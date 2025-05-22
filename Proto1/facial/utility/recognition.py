import cv2
import dlib
import numpy as np
import os
import sys
from pathlib import Path
import time
import faiss  # Facebook AI Similarity Search library
import sqlite3
import pickle
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import itertools
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from data.users import user_manager
from engine.Core import DynamicFaceEvaluator

# Add this standalone worker function at module level (outside any class)
def _process_face_image_worker(args):
    """Standalone worker function for processing a face image
    
    This runs in a separate process and loads its own dlib models
    to avoid pickle serialization issues.
    """
    image_tuple, models_dir = args
    img_path, img = image_tuple
    user_id = Path(img_path).parent.name  # Extract user_id from path
    
    try:
        # Initialize detection components for this worker process
        # These are loaded once per process, not per image
        detector = dlib.get_frontal_face_detector()
        shape_predictor_path = models_dir / "shape_predictor_68_face_landmarks.dat"
        face_rec_model_path = models_dir / "dlib_face_recognition_resnet_model_v1.dat"
        
        # Only load the models if they haven't been loaded yet in this process
        if not hasattr(_process_face_image_worker, 'shape_predictor'):
            _process_face_image_worker.shape_predictor = dlib.shape_predictor(str(shape_predictor_path))
            _process_face_image_worker.face_rec_model = dlib.face_recognition_model_v1(str(face_rec_model_path))
            print(f"Worker process {os.getpid()} initialized dlib models")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        if len(faces) == 0:
            return None, user_id, img_path
        
        # Use the largest face if multiple detected
        face_rect = max(faces, key=lambda rect: rect.area())
        
        # Get face landmarks
        shape = _process_face_image_worker.shape_predictor(img, face_rect)
        
        # Compute face encoding
        face_encoding = _process_face_image_worker.face_rec_model.compute_face_descriptor(img, shape)
        face_encoding_np = np.array(face_encoding)
        
        return face_encoding_np, user_id, img_path
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None, user_id, img_path

class DatabaseManager:
    """Context manager for database connections to ensure proper resource cleanup"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
    
    def __enter__(self):
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            self.cursor = self.conn.cursor()
            return self.conn, self.cursor
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to open database at {self.db_path}: {e}. Please ensure the data directory exists and is writable.")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None:
                # No exception occurred, commit changes
                try:
                    self.conn.commit()
                except sqlite3.Error as e:
                    print(f"Warning: Failed to commit database changes: {e}")
            
            # Always close the connection
            try:
                self.conn.close()
            except sqlite3.Error as e:
                print(f"Warning: Error closing database connection: {e}")
        
        # Return False to propagate exceptions
        return False

class FaceRecognizer:
    """Class to recognize faces of registered users"""
    
    def __init__(self, recognition_threshold=0.6):
        """Initialize the face recognizer with models and thresholds"""
        # Path to face recognition model
        models_dir = Path(__file__).parent.parent / "models"
        self.face_rec_model_path = models_dir / "dlib_face_recognition_resnet_model_v1.dat"
        self.shape_predictor_path = models_dir / "shape_predictor_68_face_landmarks.dat"
        
        # Initialize face detection/analysis components
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(str(self.shape_predictor_path))
        self.face_rec_model = dlib.face_recognition_model_v1(str(self.face_rec_model_path))
        
        # Set recognition threshold (lower = stricter matching)
        self.recognition_threshold = recognition_threshold
        
        # Use the face evaluator for better face position analysis
        self.face_evaluator = DynamicFaceEvaluator()
        
        # Add dimension for face encodings (needed for FAISS)
        self.dimension = 128  # dlib face encodings are 128-dimensional
        
        # Database and index paths
        self.db_path = Path(__file__).parent.parent / "data" / "face_encodings.db"
        self.index_path = Path(__file__).parent.parent / "data" / "face_index.faiss"
        
        # Multiprocessing configuration
        self.use_multiprocessing = True
        self.max_workers = min(multiprocessing.cpu_count(), 8)  # Limit to 8 cores max
        
        # Metrics tracking
        self.enable_metrics = True
        self.metrics = {}
        
        # Cache of known face encodings
        self.known_face_encodings = []
        self.known_user_ids = []
        self.load_known_faces()
        
    def _process_face_image(self, image_tuple):
        """Process a single face image (for multiprocessing)"""
        img_path, img = image_tuple
        user_id = Path(img_path).parent.name  # Extract user_id from path
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray)
            if len(faces) == 0:
                return None, user_id, img_path
            
            # Use the largest face if multiple detected
            face_rect = max(faces, key=lambda rect: rect.area())
            
            # Get face landmarks
            shape = self.shape_predictor(img, face_rect)
            
            # Compute face encoding
            face_encoding = self.face_rec_model.compute_face_descriptor(img, shape)
            face_encoding_np = np.array(face_encoding)
            
            return face_encoding_np, user_id, img_path
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None, user_id, img_path

    def load_known_faces(self, force_rebuild=False):
        """Load faces with incremental updates and improved error handling"""
        load_metrics = {
            'start_time': time.time(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'user_metrics': {},
            'phases': {}
        }
        
        print("Loading face recognition database...")
        start_time = time.time()
        
        try:
            # Phase 1: Setup 
            phase1_start = time.time()
            
            # Get current database state
            try:
                with DatabaseManager(self.db_path) as (conn, cursor):
                    # Create tables if they don't exist
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS users (
                            id TEXT PRIMARY KEY,
                            first_name TEXT,
                            last_name TEXT,
                            capture_date REAL,
                            last_updated REAL
                        )
                    """)
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS face_encodings (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id TEXT,
                            encoding BLOB,
                            image_path TEXT,
                            last_updated REAL,
                            FOREIGN KEY (user_id) REFERENCES users(id)
                        )
                    """)
                    
                    # Get existing users from database
                    cursor.execute("SELECT id, last_updated FROM users")
                    existing_users = {row[0]: row[1] for row in cursor.fetchall()}
                    
                    # Get existing image paths from database to avoid reprocessing
                    cursor.execute("SELECT image_path FROM face_encodings")
                    existing_images = {row[0] for row in cursor.fetchall()}
            except Exception as e:
                print(f"Database initialization error: {e}")
                print(f"Please check if the data directory exists and is writable: {Path(self.db_path).parent}")
                # Create a basic in-memory fallback database for this session
                conn = sqlite3.connect(":memory:")
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE users (id TEXT PRIMARY KEY, first_name TEXT, last_name TEXT, capture_date REAL, last_updated REAL)")
                cursor.execute("CREATE TABLE face_encodings (id INTEGER PRIMARY KEY, user_id TEXT, encoding BLOB, image_path TEXT, last_updated REAL)")
                existing_users = {}
                existing_images = set()
                print("Using in-memory database for this session (changes won't be saved)")
            
            # Get all registered users
            try:
                all_users = user_manager.get_all_users()
                users_dict = {user['id']: user for user in all_users}
                
                if not all_users:
                    print("No registered users found. Please register users first.")
            except Exception as e:
                print(f"Error getting user data: {e}")
                all_users = []
                users_dict = {}
            
            # Identify users to add, update, or remove
            users_to_add = [u for u in all_users if u['id'] not in existing_users]
            users_to_update = [u for u in all_users if u['id'] in existing_users]
            users_to_remove = [uid for uid in existing_users if uid not in users_dict]
            
            load_metrics['phases']['setup_time'] = time.time() - phase1_start
            
            # Quick check for unchanged state - if nothing changed and index exists, load it
            if not force_rebuild and not (users_to_add or users_to_update or users_to_remove):
                if os.path.exists(self.index_path):
                    print("No user changes detected, checking if index is up to date...")
                    if not self._should_rebuild_index():
                        print("Face database is up to date. Loading existing index.")
                        self._load_faiss_index()
                        
                        # Print summary
                        print(f"\nFace recognition database loaded in {time.time() - start_time:.2f} seconds")
                        print(f"Face database contains {len(self.known_face_encodings)} encodings from {len(set(self.known_user_ids))} users")
                        print("Face recognition is ready to use!")
                        
                        return 0
            
            # Process database updates in batches to avoid long transactions
            with DatabaseManager(self.db_path) as (conn, cursor):
                # Phase 2: Database updates
                phase2_start = time.time()
                
                # Handle user removals
                if users_to_remove:
                    for user_id in users_to_remove:
                        cursor.execute("DELETE FROM face_encodings WHERE user_id = ?", (user_id,))
                        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                    print(f"Removed {len(users_to_remove)} users from database")
                
                # Add new users
                for user in users_to_add:
                    cursor.execute(
                        "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                        (user['id'], user['first_name'], user['last_name'],
                         user.get('capture_date', time.time()), time.time())
                    )
                
                # Update existing users if needed
                for user in users_to_update:
                    cursor.execute(
                        "UPDATE users SET first_name = ?, last_name = ?, last_updated = ? WHERE id = ?",
                        (user['first_name'], user['last_name'], time.time(), user['id'])
                    )
                
                # Only collect and process images for new/updated users
                users_to_process = users_to_add + users_to_update
                load_metrics['phases']['database_updates_time'] = time.time() - phase2_start
                
                # Phase 3: Process only changed users' images
                if users_to_process or force_rebuild:
                    processing_start = time.time()
                    
                    # Collect images only for users that need updating
                    all_image_tuples = []
                    failed_users = []
                    
                    if force_rebuild:
                        # Process all users if doing a full rebuild
                        for user in all_users:
                            try:
                                image_tuples = user_manager.load_user_face_images(user['id'])
                                all_image_tuples.extend(image_tuples)
                            except Exception as e:
                                print(f"Error loading images for user {user['id']}: {e}")
                                failed_users.append(user['id'])
                        
                        # Clear all existing encodings
                        cursor.execute("DELETE FROM face_encodings")
                    else:
                        # Process only new/updated users
                        for user in users_to_process:
                            # Delete any existing encodings for this user
                            cursor.execute("DELETE FROM face_encodings WHERE user_id = ?", (user['id'],))
                            
                            # Get this user's images
                            try:
                                image_tuples = user_manager.load_user_face_images(user['id'])
                                all_image_tuples.extend(image_tuples)
                            except Exception as e:
                                print(f"Error loading images for user {user['id']}: {e}")
                                failed_users.append(user['id'])
                    
                    if failed_users:
                        print(f"Warning: Failed to load images for {len(failed_users)} users")
                    
                    if not all_image_tuples:
                        print("No face images found to process. Please add images for registered users.")
                    else:
                        print(f"Found {len(all_image_tuples)} images to process")
                    
                        # Process images in parallel as before
                        models_dir = Path(__file__).parent.parent / "models"
                        worker_args = [(img_tuple, models_dir) for img_tuple in all_image_tuples]
                        
                        try:
                            from tqdm import tqdm
                            # Create a wrapper to sanitize results from multiprocessing
                            results = []
                            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                                for result in tqdm(
                                    executor.map(_process_face_image_worker, worker_args),
                                    total=len(all_image_tuples),
                                    desc="Processing face images",
                                    unit="img"
                                ):
                                    # Verify the result is valid (non-None parts where needed)
                                    if isinstance(result, tuple) and len(result) == 3:
                                        encoding, user_id, img_path = result
                                        if user_id is not None and img_path is not None:
                                            results.append(result)
                        except Exception as e:
                            print(f"Error in parallel image processing: {e}")
                            print("Falling back to sequential processing")
                            
                            # Process sequentially as fallback
                            results = []
                            for arg in tqdm(worker_args, desc="Processing images sequentially"):
                                try:
                                    result = self._process_face_image(arg[0])
                                    results.append(result)
                                except Exception as img_err:
                                    print(f"Error processing image: {img_err}")
                        
                        # Insert new encodings in batches
                        batch_size = 50
                        encoding_batches = []
                        current_batch = []
                        
                        for encoding, user_id, img_path in results:
                            if encoding is not None:
                                try:
                                    # Store as direct binary representation instead of pickle for faster loading
                                    encoding_blob = encoding.tobytes()  # Direct binary conversion
                                    current_batch.append((user_id, encoding_blob, img_path, time.time()))
                                    
                                    if len(current_batch) >= batch_size:
                                        encoding_batches.append(current_batch)
                                        current_batch = []
                                except Exception as e:
                                    print(f"Error preparing encoding for {img_path}: {e}")
                        
                        if current_batch:
                            encoding_batches.append(current_batch)
                        
                        # Insert batches with error handling
                        success_count = 0
                        for batch in encoding_batches:
                            try:
                                cursor.executemany(
                                    "INSERT INTO face_encodings (user_id, encoding, image_path, last_updated) VALUES (?, ?, ?, ?)",
                                    batch
                                )
                                success_count += len(batch)
                            except Exception as e:
                                print(f"Error inserting batch into database: {e}")
                        
                        print(f"Successfully stored {success_count} face encodings in database")
                    
                    load_metrics['phases']['image_processing_time'] = time.time() - processing_start
                
                # Phase 4: Rebuild FAISS index from current database, but only if needed
                faiss_start = time.time()
                index_needs_rebuild = force_rebuild or users_to_process or self._should_rebuild_index()
                
                if index_needs_rebuild:
                    print("Rebuilding FAISS index...")
                    self._rebuild_faiss_index(conn, cursor)
                else:
                    print("FAISS index is up to date, loading existing one...")
                    self._load_faiss_index()
                    
                load_metrics['phases']['faiss_rebuild_time'] = time.time() - faiss_start
            
            # Overall metrics
            elapsed_time = time.time() - start_time
            load_metrics['total_time'] = elapsed_time
            self.metrics['last_load'] = load_metrics
            
            # Print helpful summary
            print(f"\nFace recognition database loaded in {elapsed_time:.2f} seconds")
            print(f"Face database contains {len(self.known_face_encodings)} encodings from {len(set(self.known_user_ids))} users")
            if self.known_face_encodings:
                print("Face recognition is ready to use!")
            else:
                print("\nWarning: No face encodings were loaded. Recognition will not work.")
                print("Please check that:")
                print("1. You have registered users in the system")
                print("2. Users have face images assigned to them")
                print("3. The face images contain detectable faces")
            
            return len(users_to_process)
            
        except Exception as e:
            print(f"Unexpected error in load_known_faces: {e}")
            traceback.print_exc()
            return 0

    def _rebuild_faiss_index(self, conn=None, cursor=None):
        """Rebuild FAISS index from the current database with optimized storage
        
        Args:
            conn: Optional existing database connection
            cursor: Optional existing database cursor
        """
        external_connection = conn is not None and cursor is not None
        
        try:
            # Use provided connection or create a new one
            if not external_connection:
                with DatabaseManager(self.db_path) as (conn, cursor):
                    return self._rebuild_faiss_index_internal(conn, cursor)
            else:
                return self._rebuild_faiss_index_internal(conn, cursor)
        except Exception as e:
            print(f"Error rebuilding FAISS index: {e}")
            # Initialize empty index as fallback
            self.index = faiss.IndexFlatL2(self.dimension)
            self.known_user_ids = []
            self.known_face_encodings = []
            return False

    def _rebuild_faiss_index_internal(self, conn, cursor):
        """Internal implementation of FAISS index rebuilding"""
        rebuild_start = time.time()
        
        # Create a new index
        try:
            self.index = faiss.IndexFlatL2(self.dimension)
            
            # Get all encodings from database
            cursor.execute("SELECT user_id, encoding FROM face_encodings")
            results = cursor.fetchall()
            
            if not results:
                print("Warning: No face encodings found in the database.")
                return False
                
            # Extract encodings and user IDs
            user_ids = []
            all_encodings = []
            
            for user_id, encoding_blob in results:
                # Process encoding blob
                if isinstance(encoding_blob, bytes):
                    try:
                        # First try direct numpy loading (faster)
                        encoding = np.frombuffer(encoding_blob, dtype=np.float64)
                        if len(encoding) != self.dimension:
                            # If dimension doesn't match, it's probably pickled
                            encoding = np.array(pickle.loads(encoding_blob))
                    except Exception as e:
                        try:
                            # Fall back to pickle if direct conversion fails
                            encoding = np.array(pickle.loads(encoding_blob))
                        except Exception as pickle_err:
                            print(f"Warning: Failed to decode encoding for user {user_id}: {pickle_err}")
                            continue
                    
                    all_encodings.append(encoding)
                    user_ids.append(user_id)
            
            # Add to FAISS index
            if all_encodings:
                encodings_array = np.array(all_encodings).astype('float32')
                
                # Configure FAISS for multithreading if available
                if hasattr(faiss, 'omp_set_num_threads'):
                    faiss.omp_set_num_threads(self.max_workers)
                    
                self.index.add(encodings_array)
                
                # Save index and mapping
                index_dir = Path(self.index_path).parent
                if not index_dir.exists():
                    index_dir.mkdir(parents=True, exist_ok=True)
                    
                try:
                    faiss.write_index(self.index, str(self.index_path))
                    with open(str(self.index_path).replace('.faiss', '_map.pkl'), 'wb') as f:
                        pickle.dump(user_ids, f)
                    print(f"FAISS index saved with {len(user_ids)} face encodings")
                except Exception as e:
                    print(f"Warning: Failed to save FAISS index: {e}")
                    print(f"You may not have write permissions to {self.index_path}")
                    print(f"Recognition will work for this session but won't persist between runs")
            
            # Update in-memory caches
            self.known_user_ids = user_ids
            self.known_face_encodings = all_encodings
            
            print(f"FAISS index rebuilt with {len(all_encodings)} face encodings in {time.time() - rebuild_start:.2f}s")
            return True
            
        except Exception as e:
            print(f"Error in FAISS index rebuild: {e}")
            traceback.print_exc()
            return False

    def _load_faiss_index(self):
        """Load the FAISS index and user mapping from disk with memory mapping"""
        load_start = time.time()
        
        try:
            if not os.path.exists(self.index_path):
                print(f"FAISS index not found at {self.index_path}")
                print("This is normal for first-time use or after clearing the database.")
                return False
                
            try:
                # Load FAISS index with memory mapping for faster loading and reduced memory usage
                self.index = faiss.read_index(str(self.index_path), faiss.IO_FLAG_MMAP)
                print(f"Loaded FAISS index using memory mapping")
            except Exception as faiss_error:
                print(f"Warning: Memory-mapped FAISS index loading failed: {faiss_error}")
                print("Falling back to standard index loading...")
                try:
                    self.index = faiss.read_index(str(self.index_path))
                    print("Successfully loaded FAISS index with standard method")
                except Exception as e:
                    print(f"Error: Failed to load FAISS index: {e}")
                    print("Initializing empty index. Recognition may not work correctly.")
                    self.index = faiss.IndexFlatL2(self.dimension)
                    return False
                    
            # Load user ID mapping
            map_path = str(self.index_path).replace('.faiss', '_map.pkl')
            if not os.path.exists(map_path):
                print(f"Warning: FAISS index mapping file not found at {map_path}")
                print("Face recognition will not work correctly.")
                return False
                
            try:
                with open(map_path, 'rb') as f:
                    self.known_user_ids = pickle.load(f)
            except Exception as e:
                print(f"Error loading user ID mapping: {e}")
                return False
            
            # Load encodings from database for in-memory cache
            try:
                with DatabaseManager(self.db_path) as (conn, cursor):
                    cursor.execute("SELECT encoding FROM face_encodings ORDER BY id")
                    
                    # Directly convert binary blobs to numpy arrays without pickle
                    self.known_face_encodings = []
                    for row in cursor.fetchall():
                        encoding_blob = row[0]
                        if isinstance(encoding_blob, bytes):
                            try:
                                # First try direct numpy loading (faster)
                                encoding = np.frombuffer(encoding_blob, dtype=np.float64)
                                if len(encoding) != self.dimension:
                                    # If dimension doesn't match, it's probably pickled
                                    encoding = np.array(pickle.loads(encoding_blob))
                            except Exception:
                                # Fall back to pickle if direct conversion fails
                                try:
                                    encoding = np.array(pickle.loads(encoding_blob))
                                except Exception:
                                    # Skip this encoding if both methods fail
                                    continue
                            
                            self.known_face_encodings.append(encoding)
                    
            except Exception as e:
                print(f"Error loading face encodings from database: {e}")
                print("Using only FAISS index for recognition (slower but still functional)")
                
            load_time = time.time() - load_start
            print(f"Loaded {len(self.known_face_encodings)} face encodings in {load_time:.3f}s")
            
            # Verify consistency between index and mapping
            if len(self.known_user_ids) != self.index.ntotal:
                print(f"Warning: User ID mapping size ({len(self.known_user_ids)}) doesn't match FAISS index size ({self.index.ntotal})")
                print("This may cause incorrect recognition results.")
            
            return True
                
        except Exception as e:
            print(f"Unexpected error loading FAISS index: {e}")
            traceback.print_exc()
            
            # Initialize empty index as fallback
            self.index = faiss.IndexFlatL2(self.dimension)
            self.known_user_ids = []
            self.known_face_encodings = []
            return False

    def _should_rebuild_index(self):
        """Check if FAISS index needs rebuilding based on database updates
        
        Returns:
            bool: True if index needs rebuilding, False otherwise
        """
        try:
            # Check if index file exists
            if not os.path.exists(self.index_path):
                print("Index file doesn't exist - needs rebuild")
                return True
                
            # Check if mapping file exists
            map_path = str(self.index_path).replace('.faiss', '_map.pkl')
            if not os.path.exists(map_path):
                print("Index mapping file doesn't exist - needs rebuild")
                return True
                
            # Get last modification time of index file
            index_mtime = os.path.getmtime(self.index_path)
            index_mod_time = datetime.fromtimestamp(index_mtime)
            
            # Get latest update timestamp from database
            with DatabaseManager(self.db_path) as (conn, cursor):
                # First check if there are any encodings at all in the database
                cursor.execute("SELECT COUNT(*) FROM face_encodings")
                count = cursor.fetchone()[0]
                
                # If database is empty but we have an index file, something's wrong
                if count == 0:
                    print("Database is empty but index exists - needs rebuild")
                    return True
                
                # Check index size vs database record count
                index = faiss.read_index(str(self.index_path))
                if index.ntotal != count:
                    print(f"Index size ({index.ntotal}) doesn't match database record count ({count}) - needs rebuild")
                    return True
                    
                # Check most recent update in database
                cursor.execute("SELECT MAX(last_updated) FROM face_encodings")
                latest_db_time = cursor.fetchone()[0]
                
                if latest_db_time is None:
                    print("No timestamps in database - needs rebuild")
                    return True
                    
                db_update_time = datetime.fromtimestamp(latest_db_time)
                
                # Also check if users have been changed/removed
                cursor.execute("SELECT MAX(last_updated) FROM users")
                latest_user_time = cursor.fetchone()[0]
                
                if latest_user_time is not None:
                    user_update_time = datetime.fromtimestamp(latest_user_time)
                    db_update_time = max(db_update_time, user_update_time)
                
                # Compare timestamps with 1 second buffer for filesystem precision
                needs_rebuild = db_update_time > index_mod_time + timedelta(seconds=1)
                
                if needs_rebuild:
                    print(f"Database updated at {db_update_time}, index from {index_mod_time}")
                    print("FAISS index needs rebuilding based on database changes")
                else:
                    print(f"FAISS index is up to date (index: {index_mod_time}, db: {db_update_time})")
                    
                return needs_rebuild
                
        except Exception as e:
            print(f"Error checking if index needs rebuilding: {e}")
            traceback.print_exc()
            # If we can't determine, rebuild to be safe
            return True

    def recognize_face(self, frame, min_quality_score=70):
        """Recognize a face in the given frame"""
        # Evaluate face quality
        score, feedback, annotated = self.face_evaluator.evaluate_face_position(frame)
        
        # If face quality is too low, don't attempt recognition
        if score < min_quality_score:
            return None, score, "Face quality too low for reliable recognition", annotated
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        if len(faces) == 0:
            return None, score, "No face detected", annotated
        
        # Use the largest face if multiple detected
        face_rect = max(faces, key=lambda rect: rect.area())
        
        # Get face landmarks
        shape = self.shape_predictor(frame, face_rect)
        
        # Compute face encoding
        face_encoding = self.face_rec_model.compute_face_descriptor(frame, shape)
        face_encoding_np = np.array(face_encoding)
        
        # If no known faces, return unknown
        if len(self.known_face_encodings) == 0:
            return None, score, "No known faces to compare with", annotated
        
        # Calculate distances to all known faces
        distances = []
        for known_encoding in self.known_face_encodings:
            # Euclidean distance between face encodings
            distance = np.linalg.norm(face_encoding_np - known_encoding)
            distances.append(distance)
        
        # Find the closest match
        best_match_idx = np.argmin(distances)
        best_match_distance = distances[best_match_idx]
        
        # Check if the match is close enough
        if best_match_distance <= self.recognition_threshold:
            recognized_user_id = self.known_user_ids[best_match_idx]
            
            # Get user details
            user_info = user_manager.get_user_by_id(recognized_user_id)
            first_name = user_info.get('first_name', 'Unknown')
            last_name = user_info.get('last_name', 'User')
            
            # Add recognition info to the annotated frame
            cv2.putText(annotated, f"Recognized: {first_name} {last_name}", 
                       (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated, f"Match confidence: {(1-best_match_distance)*100:.1f}%", 
                       (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            return user_info, score, f"Recognized: {first_name} {last_name}", annotated
        else:
            # Add unknown user info to the annotated frame
            cv2.putText(annotated, "Unknown person", 
                       (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(annotated, f"Best match distance: {best_match_distance:.3f}", 
                       (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            return None, score, "Unknown person", annotated
    
    def recognition_loop(self):
        """Run a continuous recognition loop with camera input"""
        print("\nStarting face recognition...")
        print("Press 'q' or ESC to quit, 'r' to reload known faces")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # FPS tracking
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading from camera")
                break
            
            # Update FPS calculation
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Perform recognition
            user, score, message, annotated = self.recognize_face(frame)
            
            # Add FPS display
            cv2.putText(annotated, f"FPS: {fps:.1f}", 
                       (annotated.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 255), 2)
            
            # Display the annotated frame
            cv2.imshow("Face Recognition", annotated)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q to quit
                break
            elif key == ord('r'):  # r to reload face database
                print("Reloading face database...")
                self.load_known_faces()
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
    
    def recognize_single_frame(self, frame=None):
        """Capture a single frame and perform recognition, or use provided frame"""
        if frame is None:
            # Initialize camera
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Capture a single frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print("Error reading from camera")
                return None, 0, "Camera error", None
        
        # Perform recognition
        return self.recognize_face(frame)

class OptimizedFaceRecognizer(FaceRecognizer):
    def __init__(self, recognition_threshold=0.6, use_multiprocessing=True):
        """Initialize with optimized storage, searching and parallel processing"""
        # Enable detailed performance metrics
        self.enable_metrics = True
        self.metrics = {}
        
        # Start init timing
        init_start = time.time()
        
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = min(multiprocessing.cpu_count(), 8)  # Limit to 8 cores max
        
        super().__init__(recognition_threshold)
        
        # Replace in-memory arrays with indexed storage
        self.db_path = Path(__file__).parent.parent / "data" / "face_encodings.db"
        self.index_path = Path(__file__).parent.parent / "data" / "face_index.faiss"
        
        # Initialize database connection
        db_start = time.time()
        self._init_database()
        if self.enable_metrics:
            self.metrics['database_init_time'] = time.time() - db_start
        
        # Set up FAISS index for fast nearest-neighbor search
        index_start = time.time()
        self.dimension = 128  # dlib face encodings are 128-dimensional
        self.index = None
        self._init_index()
        if self.enable_metrics:
            self.metrics['index_init_time'] = time.time() - index_start
        
        # Add thread pool for parallel tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Enhanced cache with frequency tracking
        self.cache_size = 100  # Increased cache size
        self.user_cache = {}
        self.access_frequency = {}
        
        # Batch processing configuration
        self.batch_size = 5  # Max faces to process in one batch
        
        # Record total initialization time
        if self.enable_metrics:
            self.metrics['total_init_time'] = time.time() - init_start
            print(f"Recognizer initialized in {self.metrics['total_init_time']:.2f} seconds")
        
    def recognize_face(self, frame, min_quality_score=70):
        """Single face recognition with timing metrics"""
        recog_start = time.time()
        
        # Get metrics for each step if enabled
        recognition_metrics = {} if self.enable_metrics else None
        
        # Evaluate face quality
        eval_start = time.time()
        score, feedback, annotated = self.face_evaluator.evaluate_face_position(frame)
        if recognition_metrics:
            recognition_metrics['face_evaluation_time'] = time.time() - eval_start
        
        # Actual recognition (use multi-face method internally)
        recognize_start = time.time()
        results, score, message, annotated = self.recognize_faces(frame, min_quality_score)
        if recognition_metrics:
            recognition_metrics['recognition_time'] = time.time() - recognize_start
        
        if not results:
            if recognition_metrics:
                recognition_metrics['total_time'] = time.time() - recog_start
                self.metrics['last_recognition'] = recognition_metrics
            return None, score, message, annotated
        
        # Return the first (usually largest) face result in the original format
        result = results[0]
        
        # Add total time to metrics
        if recognition_metrics:
            recognition_metrics['total_time'] = time.time() - recog_start
            recognition_metrics['faces_detected'] = len(results)
            recognition_metrics['faces_recognized'] = sum(1 for r in results if r['user_info'] is not None)
            
            # Add metrics to annotated frame if no user was found (to help debugging)
            if result['user_info'] is None:
                y_pos = 420
                for metric, value in recognition_metrics.items():
                    if isinstance(value, float):
                        text = f"{metric}: {value*1000:.1f}ms"
                    else:
                        text = f"{metric}: {value}"
                    cv2.putText(annotated, text, (20, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_pos += 25
            
            # Store metrics
            self.metrics['last_recognition'] = recognition_metrics
        
        if result['user_info'] is not None:
            return result['user_info'], score, message, annotated
        else:
            return None, score, message, annotated

def get_recognition_model():
    """Returns a configured face recognition model for use by other modules"""
    # Should use optimized version instead
    try:
        recognizer = OptimizedFaceRecognizer()
        print("Using optimized face recognizer")
    except Exception as e:
        print(f"Falling back to standard recognizer: {e}")
        recognizer = FaceRecognizer()
    return recognizer

def recognize_face(face_image, recognition_model=None):
    """Recognize a face and return name and confidence score
    
    Args:
        face_image: Image containing a face to recognize
        recognition_model: Optional pre-loaded recognition model
        
    Returns:
        Tuple of (name, confidence_score)
    """
    if recognition_model is None:
        recognition_model = get_recognition_model()
    
    # Get recognition result (expecting a tuple of result, score, message, annotated_image)
    result, score, message, _ = recognition_model.recognize_face(face_image)
    
    # Extract name if result is a dictionary 
    if isinstance(result, dict):
        if 'first_name' in result and 'last_name' in result:
            name = f"{result['first_name']} {result['last_name']}"
        else:
            name = str(result.get('id', 'Unknown'))
        return name, score
    
    return result, score

def main():
    """Main function to run the face recognition utility"""
    print("=== Face Recognition Utility ===")
    
    # Initialize face recognizer
    recognizer = FaceRecognizer()
    
    # Start recognition loop
    recognizer.recognition_loop()

if __name__ == "__main__":
    main()