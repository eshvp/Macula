import os
import pickle
import sqlite3
import numpy as np
import faiss
from pathlib import Path
from .base import FaceRecognizer

class OptimizedFaceRecognizer(FaceRecognizer):
    def __init__(self, recognition_threshold=0.6):
        """Initialize with optimized storage and searching"""
        super().__init__(recognition_threshold)
        
        # Replace in-memory arrays with indexed storage
        self.db_path = Path(__file__).parent.parent / "data" / "face_encodings.db"
        self.index_path = Path(__file__).parent.parent / "data" / "face_index.faiss"
        
        # Initialize database connection
        self._init_database()
        
        # Set up FAISS index for fast nearest-neighbor search
        self.dimension = 128  # dlib face encodings are 128-dimensional
        self.index = None
        self._init_index()
        
        # Cache for frequently accessed users
        self.cache_size = 50
        self.user_cache = {}
        
    def _init_database(self):
        """Initialize SQLite database for face encodings"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            created_at TIMESTAMP,
            last_access TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            encoding BLOB,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def _init_index(self):
        """Initialize FAISS index for fast similarity search"""
        if os.path.exists(self.index_path):
            # Load existing index
            self.index = faiss.read_index(str(self.index_path))
            print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            # Create new index - using L2 distance and flat index for accuracy
            self.index = faiss.IndexFlatL2(self.dimension)
            print("Created new FAISS index")
            
    def load_known_faces(self):
        """Load faces into index instead of memory arrays"""
        # Check if database already has data
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        if user_count > 0 and os.path.exists(self.index_path):
            print(f"Using existing database with {user_count} users")
            # Load the FAISS index instead of rebuilding
            return
        
        print("Building optimized face recognition index...")
        start_time = time.time()
        
        # Clear index and rebuild
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Get all registered users
        all_users = user_manager.get_all_users()
        total_images = 0
        
        # Create database connection
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM face_encodings")
        cursor.execute("DELETE FROM users")
        
        # Temporary arrays to build the index
        all_encodings = []
        user_ids = []
        
        for user in all_users:
            user_id = user['id']
            
            # Get creation time from user data or use current time
            # If user data contains capture_date, use that
            created_at = user.get('capture_date', time.time())
            
            # Add user to database with both timestamps
            cursor.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                (user_id, user['first_name'], user['last_name'], created_at, time.time())
            )
            
            # Load user's face images
            image_tuples = user_manager.load_user_face_images(user_id)
            if not image_tuples:
                continue
            
            for img_path, img in image_tuples:
                # Process and get face encoding
                encoding = self._process_face_image(img)
                if encoding is None:
                    continue
                
                # Add to temporary arrays
                all_encodings.append(encoding)
                user_ids.append(user_id)
                
                # Store in database
                encoding_blob = pickle.dumps(encoding)
                cursor.execute(
                    "INSERT INTO face_encodings (user_id, encoding) VALUES (?, ?)",
                    (user_id, encoding_blob)
                )
                
                total_images += 1
        
        # Add all encodings to FAISS index at once (much faster than one at a time)
        if all_encodings:
            encodings_array = np.array(all_encodings).astype('float32')
            self.index.add(encodings_array)
            
            # Map FAISS indices to user IDs
            with open(str(self.index_path).replace('.faiss', '_map.pkl'), 'wb') as f:
                pickle.dump(user_ids, f)
        
        conn.commit()
        conn.close()
        
        # Save index
        faiss.write_index(self.index, str(self.index_path))
        
        elapsed_time = time.time() - start_time
        print(f"Built index with {total_images} face encodings from {len(all_users)} users in {elapsed_time:.2f} seconds")
        return total_images
        
    def _process_face_image(self, img):
        """Extract face encoding from an image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:
            return None
            
        face_rect = max(faces, key=lambda rect: rect.area())
        shape = self.shape_predictor(img, face_rect)
        face_encoding = self.face_rec_model.compute_face_descriptor(img, shape)
        return np.array(face_encoding)
        
    def recognize_face(self, frame, min_quality_score=70):
        """Optimized face recognition using FAISS index"""
        # Face quality evaluation stays the same
        score, feedback, annotated = self.face_evaluator.evaluate_face_position(frame)
        
        if score < min_quality_score:
            return None, score, "Face quality too low for reliable recognition", annotated
            
        # Get face encoding
        face_encoding = self._process_face_image(frame)
        if face_encoding is None:
            return None, score, "No face detected", annotated
            
        # Check if index is empty
        if self.index.ntotal == 0:
            return None, score, "No known faces to compare with", annotated
            
        # Search in FAISS index
        query = np.array([face_encoding]).astype('float32')
        distances, indices = self.index.search(query, 1)  # Get closest match
        
        distance = distances[0][0]
        best_match_idx = indices[0][0]
        
        # Convert distance to confidence (1.0 is perfect match)
        confidence = max(0, 1.0 - distance)
        
        # Check if the match is close enough
        if distance <= self.recognition_threshold:
            # Load user ID mapping
            with open(str(self.index_path).replace('.faiss', '_map.pkl'), 'rb') as f:
                user_ids = pickle.load(f)
            
            # Get user ID and info
            user_id = user_ids[best_match_idx]
            
            # Check cache first
            if user_id in self.user_cache:
                user_info = self.user_cache[user_id]
            else:
                # Get from database/user manager
                user_info = user_manager.get_user_by_id(user_id)
                
                # Update cache (LRU-like)
                if len(self.user_cache) >= self.cache_size:
                    # Remove least recently used
                    lru_id = min(self.user_cache, key=lambda k: self.user_cache[k].get('_last_access', 0))
                    del self.user_cache[lru_id]
                
                # Add to cache with timestamp
                user_info['_last_access'] = time.time()
                self.user_cache[user_id] = user_info
            
            # Add recognition info to annotated frame (same as before)
            first_name = user_info.get('first_name', 'Unknown')
            last_name = user_info.get('last_name', 'User')
            
            cv2.putText(annotated, f"Recognized: {first_name} {last_name}", 
                      (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated, f"Match confidence: {confidence*100:.1f}%", 
                      (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            return user_info, score, f"Recognized: {first_name} {last_name}", annotated
        else:
            # Unknown person (same as before)
            cv2.putText(annotated, "Unknown person", 
                      (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(annotated, f"Best match distance: {distance:.3f}", 
                      (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            return None, score, "Unknown person", annotated