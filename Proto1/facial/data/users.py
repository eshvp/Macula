import os
import json
import time
from pathlib import Path
import cv2

class UserManager:
    """Class to manage user data storage and retrieval"""
    
    def __init__(self):
        """Initialize the user manager with the users directory path"""
        self.users_directory = Path(__file__).parent.parent / "users"
        os.makedirs(self.users_directory, exist_ok=True)
    
    def get_all_users(self):
        """Get list of all registered users"""
        users = []
        
        # Check if the users directory exists
        if not self.users_directory.exists():
            return users
        
        # List all user directories
        for user_dir in self.users_directory.iterdir():
            if user_dir.is_dir() and (user_dir / "info.txt").exists():
                # Parse info.txt for user details
                user_info = self._parse_user_info(user_dir)
                if user_info:
                    users.append(user_info)
        
        return users
    
    def _parse_user_info(self, user_dir):
        """Parse the info.txt file to extract user information"""
        info_path = user_dir / "info.txt"
        if not info_path.exists():
            return None
        
        user_info = {
            "id": user_dir.name,
            "directory": str(user_dir)
        }
        
        # Read info file
        with open(info_path, "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    user_info[key.strip().lower().replace(" ", "_")] = value.strip()
        
        # Count images
        image_count = sum(1 for file in user_dir.glob("face_*.jpg"))
        user_info["image_count"] = image_count
        
        return user_info
    
    def get_user_by_id(self, user_id):
        """Get user information by user_id (firstname_lastname)"""
        user_dir = self.users_directory / user_id
        
        if not user_dir.exists() or not user_dir.is_dir():
            return None
        
        return self._parse_user_info(user_dir)
    
    def get_user_images(self, user_id):
        """Get list of face image paths for a specific user"""
        user_dir = self.users_directory / user_id
        
        if not user_dir.exists() or not user_dir.is_dir():
            return []
        
        # Get all face images sorted by number
        image_files = sorted(
            user_dir.glob("face_*.jpg"),
            key=lambda x: int(x.stem.split('_')[1])
        )
        
        return [str(img_path) for img_path in image_files]
    
    def load_user_face_images(self, user_id):
        """Load face images for a specific user as OpenCV images"""
        image_paths = self.get_user_images(user_id)
        images = []
        
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                # Store as tuple with path and image
                images.append((path, img))
        
        return images
    
    def create_user(self, first_name, last_name, face_frames, frame_scores):
        """Create a new user entry (wrapper around the save_user_faces function)"""
        from utility.save import save_user_faces
        
        # Use the existing save_user_faces function
        save_dir = save_user_faces(first_name, last_name, face_frames, frame_scores)
        
        # Get the user_id
        user_id = f"{first_name.lower()}_{last_name.lower()}"
        
        return self.get_user_by_id(user_id)
    
    def delete_user(self, user_id):
        """Delete a user and their face data"""
        user_dir = self.users_directory / user_id
        
        if not user_dir.exists() or not user_dir.is_dir():
            return False
        
        # Remove all files in the user directory
        for file_path in user_dir.glob("*"):
            os.remove(file_path)
        
        # Remove the directory
        os.rmdir(user_dir)
        return True
    
    def export_user_list(self, output_file=None):
        """Export list of users to JSON file"""
        users = self.get_all_users()
        
        if output_file is None:
            output_file = self.users_directory / "user_list.json"
        
        with open(output_file, "w") as f:
            json.dump(users, f, indent=2)
        
        return output_file

# Singleton instance for easy import
user_manager = UserManager()

def get_all_users():
    """Get all registered users"""
    return user_manager.get_all_users()

def get_user(user_id):
    """Get user by ID"""
    return user_manager.get_user_by_id(user_id)

def get_user_images(user_id):
    """Get user's face images"""
    return user_manager.get_user_images(user_id)

def create_user(first_name, last_name, face_frames, frame_scores):
    """Create a new user"""
    return user_manager.create_user(first_name, last_name, face_frames, frame_scores)

def delete_user(user_id):
    """Delete a user"""
    return user_manager.delete_user(user_id)

from data.users import get_all_users, get_user, create_user

# List all users
all_users = get_all_users()
for user in all_users:
    print(f"User: {user['first_name']} {user['last_name']}")

# Get specific user
user = get_user("firstname_lastname")
if user:
    print(f"Found user: {user['first_name']} {user['last_name']}")
    print(f"Number of images: {user['image_count']}")