import json
import os

class UserProfiles:
    def __init__(self):
        self.users = {}  # Dictionary to store user data
        self.json_file = os.path.join(os.path.dirname(__file__), "users.json")
        self._load_users()
        
    def _load_users(self):
        """Load users from JSON file if it exists"""
        if os.path.exists(self.json_file):
            with open(self.json_file, 'r') as f:
                self.users = json.load(f)
                
    def _save_users(self):
        """Save users to JSON file"""
        with open(self.json_file, 'w') as f:
            json.dump(self.users, f, indent=4)
            
    def add_user(self, user_id: str, user_data: dict):
        """
        Add a new user to the profiles
        
        Args:
            user_id: Unique identifier for the user
            user_data: Dictionary containing user information (name, face encodings, etc)
        """
        self.users[user_id] = user_data
        self._save_users()
        
    def get_user(self, user_id: str) -> dict:
        """
        Retrieve user data by ID
        
        Args:
            user_id: ID of user to retrieve
            
        Returns:
            Dictionary containing user data or None if not found
        """
        return self.users.get(user_id)
    
    def get_all_users(self) -> dict:
        """
        Get all stored user profiles
        
        Returns:
            Dictionary containing all user data
        """
        return self.users
        
    def remove_user(self, user_id: str) -> bool:
        """
        Remove a user from profiles
        
        Args:
            user_id: ID of user to remove
            
        Returns:
            True if user was removed, False if not found
        """
        if user_id in self.users:
            del self.users[user_id]
            self._save_users()
            return True
        return False