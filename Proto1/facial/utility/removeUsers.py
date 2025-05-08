import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from data.users import user_manager
from utility.recognition import get_recognition_model

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def list_users():
    """List all users with index numbers"""
    users = user_manager.get_all_users()
    
    if not users:
        print("\nNo users registered in the system.")
        return []
    
    print("\nRegistered Users:")
    print("=" * 50)
    print(f"{'#':3} {'Name':<30} {'ID':<20}")
    print("-" * 50)
    
    for i, user in enumerate(users, 1):
        name = f"{user['first_name']} {user['last_name']}"
        print(f"{i:2}. {name:<30} {user['id']:<20}")
    
    print("=" * 50)
    return users

def remove_user(user_id):
    """Remove a user from the database"""
    try:
        success = user_manager.delete_user(user_id)
        if success:
            print(f"\nSuccessfully removed user: {user_id}")
            return True
        else:
            print(f"\nError: Could not remove user {user_id}")
            return False
    except Exception as e:
        print(f"\nError removing user: {e}")
        return False

def main():
    """Main function to run the user removal utility"""
    while True:
        clear_screen()
        print("=== User Removal Utility ===")
        
        # List all users
        users = list_users()
        
        if not users:
            choice = input("\nPress Enter to refresh or 'q' to quit: ").lower()
            if choice == 'q':
                break
            continue
        
        print("\nOptions:")
        print("1-N: Remove user by number")
        print("r: Refresh list")
        print("q: Quit")
        
        choice = input("\nEnter choice: ").lower()
        
        if choice == 'q':
            break
        elif choice == 'r':
            continue
        else:
            try:
                idx = int(choice)
                if 1 <= idx <= len(users):
                    user = users[idx-1]
                    name = f"{user['first_name']} {user['last_name']}"
                    
                    # Confirm deletion
                    confirm = input(f"\nAre you sure you want to remove {name}? (y/n): ").lower()
                    if confirm == 'y':
                        remove_user(user['id'])
                        input("\nPress Enter to continue...")
                else:
                    print("\nInvalid user number")
                    input("Press Enter to continue...")
            except ValueError:
                print("\nInvalid input")
                input("Press Enter to continue...")

if __name__ == "__main__":
    main()