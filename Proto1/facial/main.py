import os
import sys
import time
from pathlib import Path

# Ensure utility module is in the path
sys.path.append(str(Path(__file__).parent))
from utility import save
from utility.recognition import FaceRecognizer
from utility.metrics import RecognitionMetrics

def clear_screen():
    """Clear the terminal screen based on OS"""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_menu():
    """Display the main menu options"""
    clear_screen()
    print("=" * 40)
    print("       FACE RECOGNITION SYSTEM")
    print("=" * 40)
    print("1. Register New Face")
    print("2. Recognize Face")
    print("3. Metrics and Reports")
    print("4. Exit")
    print("=" * 40)

def display_metrics_menu():
    """Display the metrics submenu options"""
    clear_screen()
    print("=" * 40)
    print("       METRICS AND REPORTS")
    print("=" * 40)
    print("1. Start Real-time Monitoring with Metrics")
    print("2. Generate Recognition Report")
    print("3. Visualize Current Metrics")
    print("4. Return to Main Menu")
    print("=" * 40)

def metrics_menu():
    """Handle metrics submenu interaction"""
    metrics = RecognitionMetrics()
    
    while True:
        display_metrics_menu()
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            clear_screen()
            print("Starting real-time monitoring with metrics collection...\n")
            
            try:
                duration = int(input("Enter monitoring duration in seconds (0 for unlimited): "))
            except ValueError:
                duration = 60
                print(f"Invalid input. Using default duration: {duration} seconds")
            
            print("\nPress 'q' to stop monitoring early")
            time.sleep(2)
            
            report_path = metrics.real_time_monitoring(duration=duration)
            
            if report_path:
                print(f"\nMonitoring complete! Report saved to: {report_path}")
            
            input("\nPress Enter to return to the metrics menu...")
            
        elif choice == '2':
            clear_screen()
            print("Generating recognition metrics report...\n")
            
            if not metrics.metrics_data:
                print("No metrics data available. Please run monitoring first.")
                input("\nPress Enter to return to the metrics menu...")
                continue
                
            format_options = ['csv', 'excel', 'html']
            print("Available report formats:")
            for i, fmt in enumerate(format_options, 1):
                print(f"{i}. {fmt.upper()}")
                
            try:
                fmt_choice = int(input("\nSelect format (1-3): "))
                format_type = format_options[fmt_choice-1] if 1 <= fmt_choice <= 3 else 'csv'
            except (ValueError, IndexError):
                format_type = 'csv'
                
            report_path = metrics.generate_report(format=format_type)
            
            if report_path:
                print(f"\nReport generated successfully: {report_path}")
                
            input("\nPress Enter to return to the metrics menu...")
            
        elif choice == '3':
            clear_screen()
            print("Generating visualization of recognition metrics...\n")
            
            if not metrics.metrics_data:
                print("No metrics data available. Please run monitoring first.")
                input("\nPress Enter to return to the metrics menu...")
                continue
                
            viz_path = metrics.visualize_metrics()
            
            if viz_path:
                print(f"\nVisualization generated successfully: {viz_path}")
                
            input("\nPress Enter to return to the metrics menu...")
            
        elif choice == '4':
            return
            
        else:
            print("Invalid choice. Please enter 1-4.")
            input("Press Enter to continue...")

def main():
    """Main application entry point with menu system"""
    while True:
        display_menu()
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            clear_screen()
            # Run the face registration utility
            save.main()
            input("\nPress Enter to return to the main menu...")
        
        elif choice == '2':
            clear_screen()
            print("Starting face recognition...\n")
            # Initialize the face recognizer
            recognizer = FaceRecognizer()
            # Run the recognition loop
            recognizer.recognition_loop()
            input("\nPress Enter to return to the main menu...")
        
        elif choice == '3':
            # Run the metrics menu
            metrics_menu()
        
        elif choice == '4':
            print("Exiting application. Goodbye!")
            sys.exit(0)
        
        else:
            print("Invalid choice. Please enter 1-4.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()