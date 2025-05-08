import os
import csv
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to import from engine
sys.path.append(str(Path(__file__).parent.parent))
from utility.recognition import recognize_face, get_recognition_model

class RecognitionMetrics:
    def __init__(self):
        """Initialize the recognition metrics tracking system"""
        # Create results directory if it doesn't exist
        self.results_dir = Path(__file__).parent.parent / "data" / "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize counters
        self.recognized_count = 0
        self.unrecognized_count = 0
        
        # Initialize data storage
        self.metrics_data = []
        
        # Initialize recognition model
        self.recognition_model = get_recognition_model()
        
        print(f"Metrics system initialized. Results will be saved to {self.results_dir}")

    def record_recognition_event(self, face_image, recognition_result, confidence_score):
        """
        Record a facial recognition event
        
        Args:
            face_image: The image containing the face
            recognition_result: The name of the recognized person or None
            confidence_score: The confidence score of the recognition
        """
        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine if recognized
        is_recognized = recognition_result is not None
        
        # Update counters
        if is_recognized:
            self.recognized_count += 1
            status = "Identified"
        else:
            self.unrecognized_count += 1
            status = "Not Identified"
            recognition_result = "Unknown"
        
        # Record the event
        event = {
            "Status": status,
            "Name": recognition_result,
            "Timestamp": timestamp_str,
            "Confidence": confidence_score,
            "Recognized_Users": self.recognized_count,
            "Unrecognized_Users": self.unrecognized_count
        }
        
        self.metrics_data.append(event)
        return event
        
    def generate_report(self, format="csv"):
        """
        Generate a report of recognition metrics
        
        Args:
            format: Output format ("csv", "excel", "html")
            
        Returns:
            Path to the generated report file
        """
        if not self.metrics_data:
            print("No metrics data to report")
            return None
            
        # Create unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.metrics_data)
        
        # Add summary row
        summary = {
            "Status": "SUMMARY",
            "Name": f"Total: {len(self.metrics_data)}",
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Confidence": df["Confidence"].mean() if df["Confidence"].any() else 0,
            "Recognized_Users": self.recognized_count,
            "Unrecognized_Users": self.unrecognized_count
        }
        
        # Save in requested format
        if format.lower() == "csv":
            filename = self.results_dir / f"recognition_metrics_{timestamp}.csv"
            df.to_csv(filename, index=False)
            
            # Append summary in a separate file
            summary_file = self.results_dir / f"recognition_summary_{timestamp}.csv"
            pd.DataFrame([summary]).to_csv(summary_file, index=False)
            
        elif format.lower() == "excel":
            filename = self.results_dir / f"recognition_metrics_{timestamp}.xlsx"
            with pd.ExcelWriter(filename) as writer:
                df.to_excel(writer, sheet_name="Recognition Data", index=False)
                pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
                
        elif format.lower() == "html":
            filename = self.results_dir / f"recognition_metrics_{timestamp}.html"
            html_content = f"""
            <html>
            <head>
                <title>Facial Recognition Metrics</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .summary {{ font-weight: bold; background-color: #e6f7ff; }}
                </style>
            </head>
            <body>
                <h1>Facial Recognition Metrics</h1>
                <h2>Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</h2>
                
                <h3>Metrics Table</h3>
                {df.to_html(index=False)}
                
                <h3>Summary</h3>
                {pd.DataFrame([summary]).to_html(index=False)}
            </body>
            </html>
            """
            with open(filename, 'w') as f:
                f.write(html_content)
        else:
            print(f"Unsupported format: {format}")
            return None
            
        print(f"Report generated: {filename}")
        return filename
        
    def visualize_metrics(self):
        """Create visualization of the recognition metrics"""
        if not self.metrics_data:
            print("No metrics data to visualize")
            return None
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create DataFrame
        df = pd.DataFrame(self.metrics_data)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Recognition Pie Chart
        labels = ['Identified', 'Not Identified']
        sizes = [self.recognized_count, self.unrecognized_count]
        colors = ['#66b3ff', '#ff9999']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        ax1.set_title('Recognition Ratio')
        
        # Plot 2: Confidence Score Distribution
        if 'Confidence' in df and df['Confidence'].notna().any():
            ax2.hist(df['Confidence'], bins=10, alpha=0.7, color='#66b3ff')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Count')
            ax2.set_title('Confidence Score Distribution')
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.results_dir / f"recognition_viz_{timestamp}.png"
        plt.savefig(viz_file)
        plt.close()
        
        print(f"Visualization saved: {viz_file}")
        return viz_file
        
    def process_and_record(self, face_image):
        """
        Process a face image through recognition and record metrics
        
        Args:
            face_image: Image containing a face to recognize
            
        Returns:
            Dictionary with recognition results and metrics
        """
        # Use the recognition module to identify the face
        result, confidence = recognize_face(face_image, self.recognition_model)
        
        # Extract just the name if result is a dictionary
        if isinstance(result, dict):
            if 'first_name' in result and 'last_name' in result:
                name = f"{result['first_name']} {result['last_name']}"
            else:
                name = str(result.get('id', 'Unknown'))
        else:
            name = result
        
        # Record the event with just the name
        event = self.record_recognition_event(face_image, name, confidence)
        
        return event
        
    def real_time_monitoring(self, camera_source=0, duration=60, resolution=(1280, 720)):
        """
        Monitor camera feed for faces and record metrics in real-time
        
        Args:
            camera_source: Camera index or path to video file
            duration: Duration in seconds to monitor (0 for indefinite)
            resolution: Tuple (width, height) for camera resolution
            
        Returns:
            Path to the generated report
        """
        import cv2
        
        # Utilize the face detector and tracker from Core
        from engine.Core import DynamicFaceEvaluator
        
        # Initialize camera with proper resolution
        cap = cv2.VideoCapture(camera_source)
        
        # Set high-resolution capture (same as in Core.py and recognition.py)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Create window that can be resized by user
        cv2.namedWindow("Recognition Monitoring", cv2.WINDOW_NORMAL)
        
        face_evaluator = DynamicFaceEvaluator()
        
        start_time = time.time()
        last_recognition_time = 0
        recognition_interval = 2.0  # Seconds between recognitions
        
        # For FPS calculation
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        
        print("Starting real-time monitoring...")
        print("Press 'q' to stop and generate report")
        
        try:
            while True:
                # Check if duration has elapsed
                if duration > 0 and (time.time() - start_time) > duration:
                    break
                    
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Update FPS counter
                frame_count += 1
                elapsed_time = time.time() - fps_start_time
                if elapsed_time > 1.0:  # Update FPS every second
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    fps_start_time = time.time()
                    
                # Evaluate face position
                score, feedback, annotated = face_evaluator.evaluate_face_position(frame)
                
                # Add FPS to display
                cv2.putText(annotated, f"FPS: {fps:.1f}", 
                          (annotated.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 255, 255), 2)
                
                # Display frame with annotations
                cv2.imshow("Recognition Monitoring", annotated)
                
                # Only recognize at intervals to avoid constant processing
                current_time = time.time()
                if current_time - last_recognition_time > recognition_interval:
                    if score > 70:  # Only recognize when face quality is good
                        # Process recognition
                        event = self.process_and_record(frame)
                        
                        # Display recognition result
                        status_text = f"{event['Status']}: {event['Name']} ({event['Confidence']:.2f})"
                        cv2.putText(annotated, status_text, (20, 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        last_recognition_time = current_time
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Error in monitoring: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
        # Generate report
        return self.generate_report(format="html")
        
    def clear_metrics(self):
        """Clear all stored metrics data"""
        self.metrics_data = []
        self.recognized_count = 0
        self.unrecognized_count = 0
        print("Metrics data cleared")

def main():
    """Main function to run the metrics utility interactively"""
    metrics = RecognitionMetrics()
    
    print("\n=== Facial Recognition Metrics Utility ===")
    print("1. Start real-time monitoring")
    print("2. Generate report from existing data")
    print("3. Clear metrics data")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == '1':
        duration = input("Enter monitoring duration in seconds (0 for indefinite): ")
        try:
            duration = int(duration)
            report_path = metrics.real_time_monitoring(duration=duration)
            if report_path:
                print(f"Report generated: {report_path}")
        except ValueError:
            print("Invalid duration. Using default 60 seconds.")
            metrics.real_time_monitoring(duration=60)
    elif choice == '2':
        format_choice = input("Enter report format (csv, excel, html): ").lower()
        if format_choice not in ['csv', 'excel', 'html']:
            format_choice = 'csv'
        metrics.generate_report(format=format_choice)
        metrics.visualize_metrics()
    elif choice == '3':
        metrics.clear_metrics()
    elif choice == '4':
        print("Exiting...")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()