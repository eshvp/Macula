import cv2
import multiprocessing
from multiprocessing import Queue, Process
from .base import FaceRecognizer
from .utils import get_recognition_model

def process_frame_worker(frame_queue, result_queue, stop_event):
    """Worker process to handle face recognition"""
    recognizer = get_recognition_model()
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)  # Get a frame from the queue
            result = recognizer.recognize_face(frame)  # Perform recognition
            result_queue.put(result)  # Put the result into the result queue
        except Exception as e:
            if not stop_event.is_set():
                print(f"Worker error: {e}")

def main():
    """Main function to run the face recognition utility with multi-processing"""
    print("=== Face Recognition Utility ===")

    # Queues for frame processing
    frame_queue = Queue(maxsize=10)  # Queue for frames to be processed
    result_queue = Queue(maxsize=10)  # Queue for results from workers
    stop_event = multiprocessing.Event()  # Event to signal workers to stop

    # Determine the number of worker processes based on available CPU cores
    num_workers = max(1, multiprocessing.cpu_count() - 1)  # Use all but one core
    print(f"Starting {num_workers} worker processes for face recognition")

    # Start worker processes
    workers = []
    for _ in range(num_workers):
        worker = Process(target=process_frame_worker, args=(frame_queue, result_queue, stop_event))
        worker.start()
        workers.append(worker)

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        stop_event.set()
        for worker in workers:
            worker.join()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Add frame to the frame queue
            if not frame_queue.full():
                frame_queue.put(frame.copy())

            # Get results from the result queue
            if not result_queue.empty():
                result = result_queue.get()
                # Process and display the result (e.g., draw bounding boxes, names, etc.)
                # For now, just print the result
                print(result)

            # Show the frame (optional)
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
    finally:
        # Stop all worker processes
        stop_event.set()
        for worker in workers:
            worker.join()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()