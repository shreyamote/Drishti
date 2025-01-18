import cv2
import pyttsx3
import time
from ultralytics import YOLO
import logging
import sys

class IntelligentAssistant:
    def _init_(self, model_path="yolov8n.pt", confidence_threshold=0.3):
        """Initialize the Intelligent Assistant"""
        self.setup_logging()
        self.initialize_components(model_path, confidence_threshold)
        self.load_labels()
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('assistant.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_components(self, model_path, confidence_threshold):
        """Initialize components"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)
            self.engine.setProperty("volume", 1.0)
            self.model = YOLO(model_path)
            self.confidence_threshold = confidence_threshold
            self.running = False
            
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            raise

    def load_labels(self):
        """Load COCO dataset labels"""
        self.labels = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def speak(self, text):
        """Convert text to speech"""
        try:
            print(f"Assistant: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            self.logger.error(f"Speech error: {str(e)}")

    def create_detection_announcement(self, detections):
        """Create announcement for detected objects only"""
        if not detections:
            return "No objects detected"
        
        # Count objects
        object_count = {}
        for obj in detections:
            object_count[obj] = object_count.get(obj, 0) + 1
        
        # Format object counts
        objects_text = ", ".join(
            f"{count} {obj}{'s' if count > 1 else ''}" 
            for obj, count in object_count.items()
        )
        return f"Detected: {objects_text}"

    def process_frame(self, frame):
        """Process a single frame for object detection with hazard detection"""
        detected_objects = []
        
        try:
            results = self.model(frame, stream=True)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Ensure box has a valid confidence score
                    confidence = box.conf.numpy()[0]
                    if confidence > self.confidence_threshold:
                        class_id = int(box.cls.numpy()[0])
                        detected_object = self.labels[class_id]
                        detected_objects.append(detected_object)
                        
                        # Hazard detection logic
                        if detected_object in ['car', 'truck', 'bicycle']:
                            self.speak(f"Warning: {detected_object} ahead!")
                            
                            # For proximity warnings, use the center of the box
                            center_x = box.xywh[0][0].numpy()
                            if center_x < frame.shape[1] / 2:
                                self.speak(f"{detected_object} detected to the left")
                            else:
                                self.speak(f"{detected_object} detected to the right")
        
            return detected_objects
        
        except Exception as e:
            self.logger.error(f"Frame processing error: {str(e)}")
            return []

    def run(self):
        """Main loop"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.logger.error("Failed to open camera")
                self.speak("Camera not accessible")
                return

            self.running = True
            self.speak("Starting object detection")
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Failed to read frame")
                    self.speak("Camera feed lost")
                    break

                # Detect objects
                detected_objects = self.process_frame(frame)
                
                # Create and speak announcement
                announcement = self.create_detection_announcement(detected_objects)
                self.speak(announcement)
                
                # Small delay to prevent too frequent announcements
                time.sleep(0.5)

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
            self.speak("Shutting down")
        except Exception as e:
            self.logger.error(f"Runtime error: {str(e)}")
            self.speak("An error occurred")
        finally:
            self.running = False
            if 'cap' in locals():
                cap.release()
            self.logger.info("Shutdown complete")

if __name__ == "_main_":
    try:
        assistant = IntelligentAssistant()
        assistant.run()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)
