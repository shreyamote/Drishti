import cv2
import pyttsx3
import time
from ultralytics import YOLO
import logging
import sys
import speech_recognition as sr
import threading
import queue
import requests
import phonenumbers
import datetime
import webbrowser
import json
import os
from typing import Dict, Any, List
from deep_translator import GoogleTranslator

class IntelligentAssistant:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.3):
        """Initialize the Intelligent Assistant"""
        self.setup_logging()
        self.initialize_components(model_path, confidence_threshold)
        self.load_labels()
        self.assistant_mode = False
        self.load_api_keys()
        self.language = self.select_language()
        self.load_object_descriptions()
        
    def select_language(self) -> str:
        """Let user choose language"""
        self.speak_english("Please choose your language. Say 'English' या 'Hindi' बोलें")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio).lower()
                if 'hindi' in text:
                    self.speak_english("Hindi language selected")
                    return 'hi'
                else:
                    self.speak_english("English language selected")
                    return 'en'
            except Exception:
                self.speak_english("No selection detected, defaulting to English")
                return 'en'

    def load_api_keys(self):
        """Load API keys from environment variables"""
        self.weather_api_key = os.getenv('WEATHER_API_KEY')

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
            
            self.recognizer = sr.Recognizer()
            self.command_queue = queue.Queue()
            
            self.running = False
            self.paused = False
            
            self.context: Dict[str, Any] = {}
            self.conversation_history = []
            self.current_detections = []
            
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            raise

    def load_object_descriptions(self):
        """Load object descriptions and suggestions"""
        self.object_info = {
            "person": {
                "description": {
                    "en": "A human being",
                    "hi": "एक इंसान"
                },
                "suggestion": {
                    "en": "Consider maintaining appropriate social distance",
                    "hi": "उचित सामाजिक दूरी बनाए रखें"
                }
            },
            "car": {
                "description": {
                    "en": "A four-wheeled motor vehicle",
                    "hi": "एक चार पहिया वाहन"
                },
                "suggestion": {
                    "en": "Maintain safe distance from vehicles",
                    "hi": "वाहनों से सुरक्षित दूरी बनाए रखें"
                }
            },
            "default": {
                "description": {
                    "en": "An object in view",
                    "hi": "एक वस्तु दृश्य में"
                },
                "suggestion": {
                    "en": "Observe with caution",
                    "hi": "सावधानी से देखें"
                }
            }
        }

    def translate_text(self, text: str, target_lang: str) -> str:
        """Translate text between English and Hindi"""
        try:
            if target_lang == self.language:
                return text
            translator = GoogleTranslator(source='auto', target=target_lang)
            return translator.translate(text)
        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            return text

    def speak_english(self, text):
        """Speak in English regardless of selected language"""
        try:
            print(f"Assistant: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            self.logger.error(f"Speech error: {str(e)}")

    def speak(self, text):
        """Speak in selected language"""
        try:
            if self.language == 'hi':
                translated_text = self.translate_text(text, 'hi')
            else:
                translated_text = text
            print(f"Assistant: {translated_text}")
            self.engine.say(translated_text)
            self.engine.runAndWait()
        except Exception as e:
            self.logger.error(f"Speech error: {str(e)}")

    def get_weather(self, location: str) -> str:
        """Get weather information for a location"""
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.weather_api_key}&units=metric"
            response = requests.get(url)
            data = response.json()
            
            if response.status_code == 200:
                temp = data['main']['temp']
                condition = data['weather'][0]['description']
                return self.translate_text(
                    f"The weather in {location} is {condition} with a temperature of {temp}°C",
                    self.language
                )
            else:
                return self.translate_text(f"Sorry, I couldn't get the weather for {location}", self.language)
        except Exception as e:
            self.logger.error(f"Weather API error: {str(e)}")
            return self.translate_text("Sorry, there was an error getting the weather information", self.language)

    def make_phone_call(self, number: str) -> str:
        """Simulate making a phone call"""
        try:
            parsed_number = phonenumbers.parse(number, "US")
            if phonenumbers.is_valid_number(parsed_number):
                return self.translate_text(f"Initiating phone call to {number}", self.language)
            return self.translate_text("Invalid phone number", self.language)
        except Exception as e:
            return self.translate_text("Sorry, I couldn't process that phone number", self.language)

    def get_object_info(self, object_name: str) -> tuple:
        """Get description and suggestion for an object"""
        info = self.object_info.get(object_name, self.object_info["default"])
        description = info["description"][self.language]
        suggestion = info["suggestion"][self.language]
        
        if object_name not in self.object_info:
            translated_name = self.translate_text(object_name, self.language)
            if self.language == 'hi':
                description = f"{translated_name} - {description}"
            else:
                description = f"{object_name} - {description}"
                
        return description, suggestion

    def create_detection_announcement(self, detections):
        """Create detailed announcement for detected objects"""
        if not detections:
            return self.translate_text("No objects detected", self.language)
        
        object_count = {}
        descriptions = []
        
        for obj in detections:
            object_count[obj] = object_count.get(obj, 0) + 1
            if obj not in [desc.split(':')[0] for desc in descriptions]:
                description, suggestion = self.get_object_info(obj)
                descriptions.append(f"{obj}: {description}. {suggestion}")
        
        counts_text = ", ".join(
            f"{count} {obj}{'s' if count > 1 else ''}" 
            for obj, count in object_count.items()
        )
        
        detailed_text = "\n".join(descriptions)
        
        return f"I can see: {counts_text}\n\nDetails:\n{detailed_text}"

    def process_assistant_command(self, command: str) -> str:
        """Process general assistant commands"""
        command = command.lower()
        
        # Weather queries
        if "weather" in command:
            location = command.replace("weather", "").replace("in", "").strip()
            if location:
                return self.get_weather(location)
            return self.translate_text("Please specify a location for the weather", self.language)
            
        # Translation command
        elif "translate" in command:
            text_to_translate = command.replace("translate", "").strip()
            if self.language == 'hi':
                return f"English translation: {self.translate_text(text_to_translate, 'en')}"
            else:
                return f"Hindi translation: {self.translate_text(text_to_translate, 'hi')}"
            
        # Phone calls
        elif "call" in command:
            number = ''.join(filter(str.isdigit, command))
            if number:
                return self.make_phone_call(number)
            return self.translate_text("Please provide a phone number to call", self.language)
            
        # Time queries
        elif "time" in command:
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            return self.translate_text(f"The current time is {current_time}", self.language)
            
        # Web searches
        elif "search for" in command:
            query = command.replace("search for", "").strip()
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return self.translate_text(f"Searching for {query}", self.language)
            
        # Object detection specific commands
        elif "what do you see" in command:
            return self.create_detection_announcement(self.current_detections)
            
        # Language switch command
        elif "change language" in command:
            self.language = 'en' if self.language == 'hi' else 'hi'
            return self.translate_text(
                f"Language changed to {'Hindi' if self.language == 'hi' else 'English'}", 
                self.language
            )
            
        # Exit assistant mode
        elif command in ["goodbye", "stop", "exit", "बंद करो"]:
            self.assistant_mode = False
            return self.translate_text("Goodbye! Returning to normal monitoring mode", self.language)
            
        # Help command
        elif "help" in command:
            help_text = """I can help you with:
                   - Weather information (e.g., 'weather in London')
                   - Translations (e.g., 'translate hello')
                   - Making phone calls (e.g., 'call 123456789')
                   - Telling the time (e.g., 'what time is it')
                   - Web searches (e.g., 'search for cats')
                   - Describing what I see (e.g., 'what do you see')
                   - Changing language (say 'change language')
                   Say 'goodbye' to exit assistant mode"""
            return self.translate_text(help_text, self.language)
                   
        # Default response
        return self.translate_text(
            "I'm sorry, I didn't understand that command. Say 'help' for a list of things I can do.",
            self.language
        )

    def process_frame(self, frame):
        """Process a single frame for object detection with hazard detection"""
        detected_objects = []
        
        try:
            results = self.model(frame, stream=True)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = box.conf.numpy()[0]
                    if confidence > self.confidence_threshold:
                        class_id = int(box.cls.numpy()[0])
                        detected_object = self.labels[class_id]
                        detected_objects.append(detected_object)
                        
                        # Only announce hazards when not in assistant mode
                        if not self.assistant_mode and detected_object in ['car', 'truck', 'bicycle']:
                            warning = self.translate_text(f"Warning: {detected_object} ahead!", self.language)
                            self.speak(warning)
                            
                            # Proximity warning
                            center_x = box.xywh[0][0].numpy()
                            if center_x < frame.shape[1] / 2:
                                position = self.translate_text(
                                    f"{detected_object} detected to the left",
                                    self.language
                                )
                            else:
                                position = self.translate_text(
                                    f"{detected_object} detected to the right",
                                    self.language
                                )
                            self.speak(position)
        
            return detected_objects
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {str(e)}")
            return []

    def listen_for_wake_word(self):
        """Listen for the wake word and subsequent commands"""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while self.running:
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    text = self.recognizer.recognize_google(audio).lower()
                    
                    if not self.assistant_mode and "hello assistant" in text:
                        self.assistant_mode = True
                        self.command_queue.put("hello")
                    elif self.assistant_mode:
                        self.command_queue.put(text)
                        
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except sr.RequestError:
                    self.logger.error("Could not request results from speech recognition service")
                except Exception as e:
                    self.logger.error(f"Voice recognition error: {str(e)}")

    def run(self):
        """Main loop"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.logger.error("Failed to open camera")
                self.speak("Camera not accessible")
                return

            self.running = True
            self.speak("Starting enhanced assistant system")
            
            voice_thread = threading.Thread(target=self.listen_for_wake_word, daemon=True)
            voice_thread.start()
            
            while self.running:
                if self.assistant_mode:
                    try:
                        command = self.command_queue.get_nowait()
                        if command == "hello":
                            response = self.translate_text(
                                "Hello! I'm here to help. You can ask about the weather, translations, make calls, or say 'what do you see' for object detection.",
                                self.language
                            )
                        else:
                            response = self.process_assistant_command(command)
                        self.speak(response)
                    except queue.Empty:
                        pass

                # Continue object detection regardless of mode
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Failed to read frame")
                    self.speak("Camera feed lost")
                    break

                self.current_detections = self.process_frame(frame)
                
                # Only make automatic announcements in normal mode
                if not self.assistant_mode:
                    announcement = self.create_detection_announcement(self.current_detections)
                    self.speak(announcement)
                
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

if __name__ == "__main__":
    try:
        assistant = IntelligentAssistant()
        assistant.run()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

