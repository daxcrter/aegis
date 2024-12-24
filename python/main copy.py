import asyncio
import cv2
import numpy as np
import websockets
from dataclasses import dataclass
from datetime import datetime
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import threading
import time
import face_recognition
import os
import queue

@dataclass
class Frame:
    faces: int
    motion: float
    names: list

class Cam:
    def __init__(self, url: str = "ws://localhost:5000"):
        self.url = url
        self.prev = None
        self.bgsub = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        
        # Face setup
        self.mpface = mp.solutions.face_detection
        self.detector = self.mpface.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        
        # Face recognition
        self.faces = []
        self.names = []
        self.loadfaces("faces")
        
        # FPS calculation
        self.fps = 0
        self.frame_times = []
        self.last_frame_time = time.time()
        
        # Motion
        self.lastmotion = 0.0
        self.threshold = 0.5
        
        # Stats
        self.frames = 0
        self.errors = 0
        self.maxerrors = 5
        
        # Frame handling
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        
        # UI
        self.font = ImageFont.truetype("fonts/CalSans.woff2", 20)
        self.bold = ImageFont.truetype("fonts/CalSans.woff2", 24)
        
        # Recording animation
        self.rec_visible = True
        self.last_rec_toggle = time.time()
        self.rec_toggle_interval = 0.5  # Toggle every 0.5 seconds
        
        # Default frame
        self.blank = np.zeros((480, 640, 3), dtype=np.uint8)

        # Frame queues
        self.raw_queue = queue.Queue(maxsize=2)  # Queue for raw frame batches
        self.processed_queue = queue.Queue(maxsize=2)  # Queue for processed frame batches

    def loadfaces(self, dir: str):
        """Load faces from directory"""
        for file in os.listdir(dir):
            if file.endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(dir, file)
                img = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(img)
                
                if encodings:
                    self.faces.append(encodings[0])
                    name = os.path.splitext(file)[0]
                    self.names.append(name)
                    print(f"Loaded face: {name}")

    def overlay(self, frame):
        """Create overlay with UI elements"""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
    
        # Layout
        pad = 30
        top = 60
        rad = 12
        center = top // 2
    
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], top), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
        now = datetime.now().strftime("%H:%M:%S")
    
        timewidth = draw.textlength(now, font=self.bold)
        fpswidth = draw.textlength(f"{self.fps:.1f} FPS", font=self.font)
        right = frame.shape[1] - pad
        
        # Right side
        draw.text((right - timewidth, 10), now, font=self.bold, fill=(255, 255, 255))
        draw.text((right - fpswidth, 35), f"{self.fps:.1f} FPS", font=self.font, fill=(200, 200, 200))
        
        # Recording indicator
        x = pad + rad
        
        # Toggle REC visibility
        current_time = time.time()
        if current_time - self.last_rec_toggle >= self.rec_toggle_interval:
            self.rec_visible = not self.rec_visible
            self.last_rec_toggle = current_time
        
        if self.rec_visible:
            # Main circle
            draw.ellipse(
                [(x - rad, center - rad), (x + rad, center + rad)],
                fill=(255, 0, 0)
            )
            
            # REC text
            draw.text((x + rad + 8, 20), "REC", font=self.bold, fill=(255, 255, 255))

        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return frame

    def display(self):
        """Display thread"""
        cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera', 640, 480)
        cv2.imshow('Camera', self.blank)

        while self.running:
            try:
                batch = self.processed_queue.get(timeout=0.1)
                for frame in batch:
                    if not self.running:
                        break
                    frame_with_overlay = self.overlay(frame)
                    cv2.imshow('Camera', frame_with_overlay)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
            except queue.Empty:
                continue

        cv2.destroyAllWindows()

    def analyze(self, frame: np.ndarray) -> Frame:
        """Analyze frame"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb)
            
            faces = 0
            names = []
            
            if results.detections:
                faces = len(results.detections)
                locs = []
                
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    h, w = frame.shape[:2]
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    locs.append((y, x + width, y + height, x))
                
                encodings = face_recognition.face_encodings(rgb, locs)
                
                for (det, enc) in zip(results.detections, encodings):
                    bbox = det.location_data.relative_bounding_box
                    h, w = frame.shape[:2]
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    matches = face_recognition.compare_faces(self.faces, enc, tolerance=0.6)
                    name = "Unknown"
                    
                    if True in matches:
                        idx = matches.index(True)
                        name = self.names[idx]
                    
                    names.append(name)
                    
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    label = f"{int(det.score[0] * 100)}% - {name}"
                    cv2.putText(frame, label, (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Motion
            mask = self.bgsub.apply(frame)
            motion = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
            
            if motion > 0.01:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 100:
                        x, y, w, h = cv2.boundingRect(cnt)
            
            return Frame(faces, motion, names)
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return Frame(0, 0.0, [])

    def update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        self.frame_times.append(current_time - self.last_frame_time)
        self.last_frame_time = current_time

        # Keep only the last 10 frame times for a moving average
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)

        if self.frame_times:
            self.fps = 1 / (sum(self.frame_times) / len(self.frame_times))

    async def process(self, data: bytes) -> np.ndarray:
        """Process frame data"""
        try:
            arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Decode failed")
            
            return frame
            
        except Exception as e:
            print(f"Process error: {e}")
            self.errors += 1
            return None

    def process_batch(self):
        """Process a batch of frames in the background"""
        while self.running:
            try:
                batch = self.raw_queue.get(timeout=0.1)
                processed_batch = []
                for frame in batch:
                    processed_frame = self.analyze(frame)
                    processed_batch.append(frame)  # Append the original frame with analysis results
                    self.update_fps()
                self.processed_queue.put(processed_batch)
            except queue.Empty:
                continue

    async def run(self):
        """Main loop"""
        display_thread = threading.Thread(target=self.display)
        display_thread.start()

        process_thread = threading.Thread(target=self.process_batch)
        process_thread.start()
        
        while self.running:
            try:
                async with websockets.connect(self.url, max_size=100 * 1024 * 1024) as ws:
                    print(f"Connected to {self.url}")
                    self.errors = 0

                    while self.running:
                        try:
                            # Receive frame count
                            frame_count_data = await asyncio.wait_for(ws.recv(), timeout=0.5)
                            frame_count = int.from_bytes(frame_count_data, byteorder='little')

                            # Receive frames
                            batch = []
                            for _ in range(frame_count):
                                data = await asyncio.wait_for(ws.recv(), timeout=0.5)
                                frame = await self.process(data)
                                if frame is not None:
                                    batch.append(frame)

                            if batch:
                                self.raw_queue.put(batch)
                            
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            print(f"Frame error: {e}")
                            await asyncio.sleep(0.1)

            except websockets.exceptions.ConnectionClosed:
                print("Connection lost, reconnecting...")
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(1)
        
        display_thread.join()
        process_thread.join()

if __name__ == "__main__":
    cam = Cam()
    asyncio.run(cam.run())