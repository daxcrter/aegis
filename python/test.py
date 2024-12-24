import asyncio
import cv2
import numpy as np
import websockets
import mediapipe as mp
import face_recognition
import logging
from datetime import datetime
from pathlib import Path
import math
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

class Processor:
    def __init__(self):
        self.face_det = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
        self.faces: Dict[str, List[np.ndarray]] = {}
        self.last_frame = None
        self.mot_thresh = 30
        self.angle = 0
        
        self._load_faces()
    
    def _load_faces(self):
        face_dir = Path('faces')
        if not face_dir.exists():
            log.error("[ERROR] No faces dir")
            return
        
        for f in face_dir.glob('*'):
            if not f.is_file():
                continue
            
            name = f.stem.split('-')[0]
            img = face_recognition.load_image_file(str(f))
            enc = face_recognition.face_encodings(img)
            
            if enc:
                if name not in self.faces:
                    self.faces[name] = []
                self.faces[name].append(enc[0])
        
        log.info(f"[SYSTEM] Loaded {len(self.faces)} faces")
    
    def _detect_motion(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.last_frame is None:
            self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self.last_frame, gray)
        self.last_frame = gray
        
        _, thresh = cv2.threshold(diff, self.mot_thresh, 255, cv2.THRESH_BINARY)
        cons, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not cons:
            return []
        
        # Single box for all motion
        xmin = ymin = float('inf')
        xmax = ymax = float('-inf')
        
        for con in cons:
            x, y, w, h = cv2.boundingRect(con)
            xmin = min(xmin, x)
            ymin = min(ymin, y)
            xmax = max(xmax, x + w)
            ymax = max(ymax, y + h)
        
        return [(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin))]
    
    def _draw_rec(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        r = 15
        center = (w - r - 10, r + 10)
        
        pulse = abs(math.sin(self.angle)) * 5
        self.angle += 0.1
        
        cv2.circle(frame, center, int(r + pulse), (0, 0, 255), 2)
        cv2.circle(frame, center, int(r - 5), (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 70, r * 2 + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        # Face detection
        res = self.face_det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
        if res.detections:
            for det in res.detections:
                box = det.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y = int(box.xmin * w), int(box.ymin * h)
                width, height = int(box.width * w), int(box.height * h)
            
                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
            
                # Only attempt face encoding if we have a valid region
                if width > 0 and height > 0:
                    face_img = frame[y:y+height, x:x+width]
                    # Convert to RGB as face_recognition expects RGB
                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face = face_recognition.face_encodings(face_img_rgb)
                
                    name = "Unknown"
                    if face:
                        for known_name, known_faces in self.faces.items():
                            if any(face_recognition.compare_faces(known_faces, face[0])):
                                name = known_name
                                break
                
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    conf = f"{det.score[0]*100:.1f}%"
                    cv2.putText(frame, f"{conf} {name}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
        # Motion detection
        for x, y, w, h in self._detect_motion(frame):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
        self._draw_rec(frame)
    
        time = datetime.now().strftime("%I:%M %p")
        cv2.putText(frame, time, (frame.shape[1] - 100, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
        return frame
    
class Client:
    def __init__(self, url: str = "ws://localhost:5000"):
        self.url = url
        self.proc = Processor()
        self.delay = 1
        self.max_delay = 30

        # Initialize VideoWriter
        self.out = None
        self.frame_width = 640  # Set default width
        self.frame_height = 480  # Set default height
        self.fps = 30  # Set default FPS
        self.video_file = "processed_video.avi"  # Output file name

    async def connect(self):
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                   
                    await self._handle(ws)
            except Exception as e:
                log.error(f"[ERROR] Connection failed: {e}")
                log.info(f"[WEBSOCKET] Retry in {self.delay}s")
                await asyncio.sleep(self.delay)
                self.delay = min(self.delay * 2, self.max_delay)
    
    async def _handle(self, ws):
        while True:
            try:
                data = await ws.recv()
                log.info("[VIDEO] Chunk received")
                
                arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    log.error("[ERROR] Frame decode failed")
                    continue
                
                if self.out is None:
                    # Initialize VideoWriter once we get the first frame
                    self.frame_height, self.frame_width = frame.shape[:2]
                    self.out = cv2.VideoWriter(
                        self.video_file,
                        cv2.VideoWriter_fourcc(*'XVID'),  # Codec (e.g., XVID)
                        self.fps,
                        (self.frame_width, self.frame_height)
                    )

                frame = self.proc.process(frame)

                # Write processed frame to the output video file
                self.out.write(frame)

                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except websockets.exceptions.ConnectionClosed:
                log.info("[WEBSOCKET] Disconnected")
                break
            except Exception as e:
                log.error(f"[ERROR] Processing error: {e}")
                break

    def release_video_writer(self):
        if self.out:
            self.out.release()
            log.info(f"[VIDEO] Saved processed video to {self.video_file}")

async def main():
    client = Client()
    try:
        await client.connect()
    finally:
        client.release_video_writer()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        cv2.destroyAllWindows()
