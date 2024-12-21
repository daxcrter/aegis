import asyncio
import cv2
import json
import numpy as np
import websockets
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

@dataclass
class Frame:
    faces: int
    motion: float

class Cam:
    def __init__(self, url: str = "ws://localhost:5000"):
        self.url = url
        self.prev = None
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        
        # Face detection
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        
        # FPS settings
        self.min_fps = 10
        self.max_fps = 30
        self.fps = self.min_fps
        self.last_fps_update = 0
        self.fps_cooldown = 2.0
        
        # Stats
        self.frames = 0
        self.errors = 0
        self.max_errors = 5
        
        # Load Montserrat font
        try:
            self.font = ImageFont.truetype("Montserrat-Regular.ttf", 20)
            self.font_bold = ImageFont.truetype("Montserrat-Bold.ttf", 24)
        except:
            # Fallback to a default font if Montserrat is not available
            print("Montserrat font not found, using default font")
            self.font = None
            self.font_bold = None
        
        # Recording indicator animation
        self.rec_alpha = 0
        self.rec_increasing = True

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw camera overlay with Montserrat font"""
        # Convert to PIL Image for better text rendering
        pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_im)
        
        # Background overlay for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Current time
        time = datetime.now().strftime("%H:%M:%S")
        date = datetime.now().strftime("%Y-%m-%d")
        
        # Recording indicator animation
        if self.rec_increasing:
            self.rec_alpha += 0.1
            if self.rec_alpha >= 1.0:
                self.rec_increasing = False
        else:
            self.rec_alpha -= 0.1
            if self.rec_alpha <= 0.0:
                self.rec_increasing = True
        
        # Draw text with Montserrat (or fallback to OpenCV if font not available)
        if self.font and self.font_bold:
            # Time and date
            draw.text((20, 10), f"{time}", font=self.font_bold, fill=(255, 255, 255))
            draw.text((20, 40), f"{date}", font=self.font, fill=(200, 200, 200))
            
            # FPS and recording status
            draw.text((200, 10), f"FPS: {self.fps}", font=self.font, fill=(200, 200, 200))
            rec_color = (255, 0, 0) if self.rec_alpha > 0.5 else (150, 0, 0)
            draw.text((200, 40), "REC", font=self.font_bold, fill=rec_color)
            
            # Convert back to OpenCV format
            frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        else:
            # Fallback to OpenCV text rendering
            cv2.putText(frame, f"{time}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"{date}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.putText(frame, f"FPS: {self.fps}", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            rec_color = (0, 0, 255) if self.rec_alpha > 0.5 else (0, 0, 150)
            cv2.putText(frame, "REC", (200, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2)
        
        return frame

    def analyze(self, frame: np.ndarray) -> Frame:
        """Analyze frame for faces and motion"""
        try:
            # Face detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb)
            
            faces = 0
            if results.detections:
                faces = len(results.detections)
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    h, w = frame.shape[:2]
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Draw face box
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    
                    # Confidence score
                    conf = f"{int(det.score[0] * 100)}%"
                    cv2.putText(frame, conf, (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Motion detection
            mask = self.bg_sub.apply(frame)
            motion = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
            
            if motion > 0.01:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 100:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            return Frame(faces, motion)
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return Frame(0, 0.0)

    def calc_fps(self, frame: Frame) -> int:
        """Calculate target FPS based on activity"""
        base = self.min_fps
        
        # Increase for faces
        if frame.faces > 0:
            base += frame.faces * 5
        
        # Increase for motion
        if frame.motion > 0.01:
            motion_fps = int(frame.motion * 100)
            base += motion_fps
        
        return min(max(base, self.min_fps), self.max_fps)

    async def process(self, data: bytes) -> Optional[np.ndarray]:
        """Process received frame data"""
        try:
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Frame decode failed")
            
            return frame
            
        except Exception as e:
            print(f"Process error: {e}")
            self.errors += 1
            return None

    async def run(self):
        """Main loop"""
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    print(f"Connected to {self.url}")
                    self.errors = 0

                    while True:
                        # Get frame
                        data = await ws.recv()
                        frame = await self.process(data)
                        
                        if frame is None:
                            if self.errors >= self.max_errors:
                                print("Too many errors, reconnecting...")
                                break
                            continue

                        # Analyze and update FPS
                        analysis = self.analyze(frame)
                        target = self.calc_fps(analysis)
                        
                        if target != self.fps:
                            req = {
                                "target_fps": target,
                                "faces_detected": analysis.faces,
                                "motion_level": float(analysis.motion)
                            }
                            await ws.send(json.dumps(req))
                            self.fps = target
                        
                        # Add overlay and display
                        frame = self.draw_overlay(frame)
                        cv2.imshow('Security Camera', frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            return

            except websockets.exceptions.ConnectionClosed:
                print("Connection lost , reconnecting...")
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(1)

if __name__ == "__main__":
    cam = Cam()
    asyncio.run(cam.run())