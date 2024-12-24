import asyncio
import cv2
import numpy as np
import websockets
import mediapipe as mp
import face_recognition
from datetime import datetime
from pathlib import Path
from queue import Queue
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from PIL import ImageFont, ImageDraw, Image
import time
import uuid
import os

@dataclass
class Frame:
    id: str
    timestamp: float
    data: np.ndarray

class Storage:
    def __init__(self, directory: str = "frames"):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True)
        
    def save(self, frame: Frame) -> None:
        path = self.directory / f"{frame.id}.npy"
        np.save(str(path), frame.data)
        
    def load(self, frame_id: str) -> Optional[np.ndarray]:
        path = self.directory / f"{frame_id}.npy"
        if path.exists():
            return np.load(str(path))
        return None
        
    def cleanup(self, keep_ids: List[str]) -> None:
        for path in self.directory.glob("*.npy"):
            if path.stem not in keep_ids:
                path.unlink()

class Queue:
    def __init__(self, maxsize: int = 100):
        self.raw = asyncio.Queue(maxsize)
        self.processed = asyncio.Queue(maxsize)
        self.frames: Dict[str, float] = {}
        self.storage = Storage()
        self.cleanup_threshold = maxsize * 2
        
    async def add_raw(self, data: np.ndarray) -> None:
        frame_id = str(uuid.uuid4())
        frame = Frame(frame_id, time.time(), data)
        self.frames[frame_id] = frame.timestamp
        self.storage.save(frame)
        await self.raw.put(frame_id)
        
        if len(self.frames) > self.cleanup_threshold:
            self._cleanup()
            
    async def add_processed(self, frame_id: str) -> None:
        await self.processed.put(frame_id)
        
    async def get_raw(self) -> Optional[Tuple[str, np.ndarray]]:
        if self.raw.empty():
            return None
        frame_id = await self.raw.get()
        data = self.storage.load(frame_id)
        return frame_id, data
        
    async def get_processed(self) -> Optional[np.ndarray]:
        if self.processed.empty():
            return np.zeros((480, 640, 3), dtype=np.uint8)
        frame_id = await self.processed.get()
        return self.storage.load(frame_id)
        
    def _cleanup(self) -> None:
        current = time.time()
        expired = [fid for fid, ts in self.frames.items() 
                  if current - ts > 30]  # Keep last 30 seconds
        for fid in expired:
            del self.frames[fid]
        self.storage.cleanup(list(self.frames.keys()))

class Processor:
    def __init__(self):
        self.face_det = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.7)
        self.faces: Dict[str, List[np.ndarray]] = {}
        self.last_frame = None
        self._load_faces()
        
    def _load_faces(self):
        face_dir = Path('faces')
        if not face_dir.exists():
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

    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        res = self.face_det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if res.detections:
            for det in res.detections:
                box = det.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y = int(box.xmin * w), int(box.ymin * h)
                width, height = int(box.width * w), int(box.height * h)
                
                if width > 0 and height > 0:
                    face_img = frame[y:y+height, x:x+width]
                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face = face_recognition.face_encodings(face_img_rgb)
                    
                    name = "Unknown"
                    if face:
                        for known_name, known_faces in self.faces.items():
                            if any(face_recognition.compare_faces(
                                known_faces, face[0])):
                                name = known_name
                                break
                    
                    cv2.rectangle(frame, (x, y), (x + width, y + height),
                                (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame

class Client:
    def __init__(self, url: str = "ws://localhost:5000"):
        self.url = url
        self.processor = Processor()
        self.queue = Queue()
        self.running = True
        
    async def connect(self):
        while self.running:
            try:
                async with websockets.connect(self.url) as ws:
                    await asyncio.gather(
                        self._receive(ws),
                        self._process(),
                        self._display()
                    )
            except Exception:
                await asyncio.sleep(1)
                
    async def _receive(self, ws):
        while self.running:
            try:
                data = await ws.recv()
                arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    await self.queue.add_raw(frame)
            except:
                break
                
    async def _process(self):
        while self.running:
            frame_data = await self.queue.get_raw()
            if frame_data:
                frame_id, frame = frame_data
                processed = self.processor.process(frame)
                await self.queue.add_processed(frame_id)
            await asyncio.sleep(0.01)
                
    async def _display(self):
        while self.running:
            frame = await self.queue.get_processed()
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
            await asyncio.sleep(0.01)
        
    def stop(self):
        self.running = False

async def main():
    client = Client()
    try:
        await client.connect()
    finally:
        client.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        cv2.destroyAllWindows()