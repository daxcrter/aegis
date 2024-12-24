use tokio::{self, sync::mpsc, time::Duration};
use tokio_tungstenite::accept_async;
use futures::{SinkExt, StreamExt};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraFormat, FrameFormat, CameraIndex, RequestedFormat, RequestedFormatType, Resolution},
    Camera,
};
use std::sync::Arc;
use tokio::sync::Mutex;
use chrono::Local;
use log::{error, info};
use std::collections::VecDeque;
use std::thread;

#[derive(Clone)]
struct Frame {
    data: Vec<u8>,
    ts: i64,
}

#[derive(Clone)]
struct Chunk {
    frames: Vec<Frame>,
    start_ts: i64,
}

struct Queue {
    chunks: VecDeque<Chunk>,
    current_chunk: Option<Chunk>,
    chunk_duration: i64,
}

impl Queue {
    fn new(chunk_duration: i64) -> Self {
        Self {
            chunks: VecDeque::new(),
            current_chunk: None,
            chunk_duration,
        }
    }

    fn push_frame(&mut self, frame: Frame) {
        match &mut self.current_chunk {
            Some(chunk) => {
                // If chunk duration exceeded, finalize current chunk
                if frame.ts - chunk.start_ts >= self.chunk_duration {
                    let complete_chunk = std::mem::replace(&mut self.current_chunk, 
                        Some(Chunk {
                            frames: vec![frame],
                            start_ts: frame.ts,
                        })
                    );
                    if let Some(chunk) = complete_chunk {
                        self.chunks.push_back(chunk);
                    }
                } else {
                    chunk.frames.push(frame);
                }
            }
            None => {
                // Start new chunk
                self.current_chunk = Some(Chunk {
                    frames: vec![frame],
                    start_ts: frame.ts,
                });
            }
        }
    }

    fn pop_chunk(&mut self) -> Option<Chunk> {
        self.chunks.pop_front()
    }

    fn finalize_current(&mut self) {
        if let Some(chunk) = self.current_chunk.take() {
            if !chunk.frames.is_empty() {
                self.chunks.push_back(chunk);
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let addr = "127.0.0.1:5000";
    let queue = Arc::new(Mutex::new(Queue::new(10))); // 10 second chunks
    let (tx, rx) = mpsc::channel(32);
    let rx = Arc::new(Mutex::new(rx));
    
    // Init camera in a separate thread
    let q = Arc::clone(&queue);
    let tx_clone = tx.clone();
    
    thread::spawn(move || {
        let mut cam = match Camera::new(
            CameraIndex::Index(0), 
            RequestedFormat::new::<RgbFormat>(
                RequestedFormatType::Exact(
                    CameraFormat::new(
                        Resolution::new(1280, 720),
                        FrameFormat::MJPEG,
                        30,
                    ),
                ),
            ),
        ) {
            Ok(cam) => cam,
            Err(e) => {
                error!("[ERROR] Failed to initialize camera: {}", e);
                return;
            }
        };

        if let Err(e) = cam.open_stream() {
            error!("[ERROR] Failed to open camera stream: {}", e);
            return;
        }

        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut last_finalize = Local::now().timestamp();
        
        loop {
            match cam.frame() {
                Ok(frame) => {
                    let current_ts = Local::now().timestamp();
                    let frame = Frame {
                        data: frame.buffer().to_vec(),
                        ts: current_ts,
                    };
                    
                    rt.block_on(async {
                        let mut queue = q.lock().await;
                        queue.push_frame(frame.clone());
                        
                        // Check if it's time to finalize current chunk
                        if current_ts - last_finalize >= 10 {
                            queue.finalize_current();
                            // Send the latest complete chunk
                            if let Some(chunk) = queue.pop_chunk() {
                                if let Err(e) = tx_clone.send(chunk).await {
                                    error!("[ERROR] Send failed: {}", e);
                                }
                            }
                            last_finalize = current_ts;
                        }
                    });
                }
                Err(e) => {
                    error!("[ERROR] Frame capture failed: {}", e);
                }
            }
            
            thread::sleep(Duration::from_millis(33)); // ~30 fps
        }
    });
    
    // WS server
    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    while let Ok((stream, _)) = listener.accept().await {
        let ws = accept_async(stream).await?;
        let (mut sender, _) = ws.split();
        let rx = Arc::clone(&rx);
        let q = Arc::clone(&queue);
        
        tokio::spawn(async move {
            // Send queued chunks first
            {
                let mut queue = q.lock().await;
                queue.finalize_current(); // Finalize any in-progress chunk
                while let Some(chunk) = queue.pop_chunk() {
                    // Combine all frames in chunk into single buffer
                    let mut combined_data = Vec::new();
                    for frame in chunk.frames {
                        combined_data.extend(frame.data);
                    }
                    
                    if let Err(e) = sender.send(combined_data.into()).await {
                        error!("[ERROR] Queue send failed: {}", e);
                        break;
                    }
                }
            }
            
            // Handle new chunks
            let mut rx = rx.lock().await;
            while let Some(chunk) = rx.recv().await {
                let mut combined_data = Vec::new();
                for frame in chunk.frames {
                    combined_data.extend(frame.data);
                }
                
                match sender.send(combined_data.into()).await {
                    Ok(_) => {}
                    Err(e) => {
                        error!("[ERROR] Send failed: {}", e);
                        q.lock().await.push_frame(Frame {
                            data: combined_data,
                            ts: Local::now().timestamp(),
                        });
                        break;
                    }
                }
            }
        });
    }
    
    Ok(())
}