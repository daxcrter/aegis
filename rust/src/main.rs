use anyhow::Result;
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt, TryStreamExt}; 
use image::{ImageBuffer, Rgb};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution},
    Camera, query,
};
use serde::{Deserialize, Serialize};
use std::{
    sync::Arc,
    time::Duration,
};
use tokio::{
    sync::Mutex,
    time::sleep,
};
use tokio_tungstenite::{
    accept_async,
    tungstenite::Message,
};

// Constants
const MIN_FPS: u32 = 10;
const MAX_FPS: u32 = 30;
const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const PORT: u16 = 5000;

#[derive(Serialize, Deserialize, Debug)]
struct FrameRateRequest {
    target_fps: u32,
    faces_detected: u32,
    motion_level: f32,
}

#[derive(Serialize, Deserialize, Debug)]
struct CameraState {
    current_fps: u32,
    is_streaming: bool,
    error_count: u32,
}

struct AdaptiveCamera {
    camera: Camera,
    state: Arc<Mutex<CameraState>>,
}

impl AdaptiveCamera {
    fn new() -> Result<Self> {
        let fmt = RequestedFormat::new::<RgbFormat>(
            RequestedFormatType::Exact(
                CameraFormat::new(
                    Resolution::new(WIDTH, HEIGHT),
                    FrameFormat::NV12,
                    MIN_FPS,
                )
            )
        );

        let mut camera = Camera::new(
            CameraIndex::Index(0),
            fmt,
        )?;

        camera.open_stream()?;

        let state = Arc::new(Mutex::new(CameraState {
            current_fps: MIN_FPS,
            is_streaming: true,
            error_count: 0,
        }));

        Ok(Self { camera, state })
    }

    async fn update_framerate(&mut self, request: FrameRateRequest) -> Result<()> {
        let mut state = self.state.lock().await;
        
        // Calculate new FPS based on detection results
        let target_fps = (request.target_fps)
            .min(MAX_FPS)
            .max(MIN_FPS);

        if target_fps != state.current_fps {
            // Update camera format with new FPS
            let fmt = RequestedFormat::new::<RgbFormat>(
                RequestedFormatType::Exact(
                    CameraFormat::new(
                        Resolution::new(WIDTH, HEIGHT),
                        FrameFormat::NV12,
                        target_fps,
                    )
                )
            );

            // Safely update camera settings
            self.camera.stop_stream()?;
            self.camera.set_camera_requset(fmt)?;  // Fixed typo in method name
            self.camera.open_stream()?;
            
            state.current_fps = target_fps;
        }

        Ok(())
    }

    async fn capture_frame(&mut self) -> Result<Vec<u8>> {
        // Move state acquisition after potential error handling
        match self.camera.frame() {
            Ok(frame) => {
                let rgb = frame.decode_image::<RgbFormat>()?;
                let resolution = frame.resolution();
                
                // Create image buffer
                let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(
                    resolution.width(),
                    resolution.height(),
                    rgb.into_raw(),
                ).ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;

                // Encode to JPEG
                let mut jpg = Vec::new();
                let mut enc = image::codecs::jpeg::JpegEncoder::new(&mut jpg);
                enc.encode(
                    img.as_raw(),
                    resolution.width(),
                    resolution.height(),
                    image::ColorType::Rgb8,
                )?;

                let mut state = self.state.lock().await;
                state.error_count = 0;
                Ok(jpg)
            }
            Err(e) => {
                let mut state = self.state.lock().await;
                state.error_count += 1;
                
                // If too many errors, try to reset camera
                if state.error_count > 5 {
                    drop(state);  // Drop the lock before calling reset_camera
                    self.reset_camera().await?;
                }
                
                Err(e.into())
            }
        }
    }

    async fn reset_camera(&mut self) -> Result<()> {
        let mut state = self.state.lock().await;
        
        // Stop current stream
        self.camera.stop_stream()?;
        
        // Reset to minimum FPS
        let fmt = RequestedFormat::new::<RgbFormat>(
            RequestedFormatType::Exact(
                CameraFormat::new(
                    Resolution::new(WIDTH, HEIGHT),
                    FrameFormat::NV12,
                    MIN_FPS,
                )
            )
        );

        self.camera.set_camera_requset(fmt)?;  // Fixed typo in method name
        self.camera.open_stream()?;
        
        state.current_fps = MIN_FPS;
        state.error_count = 0;
        
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Query available cameras
    let cameras = query(ApiBackend::Auto)?;
    for camera in cameras {
        println!("Found camera: {:?}", camera);
    }

    let camera = Arc::new(Mutex::new(AdaptiveCamera::new()?));
    println!("Camera initialized at {}x{}", WIDTH, HEIGHT);

    let addr = format!("127.0.0.1:{}", PORT);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("WebSocket server listening on ws://{}", addr);

    while let Ok((stream, _)) = listener.accept().await {
        let camera = camera.clone();
        
        tokio::spawn(async move {
            let ws_stream = accept_async(stream).await.expect("Failed to accept connection");
            let (mut ws_tx, mut ws_rx) = ws_stream.split();
            
            println!("New client connected");

            loop {
                let mut camera = camera.lock().await;
                
                // Handle incoming messages (frame rate requests)
                if let Ok(Some(msg)) = ws_rx.try_next().await {  // Added .await
                    if let Ok(request) = serde_json::from_str(&String::from_utf8_lossy(&msg.into_data())) {
                        if let Err(e) = camera.update_framerate(request).await {
                            eprintln!("Failed to update framerate: {}", e);
                        }
                    }
                }

                // Capture and send frame
                match camera.capture_frame().await {
                    Ok(jpg) => {
                        if let Err(e) = ws_tx.send(Message::Binary(Bytes::from(jpg))).await {
                            eprintln!("Failed to send frame: {}", e);
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("Frame capture error: {}", e);
                        sleep(Duration::from_millis(100)).await;
                    }
                }

                let state = camera.state.lock().await;
                let delay = Duration::from_secs(1) / state.current_fps;
                drop(state);
                
                sleep(delay).await;
            }
        });
    }

    Ok(())
}