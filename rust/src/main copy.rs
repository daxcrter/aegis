use anyhow::Result;
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt, FutureExt};
use image::{ImageBuffer, Rgb};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution},
    Camera,
};

use serde::Deserialize;
use std::time::Duration;
use tokio::time::{sleep, Instant};
use tokio_tungstenite::{accept_async, tungstenite::Message};


const PORT: u16 = 5000;

#[derive(Deserialize)]
struct Req {
    target_fps: u32,
    faces_detected: u32,
    motion_level: f32,
}

struct Cam {
    dev: Camera,
}
unsafe impl Send for Cam {}
impl Cam {
    fn new() -> Result<Self> {
        let fmt = RequestedFormat::new::<RgbFormat>(
            RequestedFormatType::Exact(
                CameraFormat::new(
                    Resolution::new(1280, 720),
                    FrameFormat::MJPEG,
                    30,
                ),
            ),
        );

        let mut dev = Camera::new(CameraIndex::Index(0), fmt)?;
        dev.open_stream()?;

        Ok(Self { 
            dev,
 
        })
    }


    fn capture(&mut self) -> Result<Vec<u8>> {
        let frame = self.dev.frame()?;
        let rgb = frame.decode_image::<RgbFormat>()?;
        let res = frame.resolution();
        
        let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(
            res.width(),
            res.height(),
            rgb.into_raw(),
        ).unwrap();

        image::imageops::flip_horizontal_in_place(&mut img);

        let mut png = Vec::new();
        let mut enc = image::codecs::png::PngEncoder::new(&mut png);
        enc.encode(
            img.as_raw(),
            res.width(),
            res.height(),
            image::ColorType::Rgb8,
        )?;

        Ok(png)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let addr = format!("127.0.0.1:{PORT}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("Listening on ws://{addr}");

    while let Ok((stream, _)) = listener.accept().await {
        let mut cam = Cam::new()?;
        
        tokio::spawn(async move {
            if let Ok(ws) = accept_async(stream).await {
                let (mut tx,  rx) = ws.split();
                
                println!("Client connected");
                
                loop {

                    match cam.capture() {
                        Ok(png) => {
                            if let Err(e) = tx.send(Message::Binary(Bytes::from(png))).await {
                                eprintln!("Send failed: {e}");
                                break;
                            }
                        }
                        Err(e) => {
                            eprintln!("Capture failed: {e}");
                        }
                    }

              
                }
            }
        });
    }

    Ok(())
}