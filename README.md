
Aegis is an open-source security tool written in Rust and Python that uses your PC's camera to deliver face recognition, face detection, and motion detection. Developed with a focus on efficiency, it combines the performance of **Rust** with the flexibility of **Python** to create a lightweight yet powerful system.


*What Aegis Is Not*

- A software designed to run on your primary PC: Aegis is intended to run on dedicated servers or older PCs, repurposing them for security purposes. 
- A multi-system full-fledged security solution: Each Aegis instance is meant for only one PC's camera, although you can split the worload by deploying Aegis to two different computers.
  
---

*Current Capabilities:*

- **Face Recognition**: Recognizes and matches faces, enabling identification and tracking.
- **Face Detection**: Detects the presence of faces in the camera's frame.
- **Motion Detection**: Tracks movement in real-time.

*Upcoming Additions:*

- **Sleep Mode**: Reduces system usage by pausing operations when no motion or faces are detected.
- **UI for Monitoring**: A web interface running on `localhost` to actively monitor the camera's feed and manage settings.
- **Further Optimization**: Enhancements to make the software even more resource-efficient.

## **Getting Started**

Installation and usage instructions are coming soon. Stay tuned for updates in this repository. Aegis works on Linux and Windows. Actively tested on **Ubuntu** and **Windows 11**. Compatibility with macOS is uncertain.

## **Status**

Aegis is currently in development. While it offers functional features, additional refinements and new capabilities are on the way. Contributions are welcome! Whether it's a bug report, feature suggestion, or pull request, your input helps improve Aegis.


### **Legal**

Aegis is an open-source project and we do not collect any data in any form/mean from your use of the software.

This project is licensed under the [MIT License](LICENSE).
