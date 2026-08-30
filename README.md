# PROJECT CAM 🎥

A comprehensive collection of computer vision applications built with Python, OpenCV, and MediaPipe. This project includes gesture recognition, hand tracking, face detection, and interactive applications powered by real-time camera input.

## Features 🚀

### Core Applications

1. **Finger Counter** (`01_finger_counter.py`)
   - Real-time hand detection and finger counting
   - Uses MediaPipe hand tracking for accurate gesture recognition

2. **Virtual Mouse** (`02_virtual_mouse.py`)
   - Control your mouse cursor with hand gestures
   - Intuitive gesture-based interaction without physical input devices

3. **Volume Control** (`03_volume_control.py`)
   - Adjust system volume using hand gestures
   - Real-time audio control through camera input

4. **Brightness Control** (`04_brightness_control.py`)
   - Control screen brightness with hand gestures
   - Seamless desktop display adjustment

5. **Rock, Paper, Scissors** (`05_rock_paper_scissors.py`)
   - Play the classic game against the computer using hand gestures
   - Real-time hand gesture classification

6. **Drawing Board** (`06_drawing_board.py`)
   - Draw on a virtual canvas using your fingers
   - Creative digital art with gesture control

7. **Gesture Presentation** (`07_gesture_presentation.py`)
   - Navigate presentations using hand gestures
   - Control slides without keyboard or mouse

8. **Gesture Media Player** (`08_gesture_media_player.py`)
   - Control media playback (play, pause, next, previous) with gestures
   - Hands-free multimedia control

9. **Mask Detection** (`09_mask_detection.py`)
   - Detect whether people are wearing face masks
   - Uses TensorFlow/Keras for deep learning classification

10. **Face Attendance System** (`10_face_attendance.py`)
    - Automated attendance tracking using facial recognition
    - Identify and log individuals in real-time

11. **Drowsiness Detection** (`11_drowsiness_detection.py`)
    - Monitor driver alertness and detect drowsiness
    - Eye Aspect Ratio (EAR) based detection algorithm

12. **Sign Language Recognition** (`12_sign_language.py`)
    - Recognize and translate sign language gestures
    - Machine learning classifier for gesture interpretation

## Technology Stack 🛠️

### Core Libraries
- **OpenCV** - Computer vision and image processing
- **MediaPipe** - Hand and pose tracking
- **NumPy** - Numerical computations
- **CVZone** - Simplified OpenCV operations

### System Integration
- **PyAutoGUI** - Mouse and keyboard automation
- **PyCaw** - Windows audio control
- **Screen Brightness Control** - Display brightness adjustment

### Machine Learning
- **TensorFlow** - Deep learning framework
- **Keras** - Neural network API
- **Scikit-learn** - Machine learning algorithms
- **Dlib** - Face recognition
- **SciPy** - Scientific computing

### Data Processing
- **Pandas** - Data manipulation and analysis

## Installation 📦

### Prerequisites
- Python 3.8 or higher
- A working webcam
- Windows/Linux/macOS

### Setup

1. Clone the repository:
```bash
git clone https://github.com/envydot/PROJECT_CAM.git
cd PROJECT_CAM
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   - **Linux/macOS:**
     ```bash
     source .venv/bin/activate
     ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 🎮

Run any application directly from the command line:

```bash
python 01_finger_counter.py
python 02_virtual_mouse.py
python 03_volume_control.py
# ... and so on
```

Press `Q` or `ESC` to exit most applications.

## Project Structure 📁

```
PROJECT_CAM/
├── apps/                          # Main application scripts
│   ├── 01_finger_counter.py
│   ├── 02_virtual_mouse.py
│   ├── 03_volume_control.py
│   ├── 04_brightness_control.py
│   ├── 05_rock_paper_scissors.py
│   ├── 06_drawing_board.py
│   ├── 07_gesture_presentation.py
│   ├── 08_gesture_media_player.py
│   ├── 09_mask_detection.py
│   ├── 10_face_attendance.py
│   ├── 11_drowsiness_detection.py
│   └── 12_sign_language.py
├── data/                          # Data files and models
├── moduels/                       # Custom modules and utilities
├── requirements.txt               # Project dependencies
├── .venv/                         # Virtual environment (not committed)
└── README.md                      # This file
```

## Requirements 📋

All dependencies are listed in `requirements.txt`. Key requirements include:
- opencv-python
- mediapipe
- numpy
- cvzone
- pyautogui
- pycaw
- screen-brightness-control
- scipy
- face-recognition
- dlib
- tensorflow
- keras
- scikit-learn
- pandas

## Common Issues & Troubleshooting 🔧

### Camera Not Detected
- Ensure your webcam is connected and recognized by your operating system
- Check camera permissions in your system settings
- Try running `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

### dlib Installation Issues (Windows)
- Install Microsoft C++ Build Tools
- Use pre-built wheels: `pip install dlib --only-binary :all:`

### TensorFlow GPU Issues
- For GPU acceleration, install CUDA and cuDNN
- CPU mode works fine for most applications

### Audio/Brightness Control (Windows-specific)
- Ensure you have appropriate permissions for audio/display control
- Some applications may require administrator privileges

## Performance Tips ⚡

- Reduce frame resolution for faster processing
- Use lower detection confidence thresholds if tracking is lost
- Run on GPU if available (CUDA support)
- Close unnecessary background applications

## Future Enhancements 🔮

- [ ] Real-time pose estimation and skeleton tracking
- [ ] Multi-hand gesture recognition
- [ ] Emotion detection
- [ ] 3D hand pose estimation
- [ ] Mobile app version
- [ ] Web-based interface
- [ ] Recording and replay capabilities

## Contributing 🤝

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

## License 📄

This project is provided as-is for educational and personal use.

## Author ✏️

**DASH** - Computer Vision Enthusiast

## Acknowledgments 🙏

- Google MediaPipe team for excellent pose and hand tracking models
- OpenCV community for comprehensive computer vision tools
- TensorFlow/Keras for deep learning capabilities

## Contact & Support 💬

For issues, questions, or suggestions, please create an issue on GitHub.

---

**Happy Coding!** 🎉

Feel free to explore, modify, and build upon these applications!
