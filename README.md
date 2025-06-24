# Player Re-Identification

**Player_ReID** is a computer vision project designed to detect, track, and re-identify soccer players in video footage.  
Leveraging YOLOv11 for object detection and custom tracking algorithms, the system maintains consistent player identities even when they leave and re-enter the frame.<br><br>
🎯 OSNet (Omni-Scale Network) is a lightweight yet powerful convolutional neural network designed for person re-identification.<br>
I used : osnet_x1_0 — a mid-size model balancing speed and accuracy<br><br>
Pretrained on MSMT17, a large and diverse dataset for real-world person re-identification
The model extracts robust appearance embeddings for each player, allowing BoT-SORT to match players across frames even when they are not continuously visible.

## 🔗 Summary of Pipeline
1. Player Detection: YOLOv11 detects all players in each frame.

2. ReID Embedding: OSNet generates appearance embeddings for each detection.

3. Tracking: ByteTrack uses both motion and appearance to track players over time.

4. Re-Identification: When a player re-enters, BoT-SORT compares embeddings and reassigns the correct ID if the similarity is high.

This combination ensures accurate, real-time player tracking and re-identification with minimal ID switches.
<br>

## 📁 Project Structure
```
Player_ReID/
├── input/               
├── models/              
├── output.mp4           
├── runs/                
├── stubs/               
├── team_assigner/      
├── tracker/            
├── utils/               
├── main.py              
├── requirements.txt     
└── README.md            
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)
- [YOLOv11](https://github.com/ultralytics/yolov11) dependencies

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Ayxsh03/Player_ReID.git
   cd Player_ReID
   ```
2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. **Install the required packages:**
```bash
pip install -r requirements.txt
```
4. **Download the YOLOv11 model:**

Place the best.pt model file into the models/ directory.

## 🎯 Usage
To run the player re-identification system on an input video:

```bash
python main.py
```

## 🧠 Features
1. Player Detection: Utilises YOLOv5 for accurate player detection in each frame.

2. Tracking: Implements custom tracking algorithms to maintain player identities across frames.

3. Re-Identification: Reassigns consistent IDs to players re-entering the frame after temporary occlusions or exits.

4. Team Assignment: Assigns players to teams based on jersey colours or predefined criteria.

5. Output Generation: Produces annotated videos with bounding boxes and player IDs.

## 🛠️ Customization
1. Adjust Detection Confidence

2. Modify the confidence threshold in the detection module to fine-tune sensitivity.

3. Change Tracking Parameters:

4. Tweak parameters like maximum disappearance frames or distance thresholds in the tracking module for optimal performance.

## 📈 Sample Results
![Demo](runs/output.gif)
The above GIF demonstrates the system's ability to track and re-identify players throughout the match.

## 🤝 Contributing
Contributions are welcome! To contribute:

1. Fork the repository.

2. Create a new branch: git checkout -b feature/YourFeature

3. Commit your changes: git commit -m 'Add your feature'

4. Push to the branch: git push origin feature/YourFeature

5. Open a pull request.


