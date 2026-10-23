# YOLO11 Weapon Detection

## Files
- app.py: Streamlit website with webcam and image upload
- best.pt: Trained YOLO model
- requirements.txt: Python dependencies

## Render deployment
Create a Python Web Service connected to this repository.

Build command:
pip install -r requirements.txt

Start command:
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true

The live webcam uses browser WebRTC. Camera permission and HTTPS are required
for remote website access. Some networks may require additional WebRTC setup.
