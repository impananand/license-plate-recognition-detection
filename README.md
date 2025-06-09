# License Plate Recognition (LPR) Web App

A Flask-based web application for automatic license plate detection and recognition from uploaded videos.  
It uses YOLO (via Ultralytics) for plate detection and EasyOCR or Pytesseract for text recognition.  
The app provides a simple web interface for uploading videos and viewing results.

---

## Features

- Upload a video and detect license plates frame-by-frame.
- OCR (EasyOCR or Pytesseract) extracts plate numbers.
- Results page shows detected plates, OCR text, confidence, and annotated output video.
- All results and uploads are saved for review.

---

## Requirements

- Python 3.10 (recommended)
- pip

### Python Packages

- Flask
- opencv-python
- numpy
- ultralytics
- easyocr (optional, for better OCR)
- pytesseract (fallback OCR)
- tqdm (optional, for progress bars)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/lpr.git
cd lpr
```

### 2. Create and Activate a Virtual Environment

```bash
# Windows example with Python 3.10
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install manually:

```bash
pip install flask opencv-python numpy ultralytics easyocr pytesseract tqdm
```

### 4. Download or Train YOLO Model

- Place your YOLOv8 model weights (e.g., `best.pt`) in the project root or set the `YOLO_MODEL_PATH` environment variable.

### 5. (Optional) Install Tesseract for Windows

- Download from: https://github.com/tesseract-ocr/tesseract
- Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

### 6. Run the Application

```bash
python app.py
```

- The app will be available at [http://localhost:5000](http://localhost:5000)

---

## Usage

1. Open the web app in your browser.
2. Upload a video file (supported: mp4, avi, mov, mkv).
3. Wait for processing to complete.
4. View detected plates, OCR results, and download the annotated video.

---

## Project Structure

```
lpr/
│
├── app.py                  # Main Flask app
├── requirements.txt        # Python dependencies
├── uploads/                # Uploaded videos (auto-created)
├── results/                # Output videos and cropped plates (auto-created)
│   └── cropped_plates/
│   └── processed_plates/
├── templates/
│   ├── index.html
│   └── results.html
├── .gitignore
└── best.pt                 # (Your YOLO model weights)
```

---

## Notes

- The app will use EasyOCR if available, otherwise falls back to Pytesseract.
- For best results, use a well-trained YOLO model for license plate detection.
- Logs are saved to `app.log`.

---

## Troubleshooting

- **Orange line under imports in VS Code:**  
  Ensure your selected Python interpreter matches your virtual environment.
- **OCR not working:**  
  Make sure EasyOCR or Tesseract is installed and available.
- **Model not found:**  
  Place your YOLO weights file (`best.pt`) in the project root or set the `YOLO_MODEL_PATH` environment variable.

---

## License

MIT License

---

## Credits

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
