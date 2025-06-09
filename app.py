import cv2
import numpy as np
import os
from pathlib import Path
import time
import logging
import secrets
import sys
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename

# Setup logging with UTF-8 encoding
class UTF8StreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            stream.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see EasyOCR output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        UTF8StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Check platform and set encoding for Windows compatibility
if sys.platform.startswith('win'):
    try:
        import _locale
        _locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])
    except Exception as e:
        logger.warning(f"Failed to set locale encoding: {e}")

# Check core dependencies
try:
    import cv2
    import numpy as np
    import secrets
    logger.info("[+] Core dependencies imported successfully")
except ImportError as e:
    logger.error(f"Failed to import core dependencies: {e}")
    exit()

# Check if ultralytics is available
try:
    from ultralytics import YOLO
    logger.info("[+] Ultralytics imported successfully")
except ImportError:
    logger.error("Failed to import ultralytics: Please install with 'pip install ultralytics'")
    exit()

# OCR availability flags
EASYOCR_AVAILABLE = False
PYTESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    logger.info("[+] EasyOCR imported successfully")
except ImportError:
    logger.warning("EasyOCR not available: Please install with 'pip install easyocr'")

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
    logger.info("[+] Pytesseract imported successfully")
except ImportError:
    logger.warning("Pytesseract not available: Please install with 'pip install pytesseract'")

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size
MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'best.pt')  # Configurable model path

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class EnhancedLicensePlateDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.ocr = None
        self.ocr_type = None
        self.crop_dir = os.path.join(app.config['RESULTS_FOLDER'], 'cropped_plates')
        self.processed_dir = os.path.join(app.config['RESULTS_FOLDER'], 'processed_plates')
        os.makedirs(self.crop_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.load_model()
        self.load_ocr()

    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            self.model = YOLO(self.model_path)
            logger.info("[+] YOLO Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def load_ocr(self):
        logger.info("[+] Attempting to load OCR")
        global EASYOCR_AVAILABLE, PYTESSERACT_AVAILABLE
        
        # Try EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                import easyocr
                self.ocr = easyocr.Reader(['en'], gpu=False)
                self.ocr_type = "EasyOCR"
                logger.info("[+] EasyOCR loaded successfully")
                return True
            except Exception as e:
                logger.error(f"EasyOCR failed: {e}")
                EASYOCR_AVAILABLE = False
        
        # Try Pytesseract
        if PYTESSERACT_AVAILABLE:
            try:
                import pytesseract
                # Set Tesseract path for Windows
                if sys.platform.startswith('win'):
                    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                self.ocr = pytesseract
                self.ocr_type = "Pytesseract"
                logger.info("[+] Pytesseract loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Pytesseract failed: {e}")
                PYTESSERACT_AVAILABLE = False
        
        logger.error("No OCR library loaded. Running in detection-only mode")
        self.ocr = None
        self.ocr_type = None
        return False

    def preprocess_license_plate(self, plate_img, plate_id):
        processed_images = {}
        processed_images['original'] = plate_img.copy()
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        processed_images['grayscale'] = gray
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        processed_images['bilateral_filter'] = bilateral
        adaptive_thresh = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        processed_images['adaptive_threshold'] = adaptive_thresh
        self.save_processed_images(processed_images, plate_id)
        return processed_images

    def save_processed_images(self, processed_images, plate_id):
        plate_folder = os.path.join(self.processed_dir, f'plate_{plate_id}')
        os.makedirs(plate_folder, exist_ok=True)
        for technique_name, img in processed_images.items():
            filename = os.path.join(plate_folder, f'{technique_name}.jpg')
            try:
                if len(img.shape) == 2:
                    cv2.imwrite(filename, img)
                else:
                    cv2.imwrite(filename, img)
            except Exception as e:
                logger.error(f"Failed to save processed image {technique_name} for plate_{plate_id}: {e}")

    def recognize_text_with_ocr(self, processed_images, plate_id):
        if self.ocr is None:
            return {}
        ocr_results = {}
        techniques_to_try = ['adaptive_threshold', 'bilateral_filter', 'grayscale']
        for technique in techniques_to_try:
            if technique in processed_images:
                try:
                    img = processed_images[technique]
                    if len(img.shape) == 2:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    else:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    texts = []
                    confidences = []
                    if self.ocr_type == "EasyOCR":
                        result = self.ocr.readtext(img_rgb, detail=1, min_size=10)
                        logger.debug(f"EasyOCR result for {technique} on plate_{plate_id}: {result}")
                        if result:
                            for detection in result:
                                text = detection[1]  # Text
                                confidence = detection[2]  # Confidence
                                if confidence > 0.1:  # Lowered threshold
                                    texts.append(text)
                                    confidences.append(confidence)
                    elif self.ocr_type == "Pytesseract":
                        text = self.ocr.image_to_string(img_rgb, lang='eng')
                        logger.debug(f"Pytesseract result for {technique} on plate_{plate_id}: {text}")
                        texts = [text.strip()] if text.strip() else []
                        confidences = [0.5] * len(texts)  # Pytesseract doesn't provide confidence
                    cleaned_texts = [''.join(c for c in text if c.isalnum() or c.isspace()).strip()
                                    for text in texts if len(''.join(c for c in text if c.isalnum() or c.isspace()).strip()) >= 2]
                    ocr_results[technique] = {
                        'texts': cleaned_texts,
                        'confidences': confidences,
                        'combined_text': ' '.join(cleaned_texts) if cleaned_texts else ''
                    }
                except Exception as e:
                    logger.error(f"OCR failed for {technique} on plate_{plate_id}: {e}")
                    ocr_results[technique] = {'texts': [], 'confidences': [], 'combined_text': ''}
        return ocr_results

    def get_best_ocr_result(self, ocr_results):
        best_result = None
        best_score = 0
        best_technique = ""
        for technique, result in ocr_results.items():
            if result['texts']:
                avg_confidence = np.mean(result['confidences']) if result['confidences'] else 0
                combined_text = result['combined_text']
                text_score = 0
                if combined_text:
                    has_letters = any(c.isalpha() for c in combined_text)
                    has_numbers = any(c.isdigit() for c in combined_text)
                    length_score = min(len(combined_text) / 10, 1)
                    text_score = (1.0 if has_letters and has_numbers else 0.7 if has_letters or has_numbers else 0.3) * length_score
                total_score = (avg_confidence * 0.7) + (text_score * 0.3)
                if total_score > best_score:
                    best_score = total_score
                    best_result = result
                    best_technique = technique
        logger.debug(f"Best OCR result - Technique: {best_technique}, Score: {best_score}, Text: {best_result['combined_text'] if best_result else 'None'}")
        return best_result, best_technique, best_score

    def crop_license_plate(self, frame, box, padding=10):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        cropped_plate = frame[y1:y2, x1:x2]
        return cropped_plate, (x1, y1, x2, y2)

    def process_video(self, video_path, session_id):
        if self.model is None:
            return None, "Model not loaded"
        if not os.path.exists(video_path):
            return None, f"Video file not found: {video_path}"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Failed to open video. Ensure the video format is supported (mp4, avi, mov, mkv)."
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0 or total_frames <= 0:
                cap.release()
                return None, "Invalid video properties (e.g., zero FPS or frame count)."
            
            output_path = os.path.join(app.config['RESULTS_FOLDER'], f'{session_id}_output.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            plate_results = {}  # Store OCR results for each plate
            plate_count = 0
            results_list = []
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    results = self.model(frame, conf=0.5, verbose=False)
                    annotated_frame = frame.copy()
                    frame_results = []
                    
                    if results and results[0].boxes is not None:
                        for i, box in enumerate(results[0].boxes):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = box.conf[0].cpu().numpy()
                            cropped_plate, crop_coords = self.crop_license_plate(frame, box)
                            
                            if cropped_plate.size > 0:
                                plate_count += 1
                                crop_filename = os.path.join(self.crop_dir, f'{session_id}_plate_{plate_count}.jpg')
                                cv2.imwrite(crop_filename, cropped_plate)
                                
                                processed_images = self.preprocess_license_plate(cropped_plate, plate_count)
                                ocr_results = self.recognize_text_with_ocr(processed_images, plate_count)
                                best_result, best_technique, best_score = self.get_best_ocr_result(ocr_results)
                                
                                # Store the OCR result for this plate
                                plate_results[f'{session_id}_plate_{plate_count}.jpg'] = {
                                    'text': best_result['combined_text'] if best_result else 'No text',
                                    'confidence': best_score if best_result else 0.0
                                }
                                
                                # Draw bounding box and text on the frame
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                display_text = f"Plate: {best_result['combined_text'] if best_result else 'No text'} ({best_score:.2f})"
                                # Adjust text position to ensure it's visible
                                text_y = max(20, y1 - 10)  # Ensure text doesn't go off-screen
                                cv2.putText(annotated_frame, display_text, (x1, text_y), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Yellow text for visibility
                                
                                frame_results.append({
                                    'plate_id': plate_count,
                                    'text': best_result['combined_text'] if best_result else '',
                                    'confidence': best_score if best_result else 0.0,
                                    'technique': best_technique,
                                    'crop_path': crop_filename
                                })
                    
                    results_list.append({
                        'frame': frame_count,
                        'plates': frame_results
                    })
                    out.write(annotated_frame)
                    frame_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {e}")
            
            cap.release()
            out.release()
            total_time = time.time() - start_time
            
            summary = {
                'total_frames': frame_count,
                'total_plates': plate_count,
                'processing_time': total_time,
                'output_video': output_path,
                'results': results_list,
                'plate_results': plate_results  # Pass OCR results to the results page
            }
            return summary, None
        except Exception as e:
            cap.release()
            if 'out' in locals():
                out.release()
            logger.error(f"Error in video processing: {e}")
            return None, f"Error processing video: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        session_id = secrets.token_urlsafe(16)
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_{filename}')
        try:
            file.save(video_path)
        except Exception as e:
            logger.error(f"Failed to save uploaded video: {e}")
            return jsonify({'error': 'Failed to save video file'}), 500
        
        detector = EnhancedLicensePlateDetector(MODEL_PATH)
        summary, error = detector.process_video(video_path, session_id)
        
        if error:
            logger.error(f"Video processing failed: {error}")
            return jsonify({'error': error}), 500
        
        # Build query parameters for OCR results
        query_params = {}
        for image_name, result in summary['plate_results'].items():
            query_params[f'text_{image_name}'] = result['text']
            query_params[f'conf_{image_name}'] = str(result['confidence'])
        
        results_url = url_for('show_results', session_id=session_id, **query_params)
        return jsonify({
            'session_id': session_id,
            'summary': summary,
            'results_url': results_url
        })
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/results/<session_id>')
def show_results(session_id):
    results_dir = os.path.join(app.config['RESULTS_FOLDER'], 'cropped_plates')
    results = []
    try:
        for file in os.listdir(results_dir):
            if file.startswith(session_id):
                results.append({
                    'image': file,
                    'path': f'/results/cropped_plates/{file}',
                    'text': request.args.get(f'text_{file}', 'No text'),
                    'confidence': float(request.args.get(f'conf_{file}', 0.0))
                })
        output_video = f'{session_id}_output.mp4'
        return render_template('results.html', session_id=session_id, results=results, video=output_video)
    except Exception as e:
        logger.error(f"Error retrieving results for session {session_id}: {e}")
        return jsonify({'error': 'Failed to load results'}), 500

@app.route('/results/<path:filepath>')
def serve_result(filepath):
    try:
        return send_from_directory(app.config['RESULTS_FOLDER'], filepath)
    except Exception as e:
        logger.error(f"Error serving file {filepath}: {e}")
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    logger.warning(" * Debugger is active!")
    logger.info(" * Debugger PIN: 920-919-509")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
