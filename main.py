import cv2
import threading
import numpy as np
from deepface import DeepFace

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Define your Target People
PEOPLE_CONFIG = {
    "Naitik": "./Images/naitik4.png",
    "Isha": "./Images/isha.png",
    "Keval": "./Images/keval2.png",
    # "Boss": "./Images/boss.jpg" 
}

# Define your Video Sources (Integers for USB, Strings for IP/URL)
# You can add as many as you want: [0, 1, "http://..."]
CAMERA_SOURCES = [
    "http://192.168.1.5:8080/video",  # Camera ID 0
    0                                 # Camera ID 1 (Webcam)
]

CHOSEN_MODEL = "ArcFace"
DETECTOR_BACKEND = "opencv"

# ==========================================
# 2. SETUP & LOADING
# ==========================================

print("Initializing System...")

# Runtime storage
# Structure: "Name": { "best_cam_index": int, "distance": float, "is_match": bool }
people_status = {} 
valid_people = []

# Load Reference Images
for name, path in PEOPLE_CONFIG.items():
    img = cv2.imread(path)
    if img is not None:
        # Initial state
        people_status[name] = {"best_cam_index": -1, "distance": 10.0, "is_match": False}
        
        # Store ref image in memory separately to avoid reloading
        # We attach it to the key for easy access
        people_status[name]["ref_img"] = img
        valid_people.append(name)
        print(f"Loaded: {name}")
    else:
        print(f"Error loading {name}")

# Initialize Cameras
caps = []
for source in CAMERA_SOURCES:
    cap = cv2.VideoCapture(source)
    # limit resolution for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    caps.append(cap)

# Pre-build model
try:
    DeepFace.build_model(CHOSEN_MODEL)
    print("Model Ready.")
except Exception as e:
    print(f"Model Error: {e}")

# ==========================================
# 3. BACKGROUND WORKER
# ==========================================
is_processing = False

def process_multi_camera(frames_dict):
    """
    frames_dict: { 0: frame_img, 1: frame_img }
    """
    global is_processing, people_status
    
    try:
        # For every person...
        for name in valid_people:
            current_best_cam = -1
            current_best_dist = 10.0 # High number = bad match
            found_match = False
            
            ref_img = people_status[name]["ref_img"]

            # ...Check every camera
            for cam_idx, frame in frames_dict.items():
                try:
                    # Run Verification
                    result = DeepFace.verify(
                        img1_path = frame,
                        img2_path = ref_img.copy(),
                        model_name = CHOSEN_MODEL,
                        detector_backend = DETECTOR_BACKEND,
                        enforce_detection = False
                    )
                    
                    if result["verified"]:
                        found_match = True
                        distance = result["distance"] # Lower is better
                        
                        # Logic: Is this camera better than the previous best?
                        if distance < current_best_dist:
                            current_best_dist = distance
                            current_best_cam = cam_idx
                            
                except Exception:
                    pass # Face not detected in this specific camera, skip
            
            # Update Global Status for this person after checking all cams
            people_status[name]["is_match"] = found_match
            people_status[name]["best_cam_index"] = current_best_cam
            people_status[name]["distance"] = current_best_dist

    except Exception as e:
        print(f"Thread Error: {e}")
    finally:
        is_processing = False

# ==========================================
# 4. MAIN LOOP
# ==========================================
counter = 0

while True:
    # 1. Capture frames from ALL cameras
    current_frames = {}
    valid_capture = False
    
    # We define a common black screen size based on first camera
    ret0, _ = caps[0].read() # Dummy read to check connectivity
    
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            current_frames[i] = frame
            valid_capture = True
    
    if valid_capture:
        # Use dimensions of the first available frame for the black screen
        first_frame = list(current_frames.values())[0]
        h, w, _ = first_frame.shape
        black_screen = np.zeros((h, w, 3), dtype="uint8")

        # 2. Run Thread (If idle)
        if counter % 30 == 0 and not is_processing:
            is_processing = True
            # We must pass COPIES of frames to thread
            frames_copy = {k: v.copy() for k, v in current_frames.items()}
            thread = threading.Thread(target=process_multi_camera, args=(frames_copy,))
            thread.start()
        
        counter += 1

        # 3. Display Logic (The Window Manager)
        for name in valid_people:
            status = people_status[name]
            
            if status["is_match"] and status["best_cam_index"] in current_frames:
                # Get the frame from the WINNING camera
                cam_idx = status["best_cam_index"]
                display_frame = current_frames[cam_idx].copy()
                
                # Overlay Details
                dist_str = "{:.4f}".format(status["distance"])
                cv2.rectangle(display_frame, (10, 10), (w-10, h-10), (0, 255, 0), 4)
                
                # Header
                cv2.putText(display_frame, f"WINNER: CAMERA {cam_idx}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Person Name
                cv2.putText(display_frame, f"TARGET: {name}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                # Score
                cv2.putText(display_frame, f"Dist: {dist_str}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            else:
                # Black Screen
                display_frame = black_screen.copy()
                cv2.putText(display_frame, f"Searching: {name}...", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                
            # Show Window
            cv2.imshow(f"Monitor - {name}", display_frame)

    else:
        print("Waiting for cameras...")

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Cleanup
for cap in caps:
    cap.release()
cv2.destroyAllWindows()