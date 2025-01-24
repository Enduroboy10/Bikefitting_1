import tkinter as tk
from tkinter import ttk, filedialog
from ttkthemes import ThemedTk
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np
import base64
from matplotlib import pyplot as plt
from inference_sdk import InferenceHTTPClient

# Global variables
selected_video_path = None
cap = None
video_running = True
client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="ypbzOm53pf6plfZ7C8a0")
highest_frame = None
highest_y = float('inf')

# Function to stop video
def stop_video():
    global video_running
    video_running = False
    video_label.config(image=None, text="No video playing")

def select_video():
    global selected_video_path
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv;*.mov")])
    if file_path:
        selected_video_path = file_path
        play_video(file_path)

def play_video(file_path):
    global cap, video_running
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened():
        video_running = True
        display_frame()

def display_frame():
    global video_running
    if not video_running:
        return
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(10, display_frame)
    else:
        cap.release()

def extract_frames():
    if not selected_video_path:
        print("No video selected for processing.")
        return

    extract_button.config(state=tk.DISABLED)
    threading.Thread(target=find_highest_frame).start()

def find_highest_frame():
    global cap, highest_frame, highest_y
    cap = cv2.VideoCapture(selected_video_path)
    highest_y = float('inf')
    highest_frame = None

    frame_count = 0  # Track the frame index
    best_frame_index = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run inference on the current frame
        inference_result = run_inference(frame)
        if inference_result and 'predictions' in inference_result:
            for prediction in inference_result.get('predictions', []):
                for point in prediction.get('keypoints', []):
                    # Ensure 'class' and 'confidence' keys exist
                    if 'class' in point and 'confidence' in point:
                        if point['class'] == 'yellow_dot' and point['confidence'] > 0.5:
                            y = point['y']
                            if y < highest_y:
                                highest_y = y
                                highest_frame = frame
                                best_frame_index = frame_count

        # Stop processing if the yellow dot is found
        if highest_frame is not None:
            break

    cap.release()
    if highest_frame is not None:
        print(f"Frame with highest yellow dot found at index: {best_frame_index}")
        visualize_inference(highest_frame, run_inference(highest_frame))
    else:
        print("No valid frame with yellow dot found.")

    extract_button.config(state=tk.NORMAL)

def run_inference(frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        response = client.infer(image_base64, model_id="bikefitting/1")
        print(f"Inference response: {response}")  # Debug log
        return response
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def visualize_inference(frame, inference_result):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    keypoints_dict = {}

    for prediction in inference_result.get('predictions', []):
        keypoints = prediction.get('keypoints', [])
        for point in keypoints:
            if 'x' in point and 'y' in point and 'confidence' in point:
                x, y, confidence = point['x'], point['y'], point['confidence']
                if confidence > 0.5:
                    plt.scatter(x, y, color='red', s=20, marker='o')
                    if 'class' in point:
                        keypoints_dict[point['class']] = (x, y)

    if all(k in keypoints_dict for k in ['hufte', 'knie', 'knochel']):
        hufte = keypoints_dict['hufte']
        knie = keypoints_dict['knie']
        knochel = keypoints_dict['knochel']

        angle = calculate_angle(knochel, knie, hufte)
        adjusted_angle = 180 - angle

        plt.plot([hufte[0], knie[0]], [hufte[1], knie[1]], color='black', linewidth=2)
        plt.plot([knie[0], knochel[0]], [knie[1], knochel[1]], color='black', linewidth=2)
        
        plt.text(knie[0], knie[1] - 20, f"{adjusted_angle:.1f}°", fontsize=12, color='blue', 
                 bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))

    plt.axis('off')
    plt.show()

# Create the main window
root = ThemedTk(theme="arc")
root.title("Video Analysis Interface")
root.geometry("1000x600")

# Create video frame
video_frame = ttk.Frame(root)
video_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

video_label = ttk.Label(video_frame, text="No video selected", anchor="center", relief="sunken")
video_label.pack(fill="both", expand=True)

# Create controls frame
controls_frame = ttk.Frame(root)
controls_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# Add widgets to controls frame
ttk.Label(controls_frame, text="Controls", font=("Helvetica", 16)).pack(pady=10)

select_button = ttk.Button(controls_frame, text="Select Video", command=select_video)
select_button.pack(pady=5)

extract_button = ttk.Button(controls_frame, text="Run Analysis", command=extract_frames)
extract_button.pack(pady=5)

stop_button = ttk.Button(controls_frame, text="Stop Video", command=stop_video)
stop_button.pack(pady=5)

progress_bar = ttk.Progressbar(controls_frame, orient="horizontal", length=200, mode="determinate")
progress_bar.pack(pady=20)

# Start the application
root.mainloop()
