import tkinter as tk
from tkinter import ttk, filedialog
from ttkthemes import ThemedTk  # Install ttkthemes for modern themes
import cv2
from PIL import Image, ImageTk
import threading
import os

# Global variables
selected_video_path = None
cap = None
video_running = True

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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(10, display_frame)
    else:
        cap.release()

def stop_video():
    global video_running
    video_running = False
    video_label.config(image=None, text="No video playing")

def extract_frames():
    if not selected_video_path:
        print("No video selected for processing.")
        return

    extract_button.config(state=tk.DISABLED)
    extraction_thread = threading.Thread(target=process_video)
    extraction_thread.start()

def process_video():
    print(f"Processing video: {selected_video_path}")
    # Simulate processing
    import time
    for i in range(101):
        progress_bar["value"] = i
        root.update_idletasks()
        time.sleep(0.05)
    print("Processing complete.")
    extract_button.config(state=tk.NORMAL)

# Create the main window
root = ThemedTk(theme="arc")  # Use a modern theme
root.title("Video Frame Extractor")
root.geometry("900x600")

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

extract_button = ttk.Button(controls_frame, text="Extract Frames", command=extract_frames)
extract_button.pack(pady=5)

stop_button = ttk.Button(controls_frame, text="Stop Video", command=stop_video)
stop_button.pack(pady=5)

progress_bar = ttk.Progressbar(controls_frame, orient="horizontal", length=200, mode="determinate")
progress_bar.pack(pady=20)

# Start the application
root.mainloop()
