import cv2
import numpy as np
from scipy.signal import butter, filtfilt, argrelextrema
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import base64
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from PIL import Image
from inference_sdk import InferenceHTTPClient
import os
# Define the low-pass filter
def butter_lowpass(cutoff, fs, order=2):
    nyquist = 0.5 * fs  # Nyquist frequency is half of the sampling rate
    normal_cutoff = cutoff / nyquist  # Normalize the cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # Filter coefficients
    return b, a


def extract_fr(path):
    lower_yellow = np.array([20, 100, 100])  # Lower bound for yellow
    upper_yellow = np.array([30, 255, 255])  # Upper bound for yellow

    # Open the input video
    input_video = cv2.VideoCapture(path)

    # Get video properties
    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize a list to store the y-coordinates and frames
    y_coordinates = []
    frames = []

    # Process the video frame by frame
    frame_number = 0
    while input_video.isOpened():
        ret, frame = input_video.read()
        if not ret:
            break  # Exit loop if no more frames are available

        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for yellow
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Find the coordinates of yellow pixels
        yellow_points = np.column_stack(np.where(mask > 0))

        if yellow_points.size > 0:  # Check if there are yellow points
            # Calculate the mean coordinates
            mean_point = np.mean(yellow_points, axis=0).astype(int)  # (y, x) format
            y_coordinates.append(mean_point[0])  # Save only the y-coordinate

            frames.append(frame.copy())  # Save the frame for potential extraction

        # Display progress (optional)
        frame_number += 1
        print(f"Processing frame {frame_number}/{frame_count}", end="\r")

    # Release the video file
    input_video.release()

    # Apply low-pass filter to smooth the y-coordinates
    if y_coordinates:
        # Apply the filter
        cutoff_frequency = 1.0  # Hz
        fs = fps  # Sampling rate is the frames per second of the video
        b, a = butter_lowpass(cutoff_frequency, fs)  # Get filter coefficients
        smoothed_y = filtfilt(b, a, y_coordinates)  # Apply the low-pass filter to y-coordinates

        # Find indices of minima and maxima
        minima_indices = argrelextrema(np.array(smoothed_y), np.less)[0]
        maxima_indices = argrelextrema(np.array(smoothed_y), np.greater)[0]

        # Initialize lists to store frames
        minima_frames = []
        maxima_frames = []

        # Save frames at minima
        for idx in minima_indices:
            minima_frames.append(frames[idx])  # Add frame to minima list

        # Save frames at maxima
        for idx in maxima_indices:
            maxima_frames.append(frames[idx])  # Add frame to maxima list

    else:
        print(f"No yellow points detected in video.")
    return maxima_frames


# Function to convert image to base64
def image_to_base64(image):
    # Convert to RGB (OpenCV loads in BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert image to base64
    _, encoded_image = cv2.imencode('.jpg', image_rgb)
    return base64.b64encode(encoded_image).decode('utf-8')

def get_interf(image):
    CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com",api_key="mrzTH1atJSeLm8Sxk1WS")
    # Convert image to base64 and run inference
    image_base64 = image_to_base64(image)
    response = CLIENT.infer(image_base64,model_id="bike_3/1")
    return response

# Function to calculate angle between three points
def calculate_angle(p1, p2, p3):
    # Vectors
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Dot product and magnitudes
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Cosine of the angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    
    # Angle in radians and degrees
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg, angle_rad

# Function to display the image and mark keypoints
def show_inference_result_with_keypoints(inference_response, image):
    keypoints_dict = {}
    
    # Loop through predictions and extract keypoints
    for prediction in inference_response.get('predictions', []):
        keypoints = prediction.get('keypoints', [])
        
        for point in keypoints:
            if 'class' in point:  # Ensure 'class' exists
                x, y, confidence = int(point['x']), int(point['y']), point['confidence']
                if confidence > 0.5:  # Filter out points with low confidence
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red keypoints
                    keypoints_dict[point['class']] = (x, y)
            else:
                print(f"Warning: Missing 'class' in point {point}")
    
    # Connect keypoints (Hufte -> Knie -> Knochel -> Fus)
    if all(k in keypoints_dict for k in ['Hufte', 'Knie', 'Knochel', 'Fus']):
        hufte = keypoints_dict['Hufte']
        knie = keypoints_dict['Knie']
        knochel = keypoints_dict['Knochel']
        fus = keypoints_dict['Fus']
        
        # Calculate the adjusted knee angle
        flexion_angle = calculate_angle(knochel, knie, hufte)[0]
        adjusted_angle = 180 - flexion_angle
        print(f"Adjusted knee angle: {adjusted_angle:.2f} degrees")
        
        # Draw solid lines between keypoints
        cv2.line(image, hufte, knie, (0, 0, 0), 2)
        cv2.line(image, knie, knochel, (0, 0, 0), 2)
        cv2.line(image, knochel, fus, (0, 0, 0), 2)

        
        # Extend the line from hufte to knie as dashed
        extension_factor = 0.4
        extended_hufte_x = int(knie[0] + extension_factor * (knie[0] - hufte[0]))
        extended_hufte_y = int(knie[1] + extension_factor * (knie[1] - hufte[1]))
        for i in range(0, 10):  # Draw dashed line
            t = i / 10.0
            x = int((1 - t) * knie[0] + t * extended_hufte_x)
            y = int((1 - t) * knie[1] + t * extended_hufte_y)
            cv2.circle(image, (x, y), 2, (0, 0, 0), -1)
        
        # Create arc between dashed line and knie-knochel line
        arc_radius = 40
        arc_center = knie
        theta1 = np.degrees(np.arctan2(extended_hufte_y - knie[1], extended_hufte_x - knie[0]))
        theta2 = np.degrees(np.arctan2(knochel[1] - knie[1], knochel[0] - knie[0]))
        
        if theta1 < theta2:
            theta1 += 360
        
        for theta in np.linspace(theta2, theta1, 100):
            arc_x = int(arc_center[0] + arc_radius * np.cos(np.radians(theta)))
            arc_y = int(arc_center[1] + arc_radius * np.sin(np.radians(theta)))
            cv2.circle(image, (arc_x, arc_y), 1, (255, 0, 0), -1)  # Blue arc
        
        # Draw the dashed line from knie to start of the arc
        arc_start_x = int(arc_center[0] + arc_radius * np.cos(np.radians(theta2)))
        arc_start_y = int(arc_center[1] + arc_radius * np.sin(np.radians(theta2)))
        cv2.line(image, knie, (arc_start_x, arc_start_y), (0, 0, 0), 1, cv2.LINE_AA)
        
        # Add angle label
        label_x = knie[0] + 20
        label_y = knie[1] - 20
        cv2.putText(image, f'{adjusted_angle:.1f}degrees', (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    return image,adjusted_angle
