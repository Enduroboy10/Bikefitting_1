import cv2
import numpy as np
import os
from scipy.signal import butter, filtfilt, argrelextrema

# Low-pass filter function
def low_pass_filter(data, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Updated video processing function
def process_videos(
    video_file, 
    output_folder, 
    lower_color=np.array([20, 100, 100]), 
    upper_color=np.array([30, 255, 255]), 
    cutoff_frequency=1.0
):
    """
    Processes a single video to extract frames where yellow markers are detected.
    Saves frames corresponding to the highest (Oben) and lowest (Unten) y-coordinate.

    Parameters:
    - video_file: Path to the video file.
    - output_folder: Path to the folder where processed frames will be saved.
    - lower_color: Lower bound for the color in HSV space (default is yellow).
    - upper_color: Upper bound for the color in HSV space (default is yellow).
    - cutoff_frequency: Cutoff frequency for the low-pass filter.

    Returns:
    - None
    """
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the output folder in the script's directory if it doesn't exist
    output_folder = os.path.join(script_dir, output_folder)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    else:
        print(f"Output folder already exists: {output_folder}")

    print(f"Processing video: {video_file}")

    # Open the input video
    input_video = cv2.VideoCapture(video_file)

    # Get video properties
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

        # Create a mask for the specified color
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Find the coordinates of color pixels
        color_points = np.column_stack(np.where(mask > 0))

        if color_points.size > 0:
            # Calculate the mean coordinates
            mean_point = np.mean(color_points, axis=0).astype(int)  # (y, x) format
            y_coordinates.append(mean_point[0])  # Save only the y-coordinate
            frames.append(frame.copy())  # Save the frame for potential extraction

        # Display progress (optional)
        frame_number += 1
        print(f"Processing frame {frame_number}/{frame_count}", end="\r")

    # Release the video file
    input_video.release()
    cv2.destroyAllWindows()

    # Process extracted coordinates if available
    if y_coordinates:
        # Apply low-pass filter to smooth the y-coordinates
        smoothed_y = low_pass_filter(y_coordinates, cutoff=cutoff_frequency, fs=fps)

        # Find indices of minima and maxima
        minima_indices = argrelextrema(np.array(smoothed_y), np.less)[0]
        maxima_indices = argrelextrema(np.array(smoothed_y), np.greater)[0]

        # Create subfolders for minima (Unten) and maxima (Oben) based on the y-coordinate extremes
        highest_folder = os.path.join(output_folder, "Highest_Y_Oben")  # Changed to "Highest_Y_Oben"
        lowest_folder = os.path.join(output_folder, "Lowest_Y_Unten")  # Changed to "Lowest_Y_Unten"

        # Ensure subfolders exist
        if not os.path.exists(highest_folder):
            os.makedirs(highest_folder)
            print(f"Created highest folder: {highest_folder}")
        else:
            print(f"Highest folder already exists: {highest_folder}")

        if not os.path.exists(lowest_folder):
            os.makedirs(lowest_folder)
            print(f"Created lowest folder: {lowest_folder}")
        else:
            print(f"Lowest folder already exists: {lowest_folder}")

        # Save frame at the highest point (maxima)
        if maxima_indices.size > 0:
            highest_idx = maxima_indices[np.argmax(np.array(smoothed_y)[maxima_indices])]
            highest_frame_path = os.path.join(highest_folder, f"frame_highest_{highest_idx}.jpg")
            cv2.imwrite(highest_frame_path, frames[highest_idx])  # Save frame at highest y-coordinate
            print(f"Saved highest point frame at index {highest_idx}: {highest_frame_path}")

        # Save frame at the lowest point (minima)
        if minima_indices.size > 0:
            lowest_idx = minima_indices[np.argmin(np.array(smoothed_y)[minima_indices])]
            lowest_frame_path = os.path.join(lowest_folder, f"frame_lowest_{lowest_idx}.jpg")
            cv2.imwrite(lowest_frame_path, frames[lowest_idx])  # Save frame at lowest y-coordinate
            print(f"Saved lowest point frame at index {lowest_idx}: {lowest_frame_path}")

        print(f"Extracted frames at highest and lowest points for '{os.path.basename(video_file)}'.")
    else:
        print(f"No color points detected in '{os.path.basename(video_file)}'.")

    print("Processing completed.")
