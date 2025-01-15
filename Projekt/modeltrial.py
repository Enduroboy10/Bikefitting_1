import numpy as np
import base64
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from PIL import Image
from inference_sdk import InferenceHTTPClient

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to display the image
def show_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')  # Hide axis
    plt.show()

# Function to calculate angle between three points
def calculate_angle(hufte, knie, knochel):
    # Vectors
    v1 = np.array([knie[0] - hufte[0], knie[1] - hufte[1]])
    v2 = np.array([knochel[0] - knie[0], knochel[1] - knie[1]])
    
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
def show_inference_result_with_keypoints(inference_response, image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    keypoints_dict = {}
    
    # Loop through predictions and plot keypoints
    for prediction in inference_response.get('predictions', []):
        keypoints = prediction.get('keypoints', [])
        
        for point in keypoints:
            x, y, confidence = point['x'], point['y'], point['confidence']
            if confidence > 0.5:  # Filter out points with low confidence
                plt.scatter(x, y, color='red', s=20, marker='o')  # Decreased size of points
                keypoints_dict[point['class_name']] = (x, y)
    
    # Connect keypoints (hufte -> knie -> knochel -> fus)
    if all(k in keypoints_dict for k in ['hufte', 'knie', 'knochel', 'fus']):
        hufte = keypoints_dict['hufte']
        knie = keypoints_dict['knie']
        knochel = keypoints_dict['knochel']
        fus = keypoints_dict['fus']
        
        # Calculate knee angle
        knee_angle, knee_angle_rad = calculate_angle(hufte, knie, knochel)
        print(f"Knee angle: {knee_angle:.2f} degrees")
        
        # Plot lines between keypoints
        plt.plot([hufte[0], knie[0]], [hufte[1], knie[1]], color='black', linewidth=2)
        plt.plot([knie[0], knochel[0]], [knie[1], knochel[1]], color='black', linewidth=2)
        plt.plot([knochel[0], fus[0]], [knochel[1], fus[1]], color='black', linewidth=2)
        
        # Create arc for the knee angle
        arc_radius = 40
        arc_center = knie
        theta1 = np.degrees(np.arctan2(knie[1] - hufte[1], knie[0] - hufte[0]))
        theta2 = np.degrees(np.arctan2(knochel[1] - knie[1], knochel[0] - knie[0]))
        
        if theta2 < theta1:
            theta2 += 360
        
        angle_points = np.linspace(theta1, theta1 + knee_angle, 100)
        arc_x = arc_center[0] + arc_radius * np.cos(np.radians(angle_points))
        arc_y = arc_center[1] + arc_radius * np.sin(np.radians(angle_points))
        
        plt.plot(arc_x, arc_y, color='blue', linewidth=2)  # Angle arc
        
        # Add arrows indicating the measured angle
        arrow1_x = knie[0] + 30 * np.cos(np.radians(theta1))
        arrow1_y = knie[1] + 30 * np.sin(np.radians(theta1))
        arrow2_x = knie[0] + 30 * np.cos(np.radians(theta1 + knee_angle))
        arrow2_y = knie[1] + 30 * np.sin(np.radians(theta1 + knee_angle))
        
        plt.annotate('', xy=(arrow1_x, arrow1_y), xytext=(knie[0], knie[1]),
                     arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5))
        plt.annotate('', xy=(arrow2_x, arrow2_y), xytext=(knie[0], knie[1]),
                     arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5))
        
        # Add label for the angle
        label_x = arc_center[0] + (arc_radius + 10) * np.cos(np.radians(theta1 + knee_angle / 2))
        label_y = arc_center[1] + (arc_radius + 10) * np.sin(np.radians(theta1 + knee_angle / 2))
        plt.text(label_x, label_y, f'{knee_angle:.1f}°', fontsize=12, color='black',
                 bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))  # Angle label

    plt.axis('off')
    plt.show()

# Create an inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ypbzOm53pf6plfZ7C8a0"
)

# Display the original image before inference
image_path = "c:/Users/nmuig/Documents/Uni/Project/Projekt_1/Projekt/frame_absolute_highest_22.jpg"
show_image(image_path)

# Convert image to base64 and run inference
image_base64 = image_to_base64(image_path)
response = CLIENT.infer(
    image_base64, 
    model_id="bikefitting/1"
)

print(response)

# Display the image after inference with keypoints and lines
show_inference_result_with_keypoints(response, image_path)
