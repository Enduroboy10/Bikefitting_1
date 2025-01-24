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
def show_inference_result_with_keypoints(inference_response, image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    keypoints_dict = {}
    
    # Loop through predictions and plot keypoints
    for prediction in inference_response.get('predictions', []):
        keypoints = prediction.get('keypoints', [])
        
        for point in keypoints:
            if 'class' in point:  # Ensure 'class' exists
                x, y, confidence = point['x'], point['y'], point['confidence']
                if confidence > 0.5:  # Filter out points with low confidence
                    plt.scatter(x, y, color='red', s=20, marker='o')  # Decreased size of points
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
        
        # Plot solid lines between keypoints
        plt.plot([hufte[0], knie[0]], [hufte[1], knie[1]], color='black', linewidth=2)
        plt.plot([knie[0], knochel[0]], [knie[1], knochel[1]], color='black', linewidth=2)
        
        # Extend the line from hufte to knie as dashed
        extension_factor = 0.4  # Shorten extension factor to 2/3
        extended_hufte_x = knie[0] + extension_factor * (knie[0] - hufte[0])
        extended_hufte_y = knie[1] + extension_factor * (knie[1] - hufte[1])
        plt.plot([knie[0], extended_hufte_x], [knie[1], extended_hufte_y], color='black', linestyle='dashed', linewidth=1)
        
        # Create arc between the dashed line and knie-knochel line
        arc_radius = 40
        arc_center = knie
        theta1 = np.degrees(np.arctan2(extended_hufte_y - knie[1], extended_hufte_x - knie[0]))
        theta2 = np.degrees(np.arctan2(knochel[1] - knie[1], knochel[0] - knie[0]))
        
        if theta1 < theta2:
            theta1 += 360
        
        angle_points = np.linspace(theta2, theta1, 100)
        arc_x = arc_center[0] + arc_radius * np.cos(np.radians(angle_points))
        arc_y = arc_center[1] + arc_radius * np.sin(np.radians(angle_points))
        
        # Plot the arc
        plt.plot(arc_x, arc_y, color='blue', linewidth=2)
        
        # Plot the dashed line from knie to the start of the arc
        plt.plot([knie[0], arc_x[0]], [knie[1], arc_y[0]], color='black', linestyle='dashed', linewidth=1)
        
        # Add angle label at the fixed position in the image
        label_x = knie[0] + 20
        label_y = knie[1] - 20
        plt.text(label_x, label_y, f'{adjusted_angle:.1f}°', fontsize=12, color='black',
                 bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))

    plt.axis('off')
    plt.show()

# Create an inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="mrzTH1atJSeLm8Sxk1WS"
)

# Display the original image before inference
image_path = "c:/Users/nmuig/Documents/Uni/Project/Projekt_1/Projekt/frame_absolute_highest_22.jpg"
show_image(image_path)

# Convert image to base64 and run inference
image_base64 = image_to_base64(image_path)
response = CLIENT.infer(
    image_base64, 
    model_id="bike_3/1"
)

print(response)

# Display the image after inference with keypoints and lines
show_inference_result_with_keypoints(response, image_path)
