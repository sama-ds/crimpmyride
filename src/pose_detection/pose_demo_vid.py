import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculate the angle between three points in 3D space."""
    # Convert points to numpy arrays
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Create vectors
    ba = a - b
    bc = c - b
    
    # Calculate the dot product
    dot_product = np.dot(ba, bc)
    
    # Calculate the magnitude of the vectors
    ba_magnitude = np.linalg.norm(ba)
    bc_magnitude = np.linalg.norm(bc)
    
    # Calculate the angle in radians
    angle = np.arccos(dot_product / (ba_magnitude * bc_magnitude))
    
    # Convert the angle to degrees
    angle = np.degrees(angle)
    
    return angle


def get_joint_loc(landmarks, joint):
    """Get the 3D coordinates of a specific joint from the landmarks."""
    return [
        landmarks[mp_pose.PoseLandmark[joint].value].x, 
        landmarks[mp_pose.PoseLandmark[joint].value].y,
        landmarks[mp_pose.PoseLandmark[joint].value].z,
        ]

def calc_centre_of_mass(
        landmarks, 
        segment_weights=None):
    """Calculate the center of mass of a human body based on landmark coordinates.

    Args:
        landmarks (dict): A dictionary containing landmark coordinates.
        segment_weights (dict, optional): A dictionary containing segment weights. 
            If not provided, default weights will be used.

    Returns:
        list: A list containing the x, y, and z coordinates of the center of mass.

    """
    if segment_weights is None:
        segment_weights = {
            'head': 0.07,
            'left_lower_arm': 0.016,
            'right_lower_arm': 0.016,
            'left_upper_arm': 0.02,
            'right_upper_arm': 0.02,
            'left_thigh': 0.1,
            'right_thigh': 0.1,
            'left_calf': 0.05,
            'right_calf': 0.05,
        }
    # Define the segments in terms of start and end landmarks
    segments = {
        'head': ('NOSE', 'NOSE'),  # Placeholder, assuming the head's center is at the nose
        'left_lower_arm': ('LEFT_WRIST', 'LEFT_ELBOW'),
        'right_lower_arm': ('RIGHT_WRIST', 'RIGHT_ELBOW'),
        'left_upper_arm': ('LEFT_ELBOW', 'LEFT_SHOULDER'),
        'right_upper_arm': ('RIGHT_ELBOW', 'RIGHT_SHOULDER'),
        'left_thigh': ('LEFT_HIP', 'LEFT_KNEE'),
        'right_thigh': ('RIGHT_HIP', 'RIGHT_KNEE'),
        'left_calf': ('LEFT_KNEE', 'LEFT_ANKLE'),
        'right_calf': ('RIGHT_KNEE', 'RIGHT_ANKLE'),
    }

    segment_centers = {}
    total_mass = sum(segment_weights.values())

    for segment, (start_joint, end_joint) in segments.items():
        start_loc = get_joint_loc(landmarks, start_joint)
        end_loc = get_joint_loc(landmarks, end_joint)
        # Calculate the midpoint (center of mass) for each segment
        segment_center = [(start + end) / 2 for start, end in zip(start_loc, end_loc)]
        segment_centers[segment] = segment_center

    # Calculate the total center of mass, weighted by segment masses
    total_center_of_mass = [sum(segment_centers[segment][i] * segment_weights[segment] 
                                for segment in segment_centers) / total_mass 
                            for i in range(3)]

    return total_center_of_mass

def visualize_point(image, loc, width, height, label=None, angle=None, colour=(0, 255, 0), base_size=10.0, shape='circle'):
    """
    Visualizes the joint on an image. Optionally visualizes the angle of the joint if provided.

    Args:
        image (numpy.ndarray): The input image.
        loc (tuple): The x, y, and z coordinates of the joint.
        width (int): The width of the image.
        height (int): The height of the image.
        label (str, optional): The label of the joint. Default is None.
        angle (float, optional): The angle of the joint. Default is None.
        colour (tuple, optional): The color of the point and text. Default is (0, 255, 0).
        base_dot_size (float, optional): The base size of the point. Default is 10.0.
        shape (str, optional): The shape to be drawn at the joint. Can be 'circle' or 'cross'. Default is 'circle'.

    Returns:
        None

    Raises:
        None
    """
    # Extract the x, y, and z coordinates
    x, y, z = loc

    # Normalize and scale the dot size based on z
    mark_size = max(base_size - z * 2,1)

    thickness = 2  # Base thickness

    # Scale the x and y coordinates to image dimensions
    scaled_loc = np.multiply([x, y], [width, height]).astype(int)

    # Determine the text to be drawn
    text = ""
    if label is not None:
        text = f"{label}"
    if angle is not None:
        text += f" {str(round(angle))}"

    # Draw the point based on the specified shape
    if shape == 'circle':
        # Draw a circle
        cv2.circle(image, tuple(scaled_loc), int(mark_size), colour, -1)
    elif shape == 'cross':
        # Draw a cross

        cv2.drawMarker(image, tuple(scaled_loc), colour, markerType=cv2.MARKER_CROSS, markerSize=int(mark_size), thickness=thickness)

    # Draw the text
    if text != "":
        cv2.putText(image, text, tuple(scaled_loc), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, thickness, cv2.LINE_AA)



  

def calculate_joint_angle(landmarks, joint1, joint2, joint3):
    # Get coordinates
    joint1_coords = get_joint_loc(landmarks, joint1)
    joint2_coords = get_joint_loc(landmarks, joint2)
    joint3_coords = get_joint_loc(landmarks, joint3)

    # Calculate angle
    angle = calculate_angle(joint1_coords, joint2_coords, joint3_coords)

    return angle

# Function to be called whenever the trackbar position is changed
def on_trackbar(val):
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    ret, frame = cap.read()
    if ret:
        process_frame(frame)
    else:
        print("Error: Cannot read frame.")

# Initialize video file capture
video_path = 'boulder.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow('Mediapipe Feed')
# Create a trackbar for navigating through the video
cv2.createTrackbar('Frame', 'Mediapipe Feed', 0, total_frames-1, on_trackbar)

def process_frame(image):
    # [Your processing code here. For example, converting image colors, making detections, drawing landmarks, etc.]
    # Make sure to use 'image' argument as your frame input for processing and visualization within this function.

    ret, frame = cap.read()
    
    # Recolor image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Make detection
    results = pose.process(image)

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        landmarks = results.pose_landmarks.landmark

        # Calculate angles
        l_elbow_ang = calculate_joint_angle(landmarks, "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST")
        r_elbow_ang = calculate_joint_angle(landmarks, "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST")
        l_hip_ang = calculate_joint_angle(landmarks, "LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE")
        r_hip_ang = calculate_joint_angle(landmarks, "RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE")
        l_knee_ang = calculate_joint_angle(landmarks, "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE")
        r_knee_ang = calculate_joint_angle(landmarks, "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE")
        l_ankle_ang = calculate_joint_angle(landmarks, "LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT_INDEX")
        r_anke_ang = calculate_joint_angle(landmarks, "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT_INDEX")
        l_armpit_ang = calculate_joint_angle(landmarks, "LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP")
        r_armpit_ang = calculate_joint_angle(landmarks, "RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP")

        l_eblow_pos = get_joint_loc(landmarks, "LEFT_ELBOW")
        r_elbow_pos = get_joint_loc(landmarks, "RIGHT_ELBOW")
        l_hip_pos = get_joint_loc(landmarks, "LEFT_HIP")
        r_hip_pos = get_joint_loc(landmarks, "RIGHT_HIP")
        l_knee_pos = get_joint_loc(landmarks, "LEFT_KNEE")
        r_knee_pos = get_joint_loc(landmarks, "RIGHT_KNEE")
        l_ankle_pos = get_joint_loc(landmarks, "LEFT_ANKLE")
        r_ankle_pos = get_joint_loc(landmarks, "RIGHT_ANKLE")
        l_armpit_pos = get_joint_loc(landmarks, "LEFT_SHOULDER")
        r_armpit_pos = get_joint_loc(landmarks, "RIGHT_SHOULDER")

        #Calculate centre of mass
        com = calc_centre_of_mass(landmarks)

        #Visualise angles
        visualize_point(
            image=image, 
            angle=l_elbow_ang, 
            loc=l_eblow_pos, 
            #label="L Elbow", 
            width=width, 
            height=height,
        )
        visualize_point(
            image=image, 
            angle=r_elbow_ang, 
            loc=r_elbow_pos, 
            #label="R Elbow", 
            width=width, 
            height=height
        )
        visualize_point(
            image=image, 
            angle=l_hip_ang, 
            loc=l_hip_pos, 
            #label="L Hip", 
            width=width, 
            height=height
        )
        visualize_point(
            image=image, 
            angle=r_hip_ang, 
            loc=r_hip_pos, 
            #label="R Hip", 
            width=width, 
            height=height
        )
        visualize_point(
            image=image, 
            angle=l_knee_ang, 
            loc=l_knee_pos, 
            #label="L Knee", 
            width=width, 
            height=height
        )
        visualize_point(
            image=image, 
            angle=r_knee_ang, 
            loc=r_knee_pos, 
            #label="R Knee", 
            width=width, 
            height=height
        )

        visualize_point(
            image=image,
            angle=l_ankle_ang,
            loc=l_ankle_pos,
            #label="L Ankle",
            width=width,
            height=height
        )

        visualize_point(
            image=image,
            angle=r_anke_ang,
            loc=r_ankle_pos,
            #label="R Ankle",
            width=width,
            height=height
        )

        visualize_point(
            image=image,
            angle=l_armpit_ang,
            loc=l_armpit_pos,
            #label="L Arm",
            width=width,
            height=height
        )

        visualize_point(
            image=image,
            angle=r_armpit_ang,
            loc=r_armpit_pos,
            #label="R Arm",
            width=width,
            height=height
        )

        #Visualise centre of mass
        visualize_point(
            image=image, 
            loc=com, 
            width=width, 
            height=height, 
            colour=(0, 0, 255),
            base_size=20.0,
            shape = 'cross',
            )
        

    except Exception as e:
        print(f"Error: {e}")
        # pass
    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                )          

    
    cv2.imshow('Mediapipe Feed', image)     
        

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        # The frame processing is now handled by the on_trackbar callback function.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
