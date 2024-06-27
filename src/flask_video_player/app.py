from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    jsonify,
)
import os
from werkzeug.utils import secure_filename
import cv2
import mediapipe as mp
import json
import math

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Setup MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video_file = request.files["video"]
        if video_file:
            filename = secure_filename(video_file.filename)
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            video_file.save(video_path)
            print(video_file, "saved")
            data = process_video(video_path)
            # Save the JSON data
            json_filename = os.path.splitext(filename)[0] + ".json"
            print(json_filename, "created")
            with open(
                os.path.join(app.config["UPLOAD_FOLDER"], json_filename), "w"
            ) as f:
                json.dump(data, f)
            return redirect(
                url_for("display_video", filename=filename, data_filename=json_filename)
            )
    return render_template("index.html")


def calculate_angle(point1, center, point2):
    """
    Calculate the angle (in radians) between three points.

    Parameters:
    - point1, point2: Dictionaries with 'x' and 'y' coordinates of the points.
    - center: Dictionary with 'x' and 'y' coordinates of the center point.

    Returns:
    - Angle in radians between the vectors formed by (center, point1) and (center, point2).
    """
    vector1 = (point1["x"] - center["x"], point1["y"] - center["y"])
    vector2 = (point2["x"] - center["x"], point2["y"] - center["y"])

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = (vector1[0] ** 2 + vector1[1] ** 2) ** 0.5
    magnitude2 = (vector2[0] ** 2 + vector2[1] ** 2) ** 0.5

    if magnitude1 * magnitude2 == 0:
        return 0  # Avoid division by zero

    cos_theta = dot_product / (magnitude1 * magnitude2)
    angle_rad = max(-1.0, min(1.0, cos_theta))  # Ensure cos_theta is within [-1, 1]
    angle_rad = math.acos(angle_rad)

    return angle_rad


def extract_angles(joints):
    """
    Extract angles between specific joints.

    Parameters:
    - joints: Dictionary containing joint positions extracted from pose landmarks.

    Returns:
    - Dictionary with angle measurements between specified joints.
    """
    angles = {
        "upper_arm_torso": calculate_angle(
            joints["left_shoulder"], joints["left_elbow"], joints["left_wrist"]
        ),
        "upper_arm_lower_arm": calculate_angle(
            joints["left_shoulder"], joints["left_elbow"], joints["left_wrist"]
        ),
        "upper_leg_torso": calculate_angle(
            joints["left_shoulder"], joints["left_elbow"], joints["left_wrist"]
        ),
        "upper_leg_lower_leg": calculate_angle(
            joints["left_shoulder"], joints["left_elbow"], joints["left_wrist"]
        ),
    }
    return angles


def process_video(video_path):
    """Extract shoulder data from the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        results_data = []
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process the image and find the pose
            results = pose.process(image)

            if results.pose_landmarks:
                # Extract coordinates
                joints = extract_joints(results.pose_landmarks)
                lines = extract_lines(joints)

                # Calculate angles
                angles = extract_angles(joints)

                frame_data = {
                    "timestamp": cap.get(cv2.CAP_PROP_POS_MSEC),
                    "frame": frame_idx,
                    "joints": joints,
                    "lines": lines,
                    "angles": angles,
                }
                results_data.append(frame_data)

            frame_idx += 1

    return results_data


def process_video_frame(frame_index):
    """Extract shoulder data from the video frame at the specified index."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES, frame_index
        )  # Seek to the specified frame index
        success, frame = cap.read()
        if not success:
            return {"error": "Could not read frame"}

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and find the pose
        results = pose.process(image)

        if results.pose_landmarks:
            # Extract coordinates
            joints = extract_joints(results.pose_landmarks)
            lines = extract_lines(joints)
            frame_data = {"frame": frame_index, "joints": joints, "lines": lines}
            return frame_data
        else:
            return {"error": "No pose landmarks found"}

    cap.release()


def extract_joints(landmarks):
    joints = {
        "left_shoulder": extract_joint_loc(
            landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER
        ),
        "right_shoulder": extract_joint_loc(
            landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER
        ),
        "left_elbow": extract_joint_loc(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW),
        "right_elbow": extract_joint_loc(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW),
        "left_wrist": extract_joint_loc(landmarks, mp_pose.PoseLandmark.LEFT_WRIST),
        "right_wrist": extract_joint_loc(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST),
        "left_hip": extract_joint_loc(landmarks, mp_pose.PoseLandmark.LEFT_HIP),
        "right_hip": extract_joint_loc(landmarks, mp_pose.PoseLandmark.RIGHT_HIP),
        "left_knee": extract_joint_loc(landmarks, mp_pose.PoseLandmark.LEFT_KNEE),
        "right_knee": extract_joint_loc(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE),
        "left_ankle": extract_joint_loc(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE),
        "right_ankle": extract_joint_loc(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE),
    }
    return joints


def extract_joint_loc(landmarks, landmark):
    return {
        "x": landmarks.landmark[landmark].x,
        "y": landmarks.landmark[landmark].y,
        "z": landmarks.landmark[landmark].z,
    }


def extract_lines(joints):
    lines = [
        {"start": "left_wrist", "end": "left_elbow"},
        {"start": "left_elbow", "end": "left_shoulder"},
        {"start": "left_shoulder", "end": "left_hip"},
        {"start": "left_hip", "end": "left_knee"},
        {"start": "left_knee", "end": "left_ankle"},
        {"start": "right_wrist", "end": "right_elbow"},
        {"start": "right_elbow", "end": "right_shoulder"},
        {"start": "right_shoulder", "end": "right_hip"},
        {"start": "right_hip", "end": "right_knee"},
        {"start": "right_knee", "end": "right_ankle"},
        {"start": "left_shoulder", "end": "right_shoulder"},
        {"start": "left_hip", "end": "right_hip"},
    ]
    return lines


@app.route("/display/<filename>/<data_filename>")
def display_video(filename, data_filename):
    return render_template(
        "display_video.html", video_filename=filename, data_filename=data_filename
    )


@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/json/<filename>")
def get_json(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# Update your Flask route to render the HTML template
@app.route("/display_processed_video/<filename>/<data_filename>")
def display_processed_video(filename, data_filename):
    return render_template(
        "processed_video.html", video_filename=filename, data_filename=data_filename
    )


# Flask route for fetching frame data
@app.route("/frame_data/<int:frame_index>")
def get_frame_data(frame_index):
    # Process the video frame at the specified index and return the frame data as JSON
    frame_data = process_video_frame(frame_index)
    return jsonify(frame_data)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1")
