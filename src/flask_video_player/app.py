from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import mediapipe as mp
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Setup MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_file = request.files['video']
        if video_file:
            filename = secure_filename(video_file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(video_path)
            print(video_file, "saved")
            data = process_video(video_path)
            # Save the JSON data
            json_filename = os.path.splitext(filename)[0] + '.json'
            print(json_filename, "created")
            with open(os.path.join(app.config['UPLOAD_FOLDER'], json_filename), 'w') as f:
                json.dump(data, f)
            return redirect(url_for('display_video', filename=filename, data_filename=json_filename))
    return render_template('index.html')

def process_video(video_path):
    """Extract shoulder data from the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
                joints = {
                    'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC),
                    'frame': frame_idx,
                    'left_shoulder': {
                        'x': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        'y': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                        'z': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z,
                    },
                    'right_shoulder': {
                        'x': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                        'y': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                        'z': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z,
                    },
                    'left_elbow': {
                        'x': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                        'y': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                        'z': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z,
                    },
                    'right_elbow': {
                        'x': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                        'y': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                        'z': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z,
                    },
                    'left_hip': {
                        'x': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                        'y': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y,
                        'z': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z,
                    },
                    'right_hip': {
                        'x': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,
                        'y': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y,
                        'z': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z,
                    },
                    'left_knee': {
                        'x': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                        'y': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y,
                        'z': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z,
                    },
                    'right_knee': {
                        'x': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                        'y': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                        'z': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].z,
                    },
                    'left_ankle': {
                        'x': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                        'y': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                        'z': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].z,
                    },
                    'right_ankle': {
                        'x': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                        'y': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                        'z': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].z,
                    },
                    'left_armpit': {
                        'x': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        'y': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                        'z': results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z,
                    },
                    'right_armpit': {
                        'x': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                        'y': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                        'z': results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z,
                    }
                }

                results_data.append(joints)
            frame_idx += 1

        return results_data

@app.route('/display/<filename>/<data_filename>')
def display_video(filename, data_filename):
    return render_template('display_video.html', video_filename=filename, data_filename=data_filename)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/json/<filename>')
def get_json(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
