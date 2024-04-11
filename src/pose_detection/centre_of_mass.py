import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the points for key locations on the body
# Example coordinates (x, y, z)
key_points = {
    'left_hand': (-20, 80, 60),
    'left_elbow': (-15, 90, 30),
    'left_shoulder': (-10, 140, 0),
    'right_hand': (20, 80, 60),
    'right_elbow': (15, 90, 30),
    'right_shoulder': (10, 140, 0),
    'head_top': (0, 160, 0),
    'left_hip': (-5, 90, -20),
    'right_hip': (5, 90, -20),
    'left_knee': (-5, 70, 30),
    'right_knee': (5, 70, 30),
    'left_foot': (-5, 30, 80),
    'right_foot': (5, 30, 80)
}

# Set up the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the key points
for point in key_points.values():
    ax.scatter(*point, color='blue')

# Connect the points to represent the limbs and torso
limbs = [
    ('left_hand', 'left_elbow'), ('left_elbow', 'left_shoulder'),
    ('right_hand', 'right_elbow'), ('right_elbow', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_foot'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_foot'),
    ('left_shoulder', 'head_top'), ('right_shoulder', 'head_top')
]

for limb in limbs:
    start, end = limb
    ax.plot([key_points[start][0], key_points[end][0]],
            [key_points[start][1], key_points[end][1]],
            [key_points[start][2], key_points[end][2]], 'gray')

# Set labels
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')

# Calculate centers of mass for segments based on the key points (simplified approach)
# Here, the center of mass for each limb is approximated as the midpoint of the limb
segment_centers = {
    'left_arm': ((key_points['left_hand'][0] + key_points['left_elbow'][0]) / 2,
                 (key_points['left_hand'][1] + key_points['left_elbow'][1]) / 2,
                 (key_points['left_hand'][2] + key_points['left_elbow'][2]) / 2),
    'right_arm': ((key_points['right_hand'][0] + key_points['right_elbow'][0]) / 2,
                  (key_points['right_hand'][1] + key_points['right_elbow'][1]) / 2,
                  (key_points['right_hand'][2] + key_points['right_elbow'][2]) / 2),
    'left_leg': ((key_points['left_foot'][0] + key_points['left_knee'][0]) / 2,
                 (key_points['left_foot'][1] + key_points['left_knee'][1]) / 2,
                 (key_points['left_foot'][2] + key_points['left_knee'][2]) / 2),
    'right_leg': ((key_points['right_foot'][0] + key_points['right_knee'][0]) / 2,
                  (key_points['right_foot'][1] + key_points['right_knee'][1]) / 2,
                  (key_points['right_foot'][2] + key_points['right_knee'][2]) / 2),
    'torso': ((key_points['left_shoulder'][0] + key_points['right_hip'][0]) / 2,
              (key_points['left_shoulder'][1] + key_points['right_hip'][1]) / 2,
              (key_points['left_shoulder'][2] + key_points['right_hip'][2]) / 2),
    'head': key_points['head_top']  # Approximation for visualization
}

# Set up the plot again
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the key points
for point in key_points.values():
    ax.scatter(*point, color='blue')

# Connect the points to represent the limbs and torso
for limb in limbs:
    start, end = limb
    ax.plot([key_points[start][0], key_points[end][0]],
            [key_points[start][1], key_points[end][1]],
            [key_points[start][2], key_points[end][2]], 'gray')

# Plot segment centers of mass
for center in segment_centers.values():
    ax.scatter(*center, color='green')

# Assuming each segment has equal mass for simplicity and calculating the overall center of mass
segment_mass = 70 / len(segment_centers)  # Simplified equal mass for each segment
total_center_of_mass = [sum(x * segment_mass for x in coords) / 70 for coords in zip(*segment_centers.values())]

# Plot the overall center of mass
ax.scatter(*total_center_of_mass, color='red', s=100, label='Overall Center of Mass')

# Set labels and legend
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.legend()

plt.show()

