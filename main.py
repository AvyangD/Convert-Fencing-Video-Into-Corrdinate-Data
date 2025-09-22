import cv2
import mediapipe as mp
import pandas as pd
import os


# video_path = '/workspaces/Convert-Fencing-Video-Into-Corrdinate-Data/Fencing_data5.mp4'  
video_path = 'Fencing_data5.mp4'
print(video_path)

if not os.path.isfile(video_path):
     print(f"Error: Video file does not exist at path: {video_path}")
     exit()
else:
     print(f"Video file found: {video_path}")


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

print(f"Processing video: {video_path}")
print("Working directory:", os.getcwd())


frame_data = []
frame_count = 0


while True:
    success, frame = cap.read()
    if not success:
        print("Reached end of video or failed to read a frame.")
        break

 
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        row = {'frame': frame_count}
        for i, lm in enumerate(landmarks):
            row[f'x_{i}'] = lm.x
            row[f'y_{i}'] = lm.y
            row[f'z_{i}'] = lm.z
            row[f'visibility_{i}'] = lm.visibility
        frame_data.append(row)
        
    else:
        print(f"No pose detected in frame {frame_count}")

    frame_count += 1

cap.release()


if frame_data:
    df = pd.DataFrame(frame_data)
    df.to_csv('fencing_pose_data.csv', index=False)
    print("CSV file created with pose coordinates.")
else:
    print("No pose data extracted. CSV not created.")