

from ultralytics import YOLOv10
MODEL_PATH = 'yolov10n.pt'
model = YOLOv10 (MODEL_PATH)

image_path = './images/HCMC_Street.jpg'
result = model(source=image_path) [0]

result.save('./images/HCMC_Street_predict.jpg')


import cv2
import os
from pytube import YouTube
from ultralytics import YOLOv10
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Define the YouTube video URL
YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=9PRMLbfEFXg"

# Function to download the YouTube video
def download_youtube_video(youtube_url, output_path):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(progressive=False, file_extension='mp4').order_by('resolution').desc().first()
    stream.download(output_path=output_path)
    return os.path.join(output_path, stream.default_filename)

# Path in Google Drive
base_path = '/content/drive/My Drive/Colab Notebooks'  # Adjust the folder path as needed

# Ensure the directory for storing videos exists
video_storage_path = os.path.join(base_path, 'videos')
if not os.path.exists(video_storage_path):
    os.makedirs(video_storage_path)

# Download the video to the local directory
video_path = download_youtube_video(YOUTUBE_VIDEO_URL, video_storage_path)

# Load the YOLOv10 model
model = YOLOv10("yolov10n.pt")

# Function to extract frames from a video
def extract_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{i}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)

    cap.release()
    return frame_paths

# Extract frames from the downloaded video
frame_output_dir = os.path.join(base_path, 'frames')
frame_paths = extract_frames(video_path, frame_output_dir)

# Process each frame with the YOLOv10 model
processed_frame_dir = os.path.join(base_path, 'processed_frames')
if not os.path.exists(processed_frame_dir):
    os.makedirs(processed_frame_dir)

processed_frame_paths = []
for i, frame_path in enumerate(frame_paths):
    frame = cv2.imread(frame_path)
    results = model(frame)
    processed_frame_path = os.path.join(processed_frame_dir, f"frame_{i}.jpg")
    results[0].save(processed_frame_path)
    processed_frame_paths.append(processed_frame_path)

# Function to merge frames into a video
def frames_to_video(frame_paths, output_video_path, fps=30):
    # Read the first frame to get the video size
    frame = cv2.imread(frame_paths[0])
    height, width, layers = frame.shape
    size = (width, height)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()

# Merge processed frames into a video
processed_video_path = os.path.join(base_path, 'processed_video.mp4')
frames_to_video(processed_frame_paths, processed_video_path)

# Function to combine video with original audio
def combine_video_audio(video_path, audio_path, output_path):
    video_clip = VideoFileClip(video_path)
    original_clip = VideoFileClip(audio_path)
    final_clip = video_clip.set_audio(original_clip.audio)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

# Combine the processed video with the original audio
final_video_path = os.path.join(base_path, 'final_video_with_audio.mp4')
combine_video_audio(processed_video_path, video_path, final_video_path)

print(f"Final video with original audio saved to: {final_video_path}")



YAML_PATH = '/content/yolov10/Safety_Helmet_Dataset.zip'
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 256

model.train ( data = YAML_PATH,
 epochs = EPOCHS,
 batch = BATCH_SIZE,
 imgsz = IMG_SIZE )