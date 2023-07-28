import os
import imageio
from datetime import datetime


def get_file_creation_time(file_path):
    # Get the creation time of a file
    timestamp = os.path.getmtime(file_path)
    return datetime.fromtimestamp(timestamp)

    

    # Folder path containing the images
folder_path = 'C:/Users/ycche/git repo/slmTophat/tempPNG'

# List to store image file names
image_files = []

# Iterate through the files in the folder
for file in os.listdir(folder_path):
    if file.startswith('plot_') and file.endswith('.png'):
        image_files.append(os.path.join(folder_path, file))

# Sort the image files based on creation time
image_files.sort(key=get_file_creation_time)

# Create an empty list to store frames
frames = []

# Read the images and add them to the frames list
for image_file in image_files:
    frame = imageio.imread(image_file)
    frames.append(frame)

# Set the file path and name for the output video
output_path = 'output.mp4'

writer = imageio.get_writer(output_path, format='mp4', mode='I', fps=10)

# Write the frames to the video
for frame in frames:
    writer.append_data(frame)

# Close the writer
writer.close()