import csv
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import deeplabcut
import os
import cv2
import imageio
import shutil

def extract_uniformly_spread_frames(video_path, output_folder, num_frames=20):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    video = imageio.get_reader(video_path)
    
    # Get the total number of frames in the video
    total_frames = video.count_frames()
    
    # Calculate the step size for evenly spreading the frames
    step_size = max(total_frames // num_frames, 1)
    
    # Extract the frames at evenly spaced indices and save them as images
    for i in range(0, total_frames, step_size):
        frame = video.get_data(i)
        frame_filename = f"{output_folder}/frame_{i+1:03d}.jpg"
        imageio.imwrite(frame_filename, frame)


def create_video_from_frames(frames_folder, output_video_path):
    # Get the list of image files in the frames folder
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg")])
    
    # Read the dimensions of the first image to determine the frame size
    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    frame_height, frame_width, _ = first_frame.shape
    size = (frame_width, frame_height)
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
    out = cv2.VideoWriter(output_video_path, fourcc, 30, size)

    # Loop through the image files and add them to the video
    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        out.write(frame)
        
        # Delete the extracted frame after writing it to the video
        os.remove(frame_path)
    
    # Release the video writer object
    out.release()


def eucledian_distance(x1, y1, x2, y2):
    return sqrt((float(x1) - float(x2)) ** 2 + (float(y1) - float(y2)) ** 2)


def np_distance(csv_file):
    parts = csv_file.split('DLC_resnet50_11times60frames_cropedAug4shuffle1_800000.csv', 1)  # Split into two parts at the first occurrence of target_sequence
    if len(parts) > 1:
        video_name = parts[0]
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        x_list = []
        y_list = []
        for row in reader:
            if (row[0] in ['scorer', 'bodyparts', 'coords']):
                continue
            else:
                x_distance = eucledian_distance(row[1], row[2], row[4], row[5])
                y_distance = eucledian_distance(row[7], row[8], row[10], row[11])
                x_list.append(x_distance)
                y_list.append(y_distance)
        x_array = np.array(x_list)
        y_array = np.array(y_list)
        np.save(video_name + '_x_array.npy', x_array)
        np.save(video_name + '_y_array.npy', y_array)
        return x_array, y_array


def find_min_max(lst):
    if not lst:
        raise ValueError("The list is empty.")
    
    min_val = lst[0]
    max_val = lst[0]

    for num in lst:
        if num < min_val:
            min_val = num
        if num > max_val:
            max_val = num

    return min_val, max_val


def window_creation(csv_file, gage=15):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        x_min = []
        x_max = []
        y_min = []
        y_max = []
        for row in reader:
            if (row[0] in ['scorer', 'bodyparts', 'coords']):
                continue
            else:
                x_min.append(row[1])
                x_max.append(row[4])
                y_min.append(row[8])
                y_max.append(row[11])
        
        # starting point of window (top left corner)
        x = int(float(find_min_max(x_min)[0])) - gage
        y = int(float(find_min_max(y_min)[0])) - gage

        #height of window
        height = int(float(find_min_max(y_max)[1])) + gage + 1 - y

        #width of window
        width = int(float(find_min_max(x_max)[1])) + gage + 1 - x
        print(x, y, width, height)
        return x, y, width, height


def crop_and_downsample_video(input_video_path, output_video_path, crop_region, skip_frames):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Get the original frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the coordinates of the crop region (x, y, width, height)
    x, y, width, height = crop_region

    # Calculate the total number of frames in the original video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the step size for frame skipping
    step_size = max(skip_frames, 1)

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
    out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    frame_index = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Only process frames at specified intervals
        if frame_index % step_size == 0:
            # Crop the frame to the specified region
            cropped_frame = frame[y:y+height, x:x+width]

            # Write the cropped frame to the output video
            out.write(cropped_frame)

        frame_index += 1

        # Break the loop if we've processed all frames
        if frame_index >= total_frames:
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()


def organize_video_files(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder path '{folder_path}' does not exist.")
        return
    
    video_files = [file for file in os.listdir(folder_path) if any(file.lower().endswith(ext) for ext in ["mp4", "avi", "mov", "mkv"])]  # Add other video extensions as needed
    
    if not video_files:
        print("No video files found in the folder.")
        return
    
    # creates folder for each video
    for video_file in video_files:
        file_name = os.path.splitext(video_file)[0]
        subfolder_path = os.path.join(folder_path, file_name)
        
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        
        video_file_path = os.path.join(folder_path, video_file)
        destination_path = os.path.join(subfolder_path, video_file)
        
        shutil.move(video_file_path, destination_path)
    
    print("Organizing completed successfully!")


def process_subfolders(main_folder, numpy_array, frame_skip):
    if not os.path.exists(main_folder):
        print(f"The folder '{main_folder}' does not exist.")
        return

    subfolders = [name for name in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, name))]

    # Process the analysis of each video folder
    for subfolder_name in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder_name)
        print(subfolder_path)
        video_file = [file for file in os.listdir(subfolder_path)]
        print(video_file)
        video_file_path = os.path.join(subfolder_path, video_file[0])
        print(video_file_path)
        analysis_of_folder(video_file_path, numpy_array, frame_skip)


def analysis_of_folder(video_file_path, numpy_array, frame_skip):
    # uptain subfolder path and video name
    folder_path = os.path.dirname(video_file_path)
    video_name = os.path.basename(folder_path)

    # Upload videos and projects
    project_name = '11times60frames-Ben-2023-07-24'
    crop_project_name = '11times60frames_croped-Ben-2023-08-04'
    path_config_file = project_name + '/config.yaml'
    crop_path_config_file = crop_project_name + '/config.yaml'

    # Video extraction of 20 frames
    extract_uniformly_spread_frames(video_file_path, folder_path, num_frames=20)
    create_video_from_frames(folder_path, folder_path+'/20frames.mp4')

    # Video analysis of 20 frames
    deeplabcut.analyze_videos(path_config_file, folder_path+'/20frames.mp4', save_as_csv=True)

    # crop and downsample video
    filter_aberrant_data(folder_path+'/20framesDLC_resnet50_11times60framesJul24shuffle1_180000.csv')
    window = window_creation(folder_path+'/cleaned_20framesDLC_resnet50_11times60framesJul24shuffle1_180000.csv')
    crop_video_path = folder_path+'/'+video_name+'_analysed.mp4'
    crop_and_downsample_video(video_file_path, crop_video_path, window, frame_skip)

    # removal of data from shorten video
    os.remove(folder_path+'/20frames.mp4')
    os.remove(folder_path+'/20framesDLC_resnet50_11times60framesJul24shuffle1_180000_meta.pickle')
    os.remove(folder_path+'/20framesDLC_resnet50_11times60framesJul24shuffle1_180000.csv')
    os.remove(folder_path+'/cleaned_20framesDLC_resnet50_11times60framesJul24shuffle1_180000.csv')
    os.remove(folder_path+'/20framesDLC_resnet50_11times60framesJul24shuffle1_180000.h5')

    # Video analysis of full video
    deeplabcut.analyze_videos(crop_path_config_file, crop_video_path, save_as_csv=True)

    # numpy creation
    if numpy_array == True:
        csv_file = folder_path+'/'+video_name+'_analysedDLC_resnet50_11times60frames_cropedAug4shuffle1_800000.csv'
        # print(csv_file)
        np_distance(csv_file)
    
    # removal of croped video
    os.remove(crop_video_path)
    return f'Analysis complete of {video_name}'


def analysis(video_folder, numpy_array=True, frame_skip = 1):

    # creates folders
    organize_video_files(video_folder)

    # start analysis of each folder individually
    process_subfolders(video_folder, numpy_array, frame_skip)
    return 'Analysis complete all videos'


def dlc_plot(main_folder):
    if not os.path.exists(main_folder):
        print(f"The folder '{main_folder}' does not exist.")
        return

    subfolders = [name for name in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, name))]

    # plots a graph for each video
    for subfolder_name in subfolders:
        x = None
        y = None
        subfolder_path = os.path.join(main_folder, subfolder_name)
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('_x_array.npy'):
                x = np.load(os.path.join(subfolder_path, file_name))
            elif file_name.endswith('_y_array.npy'):
                y = np.load(os.path.join(subfolder_path, file_name))
        
        if x is None and y is None:
            raise ValueError("Both x and y arrays are missing.")
        
        if x is not None and y is not None:
            plt.plot(x, 'r', label='Horizontal')
            plt.plot(y, 'b', label='Vertical')
            plt.title("Plot of horizontal and vertical pupil diameters, " + subfolder_name)
        elif x is not None:
            plt.plot(x, 'r', label='Horizontal')
            plt.title("Plot of horizontal pupil diameters, " + subfolder_name)
        elif y is not None:
            plt.plot(y, 'b', label='Vertical')
            plt.title("Plot of vertical pupil diameters, " + subfolder_name)
        
        plt.xlabel("Frames")
        plt.ylabel("Diameters in pixel")
        plt.legend()
        
        # Save the plot in the subfolder
        plot_filename = os.path.join(subfolder_path, f"{subfolder_name}_plot.png")
        plt.savefig(plot_filename)
        
        # Display the plot
        plt.show()


def filter_aberrant_data(csv_file_path):
    folder_path, file_name = os.path.split(csv_file_path)
    cleaned_file_name = "cleaned_" + file_name

    # Create a temporary list to store the filtered data
    cleaned_data = []

    # Read the CSV file and filter out aberrant data
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)  # Read the header row and store it in a separate variable
        cleaned_data.append(header)  # Add the header to the cleaned data

        for i, row in enumerate(reader, start=1):
            if i <= 3:  # Preserve the first 3 rows
                cleaned_data.append(row)
            else:
                try:
                    # Assuming column indices 2, 5, 8, and 11 contain numerical data
                    values_to_check = [float(row[2]), float(row[5]), float(row[8]), float(row[11])]
                    if all(value >= 0.8 for value in values_to_check):
                        cleaned_data.append(row)
                except ValueError:
                    # In case any of the values are not numerical, skip the row
                    pass

    # Write the cleaned data to a new CSV file
    cleaned_file_path = os.path.join(folder_path, cleaned_file_name)
    with open(cleaned_file_path, 'w', newline='') as cleaned_csv_file:
        writer = csv.writer(cleaned_csv_file)
        writer.writerows(cleaned_data)