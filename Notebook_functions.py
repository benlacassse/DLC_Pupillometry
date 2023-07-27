import csv
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt

def eucledian_distance(x1, y1, x2, y2):
    return sqrt((float(x1) - float(x2)) ** 2 + (float(y1) - float(y2)) ** 2)


def np_distance(csv_file):
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
        return x_array, y_array


def dlc_plot(x=None, y=None, filter=None):
    if x is None and y is None:
        raise ValueError("Both x and y arrays are missing.")
        
    if filter==None:
        if x is not None and y is not None:
            plt.plot(x, 'r')
            plt.plot(y, 'b')
            plt.title("Plot of horizontal and vertical pupil diameters, unfiltered")
        elif x is not None:
            plt.plot(x, 'r')
            plt.title("lot of horizontal pupil diameters, unfiltered")
        elif y is not None:
            plt.plot(y, 'b')
            plt.title("Plot of vertical pupil diameters, unfiltered")

    else:
        x = filter(x)
        y = filter(y)
        if x is not None and y is not None:
            plt.plot(x, 'r')
            plt.plot(y, 'b')
            plt.title("Plot of horizontal and vertical pupil diameters" + str(filter))
        elif x is not None:
            plt.plot(x, 'r')
            plt.title("lot of horizontal pupil diameters, unfiltered")
        elif y is not None:
            plt.plot(y, 'b')
            plt.title("Plot of vertical pupil diameters, unfiltered")
    
    plt.xlabel("Frames")
    plt.ylabel("Diameters in pixel")
    plt.show()

test = 'CVR_M36_15juinDLC_resnet50_6times60framesJul1shuffle1_70000.csv'  
dlc_plot(x=np_distance(test)[0], y=np_distance(test)[1], filter = savgol_filter)