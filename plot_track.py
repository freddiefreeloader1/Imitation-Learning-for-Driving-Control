
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml



def plot_track(fig, ax, track_data):

    data_track = []
    L_track = track_data["track"]["trackLength"]
    arcLength = np.array(track_data["track"]["arcLength"])

    ind = np.where(arcLength <= L_track)[0]
    np.append(ind, ind[-1]+1)

    xCoords = np.array(track_data["track"]["xCoords"]) - np.array(track_data["track"]["x_init"])
    yCoords = np.array(track_data["track"]["yCoords"]) - np.array(track_data["track"]["y_init"])
    tangentAngle = np.array(track_data["track"]["tangentAngle"])
    curvature = np.array(track_data["track"]["curvature"])
    arcLength = np.array(track_data["track"]["arcLength"])

    print(np.array(track_data["track"]["arcLength"])[ind[-1]+1].tolist())

    filtered_xCoords = xCoords[ind].tolist()
    filtered_yCoords = yCoords[ind].tolist()
    filtered_tangentAngle = tangentAngle[ind].tolist()
    filtered_curvature = curvature[ind].tolist()
    filtered_arcLength = arcLength[ind].tolist()

    data_track.append(filtered_xCoords)
    data_track.append(filtered_yCoords)
    data_track.append(filtered_tangentAngle)
    data_track.append(filtered_curvature)
    data_track.append(filtered_arcLength)

    data_track = np.array(data_track)
    print(data_track.shape)

    x_track = data_track[0, :].T
    y_track = data_track[1, :].T
    theta_track = data_track[2, :].T

    left_track_x = []
    left_track_y = []
    right_track_x = []
    right_track_y = []
    max_dist = 0.84/2

    for i in range(len(x_track)):
        x_t = x_track[i]
        y_t = y_track[i]
        theta_t = theta_track[i]
        orth_t = (theta_t + np.pi / 2) % (2 * np.pi)

        left_track_x.append(x_t + max_dist * np.cos(orth_t))
        left_track_y.append(y_t + max_dist * np.sin(orth_t))
        right_track_x.append(x_t - max_dist * np.cos(orth_t))
        right_track_y.append(y_t - max_dist * np.sin(orth_t))
        

    x_coords = x_track 
    y_coords = y_track

    ax.plot(x_coords, y_coords, 'b--', label='Track')
    ax.plot(x_coords[0], y_coords[0], 'go', label='Start')
    ax.plot(x_coords[-1], y_coords[-1], 'ro', label='End')

    ax.plot(left_track_x, left_track_y, 'b-')
    ax.plot(right_track_x, right_track_y, 'b-')

    ax.set_xlabel('X coordinates')
    ax.set_ylabel('Y coordinates')
    ax.legend()
    ax.set_title('Track with Speeds in Body Frame')
    ax.grid(True)
    #ax.set_aspect('equal', adjustable='box')
    aspect_ratio = 1.0
    ax.set_aspect(aspect_ratio)

    ax.set_xlim([min(x_coords) - 1.5, max(x_coords) + 1.5])
    ax.set_ylim([min(y_coords) - 1.5, max(y_coords) + 1.5])


    return 

if __name__ == "__main__":
    with open("la_track.yaml", 'r') as stream:
        try:
            track_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    fig, ax = plt.subplots()
    plot_track(fig, ax, track_data)
    plt.show()