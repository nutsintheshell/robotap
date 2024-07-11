import numpy as np
import math
import mediapy as media
from tapnet.utils import viz_utils
therehold_ex = 10000
therehold_in = 1e-4
track_4 = np.load('tracks_4.npy')
track_1 = np.load('tracks_1.npy')
track_2 = np.load('tracks_2.npy')
track_3 = np.load('tracks_3.npy')
track_5 = np.load('tracks_5.npy')
visibles_4 = np.load('visibles_4.npy')
visibles_1 = np.load('visibles_1.npy')
visibles_2 = np.load('visibles_2.npy')
visibles_3 = np.load('visibles_3.npy')
visibles_5 = np.load('visibles_5.npy')
def compute_distance(points):
    dis = 0
    print(points.shape)
    for i in range(points.shape[0]):
        dis += math.sqrt((points[i, 0]-points[(i+1)%points.shape[1], 0])**2 + (points[i, 1]-points[(i+1)%points.shape[1], 1])**2)
    return dis

active_points = []
tracks = []
visibles = []
for i in range(track_1.shape[0]):
    distance_ex = compute_distance(np.concatenate((np.expand_dims(track_1[i, -1, :], axis=0), 
                                     np.expand_dims(track_2[i, -1, :], axis=0), 
                                     np.expand_dims(track_3[i, -1, :], axis=0), 
                                     np.expand_dims(track_4[i, -1, :], axis=0), 
                                     np.expand_dims(track_5[i, -1, :], axis=0))))
    distance_in = compute_distance(np.concatenate((np.expand_dims(track_1[i, 0, :], axis=0), 
                                                   np.expand_dims(track_1[i, -1, :], axis=0))))
    distance_in += compute_distance(np.concatenate((np.expand_dims(track_2[i, 0, :], axis=0), 
                                                   np.expand_dims(track_2[i, -1, :], axis=0))))
    distance_in += compute_distance(np.concatenate((np.expand_dims(track_3[i, 0, :], axis=0), 
                                                   np.expand_dims(track_3[i, -1, :], axis=0))))
    distance_in += compute_distance(np.concatenate((np.expand_dims(track_4[i, 0, :], axis=0), 
                                                   np.expand_dims(track_4[i, -1, :], axis=0))))
    distance_in += compute_distance(np.concatenate((np.expand_dims(track_5[i, 0, :], axis=0), 
                                                   np.expand_dims(track_5[i, -1, :], axis=0))))
    
    
    
    
    if distance_ex < therehold_ex and distance_in > therehold_in and visibles_1[i, -1] > 0.5 \
        and visibles_2[i, -1] > 0.5 and visibles_3[i, -1] > 0.5 and  visibles_4[i, -1] > 0.5 and visibles_5[i, -1] > 0.5:
        active_points.append(i)
        tracks.append(track_4[i, :, :])
        visibles.append(visibles_4[i, :])
    
np.save('active_points.npy', np.array(active_points, dtype=np.int32))
print( np.array(active_points, dtype=np.int32).shape)
video = media.read_video('./demo_4.mp4')

print(np.array(tracks).shape)
print(np.array(visibles).shape)
video_viz = viz_utils.paint_point_track(video, np.array(tracks), np.array(visibles))
media.write_video('./active.mp4', video_viz, fps=20)
