# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Live Demo for Online TAPIR."""

import functools
import time

import cv2
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tapnet import tapir_model
from tapnet.utils import model_utils
import mediapy as media
import robosuite as suite
import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob
from tapnet.utils import viz_utils

import h5py
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper


demo_path = 'path/to/demo/video'
track_path = 'path/to/tracks/of/points'
active_points_path = 'path/to/active/points/id'

therehold = 1e3
gap_pre = 1
paint = True
#prepare for the robot environment
env = suite.make(env_name='Lift', robots='Panda', controller_configs=load_controller_config(default_controller="OSC_POSE", ), 
                 has_renderer=True, 
                 has_offscreen_renderer=False,
                 render_camera='robot0_eye_in_hand',
                 ignore_done=True,
                 use_camera_obs=False,
                 reward_shaping=True,
                 control_freq=20,)

env = VisualizationWrapper(env)
env.viewer.set_camera(5)
config = {
    "env_name": 'Lift',
    "robots": 'Panda',
    "controller_configs": load_controller_config(default_controller="OSC_POSE"), 
}

env_info = json.dumps(config)

env.reset()
#env.render()
frame = env.viewer.save()

NUM_POINTS = 30


def load_checkpoint(checkpoint_path):
  ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
  return ckpt_state["params"], ckpt_state["state"]


tapir = lambda: tapir_model.TAPIR(
    use_causal_conv=True, bilinear_interp_with_depthwise_conv=False
)


def build_online_model_init(frames, points):
  tapir_instance = tapir()
  feature_grids = tapir_instance.get_feature_grids(frames, is_training=False)
  features = tapir_instance.get_query_features(
      frames,
      is_training=False,
      query_points=points,
      feature_grids=feature_grids,
  )
  return features

def build_online_model_init(frames, points):
  tapir_instance = tapir()
  feature_grids = tapir_instance.get_feature_grids(frames, is_training=False)
  features = tapir_instance.get_query_features(
      frames,
      is_training=False,
      query_points=points,
      feature_grids=feature_grids,
  )
  return features

def build_online_model_predict(frames, features, causal_context):
  """Compute point tracks and occlusions given frames and query points."""
  tapir_instance = tapir()
  feature_grids = tapir_instance.get_feature_grids(frames, is_training=False)
  trajectories = tapir_instance.estimate_trajectories(
      frames.shape[-3:-1],
      is_training=False,
      feature_grids=feature_grids,
      query_features=features,
      query_points_in_video=None,
      query_chunk_size=64,
      causal_context=causal_context,
      get_causal_context=True,
  )
  causal_context = trajectories["causal_context"]
  del trajectories["causal_context"]
  return {k: v[-1] for k, v in trajectories.items()}, causal_context


def get_frame(video_capture):
  r_val, image = video_capture.read()
  trunc = np.abs(image.shape[1] - image.shape[0]) // 2
  if image.shape[1] > image.shape[0]:
    image = image[:, trunc:-trunc]
  elif image.shape[1] < image.shape[0]:
    image = image[trunc:-trunc]
  return r_val, image

def get_points_feature(params, state, rng, acpoints, demo_path, track_path, frame_id=0):
  image = media.read_video(demo_path)[frame_id]
  query_points = np.load(track_path)[:, frame_id, :].take(acpoints.tolist(), 0)
  for point in query_points:
    point[0], point[1] = point[1], point[0]
    point[1] = image.shape[1] - point[1]
  #print(query_points.shape)
  query_points = np.concatenate([np.zeros((query_points.shape[0], 1)), query_points], axis=1)
  query_features, _ = online_init_apply(
      frames=model_utils.preprocess_frames(image[None, None]),
      points=query_points[None],
  )
  causal_state = hk.transform_with_state(lambda : tapir().construct_initial_causal_state(
      NUM_POINTS, len(query_features.resolutions) - 1
  )).apply(params=params, state=state, rng=rng)[0]

  return query_features, causal_state

def compute_jacobi(pos):
  for i in range(pos.shape[0]):
    if i == 0:
      j = np.array([[1, 0,-(pos[i, 0]-400), -(pos[i, 1]-640)], 
                    [0, 1, -(pos[i, 1]-640), (pos[i, 0]-400)]])
    else:
      j = np.concatenate([j, np.array([[1, 0,-(pos[i, 0]-400), -(pos[i, 1]-640)], 
                                       [0, 1, -(pos[i, 1]-640), (pos[i, 0]-400)]])], axis=0)
  #print(j.shape)
  return j
      
def jacobi_sim(pos):
  for i in range(pos.shape[0]):
    if i == 0:
      j = np.array([[1, 0], [0, 1]])
    else:
      j = np.concatenate([j, np.array([[1, 0], [0, 1]])], axis=0)
  return j

def get_action(pos, demo_pos):
  #print(obs.shape)
  jacobi_p = compute_jacobi(pos=pos)
  jacobi_g = compute_jacobi(pos=demo_pos)
  #print(jacobi.shape)
  #print(pos.shape)
  #print(demo_pos.shape)
  #vs = np.linalg.solve(jacobi.transpose()@jacobi, jacobi.transpose()@(demo_pos.flatten('C')-pos.flatten('C')))
  vs = 0.5*(np.linalg.lstsq(jacobi_p, (demo_pos.flatten('C')-pos.flatten('C')))[0] - np.linalg.lstsq(jacobi_g, (pos.flatten('C')-demo_pos.flatten('C')))[0])
  #print(vs)
  #print(np.array(vs).shape)
  #vs = np.array(vs).reshape(4)
  vs = 0.5*vs/np.linalg.norm(vs[0:3])

  return np.concatenate([vs[0:3], np.array([0, 0, vs[3]]), [0]])
def getac(pos, demo_pos):
  jacobi = jacobi_sim(pos)
  vs = np.linalg.lstsq(jacobi, (demo_pos.flatten('C')-pos.flatten('C')))[0]
  vs = vs/np.linalg.norm(vs)
  return np.concatenate([vs, np.zeros(5)], axis=0)

def get_vector(pos, demo_pos):
  vector = np.zeros(2)
  for i in range(pos.shape[0]):
    vector += demo_pos[i] - pos[i]
  vector = 100*vector/np.linalg.norm(vector)
  return vector

def get_gap(pos, demo_pos, gap_pre):
  num = 0
  for i in range(len(pos)):
    if np.linalg.norm(pos[i]-demo_pos[i]) < gap_pre:
      num += 1
  return num/float(len(pos)) > 0.3


print("Welcome to the TAPIR live demo.")
print("Please note that if the framerate is low (<~12 fps), TAPIR performance")
print("may degrade and you may need a more powerful GPU.")

print("Loading checkpoint...")
# --------------------
# Load checkpoint and initialize
params, state = load_checkpoint(
    "checkpoints/causal_tapir_checkpoint.npy"
)

print("Creating model...")
online_init = hk.transform_with_state(build_online_model_init)
online_init_apply = jax.jit(online_init.apply)

online_predict = hk.transform_with_state(build_online_model_predict)
online_predict_apply = jax.jit(online_predict.apply)

rng = jax.random.PRNGKey(42)
online_init_apply = functools.partial(
    online_init_apply, params=params, state=state, rng=rng
)
online_predict_apply = functools.partial(
    online_predict_apply, params=params, state=state, rng=rng
)
'''
update_query_features_apply = functools.partial(
  hk.transform_with_state(lambda **kwargs: tapir().update_query_features(**kwargs)).apply,
  params=params, state=state, rng=rng
)
'''
# --------------------
# Start point tracking
#vc = cv2.VideoCapture(0)

#vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

#if vc.isOpened():  # try to get the first frame
  #rval, frame = get_frame(vc)
#else:
  #raise ValueError("Unable to open camera.")
'''
video = media.read_video('./demo_4.mp4')
frame = video[0, :, :, :]
print(frame.shape)
'''
#pos = tuple()
query_frame = True
have_point = [True] * NUM_POINTS
query_features = None
causal_state = None
next_query_idx = 0


print("Compiling jax functions (this may take a while...)")
# --------------------
# Call one time to compile
'''
query_points = jnp.zeros([NUM_POINTS, 3], dtype=jnp.float32)
print(frame.shape)
print(query_points[None, 0:1].shape)
query_features, _ = online_init_apply(
    frames=model_utils.preprocess_frames(frame[None, None]),
    points=query_points[None, 0:1],
)
jax.block_until_ready(query_features)

query_features, _ = online_init_apply(
    frames=model_utils.preprocess_frames(frame[None, None]),
    points=query_points[None],
)
causal_state = hk.transform_with_state(lambda : tapir().construct_initial_causal_state(
    NUM_POINTS, len(query_features.resolutions) - 1
)).apply(params=params, state=state, rng=rng)[0]

update_query_features_apply = functools.partial(
  hk.transform_with_state(lambda **kwargs: tapir().update_query_features(**kwargs)).apply,
  params=params, state=state, rng=rng
)


(prediction, causal_state), _ = online_predict_apply(
    frames=model_utils.preprocess_frames(frame[None, None]),
    features=query_features,
    causal_context=causal_state,
)

jax.block_until_ready(prediction["tracks"])

last_click_time = 0


def mouse_click(event, x, y, flags, param):
  del flags, param
  global pos, query_frame, last_click_time

  # event fires multiple times per click sometimes??
  if (time.time() - last_click_time) < 0.5:
    return

  if event == cv2.EVENT_LBUTTONDOWN:
    pos = (y, frame.shape[1] - x)
    query_frame = True
    last_click_time = time.time()

'''
#cv2.namedWindow("Point Tracking")
#cv2.setMouseCallback("Point Tracking", mouse_click)

t = time.time()
step_counter = 0

print("Press ESC to exit.")
active_points = np.load(active_points_path)

query_features, causal_state = get_points_feature(params=params, state=state, rng=rng, acpoints=active_points, demo_path=demo_path, track_path=track_path)
# video = media.read_video('./demo_4.mp4')
demo_tracks = np.load(track_path).take(active_points.tolist(), 0)
video_length = demo_tracks.shape[1]+1
for point in demo_tracks:
  for frame in point:
    frame[0], frame[1] = frame[1], frame[0]
#print(demo_tracks.shape)
gap = False
vid = []
start = 1
save = 1
count_list = []
def deal_track(tracks):
  for point in tracks:
    point[0], point[1] = point[1], point[0]
  return tracks

def paint_points(image, points, vis, color=(255, 0, 0)):
  for i in range(points.shape[0]):
    if int(vis[i]) > 0.5:
      image = cv2.circle(image, center=(int(points[i][1]), int(points[i][0])), radius=10, color=color, thickness=-1)
  return image

try:
  for i in range(video_length):
    #rval, frame = get_frame(vc)
    #frame = video[i+1, :, :, :]
    print('---begin ' + str(i+1)+'th frame---')
    count = 0
    while not gap:
      if start :
        frame = env.viewer.save()
      else:
        #print(demo_tracks[:, i, :].shape)
        track_select = np.array([track_compute[j] for j in range(len(track_compute)) if visibles[j] > 0.5])
        demo_select = np.array([demo_tracks[j, i, :] for j in range(len(track_compute)) if visibles[j] > 0.5])
        #print(track_select.shape[0])
        if track_select.shape[0] > 0:
          action = get_action(track_select, demo_select)
        else:
          action = get_action(track_compute, demo_tracks[:, i, :])
        #print(action)
        obs, reawrd, done, _ = env.step(action)
        frame = env.viewer.save()
        #env.render()
      #print(frame.shape)
      #vid.append(np.array(frame).copy())
      '''
      query_points = jnp.array((0,) + pos, dtype=jnp.float32)
      init_query_features, _ = online_init_apply(
          frames=model_utils.preprocess_frames(jnp.array(frame)[None, None]),
          points=query_points[None, None],
      )
      query_frame = False
      query_features, causal_state = update_query_features_apply(
          query_features=query_features, 
          new_query_features=init_query_features, 
          idx_to_update=np.array([next_query_idx]), 
          causal_state=causal_state
      )[0]
      '''
      #have_point[next_query_idx] = True
      #next_query_idx = (next_query_idx + 1) % NUM_POINTS
      (prediction, causal_state), _ = online_predict_apply(
          frames=model_utils.preprocess_frames(frame[None, None]),
          features=query_features,
          causal_context=causal_state,
      )
      track = prediction["tracks"][0, :, 0]
      occlusion = prediction["occlusion"][0, :, 0]
      expected_dist = prediction["expected_dist"][0, :, 0]
      visibles = model_utils.postprocess_occlusions(occlusion, expected_dist)
      track = np.round(track)
      track_compute = deal_track(np.array(track))
      vector = get_vector(track_compute, demo_tracks[:, i, :])
      frame = paint_points(np.array(frame), np.array(track_compute), vis=visibles)
      frame = cv2.arrowedLine(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), (int(frame.shape[1]/2+vector[1]), int(frame.shape[0]/2+vector[0])), (0, 255, 0), 2, 0, 0, 0.2)
      if not start:
        frame = cv2.arrowedLine(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), (int(frame.shape[1]/2-100*action[1]), int(frame.shape[0]/2-100*action[0])), (0, 255, 255), 2, 0, 0, 0.2)
      frame = paint_points(frame, demo_tracks[:, i, :], vis=np.ones(demo_tracks.shape[0]), color=(0, 0, 255))
      vid.append(np.array(frame).copy())
      #print(track.shape)
      if start:
        tracks = np.expand_dims(track, axis=1).copy()
        vis = np.expand_dims(visibles, axis=1).copy()
        start = 0
      else:
        tracks = np.concatenate([tracks, np.expand_dims(track, axis=1).copy()], axis=1).copy()
        vis = np.concatenate([vis, np.expand_dims(visibles, axis=1).copy()], axis=1).copy()
      if i == 0:
        gap = get_gap(track, demo_tracks[:, i, :], gap_pre=gap_pre)
      else:
        gap = 1e10
      gap_pre *= 1.3
      count += 1
    gap = False
    gap_pre = 1
    if save:
      com_tracks = np.expand_dims(track, axis=1).copy()
      save = 0
    else:
      com_tracks = np.concatenate([com_tracks, np.expand_dims(track, axis=1).copy()], axis=1).copy()
    count_list.append(count)
    '''
      for i, _ in enumerate(have_point):
        if visibles[i] and have_point[i]:
          cv2.circle(
              frame, (int(track[i, 0]), int(track[i, 1])), 5, (255, 0, 0), -1
          )
          if track[i, 0] < 16 and track[i, 1] < 16:
            print((i, next_query_idx))
    
    cv2.imshow("Point Tracking", frame[:, ::-1])
    if pos:
      step_counter += 1
      if time.time() - t > 5:
        print(f"{step_counter/(time.time()-t)} frames per second")
        t = time.time()
        step_counter = 0
    else:
      t = time.time()
    key = cv2.waitKey(1)

    if key == 27:  # exit on ESC
      break
    '''
  #cv2.destroyWindow("Point Tracking")
  #vc.release()
finally:
  print(np.array(vid).shape)
  np.save('track_test.npy', tracks)
  np.save('visibles_test.npy', vis)
  np.save('count.npy', count_list)
  #np.save('com_tracks4.npy', com_tracks)
  #vid = viz_utils.paint_point_track(np.array(vid), tracks, vis)
  media.write_video('./imitation.mp4', vid, fps=20)
