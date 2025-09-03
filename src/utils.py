import os
import cv2
import torch
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

CLASS_NAMES = {
  0: 'Car',
  1: 'Pedestrian',
  2: 'Motorcycle',
  3: 'Traffic-Sign',
  4: 'Traffic-Light'
}


def iou(box1, box2):
  """
  Calculates iou for two rectangles

  :param box1: rectangle 1
  :param box2: rectangle 2
  :return: iou
  """
  x1 = max(box1[0], box2[0])
  y1 = max(box1[1], box2[1])
  x2 = min(box1[2], box2[2])
  y2 = min(box1[3], box2[3])
  inter = max(0, x2 - x1) * max(0, y2 - y1)
  area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
  area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
  union = area1 + area2 - inter
  return inter / union if union > 0 else 0


def write_frames_with_masks(input_dir, output_dir, model, do_ewm_smoothing):
  """
  Applies object detection boxes to files from input_dir using model, writes
  them to output_dir, optionally performs exponential smoothing

  :param input_dir: directory with video frames
  :param output_dir: directory where frames with object detection boxes should be written
  """
  video_frame_height = 1080
  video_frame_width = 1920
  video_frame_new_height = video_frame_height / video_frame_width * 1024
  video_frame_height_one_side_padding = (1024 - video_frame_new_height) / 2

  ewm_alpha = 0.5

  transform = A.Compose([
      A.LongestMaxSize(1024),
      A.PadIfNeeded(1024, 1024, border_mode=0, value=0),
      A.Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)),
      ToTensorV2()
  ])
    
  os.makedirs(output_dir, exist_ok=True)

  frame_files = sorted(os.listdir(input_dir))

  object_tracks = {}

  next_id = 0
  used_ids = set()

  for frame_idx, filename in tqdm(enumerate(frame_files)):
    img_path = os.path.join(input_dir, filename)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
      preds = retinanet_model(image_tensor)[0]

    boxes = preds['boxes'].cpu().numpy()
    scores = preds['scores'].cpu().numpy()
    labels = preds['labels'].cpu().numpy()

    filtered = [(box, label, score) for box, label, score in zip(boxes, labels, scores) if score > 0.5]
  
    new_smoothed = []

    for box, label, score in filtered:
      box = np.array(box)
      matched_id = None
      best_iou = 0.0

      for obj_id, track in object_tracks.items():
        if obj_id in used_ids:
          continue
        if track['label'] != label:
          continue
        iou_val = iou(box, track['box'])
        if iou_val > best_iou and iou_val > 0.5:
          matched_id = obj_id
          best_iou = iou_val

      if matched_id is None or not do_ewm_smoothing:
        smoothed_box = box
        smoothed_score = score
      else:
        prev_box = object_tracks[matched_id]['box']
        prev_score = object_tracks[matched_id]['score']
        smoothed_box = ewm_alpha * box + (1 - ewm_alpha) * prev_box
        smoothed_score = ewm_alpha * score + (1 - ewm_alpha) * prev_score
        used_ids.add(matched_id)

      next_id += 1
      object_tracks[next_id] = {'box': smoothed_box, 'label': label, 'score': smoothed_score}

      new_smoothed.append((smoothed_box, label, score))

      for box, label, score in new_smoothed:
        x1, y1, x2, y2 = map(int, box)
        y1 -= video_frame_height_one_side_padding
        y2 -= video_frame_height_one_side_padding
        x1 = int(x1 / 1024 * video_frame_width)
        x2 = int(x2 / 1024 * video_frame_width)
        y1 = int(y1 / video_frame_new_height * video_frame_height)
        y2 = int(y2 / video_frame_new_height * video_frame_height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = f'{CLASS_NAMES[label]}: {score:.2f}'
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

      cv2.imwrite(os.path.join(output_dir, filename), image)


def combine_frames_into_video(output_dir, video_name):
  """
  Creates a video named video_name from frames in output_dir

  :param output_dir: folder with frames for the video
  :param video_name: name of the video
  """
  frame_example = cv2.imread(os.path.join(output_dir, frame_files[0]))
  h, w, _ = frame_example.shape

  out_path = f'{video_name}.mp4'
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  writer = cv2.VideoWriter(out_path, fourcc, 30, (w, h))

  for filename in tqdm(frame_files):
    frame = cv2.imread(os.path.join(output_dir, filename))
    writer.write(frame)

  writer.release()