# from ultralytics import YOLO
# from ultralytics.trackers.bot_sort import BOTSORT
# import cv2
# import pickle
# import os
# from utils import get_center, get_width
# from types import SimpleNamespace

# class ReIDTracker:
#     def __init__(self, model_path):
#         self.model = YOLO(model_path)
#         args = self.get_default_botsort_args()
#         self.tracker = BOTSORT(args)

#     @staticmethod
#     def get_default_botsort_args():
#         return SimpleNamespace(
#             track_high_thresh=0.6,
#             track_low_thresh=0.1,
#             new_track_thresh=0.6,
#             track_buffer=30,
#             match_thresh=0.8,
#             aspect_ratio_thresh=1.6,
#             min_box_area=10,
#             mot20=False,
#             ema_alpha=0.9,
#             frame_rate=30,
#             with_reid=False,
#             fast_reid_config=None,
#             fast_reid_weights=None,
#             device='cpu',
#             gmc_method='sift',
#             proximity_thresh=0.5,
#             appearance_thresh=0.25
#         )

#     def detect_frames(self, frames):
#         batch_size = 15
#         detections = []
#         for i in range(0, len(frames), batch_size):
#             batch_detections = self.model.predict(frames[i:i + batch_size], conf=0.1)
#             detections += batch_detections
#         return detections

#     def object_tracks(self, frames, read_from_stub=False, stub_path=None):
#         if read_from_stub and stub_path is not None and os.path.exists(stub_path):
#             with open(stub_path, 'rb') as f:
#                 tracks = pickle.load(f)
#                 return tracks

#         detections = self.detect_frames(frames)

#         tracks = {
#             "player": [],
#             "referee": [],
#             "ball": [],
#             "goalkeeper": []
#         }

#         for frame_num, result in enumerate(detections):
#             names = result.names
#             class_names_inverse = {v: k for k, v in names.items()}

#             frame_tracks = self.tracker.update(
#                 [[*box.xyxy[0].tolist(), conf.item(), int(cls.item())]
#                  for box, cls, conf in zip(result.boxes, result.boxes.cls, result.boxes.conf)],
#                 frames[frame_num]
#             )

#             tracks["player"].append({})
#             tracks["referee"].append({})
#             tracks["ball"].append({})
#             tracks["goalkeeper"].append({})

#             for det in frame_tracks:
#                 x1, y1, x2, y2, track_id, cls_id = int(det[0]), int(det[1]), int(det[2]), int(det[3]), int(det[4]), int(det[5])
#                 bbox = [x1, y1, x2, y2]

#                 if cls_id == class_names_inverse.get("player"):
#                     tracks["player"][frame_num][track_id] = {"bbox": bbox}
#                 elif cls_id == class_names_inverse.get("referee"):
#                     tracks["referee"][frame_num][track_id] = {"bbox": bbox}
#                 elif cls_id == class_names_inverse.get("goalkeeper"):
#                     tracks["goalkeeper"][frame_num][track_id] = {"bbox": bbox}
#                 elif cls_id == class_names_inverse.get("ball"):
#                     tracks["ball"][frame_num]["ball"] = {"bbox": bbox}

#         if stub_path is not None:
#             with open(stub_path, 'wb') as f:
#                 pickle.dump(tracks, f)

#         return tracks

#     def draw_elipse(self, frame, bbox, color, track_id=None):
#         y2 = int(bbox[3])
#         x_center, _ = get_center(bbox)
#         width = get_width(bbox)

#         cv2.ellipse(
#             frame,
#             (int(x_center), int(y2)),
#             (int(width), int(width / 4)),
#             0,
#             -45,
#             235,
#             color,
#             2,
#             cv2.LINE_4
#         )

#         if track_id is not None:
#             rect_w, rect_h = 40, 20
#             x1_rect = int(x_center - rect_w / 2)
#             y1_rect = int(y2 - rect_h / 2) + 15
#             x2_rect = int(x_center + rect_w / 2)
#             y2_rect = int(y2 + rect_h / 2) + 15

#             x1_text = x1_rect + 12
#             if track_id > 99:
#                 x1_text -= 10

#             cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)
#             cv2.putText(frame, f"{track_id}", (x1_text, y1_rect + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

#         return frame

#     def draw_annotations(self, video_frames, tracks):
#         annotated_frames = []
#         for frame_num, frame in enumerate(video_frames):
#             frame = frame.copy()

#             player_dict = tracks["player"][frame_num]
#             ball_dict = tracks["ball"][frame_num]
#             referee_dict = tracks["referee"][frame_num]
#             goalkeeper_dict = tracks["goalkeeper"][frame_num]

#             for track_id, player in player_dict.items():
#                 frame = self.draw_elipse(frame, player["bbox"], (100, 175, 250), track_id)

#             for track_id, goalkeeper in goalkeeper_dict.items():
#                 frame = self.draw_elipse(frame, goalkeeper["bbox"], (100, 175, 250), track_id)

#             for track_id, referee in referee_dict.items():
#                 frame = self.draw_elipse(frame, referee["bbox"], (0, 255, 255))

#             annotated_frames.append(frame)

#         return annotated_frames

from ultralytics import YOLO
import cv2
import pickle
import os
from utils import get_center, get_width
import torch
import torchreid
from sklearn.metrics.pairwise import cosine_similarity as cs

class ReIDTracker:
    # def __init__(self, model_path):
    #     self.model = YOLO(model_path)
    def __init__(self, detector_path, reid_model_path, device='cpu'):
            # Detection model (YOLO)
            self.detector = YOLO(detector_path)
            # ReID model (OSNet)
            self.device = device
            self.reid_model = torchreid.models.build_model(
                name='osnet_x1_0',
                num_classes=4101,
                loss='softmax',
                pretrained=False
            )
            self.reid_model.load_state_dict(torch.load(reid_model_path, map_location=device))
            self.reid_model.eval().to(device)
            _,self.transforms = torchreid.data.transforms.build_transforms(
                height=256, width=128, is_train=False
            )
            self.embedding_db = {}
            self.next_id = 0

    def cosine_similarity(self, a, b):
        return cs([a], [b])[0][0]

    def extract(self, img):
        import cv2
        from PIL import Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = self.transforms(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.reid_model(tensor)
        return features.squeeze().cpu().numpy()

    def object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
                return tracks

        tracks = {
            "player": [],
            "referee": [],
            "ball": [],
            "goalkeeper": []
        }

        for frame_num, frame in enumerate(frames):
            results = self.detector.track(frame, persist=True, conf=0.1, tracker="botsort.yaml")[0]

            tracks["player"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})
            tracks["goalkeeper"].append({})

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls.item())
                class_name = self.detector.names[cls_id]
                bbox = [x1, y1, x2, y2]

                if class_name == "player" or class_name == "goalkeeper":
                    cropped = frame[y1:y2, x1:x2]
                    embedding = self.extract(cropped)

                    matched_id = None
                    for track_id, stored_embedding in self.embedding_db.items():
                        similarity = self.cosine_similarity(embedding, stored_embedding)
                        if similarity > 0.1:  # Tune this threshold!
                            matched_id = track_id
                            break
                    if matched_id is None:
                        matched_id = self.next_id
                        self.next_id += 1
                        self.embedding_db[matched_id] = embedding

                    if class_name == "player":
                        tracks["player"][frame_num][matched_id] = {"bbox": bbox}

                    else:
                        tracks["goalkeeper"][frame_num][matched_id] = {"bbox": bbox}

                elif class_name == "referee":
                    track_id = int(box.id.item()) if box.id is not None else -1
                    tracks["referee"][frame_num][track_id] = {"bbox": bbox}

                elif class_name == "ball":
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_elipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center(bbox)
        width = get_width(bbox)

        cv2.ellipse(
            frame,
            (int(x_center), int(y2)),
            (int(width), int(width / 4)),
            0,
            -45,
            235,
            color,
            2,
            cv2.LINE_4
        )

        if track_id is not None:
            rect_w, rect_h = 40, 20
            x1_rect = int(x_center - rect_w / 2)
            y1_rect = int(y2 - rect_h / 2) + 15
            x2_rect = int(x_center + rect_w / 2)
            y2_rect = int(y2 + rect_h / 2) + 15

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)
            cv2.putText(frame, f"{track_id}", (x1_text, y1_rect + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks):
        annotated_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["player"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referee"][frame_num]
            goalkeeper_dict = tracks["goalkeeper"][frame_num]

            for track_id, player in player_dict.items():
                frame = self.draw_elipse(frame, player["bbox"], (100, 175, 250), track_id)

            for track_id, goalkeeper in goalkeeper_dict.items():
                frame = self.draw_elipse(frame, goalkeeper["bbox"], (100, 175, 250), track_id)

            for track_id, referee in referee_dict.items():
                frame = self.draw_elipse(frame, referee["bbox"], (0, 255, 255))

            annotated_frames.append(frame)

        return annotated_frames