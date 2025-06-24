from ultralytics import YOLO
import supervision as sv
import cv2
import pickle
import os
from utils import get_center,get_width

class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 15
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_detections = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += batch_detections
        return detections
    
    def object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
                return tracks
        
        detections = self.detect_frames(frames)

        tracks={
            "player": [],
            "referee": [],
            "ball": [],
            "goalkeeper": []
        }

        for frame_num, detections in enumerate(detections):
            class_names = detections.names
            class_names_inverse = {v: k for k, v in class_names.items()}
            
            # Convert detections to supervision format
            detections_supervision = sv.Detections.from_ultralytics(detections)

            # Track objects
            detection_tracks = self.tracker.update_with_detections(detections_supervision)

            tracks["player"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})
            tracks["goalkeeper"].append({})

            for frame_detections in detection_tracks:
                bbox= frame_detections[0].tolist()
                class_id = frame_detections[3]
                track_id = frame_detections[4]

                if class_id == class_names_inverse['player']:
                    tracks["player"][frame_num][track_id] = {"bbox":bbox}
                if class_id == class_names_inverse['referee']:
                    tracks["referee"][frame_num][track_id] = {"bbox":bbox}
                if class_id == class_names_inverse['goalkeeper']:
                    tracks["goalkeeper"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detections in detections_supervision:
                bbox= frame_detections[0].tolist()
                class_id = frame_detections[3]
                if class_id == class_names_inverse['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_elipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3]) # bottom y-coordinate
        x_center, _ = get_center(bbox)
        width = get_width(bbox)

        cv2.ellipse(
            frame,
            (int(x_center), int(y2)),              # center
            (int(width), int(width / 4)),  # axes
            0,                           # angle
            -45,                          # startAngle
            235,                         # endAngle
            color,                       # color
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = int(x_center - rectangle_width / 2)
        x2_rect = int(x_center + rectangle_width / 2)
        y1_rect = int(y2 - rectangle_height/2) + 15
        y2_rect = int(y2 + rectangle_height/2) + 15

        x1_text = x1_rect + 12
        if track_id is not None and track_id > 99:
            x1_text -= 10

        if(track_id is not None):
            cv2.rectangle(
                frame,
                (x1_rect, y1_rect),
                (x2_rect, y2_rect),
                color,
                cv2.FILLED
            )
            cv2.putText(
                frame,
                f"{track_id}",
                (x1_text, y1_rect + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
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
                colour = player.get("team_colours", (255, 0, 0))  # Default to blue if no colour is set
                frame = self.draw_elipse(frame, player["bbox"], colour, track_id)
            
            for track_id, goalkeeper in goalkeeper_dict.items():
                frame = self.draw_elipse(frame, goalkeeper["bbox"], (100, 175, 250), track_id)

            for track_id, referee in referee_dict.items():
                frame = self.draw_elipse(frame, referee["bbox"], (0, 255, 255))

            annotated_frames.append(frame)

        return annotated_frames