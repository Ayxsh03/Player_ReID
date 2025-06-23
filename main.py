import numpy as np

from utils import read_video, save_video
from tracker import Tracker 

def main():
    video_frames = read_video('input/15sec_input_720p.mp4')
    tracker = Tracker('models/best.pt')

    tracks = tracker.object_tracks(video_frames,read_from_stub=True, stub_path='stubs/tracks_stub.pkl')

    output_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_frames, 'runs/output.mp4')


if __name__ == "__main__":
    main()