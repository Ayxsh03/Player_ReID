import numpy as np

from utils import read_video, save_video
from tracker import Tracker, ReIDTracker
from team_assigner import TeamAssigner

def main():
    video_frames = read_video('input/15sec_input_720p.mp4')
    #trackers = Tracker('models/best.pt')
    trackers = ReIDTracker('models/best.pt','models/osnet_x1_0_msmt17.pt')

    tracks = trackers.object_tracks(video_frames, read_from_stub=True, stub_path='stubs/tracks_stub.pkl')


    team_assigner = TeamAssigner()
    team_assigner.assign_teams_color(video_frames[0], tracks['player'][0])

    for frame_num, player_track in enumerate(tracks['player']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['player'][frame_num][player_id]['team'] = team 
            tracks['player'][frame_num][player_id]['team_colours'] = team_assigner.team_colours[team]


    output_frames = trackers.draw_annotations(video_frames, tracks)
    save_video(output_frames, 'runs/output.mp4')


if __name__ == "__main__":
    main()