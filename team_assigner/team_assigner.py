from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colours ={}
        self.player_team = {}

    def get_clustering_model(self, image):
        image2D=image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(image2D)
        return kmeans 

    def get_player_colour(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        top_half_image = image[0: image.shape[0] // 2, :]
        kmeans = self.get_clustering_model(top_half_image)

        labels = kmeans.labels_
        clustered_image=labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        corner_cluster = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1 - non_player_cluster

        player_colour = kmeans.cluster_centers_[player_cluster]
        player_colour = np.round(player_colour).astype(int)

        return player_colour
    
    def assign_teams_color(self, frame, player_detections):
        player_colour = []
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            colour = self.get_player_colour(frame, bbox)
            player_colour.append(colour) 

        kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0)
        kmeans.fit(player_colour)
        
        self.kmeans = kmeans

        self.team_colours[1] = kmeans.cluster_centers_[0]
        self.team_colours[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team:
            return self.player_team[player_id]
        
        player_colour = self.get_player_colour(frame, player_bbox)

        team_id = self.kmeans.predict(player_colour.reshape(1,-1))[0]
        team_id += 1

        # if player_id == 106:  # Special case for goalkeeper
        #     team_id = 1
        # if player_id == 221:  # Special case for goalkeeper
        #     team_id = 2



        self.player_team[player_id] = team_id
        return team_id
    

        
        