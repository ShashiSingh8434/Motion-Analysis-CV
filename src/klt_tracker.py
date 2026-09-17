import cv2
import numpy as np

class KLTTracker:
    def __init__(self, max_points=100, quality_level=0.3, min_distance=7, block_size=7):
        self.feature_params = dict(maxCorners=max_points,
                                   qualityLevel=quality_level,
                                   minDistance=min_distance,
                                   blockSize=block_size)
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray = None
        self.p0 = None
        
        self.tracks = [] # List of active tracks, each track is a list of (x,y)
        self.track_len = 50 # How many points to keep in track history
        
        self.total_detected = 0
        self.total_tracked = 0
        self.displacements = []
        
    def compute(self, current_gray):
        if self.prev_gray is None:
            self.prev_gray = current_gray
            self.p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            if self.p0 is not None:
                self.tracks = [[tuple(p[0])] for p in self.p0]
                self.total_detected += len(self.p0)
            return self.tracks, 0.0

        if self.p0 is None or len(self.p0) == 0:
            self.p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            if self.p0 is not None:
                self.tracks = [[tuple(p[0])] for p in self.p0]
                self.total_detected += len(self.p0)
            else:
                self.prev_gray = current_gray
                return self.tracks, 0.0

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, current_gray, self.p0, None, **self.lk_params)
        
        avg_displacement = 0.0
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            
            valid_indices = np.where(st == 1)[0]
            new_tracks = []
            
            frame_displacement = 0.0
            tracked_count = 0
            
            for i, idx in enumerate(valid_indices):
                track = self.tracks[idx]
                track.append(tuple(good_new[i]))
                if len(track) > self.track_len:
                    del track[0]
                new_tracks.append(track)
                
                dist = np.linalg.norm(good_new[i] - good_old[i])
                frame_displacement += dist
                tracked_count += 1
                
            self.tracks = new_tracks
            self.p0 = good_new.reshape(-1, 1, 2)
            self.total_tracked += len(good_new)
            
            if tracked_count > 0:
                avg_displacement = frame_displacement / tracked_count
                self.displacements.append(avg_displacement)
                
        # Re-detect if we lose too many points
        if self.p0 is None or len(self.p0) < 10:
            new_features = cv2.goodFeaturesToTrack(current_gray, mask=None, **self.feature_params)
            if new_features is not None:
                self.p0 = new_features
                self.tracks = [[tuple(p[0])] for p in self.p0]
                self.total_detected += len(self.p0)

        self.prev_gray = current_gray
        return self.tracks, avg_displacement
