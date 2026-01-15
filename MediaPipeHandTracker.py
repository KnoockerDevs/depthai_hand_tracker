"""
MediaPipe-based Hand Tracker for standard RGB webcams.
Provides the same interface as HandTrackerEdge for compatibility with HandController.
"""

import cv2
import numpy as np
from mediapipe_utils import HandRegion, recognize_gesture

# Check if mediapipe is installed
try:
    import mediapipe as mp
except ImportError:
    print("ERROR: The 'mediapipe' package is required for RGB camera mode.")
    print("Install it with: pip install mediapipe")
    raise

# Try the new task-based API first, fall back to legacy solutions API
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    USE_TASKS_API = True
except ImportError:
    USE_TASKS_API = False


class MediaPipeHandTracker:
    """
    Hand tracker using MediaPipe Hands solution for standard RGB webcams.
    
    Provides the same interface as HandTrackerEdge:
    - img_w, img_h: frame dimensions
    - next_frame(): returns (frame, hands, bag)
    - exit(): cleanup resources
    
    Arguments:
    - camera_id: webcam device ID (default 0)
    - resolution: tuple (width, height) for camera resolution
    - max_num_hands: maximum number of hands to detect (1 or 2)
    - min_detection_confidence: minimum confidence for detection
    - min_tracking_confidence: minimum confidence for tracking
    - use_gesture: whether to recognize gestures (default True)
    """
    
    def __init__(self,
                 camera_id=0,
                 resolution=(1280, 720),
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 use_gesture=True,
                 # Ignored parameters for compatibility with HandTrackerEdge config
                 pd_score_thresh=None,
                 pd_nms_thresh=None,
                 lm_score_thresh=None,
                 solo=None,
                 internal_fps=None,
                 internal_frame_height=None,
                 xyz=None,
                 **kwargs):
        
        self.camera_id = camera_id
        self.use_gesture = use_gesture
        self.max_num_hands = max_num_hands
        
        # Use solo parameter if provided (solo=True means 1 hand, solo=False means 2 hands)
        if solo is not None:
            self.max_num_hands = 1 if solo else 2
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Get actual resolution (may differ from requested)
        self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"MediaPipe Hand Tracker initialized")
        print(f"Camera resolution: {self.img_w} x {self.img_h}")
        
        # Initialize MediaPipe Hands based on available API
        self.use_tasks_api = USE_TASKS_API
        self.hands = None
        self._detection_result = None
        
        if self.use_tasks_api:
            # New task-based API (MediaPipe 0.10+)
            base_options = python.BaseOptions(model_asset_path=self._get_model_path())
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.hands = vision.HandLandmarker.create_from_options(options)
            self._timestamp_ms = 0
        else:
            # Legacy solutions API
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
    
    def _get_model_path(self):
        """Get path to hand landmarker model, downloading if necessary."""
        import os
        from pathlib import Path
        import urllib.request
        
        # Store model in the same directory as this script
        model_dir = Path(__file__).parent / "models"
        model_path = model_dir / "hand_landmarker.task"
        
        if not model_path.exists():
            print("Downloading MediaPipe hand landmarker model...")
            model_dir.mkdir(parents=True, exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print(f"Model downloaded to {model_path}")
        
        return str(model_path)
        
    def _create_hand_region(self, hand_landmarks, handedness, frame_width, frame_height):
        """
        Convert MediaPipe hand landmarks (legacy API) to HandRegion object compatible with existing code.
        """
        hand = HandRegion()
        
        # Extract 2D landmarks in pixel coordinates
        landmarks_2d = []
        for lm in hand_landmarks.landmark:
            x = int(lm.x * frame_width)
            y = int(lm.y * frame_height)
            landmarks_2d.append([x, y])
        hand.landmarks = np.array(landmarks_2d, dtype=np.int32)
        
        # Extract normalized 3D landmarks (for gesture recognition)
        norm_landmarks = []
        for lm in hand_landmarks.landmark:
            norm_landmarks.append([lm.x, lm.y, lm.z])
        hand.norm_landmarks = np.array(norm_landmarks, dtype=np.float32)
        
        # Handedness (MediaPipe returns "Left" or "Right" from subject's perspective)
        hand_label = handedness.classification[0].label.lower()
        hand.label = hand_label
        hand.handedness = 1.0 if hand.label == "right" else 0.0
        
        # Calculate bounding rectangle from landmarks
        x_coords = hand.landmarks[:, 0]
        y_coords = hand.landmarks[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        hand.rect_x_center_a = (x_min + x_max) / 2
        hand.rect_y_center_a = (y_min + y_max) / 2
        hand.rect_w_a = x_max - x_min
        hand.rect_h_a = y_max - y_min
        hand.rotation = 0  # MediaPipe doesn't provide rotation directly
        
        # Calculate rect_points (4 corners of bounding box)
        hand.rect_points = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ]
        
        # Landmark score (use average visibility/presence)
        hand.lm_score = handedness.classification[0].score
        
        # Gesture recognition
        hand.gesture = None
        if self.use_gesture:
            recognize_gesture(hand)
        
        return hand
    
    def _create_hand_region_tasks(self, hand_landmarks, handedness, frame_width, frame_height):
        """
        Convert MediaPipe hand landmarks (tasks API) to HandRegion object compatible with existing code.
        """
        hand = HandRegion()
        
        # Extract 2D landmarks in pixel coordinates
        landmarks_2d = []
        for lm in hand_landmarks:
            x = int(lm.x * frame_width)
            y = int(lm.y * frame_height)
            landmarks_2d.append([x, y])
        hand.landmarks = np.array(landmarks_2d, dtype=np.int32)
        
        # Extract normalized 3D landmarks (for gesture recognition)
        norm_landmarks = []
        for lm in hand_landmarks:
            norm_landmarks.append([lm.x, lm.y, lm.z])
        hand.norm_landmarks = np.array(norm_landmarks, dtype=np.float32)
        
        # Handedness (MediaPipe returns "Left" or "Right" from subject's perspective)
        hand_label = handedness[0].category_name.lower()
        hand.label = hand_label
        hand.handedness = 1.0 if hand.label == "right" else 0.0
        
        # Calculate bounding rectangle from landmarks
        x_coords = hand.landmarks[:, 0]
        y_coords = hand.landmarks[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        hand.rect_x_center_a = (x_min + x_max) / 2
        hand.rect_y_center_a = (y_min + y_max) / 2
        hand.rect_w_a = x_max - x_min
        hand.rect_h_a = y_max - y_min
        hand.rotation = 0  # MediaPipe doesn't provide rotation directly
        
        # Calculate rect_points (4 corners of bounding box)
        hand.rect_points = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ]
        
        # Landmark score (use confidence from handedness)
        hand.lm_score = handedness[0].score
        
        # Gesture recognition
        hand.gesture = None
        if self.use_gesture:
            recognize_gesture(hand)
        
        return hand
    
    def next_frame(self):
        """
        Capture next frame and detect hands.
        
        Returns:
            frame: BGR image (numpy array)
            hands: list of HandRegion objects
            bag: None (for compatibility)
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, [], None
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        hands = []
        
        if self.use_tasks_api:
            # New task-based API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            self._timestamp_ms += 33  # Approximate 30 FPS
            results = self.hands.detect_for_video(mp_image, self._timestamp_ms)
            
            if results.hand_landmarks and results.handedness:
                for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                    hand = self._create_hand_region_tasks(
                        hand_landmarks,
                        handedness,
                        self.img_w,
                        self.img_h
                    )
                    hands.append(hand)
        else:
            # Legacy solutions API
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand = self._create_hand_region(
                        hand_landmarks, 
                        handedness,
                        self.img_w, 
                        self.img_h
                    )
                    hands.append(hand)
        
        return frame, hands, None
    
    def exit(self):
        """Release resources."""
        if self.hands:
            if self.use_tasks_api:
                self.hands.close()
            else:
                self.hands.close()
        if self.cap:
            self.cap.release()
        print("MediaPipe Hand Tracker closed")

