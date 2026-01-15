"""
Simple renderer for MediaPipe Hand Tracker.
Draws hand landmarks and gestures on the frame.
"""

import cv2
import numpy as np

LINES_HAND = [[0,1],[1,2],[2,3],[3,4], 
            [0,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20],[0,17]]


class MediaPipeHandTrackerRenderer:
    """
    Simple renderer for MediaPipe hand tracking results.
    Compatible interface with HandTrackerRenderer.
    """
    
    def __init__(self, tracker, output=None):
        self.tracker = tracker
        self.show_landmarks = True
        self.show_gesture = True
        self.show_fps = True
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = None
        
        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(output, fourcc, 30, (tracker.img_w, tracker.img_h))
    
    def draw_hand(self, hand):
        """Draw landmarks and gesture for a single hand."""
        if not hasattr(hand, 'landmarks') or hand.landmarks is None:
            return
            
        # Calculate line thickness based on hand size
        thick_coef = getattr(hand, 'rect_w_a', 200) / 400
        
        if self.show_landmarks:
            # Draw hand skeleton lines
            lines = [np.array([hand.landmarks[point] for point in line]).astype(np.int32) 
                     for line in LINES_HAND]
            color = (0, 255, 0) if getattr(hand, 'handedness', 0.5) > 0.5 else (0, 0, 255)
            cv2.polylines(self.frame, lines, False, color, int(1 + thick_coef * 3), cv2.LINE_AA)
            
            # Draw landmark points
            radius = int(1 + thick_coef * 5)
            
            # Color based on finger states if available (for gesture)
            if hasattr(hand, 'thumb_state'):
                finger_color = {1: (0, 255, 0), 0: (0, 0, 255), -1: (0, 255, 255)}
                cv2.circle(self.frame, tuple(hand.landmarks[0]), radius, finger_color[-1], -1)
                for i in range(1, 5):
                    cv2.circle(self.frame, tuple(hand.landmarks[i]), radius, finger_color[hand.thumb_state], -1)
                for i in range(5, 9):
                    cv2.circle(self.frame, tuple(hand.landmarks[i]), radius, finger_color[hand.index_state], -1)
                for i in range(9, 13):
                    cv2.circle(self.frame, tuple(hand.landmarks[i]), radius, finger_color[hand.middle_state], -1)
                for i in range(13, 17):
                    cv2.circle(self.frame, tuple(hand.landmarks[i]), radius, finger_color[hand.ring_state], -1)
                for i in range(17, 21):
                    cv2.circle(self.frame, tuple(hand.landmarks[i]), radius, finger_color[hand.little_state], -1)
            else:
                # Default coloring
                for x, y in hand.landmarks[:, :2]:
                    cv2.circle(self.frame, (int(x), int(y)), radius, (0, 128, 255), -1)
        
        # Draw gesture label
        if self.show_gesture and hasattr(hand, 'gesture') and hand.gesture:
            info_ref_x = hand.landmarks[0, 0]
            info_ref_y = np.max(hand.landmarks[:, 1])
            cv2.putText(self.frame, hand.gesture, (info_ref_x - 20, info_ref_y - 50),
                       cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    
    def draw(self, frame, hands, bag=None):
        """Draw all hands on frame."""
        self.frame = frame
        for hand in hands:
            self.draw_hand(hand)
        return self.frame
    
    def exit(self):
        """Cleanup resources."""
        if self.output:
            self.output.release()
        cv2.destroyAllWindows()
    
    def waitKey(self, delay=1):
        """Display frame and wait for key."""
        # Simple FPS calculation
        import time
        self.frame_count += 1
        current_time = time.time()
        if self.last_fps_time is None:
            self.last_fps_time = current_time
        elif current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        
        if self.show_fps:
            cv2.putText(self.frame, f"FPS: {self.fps:.1f}", (50, 50),
                       cv2.FONT_HERSHEY_PLAIN, 2, (240, 180, 100), 2)
        
        # Mirror the display horizontally for natural interaction
        display_frame = cv2.flip(self.frame, 1)
        cv2.imshow("Hand tracking", display_frame)
        if self.output:
            self.output.write(self.frame)
        
        key = cv2.waitKey(delay)
        if key == 32:  # Space bar to pause
            key = cv2.waitKey(0)
        elif key == ord('4'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('7'):
            self.show_gesture = not self.show_gesture
        elif key == ord('f'):
            self.show_fps = not self.show_fps
        
        return key

