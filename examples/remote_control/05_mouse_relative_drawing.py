#!/usr/bin/env python3

import argparse
from time import monotonic


README = """
Move your mouse pointer with your hand.

ACTIVATION: Show PEACE pose with BOTH hands for 1 second to activate mouse control.
A happy sound will play when activated.

DEACTIVATION: If no mouse control (movement or click) for 2 seconds, control is
automatically deactivated. Show PEACE pose again to reactivate.

The pointer moves when your hand is doing the ONE, TWO, FIVE, or FIST pose.
The difference between ONE/FIVE and TWO/FIST is that in TWO/FIST the left button
is also pressed (drag mode).

The mouse location is calculated from the index finger tip location.
An double exponential filter is used to limit jittering.

If you have multiple screens, you may have to modify the line:
monitor = get_monitors()[0]

"""

from HandController import HandController

import sys
import ctypes
from ctypes import wintypes

# Controlling the mouse
try:
    from pynput.mouse import Button, Controller
except ModuleNotFoundError:
    print("To run this demo, you need the python package: pynput")
    print("Can be installed with: pip install pynput")
    import sys
    sys.exit()
mouse = Controller()

_SendInput = None
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000

# Primary monitor bounds (set after monitor detection)
_primary_bounds = (0, 0, 0, 0)

if sys.platform == "win32":
    try:
        ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class _INPUTUNION(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT)]

        class INPUT(ctypes.Structure):
            _anonymous_ = ("union",)
            _fields_ = [("type", wintypes.DWORD),
                        ("union", _INPUTUNION)]

        user32 = ctypes.windll.user32
        _SendInput = user32.SendInput
        # Get primary screen dimensions (not virtual desktop)
        SM_CXSCREEN = 0  # Primary screen width
        SM_CYSCREEN = 1  # Primary screen height
        _primary_width = user32.GetSystemMetrics(SM_CXSCREEN)
        _primary_height = user32.GetSystemMetrics(SM_CYSCREEN)
        _primary_bounds = (0, 0, _primary_width, _primary_height)
    except (AttributeError, OSError, ValueError):
        _SendInput = None
        MOUSEINPUT = None
        INPUT = None
else:
    MOUSEINPUT = None
    INPUT = None


def _send_input_mouse(dx, dy, flags):
    if _SendInput is None:
        return False
    inp = INPUT(type=0)
    inp.mi = MOUSEINPUT(dx=int(dx), dy=int(dy), mouseData=0, dwFlags=flags, time=0, dwExtraInfo=0)
    sent = _SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    return sent == 1


def _send_input_mouse_absolute(x, y):
    """Move cursor to absolute position on PRIMARY monitor only."""
    if _SendInput is None:
        return False
    left, top, width, height = _primary_bounds
    if width <= 0 or height <= 0:
        return False
    # Clamp coordinates to primary monitor bounds
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    # Normalize to 0-65535 range for primary screen (no VIRTUALDESK flag = primary only)
    norm_x = int(round(x * 65535 / max(1, width - 1)))
    norm_y = int(round(y * 65535 / max(1, height - 1)))
    # Use MOUSEEVENTF_ABSOLUTE without VIRTUALDESK to constrain to primary monitor
    return _send_input_mouse(norm_x, norm_y, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)


def set_cursor_absolute(x, y):
    """Set cursor position on primary monitor. x,y are relative to primary monitor (0,0 = top-left)."""
    if not _send_input_mouse_absolute(x, y):
        # Fallback using pynput - coordinates are screen-absolute
        mouse.position = (int(round(x)), int(round(y)))


def send_left_down():
    if not _send_input_mouse(0, 0, MOUSEEVENTF_LEFTDOWN):
        mouse.press(Button.left)


def send_left_up():
    if not _send_input_mouse(0, 0, MOUSEEVENTF_LEFTUP):
        mouse.release(Button.left)

# Get screen resolution - use the PRIMARY monitor only
try:
    from screeninfo import get_monitors
except ModuleNotFoundError:
    print("To run this demo, you need the python package: screeninfo")
    print("Can be installed with: pip install screeninfo")
    import sys
    sys.exit()

def get_primary_monitor():
    """Get the primary monitor. Falls back to first monitor if no primary is found."""
    monitors = get_monitors()
    for m in monitors:
        if getattr(m, 'is_primary', False):
            return m
    # Fallback: return the monitor at position (0,0) or first one
    for m in monitors:
        if m.x == 0 and m.y == 0:
            return m
    return monitors[0] if monitors else None

monitor = get_primary_monitor()
print(f"Using primary monitor: {monitor}")

# Smoothing filter
import numpy as np
class DoubleExpFilter:
    def __init__(self,smoothing=0.65,
                 correction=1.0,
                 prediction=0.85,
                 jitter_radius=250.,
                 max_deviation_radius=540.,
                 out_int=False):
        self.smoothing = smoothing
        self.correction = correction
        self.prediction = prediction
        self.jitter_radius = jitter_radius
        self.max_deviation_radius = max_deviation_radius
        self.count = 0
        self.filtered_pos = 0
        self.trend = 0
        self.raw_pos = 0
        self.out_int = out_int
        self.enable_scrollbars = False
    
    def reset(self):
        self.count = 0
        self.filtered_pos = 0
        self.trend = 0
        self.raw_pos = 0
    
    def update(self, pos):
        raw_pos = np.asanyarray(pos)
        if self.count > 0:
            prev_filtered_pos = self.filtered_pos
            prev_trend = self.trend
            prev_raw_pos = self.raw_pos
        if self.count == 0:
            self.shape = raw_pos.shape
            filtered_pos = raw_pos
            trend = np.zeros(self.shape)
            self.count = 1
        elif self.count == 1:
            filtered_pos = (raw_pos + prev_raw_pos)/2
            diff = filtered_pos - prev_filtered_pos
            trend = diff*self.correction + prev_trend*(1-self.correction)
            self.count = 2
        else:
            # First apply jitter filter
            diff = raw_pos - prev_filtered_pos
            length_diff = np.linalg.norm(diff)
            if length_diff <= self.jitter_radius:
                alpha = pow(length_diff/self.jitter_radius,1.5)
                # alpha = length_diff/self.jitter_radius
                filtered_pos = raw_pos*alpha \
                                + prev_filtered_pos*(1-alpha)
            else:
                filtered_pos = raw_pos
            # Now the double exponential smoothing filter
            filtered_pos = filtered_pos*(1-self.smoothing) \
                        + self.smoothing*(prev_filtered_pos+prev_trend)
            diff = filtered_pos - prev_filtered_pos
            trend = self.correction*diff + (1-self.correction)*prev_trend
        # Predict into the future to reduce the latency
        predicted_pos = filtered_pos + self.prediction*trend
        # Check that we are not too far away from raw data
        diff = predicted_pos - raw_pos
        length_diff = np.linalg.norm(diff)
        if length_diff > self.max_deviation_radius:
            predicted_pos = predicted_pos*self.max_deviation_radius/length_diff \
                        + raw_pos*(1-self.max_deviation_radius/length_diff)
        # Save the data for this frame
        self.raw_pos = raw_pos
        self.filtered_pos = filtered_pos
        self.trend = trend
        # Output the data
        if self.out_int:
            return predicted_pos.astype(int)
        else:
            return predicted_pos

smooth = DoubleExpFilter(smoothing=0.2, prediction=0.2, jitter_radius=200, out_int=False)

# Camera image size (populated after tracker initialization)
controller = None
cam_width = None
cam_height = None

# Drag state tracking
left_down = False
cursor_initialized = False

# Activation state - requires PEACE pose with both hands for 1 second
activated = False
activation_start_time = None
last_control_time = None  # Track last mouse control activity for auto-deactivation
ACTIVATION_DURATION = 1.0  # seconds required to hold PEACE with both hands
DEACTIVATION_TIMEOUT = 2.0  # seconds of inactivity before deactivating

# Sound for activation feedback
def play_activation_sound():
    """Play a happy sound when mouse control is activated."""
    try:
        import winsound
        # Play a pleasant ascending three-tone sound
        winsound.Beep(523, 150)  # C5
        winsound.Beep(659, 150)  # E5
        winsound.Beep(784, 200)  # G5
    except Exception:
        # Fallback: print message if sound fails
        print("🎉 Mouse control ACTIVATED!")

def play_deactivation_sound():
    """Play a descending sound when mouse control is deactivated."""
    try:
        import winsound
        # Play a descending two-tone sound
        winsound.Beep(659, 150)  # E5
        winsound.Beep(440, 200)  # A4
    except Exception:
        # Fallback: print message if sound fails
        print("Mouse control deactivated")

def check_activation(hands):
    """Check if both hands are showing PEACE pose for activation."""
    global activated, activation_start_time
    
    if activated:
        return  # Already activated, nothing to do
    
    # Check if we have exactly 2 hands both doing PEACE
    if len(hands) >= 2:
        peace_count = sum(1 for h in hands if h.gesture == "PEACE")
        if peace_count >= 2:
            # Both hands showing PEACE
            if activation_start_time is None:
                activation_start_time = monotonic()
                print("Activation pose detected... hold for 1 second")
            elif monotonic() - activation_start_time >= ACTIVATION_DURATION:
                activated = True
                print("✓ Mouse control ACTIVATED!")
                play_activation_sound()
            return
    
    # Reset if conditions not met
    if activation_start_time is not None:
        activation_start_time = None

def check_deactivation():
    """Check if mouse control should be deactivated due to inactivity."""
    global activated, last_control_time
    
    if not activated:
        return  # Already deactivated
    
    if last_control_time is None:
        return  # No activity yet, don't deactivate
    
    if monotonic() - last_control_time >= DEACTIVATION_TIMEOUT:
        activated = False
        last_control_time = None
        smooth.reset()  # Reset filter to avoid jumpy cursor on reactivation
        print("✗ Mouse control DEACTIVATED (no activity for 2 seconds)")
        play_deactivation_sound()

def on_frame(frame, hands, bag):
    """Called every frame to check for activation/deactivation."""
    check_activation(hands)
    check_deactivation()

# Cursor tuning
CURSOR_GAIN_X = 1.8  # >1 increases how far horizontal motions move the cursor
CURSOR_GAIN_Y = 1.8  # >1 increases how far vertical motions move the cursor


def _apply_cursor_gain(value, gain):
    """Expand motions around the center while keeping them within [0, 1]."""
    scaled = 0.5 + (value - 0.5) * gain
    return max(0.0, min(1.0, scaled))


def move(event):
    global cursor_initialized, cam_width, cam_height, last_control_time
    if not activated:
        return  # Mouse control not yet activated
    if not cam_width or not cam_height:
        return
    # Update last activity time to prevent deactivation
    last_control_time = monotonic()
    # Use index finger tip for ONE and TWO poses, wrist (point 0) for other poses
    if event.pose in ['ONE', 'TWO']:
        x, y = event.hand.landmarks[8,:2]  # Index finger tip
    else:
        x, y = event.hand.landmarks[0,:2]  # Wrist (point 0)
    x /= cam_width
    x = 1 - x
    y /= cam_height
    # Apply gain to amplify relative motions
    x = _apply_cursor_gain(x, CURSOR_GAIN_X)
    y = _apply_cursor_gain(y, CURSOR_GAIN_Y)
    # Map the normalized coordinates across the primary monitor bounds
    # Use primary monitor dimensions (0,0 is top-left corner of primary monitor)
    mx = max(0, min(monitor.width - 1, monitor.width * x))
    my = max(0, min(monitor.height - 1, monitor.height * y))
    mx, my = smooth.update((mx, my))

    # Position is relative to primary monitor (0,0 is top-left corner)
    set_cursor_absolute(mx, my)
    cursor_initialized = True

def press_release(event):
    global left_down, last_control_time
    if not activated:
        return  # Mouse control not yet activated
    # Update last activity time to prevent deactivation
    last_control_time = monotonic()
    if event.trigger == "enter": 
        left_down = True
        send_left_down()
    elif event.trigger == "leave":
        left_down = False
        send_left_up()

def click(event):
    send_left_down()
    send_left_up()

def parse_args():
    parser = argparse.ArgumentParser(description="Hand-tracking mouse controller demo.")
    parser.add_argument(
        "--no-renderer",
        action="store_true",
        help="Disable on-screen visualization windows."
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Skip printing the README banner (useful when running as an executable)."
    )
    return parser.parse_args()


def run_controller(args):
    global controller, cam_width, cam_height
    config = {
        'renderer': {'enable': not args.no_renderer},
        'on_frame': 'on_frame',  # Callback for activation pose detection
        'pose_actions': [
            {'name': 'MOVE', 'pose': ['ONE', 'TWO', 'FIVE', 'FIST', 'OK'], 'callback': 'move', "trigger": "continuous", "first_trigger_delay": 0.05,},
            {'name': 'CLICK', 'pose': ['TWO', 'FIST', 'OK'], 'callback': 'press_release', "trigger": "enter_leave", "first_trigger_delay": 0.05, "next_trigger_delay": 0.2, "max_missing_frames": 30},
        ],
        'tracker': {
            'args': {
                'resolution': '12mp',
                'xyz': False,
                'solo': False  # Enable two-hand tracking for activation pose
            }
        }
    }

    controller = HandController(config)
    cam_width = controller.tracker.img_w
    cam_height = controller.tracker.img_h
    print(f"Tracking camera frame size: {cam_width} x {cam_height}")
    controller.loop()


def main():
    args = parse_args()
    if not args.no_readme:
        print(README)
    run_controller(args)


if __name__ == "__main__":
    main()