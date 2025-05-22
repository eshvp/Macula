import ctypes
import platform
import sys


def get_primary_display_resolution():
    """
    Get the primary display resolution for the current system.
    Returns a tuple of (width, height) in pixels.
    """
    system = platform.system()

    if system == 'Windows':
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    elif system == 'Linux':
        try:
            import subprocess
            output = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4',
                                      shell=True, stdout=subprocess.PIPE).communicate()[0]
            resolution = output.decode("utf-8").strip().split('x')
            if len(resolution) == 2:
                return int(resolution[0]), int(resolution[1])
        except:
            pass

    elif system == 'Darwin':  # macOS
        try:
            import AppKit
            screen = AppKit.NSScreen.mainScreen()
            frame = screen.frame()
            return int(frame.size.width), int(frame.size.height)
        except:
            pass

    # Fallback to a reasonable default if detection fails
    return 1920, 1080


def get_optimal_camera_resolution(display_width, display_height):
    """
    Determine optimal camera resolution based on display size.

    Returns a tuple of (width, height) for camera settings.
    """
    # Standard camera resolutions (width, height)
    camera_resolutions = [
        (640, 480),  # VGA
        (1280, 720),  # 720p
        (1920, 1080),  # 1080p
        (3840, 2160)  # 4K
    ]

    # Find best resolution based on display size
    # Typically want camera resolution close to but not exceeding display resolution
    for res in sorted(camera_resolutions, reverse=True):
        # If camera resolution is at most 80% of display resolution, it's good
        if res[0] <= display_width * 0.8 and res[1] <= display_height * 0.8:
            return res

    # Fallback to 720p if nothing else works
    return 1280, 720


def get_recommended_window_size(display_width, display_height):
    """
    Calculate a reasonable window size based on display dimensions.
    Returns (width, height) in pixels.
    """
    # Use 70% of available display as maximum size
    max_width = int(display_width * 0.7)
    max_height = int(display_height * 0.7)

    # Maintain 16:9 aspect ratio if possible
    if max_width / max_height > 16 / 9:
        # Display is wider than 16:9
        width = int(max_height * 16 / 9)
        height = max_height
    else:
        # Display is taller than 16:9
        width = max_width
        height = int(max_width * 9 / 16)

    return width, height