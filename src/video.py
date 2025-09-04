import cv2
import subprocess
import numpy as np

class FFmpegVideoWriter:
    def __init__(self, output_path: str, fps: float, frame_size: tuple[int, int]):
        self.output_path = output_path
        self.width, self.height = frame_size
        self.fps = fps

        # Launch ffmpeg process
        self.process = subprocess.Popen([
            'ffmpeg',
            '-y',  # overwrite
            '-loglevel', 'error',  # only show error messages
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',  # read from stdin
            '-an',  # no audio
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            self.output_path
        ], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        """Write a single frame"""
        self.process.stdin.write(frame.tobytes())

    def release(self):
        self.process.stdin.close()
        self.process.wait()

class VideoReader:
    """Video reader using cv2"""
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (self.width, self.height)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def read(self):
        return self.cap.read()
    
    def release(self):
        self.cap.release()
        
