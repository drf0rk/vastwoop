import os
import tempfile
import subprocess
import numpy as np
from pathlib import Path

# Improved FFMPEG writer with better error handling for Gradio environments
class ImprovedFFmpegWriter:
    def __init__(self, filename, fps=30, quality=8, output_params=None):
        """
        Initialize FFMPEG writer with improved path handling for Gradio
        """
        # Ensure we have an absolute path and the directory exists
        self.filename = os.path.abspath(filename)
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        self.fps = fps
        self.quality = quality
        self.output_params = output_params or []
        self.proc = None
        self.dimensions = None
        
        # Temporary file for debugging
        self.log_file = tempfile.NamedTemporaryFile(delete=False, suffix='.log')
        print(f"FFMPEG log file: {self.log_file.name}")
        
    def _start_ffmpeg_process(self, frame_size):
        """Start the FFMPEG subprocess with proper error handling"""
        self.dimensions = frame_size
        
        # Build command with explicit parameters
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{frame_size[1]}x{frame_size[0]}',  # Width x height
            '-pix_fmt', 'rgb24',
            '-r', f'{self.fps}',
            '-i', '-',  # Input from pipe
            '-vcodec', 'libx264',
            '-preset', 'medium',
            '-crf', f'{self.quality}',
        ]
        
        # Add any additional output parameters
        command.extend(self.output_params)
        
        # Add output filename
        command.append(self.filename)
        
        print(f"Starting FFMPEG with command: {' '.join(command)}")
        
        try:
            # Start process with pipe for stdin and capture stderr for logging
            self.proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=self.log_file,
                bufsize=10**8  # Larger buffer
            )
        except Exception as e:
            print(f"Failed to start FFMPEG process: {e}")
            raise
    
    def write_frame(self, frame):
        """Write a frame to the video file with improved error handling"""
        # Initialize on first frame
        if self.proc is None:
            self._start_ffmpeg_process(frame.shape)
        
        # Check if dimensions match
        if frame.shape[:2] != self.dimensions:
            raise ValueError(f"Frame size ({frame.shape[1]}x{frame.shape[0]}) doesn't match writer dimensions ({self.dimensions[1]}x{self.dimensions[0]})")
        
        # Ensure frame is correctly formatted
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        # Try to write with retry logic
        try:
            self.proc.stdin.write(frame.tobytes())
            self.proc.stdin.flush()  # Force the buffer to be written
        except BrokenPipeError:
            print("Broken pipe encountered, restarting FFMPEG process...")
            # Close and restart
            if self.proc:
                self._close()
            self._start_ffmpeg_process(frame.shape)
            try:
                # Try again with the new process
                self.proc.stdin.write(frame.tobytes())
                self.proc.stdin.flush()
            except Exception as e:
                print(f"Failed to write frame after restart: {e}")
                # Log current state
                self._debug_log()
                raise
    
    def _debug_log(self):
        """Print debug information to help diagnose issues"""
        print("=== DEBUG INFORMATION ===")
        print(f"Output path: {self.filename}")
        print(f"Directory exists: {os.path.exists(os.path.dirname(self.filename))}")
        print(f"Directory writable: {os.access(os.path.dirname(self.filename), os.W_OK)}")
        print(f"Current working directory: {os.getcwd()}")
        print("========================")
    
    def _close(self):
        """Safely close the FFMPEG process"""
        if self.proc:
            try:
                self.proc.stdin.close()
            except:
                pass
            
            try:
                # Wait for process to finish
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("FFMPEG process didn't terminate, killing it")
                self.proc.kill()
            
            self.proc = None
    
    def close(self):
        """Close the writer properly"""
        self._close()
        print(f"FFMPEG process finished. Check log at {self.log_file.name}")
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def __del__(self):
        self.close()
