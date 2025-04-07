# Replace the content of /vastwoop/roop/ffmpeg_writer.py with this code

import os
import subprocess
import numpy as np
import tempfile
from pathlib import Path

class FFmpegWriter:
    """
    Improved FFmpegWriter with better error handling and path management for Gradio environments
    """
    def __init__(self, filename, fps=30, codec='libx264', pixfmt='rgb24', output_params=None):
        """
        Initialize FFmpeg writer with robust path handling
        """
        # Ensure absolute path and create directory if needed
        self.filename = os.path.abspath(filename)
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        self.fps = fps
        self.codec = codec
        self.pixfmt = pixfmt
        self.output_params = output_params or []
        self.proc = None
        self.dimensions = None
        
        # Create a log file for debugging
        self.log_file = open(os.path.join(tempfile.gettempdir(), 'ffmpeg_roop.log'), 'a')
        self.log_file.write(f"\n\n--- New FFmpeg session for {self.filename} ---\n")
        self.log_file.flush()
        
    def _start_ffmpeg_process(self, frame_size):
        """Start the FFmpeg process with reliable parameters"""
        self.dimensions = frame_size
        height, width = frame_size[0], frame_size[1]
        
        # Build FFmpeg command with explicit parameters
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', self.pixfmt,
            '-r', f'{self.fps}',
            '-i', '-',  # Input from pipe
            '-vcodec', self.codec,
        ]
        
        # Add custom output parameters
        command.extend(self.output_params)
        
        # Add output filename
        command.append(self.filename)
        
        self.log_file.write(f"FFmpeg command: {' '.join(command)}\n")
        self.log_file.flush()
        
        try:
            # Start process with larger buffer
            self.proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            self.log_file.write("FFmpeg process started successfully\n")
            self.log_file.flush()
            
        except Exception as e:
            self.log_file.write(f"Failed to start FFmpeg: {str(e)}\n")
            self.log_file.flush()
            raise IOError(f"Error starting FFmpeg process: {str(e)}")
    
    def write_frame(self, frame):
        """Write a frame with error handling and recovery"""
        if self.proc is None:
            self._start_ffmpeg_process(frame.shape)
        
        # Ensure frame is in the correct format
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        try:
            # Write frame to FFmpeg
            self.proc.stdin.write(frame.tobytes())
            self.proc.stdin.flush()  # Force buffer flush
            
        except BrokenPipeError:
            self.log_file.write("BrokenPipeError encountered, attempting to restart FFmpeg\n")
            self.log_file.flush()
            
            # Clean up old process
            self._close_proc()
            
            # Restart with same dimensions
            self._start_ffmpeg_process(frame.shape)
            
            try:
                # Try writing the frame again
                self.proc.stdin.write(frame.tobytes())
                self.proc.stdin.flush()
                self.log_file.write("Successfully wrote frame after restarting FFmpeg\n")
                self.log_file.flush()
                
            except Exception as e:
                self.log_file.write(f"Failed to write frame after restart: {str(e)}\n")
                self.log_file.flush()
                error = f"FFMPEG encountered the following error while writing file: {str(e)}"
                raise IOError(error)
                
        except Exception as e:
            self.log_file.write(f"Error writing frame: {str(e)}\n")
            self.log_file.flush()
            error = f"FFMPEG encountered the following error while writing file: {str(e)}"
            raise IOError(error)
    
    def _close_proc(self):
        """Safely close the FFmpeg process"""
        if self.proc:
            try:
                self.proc.stdin.close()
            except:
                pass
                
            try:
                # Wait for process to finish
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.log_file.write("FFmpeg process didn't terminate, killing it\n")
                self.log_file.flush()
                self.proc.kill()
                
            self.proc = None
    
    def close(self):
        """Close the writer and finalize the video file"""
        self._close_proc()
        self.log_file.write(f"FFmpeg writer closed for {self.filename}\n")
        self.log_file.flush()
        self.log_file.close()
        
    def __del__(self):
        """Destructor to ensure resources are released"""
        self.close()
