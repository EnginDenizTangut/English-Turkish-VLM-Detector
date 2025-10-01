#!/usr/bin/env python3
"""
Modern GUI for English-Turkish VLM Detector
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import os
import time
from main import VLMDetector, VideoProcessor

class VLMDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("English-Turkish VLM Detector")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize detector
        self.detector = VLMDetector(mode='detection')
        self.video_processor = VideoProcessor(self.detector)
        self.current_mode = 'detection'
        
        # Variables
        self.current_image_path = None
        self.current_video_path = None
        self.original_image = None
        self.result_image = None
        self.is_video_mode = False
        
        # Video player variables
        self.video_cap = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.video_thread = None
        self.detection_enabled = False
        self.detection_frame_skip = 3  # Process every 3rd frame for smooth flow
        self.frame_counter = 0
        
        # Configure style
        self.setup_styles()
        
        # Create GUI
        self.create_widgets()
        
    def setup_styles(self):
        """Configure modern styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure button style
        style.configure('Modern.TButton',
                       font=('Arial', 10, 'bold'),
                       padding=(10, 5),
                       background='#4CAF50',
                       foreground='white')
        
        style.map('Modern.TButton',
                 background=[('active', '#45a049')])
        
        # Configure entry style
        style.configure('Modern.TEntry',
                       font=('Arial', 11),
                       padding=(10, 8),
                       fieldbackground='white')
        
        # Configure label style
        style.configure('Title.TLabel',
                       font=('Arial', 16, 'bold'),
                       background='#f0f0f0',
                       foreground='#333333')
        
        style.configure('Subtitle.TLabel',
                       font=('Arial', 12, 'bold'),
                       background='#f0f0f0',
                       foreground='#666666')
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎯 English-Turkish VLM Detector", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Image panels frame
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Left panel - Original image
        left_panel = ttk.LabelFrame(image_frame, text="📸 Original Image", padding="10")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(0, weight=1)
        
        self.original_canvas = tk.Canvas(left_panel, bg='white', relief=tk.SUNKEN, bd=2)
        self.original_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Right panel - Result image
        right_panel = ttk.LabelFrame(image_frame, text="🎨 Detection Result", padding="10")
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        
        self.result_canvas = tk.Canvas(right_panel, bg='white', relief=tk.SUNKEN, bd=2)
        self.result_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # Mode selection
        mode_frame = ttk.Frame(control_frame)
        mode_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(mode_frame, text="🎯 Mode:", style='Subtitle.TLabel').grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.mode_var = tk.StringVar(value='detection')
        self.mode_var.trace('w', self.on_mode_change)
        
        detection_radio = ttk.Radiobutton(mode_frame, text="🔍 Detection", variable=self.mode_var, value='detection')
        detection_radio.grid(row=0, column=1, padx=(0, 20))
        
        segmentation_radio = ttk.Radiobutton(mode_frame, text="🎨 Segmentation", variable=self.mode_var, value='segmentation')
        segmentation_radio.grid(row=0, column=2, padx=(0, 20))
        
        # Media type selection
        media_frame = ttk.Frame(control_frame)
        media_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(media_frame, text="📱 Media Type:", style='Subtitle.TLabel').grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.media_var = tk.StringVar(value='image')
        self.media_var.trace('w', self.on_media_change)
        
        image_radio = ttk.Radiobutton(media_frame, text="📸 Image", variable=self.media_var, value='image')
        image_radio.grid(row=0, column=1, padx=(0, 20))
        
        video_radio = ttk.Radiobutton(media_frame, text="🎥 Video", variable=self.media_var, value='video')
        video_radio.grid(row=0, column=2, padx=(0, 20))
        
        webcam_radio = ttk.Radiobutton(media_frame, text="📹 Webcam", variable=self.media_var, value='webcam')
        webcam_radio.grid(row=0, column=3, padx=(0, 20))
        
        # File selection
        file_frame = ttk.Frame(control_frame)
        file_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        self.file_label = ttk.Label(file_frame, text="📁 Image File:", style='Subtitle.TLabel')
        self.file_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, style='Modern.TEntry', state='readonly')
        self.file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file, style='Modern.TButton')
        self.browse_button.grid(row=0, column=2)
        
        # Video processing options (initially hidden)
        self.video_options_frame = ttk.LabelFrame(control_frame, text="🎥 Video Options", padding="10")
        self.video_options_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.video_options_frame.grid_remove()  # Hide initially
        
        # Frame skip option
        ttk.Label(self.video_options_frame, text="Frame Skip:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.frame_skip_var = tk.StringVar(value="1")
        frame_skip_entry = ttk.Entry(self.video_options_frame, textvariable=self.frame_skip_var, width=10)
        frame_skip_entry.grid(row=0, column=1, padx=(0, 20))
        ttk.Label(self.video_options_frame, text="(1 = all frames, 5 = every 5th frame)").grid(row=0, column=2, sticky=tk.W)
        
        # Max frames option
        ttk.Label(self.video_options_frame, text="Max Frames:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.max_frames_var = tk.StringVar(value="")
        max_frames_entry = ttk.Entry(self.video_options_frame, textvariable=self.max_frames_var, width=10)
        max_frames_entry.grid(row=1, column=1, padx=(0, 20))
        ttk.Label(self.video_options_frame, text="(empty = no limit)").grid(row=1, column=2, sticky=tk.W)
        
        # Webcam options (initially hidden)
        self.webcam_options_frame = ttk.LabelFrame(control_frame, text="📹 Webcam Options", padding="10")
        self.webcam_options_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.webcam_options_frame.grid_remove()  # Hide initially
        
        # Duration option
        ttk.Label(self.webcam_options_frame, text="Duration (seconds):").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.duration_var = tk.StringVar(value="30")
        duration_entry = ttk.Entry(self.webcam_options_frame, textvariable=self.duration_var, width=10)
        duration_entry.grid(row=0, column=1, padx=(0, 20))
        ttk.Label(self.webcam_options_frame, text="(0 = unlimited)").grid(row=0, column=2, sticky=tk.W)
        
        # Video player controls (initially hidden)
        self.video_player_frame = ttk.LabelFrame(control_frame, text="🎬 Video Player", padding="10")
        self.video_player_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.video_player_frame.grid_remove()  # Hide initially
        
        # Video controls
        controls_frame = ttk.Frame(self.video_player_frame)
        controls_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.play_button = ttk.Button(controls_frame, text="▶️ Play", command=self.toggle_play_pause, style='Modern.TButton')
        self.play_button.grid(row=0, column=0, padx=(0, 5))
        
        self.stop_button = ttk.Button(controls_frame, text="⏹️ Stop", command=self.stop_video, style='Modern.TButton')
        self.stop_button.grid(row=0, column=1, padx=(0, 5))
        
        # Detection toggle
        self.detection_var = tk.BooleanVar()
        self.detection_check = ttk.Checkbutton(controls_frame, text="🔍 Live Detection", 
                                             variable=self.detection_var, command=self.toggle_detection)
        self.detection_check.grid(row=0, column=2, padx=(20, 0))
        
        # Detection frame skip
        ttk.Label(controls_frame, text="Detection Skip:").grid(row=0, column=3, padx=(20, 5))
        self.detection_skip_var = tk.StringVar(value="3")
        detection_skip_combo = ttk.Combobox(controls_frame, textvariable=self.detection_skip_var, 
                                          values=["1", "2", "3", "5", "10", "15", "30"], width=5)
        detection_skip_combo.grid(row=0, column=4, padx=(0, 5))
        detection_skip_combo.bind('<<ComboboxSelected>>', self.on_detection_skip_change)
        
        # Progress bar for video
        self.video_progress = ttk.Scale(self.video_player_frame, from_=0, to=100, 
                                      orient=tk.HORIZONTAL, command=self.on_progress_change)
        self.video_progress.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Time labels
        self.time_label = ttk.Label(self.video_player_frame, text="00:00 / 00:00")
        self.time_label.grid(row=1, column=2, padx=(10, 0))
        
        # Speed control
        ttk.Label(self.video_player_frame, text="Speed:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10))
        self.speed_var = tk.StringVar(value="1.0")
        speed_combo = ttk.Combobox(self.video_player_frame, textvariable=self.speed_var, 
                                 values=["0.5", "0.75", "1.0", "1.25", "1.5", "2.0"], width=8)
        speed_combo.grid(row=2, column=1, padx=(0, 20))
        
        # Frame info
        self.frame_info_label = ttk.Label(self.video_player_frame, text="Frame: 0 / 0")
        self.frame_info_label.grid(row=2, column=2, padx=(10, 0))
        
        # Prompt input
        prompt_frame = ttk.Frame(control_frame)
        prompt_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        prompt_frame.columnconfigure(0, weight=1)
        
        ttk.Label(prompt_frame, text="💬 Detection Prompt:", style='Subtitle.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.prompt_var = tk.StringVar()
        self.prompt_entry = ttk.Entry(prompt_frame, textvariable=self.prompt_var, style='Modern.TEntry')
        self.prompt_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.prompt_entry.bind('<Return>', lambda e: self.detect_objects())
        
        # Example prompts
        example_frame = ttk.Frame(prompt_frame)
        example_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(example_frame, text="💡 Examples:", font=('Arial', 9), foreground='#888888').grid(row=0, column=0, sticky=tk.W)
        
        examples = [
            "mavi arabaları göster", "kırmızı kedileri bul", "yeşil sandalyeleri tespit et",
            "sarı meyveleri göster", "mor çiçekleri bul", "arabaları göster"
        ]
        
        for i, example in enumerate(examples):
            btn = ttk.Button(example_frame, text=example, 
                           command=lambda ex=example: self.set_prompt(ex),
                           style='TButton')
            btn.grid(row=1, column=i, padx=(0, 5), pady=2)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.detect_button = ttk.Button(button_frame, text="🔍 Detect Objects", 
                                      command=self.detect_objects, style='Modern.TButton')
        self.detect_button.grid(row=0, column=0, padx=(0, 10))
        
        self.clear_button = ttk.Button(button_frame, text="🗑️ Clear", 
                                     command=self.clear_all, style='Modern.TButton')
        self.clear_button.grid(row=0, column=1, padx=(0, 10))
        
        self.save_button = ttk.Button(button_frame, text="💾 Save Result", 
                                    command=self.save_result, style='Modern.TButton')
        self.save_button.grid(row=0, column=2)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select an image and enter a prompt")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              font=('Arial', 9), foreground='#666666')
        status_bar.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def on_mode_change(self, *args):
        """Handle mode change"""
        new_mode = self.mode_var.get()
        if new_mode != self.current_mode:
            self.current_mode = new_mode
            # Reinitialize detector with new mode
            self.detector = VLMDetector(mode=new_mode)
            
            # Update button text
            if new_mode == 'segmentation':
                self.detect_button.config(text="🎨 Segment Objects")
                self.status_var.set(f"Mode changed to Segmentation - Select an image and enter a prompt")
            else:
                self.detect_button.config(text="🔍 Detect Objects")
                self.status_var.set(f"Mode changed to Detection - Select an image and enter a prompt")
            
            # Clear current results
            self.result_canvas.delete("all")
            self.result_image = None
    
    def on_media_change(self, *args):
        """Handle media type change"""
        media_type = self.media_var.get()
        
        if media_type == 'image':
            self.is_video_mode = False
            self.file_label.config(text="📁 Image File:")
            self.browse_button.config(text="Browse")
            self.video_options_frame.grid_remove()
            self.webcam_options_frame.grid_remove()
            self.status_var.set("Ready - Select an image and enter a prompt")
            
        elif media_type == 'video':
            self.is_video_mode = True
            self.file_label.config(text="📁 Video File:")
            self.browse_button.config(text="Browse")
            self.video_options_frame.grid()
            self.webcam_options_frame.grid_remove()
            self.video_player_frame.grid()  # Show video player
            self.status_var.set("Ready - Select a video and enter a prompt")
            
        elif media_type == 'webcam':
            self.is_video_mode = True
            self.file_label.config(text="📹 Webcam:")
            self.browse_button.config(text="Start Live Webcam", command=self.start_live_webcam)
            self.video_options_frame.grid_remove()
            self.webcam_options_frame.grid_remove()  # Hide webcam options for live mode
            self.video_player_frame.grid()  # Show video player for live webcam
            self.status_var.set("Ready - Click Start Live Webcam and enter a prompt")
    
    def start_live_webcam(self):
        """Start live webcam with real-time detection"""
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a detection prompt!")
            return
        
        # Disable button and show progress
        self.browse_button.config(state='disabled')
        self.progress.start()
        self.status_var.set("Starting live webcam... Please wait")
        
        # Run live webcam in separate thread
        thread = threading.Thread(target=self.run_live_webcam, args=(prompt,))
        thread.daemon = True
        thread.start()
    
    def start_webcam(self):
        """Start webcam processing (old method)"""
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a detection prompt!")
            return
        
        duration = self.duration_var.get().strip()
        duration = int(duration) if duration.isdigit() else 30
        
        # Disable button and show progress
        self.browse_button.config(state='disabled')
        self.progress.start()
        self.status_var.set("Starting webcam... Please wait")
        
        # Run webcam processing in separate thread
        thread = threading.Thread(target=self.run_webcam, args=(prompt, duration))
        thread.daemon = True
        thread.start()
    
    def run_live_webcam(self, prompt):
        """Run live webcam with real-time detection"""
        try:
            # Open webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.root.after(0, self.update_status, "Webcam açılamadı!")
                return
            
            # Get webcam properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Live webcam başlatıldı: {width}x{height}, {fps} FPS")
            
            # Setup video player for live webcam
            self.video_cap = cap
            self.total_frames = 0  # Live stream, no total frames
            self.current_frame = 0
            self.fps = fps
            self.is_playing = True
            
            # Update UI
            self.root.after(0, self.update_status, "📹 Live webcam started - detection active!")
            self.root.after(0, self.play_button.config, {"text": "⏸️ Pause"})
            self.root.after(0, self.stop_button.config, {"command": self.stop_live_webcam})
            
            # Start live detection
            self.detection_enabled = True
            self.detection_var.set(True)
            
            frame_count = 0
            
            while self.is_playing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                self.current_frame = frame_count
                
                # Process every 3rd frame for performance
                if frame_count % 3 == 0:
                    frame = self.process_frame_for_detection(frame)
                
                # Update display
                self.root.after(0, self.update_video_display, frame)
                
                # Control frame rate
                time.sleep(1.0 / 30)  # 30 FPS
            
            cap.release()
            
        except Exception as e:
            self.root.after(0, self.update_status, f"Live webcam error: {str(e)}")
        
        finally:
            # Re-enable button and stop progress
            self.root.after(0, self.live_webcam_finished)
    
    def stop_live_webcam(self):
        """Stop live webcam"""
        self.is_playing = False
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.detection_enabled = False
        self.detection_var.set(False)
        self.update_status("Live webcam stopped")
    
    def live_webcam_finished(self):
        """Called when live webcam is finished"""
        self.browse_button.config(state='normal')
        self.progress.stop()
        self.play_button.config(text="▶️ Play")
        self.stop_button.config(command=self.stop_video)
    
    def run_webcam(self, prompt, duration):
        """Run webcam processing in background thread (old method)"""
        try:
            output_path = self.video_processor.process_webcam(prompt, duration=duration)
            
            if output_path:
                self.root.after(0, self.update_status, f"Webcam processing completed! Output: {output_path}")
            else:
                self.root.after(0, self.update_status, "Webcam processing failed!")
                
        except Exception as e:
            self.root.after(0, self.update_status, f"Error: {str(e)}")
        
        finally:
            # Re-enable button and stop progress
            self.root.after(0, self.webcam_finished)
    
    def webcam_finished(self):
        """Called when webcam processing is finished"""
        self.browse_button.config(state='normal')
        self.progress.stop()
    
    def browse_file(self):
        """Open file dialog to select image or video"""
        media_type = self.media_var.get()
        
        if media_type == 'image':
            file_types = [
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
            title = "Select Image File"
        else:  # video
            file_types = [
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv"),
                ("All files", "*.*")
            ]
            title = "Select Video File"
        
        filename = filedialog.askopenfilename(
            title=title,
            filetypes=file_types
        )
        
        if filename:
            if media_type == 'image':
                self.current_image_path = filename
                self.current_video_path = None
                self.load_original_image()
                self.status_var.set(f"Image loaded: {os.path.basename(filename)}")
            else:  # video
                self.current_video_path = filename
                self.current_image_path = None
                self.load_video_preview(filename)
                self.setup_video_player(filename)
                self.status_var.set(f"Video loaded: {os.path.basename(filename)}")
    
    def load_original_image(self):
        """Load and display original image"""
        if not self.current_image_path:
            return
        
        try:
            # Load image
            image = Image.open(self.current_image_path)
            self.original_image = image.copy()
            
            # Resize to fit canvas
            canvas_width = self.original_canvas.winfo_width()
            canvas_height = self.original_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas not yet sized, schedule for later
                self.root.after(100, self.load_original_image)
                return
            
            # Calculate resize dimensions
            img_width, img_height = image.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.original_photo = ImageTk.PhotoImage(image)
            
            # Clear canvas and display image
            self.original_canvas.delete("all")
            self.original_canvas.create_image(canvas_width//2, canvas_height//2, 
                                            image=self.original_photo, anchor=tk.CENTER)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def load_video_preview(self, video_path):
        """Load and display video preview (first frame)"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                messagebox.showerror("Error", "Failed to open video file!")
                return
            
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to read video frame!")
                cap.release()
                return
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self.original_image = image.copy()
            
            # Resize to fit canvas
            canvas_width = self.original_canvas.winfo_width()
            canvas_height = self.original_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas not yet sized, schedule for later
                self.root.after(100, lambda: self.load_video_preview(video_path))
                cap.release()
                return
            
            # Calculate resize dimensions
            img_width, img_height = image.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.original_photo = ImageTk.PhotoImage(image)
            
            # Clear canvas and display image
            self.original_canvas.delete("all")
            self.original_canvas.create_image(canvas_width//2, canvas_height//2, 
                                            image=self.original_photo, anchor=tk.CENTER)
            
            # Add video info overlay
            video_info = self.video_processor.get_video_info(video_path)
            if video_info:
                info_text = f"Video: {video_info['width']}x{video_info['height']}, {video_info['fps']:.1f} FPS, {video_info['duration']:.1f}s"
                self.original_canvas.create_text(10, 10, text=info_text, 
                                               fill='white', anchor='nw', font=('Arial', 10, 'bold'))
            
            cap.release()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video preview: {str(e)}")
    
    def setup_video_player(self, video_path):
        """Setup video player for the loaded video"""
        try:
            # Close existing video capture
            if self.video_cap:
                self.video_cap.release()
            
            # Open new video
            self.video_cap = cv2.VideoCapture(video_path)
            if not self.video_cap.isOpened():
                messagebox.showerror("Error", "Failed to open video file!")
                return
            
            # Get video properties
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30
            self.current_frame = 0
            
            # Update UI
            self.video_progress.config(to=self.total_frames)
            self.frame_info_label.config(text=f"Frame: 0 / {self.total_frames}")
            
            # Calculate duration
            duration = self.total_frames / self.fps
            self.time_label.config(text=f"00:00 / {int(duration//60):02d}:{int(duration%60):02d}")
            
            # Reset player state
            self.is_playing = False
            self.play_button.config(text="▶️ Play")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to setup video player: {str(e)}")
    
    def toggle_play_pause(self):
        """Toggle video play/pause"""
        if not self.video_cap:
            return
        
        if self.is_playing:
            self.is_playing = False
            self.play_button.config(text="▶️ Play")
            if self.video_thread:
                self.video_thread.join(timeout=0.1)
        else:
            self.is_playing = True
            self.play_button.config(text="⏸️ Pause")
            self.video_thread = threading.Thread(target=self.play_video)
            self.video_thread.daemon = True
            self.video_thread.start()
    
    def stop_video(self):
        """Stop video playback"""
        self.is_playing = False
        self.play_button.config(text="▶️ Play")
        self.current_frame = 0
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.update_video_display()
    
    def play_video(self):
        """Play video in separate thread"""
        while self.is_playing and self.video_cap:
            ret, frame = self.video_cap.read()
            if not ret:
                # End of video
                self.is_playing = False
                self.root.after(0, lambda: self.play_button.config(text="▶️ Play"))
                break
            
            self.current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.frame_counter += 1
            
            # Process frame if detection is enabled (every frame for real-time detection)
            if self.detection_enabled and self.frame_counter % self.detection_frame_skip == 0:
                frame = self.process_frame_for_detection(frame)
            
            # Update display
            self.root.after(0, self.update_video_display, frame)
            
            # Control playback speed - keep video flowing smoothly
            speed = float(self.speed_var.get())
            delay = int(1000 / (self.fps * speed))  # Normal speed, no slowing down
            time.sleep(delay / 1000.0)
    
    def process_frame_for_detection(self, frame):
        """Process frame for live detection - ultra fast real-time"""
        try:
            # Get current prompt
            prompt = self.prompt_var.get().strip()
            if not prompt:
                return frame
            
            # Direct YOLO detection on frame (ultra fast)
            results = self.detector.detect_objects_direct(frame)
            
            if results and len(results.boxes) > 0:
                # Fast class filtering without LLM (much faster)
                boxes, confidences, classes = self.fast_class_filter(results, prompt)
                
                if boxes:
                    # Draw detections directly on frame
                    frame = self.draw_detections_on_frame(frame, boxes, confidences, classes, prompt)
                    print(f"Frame {self.current_frame}: Found {len(boxes)} objects")
            
            return frame
            
        except Exception as e:
            print(f"Detection error: {e}")
            return frame
    
    def fast_class_filter(self, results, prompt):
        """Fast class filtering without LLM for real-time detection"""
        try:
            # Simple keyword matching for speed
            prompt_lower = prompt.lower()
            
            # Common Turkish to English mappings for speed
            class_mappings = {
                'araba': 'car', 'otomobil': 'car', 'taşıt': 'car', 'vasıta': 'car',
                'kamyon': 'truck', 'tır': 'truck', 'yük aracı': 'truck',
                'otobüs': 'bus', 'şehir otobüsü': 'bus',
                'motosiklet': 'motorcycle', 'moto': 'motorcycle', 'motor': 'motorcycle',
                'bisiklet': 'bicycle', 'velespit': 'bicycle', 'pedal': 'bicycle',
                'insan': 'person', 'kişi': 'person', 'adam': 'person', 'kadın': 'person',
                'kedi': 'cat', 'pisi': 'cat', 'miyav': 'cat',
                'köpek': 'dog', 'it': 'dog', 'hav hav': 'dog',
                'kuş': 'bird', 'kanatlı': 'bird',
                'sandalye': 'chair', 'oturak': 'chair', 'koltuk': 'chair',
                'masa': 'dining table', 'yemek masası': 'dining table',
                'televizyon': 'tv', 'tv': 'tv', 'ekran': 'tv',
                'laptop': 'laptop', 'dizüstü': 'laptop', 'bilgisayar': 'laptop',
                'telefon': 'cell phone', 'cep telefonu': 'cell phone', 'mobil': 'cell phone'
            }
            
            # Find matching classes
            target_classes = []
            for turkish, english in class_mappings.items():
                if turkish in prompt_lower:
                    target_classes.append(english)
            
            # If no specific class found, detect all objects
            if not target_classes:
                target_classes = list(self.detector.class_names.values())
            
            # Filter results
            filtered_boxes = []
            filtered_confidences = []
            filtered_classes = []
            
            for i, box in enumerate(results.boxes):
                class_id = int(box.cls[0])
                class_name = self.detector.class_names[class_id]
                confidence = float(box.conf[0])
                
                if class_name.lower() in [cls.lower() for cls in target_classes]:
                    filtered_boxes.append(box.xyxy[0].cpu().numpy())
                    filtered_confidences.append(confidence)
                    filtered_classes.append(class_name)
            
            return filtered_boxes, filtered_confidences, filtered_classes
            
        except Exception as e:
            print(f"Fast filter error: {e}")
            return [], [], []
    
    def draw_detections_on_frame(self, frame, boxes, confidences, classes, prompt):
        """Draw bounding boxes directly on frame"""
        try:
            # Get color from prompt
            color = self.detector.extract_color_from_query(prompt)
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{cls}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return frame
            
        except Exception as e:
            print(f"Draw detections error: {e}")
            return frame
    
    def draw_segmentation_on_frame(self, frame, masks, confidences, classes, prompt):
        """Draw segmentation masks directly on frame"""
        try:
            # Get color from prompt
            color = self.detector.extract_color_from_query(prompt)
            
            # Create overlay
            overlay = frame.copy()
            
            for i, (mask, conf, cls) in enumerate(zip(masks, confidences, classes)):
                # Resize mask to frame size
                if len(mask.shape) == 3:
                    mask = mask.squeeze()
                
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_uint8 = (mask_resized * 255).astype(np.uint8)
                
                # Create colored mask
                colored_mask = np.zeros_like(frame)
                colored_mask[mask_uint8 > 0] = color
                
                # Blend with overlay
                overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
                
                # Draw bounding box
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw label
                    label = f"{cls}: {conf:.2f}"
                    cv2.putText(overlay, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return overlay
            
        except Exception as e:
            print(f"Draw segmentation error: {e}")
            return frame
    
    def update_video_display(self, frame=None):
        """Update video display"""
        if frame is None:
            if not self.video_cap:
                return
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.video_cap.read()
            if not ret:
                return
        
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Resize to fit canvas
            canvas_width = self.original_canvas.winfo_width()
            canvas_height = self.original_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            img_width, img_height = image.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.original_photo = ImageTk.PhotoImage(image)
            
            # Clear canvas and display image
            self.original_canvas.delete("all")
            self.original_canvas.create_image(canvas_width//2, canvas_height//2, 
                                            image=self.original_photo, anchor=tk.CENTER)
            
            # Update progress and info
            progress_value = (self.current_frame / self.total_frames) * 100 if self.total_frames > 0 else 0
            self.video_progress.set(progress_value)
            
            # Update frame info
            self.frame_info_label.config(text=f"Frame: {self.current_frame} / {self.total_frames}")
            
            # Update time
            current_time = self.current_frame / self.fps
            total_time = self.total_frames / self.fps
            self.time_label.config(text=f"{int(current_time//60):02d}:{int(current_time%60):02d} / {int(total_time//60):02d}:{int(total_time%60):02d}")
            
        except Exception as e:
            print(f"Display update error: {e}")
    
    def on_progress_change(self, value):
        """Handle progress bar change (seek)"""
        if not self.video_cap or self.is_playing:
            return
        
        frame_number = int(float(value) * self.total_frames / 100)
        self.current_frame = frame_number
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.update_video_display()
    
    def toggle_detection(self):
        """Toggle live detection"""
        self.detection_enabled = self.detection_var.get()
        if self.detection_enabled:
            self.status_var.set("🚀 Ultra-fast detection enabled - video flows smoothly while catching objects!")
        else:
            self.status_var.set("Live detection disabled")
    
    def on_detection_skip_change(self, event=None):
        """Handle detection frame skip change"""
        try:
            self.detection_frame_skip = int(self.detection_skip_var.get())
            self.status_var.set(f"Detection frame skip set to {self.detection_frame_skip}")
        except ValueError:
            self.detection_frame_skip = 5
    
    def set_prompt(self, prompt):
        """Set prompt from example button"""
        self.prompt_var.set(prompt)
        self.prompt_entry.focus()
    
    def detect_objects(self):
        """Detect objects in the image or video"""
        media_type = self.media_var.get()
        
        if media_type == 'image':
            if not self.current_image_path:
                messagebox.showwarning("Warning", "Please select an image first!")
                return
        elif media_type == 'video':
            if not self.current_video_path:
                messagebox.showwarning("Warning", "Please select a video first!")
                return
        else:  # webcam
            messagebox.showwarning("Warning", "Please use 'Start Webcam' button for webcam processing!")
            return
        
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a detection prompt!")
            return
        
        # Disable button and show progress
        self.detect_button.config(state='disabled')
        self.progress.start()
        self.status_var.set("Processing... Please wait")
        
        # Run detection in separate thread
        if media_type == 'image':
            thread = threading.Thread(target=self.run_detection, args=(prompt,))
        else:  # video
            thread = threading.Thread(target=self.run_video_detection, args=(prompt,))
        thread.daemon = True
        thread.start()
    
    def run_detection(self, prompt):
        """Run detection in background thread"""
        try:
            # Run detection/segmentation
            results = self.detector.process_image(self.current_image_path, prompt)
            
            # Determine result file based on mode
            if self.current_mode == 'segmentation':
                result_path = "output_segmentation.jpg"
                mode_name = "Segmentation"
            else:
                result_path = "output_detection.jpg"
                mode_name = "Detection"
            
            if os.path.exists(result_path):
                result_image = Image.open(result_path)
                self.result_image = result_image.copy()
                
                # Update GUI in main thread
                self.root.after(0, self.display_result, result_image)
                
                # Update status
                if isinstance(results, tuple) and len(results) >= 3:
                    classes = results[2] if len(results) > 2 else []
                    self.root.after(0, self.update_status, 
                                  f"{mode_name} completed! Found {len(classes)} objects: {', '.join(classes)}")
                else:
                    self.root.after(0, self.update_status, f"{mode_name} completed!")
            else:
                self.root.after(0, self.update_status, f"No objects detected in {mode_name.lower()} mode")
                
        except Exception as e:
            self.root.after(0, self.update_status, f"Error: {str(e)}")
        
        finally:
            # Re-enable button and stop progress
            self.root.after(0, self.detection_finished)
    
    def run_video_detection(self, prompt):
        """Run video detection in background thread"""
        try:
            # Get video processing options
            frame_skip = self.frame_skip_var.get().strip()
            frame_skip = int(frame_skip) if frame_skip.isdigit() else 1
            
            max_frames = self.max_frames_var.get().strip()
            max_frames = int(max_frames) if max_frames.isdigit() else None
            
            # Process video
            results = self.video_processor.process_video_frames(
                self.current_video_path, prompt, 
                frame_skip=frame_skip, max_frames=max_frames
            )
            
            if results:
                # Update GUI in main thread
                self.root.after(0, self.update_status, 
                              f"Video processing completed! Processed {results['processed_frames']} frames. Output: {results['output_video']}")
            else:
                self.root.after(0, self.update_status, "Video processing failed!")
                
        except Exception as e:
            self.root.after(0, self.update_status, f"Error: {str(e)}")
        
        finally:
            # Re-enable button and stop progress
            self.root.after(0, self.detection_finished)
    
    def display_result(self, result_image):
        """Display result image"""
        try:
            # Get canvas dimensions
            canvas_width = self.result_canvas.winfo_width()
            canvas_height = self.result_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                self.root.after(100, lambda: self.display_result(result_image))
                return
            
            # Calculate resize dimensions
            img_width, img_height = result_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            result_image = result_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.result_photo = ImageTk.PhotoImage(result_image)
            
            # Clear canvas and display image
            self.result_canvas.delete("all")
            self.result_canvas.create_image(canvas_width//2, canvas_height//2, 
                                          image=self.result_photo, anchor=tk.CENTER)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display result: {str(e)}")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
    
    def detection_finished(self):
        """Called when detection is finished"""
        self.detect_button.config(state='normal')
        self.progress.stop()
    
    def clear_all(self):
        """Clear all inputs and images"""
        # Stop video if playing
        if self.is_playing:
            self.stop_video()
        
        # Stop live webcam if running
        if self.detection_enabled and self.video_cap:
            self.stop_live_webcam()
        
        # Close video capture
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        
        self.current_image_path = None
        self.current_video_path = None
        self.file_path_var.set("")
        self.prompt_var.set("")
        
        self.original_canvas.delete("all")
        self.result_canvas.delete("all")
        
        self.original_image = None
        self.result_image = None
        
        # Reset video player state
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.detection_enabled = False
        self.detection_var.set(False)
        self.frame_counter = 0
        self.detection_skip_var.set("3")
        
        # Reset media type to image
        self.media_var.set("image")
        self.on_media_change()
        
        self.status_var.set("Ready - Select an image and enter a prompt")
    
    def save_result(self):
        """Save result image"""
        if not self.result_image:
            messagebox.showwarning("Warning", "No result image to save!")
            return
        
        file_types = [
            ("JPEG files", "*.jpg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Save Result Image",
            defaultextension=".jpg",
            filetypes=file_types
        )
        
        if filename:
            try:
                self.result_image.save(filename)
                self.status_var.set(f"Result saved: {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Result saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = VLMDetectorGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()
