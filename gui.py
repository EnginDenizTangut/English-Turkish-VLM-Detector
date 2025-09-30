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
from main import VLMDetector

class VLMDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("English-Turkish VLM Detector")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize detector
        self.detector = VLMDetector()
        
        # Variables
        self.current_image_path = None
        self.original_image = None
        self.result_image = None
        
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
        
        # File selection
        file_frame = ttk.Frame(control_frame)
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="📁 Image File:", style='Subtitle.TLabel').grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, style='Modern.TEntry', state='readonly')
        self.file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file, style='Modern.TButton')
        self.browse_button.grid(row=0, column=2)
        
        # Prompt input
        prompt_frame = ttk.Frame(control_frame)
        prompt_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
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
        button_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
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
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def browse_file(self):
        """Open file dialog to select image"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=file_types
        )
        
        if filename:
            self.current_image_path = filename
            self.file_path_var.set(filename)
            self.load_original_image()
            self.status_var.set(f"Image loaded: {os.path.basename(filename)}")
    
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
    
    def set_prompt(self, prompt):
        """Set prompt from example button"""
        self.prompt_var.set(prompt)
        self.prompt_entry.focus()
    
    def detect_objects(self):
        """Detect objects in the image"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first!")
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
        thread = threading.Thread(target=self.run_detection, args=(prompt,))
        thread.daemon = True
        thread.start()
    
    def run_detection(self, prompt):
        """Run detection in background thread"""
        try:
            # Run detection
            boxes, confidences, classes = self.detector.process_image(self.current_image_path, prompt)
            
            # Load result image
            result_path = "output_detection.jpg"
            if os.path.exists(result_path):
                result_image = Image.open(result_path)
                self.result_image = result_image.copy()
                
                # Update GUI in main thread
                self.root.after(0, self.display_result, result_image)
                
                # Update status
                self.root.after(0, self.update_status, 
                              f"Detection completed! Found {len(classes)} objects: {', '.join(classes)}")
            else:
                self.root.after(0, self.update_status, "No objects detected")
                
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
        self.current_image_path = None
        self.file_path_var.set("")
        self.prompt_var.set("")
        
        self.original_canvas.delete("all")
        self.result_canvas.delete("all")
        
        self.original_image = None
        self.result_image = None
        
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
