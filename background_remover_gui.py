#!/usr/bin/env python3
"""
Simple GUI for Background Remover
A user-friendly interface for removing backgrounds from images using the backgroundremover library.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk
import threading
from backgroundremover.bg import remove


class BackgroundRemoverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Background Remover")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.model_choice = tk.StringVar(value="u2net")
        self.alpha_matting = tk.BooleanVar(value=False)
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Background Remover", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input file selection
        ttk.Label(main_frame, text="Input Image:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_file, width=50).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="Browse", 
                  command=self.browse_input_file).grid(row=1, column=2, pady=5)
        
        # Output file selection
        ttk.Label(main_frame, text="Output File:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_file, width=50).grid(
            row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="Browse", 
                  command=self.browse_output_file).grid(row=2, column=2, pady=5)
        
        # Model selection
        ttk.Label(main_frame, text="Model:").grid(row=3, column=0, sticky=tk.W, pady=5)
        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=5)
        
        models = ["u2net", "u2netp", "u2net_human_seg"]
        for i, model in enumerate(models):
            ttk.Radiobutton(model_frame, text=model, variable=self.model_choice, 
                           value=model).grid(row=0, column=i, padx=(0, 10))
        
        # Alpha matting option
        ttk.Checkbutton(main_frame, text="Use Alpha Matting (better quality)", 
                       variable=self.alpha_matting).grid(row=4, column=0, columnspan=3, 
                                                       sticky=tk.W, pady=10)
        
        # Process button
        self.process_button = ttk.Button(main_frame, text="Remove Background", 
                                        command=self.process_image, style="Accent.TButton")
        self.process_button.grid(row=5, column=0, columnspan=3, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to process images")
        self.status_label.grid(row=7, column=0, columnspan=3, pady=5)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding="5")
        preview_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(8, weight=1)
        
        # Input preview
        ttk.Label(preview_frame, text="Input:").grid(row=0, column=0, sticky=tk.W)
        self.input_preview = ttk.Label(preview_frame, text="No image selected", 
                                      background="lightgray", anchor="center")
        self.input_preview.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), 
                               padx=(0, 5), pady=5)
        
        # Output preview
        ttk.Label(preview_frame, text="Output:").grid(row=0, column=1, sticky=tk.W)
        self.output_preview = ttk.Label(preview_frame, text="No output yet", 
                                       background="lightgray", anchor="center")
        self.output_preview.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), 
                                padx=(5, 0), pady=5)
        
    def browse_input_file(self):
        """Browse for input image file"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select input image",
            filetypes=filetypes
        )
        
        if filename:
            self.input_file.set(filename)
            self.load_input_preview(filename)
            
            # Auto-generate output filename
            if not self.output_file.get():
                base_name = os.path.splitext(filename)[0]
                output_name = f"{base_name}_no_bg.png"
                self.output_file.set(output_name)
    
    def browse_output_file(self):
        """Browse for output file location"""
        filetypes = [
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Save output as",
            defaultextension=".png",
            filetypes=filetypes
        )
        
        if filename:
            self.output_file.set(filename)
    
    def load_input_preview(self, filename):
        """Load and display input image preview"""
        try:
            # Load and resize image for preview
            image = Image.open(filename)
            image.thumbnail((200, 200), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update preview label
            self.input_preview.configure(image=photo, text="")
            self.input_preview.image = photo  # Keep a reference
            
        except Exception as e:
            self.input_preview.configure(image="", text=f"Error loading image:\n{str(e)}")
    
    def load_output_preview(self, filename):
        """Load and display output image preview"""
        try:
            if os.path.exists(filename):
                # Load and resize image for preview
                image = Image.open(filename)
                image.thumbnail((200, 200), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image)
                
                # Update preview label
                self.output_preview.configure(image=photo, text="")
                self.output_preview.image = photo  # Keep a reference
            else:
                self.output_preview.configure(image="", text="Output file not found")
                
        except Exception as e:
            self.output_preview.configure(image="", text=f"Error loading output:\n{str(e)}")
    
    def process_image(self):
        """Process the image to remove background"""
        # Validate inputs
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input image file.")
            return
            
        if not self.output_file.get():
            messagebox.showerror("Error", "Please specify an output file.")
            return
            
        if not os.path.exists(self.input_file.get()):
            messagebox.showerror("Error", "Input file does not exist.")
            return
        
        # Start processing in a separate thread
        self.process_button.configure(state="disabled")
        self.progress.start()
        self.status_label.configure(text="Processing... Please wait.")
        
        # Run processing in background thread
        thread = threading.Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()
    
    def _process_image_thread(self):
        """Process image in background thread"""
        try:
            # Read input image
            with open(self.input_file.get(), "rb") as f:
                input_data = f.read()
            
            # Process with backgroundremover
            output_data = remove(
                input_data,
                model_name=self.model_choice.get(),
                alpha_matting=self.alpha_matting.get(),
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_structure_size=10,
                alpha_matting_base_size=1000
            )
            
            # Save output
            with open(self.output_file.get(), "wb") as f:
                f.write(output_data)
            
            # Update UI in main thread
            self.root.after(0, self._processing_complete)
            
        except Exception as e:
            # Handle errors in main thread
            self.root.after(0, lambda: self._processing_error(str(e)))
    
    def _processing_complete(self):
        """Called when processing is complete"""
        self.progress.stop()
        self.process_button.configure(state="normal")
        self.status_label.configure(text="Processing complete!")
        
        # Load output preview
        self.load_output_preview(self.output_file.get())
        
        # Show success message
        messagebox.showinfo("Success", 
                           f"Background removed successfully!\nOutput saved to:\n{self.output_file.get()}")
    
    def _processing_error(self, error_message):
        """Called when processing encounters an error"""
        self.progress.stop()
        self.process_button.configure(state="normal")
        self.status_label.configure(text="Processing failed.")
        
        # Show error message
        messagebox.showerror("Error", f"Failed to process image:\n{error_message}")


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    
    # Configure style
    style = ttk.Style()
    style.theme_use('clam')  # Use a modern theme
    
    # Create and run the application
    app = BackgroundRemoverGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
