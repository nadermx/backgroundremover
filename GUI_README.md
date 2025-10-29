# Background Remover GUI

A simple and user-friendly graphical interface for removing backgrounds from images using the backgroundremover library.

## Features

- **Easy File Selection**: Browse and select input images with a simple file dialog
- **Multiple Model Support**: Choose between different AI models (u2net, u2netp, u2net_human_seg)
- **Alpha Matting**: Optional high-quality background removal with alpha matting
- **Live Preview**: See input and output images side by side
- **Progress Indication**: Visual feedback during processing
- **Auto Output Naming**: Automatically suggests output filenames
- **Error Handling**: Clear error messages and validation

## How to Use

### Method 1: Run the GUI directly
```bash
python background_remover_gui.py
```

### Method 2: Use the batch file (Windows)
Double-click `run_gui.bat` or run it from command prompt.

## Step-by-Step Instructions

1. **Launch the Application**
   - Run `python background_remover_gui.py` or double-click `run_gui.bat`

2. **Select Input Image**
   - Click "Browse" next to "Input Image"
   - Choose your image file (supports JPG, PNG, BMP, TIFF)
   - The input preview will show your selected image

3. **Choose Output Location**
   - The output filename is auto-generated (you can change it)
   - Click "Browse" next to "Output File" to choose a different location
   - Output will be saved as PNG format

4. **Select Model (Optional)**
   - **u2net**: General purpose model (default)
   - **u2netp**: Lighter, faster model
   - **u2net_human_seg**: Optimized for human subjects

5. **Enable Alpha Matting (Optional)**
   - Check "Use Alpha Matting" for higher quality results
   - Takes longer but produces better edge quality

6. **Process the Image**
   - Click "Remove Background"
   - Wait for processing to complete
   - The output preview will show the result

## Supported File Formats

### Input Formats:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

### Output Format:
- PNG (.png) - Always with transparency

## Tips for Best Results

1. **For People**: Use `u2net_human_seg` model
2. **For Objects**: Use `u2net` or `u2netp` model
3. **High Quality**: Enable alpha matting for better edge quality
4. **Speed**: Use `u2netp` model for faster processing
5. **File Size**: PNG output files may be larger than original JPEGs

## Troubleshooting

### Common Issues:

1. **"Input file does not exist"**
   - Make sure the file path is correct
   - Check if the file was moved or deleted

2. **"Failed to process image"**
   - Ensure the image file is not corrupted
   - Try a different image format
   - Check if you have enough disk space

3. **GUI doesn't start**
   - Make sure Python and tkinter are installed
   - Run from command prompt to see error messages

4. **Slow processing**
   - This is normal for large images
   - Consider using `u2netp` model for faster processing
   - Alpha matting takes longer but produces better results

## Technical Details

- **Framework**: Python tkinter
- **Image Processing**: PIL (Pillow)
- **Background Removal**: backgroundremover library
- **Threading**: Processing runs in background thread to keep GUI responsive

## Requirements

- Python 3.6+
- backgroundremover library (already installed)
- tkinter (usually included with Python)
- PIL/Pillow (already installed)

## File Structure

```
background_remover_gui.py    # Main GUI application
run_gui.bat                  # Windows batch file to run GUI
GUI_README.md               # This documentation
```

Enjoy using the Background Remover GUI!
