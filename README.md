# BackgroundRemover
![Background Remover](https://raw.githubusercontent.com/nadermx/backgroundremover/main/examplefiles/backgroundremoverexample.png)
<img alt="background remover video" src="https://raw.githubusercontent.com/nadermx/backgroundremover/main/examplefiles/backgroundremoverprocessed.gif" height="200" /><br>
BackgroundRemover is a command line tool to remove background from [image](https://github.com/nadermx/backgroundremover#image) and [video](https://github.com/nadermx/backgroundremover#video) using AI, made by [nadermx](https://john.nader.mx) to power [https://BackgroundRemoverAI.com](https://backgroundremoverai.com). If you wonder why it was made read this [short blog post](https://johnathannader.com/my-first-open-source-project/).<br>


### Requirements

* python >= 3.6
* python3.6-dev #or what ever version of python you use
* torch and torchvision stable version (https://pytorch.org)
* ffmpeg 4.4+

* To clarify, you must install both python and whatever dev version of python you installed. IE; python3.10-dev with python3.10 or python3.8-dev with python3.8

#### How to install torch and ffmpeg

Go to https://pytorch.org and scroll down to `INSTALL PYTORCH` section and follow the instructions.

For example:

```
PyTorch Build: Stable (1.7.1)
Your OS: Windows
Package: Pip
Language: Python
CUDA: None
```

To install ffmpeg and python-dev

```
sudo apt install ffmpeg python3.6-dev
```

### Installation
To Install backgroundremover, install it from pypi

```bash
pip install --upgrade pip
pip install backgroundremover
```
Please note that when you first run the program, it will check to see if you have the u2net models, if you do not, it will pull them from this repo

It is also possible to run this without installing it via pip, just clone the git to local start a virtual env and install requirements and run
```bash
python -m backgroundremover.cmd.cli -i "video.mp4" -mk -o "output.mov"
```
and for windows
```bash
python.exe -m backgroundremover.cmd.cli -i "video.mp4" -mk -o "output.mov"
```
### Installation using Docker
```bash
git clone https://github.com/nadermx/backgroundremover.git
cd backgroundremover
docker build -t bgremover .
# Basic usage (models will be downloaded on each run)
alias backgroundremover='docker run -it --rm -v "$(pwd):/tmp" bgremover:latest'

# Recommended: Persist models between runs to avoid re-downloading
mkdir -p ~/.u2net
alias backgroundremover='docker run -it --rm -v "$(pwd):/tmp" -v "$HOME/.u2net:/root/.u2net" bgremover:latest'
```
### Usage as a cli
## Image

Remove the background from a local file image

```bash
backgroundremover -i "/path/to/image.jpeg" -o "output.png"
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif` (HEIC/HEIF support requires pillow-heif)
### Process all images in a folder

You can now remove backgrounds from all supported image or video files in a folder using the `--input-folder` (`-if`) option. You can also optionally set an output folder using `--output-folder` (`-of`). If `--output-folder` is not provided, the outputs will be saved in the same input folder, prefixed with `output_`.

### Example: Folder of Images

```bash
backgroundremover -if "/path/to/image-folder" -of "/path/to/output-folder"
```

This will process all `.jpg`, `.jpeg`, `.png`, `.heic`, and `.heif` images in the folder and save the results to the output folder.



### Advance usage for image background removal

Sometimes it is possible to achieve better results by turning on alpha matting. Example:

```bash
backgroundremover -i "/path/to/image.jpeg" -a -ae 15 -o "output.png"
```
change the model for different background removal methods between `u2netp`, `u2net`, or `u2net_human_seg`
```bash
backgroundremover -i "/path/to/image.jpeg" -m "u2net_human_seg" -o "output.png"
```

### Output only the mask (binary mask/matte)

```bash
backgroundremover -i "/path/to/image.jpeg" -om -o "mask.png"
```

### Replace background with a custom color

```bash
# Replace with red background
backgroundremover -i "/path/to/image.jpeg" -bc "255,0,0" -o "output.png"

# Replace with green background
backgroundremover -i "/path/to/image.jpeg" -bc "0,255,0" -o "output.png"

# Replace with blue background
backgroundremover -i "/path/to/image.jpeg" -bc "0,0,255" -o "output.png"
```

### Replace background with a custom image

```bash
# Replace background with another image
backgroundremover -i "/path/to/image.jpeg" -bi "/path/to/background.jpg" -o "output.png"
```

### Use with pipes (stdin/stdout)

You can use backgroundremover in Unix pipelines by reading from stdin and writing to stdout:

```bash
# Read from stdin, write to stdout
cat input.jpg | backgroundremover > output.png

# Use with other tools in a pipeline
curl https://example.com/image.jpg | backgroundremover | convert - -resize 50% smaller.png

# Equivalent explicit syntax
backgroundremover -i - -o - < input.jpg > output.png
```

Note: Pipe mode assumes image input (not video).

### Run as HTTP API Server

You can run backgroundremover as an HTTP API server:

```bash
# Start server on default port 5000
backgroundremover-server

# Specify custom host and port
backgroundremover-server --addr 0.0.0.0 --port 8080
```

API Usage:

```bash
# Upload image via POST
curl -X POST -F "file=@image.jpg" http://localhost:5000/ -o output.png

# Process from URL via GET
curl "http://localhost:5000/?url=https://example.com/image.jpg" -o output.png

# With alpha matting
curl "http://localhost:5000/?url=https://example.com/image.jpg&a=true&af=240" -o output.png

# Choose model
curl "http://localhost:5000/?url=https://example.com/image.jpg&model=u2net_human_seg" -o output.png
```

Parameters:
- `a` - Enable alpha matting
- `af` - Alpha matting foreground threshold (default: 240)
- `ab` - Alpha matting background threshold (default: 10)
- `ae` - Alpha matting erosion size (default: 10)
- `az` - Alpha matting base size (default: 1000)
- `model` - Model choice: `u2net`, `u2netp`, or `u2net_human_seg`

## Video

### remove background from video and make transparent mov

```bash
backgroundremover -i "/path/to/video.mp4" -tv -o "output.mov"
```
### Process all videos in a folder

You can now remove backgrounds from all supported image or video files in a folder using the `--input-folder` (`-if`) option. You can also optionally set an output folder using `--output-folder` (`-of`). If `--output-folder` is not provided, the outputs will be saved in the same input folder, prefixed with `output_`.

### Example: Folder of Videos to Transparent `.mov`

```bash
backgroundremover -if "/path/to/video-folder" -of "/path/to/output-folder" -tv
```

You can also combine additional options:

```bash
backgroundremover -if "videos" -of "processed" -m "u2net_human_seg" -fr 30 -tv
```

- Uses the `u2net_human_seg` model
- Overrides video framerate to 30 fps
- Outputs transparent `.mov` files into the `processed/` folder
- Supported video formats: `.mp4`, `.mov`, `.webm`, `.ogg`, `.gif`
- Output files will be named like `output_filename.ext` in the output folder

### remove background from local video and overlay it over other video
```bash
backgroundremover -i "/path/to/video.mp4" -tov -bv "/path/to/background_video.mp4" -o "output.mov"
```
### remove background from local video and overlay it over an image
```bash
backgroundremover -i "/path/to/video.mp4" -toi -bi "/path/to/background_image.png" -o "output.mov"
```

### remove background from video and make transparent gif


```bash
backgroundremover -i "/path/to/video.mp4" -tg -o "output.gif"
```
### Make matte key file (green screen overlay)

Make a matte file for premiere

```bash
backgroundremover -i "/path/to/video.mp4" -mk -o "output.matte.mp4"
```

### Advance usage for video

Change the framerate of the video (default is set to 30)

```bash
backgroundremover -i "/path/to/video.mp4" -fr 30 -tv -o "output.mov"
```

Set total number of frames of the video (default is set to -1, ie the remove background from full video)

```bash
backgroundremover -i "/path/to/video.mp4" -fl 150 -tv -o "output.mov"
```

Change the gpu batch size of the video (default is set to 1)

```bash
backgroundremover -i "/path/to/video.mp4" -gb 4 -tv -o "output.mov"
```

Change the number of workers working on video (default is set to 1)

```bash
backgroundremover -i "/path/to/video.mp4" -wn 4 -tv -o "output.mov"
```
change the model for different background removal methods between `u2netp`, `u2net`, or `u2net_human_seg` and limit the frames to 150
```bash
backgroundremover -i "/path/to/video.mp4" -m "u2net_human_seg" -fl 150 -tv -o "output.mov"
```

## As a library
### Remove background image

```python
from backgroundremover.bg import remove

def remove_bg(src_img_path, out_img_path):
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]
    f = open(src_img_path, "rb")
    data = f.read()
    img = remove(data, model_name=model_choices[0],
                 alpha_matting=True,
                 alpha_matting_foreground_threshold=240,
                 alpha_matting_background_threshold=10,
                 alpha_matting_erode_structure_size=10,
                 alpha_matting_base_size=1000)
    f.close()
    f = open(out_img_path, "wb")
    f.write(img)
    f.close()
```

### Generate only a binary mask

```python
from backgroundremover.bg import remove

f = open("input.jpg", "rb")
data = f.read()
mask = remove(data, model_name="u2net", only_mask=True)
f.close()

f = open("mask.png", "wb")
f.write(mask)
f.close()
```

### Replace background with custom color

```python
from backgroundremover.bg import remove

f = open("input.jpg", "rb")
data = f.read()
# Use RGB tuple for background color (255, 0, 0) = red
img = remove(data, model_name="u2net", background_color=(255, 0, 0))
f.close()

f = open("output.png", "wb")
f.write(img)
f.close()
```

### Replace background with custom image

```python
from backgroundremover.bg import remove

# Read input image
with open("input.jpg", "rb") as f:
    input_data = f.read()

# Read background image
with open("background.jpg", "rb") as f:
    bg_data = f.read()

# Remove background and composite over background image
result = remove(input_data, model_name="u2net", background_image=bg_data)

# Save result
with open("output.png", "wb") as f:
    f.write(result)
```

## Testing

Currently, this project does not have automated test cases. Testing is done manually using sample images and videos.

### Manual Testing

To test backgroundremover functionality:

**Test Image Background Removal:**
```bash
# Basic test
backgroundremover -i "test_image.jpg" -o "output.png"

# Test with alpha matting
backgroundremover -i "test_image.jpg" -a -ae 15 -o "output.png"

# Test mask generation
backgroundremover -i "test_image.jpg" -om -o "mask.png"

# Test custom background color
backgroundremover -i "test_image.jpg" -bc "0,255,0" -o "output.png"
```

**Test Video Processing:**
```bash
# Test transparent video
backgroundremover -i "test_video.mp4" -tv -o "output.mov"

# Test matte key
backgroundremover -i "test_video.mp4" -mk -o "matte.mov"

# Test transparent GIF
backgroundremover -i "test_video.mp4" -tg -o "output.gif"
```

**Test HTTP Server:**
```bash
# Start server
backgroundremover-server --port 5000

# Test with curl (in another terminal)
curl -X POST -F "file=@test_image.jpg" http://localhost:5000/ -o output.png
```

### Contributing Tests

Automated tests using pytest or unittest would be a valuable contribution to this project. Test cases should cover:
- Image processing with different formats (JPG, PNG, HEIC)
- Video processing with different codecs
- CLI argument validation
- HTTP API endpoints
- Model loading and inference
- Error handling

## Todo

- convert logic from video to image to utilize more GPU on image removal
- clean up documentation a bit more
- add ability to adjust and give feedback images or videos to datasets
- add ability to realtime background removal for videos, for streaming
- add ability to use other models than u2net, ie your own
- add automated test suite
- other

### Pull requests

Accepted

### If you like this library

Give a link to our project [BackgroundRemoverAI.com](https://backgroundremoverai.com) or this git, telling people that you like it or use it.

### Reason for project

We made it our own package after merging together parts of others, adding in a few features of our own via posting parts as bounty questions on superuser, etc.  As well as asked on hackernews earlier to open source the image part, so decided to add in video, and a bit more.



### References

- https://arxiv.org/pdf/2005.09007.pdf
- https://github.com/NathanUA/U-2-Net
- https://github.com/pymatting/pymatting
- https://github.com/danielgatis/rembg
- https://github.com/ecsplendid/rembg-greenscreen
- https://superuser.com/questions/1647590/have-ffmpeg-merge-a-matte-key-file-over-the-normal-video-file-removing-the-backg
- https://superuser.com/questions/1648680/ffmpeg-alphamerge-two-videos-into-a-gif-with-transparent-background/1649339?noredirect=1#comment2522687_1649339
- https://superuser.com/questions/1649817/ffmpeg-overlay-a-video-after-alphamerging-two-others/1649856#1649856

### License

- Copyright (c) 2021-present [Johnathan Nader](https://github.com/nadermx)
- Copyright (c) 2020-present [Lucas Nestler](https://github.com/ClashLuke)
- Copyright (c) 2020-present [Dr. Tim Scarfe](https://github.com/ecsplendid)
- Copyright (c) 2020-present [Daniel Gatis](https://github.com/danielgatis)

Code Licensed under [MIT License](./LICENSE.txt)
Models Licensed under [Apache License 2.0](./models/license)
