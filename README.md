# BackgroundRemover
![Background Remover](https://raw.githubusercontent.com/nadermx/backgroundremover/main/examplefiles/backgroundremoverexample.png)
<img alt="background remover video" src="https://raw.githubusercontent.com/nadermx/backgroundremover/main/examplefiles/backgroundremoverprocessed.gif" height="200" /><br>
BackgroundRemover is a command line tool to remove background from [image](https://github.com/nadermx/backgroundremover#image) and [video](https://github.com/nadermx/backgroundremover#video) using AI, made by [nadermx](https://john.nader.mx) to power [https://BackgroundRemoverAI.com](https://backgroundremoverai.com). If you wonder why it was made read this [short blog post](https://johnathannader.com/my-first-open-source-project/).<br>


### Requirements

* python >= 3.6 =< 3.11 #does not work for 3.12 yet
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
alias backgroundremover='docker run -it --rm -v "$(pwd):/tmp" bgremover:latest'
```
### Usage as a cli
## Image

Remove the background from a local file image

```bash
backgroundremover -i "/path/to/image.jpeg" -o "output.png"
```
### Process all images or videos in a folder

You can now remove backgrounds from all supported image or video files in a folder using the `--input-folder` (`-if`) option. You can also optionally set an output folder using `--output-folder` (`-of`). If `--output-folder` is not provided, the outputs will be saved in the same input folder, prefixed with `output_`.

### Example: Folder of Images

```bash
backgroundremover -if "/path/to/image-folder" -of "/path/to/output-folder"
```

This will process all `.jpg`, `.jpeg`, and `.png` images in the folder and save the results to the output folder.

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

### 💡 Notes

- Supported image formats: `.jpg`, `.jpeg`, `.png`
- Supported video formats: `.mp4`, `.mov`, `.webm`, `.ogg`, `.gif`
- Output files will be named like `output_filename.ext` in the output folder
### Advance usage for image background removal

Sometimes it is possible to achieve better results by turning on alpha matting. Example:

```bash
backgroundremover -i "/path/to/image.jpeg" -a -ae 15 -o "output.png"
```
change the model for different background removal methods between `u2netp`, `u2net`, or `u2net_human_seg`
```bash
backgroundremover -i "/path/to/image.jpeg" -m "u2net_human_seg" -o "output.png"
```
## Video

### remove background from video and make transparent mov

```bash
backgroundremover -i "/path/to/video.mp4" -tv -o "output.mov"
```
### remove background from local video and overlay it over other video
```bash
backgroundremover -i "/path/to/video.mp4" -tov "/path/to/videtobeoverlayed.mp4" -o "output.mov"
```
### remove background from local video and overlay it over an image
```bash
backgroundremover -i "/path/to/video.mp4" -toi "/path/to/videtobeoverlayed.mp4" -o "output.mov"
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

```
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

## Todo

- convert logic from video to image to utilize more GPU on image removal
- clean up documentation a bit more
- add ability to adjust and give feedback images or videos to datasets
- add ability to realtime background removal for videos, for streaming
- finish flask server api
- add ability to use other models than u2net, ie your own
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
