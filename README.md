# BackgroundRemover

A command line tool to remove background from [video](https://backgroundremover.app/video)
and [image](https://backgroundremover.app/image), brought to you
by [BackgroundRemover.app](https://backgroundremover.app) which is an app made by [nadermx](https://john.nader.mx) powered by this tool

<img alt="background remover video" src="https://backgroundremover.app/static/backgroundremover.gif" height="200" width="110" />
<img alt="green screen matte key file" src="https://backgroundremover.app/static/matte.gif" height="200" width="110" />
<img alt="background remover image" src="https://backgroundremover.app/static/backgroundremoverexample.png" height="200" />

### Requirements

* python 3.6 (only one tested so far but may work for < 3.6)
* python3.6-dev
* torch and torchvision stable version (https://pytorch.org)

* ffmpeg 4.2+

#### How to install torch and fmpeg

Go to https://pytorch.org and scroll down to `INSTALL PYTORCH` section and follow the instructions.

For example:

```
PyTorch Build: Stable (1.7.1)
Your OS: Windows
Package: Pip
Language: Python
CUDA: None
```

To install ffmpeg

```
sudo apt install ffmpeg python3.6-dev
```

To install torch:

```
pip install --upgrade pip
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### Installation

To Install backgroundremover, install it from pypi

```bash
pip install backgroundremover
```

# Usage as a cli
## Image

Remove the background from a local file image

```bash
backgroundremover -i "/path/to/image.jpeg" -o "output.png"
```

### Advance usage for image background removal

Sometimes it is possible to achieve better results by turning on alpha matting. Example:

```bash
backgroundremover -i "/path/to/image.jpeg" a -ae 15 -o "output.png"
```
change the model for diferent background removal methods between `u2netp`, `u2net`, or `u2net_human_seg`
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
backgroundremover -i "/path/to/video.mp4" -tov -tv "/path/to/videtobeoverlayed.mp4" -o "output.mov"
```

### remove background from video and make transparent gif


```bash
backgroundremover -i "/path/to/video.mp4" -tg -o "output.gif"
```
### Make matte key file (green screen overlay)

Make a matte file for premier

```bash
backgroundremover -i "/path/to/video.mp4" -mk -o "output.matte.mp4"
```

### Advance usage for video

Change the framerate of the video (default is set to 30)

```bash
backgroundremover -i "/path/to/video.mp4" -fr 30 -tv -o "output.mov"
```

Change the gpu batch size of the video (default is set to 1)

```bash
backgroundremover -i "/path/to/video.mp4" -gp 4 -tv -o "output.mov"
```

Change the number of workers working on video (default is set to 1)

```bash
backgroundremover -i "/path/to/video.mp4" -wn 4 -tv -o "output.mov"
```
change the model for diferent background removal methods between `u2netp`, `u2net`, or `u2net_human_seg`
```bash
backgroundremover -i "/path/to/video.mp4" -m "u2net_human_seg"-tv -o "output.mov"
```

## Todo

- convert logic from video to image to utilize more GPU on image removal
- remove duplicate imports from image and video of u2net models
- clean up documentation a bit more
- add ability to adjust and give feedback images or videos to datasets
- other

### Pull requests

Accepted

### If you like this library

Give a link to our project [BackgroundRemover.app](https://backgroundremover.app) or this git, telling people that you like it or use it.

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

Licensed under [MIT License](./LICENSE.txt)
