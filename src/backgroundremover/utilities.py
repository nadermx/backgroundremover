import os
import math
import torch.multiprocessing as multiprocessing
import subprocess as sp
import time
import ffmpeg
import numpy as np
import torch
import tempfile
from .bg import DEVICE, Net, iter_frames, remove_many
import shlex
import gdown
from tqdm import tqdm

multiprocessing.set_start_method('spawn', force=True)


def worker(worker_nodes,
           worker_index,
           result_dict,
           model_name,
           gpu_batchsize,
           total_frames,
           frames_dict):
    print(F"WORKER {worker_index} ONLINE")

    output_index = worker_index + 1
    base_index = worker_index * gpu_batchsize
    net = Net(model_name)
    script_net = None
    for fi in (list(range(base_index + i * worker_nodes * gpu_batchsize,
                          min(base_index + i * worker_nodes * gpu_batchsize + gpu_batchsize, total_frames)))
               for i in range(math.ceil(total_frames / worker_nodes / gpu_batchsize))):
        if not fi:
            break

        # are we processing frames faster than the frame ripper is saving them?
        last = fi[-1]
        while last not in frames_dict:
            time.sleep(0.1)

        input_frames = [frames_dict[index] for index in fi]
        if script_net is None:
            script_net = torch.jit.trace(net,
                                         torch.as_tensor(np.stack(input_frames), dtype=torch.float32, device=DEVICE))

        result_dict[output_index] = remove_many(input_frames, script_net)

        # clean up the frame buffer
        for fdex in fi:
            del frames_dict[fdex]
        output_index += worker_nodes


def capture_frames(file_path, frames_dict, prefetched_samples, total_frames):
    print(F"WORKER FRAMERIPPER ONLINE")
    for idx, frame in enumerate(iter_frames(file_path)):
        frames_dict[idx] = frame
        while len(frames_dict) > prefetched_samples:
            time.sleep(0.1)
        if idx > total_frames:
            break


def matte_key(output, file_path,
              worker_nodes,
              gpu_batchsize,
              model_name,
              frame_limit=-1,
              prefetched_batches=4,
              framerate=-1):
    manager = multiprocessing.Manager()

    results_dict = manager.dict()
    frames_dict = manager.dict()

    print(file_path)

    info = ffmpeg.probe(file_path)
    total_frames = int(info["streams"][0]["nb_frames"])

    if frame_limit != -1:
        total_frames = min(frame_limit, total_frames)

    fr = info["streams"][0]["r_frame_rate"]

    if framerate == -1:
        print(F"FRAME RATE DETECTED: {fr} (if this looks wrong, override the frame rate)")
        framerate = math.ceil(eval(fr))

    print(F"FRAME RATE: {framerate} TOTAL FRAMES: {total_frames}")

    p = multiprocessing.Process(target=capture_frames,
                                args=(file_path, frames_dict, gpu_batchsize * prefetched_batches, total_frames))
    p.start()

    # note I am deliberatley not using pool
    # we can't trust it to run all the threads concurrently (or at all)
    workers = [multiprocessing.Process(target=worker,
                                       args=(worker_nodes, wn, results_dict, model_name, gpu_batchsize, total_frames,
                                             frames_dict))
               for wn in range(worker_nodes)]
    for w in workers:
        w.start()

    command = None
    proc = None
    frame_counter = 0
    for i in range(math.ceil(total_frames / worker_nodes)):
        for wx in range(worker_nodes):

            hash_index = i * worker_nodes + 1 + wx

            while hash_index not in results_dict:
                time.sleep(0.1)

            frames = results_dict[hash_index]
            # dont block access to it anymore
            del results_dict[hash_index]

            for frame in frames:
                if command is None:
                    command = ['ffmpeg',
                               '-y',
                               '-f', 'rawvideo',
                               '-vcodec', 'rawvideo',
                               '-s', F"{frame.shape[1]}x320",
                               '-pix_fmt', 'gray',
                               '-r', F"{framerate}",
                               '-i', '-',
                               '-an',
                               '-vcodec', 'mpeg4',
                               '-b:v', '2000k',
                               '%s' % output]

                    proc = sp.Popen(command, stdin=sp.PIPE)

                proc.stdin.write(frame.tostring())
                frame_counter = frame_counter + 1

                if frame_counter >= total_frames:
                    p.join()
                    for w in workers:
                        w.join()
                    proc.stdin.close()
                    proc.wait()
                    print(F"FINISHED ALL FRAMES ({total_frames})!")
                    return

    p.join()
    for w in workers:
        w.join()
    proc.stdin.close()
    proc.wait()
    return


def transparentgif(output, file_path,
                   worker_nodes,
                   gpu_batchsize,
                   model_name,
                   frame_limit=-1,
                   prefetched_batches=4,
                   framerate=-1):
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_file = os.path.abspath("%s/matte.mp4" % tmpdirname)
        matte_key(temp_file, file_path,
                  worker_nodes,
                  gpu_batchsize,
                  model_name,
                  frame_limit,
                  prefetched_batches,
                  framerate)
        cmd = "nice -10 ffmpeg -y -i %s -i %s -filter_complex '[1][0]scale2ref[mask][main];[main][mask]alphamerge=shortest=1,fps=10,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' -shortest %s" % (
            file_path, temp_file, output)
        sp.run(shlex.split(cmd))
        print("Process finished")

    return


def transparentgifwithbackground(output, overlay, file_path,
                      worker_nodes,
                      gpu_batchsize,
                      model_name,
                      frame_limit=-1,
                      prefetched_batches=4,
                      framerate=-1):
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_file = os.path.abspath("%s/matte.mp4" % tmpdirname)
        matte_key(temp_file, file_path,
                  worker_nodes,
                  gpu_batchsize,
                  model_name,
                  frame_limit,
                  prefetched_batches,
                  framerate)
        print("Starting alphamerge")
        cmd = "nice -10 ffmpeg -y -i %s -i %s -i %s -filter_complex '[1][0]scale2ref[mask][main];[main][mask]alphamerge=shortest=1[fg];[2][fg]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2:format=auto,fps=10,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' -shortest %s" % (
            file_path, temp_file, overlay, output)
        sp.run(shlex.split(cmd))
        print("Process finished")

    return


def transparentvideo(output, file_path,
                     worker_nodes,
                     gpu_batchsize,
                     model_name,
                     frame_limit=-1,
                     prefetched_batches=4,
                     framerate=-1):
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_file = os.path.abspath("%s/matte.mp4" % tmpdirname)
        matte_key(temp_file, file_path,
                  worker_nodes,
                  gpu_batchsize,
                  model_name,
                  frame_limit,
                  prefetched_batches,
                  framerate)
        print("Starting alphamerge")
        cmd = "nice -10 ffmpeg -y -nostats -loglevel 0 -i %s -i %s -filter_complex '[1][0]scale2ref[mask][main];[main][mask]alphamerge=shortest=1' -c:v qtrle -shortest %s" % (
            file_path, temp_file, output)
        process = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, stderr = process.communicate()
        print('after call')

        if stderr:
            return "ERROR: %s" % stderr.decode("utf-8")
        print("Process finished")

    return


def transparentvideoovervideo(output, overlay, file_path,
                         worker_nodes,
                         gpu_batchsize,
                         model_name,
                         frame_limit=-1,
                         prefetched_batches=4,
                         framerate=-1):
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_file = os.path.abspath("%s/matte.mp4" % tmpdirname)
        matte_key(temp_file, file_path,
                  worker_nodes,
                  gpu_batchsize,
                  model_name,
                  frame_limit,
                  prefetched_batches,
                  framerate)
        print("Starting alphamerge")
        cmd = "nice -10 ffmpeg -y -i %s -i %s -i %s -filter_complex '[1][0]scale2ref[mask][main];[main][mask]alphamerge=shortest=1[vid];[vid][2:v]scale2ref[fg][bg];[bg][fg]overlay=shortest=1[out]' -map [out] -shortest %s" % (
            file_path, temp_file, overlay, output)
        sp.run(shlex.split(cmd))
        print("Process finished")
    return


def transparentvideooverimage(output, overlay, file_path,
                         worker_nodes,
                         gpu_batchsize,
                         model_name,
                         frame_limit=-1,
                         prefetched_batches=4,
                         framerate=-1):
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_file = os.path.abspath("%s/matte.mp4" % tmpdirname)
        matte_key(temp_file, file_path,
                  worker_nodes,
                  gpu_batchsize,
                  model_name,
                  frame_limit,
                  prefetched_batches,
                  framerate)
        print("Scale image")
        temp_image = os.path.abspath("%s/new.jpg" % tmpdirname)
        cmd = "nice -10 ffmpeg -i %s -i %s -filter_complex 'scale2ref[img][vid];[img]setsar=1;[vid]nullsink' -q:v 2 %s" % (
            overlay, file_path, temp_image)
        sp.run(shlex.split(cmd))
        print("Starting alphamerge")
        cmd = "nice -10 ffmpeg -y -i %s -i %s -i %s -filter_complex '[0][1]scale2ref[img][vid];[img]setsar=1[img];[vid]nullsink; [img][2]overlay=(W-w)/2:(H-h)/2' -shortest %s" % (
        #cmd = "nice -10 ffmpeg -y -i %s -i %s -i %s -filter_complex '[1][0]scale2ref[mask][main];[main][mask]alphamerge=shortest=1[vid];[2:v][vid]overlay[out]' -map [out] -shortest %s" % (
            temp_image, file_path, temp_file, output)
        sp.run(shlex.split(cmd))
        print("Process finished")
    return

def download_file_from_google_drive(model, path):
    head, tail = os.path.split(path)
    os.makedirs(head, exist_ok=True)
    URL = "https://drive.google.com/uc?id=%s" % model[2]

    gdown.download(URL, path, quiet=False)
