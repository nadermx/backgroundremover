import os
import math
from fractions import Fraction
import torch.multiprocessing as multiprocessing
import subprocess as sp
import time
import ffmpeg
import numpy as np
import torch
from .bg import DEVICE, Net, iter_frames, remove_many
import tempfile
import requests
from pathlib import Path

multiprocessing.set_start_method('spawn', force=True)


def _parse_frame_rate(rate_str):
    try:
        return float(Fraction(rate_str))
    except Exception:
        return float(rate_str)


def _alpha_encoding_args(output, alpha_codec, alpha_pix_fmt):
    ext = Path(output).suffix.lower()
    if alpha_codec in (None, "auto"):
        if ext == ".webm":
            alpha_codec = "libvpx-vp9"
        else:
            alpha_codec = "qtrle"

    if alpha_codec == "prores_ks":
        args = ["-c:v", "prores_ks", "-profile:v", "4"]
        pix_fmt = alpha_pix_fmt or "yuva444p10le"
        args += ["-pix_fmt", pix_fmt]
        return args

    if alpha_codec == "libvpx-vp9":
        args = ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0"]
        pix_fmt = alpha_pix_fmt or "yuva420p"
        args += ["-pix_fmt", pix_fmt]
        return args

    if alpha_codec == "qtrle":
        pix_fmt = alpha_pix_fmt or "argb"
        return ["-c:v", "qtrle", "-pix_fmt", pix_fmt]

    args = ["-c:v", alpha_codec]
    if alpha_pix_fmt:
        args += ["-pix_fmt", alpha_pix_fmt]
    return args


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


    info = ffmpeg.probe(file_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "csv=p=0",
        file_path
    ]
    framerate_output = sp.check_output(cmd, universal_newlines=True)

    total_frames = int(framerate_output.split(",")[0])
    if frame_limit != -1:
        total_frames = min(frame_limit, total_frames)

    video_stream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if not video_stream:
        raise Exception("Could not find video stream")

    frame_rate_str = video_stream.get("r_frame_rate", "0/0")
    if frame_rate_str == "0/0":
        raise Exception("Could not detect framerate of video")

    if framerate == -1:
        print(F"FRAME RATE DETECTED: {frame_rate_str} (if this looks wrong, override the frame rate)")
        framerate_str = frame_rate_str
        framerate_value = _parse_frame_rate(frame_rate_str)
    else:
        framerate_str = str(framerate)
        framerate_value = float(framerate)

    print(F"FRAME RATE: {framerate_value} TOTAL FRAMES: {total_frames}")

    p = multiprocessing.Process(target=capture_frames,
                                args=(file_path, frames_dict, gpu_batchsize * prefetched_batches, total_frames))
    p.start()

    # note I am deliberately not using pool
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

            try:
                timeout_counter = 0
                while hash_index not in results_dict:
                    time.sleep(0.1)
                    timeout_counter += 1
                    # Check if workers are still alive every 10 seconds
                    if timeout_counter % 100 == 0:
                        dead_workers = [w for w in workers if not w.is_alive()]
                        if dead_workers and hash_index not in results_dict:
                            raise RuntimeError(f"Worker process crashed while waiting for frame batch {hash_index}. Try reducing worker count with -wn 1")

                frames = results_dict[hash_index]
            except (ConnectionResetError, BrokenPipeError) as e:
                print(f"\nError: Worker connection lost (frame batch {hash_index}). This often happens with high worker counts.")
                print("Try reducing workers with -wn 1 or -wn 2")
                # Clean up
                p.terminate()
                for w in workers:
                    w.terminate()
                if proc:
                    proc.stdin.close()
                    proc.wait()
                raise RuntimeError(f"Worker connection error: {e}")
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
                               '-r', framerate_str,
                               '-i', '-',
                               '-an',
                               '-vcodec', 'mpeg4',
                               '-b:v', '2000k',
                               '%s' % output]

                    proc = sp.Popen(command, stdin=sp.PIPE)

                proc.stdin.write(frame.tobytes())
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
    temp_dir = tempfile.TemporaryDirectory()
    tmpdirname = Path(temp_dir.name)
    temp_file = os.path.abspath(os.path.join(tmpdirname, "matte.mp4"))
    matte_key(temp_file, file_path,
              worker_nodes,
              gpu_batchsize,
              model_name,
              frame_limit,
              prefetched_batches,
              framerate)
    cmd = [
        'ffmpeg', '-y', '-i', file_path, '-i', temp_file, '-filter_complex',
        '[1][0]scale2ref[mask][main];[main][mask]alphamerge,fps=10,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        '-shortest', output
    ]

    sp.run(cmd)

    print("Process finished")

    return


def transparentgifwithbackground(output, overlay, file_path,
                      worker_nodes,
                      gpu_batchsize,
                      model_name,
                      frame_limit=-1,
                      prefetched_batches=4,
                      framerate=-1):
    temp_dir = tempfile.TemporaryDirectory()
    tmpdirname = Path(temp_dir.name)
    temp_file = os.path.abspath(os.path.join(tmpdirname, "matte.mp4"))
    matte_key(temp_file, file_path,
              worker_nodes,
              gpu_batchsize,
              model_name,
              frame_limit,
              prefetched_batches,
              framerate)
    print("Starting alphamerge")
    cmd = [
        'ffmpeg', '-y', '-i', file_path, '-i', temp_file, '-i', overlay, '-filter_complex',
        '[1][0]scale2ref[mask][main];[main][mask]alphamerge[fg];[2][fg]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2:format=auto,fps=10,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        '-shortest', output
    ]
    sp.run(cmd)
    print("Process finished")
    try:
        temp_dir.cleanup()
    except PermissionError:
        pass
    return


def transparentvideo(output, file_path,
                     worker_nodes,
                     gpu_batchsize,
                     model_name,
                     frame_limit=-1,
                     prefetched_batches=4,
                     framerate=-1,
                     alpha_codec="auto",
                     alpha_pix_fmt=None):
    temp_dir = tempfile.TemporaryDirectory()
    tmpdirname = Path(temp_dir.name)
    temp_file = os.path.abspath(os.path.join(tmpdirname, "matte.mp4"))
    matte_key(temp_file, file_path,
              worker_nodes,
              gpu_batchsize,
              model_name,
              frame_limit,
              prefetched_batches,
              framerate)
    print("Starting alphamerge")
    encoding_args = _alpha_encoding_args(output, alpha_codec, alpha_pix_fmt)
    cmd = [
        'ffmpeg', '-y', '-i', file_path, '-i', temp_file, '-filter_complex',
        '[1][0]scale2ref[mask][main];[main][mask]alphamerge[v]',
        '-map', '[v]', '-map', '0:a?', '-shortest'
    ]
    cmd += encoding_args + [output]

    sp.run(cmd)
    print("Process finished")
    try:
        temp_dir.cleanup()
    except PermissionError:
        pass
    return


def transparentvideoovervideo(output, overlay, file_path,
                         worker_nodes,
                         gpu_batchsize,
                         model_name,
                         frame_limit=-1,
                         prefetched_batches=4,
                         framerate=-1,
                         alpha_codec="auto",
                         alpha_pix_fmt=None):
    temp_dir = tempfile.TemporaryDirectory()
    tmpdirname = Path(temp_dir.name)
    temp_file = os.path.abspath(os.path.join(tmpdirname, "matte.mp4"))
    matte_key(temp_file, file_path,
              worker_nodes,
              gpu_batchsize,
              model_name,
              frame_limit,
              prefetched_batches,
              framerate)
    print("Starting alphamerge")
    encoding_args = _alpha_encoding_args(output, alpha_codec, alpha_pix_fmt)
    cmd = [
        'ffmpeg', '-y', '-i', file_path, '-i', temp_file, '-i', overlay, '-filter_complex',
        '[1][0]scale2ref[mask][main];[main][mask]alphamerge[vid];[vid][2:v]scale2ref[fg][bg];[bg][fg]overlay[out]',
        '-map', '[out]', '-map', '2:a?', '-shortest'
    ]
    cmd += encoding_args + [output]
    sp.run(cmd)
    print("Process finished")
    try:
        temp_dir.cleanup()
    except PermissionError:
        pass
    return


def transparentvideooverimage(output, overlay, file_path,
                         worker_nodes,
                         gpu_batchsize,
                         model_name,
                         frame_limit=-1,
                         prefetched_batches=4,
                         framerate=-1,
                         alpha_codec="auto",
                         alpha_pix_fmt=None):
    temp_dir = tempfile.TemporaryDirectory()
    tmpdirname = Path(temp_dir.name)
    temp_file = os.path.abspath(os.path.join(tmpdirname, "matte.mp4"))
    matte_key(temp_file, file_path,
              worker_nodes,
              gpu_batchsize,
              model_name,
              frame_limit,
              prefetched_batches,
              framerate)
    print("Scale image")
    temp_image = os.path.abspath("%s/new.jpg" % tmpdirname)
    cmd = [
        'ffmpeg', '-y', '-i', overlay, '-i', file_path, '-filter_complex',
        'scale2ref[img][vid];[img]setsar=1;[vid]nullsink', '-q:v', '2', temp_image
    ]
    sp.run(cmd)
    print("Starting alphamerge")
    encoding_args = _alpha_encoding_args(output, alpha_codec, alpha_pix_fmt)
    cmd = [
        'ffmpeg', '-y', '-i', temp_image, '-i', file_path, '-i', temp_file, '-filter_complex',
        '[2][1]scale2ref[mask][main];[main][mask]alphamerge[fg];[0:v]scale2ref[bg][fg];[bg][fg]overlay=(W-w)/2:(H-h)/2:shortest=1[out]',
        '-map', '[out]', '-map', '1:a?', '-shortest'
    ]
    cmd += encoding_args + [output]
    sp.run(cmd)
    print("Process finished")
    try:
        temp_dir.cleanup()
    except PermissionError:
        pass
    return
