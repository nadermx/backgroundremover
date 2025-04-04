import argparse
import os
from distutils.util import strtobool
from .. import utilities
from ..bg import remove


def main():
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-m",
        "--model",
        default="u2net",
        type=str,
        choices=model_choices,
        help="The model name, u2net, u2netp, u2net_human_seg",
    )

    ap.add_argument(
        "-a",
        "--alpha-matting",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="When true use alpha matting cutout.",
    )

    ap.add_argument(
        "-af",
        "--alpha-matting-foreground-threshold",
        default=240,
        type=int,
        help="The trimap foreground threshold.",
    )

    ap.add_argument(
        "-ab",
        "--alpha-matting-background-threshold",
        default=10,
        type=int,
        help="The trimap background threshold.",
    )

    ap.add_argument(
        "-ae",
        "--alpha-matting-erode-size",
        default=10,
        type=int,
        help="Size of element used for the erosion.",
    )

    ap.add_argument(
        "-az",
        "--alpha-matting-base-size",
        default=1000,
        type=int,
        help="The image base size.",
    )
    ap.add_argument(
        "-wn",
        "--workernodes",
        default=1,
        type=int,
        help="Number of parallel workers"
    )

    ap.add_argument(
        "-gb",
        "--gpubatchsize",
        default=2,
        type=int,
        help="GPU batchsize"
    )

    ap.add_argument(
        "-fr",
        "--framerate",
        default=-1,
        type=int,
        help="Override the frame rate"
    )

    ap.add_argument(
        "-fl",
        "--framelimit",
        default=-1,
        type=int,
        help="Limit the number of frames to process for quick testing.",
    )
    ap.add_argument(
        "-mk",
        "--mattekey",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Output the Matte key file",
    )
    ap.add_argument(
        "-tv",
        "--transparentvideo",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Output transparent video format mov",
    )

    ap.add_argument(
        "-tov",
        "--transparentvideoovervideo",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Overlay transparent video over another video",
    )
    ap.add_argument(
        "-toi",
        "--transparentvideooverimage",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Overlay transparent video over another image",
    )
    ap.add_argument(
        "-tg",
        "--transparentgif",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Make transparent gif from video",
    )
    ap.add_argument(
        "-tgwb",
        "--transparentgifwithbackground",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Make transparent background overlay a background image",
    )

    ap.add_argument(
        "-i",
        "--input",
        nargs="?",
        default="-",
        type=argparse.FileType("rb"),
        help="Path to the input video or image.",
    )

    ap.add_argument(
        "-bi",
        "--backgroundimage",
        nargs="?",
        default="-",
        type=argparse.FileType("rb"),
        help="Path to background image.",
    )

    ap.add_argument(
        "-bv",
        "--backgroundvideo",
        nargs="?",
        default="-",
        type=argparse.FileType("rb"),
        help="Path to background video.",
    )

    ap.add_argument(
        "-o",
        "--output",
        nargs="?",
        default="-",
        type=argparse.FileType("wb"),
        help="Path to the output",
    )

    ap.add_argument(
        "-if",
        "--input-folder",
        type=str,
        help="Path to a folder containing input videos or images.",
    )

    ap.add_argument(
        "-of",
        "--output-folder",
        type=str,
        help="Path to the output folder for processed files.",
    )

    args = ap.parse_args()

    def is_video_file(filename):
        return filename.lower().endswith((".mp4", ".mov", ".webm", ".ogg", ".gif"))

    def is_image_file(filename):
        return filename.lower().endswith((".jpg", ".jpeg", ".png"))

    if args.input_folder:
        input_folder = os.path.abspath(args.input_folder)
        output_folder = os.path.abspath(args.output_folder or input_folder)
        os.makedirs(output_folder, exist_ok=True)

        files = [f for f in os.listdir(input_folder) if is_video_file(f) or is_image_file(f)]

        for f in files:
            input_path = os.path.join(input_folder, f)
            output_path = os.path.join(output_folder, f"output_{f}")

            if is_video_file(f):
                if args.mattekey:
                    utilities.matte_key(output_path, input_path,
                                        worker_nodes=args.workernodes,
                                        gpu_batchsize=args.gpubatchsize,
                                        model_name=args.model,
                                        frame_limit=args.framelimit,
                                        framerate=args.framerate)
                elif args.transparentvideo:
                    utilities.transparentvideo(output_path, input_path,
                                               worker_nodes=args.workernodes,
                                               gpu_batchsize=args.gpubatchsize,
                                               model_name=args.model,
                                               frame_limit=args.framelimit,
                                               framerate=args.framerate)
                elif args.transparentvideoovervideo:
                    utilities.transparentvideoovervideo(output_path, os.path.abspath(args.backgroundvideo.name),
                                                        input_path,
                                                        worker_nodes=args.workernodes,
                                                        gpu_batchsize=args.gpubatchsize,
                                                        model_name=args.model,
                                                        frame_limit=args.framelimit,
                                                        framerate=args.framerate)
                elif args.transparentvideooverimage:
                    utilities.transparentvideooverimage(output_path, os.path.abspath(args.backgroundimage.name),
                                                        input_path,
                                                        worker_nodes=args.workernodes,
                                                        gpu_batchsize=args.gpubatchsize,
                                                        model_name=args.model,
                                                        frame_limit=args.framelimit,
                                                        framerate=args.framerate)
                elif args.transparentgif:
                    utilities.transparentgif(output_path, input_path,
                                             worker_nodes=args.workernodes,
                                             gpu_batchsize=args.gpubatchsize,
                                             model_name=args.model,
                                             frame_limit=args.framelimit,
                                             framerate=args.framerate)
                elif args.transparentgifwithbackground:
                    utilities.transparentgifwithbackground(output_path, os.path.abspath(args.backgroundimage.name), input_path,
                                                           worker_nodes=args.workernodes,
                                                           gpu_batchsize=args.gpubatchsize,
                                                           model_name=args.model,
                                                           frame_limit=args.framelimit,
                                                           framerate=args.framerate)
            elif is_image_file(f):
                with open(input_path, "rb") as i, open(output_path, "wb") as o:
                    r = lambda i: i.buffer.read() if hasattr(i, "buffer") else i.read()
                    w = lambda o, data: o.buffer.write(data) if hasattr(o, "buffer") else o.write(data)
                    w(
                        o,
                        remove(
                            r(i),
                            model_name=args.model,
                            alpha_matting=args.alpha_matting,
                            alpha_matting_foreground_threshold=args.alpha_matting_foreground_threshold,
                            alpha_matting_background_threshold=args.alpha_matting_background_threshold,
                            alpha_matting_erode_structure_size=args.alpha_matting_erode_size,
                            alpha_matting_base_size=args.alpha_matting_base_size,
                        ),
                    )
        return

    if args.input.name.rsplit('.', 1)[1] in ['mp4', 'mov', 'webm', 'ogg', 'gif']:
        if args.mattekey:
            utilities.matte_key(os.path.abspath(args.output.name), os.path.abspath(args.input.name),
                                worker_nodes=args.workernodes,
                                gpu_batchsize=args.gpubatchsize,
                                model_name=args.model,
                                frame_limit=args.framelimit,
                                framerate=args.framerate)
        elif args.transparentvideo:
            utilities.transparentvideo(os.path.abspath(args.output.name), os.path.abspath(args.input.name),
                                       worker_nodes=args.workernodes,
                                       gpu_batchsize=args.gpubatchsize,
                                       model_name=args.model,
                                       frame_limit=args.framelimit,
                                       framerate=args.framerate)
        elif args.transparentvideoovervideo:
            utilities.transparentvideoovervideo(os.path.abspath(args.output.name), os.path.abspath(args.backgroundvideo.name),
                                                os.path.abspath(args.input.name),
                                                worker_nodes=args.workernodes,
                                                gpu_batchsize=args.gpubatchsize,
                                                model_name=args.model,
                                                frame_limit=args.framelimit,
                                                framerate=args.framerate)
        elif args.transparentvideooverimage:
            utilities.transparentvideooverimage(os.path.abspath(args.output.name), os.path.abspath(args.backgroundimage.name),
                                                os.path.abspath(args.input.name),
                                                worker_nodes=args.workernodes,
                                                gpu_batchsize=args.gpubatchsize,
                                                model_name=args.model,
                                                frame_limit=args.framelimit,
                                                framerate=args.framerate)
        elif args.transparentgif:
            utilities.transparentgif(os.path.abspath(args.output.name), os.path.abspath(args.input.name),
                                     worker_nodes=args.workernodes,
                                     gpu_batchsize=args.gpubatchsize,
                                     model_name=args.model,
                                     frame_limit=args.framelimit,
                                     framerate=args.framerate)
        elif args.transparentgifwithbackground:
            utilities.transparentgifwithbackground(os.path.abspath(args.output.name), os.path.abspath(args.backgroundimage.name), os.path.abspath(args.input.name),
                                                   worker_nodes=args.workernodes,
                                                   gpu_batchsize=args.gpubatchsize,
                                                   model_name=args.model,
                                                   frame_limit=args.framelimit,
                                                   framerate=args.framerate)

    else:
        print(args.output.name)
        r = lambda i: i.buffer.read() if hasattr(i, "buffer") else i.read()
        w = lambda o, data: o.buffer.write(data) if hasattr(o, "buffer") else o.write(data)
        w(
            args.output,
            remove(
                r(args.input),
                model_name=args.model,
                alpha_matting=args.alpha_matting,
                alpha_matting_foreground_threshold=args.alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=args.alpha_matting_background_threshold,
                alpha_matting_erode_structure_size=args.alpha_matting_erode_size,
                alpha_matting_base_size=args.alpha_matting_base_size,
            ),
        )


if __name__ == "__main__":
    main()
