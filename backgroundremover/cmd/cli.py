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
        "-om",
        "--only-mask",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Output only the binary mask (grayscale image).",
    )

    ap.add_argument(
        "-bc",
        "--background-color",
        type=str,
        default=None,
        help="Background color as RGB tuple, e.g., '255,0,0' for red or '0,255,0' for green.",
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
        help="Output a matte key video (black/white mask for video editing). For transparent video use -tv instead.",
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
        "--alpha-codec",
        default="auto",
        type=str,
        help="Codec for transparent video output (auto, prores_ks, qtrle, libvpx-vp9). Auto defaults to lossless qtrle.",
    )
    ap.add_argument(
        "--alpha-pix-fmt",
        default=None,
        type=str,
        help="Override pixel format for transparent video output (e.g., yuva444p10le).",
    )

    ap.add_argument(
        "-tov",
        "--transparentvideoovervideo",
        action="store_true",
        default=False,
        help="Overlay transparent video over another video (use -bv to specify background video)",
    )
    ap.add_argument(
        "-toi",
        "--transparentvideooverimage",
        action="store_true",
        default=False,
        help="Overlay transparent video over another image (use -bi to specify background image)",
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
        action="store_true",
        default=False,
        help="Make transparent background overlay a background image (use -bv to specify background video)",
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
        default=None,
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

    # Validate that -toi and -tov have their required background arguments
    if args.transparentvideooverimage and (not args.backgroundimage or args.backgroundimage.name == "<stdin>"):
        print("Error: -toi/--transparentvideooverimage requires -bi/--backgroundimage to specify the background image.")
        print("Example: backgroundremover -i video.mp4 -toi -bi background.png -o output.mov")
        exit(1)

    if args.transparentvideoovervideo and (not args.backgroundvideo or args.backgroundvideo.name == "<stdin>"):
        print("Error: -tov/--transparentvideoovervideo requires -bv/--backgroundvideo to specify the background video.")
        print("Example: backgroundremover -i video.mp4 -tov -bv background.mp4 -o output.mov")
        exit(1)

    if args.transparentgifwithbackground and (not args.backgroundimage or args.backgroundimage.name == "<stdin>"):
        print("Error: -tgwb/--transparentgifwithbackground requires -bi/--backgroundimage to specify the background image.")
        print("Example: backgroundremover -i video.mp4 -tgwb -bi background.png -o output.gif")
        exit(1)

    # Warn about high worker counts that may cause issues
    if args.workernodes > 4:
        print(f"Warning: Using {args.workernodes} workers. High worker counts (>4) may cause ConnectionResetError or crashes on some systems.")
        print("If you experience errors, try reducing workers with -wn 1 or -wn 2")

    # Parse background color if provided
    background_color = None
    if args.background_color:
        try:
            rgb_values = tuple(int(x.strip()) for x in args.background_color.split(','))
            if len(rgb_values) != 3 or not all(0 <= v <= 255 for v in rgb_values):
                raise ValueError("RGB values must be between 0 and 255")
            background_color = rgb_values
        except Exception as e:
            print(f"Invalid background color format. Use format '255,0,0' for red. Error: {e}")
            exit(1)

    # Read background image if provided
    background_image = None
    if args.backgroundimage and args.backgroundimage.name != "-":
        r = lambda i: i.buffer.read() if hasattr(i, "buffer") else i.read()
        background_image = r(args.backgroundimage)

    def is_video_file(filename):
        return filename.lower().endswith((".mp4", ".mov", ".webm", ".ogg", ".gif"))

    def is_image_file(filename):
        return filename.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".heif"))

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
                                               framerate=args.framerate,
                                               alpha_codec=args.alpha_codec,
                                               alpha_pix_fmt=args.alpha_pix_fmt)
                elif args.transparentvideoovervideo:
                    utilities.transparentvideoovervideo(output_path, os.path.abspath(args.backgroundvideo.name),
                                                        input_path,
                                                        worker_nodes=args.workernodes,
                                                        gpu_batchsize=args.gpubatchsize,
                                                        model_name=args.model,
                                                        frame_limit=args.framelimit,
                                                        framerate=args.framerate,
                                                        alpha_codec=args.alpha_codec,
                                                        alpha_pix_fmt=args.alpha_pix_fmt)
                elif args.transparentvideooverimage:
                    utilities.transparentvideooverimage(output_path, os.path.abspath(args.backgroundimage.name),
                                                        input_path,
                                                        worker_nodes=args.workernodes,
                                                        gpu_batchsize=args.gpubatchsize,
                                                        model_name=args.model,
                                                        frame_limit=args.framelimit,
                                                        framerate=args.framerate,
                                                        alpha_codec=args.alpha_codec,
                                                        alpha_pix_fmt=args.alpha_pix_fmt)
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
                            only_mask=args.only_mask,
                            background_color=background_color,
                            background_image=background_image,
                        ),
                    )
        return

    # Handle stdin/stdout pipe support
    # When using pipes, we assume image input unless file extension is detected
    if args.input.name == "<stdin>" or args.output.name == "<stdout>":
        # Pipe mode - assume image processing
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
                only_mask=args.only_mask,
                background_color=background_color,
                background_image=background_image,
            ),
        )
        return

    ext = os.path.splitext(args.input.name)[1].lower()

    if ext in [".mp4", ".mov", ".webm", ".ogg", ".gif"]:
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
                                       framerate=args.framerate,
                                       alpha_codec=args.alpha_codec,
                                       alpha_pix_fmt=args.alpha_pix_fmt)
        elif args.transparentvideoovervideo:
            utilities.transparentvideoovervideo(os.path.abspath(args.output.name), os.path.abspath(args.backgroundvideo.name),
                                                os.path.abspath(args.input.name),
                                                worker_nodes=args.workernodes,
                                                gpu_batchsize=args.gpubatchsize,
                                                model_name=args.model,
                                                frame_limit=args.framelimit,
                                                framerate=args.framerate,
                                                alpha_codec=args.alpha_codec,
                                                alpha_pix_fmt=args.alpha_pix_fmt)
        elif args.transparentvideooverimage:
            utilities.transparentvideooverimage(os.path.abspath(args.output.name), os.path.abspath(args.backgroundimage.name),
                                                os.path.abspath(args.input.name),
                                                worker_nodes=args.workernodes,
                                                gpu_batchsize=args.gpubatchsize,
                                                model_name=args.model,
                                                frame_limit=args.framelimit,
                                                framerate=args.framerate,
                                                alpha_codec=args.alpha_codec,
                                                alpha_pix_fmt=args.alpha_pix_fmt)
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

    elif ext in [".jpg", ".jpeg", ".png", ".heic", ".heif"]:
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
                only_mask=args.only_mask,
                background_color=background_color,
                background_image=background_image,
            ),
        )
    else:
        print(f"❌ Unsupported file type: {ext}")
        print(f"Supported image formats: .jpg, .jpeg, .png, .heic, .heif")
        print(f"Supported video formats: .mp4, .mov, .webm, .ogg, .gif")
        exit(1)


if __name__ == "__main__":
    main()
