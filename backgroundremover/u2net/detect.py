import errno
import os
import sys
import numpy as np
import torch
from hsh.library.hash import Hasher
from PIL import Image
from torchvision import transforms

from . import data_loader, u2net
from .. import github


def load_model(model_name: str = "u2net"):
    hasher = Hasher()

    model = {
        'u2netp': (u2net.U2NETP,
                   'e4f636406ca4e2af789941e7f139ee2e',
                   '1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy',
                   'U2NET_PATH'),
        'u2net': (u2net.U2NET,
                  '09fb4e49b7f785c9f855baf94916840a',
                  '1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ',
                  'U2NET_PATH'),
        'u2net_human_seg': (u2net.U2NET,
                            '347c3d51b01528e5c6c071e3cff1cb55',
                            '1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P',
                            'U2NET_PATH')
    }[model_name]

    if model_name == "u2netp":
        net = u2net.U2NETP(3, 1)
        path = os.environ.get(
            "U2NETP_PATH",
            os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
        )
        if (
            not os.path.exists(path)
            #or hasher.md5(path) != "e4f636406ca4e2af789941e7f139ee2e"
        ):
            github.download_files_from_github(
                path, model_name
            )

    elif model_name == "u2net":
        net = u2net.U2NET(3, 1)
        path = os.environ.get(
            "U2NET_PATH",
            os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
        )

        print(f"DEBUG: path to be checked: {path}")

        if (
            not os.path.exists(path)
            #or hasher.md5(path) != "09fb4e49b7f785c9f855baf94916840a"
        ):
            github.download_files_from_github(
                path, model_name
            )

    elif model_name == "u2net_human_seg":
        net = u2net.U2NET(3, 1)
        path = os.environ.get(
            "U2NET_PATH",
            os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
        )
        if (
            not os.path.exists(path)
            #or hasher.md5(path) != "347c3d51b01528e5c6c071e3cff1cb55"
        ):
            github.download_files_from_github(
                path, model_name
            )

    else:
        print("Choose between u2net, u2net_human_seg or u2netp", file=sys.stderr)

    try:
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(path))
            net.to(torch.device("cuda"))
        else:
            net.load_state_dict(
                torch.load(
                    path,
                    map_location="cpu",
                )
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), model_name + ".pth"
        )

    net.eval()

    return net


def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)

    return dn


def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3

    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose(
        [data_loader.RescaleT(320), data_loader.ToTensorLab(flag=0)]
    )
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})

    return sample


def predict(net, item):
    sample = preprocess(item)

    with torch.no_grad():

        if torch.cuda.is_available():
            inputs_test = torch.cuda.FloatTensor(
                sample["image"].unsqueeze(0).cuda().float()
            )
        else:
            inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        pred = d1[:, 0, :, :]
        predict = norm_pred(pred)

        predict = predict.squeeze()
        predict_np = predict.cpu().detach().numpy()
        img = Image.fromarray(predict_np * 255).convert("RGB")

        del d1, d2, d3, d4, d5, d6, d7, pred, predict, predict_np, inputs_test, sample
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return img
