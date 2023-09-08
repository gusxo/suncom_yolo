import utils
import sys
import glob
import os
from pathlib import Path
import argparse
import cv2
from PIL import Image
import matplotlib.pyplot as plt

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", action="store", help="target video pathes", type=str)
    parser.add_argument("model", action="store", type=str, help="yolo model ckpt")
    parser.add_argument("save", action="store", type=str, help="save dir")
    parser.add_argument("cut", action="store", nargs=4, type=int, help="box range for line tracking")
    parser.add_argument("--save_images", action="store_true", dest="save_images", help="save all images from througth process")
    parser.add_argument("--dup_rate", action="store", type=float, dest="dup_rate", help="allow duplicate area rate", default=0.8)
    parser.add_argument("--tracking_limit", action="store", type=int, dest="tracking_limit", help="limit count for for un-tracking", default=1)
    parser.add_argument("--tracking_dup_rate", action="store", type=float, dest="tracking_dup_rate", help="duplicate area rate for determind of tracking", default=0.5)
    parser.add_argument("--frame_distance", action="store", type=float, dest="frame_distance", help="determind each frames time distance", default=0.2)
    parser.add_argument("--save_only_track", action="store_true", dest="save_only_track", help="save tracking images.")
    args = parser.parse_args()

    print(f"your arguments is : {args}")

    try:
        import ultralytics
        ultralytics.checks()

        import torch
        torch.cuda.is_available()

        from ultralytics import YOLO
        model = YOLO(args.model)
    except Exception as e:
        raise Exception("failed to load YOLO lib / model.")
    
    save_root = f"{args.save}/{Path(os.path.abspath(args.video)).stem}"
    
    images, meta = utils.extract_frames([args.video], frame_distance=args.frame_distance, sortlist=False, verbose=1, raise_exception=True, continued_video=False)

    if args.save_images:
        savedir = f"{save_root}/target_images/"
        if not os.path.isdir(savedir):
            Path(savedir).mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            cv2.imwrite(f"{savedir}/{i}.png", img)
        
    preds = model.predict(source=images)

    if args.save_images:
        savedir = f"{save_root}/detect_images/"
        if not os.path.isdir(savedir):
            Path(savedir).mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(preds):
            im_array = img.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.save(f"{savedir}/{i}.png")  # save image

    # names: {0: 'line', 1: 'sardine', 2: 'saurel', 3: 'squid'}
    lines = utils.yolo_cut_by_range(preds, 0, *args.cut, args.dup_rate)

    if args.save_images:
        savedir = f"{save_root}/selected_lines/"
        if not os.path.isdir(savedir):
            Path(savedir).mkdir(parents=True, exist_ok=True)
        for i, (img, line_info) in enumerate(zip(images, lines)):
            img = cv2.rectangle(img, (args.cut[0], args.cut[1]), ((img.shape[1] if args.cut[2] == -1 else args.cut[2]), (img.shape[0] if args.cut[3] == -1 else args.cut[3])), (0, 0, 255), 3)
            for xyxy, conf in line_info:
                img = cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 255), 3)
            cv2.imwrite(f"{savedir}/{i}.png", img)

    track_limit:int = args.tracking_limit
    track_dup_rate:float = args.tracking_dup_rate





    