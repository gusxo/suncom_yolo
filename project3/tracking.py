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
    parser.add_argument("--dup_rate", action="store", type=float, dest="dup_rate", help="allow duplicate area rate", default=0.6)
    parser.add_argument("--tracking_limit", action="store", type=int, dest="tracking_limit", help="limit count for for un-tracking", default=1)
    parser.add_argument("--tracking_dup_rate", action="store", type=float, dest="tracking_dup_rate", help="duplicate area rate for determind of tracking", default=0.6)
    parser.add_argument("--frame_distance", action="store", type=float, dest="frame_distance", help="determind each frames time distance", default=0.2)
    parser.add_argument("--save_only_track", action="store_true", dest="save_only_track", help="save tracking images.")
    parser.add_argument("--save_video", action="store_true", dest="save_video", help="save tracking images by video.")
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
        
    preds = model.predict(source=images, stream=True)

    if args.save_images:
        savedir = f"{save_root}/detect_images/"
        if not os.path.isdir(savedir):
            Path(savedir).mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(preds):
            im_array = img.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.save(f"{savedir}/{i}.png")  # save image

    print(f"start range check & duplicate removal...", flush=True, end="")
    # names: {0: 'line', 1: 'sardine', 2: 'saurel', 3: 'squid'}
    detected_lines = utils.yolo_cut_by_range(preds, 0, *args.cut, args.dup_rate)
    print(f"\tdone.", flush=True)

    if args.save_images:
        savedir = f"{save_root}/selected_lines/"
        if not os.path.isdir(savedir):
            Path(savedir).mkdir(parents=True, exist_ok=True)
        for i, (img, line_info) in enumerate(zip(images, detected_lines)):
            img = cv2.rectangle(img, (args.cut[0], args.cut[1]), ((img.shape[1] if args.cut[2] == -1 else args.cut[2]), (img.shape[0] if args.cut[3] == -1 else args.cut[3])), (0, 0, 255), 3)
            for xyxy, conf in line_info:
                img = cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 255), 3)
            cv2.imwrite(f"{savedir}/{i}.png", img)

    track_limit:int = args.tracking_limit
    track_dup_rate:float = args.tracking_dup_rate

    print(f"start tracking ...", flush=True, end="")
    track_lines = utils.yolo_tracking(detected_lines, track_dup_rate, track_limit)
    print(f"\ttrack result : {len(track_lines)} line objects.")

    if args.save_only_track or args.save_images or args.save_video:
        savedir = f"{save_root}/tracking/"
        if not os.path.isdir(savedir):
            Path(savedir).mkdir(parents=True, exist_ok=True)
        track_images = images
        for i in range(len(images)):
            track_images[i] = cv2.rectangle(track_images[i], (args.cut[0], args.cut[1]), ((track_images[i].shape[1] if args.cut[2] == -1 else args.cut[2]), (track_images[i].shape[0] if args.cut[3] == -1 else args.cut[3])), (0, 0, 255), 3)

        colormap = plt.get_cmap("hsv")
        cmapfunc = lambda x : list(map(lambda y : int(y * 255), colormap((x << 4)%255)[-2::-1]))

        for track_id, line_obj in enumerate(track_lines):
            track_start_index = line_obj[0]
            track_end_index = line_obj[1]
            for i, box in enumerate(line_obj[2:3 + (track_end_index - track_start_index)]):
                track_images[track_start_index + i] = cv2.rectangle(track_images[track_start_index + i], 
                                                                    (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                                                    cmapfunc(track_id), 3)
        
        if args.save_only_track or args.save_images:
            for i, img in enumerate(track_images):
                cv2.imwrite(f"{savedir}/{i}.png", img)
        
        if args.save_video:
            video = cv2.VideoWriter(f"{savedir}/video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), args.frame_distance, track_images[0].shape[1::-1], True)
            for img in track_images:
                video.write(img)
            cv2.destroyAllWindows()
            video.release()


    