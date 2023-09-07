import utils
import sys
import glob
import os
from pathlib import Path
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_folder", action="store", help="extracted frames folder path", type=str)
    parser.add_argument("video_folder", action="store", help="original videos folder.", type=str)
    parser.add_argument("model", action="store", type=str, help="yolo model ckpt")
    parser.add_argument("save", action="store", type=str, help="save dir")
    parser.add_argument("--meta", action="store", dest="meta", default=None, type=str, help="additional info for search original video.")
    parser.add_argument("--cut", action="store", nargs=4, type=float, dest="cut", default=None)
    args = parser.parse_args()

    targets = glob.glob(f"{args.frames_folder}/**/????_??_??/", recursive=True)
    targets = sorted(targets)
    print(f"{len(targets)} frames folder searched.")

    if args.meta is None:
        try:
            with open(f"{args.frames_folder}/cam.txt", "r") as f:
                cam_name = f.readline().strip()
            
        except Exception as e:
            raise Exception("it required a original_videos_folder path's meta information : 'cam.txt'.")
    else:
        cam_name = args.meta
    
    try:
        import ultralytics
        ultralytics.checks()

        import torch
        torch.cuda.is_available()

        from ultralytics import YOLO
        model = YOLO(args.model)
    except Exception as e:
        raise Exception("failed to load YOLO lib / model.")

    videoes = []
    target_days = []
    print(f"start video / frames matching...", flush=True)
    for target in targets:
        target_day = target[utils.find(target, '/')[-2]+1:utils.find(target, '/')[-1]].split("_")
        target_video_sentense = f"{args.video_folder}/**/{target_day[0]}/{target_day[1]}/{target_day[2]}/{cam_name}/"
        video_folder = glob.glob(target_video_sentense, recursive=True)
        if len(video_folder) == 0:
            raise Exception(f"{target_video_sentense} : original video is not found.")
        if len(video_folder) >= 2:
            raise Exception(f"{target_video_sentense} : duplicate folders.")
        videoes.append(video_folder[0])
        target_days.append(target_day)

    savedir = args.save
    if not os.path.isdir(savedir):
        Path(savedir).mkdir(parents=True, exist_ok=True)

    for i in range(len(targets)):
        target_folder:str = targets[i]
        video_folder:str = videoes[i]
        target_day = target_days[i]
        print(f"{i+1}/{len(targets)} : target ({target_folder}), video ({video_folder}), day ({target_day})")
        #
        utils.detect_a_folder(
            path=target_folder,
            model=model,
            verbose=1,
            save=f"{savedir}/{'_'.join(target_day)}.csv",
            after_process=utils.extract_timing,
            video_folder=video_folder,
            valid_range=args.cut
        )

        
