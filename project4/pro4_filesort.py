from utils import find
import glob
import pandas as pd
import numpy as np
import cv2

targets = glob.glob("/home/user/mount2/**/*.mp4", recursive=True)
savedir = f"/home/user/mount2/index.csv"
result = []

print(f"{len(targets)} target video searched.")

for i, target in enumerate(targets):
    print(f"{i+1}/{len(targets)} : {target}", flush=True, end=" ... ")
    filename = target[find(target, "/")[-1]+1:]
    date_start_idx = 1

    if len(filename) <= 18:
        print(f"drop : file name length <= 18")
        continue

    if filename[0] != "f":
        print(f"drop : file name is not start with 'f'")
        continue

    if filename[1] == "c":
        date_start_idx = 2
    elif ord(filename[1]) < ord("0") or ord(filename[1]) > ord("9"):
        print(f"drop : file name is not start with 'fc'")
        continue

    date = f"{filename[date_start_idx:date_start_idx+4]}_{filename[date_start_idx+4:date_start_idx+6]}_{filename[date_start_idx+6:date_start_idx+8]}"
    try:
        map(int, date.split("_"))
    except Exception as e:
        print(f"drop : failed to extract date")
        continue

    shipname = f"{filename[date_start_idx+14:-4]}"
    date_fulltext = f"{filename[date_start_idx:date_start_idx+14]}"

    if len(shipname) <= 0:
        print(f"drop : shipname len 0")
        continue

    try:
        v = cv2.VideoCapture(target)
        fps = v.get(cv2.CAP_PROP_FPS)
        length = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
        sec = f"{length/fps:.1f}"
        v.release()
    except Exception as e:
        if v.isOpened():
            v.release()
        print(f"drop : failed to open video")
        continue

    result.append([date_fulltext, date, shipname, target, sec])
    print(f"check.")
    
#remove duplicated file by date_fulltext(datetime), and save it
df = pd.DataFrame(result)
df.columns = ["datetime", "date", "shipname", "abspath", "video_sec"]
shipnames = np.unique(df["shipname"])
res_df = pd.DataFrame(np.empty((0, len(df.columns))))
for shipname in shipnames:
    ship_data = np.array(df[df["shipname"] == shipname], object)
    res_df = pd.concat([res_df, pd.DataFrame(ship_data[np.unique(ship_data[:,0], return_index=1)[1]])], axis=0, ignore_index=True)

print(f"result csv rows : {res_df.shape[0]}")
print(f"{len(result) - res_df.shape[0]} rows drop to removal duplicate.")

res_df.columns = df.columns
res_df = res_df.sort_values(by=["shipname", "datetime"])
res_df.to_csv(f"{savedir}", index=True)

