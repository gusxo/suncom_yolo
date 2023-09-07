from utils import extract_frames, find
import sys
import pandas as pd
import numpy as np


df = pd.read_csv(
    "/home/user/mount2/index.csv",
    encoding="utf-8",
    index_col=0, 
    dtype={"shipname":"string", "datetime":"string", "date":"string", "video_sec":np.float32, "abspath":"string"})

#예외 처리 : 회의 내용 참조
#영생11호, 제11영생호는 제11영생호로 처리
df.loc[df.shipname == "영생11호", ["shipname"]] = "제11영생호"

ignore = pd.read_csv(
    "/home/user/mount2/ignore.csv",
    encoding="utf-8",
    index_col=0,
    dtype={"shipname":"string", "date":"string", "datetime":"string"}
)

#블랙리스트(csv) 기반으로 추출안할 영상 df에서 드랍
for i in range(ignore.shape[0]):
    row = ignore.iloc[i]
    shipname = row.shipname
    date = row.date
    datetime = row.datetime

    if date == "0" and datetime == "0":
        df = df.drop(df[df.shipname == shipname].index)
    
    elif datetime == "0":
        df = df.drop(df[(df.shipname == shipname) & (df.date == date)].index)

    elif date == "0":
        df = df.drop(df[(df.shipname == shipname) & (df.datetime == datetime)].index)


savedir = "/home/user/share/project4/"
savedir = "/home/user/mount2/frames/"

shipnames = pd.unique(df.shipname)
for ship in shipnames:
    dates = pd.unique(df[df.shipname == ship].date)
    for date in dates:
        targets = df[(df.shipname == ship) & (df.date == date)].sort_values(by="datetime").abspath
        save_target = f"{savedir}/{ship}/{date}/"
        extract_frames(targets, frame_distance=60.0, savedir=save_target, raise_exception=False, continued_video=False)