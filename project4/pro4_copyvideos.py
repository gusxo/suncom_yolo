from utils import extract_frames, find
import sys
import pandas as pd
import numpy as np

df = pd.read_csv(
    "/home/user/mount2/index.csv",
    encoding="utf-8",
    index_col=0, 
    dtype={"shipname":"string", "datetime":"string", "date":"string", "video_sec":np.float32})

#예외 처리 : 회의 내용 참조
#datetime 형식 안지켜진 재성호 파일들 처리안함
#영생11호, 제11영생호는 제11영생호로 처리
df.loc[df["shipname"] == "영생11호", ["shipname"]] = "제11영생호"
df = df[df["shipname"] != "성호"]

name = "제1창경호"
date = "2023_04_14"
a = df[(df["shipname"] == name) & (df["date"]==date)]
from shutil import copyfile
for i in range(a.shape[0]):
    abspath = a.iloc[i]["abspath"]
    print(f"{i+1}/{a.shape[0]} : {abspath}", flush=True)
    copyfile(abspath, f"/home/user/share/tmp/{name}/{a.iloc[i]['datetime']}.mp4")