from utils import extract_frames, find
import sys
import glob

if sys.argv[1] == "1":
    #배 1번에 대한 설정
    targets = glob.glob("/home/user/mount1/620동원/**/CAM4/", recursive=True)
    shipname = "ship1"

if sys.argv[1] == "2":
    # 배 2번에 대한 설정
    targets = glob.glob("/home/user/mount1/722오룡/**/CAM4/", recursive=True)
    shipname = "ship2"

if sys.argv[1] == "3":
    # 배 3번에 대한 설정
    targets = glob.glob("/home/user/mount1/723오룡/**/CAM3/", recursive=True)
    shipname = "ship3"

if sys.argv[1] == "4":
    # 배 4번에 대한 설정
    targets = glob.glob("/home/user/mount1/토니나3/**/CAM3/", recursive=True)
    shipname = "ship4"

print(f"{len(targets)} target folders searched.")
i = 0

for target in targets:
    i += 1
    print(f"[{i}/{len(targets)}]")
    print(f"target : {target}")
    sep = find(target, '/')
    savedir = f"/home/user/project3/{shipname}/{target[sep[-5]+1:sep[-2]].replace('/', '_')}/"
    print(savedir)
    extract_frames(f"{target}/*", frame_distance=60, scale=1, savedir=savedir, raise_exception=False, verbose=1)