import cv2
import os
import glob
import tqdm
from typing import Union, Literal, Tuple, Callable
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import numpy as np

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def extract_frames(filepath:Union[str, list, tuple], 
                   frame_distance:float,
                   scale:float=1, 
                   savedir:str=None,
                   sortlist:bool=True,
                   verbose:Literal[0, 1, 2]=1,
                   raise_exception:bool=True,
                   cut_range:Tuple[Tuple[int, int], Tuple[int, int]]=None,
                   continued_video:bool=True,
                   )->list:
    """
    parameters
    ----------
    `filepath` :
        파일 경로들의 배열일 경우, 해당 파일들을 순서대로 불러와서 처리함.

        string일 경우, 해당 문자열로 검색(Glob Pattern)을 하여 검색된 파일들로 처리함.

    `frame_distance`:
        프레임을 몇 초마다 추출할 지 정의
    
    `scale`:
        이미지 크기 배율

    `savedir`:
        저장 경로(폴더), 없을시 저장안함.

    `sortlist`:
        `filepath`를 정렬할 것인지 여부.

    `verbose`:
        0은 출력 없음

        1은 매 파일마다 출력

        2는 매 파일마다 출력 및 진행바

    `raise_exception`:
        파일 읽기 실패시 에러 출력 여부.

        False일시 실패하여도 해당 파일만 스킵하고 계속 진행함.

    `cut_range`:
        ((x_start, x_end), (y_start, y_end)) 형태의 tuple.

        해당 범위만큼 이미지를 자름. `scale`보다 우선됨.

    return
    ------
    a tuple, (image_list, metadata_list)

    metadata는 (n, 3) 형태의 nested list.
    
    각 행은 image_list의 index에 대응됨.
    
    column 1은 image가 추출된 영상 이름, column 0은 추출된 프레임 위치, column 2는 추출된 영상의 `filepath`에서의 인덱스.
    """
    if isinstance(filepath, str):
        filepath = glob.glob(filepath)
    
    if sortlist:
        filepath = sorted(filepath)

    v = cv2.VideoCapture()
    results = []
    start = 0
    meta = []

    for fi, f in enumerate(filepath):
        try:
            if verbose == 1 or verbose == 2:
                print(f"{fi+1}/{len(filepath)} : open {f}...", end="\n")
            v = cv2.VideoCapture(f)
            if not v.isOpened():
                raise Exception(f"failed to open video : {f}")
            fps = v.get(cv2.CAP_PROP_FPS)
            length = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            fd = int(fps * frame_distance)

            width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH)) if cut_range is None else cut_range[0][1]-cut_range[0][0]
            height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cut_range is None else cut_range[1][1]-cut_range[1][0]

            #비디오가 연속됨이 보장 안되는 경우(continued_video == False) 무조건 0초부터 읽음
            start = int(fd * start) if continued_video else 0
            r = range(start, length, fd) if verbose != 2 else tqdm.trange(start, length, fd)

            filename_no_parents = f[find(f, '/')[-1]+1:]


            if verbose == 1 or verbose == 2:
                print(f"- video is {(length / fps):.1f} sec. first snapshot is {(start / fps):.1f} sec.", end=" ", flush=True)
            if verbose == 2:
                print(flush=True)

            for i in r:
                v.set(1, i)
                ret, image = v.read()
                if not ret:
                    break
                if cut_range:
                    image = image[cut_range[1][0]:cut_range[1][1], cut_range[0][0]:cut_range[0][1]]
                image = cv2.resize(image, (int(width * scale), int(height * scale)))
                results.append(image)
                #add metadata (frame_index, file_name, index_of_file_name_in_filepath)
                meta.append([i, filename_no_parents, fi])

            if verbose == 1:
                print(f"get {len(r)} frames", end="\n", flush=True)

            if start < length:
                #안읽은 프레임 갯수에 따라 다음 영상의 시작 프레임이 몇초여야 하는지 저장
                start = (length - start) / fd
                start = (1 - (start - int(start)))
                if start == 1.0:
                    start = 0
            else:
                #영상에서 프레임 0개를 저장한 경우에는 계산식이 살짝다름.
                start = (start - length) / fd
        except Exception as e:
            if raise_exception:
                if v.isOpened():
                    v.release()
                raise e
            #if raise_exception is False, ignore exception
            print(f"exception in {f}, ignore...", flush=True)
        finally:
            if v.isOpened():
                v.release()

    assert len(results) == len(meta)

    if savedir and len(results):
        if verbose == 1 or verbose==2:
            print(f"save to {savedir} {len(results)} images...", flush=True)
        if not os.path.isdir(savedir):
            Path(savedir).mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(results):
            cv2.imwrite(f"{savedir}/{i}.png", img)
        pd.DataFrame(meta).to_csv(f"{savedir}/meta.csv",header=["frame_num", "video_name", "video_index"], index=True, index_label="image_name")

    return results, meta

# # 2개이상 탐지시 1개만 허용하는 버전
# def detect_a_folder(
#         path:str,
#         model,
#         verbose:Literal[0,1]=1,
#         save:str=None,
#         after_process:Callable=None,
#         **kwargs
# ):
#     """
#     `after_process` : 
#         매개변수가 (x, **kwargs)인 함수,

#         `after_process(return_values, **kwargs)` 형태로 호출 되고, 결과값을 대신 반환 함.    
#     return
#     ------
#     update later

#     """

#     meta_file_path = f"{path}/meta.csv"
#     if verbose: print(f"open {meta_file_path}...", flush=True)
#     meta = pd.read_csv(meta_file_path)
#     results = []

#     if verbose: print(f"start predict...", end="", flush=True)
#     preds = model.predict(f"{path}/", stream=True)
    
#     for pred in preds:
#         #predict 순서가 오름차순인것이 보장되는지 모르므로, image 번호 추출함.
#         image_num = int(pred.path.split('/')[-1][:-4])
#         #1개라도 탐지되면 dataframe 업데이트
#         boxes = pred.boxes.cpu()
#         if len(boxes.cls):
#             argmax = int(np.argmax(boxes.conf))
#             conf = float(boxes.conf[argmax])
#             cls_name = pred.names[argmax]
#             results.append([*meta.iloc[image_num, :], cls_name, conf])
#         #미탐지시 meta 정보 재활용
#         else:
#             cls_name = ""
#             results.append([*meta.iloc[image_num, :], "", 0])

#     results = pd.DataFrame(results)
#     results.columns=["image_num", "frame_num", "video_name", "video_index", "yolo_class", "conf_score"]
#     results["image_num"] = results["image_num"].astype(int)
#     results.sort_values(by="image_num")
#     if verbose: print(f"\tdone.", end="\n", flush=True)

#     if after_process:
#         results = after_process(results, **kwargs)
#     if save:
#         results.to_csv(save, index=False)

#     return results


def detect_a_folder(
        path:str,
        model,
        verbose:Literal[0,1]=1,
        save:str=None,
        after_process:Callable=None,
        valid_range:Tuple[float, float, float, float]=None,
        **kwargs
):
    """
    `after_process` : 
        매개변수가 (x, **kwargs)인 함수,

        `after_process(return_values, **kwargs)` 형태로 호출 되고, 결과값을 대신 반환 함.   

    `valid_range` :
        yolov8 xyxy format

    return
    ------
    update later
    
    """

    meta_file_path = f"{path}/meta.csv"
    if verbose: print(f"open {meta_file_path}...", flush=True)
    meta = pd.read_csv(meta_file_path)
    results = []

    if verbose: print(f"start predict...", end="", flush=True)
    preds = model.predict(f"{path}/", stream=True)
    class_names = None

    if valid_range[2] == -1:
        valid_range = preds[0].orig_shape[1]
    if valid_range[3] == -1:
        valid_range = preds[0].orig_shape[0]
    
    for pred in preds:
        if class_names is None:
            class_names = list(map(lambda i : pred.names[i], range(len(pred.names))))   #모든 클래스명 배열로 추출

        #predict 순서가 오름차순인것이 보장되는지 모르므로, image 번호 추출함.
        image_num = int(pred.path.split('/')[-1][:-4])
        #1개라도 탐지되면 dataframe 업데이트
        boxes = pred.boxes.cpu()
        conf_array = [0] * (len(class_names))

        if len(boxes.cls):
            for cls, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
                if valid_range:
                    #must bounding box is contain in valid_range box.
                    if (xyxy[0] < valid_range[0]) or (xyxy[1] < valid_range[1]) or (xyxy[2] > valid_range[2]) or (xyxy[3] > valid_range[3]):
                        continue

                cls = int(cls)
                conf = float(conf)
                if conf_array[cls] < conf:
                    conf_array[cls] = conf
        results.append([*meta.iloc[image_num, :], *conf_array])

    results = pd.DataFrame(results)
    results.columns=["image_num", "frame_num", "video_name", "video_index", *class_names]
    results["image_num"] = results["image_num"].astype(int)
    results = results.sort_values(by="image_num")
    if verbose: print(f"\tdone.", end="\n", flush=True)

    if after_process:
        results = after_process(results, **kwargs)
    if save:
        results.to_csv(save, index=False)

    return results

def extract_timing(result:pd.DataFrame, video_folder:str, fps=None, custom_video_timing_func=None):
    hours = []
    mins = []
    secs = []
    video_fullpath = []
    def default_video_timing(s):
        # 비디오 이름이 ****YYYYMMDDhhmmss.mp4 이라고 가정함.
        # [시, 분, 초]로 추출.
        # 이름 패턴 다르면 직접 함수 하나 정의해서 넘겨줄것(custom_video_timing_func)
        return [int(s[-10:-8]), int(s[-8:-6]), int(s[-6:-4])]
    
    if fps is not None:
        video_fps = fps
    
    for i in range(result.shape[0]):
        video_target = result.iloc[i]["video_name"]
        frame_target = result.iloc[i]["frame_num"]
        h, m, s = custom_video_timing_func(video_target) if custom_video_timing_func else default_video_timing(video_target)

        if fps is None:
            #fps 명시(고정)안하면 모든 영상마다 fps 확인함.
            f = f"{video_folder}/{video_target}"
            v = cv2.VideoCapture(f)
            if not v.isOpened():
                raise Exception(f"failed to open video : {f}")
            video_fps = v.get(cv2.CAP_PROP_FPS)
            v.release()

        #소수점 단위의 초는 버림.
        t = h * 3600 + m * 60 + s + frame_target//video_fps
        
        hours.append(t//3600)
        mins.append((t - hours[-1] * 3600)//60)
        secs.append(t - hours[-1] * 3600 - mins[-1] * 60)
        video_fullpath.append(f"{video_folder}/{video_target}")

    result["hour"] = hours
    result["min"] = mins
    result["sec"] = secs
    result["video_abspath"] = video_fullpath

    return result
    
def get_duplicate_area_rate(xyxy1, xyxy2):
    duplicate_area = [max(xyxy1[0], xyxy2[0]), max(xyxy1[1], xyxy2[1]), min(xyxy1[2], xyxy2[2]), min(xyxy1[3], xyxy2[3])]
    if duplicate_area[2] - duplicate_area[0] < 0 or duplicate_area[3] - duplicate_area[1] < 0:
        return [0, 0]

    duplicate_area = (duplicate_area[2] - duplicate_area[0]) * (duplicate_area[3] - duplicate_area[1])

    area1 = (xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1])

    area2 = (xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1])

    return [duplicate_area / area1, duplicate_area / area2]

def yolo_cut_by_range(preds, target_class:int, x1, y1, x2, y2, allowed_duplicate_rate:1.0):
    """
    return is 3-nested list

    1D : list of preds

    2D : target objects of pred

    3D : [xyxy, conf_score] of target-object

    """
    result = []

    if x2 == -1:
        x2 = preds[0].orig_shape[1]
    if y2 == -1:
        y2 = preds[0].orig_shape[0]

    #Union-Find Algorithm
    def union_find(parent, x):
        if parent[x] == x:
            return x
        
        parent[x] = union_find(parent, parent[x])
        return parent[x]
        
    def union_union(parent, x, y):
        xp = union_find(parent, x)
        yp = union_find(parent, y)
        parent[yp] = xp

    for pred in preds:
        boxes = pred.boxes.cpu()
        tmp_result = []
        if len(boxes.cls):
            for cls, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
                if int(cls) == target_class:
                    #must bounding box is contain in valid_range box.
                    if (xyxy[0] < x1) or (xyxy[1] < y1) or (xyxy[2] > x2) or (xyxy[3] > y2):
                        continue
                    tmp_result.append([xyxy, conf])

        #make a group by duplicated_area using union-find
        parent = np.arange(len(tmp_result))
        for i in range(len(tmp_result)):
            for j in range(i+1, len(tmp_result)):
                rate = get_duplicate_area_rate(tmp_result[i][0], tmp_result[j][0])
                if rate[0] > allowed_duplicate_rate or rate[1] > allowed_duplicate_rate:
                    union_union(parent, i, j)
        
        #sort all groups
        line_groups = {}
        for i, p in enumerate(parent):
            if p in line_groups:
                line_groups[p].append(i)
            else:
                line_groups[p] = [i]
        
        #determind one box from each groups (remove duplicated detection)
        selected_tmp_result = []
        for key, item in line_groups.items():
            max_conf = 0
            max_index = None
            for tmp_result_index in item:
                xyxy, conf = tmp_result[tmp_result_index]
                if conf > max_conf:
                    max_conf = conf
                    max_index = tmp_result_index
            selected_tmp_result.append(tmp_result[max_index])

        result.append(selected_tmp_result)

    return result