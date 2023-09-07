import glob
import argparse
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

if __name__=="__main__":

    #<program> <target csv dir> <result dir>
    target = sys.argv[1]

    savedir = sys.argv[2]

    if not os.path.isdir(savedir):
        Path(savedir).mkdir(parents=True, exist_ok=True)

    target_csv_list = glob.glob(f"{target}/*.csv")
    result = []
    for target_csv in target_csv_list:
        data = pd.read_csv(target_csv)
        date = target_csv[-14:-4]
        y = int(date[0:4])
        m = int(date[5:7])
        d = int(date[8:10])
        bigeye = data["BigeyeTuna"]
        yellowfin = data["YellowfinTuna"]
        
        bigeye_cnt = 0
        bigeye_flag = 0
        yellowfin_cnt = 0
        yellowfin_flag = 0
        for i in range(data.shape[0]):
            B = (bigeye[i] != 0)
            Y = (yellowfin[i] != 0)
            if B and Y:
                if bigeye[i] > yellowfin[i]:
                    Y = False
                else:
                    B = False

            if not B:
                if bigeye_flag:
                    bigeye_flag = 0
                    bigeye_cnt += 1
            else:
                bigeye_flag = 1

            if not Y:
                if yellowfin_flag:
                    yellowfin_flag = 0
                    yellowfin_cnt += 1
            else:
                yellowfin_flag = 1
            
        bigeye_cnt += bigeye_flag
        yellowfin_cnt += yellowfin_flag

        result.append([y, m, d, bigeye_cnt, yellowfin_cnt])
    result = pd.DataFrame(result)
    result.columns=["year", "month", "day", "BigeyeTuna", "YellowfinTuna"]
    result = result.sort_values(by=["year", "month", "day"])

    result.to_csv(f"{savedir}/day.csv", index=False)

    res2 = result.groupby(by=["year", "month"]).sum()[["BigeyeTuna","YellowfinTuna"]]
    res2.to_csv(f"{savedir}/month.csv", index=True)

