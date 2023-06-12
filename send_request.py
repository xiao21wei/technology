import csv
import json

import pandas as pd
import yaml

import requests
import time

# 从secrets.yml文件中读取IP和PORT
with open('secrets.yml', 'r', encoding='utf-8') as f:
    file_content = f.read()
    content = yaml.load(file_content, yaml.FullLoader)
    IP = content["IP"]
    PORT = content["PORT"]


def get_nodeId():
    url = "http://" + IP + ":" + PORT + "/node/info"
    response = requests.get(url)
    nodeId_data = response.json()["data"]
    return nodeId_data


def get_equipment(nodeId):
    url = "http://" + IP + ":" + PORT + "/equipment/node/" + nodeId + "/info"
    response = requests.get(url)
    equipmentUuid_data = response.json()["data"]
    return equipmentUuid_data


def get_point(equipmentUuid):
    url = "http://" + IP + ":" + PORT + "/point/" + equipmentUuid + "/info"
    response = requests.get(url)
    point_data = response.json()["data"]
    return point_data


def get_data(equipmentUuid, pointId):
    url = "http://" + IP + ":" + PORT + "/trend/" + equipmentUuid + "/" + pointId + "/real_time"
    response = requests.get(url)
    if 'data' in response.json():
        data_data = response.json()["data"]
        return data_data
    else:
        return None


def get_real_time():
    nodeId_data = get_nodeId()
    for (i, nodeId) in enumerate(nodeId_data):
        equipmentUuid_data = get_equipment(nodeId["nodeId"])
        for (j, equipmentUuid) in enumerate(equipmentUuid_data):
            point_data = get_point(equipmentUuid["equipmentUuid"])
            for (k, point) in enumerate(point_data):
                data = get_data(equipmentUuid["equipmentUuid"], point["pointId"])
                if data is not None:
                    trendTime = data["trendTime"]
                    t = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime(trendTime / 1000))
                    equipmentName = data["equipmentName"]
                    name = equipmentName + "%%" + point["pointId"] + "%%" + t
                    f = open(name + ".txt", "w")
                    trendValue = json.dumps(data["trendValue"], ensure_ascii=False)
                    f.write(trendValue)
                    waveValue = json.dumps(data["waveValue"], ensure_ascii=False)
                    f.write(waveValue)
                    f.close()


def get_history(equipmentUuid, pointId, startTime, endTime):
    url = "http://" + IP + ":" + PORT + "/trend/" + equipmentUuid + "/" + pointId + "/" + startTime + "/" + endTime + "/info"
    print(url)

    print(url)
    response = requests.get(url)
    print(response.json())
    if 'data' in response.json():
        data_data = response.json()["data"]
        return data_data
    else:
        return None


def get_proctrend_history(equipmentUuid, pointId, startTime, endTime):
    url = "http://" + IP + ":" + PORT + "/proctrend/" + equipmentUuid + "/" + pointId + "/" + startTime + "/" + endTime + "/his"
    response = requests.get(url)
    if 'data' in response.json():
        data_data = response.json()["data"]
        return data_data
    else:
        return None


def load_data(start, end):
    points = get_point("b1161555a5cf4cb0f060a7442127b7b6")
    for point in points:
        data = get_history("b1161555a5cf4cb0f060a7442127b7b6", point["pointId"], start, end)
        if data is not None and len(data["trendInfo"]) != 0:
            trendInfo = data["trendInfo"]
            f = open(point["pointId"] + ".csv", "w", encoding="utf-8", newline="")
            csv_writer = csv.writer(f)
            csv_writer.writerow(["trendTime", "all", "one", "two", "three", "half", "res"])
            for trend in trendInfo:
                trendTime = trend["trendTime"]
                t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(trendTime / 1000))
                all = trend["trendValue"][0]["all"]
                one = trend["trendValue"][0]["one"]
                two = trend["trendValue"][0]["two"]
                three = trend["trendValue"][0]["three"]
                half = trend["trendValue"][0]["half"]
                res = trend["trendValue"][0]["res"]
                csv_writer.writerow([t, all, one, two, three, half, res])
            f.close()


def load_other_data(start, end):
    data = get_history("d5c48b2bc9c34875bc719c0c6300233a", "燃气发生器转速_Ng", start, end)
    if data is not None and len(data["trendInfo"]) != 0:
        trendInfo = data["trendInfo"]
        f = open("燃气发生器转速_Ng.csv", "a", encoding="utf-8", newline="")
        csv_writer = csv.writer(f)
        for trend in trendInfo:
            trendTime = trend["trendTime"]
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(trendTime / 1000))
            three = trend["trendValue"][0]["three"]
            csv_writer.writerow([t, three])
        f.close()


def load_proctrend_data(start, end):
    data = get_proctrend_history("d5c48b2bc9c34875bc719c0c6300233a", "燃气发生器转速_Ng", start, end)
    if data is not None and len(data["trendValues"]) != 0:
        trendValues = data["trendValues"]
        # "trendValues": {"2021-11-02 11:49:38.653": [12958.29],"2021-11-02 11:49:44.833": [12964.02]}
        # 将数据写入csv文件，分别为time和Ng
        f = open("Ng.csv", "a", encoding="utf-8", newline="")
        csv_writer = csv.writer(f)
        for trendValue in trendValues:
            t = trendValue
            Ng = trendValues[trendValue][0]
            csv_writer.writerow([t, Ng])
        f.close()


# 将Ng.csv和GenPCal.csv合并，写入Ng_GenPCal.csv文件
def merge_Ng_and_GenPCal():
    Ng = pd.read_csv("Ng.csv")
    GenPCal = pd.read_csv("GenPCal.csv")
    Ng_GenPCal = pd.merge(Ng, GenPCal, on="time", how="left")
    Ng_GenPCal.to_csv("Ng_GenPCal.csv", index=False, sep=',')


if __name__ == '__main__':
    # 开始时间为2021-11-02 10:09:37.727
    # 结束时间为八天后
    # start = 1635824977727
    # # 2021-11-04 15:02:32.460
    # end = 1636047752460
    # left = start
    # right = start + 5 * 60 * 1000
    # print(left, right)
    # f = open("Ng.csv", "a", encoding="utf-8", newline="")
    # csv_writer = csv.writer(f)
    # csv_writer.writerow(["time", "Ng"])
    # f.close()
    # while left < end:
    #     load_proctrend_data(str(left), str(right))
    #     left = right
    #     right = right + 5 * 60 * 1000
    #     if right > end:
    #         right = end
    # # 输出Ng.csv文件的行数
    # f = open("Ng.csv", "r", encoding="utf-8")
    # reader = csv.reader(f)
    # print(len(list(reader)))
    # f.close()

    # data = get_proctrend_history("d5c48b2bc9c34875bc719c0c6300233a", "发电功率GenPCal", str(left), str(right))
    # # 输出data["trendValues"]的长度
    # print(len(data["trendValues"]))

    merge_Ng_and_GenPCal()
