import csv
import json
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


def load_three_data(start, end):
    data = get_history("b1161555a5cf4cb0f060a7442127b7b6", "cs4", start, end)
    if data is not None and len(data["trendInfo"]) != 0:
        trendInfo = data["trendInfo"]
        f = open("cs4.csv", "a", encoding="utf-8", newline="")
        csv_writer = csv.writer(f)
        for trend in trendInfo:
            trendTime = trend["trendTime"]
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(trendTime / 1000))
            three = trend["trendValue"][0]["three"]
            csv_writer.writerow([t, three])
        f.close()


if __name__ == '__main__':
    # 获取当前时间的UNIX毫秒时间戳和一周前的UNIX毫秒时间戳
    start = 1681885246000
    end = 1683184292000
    left = start
    right = start + 40 * 60 * 1000
    left = str(left)
    right = str(right)
    f = open("cs4.csv", "a", encoding="utf-8", newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["trendTime", "three"])
    f.close()
    while int(left) < end:
        left = int(right)
        right = left + 40 * 60 * 1000
        left = str(left)
        right = str(right)
        load_three_data(left, right)
