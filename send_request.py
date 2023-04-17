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
    print(response.json())
    if 'data' in response.json():
        data_data = response.json()["data"]
        return data_data
    else:
        return None


if __name__ == '__main__':
    # 获取当前时间的UNIX毫秒时间戳和一周前的UNIX毫秒时间戳
    now = int(time.time() * 1000)
    week_ago = now - 7 * 24 * 60 * 60 * 1000
    # 将UNIX毫秒时间戳转换为字符串
    now = str(now)
    week_ago = str(week_ago)
    print(week_ago)
    print(now)

    points = get_point("b1161555a5cf4cb0f060a7442127b7b6")
    for point in points:
        data = get_history("b1161555a5cf4cb0f060a7442127b7b6", point["pointId"], week_ago, now)
        if data is not None and len(data["trendInfo"]) != 0:
            print(data)

