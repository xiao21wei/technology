import json

import requests
import time

IP = "192.168.119.161"
PORT = "9091"


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


def get_history(equipmentUuid, pointId, startTime, endTime):
    url = "http://" + IP + ":" + PORT + "/trend/" + equipmentUuid + "/" + pointId + "/" + str(startTime) + "/" + str(endTime) + "/info"
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


if __name__ == '__main__':
    get_real_time()
