import json
import time

from send_request import get_nodeId, get_equipment, get_point, get_data, get_history

IP = "192.168.119.161"
PORT = "9091"


def real_time(pointUuid):
    nodeId_data = get_nodeId()
    for (i, nodeId) in enumerate(nodeId_data):
        equipmentUuid_data = get_equipment(nodeId["nodeId"])
        for (j, equipmentUuid) in enumerate(equipmentUuid_data):
            point_data = get_point(equipmentUuid["equipmentUuid"])
            for (k, point) in enumerate(point_data):
                if point["pointUuid"] == pointUuid:
                    print(equipmentUuid["equipmentUuid"], point["pointId"])
                    data = get_data(equipmentUuid["equipmentUuid"], point["pointId"])
                    if data is not None:
                        trendTime = data["trendTime"]
                        t = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime(trendTime / 1000))
                        name = data["equipmentName"] + "%%" + point["pointId"] + "%%" + t
                        f = open(name + ".txt", "w")
                        trendValue = json.dumps(data["trendValue"], ensure_ascii=False)
                        f.write(trendValue)
                        waveValue = json.dumps(data["waveValue"], ensure_ascii=False)
                        f.write(waveValue)
                        f.close()


def info(pointUuid, startTime, endTime):
    nodeId_data = get_nodeId()
    for (i, nodeId) in enumerate(nodeId_data):
        equipmentUuid_data = get_equipment(nodeId["nodeId"])
        for (j, equipmentUuid) in enumerate(equipmentUuid_data):
            point_data = get_point(equipmentUuid["equipmentUuid"])
            for (k, point) in enumerate(point_data):
                if point["pointUuid"] == pointUuid:
                    data = get_history(equipmentUuid["equipmentUuid"], point["pointId"], startTime, endTime)
                    if data is not None:
                        name = data["equipmentName"] + "%%" + point["pointId"]
                        f = open(name + ".txt", "w")
                        f.write(str(data["trendInfo"]))
                        f.close()


if __name__ == '__main__':
    pointUuid = input()
    real_time(pointUuid)
