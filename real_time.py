import json
import time
import yaml

from send_request import get_nodeId, get_equipment, get_point, get_data

# 从secrets.yml文件中读取IP和PORT
with open('secrets.yml', 'r', encoding='utf-8') as f:
    file_content = f.read()
    content = yaml.load(file_content, yaml.FullLoader)
    IP = content["IP"]
    PORT = content["PORT"]


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


if __name__ == '__main__':
    pointUuid = input()
    real_time(pointUuid)
