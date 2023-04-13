import os

from database import store_data
from delete import delete_file


def read_data():  # 从该目录下的txt文件中读取数据，存入数据库
    num = 0
    path = os.getcwd()
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                num = num + 1
                f = open(os.path.join(root, file), "r")
                # 获取文件的名称
                file_name = os.path.splitext(file)[0]
                name = file_name.split("%%")
                equipmentName = name[0]
                pointId = name[1]
                data_time = name[2]
                # data = f.read()
                # block_size = 10000  # 每次读取的数据块大小
                # num_blocks = (len(data) + block_size - 1) // block_size  # 计算需要读取的数据块数
                # for i in range(num_blocks):
                #     start = i * block_size
                #     end = min((i + 1) * block_size, len(data))
                #     block = data[start:end]
                #     store_data(equipmentName, pointId, data_time, block)
                f.close()
                store_data(equipmentName, pointId, data_time, file_name+'.txt')
                delete_file(file_name)
    return num


if __name__ == '__main__':
    read_data()
