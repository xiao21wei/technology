from database import select_data_pro, count_data, select_data


def select_pro():
    count = count_data()
    print("数据库中共有" + str(count) + "条数据")
    time1, time1_1 = select_data(100000)
    time2, time2_1 = select_data_pro(100000, 100)
    time3, time3_1 = select_data(1000)
    print("读取", 100000, "条数据（未存储至文本文件）所用时间为：", time1)
    print("读取", 100000, "条数据（已存储至文本文件）所用时间为：", time1_1)
    print("读取", 1000, "条数据（间隔100，未存储至文本文件）所用时间为：", time2)
    print("读取", 1000, "条数据（间隔100，已存储至文本文件）所用时间为：", time2_1)


if __name__ == '__main__':
    select_pro()
