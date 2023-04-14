from database import select_data, count_data


def select():
    count = count_data()
    print("数据库中共有" + str(count) + "条数据")
    while True:
        t1 = input()
        if t1 == '0':
            break
        else:
            time1, time2 = select_data(t1)
            print("读取", t1, "条数据（未存储至文本文件）所用时间为：", time1)
            print("读取", t1, "条数据（已存储至文本文件）所用时间为：", time2)


if __name__ == '__main__':
    select()
