from database import select_data, count_data


def select():
    count = count_data()
    print("数据库中共有" + str(count) + "条数据")
    t1 = 20000
    t2 = 100000
    time1 = select_data(t1)
    time2 = select_data(t2)
    print("读取" + str(t1) + "条数据所用时间：", time1)
    print("读取" + str(t2) + "条数据所用时间：", time2)


if __name__ == '__main__':
    select()
