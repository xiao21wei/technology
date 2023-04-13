from database import select_data, count_data


def select():
    count = count_data()
    print("数据库中共有" + str(count) + "条数据")
    while True:
        t1 = input()
        if t1 == '0':
            break
        else:
            time1 = select_data(t1)
            print("读取", t1, "条数据所用时间为：", time1)


if __name__ == '__main__':
    select()
