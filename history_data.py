import time


def get_history_data():
    # 获取当前时间的UNIX毫秒时间戳和一周前的UNIX毫秒时间戳
    now = int(time.time() * 1000)
    week_ago = now - 7 * 24 * 60 * 60 * 1000
    # 将UNIX毫秒时间戳转换为字符串
    now = str(now)
    week_ago = str(week_ago)
    print(now)
    print(week_ago)


if __name__ == '__main__':
    get_history_data()
