import jpype

from send_request import get_real_time
from read import read_data


def store():
    count = 0
    count_max = 20000
    while count < count_max:
        get_real_time()
        count = count + read_data()
    # get_real_time()
    # count = count + read_data()
    print(count)
    jpype.shutdownJVM()


if __name__ == '__main__':
    store()
