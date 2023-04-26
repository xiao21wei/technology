import os
import stat

import jpype


def delete():
    # 删除该目录下所有的txt文件
    path = os.getcwd()
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.csv':
                # 如果文件为只读属性，则去除文件的只读属性
                if os.access(os.path.join(root, file), os.F_OK):
                    os.chmod(os.path.join(root, file), stat.S_IWRITE)
                os.remove(os.path.join(root, file))


def delete_file(file_name):
    # 删除该目录下,文件名为file_name的文件
    path = os.getcwd()
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                f = open(os.path.join(root, file), "r")
                # 获取文件的名称
                name = os.path.splitext(file)[0]
                f.close()
                if name == file_name:
                    if os.access(os.path.join(root, file), os.F_OK):
                        os.chmod(os.path.join(root, file), stat.S_IWRITE)
                    os.remove(os.path.join(root, file))


if __name__ == '__main__':
    delete()
