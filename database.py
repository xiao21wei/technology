import datetime
import sys

import jpype.imports
from jpype.types import *


# jvm成功启动后导入相应的Java模块
jvmPath = r'C:\Program Files\Java\jdk-16.0.2\bin\server\jvm.dll'
jdbc_driver = r"E:\technology\project\gbasedbtjdbc_3.3.0_2P20220402_428c64.jar"
jpype.addClassPath(jdbc_driver)
if not jpype.isJVMStarted():
    jpype.startJVM(jvmPath)

from java.lang import *
from java.util import *
from java.sql import *


def connect_database():
    # 1.加载驱动
    url = "jdbc:gbasedbt-sqli://192.168.119.209:9088/gbasedb:GBASEDBTSERVER=gbaseserver;DB_LOCALE=zh_CN.utf8;CLIENT_LOCALE=zh_CN.utf8;NEWCODESET=UTF8,utf8,57372;GL_USEGLU=1"
    user = "gbasedbt"
    password = "ndtyGBase8s"
    driver_cls = JClass("com.gbasedbt.jdbc.Driver")
    # 获取驱动版本
    driver_version = driver_cls.getJDBCVersion()
    try:
        import java.lang.System as System
        conn = DriverManager.getConnection(url, user, password)
    except JException as e:
        print("连接失败")
        print(e.message())
        sys.exit(1)
    return conn


def store_data(equipmentName, pointId, data_time, data):  # 将数据导入到数据库中
    # CREATE TABLE gbasedb.info (
    # 	id SERIAL NOT NULL,
    # 	equipmentname VARCHAR(100),
    # 	pointid VARCHAR(100),
    # 	trendtime DATETIME YEAR TO SECOND,
    # 	currentvalue CLOB,
    # 	PRIMARY KEY (id) CONSTRAINT current_info_pk
    # )
    #  in datadbs1 ;
    conn = connect_database()
    # 2.创建Statement对象
    stmt = conn.createStatement()
    data_time = datetime.datetime.strptime(data_time, "%Y-%m-%d_%H_%M_%S")
    data_time = data_time.strftime("%Y-%m-%d %H:%M:%S")
    # 3.执行SQL语句
    sql = "insert into current_info(id, equipmentname, pointid, trendtime, currentvalue) " \
        "values(0, '" + equipmentName + "','" + pointId + "','" + data_time + "',FILETOCLOB('" + data + "', 'client'))"
    stmt.execute(sql)

    # # data为需要存储的数据，data_time为数据的时间
    # # 根据equipmentname,pointid,data_time,判断数据库中是否存在该数据
    # # 如果存在，将data拼接在currentvalue后面
    # # 如果不存在，将data存储在currentvalue中
    # # currentvalue为clob类型,需要使用concat函数拼接
    # # 将字符串类型的data转换成CLOB类型
    # # data = "FILETOCLOB('" + data + "', 'client')"
    # sql = "select * from info where equipmentname = '" + equipmentName + "' and pointid = '" + pointId + "' and trendtime = '" + data_time + "'"
    # rs = stmt.executeQuery(sql)
    # if rs.next():
    #     # 存在数据，需要拼接
    #     sql = "update info set currentvalue = concat(currentvalue, '" + data + "') where equipmentname = '" + equipmentName + "' and pointid = '" + pointId + "' and trendtime = '" + data_time + "'"
    #     print(sql)
    #     stmt.execute(sql)
    # else:
    #     # 不存在数据，直接插入
    #     sql = "insert into info(id, equipmentname, pointid, trendtime, currentvalue) " \
    #         "values(0, '" + equipmentName + "','" + pointId + "','" + data_time + "','" + data + "')"
    #     print(sql)
    #     stmt.execute(sql)
    # 4.关闭连接
    stmt.close()
    conn.close()


def delete_data():
    conn = connect_database()
    # 2.创建Statement对象
    stmt = conn.createStatement()
    # 3.执行SQL语句
    sql = "delete from current_info"
    stmt.execute(sql)
    # 4.关闭连接
    stmt.close()
    conn.close()


def select_data(num):  # 返回从数据库中读取到num条数据，导入到文件中，并记录读取到的数据所用的时间
    conn = connect_database()
    time1 = datetime.datetime.now()
    # 2.创建Statement对象
    stmt = conn.createStatement()
    # 3.执行SQL语句
    sql = "select * from current_info where id <= " + str(num)
    rs = stmt.executeQuery(sql)
    # 4.处理结果集，将结果导入到txt文件中
    file_name = str(num) + "_" + str(time1.strftime("%Y-%m-%d_%H_%M_%S")) + ".txt"
    with open(file_name, "w") as f:
        while rs.next():
            id = str(rs.getString(1))
            equipmentname = str(rs.getString(2))
            pointid = str(rs.getString(3))
            trendtime = str(rs.getString(4))
            f.write(id + "," + equipmentname + "," + pointid + "," + trendtime  + "\n")
    time2 = datetime.datetime.now()
    # 5.关闭结果集
    rs.close()
    # 6.关闭连接
    stmt.close()
    conn.close()
    return time2 - time1


def count_data():
    ans = 0
    conn = connect_database()
    # 2.创建Statement对象
    stmt = conn.createStatement()
    # 3.执行SQL语句
    sql = "select count(*) from current_info"
    rs = stmt.executeQuery(sql)
    # 4.处理结果集
    while rs.next():
        ans = rs.getString(1)
    # 5.关闭结果集
    rs.close()
    # 6.关闭连接
    stmt.close()
    conn.close()
    return ans
