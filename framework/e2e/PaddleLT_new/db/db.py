#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
db object
"""

import json
import traceback
from datetime import datetime
import yaml
import pymysql

# from utils.logger import logger

ACCURACY = "%.6g"


class DB(object):
    """DB class"""

    def __init__(self, storage="storage.yaml"):
        self.storage = storage
        host, port, user, password, database = self.load_storge()
        self.db = pymysql.connect(host=host, port=port, user=user, password=password, database=database, charset="utf8")
        self.cursor = self.db.cursor()
        # self.now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def load_storge(self):
        """
        解析storage.yaml的内容添加到self.db
        """
        with open(self.storage, "r") as f:
            data = yaml.safe_load(f)
        msg_dict = data.get("Config").get("layer_benchmark").get("MYSQL")
        # msg_dict = data.get("Config").get("layer_benchmark").get("DEV")
        host = msg_dict.get("host")
        port = msg_dict.get("port")
        user = msg_dict.get("user")
        password = msg_dict.get("password")
        database = msg_dict.get("db_name")
        return host, port, user, password, database

    def timestamp(self):
        """
        时间戳控制
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def insert(self, table, data):
        """插入数据"""
        id = -1
        sql_table = "`" + table + "`"
        ls = [(k, data[k]) for k in data if data[k] is not None]
        keys = ",".join(("`" + i[0] + "`") for i in ls)
        values = ",".join("%r" % i[1] for i in ls)

        sql = "INSERT INTO {table}({keys}) VALUES ({values})".format(table=sql_table, keys=keys, values=values)
        try:
            self.cursor.execute(sql)
            id = self.db.insert_id()
            self.db.commit()
        except Exception as e:
            # print(traceback.format_exc())
            print(e)
        return id

    def update(self, table, data, data_condition):
        """按照data_condition 更新数据"""
        sql_table = "`" + table + "`"
        sql = (
            "UPDATE %s SET " % sql_table
            + ",".join("%s=%r" % (("`" + k + "`"), data[k]) for k in data)
            + " WHERE "
            + " AND ".join("%s=%r" % (("`" + k + "`"), data_condition[k]) for k in data_condition)
        )

        try:
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            print(traceback.format_exc())
            print(e)

    def update_by_id(self, table, data, id):
        """按照id 更新数据"""
        sql_table = "`" + table + "`"
        sql = (
            "UPDATE %s SET " % sql_table
            + ",".join("%s=%r" % (("`" + k + "`"), data[k]) for k in data)
            + " WHERE "
            + "%s=%r" % ("`id`", id)
        )

        try:
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            print(traceback.format_exc())
            print(e)

    def select(self, table, condition_list):
        """按照condition_list 查询数据"""
        results = []
        sql_table = "`" + table + "`"
        sql = "SELECT * FROM %s " % sql_table + " WHERE " + " AND ".join("%s" % k for k in condition_list)

        try:
            self.cursor.execute(sql)
            res = self.cursor.fetchall()

            index_list = self.show_list(table=table)
            for row in res:
                tmp = {}
                for i, v in enumerate(row):
                    tmp[index_list[i]] = row[i]
                results.append(tmp)
        except Exception as e:
            print(traceback.format_exc())
            print(e)
        return results

    def select_use_date(self, table, date_str, condition_dict):
        """按照condition_list 查询数据"""
        results = []
        # sql_table = "`" + table + "`"
        # condition_list.append("DATE(update_time) = %s")
        # sql = "SELECT * FROM %s " % sql_table + " WHERE " + " AND ".join("%s" % k for k in condition_list)

        # 构建条件列表，注意日期条件的特殊处理
        conditions = []
        params = []

        # 添加日期条件
        conditions.append("DATE(update_time) = %s")
        params.append(date_str)  # 确保date_str是"YYYY-MM-DD"格式的字符串

        # 添加其他条件
        for key, value in condition_dict.items():
            # 这里假设value已经是适合数据库查询的格式（如字符串、数字等）
            # 如果value是日期或时间，请确保它在这里被正确格式化
            conditions.append(f"{key} = %s")
            params.append(value)

        # 构建完整的SQL查询语句
        sql = "SELECT * FROM {} WHERE {}".format(table, " AND ".join(conditions))
        try:
            self.cursor.execute(sql, tuple(params))
            res = self.cursor.fetchall()

            index_list = self.show_list(table=table)
            for row in res:
                tmp = {}
                for i, v in enumerate(row):
                    tmp[index_list[i]] = row[i]
                results.append(tmp)
        except Exception as e:
            print(traceback.format_exc())
            print(e)
        return results

    def select_by_id(self, table, id):
        """按照id 查询数据"""
        results = []
        sql_table = "`" + table + "`"
        sql = "SELECT * FROM %s " % sql_table + " WHERE " + "%s=%r" % ("`id`", id)

        try:
            self.cursor.execute(sql)
            res = self.cursor.fetchall()

            tmp = {}
            index_list = self.show_list(table=table)
            for row in res:
                for i, v in enumerate(row):
                    tmp[index_list[i]] = row[i]
                results.append(tmp)
        except Exception as e:
            print(traceback.format_exc())
            print(e)
        return results

    def insert_job(
        self,
        comment,
        status,
        result,
        env_info,
        framework,
        agile_pipeline_build_id,
        testing_mode,
        testing,
        plt_perf_content,
        layer_type,
        commit,
        version,
        hostname,
        hardware,
        system,
        md5_id,
        base,
        ci,
        create_time,
        update_time,
    ):
        """向job表中录入数据, 表中部分字段后续更新"""
        data = {
            "comment": comment,
            "status": status,
            "result": result,
            "env_info": env_info,
            "framework": framework,
            "agile_pipeline_build_id": agile_pipeline_build_id,
            "testing_mode": testing_mode,
            "testing": testing,
            "plt_perf_content": plt_perf_content,
            "layer_type": layer_type,
            "commit": commit,
            "version": version,
            "hostname": hostname,
            "hardware": hardware,
            "system": system,
            "md5_id": md5_id,
            "base": base,
            "ci": ci,
            "create_time": create_time,
            "update_time": update_time,
        }
        id = self.insert(table="layer_job", data=data)
        return id

    def insert_case(self, jid, case_name, result, create_time):
        """向case表中录入数据"""
        try:
            data = {
                "jid": jid,
                "case_name": case_name,
                "result": result,
                "create_time": create_time,
            }
            retry = 3
            for i in range(retry):
                case_id = self.insert(table="layer_case", data=data)
                if case_id == -1:
                    print("db ping again~~~")
                    self.db.ping(True)
                    continue
                else:
                    break
        except Exception as e:
            # self.ci_update_job(id=job_id, status="error", update_time=time_now)
            print(traceback.format_exc())
            print(e)

    def update_job(self, id, status, update_time):
        """数据录入完成后更新job表中的部分字段"""
        data = {"status": status, "update_time": update_time}
        self.update_by_id(table="layer_job", data=data, id=id)

    def select_baseline_job(self, comment, testing, plt_perf_content, base, ci, md5_id):
        """通过comment字段、ci字段、机器唯一标识码，查找baseline数据"""
        condition_list = [
            "comment = '{}'".format(comment),
            "status = 'done'",
            "testing = '{}'".format(testing),
            "plt_perf_content = '{}'".format(plt_perf_content),
            "base = '{}'".format(base),
            "ci = '{}'".format(ci),
            "md5_id = '{}'".format(md5_id),
        ]
        res = self.select(table="layer_job", condition_list=condition_list)
        baseline_job = res[-1]
        # job_id = baseline_job["id"]
        return baseline_job

    def show_list(self, table):
        """返回table中的列list"""
        results = []
        sql_table = "`" + table + "`"
        sql = "SHOW COLUMNS from {}".format(sql_table)
        try:
            self.cursor.execute(sql)
            results = [column[0] for column in self.cursor.fetchall()]
        except Exception as e:
            print(traceback.format_exc())
            print(e)
        return results


if __name__ == "__main__":
    db = DB(storage="storage.yaml")

    # table = "layer_case"
    # condition_list = ["jid = 71"]
    # res = db.select(table=table, condition_list=condition_list)
    # print("res is: ", res)

    table = "layer_job"
    condition_dict = {"md5_id": "0b00a671d6e8db2e16afc619c7289970"}
    date_str = "2024-11-03"
    res = db.select_use_date(table=table, date_str=date_str, condition_dict=condition_dict)
    print("res is: ", res)
