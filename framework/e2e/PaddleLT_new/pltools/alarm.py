#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
alarm
"""
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import yaml


class Alarm(object):
    """
    报警模块
    """

    def __init__(self, storage="apibm_config.yml"):
        """

        :param storage:
        """
        with open(storage, "r") as f:
            data = yaml.safe_load(f)
            alarm = data.get("Config").get("alarm")
            self.smtp = alarm.get("smtp")
            self.sender = alarm.get("sender")
            self.receiver = alarm.get("receiver")

    def email_send(self, receiver, subject, content, sender=None):
        """
        send
        """
        if sender is None:
            sender = self.sender
        message = MIMEText(content, "html", "utf-8")
        message["To"] = Header(", ".join(receiver))

        message["Subject"] = Header(subject, "utf-8")
        try:
            smtpObj = smtplib.SMTP(self.smtp)
            smtpObj.sendmail(sender, receiver, message.as_string())
            print("邮件发送成功")
        except smtplib.SMTPException:
            print("邮件发送失败")


if __name__ == "__main__":
    alarm = Alarm("apibm_config.yml")
    alarm.email_send(alarm.receiver, "hello", "hello world")
