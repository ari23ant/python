#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Pythonスクリプトテンプレート
- 2019/10/07| テンプレート作成
- 2020/01/14| IPythonおまじない修正とログ追加
'''
__author__ = 'ari23(Twitter: ai23ant)'
__date__ = 'yyyy/mm/dd'


import time
#import os
#import sys
from IPython import get_ipython
import logging


class Template:

    def __init__(self):
        self.string = "Hello world!"

    def Process(self):
        print(self.string)

if __name__ == '__main__':
    # IPython使用時のおまじない
    if get_ipython().__class__.__name__ == 'TerminalInteractiveShell':
        get_ipython().magic('reset -sf')
        print('IPython reset command')

    #---------- Program Start ----------#
    start_time = time.perf_counter()
    print('---------- Start ----------')

    '''
    #--- ログ用意 ---#
    # ロガー
    logger = logging.getLogger('ControlADIWatch')
    logger.setLevel(logging.DEBUG)

    # フォーマット
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # コンソール出力用
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    # ファイル出力用
    file_handler = logging.FileHandler('nk_python.log', 'a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # ロガーに追加
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    '''
 
    #--- Template ---#
    proc = Template()
    proc.Process()

    '''
    #--- 終了処理 ---#
    # 重複出力を避けるおまじない
    del logging.Logger.manager.loggerDict[logger.name]
    '''

    #---------- Program End ----------#
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print('Execution Time: ' + str(execution_time) + 's')
    print('----------  End  ----------')
