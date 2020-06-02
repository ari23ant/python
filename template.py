#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pythonスクリプトテンプレート
更新履歴
- 2019/10/07| テンプレート作成
- 2020/01/14| IPythonおまじない修正とログ追加
- 2020/02/14| versionとstatus追加、引数機能、リファクタリング
- 2020/03/06| self.string→self.msgなどリファクタリング
- 2020/04/05| リファクタリング
- 2020/06/02| plt.figure()があったら削除する機能追加
"""
__author__  = 'ari23(Twitter: @ari23ant)'
__version__ = '0.0.6'
__date__    = '2020/06/02'
__status__  = 'Development'

#import os
#import sys
import time
from IPython import get_ipython
#import logging


class Template:

    def __init__(self):
        self.msg = 'Hello world!'

    def Process(self):
        print(self.msg)
        #logger.debug(self.msg)

if __name__ == '__main__':
    # IPython使用時のおまじない
    if get_ipython().__class__.__name__ == 'TerminalInteractiveShell':
        # plt.fiugre()があったら削除
        import sys
        if 'matplotlib.pyplot' in sys.modules:
            #import matplotlib.pyplot
            #matplotlib.pyplot.close('all')
            plt.close('all')
            print('close all figures')
        # IPython Resetコマンド
        get_ipython().magic('reset -sf')
        print('IPython reset command')
    # ---------- Program Start ---------- #
    start_time = time.perf_counter()
    print('---------- Start ----------')

    '''
    # --- ログ用意 --- #
    # ロガー
    logger = logging.getLogger('Template')
    logger.setLevel(logging.DEBUG)
    # フォーマット
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # コンソール出力用
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    # ファイル出力用
    file_handler = logging.FileHandler('test.log', 'a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    # ロガーに追加
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    '''

    # --- Template --- #
    proc = Template()
    proc.Process()

    '''
    # --- Get Argument --- #
    args = sys.argv  # list
    # --- Main Process --- #
    if len(args) == 1:
        proc = Template('')
    else:  # argsの大きさが0になることはない
        proc = Template(args[1])
    proc.Process()
    '''

    '''
    # --- ログ終了処理 --- #
    # 重複出力を避けるおまじない
    del logging.Logger.manager.loggerDict[logger.name]
    '''

    # ---------- Program End ---------- #
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print('Execution Time: ' + str(execution_time) + 's')
    print('----------  End  ----------')
