#!/usr/bin/python
# -*- coding: utf-8 -*-
'''信号処理
周波数解析や周波数応答の関数
'''
__author__  = 'ari23 (Twitter: ari23ant)'
__version__ = '0.0.1'
__date__    = '2020/03/06'
__status__  = 'Development'

import time
import pandas as pd
import numpy as np
from scipy import signal
from scipy import fftpack
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set()


def fft(array, fs, win='hanning', ylabel_name='Data', xlabel_name='Time'):
    '''
    MATLAB関数fftを参考
    この関数を呼んだあと、必ずplt.show()を叩くこと
    param
    -----
    array: array_like
        波形データ
    fs: float
        サンプリング周波数[Hz]
    win: string
        窓関数指定 デフォルト値は'hanning'
    ylabel_name: string
        出力するグラフの縦軸のラベル デフォルト値は'Data'
    xlabel_name: string
        出力するグラフの横軸のラベル デフォルト値は'Time'
    '''
    # 窓関数で波形を切り出す
    L = len(array)
    if win == 'hanning':
        window = np.hanning(L)  # ハン窓
    elif win == 'hamming':
        window = np.hamming(L)  # ハミング窓
    elif win == 'square':
        window = np.ones(L)    # 矩形窓
    else:
        print('win is fault: ' + win)
        print('use square window')
        win = 'square'
        window = np.ones(L)    # 矩形窓

    array_window = array * window

    # FFT計算
    # numpy.fftよりscipy.fftpackの方が速い
    NFFT = 2**nextpow2(L)  # 計算速度向上のため解析データ数に近い2の乗数を計算
    fft_amp = fftpack.fft(array_window, NFFT)  # 周波数領域のAmplitude
    fft_fq = fftpack.fftfreq(NFFT, d=1.0/fs)  # Amplitudeに対応する周波数
    # 正の領域のみ抽出
    fft_amp = fft_amp[0: int(len(fft_amp)/2)]
    fft_fq = fft_fq[0: int(len(fft_fq)/2)]
    fft_amp = db(abs(fft_amp))  # 複素数→デシベル変換

    # グラフ表示
    plt.figure(figsize=(8, 6*1.5))
    plt.subplots_adjust(hspace=0.4)  # x軸ラベル表示のため余白調整
    # 入力データプロット
    plt.subplot(4, 1, 1)
    plt.plot(array)
    #plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    plt.grid()
    # 入力データ*窓関数プロット
    plt.subplot(4, 1, 2)
    plt.plot(array_window)
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name + '*' + win)
    plt.grid()
    # FFTプロット
    plt.subplot(4, 1, 3)
    plt.plot(fft_fq, fft_amp)
    plt.xlim(0, fs/2)
    #plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')  # |X(omega)|
    plt.grid()
    # 低周波領域FFTプロット
    plt.subplot(4, 1, 4)
    plt.plot(fft_fq, fft_amp)
    plt.xlim(0, fs/20)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')  # |X(omega)|
    plt.grid()

def nextpow2(n):  # MATLAB関数nextpow2
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return int(np.log2(2**m_i))

def fvtool(b, fs, a=1, worN=8192, fc1=None, fc2=None):
    '''
    MATLAB関数fvtoolを参考
    この関数を呼んだあと、必ずplt.show()を叩くこと
    param
    -----
    b: array_like
        伝達関数の分子の係数
    a: array_like
        伝達関数の分母の係数 デフォルト値は1
    fs: float
        サンプリング周波数[Hz]
    worN: int
        単位円の半円の分割数
        周波数に変換するときは w/pi*fn をする
    fc1: float
        周波数応答のグラフにカットオフ周波数などの印（縦線）を入れる
    fc2: float
        周波数応答のグラフにカットオフ周波数などの印（縦線）を入れる
    '''
    # ナイキスト周波数計算
    fn = fs / 2

    # 周波数応答計算
    w, h = signal.freqz(b, a, worN)
    x_freq_rspns = w / np.pi * fn
    y_freq_rspns = db(abs(h))  # 複素数→デシベル変換

    # 群遅延計算
    w, gd = signal.group_delay([b, a], worN)
    x_gd = w / np.pi * fn
    y_gd = gd

    # グラフ表示
    plt.figure(figsize=(8, 6))
    # 周波数応答プロット
    plt.subplot(2, 1, 1)
    plt.plot(x_freq_rspns, y_freq_rspns)
    if fc1 is not None:
        plt.plot([fc1, fc1], [y_freq_rspns.min(), y_freq_rspns.max()])
    if fc2 is not None:
        plt.plot([fc2, fc2], [y_freq_rspns.min(), y_freq_rspns.max()])
    #plt.ylim(-70, 2)  # MATLAB fvtool仕様
    plt.ylabel('Amplitude [dB]')
    plt.grid()
    # 群遅延プロット
    plt.subplot(2, 1, 2)
    plt.plot(x_gd, y_gd)
    plt.ylim(0, len(b))  # MATALB fvtool仕様
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Group delay [samples]')
    plt.grid()

def db(x, dBref=1):  # デシベル変換
    y = 20 * np.log10(x / dBref)
    return y

def make_MA_filter(N):
    if N % 2 == 0:
        print('N is even, and should be odd for Moving Average Filter.')

    b = np.ones(N) / N
    a = [1 for i in range(N) ]
    return b/a

def make_KZ_filter(N, _times):
    f_MA = make_MA_filter(N)
    f_KZ = f_MA.copy()

    for i in range(_times - 1):
        f_KZ = np.convolve(f_KZ, f_MA, mode='full')

    return f_KZ

def is_odd(num):
    if num % 2 == 0:
        return False
    else:
        return True
