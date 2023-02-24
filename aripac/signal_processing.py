#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Signal processing plot functions.

周波数解析や周波数応答の関数
"""
__author__  = 'ari23 (Twitter: ari23ant)'
__version__ = '0.2.7'
__date__ = '2023/02/24'
__status__ = 'Development'

import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt


def plotfft(wave, fs, win='hanning', dB=True, vert=[], title='', ylabel='Wave', xlabel='Time [ms]', alpha=0.1):
    """Plot fft result.

    MATLAB関数fftを参考に実装した。
    この関数を呼んだあと、必ずplt.show()を叩くこと。
    param
    -----
    wave: array_like
        波形データ
    fs: float
        サンプリング周波数[Hz]
    win: string
        窓関数指定
    dB: bool
        dBへの単位変換 Trueなら単位変換する
    vert: array_like
        周波数応答のグラフに目印の縦線を入れる
    title: string
        出力するグラフのタイトル
    ylabel: string
        出力するグラフの縦軸のラベル
    xlabel: string
        出力するグラフの縦軸のラベル
    alpha: float
        fftの結果の拡大率
    """
    # --- 引数確認 --- #
    if not (0 < alpha < 1):
        print('alpha is INVALID value.')
        return False

    # --- 窓関数で波形を切り出す --- #
    L = len(wave)
    if win == 'hanning':
        window = np.hanning(L)  # ハン窓
    elif win == 'hamming':
        window = np.hamming(L)  # ハミング窓
    elif win == 'square':
        window = np.ones(L)    # 矩形窓
    else:
        print(f'{win} is INVALID. Use square window.')
        win = 'square'
        window = np.ones(L)    # 矩形窓

    # 窓掛け
    wave_window = wave * window

    # --- FFT計算 --- #
    # fftpack->fftに変更
    NFFT = 2**_nextpow2(L)  # 計算速度向上のため解析データ数に近い2の乗数を計算
    fft_amp = fft.fft(wave_window, n=NFFT)  # 周波数領域のAmplitude
    fft_frq = fft.fftfreq(NFFT, d=1.0/fs)  # Amplitudeに対応する周波数

    # 正の領域のみ抽出
    fft_amp = fft_amp[0: int(len(fft_amp)/2)]
    fft_frq = fft_frq[0: int(len(fft_frq)/2)]
    if dB:
        fft_amp = _dB(abs(fft_amp))  # 複素数→デシベル変換
    else:
        fft_amp = abs(fft_amp)

    # --- グラフ表示 --- #
    # fig設定
    fig = plt.figure(figsize=(8, 6*1.5), tight_layout=True)
    fig.suptitle(title)

    # timestamp
    ts = np.array([i for i in range(L)]) * (1000. / fs)

    # 入力データ
    ax = fig.add_subplot(4, 1, 1)
    ax.plot(ts, wave)
    ax.set_ylabel(ylabel)
    ax.grid(visible=True)

    # 入力データ*窓
    ax = fig.add_subplot(4, 1, 2)
    ax.plot(ts, wave_window)
    ax.set_xlabel('Timestamp [ms]')
    ax.set_ylabel(ylabel + '*' + win)
    ax.grid(visible=True)

    # FFT
    ax = fig.add_subplot(4, 1, 3)
    ax.plot(fft_frq, fft_amp)
    ax.set_xlim(0, fs/2)
    if dB:
        ax.set_ylabel('Amplitude [dB]')  # |X(omega)|
    else:
        ax.set_ylabel('Amplitude')
    ax.grid(visible=True)

    # 低周波領域FFTプロット
    ax = fig.add_subplot(4, 1, 4)
    ax.plot(fft_frq, fft_amp)
    if vert is not None:
        for v in vert:
            ax.plot([v, v], [fft_amp.min(), fft_amp.max()], label=str(v)+'[Hz]')
        ax.legend()
    ax.set_xlim(0, fs/2*alpha)
    ax.set_xlabel('Frequency [Hz]')
    if dB:
        ax.set_ylabel('Amplitude [dB]')  # |X(omega)|
    else:
        ax.set_ylabel('Amplitude')
    ax.grid(visible=True)

    return fig


def _nextpow2(n):  # MATLAB関数nextpow2
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return int(np.log2(2**m_i))


def plotfreqz(b, fs, a=1, worN=8192, vert=[], title=''):
    """Plot freqz result.

    MATLAB関数fvtoolを参考
    この関数を呼んだあと、必ずplt.show()を叩くこと
    param
    -----
    b: array_like
        伝達関数の分子の係数
    a: array_like
        伝達関数の分母の係数
    fs: float
        サンプリング周波数[Hz]
    worN: int
        単位円の半円の分割数
        周波数に変換するときは w/pi*fn をする
    vert: array_like
        周波数応答のグラフに目印の縦線を入れる
    title: string
        出力するグラフのタイトル
    """
    # --- F特計算 --- #
    # ナイキスト周波数計算
    fn = fs / 2.0

    # 周波数応答計算
    w, h = signal.freqz(b, a, worN)
    x_freq_rspns = w / np.pi * fn
    y_freq_rspns = _dB(abs(h))  # 複素数→デシベル変換

    # 群遅延計算
    w, gd = signal.group_delay((b, a), worN)
    x_gd = w / np.pi * fn
    y_gd = gd

    # y_gdの要素にゼロがあったら削除する（SciPyのバグ？）
    zero_index = np.where(y_gd == 0.0, True, False)
    if zero_index.sum() > 0:
        print('output of signal.group_delay has zero value and delete the elements')
        nonzero_index = np.logical_not(zero_index)
        x_gd = x_gd[nonzero_index]
        y_gd = y_gd[nonzero_index]

    # --- グラフ表示 --- #
    fig = plt.figure(figsize=(8, 6), tight_layout=True)
    fig.suptitle(title)

    # 周波数応答
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(x_freq_rspns, y_freq_rspns)
    if vert is not None:
        for v in vert:
            ax.plot([v, v], [y_freq_rspns.min(), y_freq_rspns.max()], label=str(v)+'[Hz]')
        ax.legend()
    ax.set_ylabel('Amplitude [dB]')
    ax.grid(visible=True)

    # 群遅延
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(x_gd, y_gd)
    ax.set_ylim(0, len(b))  # MATLAB fvtool仕様
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Group delay [samples]')
    ax.grid(visible=True)

    return fig


def _dB(x, dBref=1):  # デシベル変換
    #y = 20 * np.log10(x / dBref)
    y = 20 * np.log10(x / dBref + 1.0e-06)
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
