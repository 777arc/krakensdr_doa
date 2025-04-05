# Standalone script to compare the performance of Kraken's DOA algorithms
#
# Copyright (C) 2025 Carl Laufer, Marc Lichtman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#
# - coding: utf-8 -*-

import queue
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, "../../_ui/_web_interface")
sys.path.insert(0, "../../_sdr/_receiver")
sys.path.insert(0, "../../_sdr/_signal_processing")

from kraken_sdr_signal_processor import SignalProcessor
from kraken_sdr_receiver import ReceiverRTLSDR
from variables import shared_path

DOA_algorithms = ["Bartlett", "Capon", "MEM", "MUSIC"] # "ROOT-MUSIC", "TNA",

if not os.path.exists(shared_path):
    os.makedirs(shared_path)

rtl = ReceiverRTLSDR(queue.Queue(1), data_interface="eth", logging_level=10)
sp = SignalProcessor(queue.Queue(1), module_receiver=rtl, logging_level=10)

N = 10000
sample_rate = 1e6
Nr = 5 # elements
d = 0.5 # half wavelength spacing
theta1 = 20 / 180 * np.pi # convert to radians
theta2 = 25 / 180 * np.pi
theta3 = -40 / 180 * np.pi
s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # Nr x 1
s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
s3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)
t = np.arange(N) / sample_rate
tone1 = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1)
tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)
X = s1 @ tone1 + s2 @ tone2 + s3 @ tone3 # note the last one is 1/10th the strength
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
X = X + 0.5*n # Nr x N

sp.DOA_expected_num_of_sources = 3 # used for MUSIC

fig, axs = plt.subplots(len(DOA_algorithms), 1, figsize=(6, 9), subplot_kw={'projection': 'polar'})
fig.subplots_adjust(hspace=-0.2)
for i, DOA_algorithm, in enumerate(DOA_algorithms):
    print(f"Running DOA algorithm {DOA_algorithm}")
    sp.DOA_algorithm = DOA_algorithm
    sp.estimate_DOA(processed_signal=X, vfo_freq=sp.module_receiver.daq_center_freq)
    theta_scan = np.linspace(0, 2*np.pi, len(sp.DOA))
    axs[i].plot(theta_scan, sp.DOA.real / np.max(sp.DOA.real), alpha=0.75)
    axs[i].plot([theta1, theta2, theta3], [1.1, 1.1, 1.1], '.', color='red')
    axs[i].set_ylabel(DOA_algorithm, rotation=0, labelpad=50)
    axs[i].set_theta_zero_location('N') # make 0 degrees point up
    axs[i].set_theta_direction(-1) # increase clockwise
    axs[i].set_rlabel_position(55)  # Move grid labels away from other labels
    axs[i].set_thetamin(-90) # only show top half
    axs[i].set_thetamax(90)
    axs[i].set_position([0, i*0.22, 1, 0.3]) # left, bottom, width, height

fig.tight_layout()
plt.savefig("DOA_algorithms_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
