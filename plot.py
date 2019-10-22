#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:03:19 2019

@author: lijiahao
"""

import numpy as np
import matplotlib.pyplot as plt


binary00 = np.loadtxt("/Users/lijiahao/Documents/UBC/CPEN_502/UBC-CPEN502/binary00.text")
binary09 = np.loadtxt("/Users/lijiahao/Documents/UBC/CPEN_502/UBC-CPEN502/binary09.text")
bipolar00 = np.loadtxt("/Users/lijiahao/Documents/UBC/CPEN_502/UBC-CPEN502/bipolar00.text")
bipolar09 = np.loadtxt("/Users/lijiahao/Documents/UBC/CPEN_502/UBC-CPEN502/bipolar09.text")

# pltBinary00 = plt
# pltBinary00.plot(binary00)
# pltBinary00.xlabel('epoch')
# pltBinary00.ylabel('error')
# pltBinary00.title('binary/momemtum = 0')
# pltBinary00.show()

# pltBinary09 = plt
# pltBinary09.plot(binary09)
# pltBinary09.xlabel('epoch')
# pltBinary09.ylabel('error')
# pltBinary09.title('binary/momemtum = 0.9')
# pltBinary09.show()

# pltBipolar00 = plt
# pltBipolar00.plot(bipolar00)
# pltBipolar00.xlabel('epoch')
# pltBipolar00.ylabel('error')
# pltBipolar00.title('bipolar/momemtum = 0')
# pltBipolar00.show()

pltBipolar09 = plt
pltBipolar09.plot(bipolar09)
pltBipolar09.xlabel('epoch')
pltBipolar09.ylabel('error')
pltBipolar09.title('bipolar/momemtum = 0.9')
pltBipolar09.show()
