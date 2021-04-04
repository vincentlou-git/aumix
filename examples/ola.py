"""
ola.py

Module for investigating whether a (window, overlap) tuple satisfies COLA or NOLA.

@author: Chan Wai Lou
"""


import scipy.signal as signal
import numpy as np
import pandas as pd


"""
To enable iSTFT, it is sufficient that COLA (Constant Overlap Add) is met.
This property ensures each data point is weighted equally, 
thereby avoiding aliasing ("roughing"), giving for a perfect reconstruction.

However, even if COLA is not met, as long as NOLA is met, iSTFT can be done. 
"""
rect120 = signal.windows.boxcar(120)
hann_sym64 = signal.windows.hann(64, sym=True)
hann_sym120 = signal.windows.hann(120, sym=True)
hann_asym120 = signal.windows.hann(120, sym=False)
blackmanharris120 = signal.windows.blackmanharris(120)
ones64 = np.ones(64, dtype="float")

tests = [
    ("Rectangular, 75%: ", rect120, 120, 90),   # True , True
    ("Rectangular, 25%: ", rect120, 120, 30),   # False, True

    ("Hann symmetrical, 50%: ", hann_sym120, 120, 60),         # False, True
    ("Hann Periodic/DFT-even, 1/2: ", hann_asym120, 120, 60),  # True , True
    ("Hann Periodic/DFT-even, 2/3: ", hann_asym120, 120, 80),  # True , True
    ("Hann Periodic/DFT-even, 3/4: ", hann_asym120, 120, 90),  # True , True

    ("Blackmanharris, 50%: ", blackmanharris120, 120, 60),     # False, True
    ("Blackmanharris, 75%: ", blackmanharris120, 120, 90),     # True , True

    # NOLA
    ("62 ones with 2 zeros appended, 50%: ", ones64, 64, 32),  # False, False

    ("Hann symmetrical, 1/64 (not enough overlap): ", hann_sym64, 64, 1),   # False, False
    ("Hann symmetrical, 2/64 (not enough overlap): ", hann_sym64, 64, 2),   # False, True
    ("Hann symmetrical, 3/64 (not enough overlap): ", hann_sym64, 64, 3)    # False, True
]

results = np.empty((len(tests), 3), dtype="<U64")   # no. tests rows, 3 columns

for i, (desc, w, nperseg, noverlap) in enumerate(tests):
    results[i] = np.array([desc, str(signal.check_COLA(w, nperseg, noverlap)), str(signal.check_NOLA(w, nperseg, noverlap))])


df = pd.DataFrame(data=results, columns=["Description", "COLA", "NOLA"])
print(df)
