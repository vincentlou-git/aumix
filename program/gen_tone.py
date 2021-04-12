"""
gen_tone.py

Program to generate a tone .wav file.

@author: Chan Wai Lou
"""

import sys

import aumix.signal.simple_signal as ss
import aumix.io.wav as wav


func_map = {
    "sine": ss.SineSignal,
    "square": ss.SquareSignal,
    "sawtooth": ss.SawtoothSignal
}


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python3 gen_tone.py <type> <frequency> <volume> <duration> <samp_rate>")
        print("type:\t\tsine, square, sawtooth;")
        print("frequency:\tnumber;")
        print("volume:\t\tnumber from 0 to 100;")
        print("duration:\tnumber in seconds;")
        print("samp_rate:\tsampling rate in Hertz.")
        sys.exit()

    if sys.argv[0].lower() not in func_map.keys():
        print("type must be sine, square, or sawtooth")

    # Give arguments names
    signal_func = func_map[sys.argv[0].lower()]
    freq = sys.argv[1]
    vol = sys.argv[2]
    duration = sys.argv[3]
    samp_rate = sys.argv[4]

    # Generate signal
    sig = signal_func(freq=freq, samp_rate=samp_rate, duration=duration)
    data = sig.data

    wav.write(f"audio/{sig}", data, samp_rate=samp_rate, amp_perc=vol/100)

    print(f"File written to 'audio/{sig}.wav'")
