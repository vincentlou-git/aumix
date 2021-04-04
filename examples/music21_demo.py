"""
music21_demo.py

Attempts to use music21.

@author: DoraMemo
"""


import music21 as m21

a4 = m21.note.Note("A4")
asharp4 = a4.transpose(1)
b4 = a4.transpose(2)
c5 = a4.transpose(3)

print(a4, asharp4, b4, c5)

melody = m21.stream.Stream()
melody.append([a4, asharp4, b4, c5])
melody.show()
melody.show("text")

# cMinor = m21.chord.Chord(["C4", "G4", "E-5"])
# cMinor.show()
