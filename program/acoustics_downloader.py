"""
acoustics_downloader.py

Downloads the acoustic (human played notes) of instruments from Music Acoustics UNSW.

@author: Chan Wai Lou
"""

import os
import re
import copy
import requests
import aumix.music.freq as afreq

folder = "acoustics_downloader"
dynamics = ["f", "mp", "p"]
note_names = copy.deepcopy(afreq.notes)
note_names = [n.replace("#", "sharp") for n in note_names]   # replace # with sharp

data = {
    "cl": {
        "url": "https://newt.phys.unsw.edu.au/music/clarinet/sounds/",   # G3.wav
        # "notes": note_names[note_names.index("E3"):note_names.index("Csharp7")]   # Need more cleansing
    },
    "fl_B": {
        "url": "https://newt.phys.unsw.edu.au/music/flute/modernB/sounds/",  # E5.B.wav
        "urlappend": ".B.wav",
        "notes": note_names[note_names.index("C4"):note_names.index("Fsharp7")]
    },
    "fl_C": {
        "url": "https://newt.phys.unsw.edu.au/music/flute/modernC/sounds/",   # C4.C.wav
        "urlappend": ".C.wav",
        "notes": note_names[note_names.index("C4"):note_names.index("Fsharp7")]
    },
    "sax_tenor": {
        "url": "https://newt.phys.unsw.edu.au/music/saxophone/tenor/sounds/",   # B3.wav
        "notes": note_names[note_names.index("B3"):note_names.index("Fsharp7")]   # Need to add A#3
    },
    "sax_alto": {
        "url": "https://newt.phys.unsw.edu.au/music/saxophone/soprano/sounds/",   # Asharp3.wav
        "notes": note_names[note_names.index("Asharp3"):note_names.index("Asharp6")]
    }
}

# Cleanse clarinet notes
cl_notes = note_names[note_names.index("E3"):note_names.index("Csharp7")]
cl_dynamics_notes = ["E3", "F3", "G4", "A6"]
for n in cl_dynamics_notes:
    cl_notes.remove(n)
    # Append each dynamic after each dynamics_note, then add to the notes list
    for dynamic in dynamics:
        cl_notes.append("".join((n, dynamic)))
data["cl"]["notes"] = cl_notes

# Add tenor sax A#3 notes
saxt_extra_notes = ["Bb3%20dynamics/Bb3%20very%20loud",
                    "Bb3%20dynamics/Bb3%20moderately%20loud",
                    "Bb3%20dynamics/Bb3%20moderately%20soft",
                    "Bb3%20dynamics/Bb3%20very%20soft"]
for en in saxt_extra_notes:
    data["sax_tenor"]["notes"].append(en)

#
# Download
#

for name, instrument_dict in data.items():

    base_url = instrument_dict["url"]
    url_append = instrument_dict.get("urlappend", ".wav")
    notes = instrument_dict["notes"]

    dir_name = f"{folder}/{name}"

    # Create directory
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for note in notes:
        # Cleanse the note name for writing to file
        filename = f"{dir_name}/{re.sub(r'[(%20)/]', '_', note)}{url_append}"

        # File exists
        if os.path.exists(filename):
            print(f"{filename} exists. Skipping")
            continue

        wav = requests.get(f"{base_url}{note}{url_append}")

        # Not found
        if wav.status_code == 404:
            print(f"{base_url}{note}{url_append} was not found. Skipping")
            continue
        elif wav.status_code != 200:
            print(f"{base_url}{note}{url_append} gives status code {wav.status_code}. Skipping")
            continue

        # Write to file
        with open(filename, "wb") as f:
            f.write(wav.content)
        print(f"Downloaded {filename}")

print("Done")
