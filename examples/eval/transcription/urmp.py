


import os
import re
import numpy as np
import music21 as m21
import mir_eval
import pandas as pd

import aumix.io.filename as af
import aumix.converter.convert as aconv


#
# Parameters
#

ids_to_test = range(1, 45)   # 16, 26 is weird, 28 has lots of initial silences

# Allow (i,j) such that the onset of reference note i is within
# onset_tol (percent) of the onset of estimated note j
onset_tol = 0.05

input_path = "D:\\Year 3\\COMP3931 Individual Project\\repo\\data\\URMP\\AnthemScore_xml"
truth_root = "../../../data/URMP/Dataset/"
write_path = "D:\\Year 3\\COMP3931 Individual Project\\repo\\examples\\eval\\transcription"


# Offset values in quarter notes for each piece/part.
manual_offset = {
    1: [0, 0.5],
    2: [0.5, 0],
    3: [0, 0],   # middle offset by 1 beat
#     4: halve duration
#     5: [1.25, 0.5]
#     6: [0, 0.3]
    10: [0, 1.75]
}

stereo_manual_offset = {
    3: 0.25,
#     4: halve duration
#     6: 0.5
}


#
# Computation
#

# Put the xml file names in a dictionary and sort them in lists
xml_filenames_dict = af.id_filename_dict(input_path,
                                         ids=ids_to_test,
                                         regexes=[f"{i}-d=(.*?).xml" for i in ids_to_test],
                                         sort_function=lambda f: int(f.split("d=")[1].split("-")[0]))

true_pattern = re.compile("Sco(.*?).mid")   # Pattern for truth file names
for dir_name, subdir_list, files in os.walk(truth_root):
    # Inside directory: dir_name

    # Find all the parts
    midi_filename = None
    for filename in files:
        if true_pattern.match(filename) is not None:
            midi_filename = os.path.join(dir_name, filename)

    # Skip the folder if no midi file can be found
    if midi_filename is None:
        continue

    # id is the first 2 characters of the dir name
    idx = int(dir_name.split("/")[-1][:2])

    if idx not in ids_to_test:
        continue

    # Load reference midi
    ref_score = m21.converter.parse(midi_filename)

    # Load transcribed scores
    os.chdir(input_path)

    est_stereo_score = m21.converter.parse(f"{idx}-stereo.xml")
    est_scores = [m21.converter.parse(f) for f in xml_filenames_dict[idx]]

    # f0_truth_1 = mir_eval.io.load_ragged_time_series("F0s_1_vn_12_Spring.txt")

    # Convert reference, store each part's score in a list
    ref_intervals, ref_pitches, est_intervals, est_pitches, ref_metronome_mark = \
        aconv.ref_est_scores_to_valued_interval(ref_score, est_scores,
                                                manual_offset=manual_offset.get(idx, None))

    # Join all parts together for another dataset
    ref_intervals_all = np.concatenate(ref_intervals)
    ref_pitches_all = np.concatenate(ref_pitches)
    est_intervals_all = np.concatenate(est_intervals)
    est_pitches_all = np.concatenate(est_pitches)

    # Remove all duplicates
    _, ids = np.unique(np.column_stack((est_intervals_all[:, 0], est_pitches_all)), axis=0, return_index=True)
    est_intervals_unique = est_intervals_all[ids]
    est_pitches_unique = est_pitches_all[ids]

    # Convert stereo score
    stereo_intervals, stereo_pitches = aconv.score_to_valued_interval(est_stereo_score,
                                                                      ref_score,
                                                                      ref_metronome_mark,
                                                                      manual_offset=stereo_manual_offset.get(idx, None),
                                                                      skip_non_measure=False,
                                                                      separate_parts=False,
                                                                      list_output=False)

    precision = np.empty(len(est_scores) + 3)
    recall = np.empty(len(est_scores) + 3)
    f_measure = np.empty(len(est_scores) + 3)
    avg_overlap_ratio = np.empty(len(est_scores) + 3)

    precision[0], recall[0], f_measure[0], avg_overlap_ratio[0] \
        = mir_eval.transcription.precision_recall_f1_overlap(ref_intervals_all, ref_pitches_all,
                                                             stereo_intervals, stereo_pitches,
                                                             offset_ratio=None,
                                                             onset_tolerance=onset_tol)

    precision[1], recall[1], f_measure[1], avg_overlap_ratio[1] \
        = mir_eval.transcription.precision_recall_f1_overlap(ref_intervals_all, ref_pitches_all,
                                                             est_intervals_all, est_pitches_all,
                                                             offset_ratio=None,
                                                             onset_tolerance=onset_tol)

    precision[2], recall[2], f_measure[2], avg_overlap_ratio[2] \
        = mir_eval.transcription.precision_recall_f1_overlap(ref_intervals_all, ref_pitches_all,
                                                             est_intervals_unique, est_pitches_unique,
                                                             offset_ratio=None,
                                                             onset_tolerance=onset_tol)

    # Evaluate each part individually
    for i in range(len(est_scores)):
        precision[i+3], recall[i+3], f_measure[i+3], avg_overlap_ratio[i+3] \
            = mir_eval.transcription.precision_recall_f1_overlap(ref_intervals[i], ref_pitches[i],
                                                                 est_intervals[i], est_pitches[i],
                                                                 offset_ratio=None,
                                                                 onset_tolerance=onset_tol)

    df = pd.DataFrame(data={
        "PieceID": idx,
        "Comparison": ["Stereo", "Separated (Combined)", "Separated (Duplicates Removed)"]
                    + [f"Separated (Part {i+1})" for i in range(len(est_scores))],
        "Precision": precision,
        "Recall": recall,
        "fmeasure": f_measure,
        "overlap": avg_overlap_ratio,
        # "# Notes": [(stereo_pitches).shape[0],
        #             (est_pitches_all).shape[0],
        #             (est_pitches_unique).shape[0]]
    })
    ok_df = df.query("fmeasure > 0.5 and overlap > 0.5")\

    ok_df.to_csv(os.path.join(write_path, "raw.txt"),
                 header=False, index=False, sep="\t", mode="a")




