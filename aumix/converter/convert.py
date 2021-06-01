"""
convert.py

Convert functions for different audio file formats.

@author: Chan Wai Lou
"""


import mido
import music21 as m21
import numpy as np
import itertools


def rough_similarity(a, b):
    return (max(a, b) - min(a, b)) / a


def rough_equal(a, b, tol=0.15):
    return ((max(a, b) - min(a, b)) / a) < tol


def rough_multiple(ref, b, tol=0.15):
    """Check whether ref is roughly a multiple of b."""

    # Roughly the same
    if rough_equal(ref, b, tol):
        return True

    # Roughly a multiple
    # e.g. 200/100 (multiple of 2)
    # e.g. 100/200 (multiple of 0.5), 150/200 (multiple of 0.75)
    multiple_ratio = (max(ref, b) % min(ref, b)) / ref
    return multiple_ratio < tol or ((1/multiple_ratio) % 1) < tol


def sec_per_quarter_ratio(ref_metronome_mark, ref, est, tol=0.15, max_den=5):

    # Iterate through all possible fractions (and multiples) limited by max_den.
    # Use the fraction (or multiple) that gives the closest match to the reference.
    fractions = [[num/den for num in range(1, max_den+1)] for den in range(2, max_den+1)]
    ratios = set(itertools.chain(*fractions)).union(set(range(1, max_den+1)))
    ratios = list(ratios)
    similarities = [rough_similarity(ref, est * ratio) for ratio in ratios]
    closest_idx = np.argmin(similarities)

    if similarities[closest_idx] <= tol:
        # Get the specified BPM and sec per quarter
        ref_sec_per_quarter = ref_metronome_mark.secondsPerQuarter()

        return ref_sec_per_quarter * ( 1 / ratios[closest_idx] )

        # # Estimate is roughly a multiple of ref. Make sec per quarter smaller
        # if est > ref:
        #     return ref_sec_per_quarter * ( 1 / round(est / ref) )
        #
        # # Ref is roughly a multiple of estimate. Make sec per quarter bigger
        # elif est <= ref:
        #     return ref_sec_per_quarter * round(ref / est)

    else:
        raise ValueError(f"There seems to be no relationship between "
                         f"ref {ref} and est {est}.")


def score_to_valued_interval(score, ref_score, ref_metronome_mark,
                             skip_non_measure=False, remove_initial_silence=True,
                             separate_parts=False, list_output=False):

    # --- Check tempo relationship with ref. Is it a multiple of it? ---

    # Check with the number of measures first
    try:
        sec_per_quarter = sec_per_quarter_ratio(ref_metronome_mark,
                                                ref=ref_score.highestTime,
                                                est=score.highestTime)
    except ValueError:
        # Ok, measures don't match. Check with tempos

        # Get tempo from score if score_bpm is not specified.
        curr_tempos = score.metronomeMarkBoundaries()
        curr_metronome_mark = curr_tempos[-1][2]
        curr_bpm = curr_metronome_mark.getQuarterBPM()

        # throw an error if tempos don't match too
        sec_per_quarter = sec_per_quarter_ratio(ref_metronome_mark,
                                                ref=ref_metronome_mark.getQuarterBPM(),
                                                est=curr_bpm)

    # Variables for storage
    final_intervals = []
    final_pitches = []

    # Convert score to comparable format (start_ms, end_ms, frequency)
    intervals = []
    pitches = []
    for stream in score.recurse(streamsOnly=True):
        if skip_non_measure and type(stream) is not m21.stream.Measure:
            continue

        for note in stream.notes:
            true_note_offset = stream.offset + note.offset
            start_ms = true_note_offset * sec_per_quarter
            end_ms = (true_note_offset + note.duration.quarterLength) * sec_per_quarter

            # Skip the note if it has 0 duration (usually happens for trills)
            if start_ms == end_ms:
                continue

            # Expand chords
            if type(note) is m21.note.Note:
                freqs = [note.pitch.frequency]
            elif type(note) is m21.chord.Chord:
                freqs = [p.frequency for p in note.pitches]
            else:
                freqs = []
                print("No frequency in " + note)

            for freq in freqs:
                intervals.append((start_ms, end_ms))
                pitches.append(freq)

        if separate_parts:
            final_intervals.append(np.array(intervals))
            final_pitches.append(np.array(pitches))

            intervals.clear()
            pitches.clear()

    if separate_parts:

        # Only remove the first silence across all parts
        # (i.e. the global offset is retained)
        if remove_initial_silence:
            silence_time = np.min([fi[0] for fi in final_intervals])
            initial_removed_intervals = [part_interval - silence_time for part_interval in final_intervals]
            final_intervals = initial_removed_intervals

        return final_intervals, final_pitches

    interval_arr = np.array(intervals)
    pitches_arr = np.array(pitches)
    if remove_initial_silence:
        interval_arr -= np.min(interval_arr)

    return ([interval_arr], [pitches_arr]) if list_output else (interval_arr, pitches_arr)


def ref_est_scores_to_valued_interval(ref_score, est_scores, tol=0.15):

    # ---Get reference tempo---

    # Try to use ref tempo
    ref_tempos = ref_score.metronomeMarkBoundaries()
    ref_metronome_mark = ref_tempos[-1][2]

    # If ref_score has default BPM (100), probe all est_scores
    # and use the averaged tempo for tempos that are "roughly the same"
    if ref_metronome_mark.getQuarterBPM() == 100:
        sum_bpms = [est_score.metronomeMarkBoundaries()[-1][2].getQuarterBPM() for est_score in est_scores]
        sum_bpms = sorted(sum_bpms)

        # Split the group at the point where the pairwise difference exceeds a threshold
        group = []
        last_gid = 0
        for i in range(len(sum_bpms)-1):
            # Replace the group if the current group size is larger
            if sum_bpms[i+1] - sum_bpms[i] > sum_bpms[0] * tol and len(sum_bpms[last_gid:i+1]) > len(group):
                group = sum_bpms[last_gid:i+1]
                last_gid = i+1
        avg_bpm = np.average(sum_bpms if len(group) == 0 else group)
        ref_metronome_mark = m21.tempo.MetronomeMark(number=avg_bpm)

    # Convert reference, store each part's score in a list
    ref_intervals, ref_pitches = score_to_valued_interval(ref_score, ref_score,
                                                          ref_metronome_mark,
                                                          skip_non_measure=False,
                                                          separate_parts=True)

    # Convert all estimated parts
    est_intervals = []
    est_pitches = []
    for score in est_scores:
        intervals, pitches = score_to_valued_interval(score, ref_score,
                                                      ref_metronome_mark,
                                                      skip_non_measure=True,
                                                      separate_parts=False,
                                                      list_output=False)
        est_intervals.append(intervals)
        est_pitches.append(pitches)

    # For each part, add the initial silence offset
    for i in range(len(est_intervals)):
        est_intervals[i] += np.min(ref_intervals[i])

    return ref_intervals, ref_pitches, est_intervals, est_pitches, ref_metronome_mark


def midi_ticks_to_ms(ticks, bpm, ppqn):
    return ticks * 60000 / (bpm * ppqn)


def midi_to_valued_interval_ticks(midi):
    # Loop through all events in the firs (info) track to find the SET_TEMPO event.
    tempo_in_ticks = [msg.tempo for msg in midi.tracks[0] if msg.type == "set_tempo"]
    if len(tempo_in_ticks) == 0:
        raise BaseException("No tempo found")

    bpm = (60 * 1000000) / tempo_in_ticks[0]
    ms_per_beat = 1000 / bpm

    msgs = [m for m in midi.tracks[1] if m.type == "note_on" or m.type == "note_off"]
    # ticks_inc = min([m.time for m in msgs if m.time > 0])

    # Perform the conversion until no more events are left in the MIDI track
    curr_ticks = 0
    silence_ticks = 0
    active_notes = dict()
    results = []

    while len(msgs) != 0:

        # Get the first event
        msg = msgs[0]

        # Move time forward by event duration.
        # For note_on, this denotes the amount of time that is silent
        # For note_off, this denotes the amount of time the note is active for
        curr_ticks += msg.time

        # Mark note as active
        if msg.type == "note_on":
            active_notes[msg.note] = curr_ticks        # put note as active (start_ticks)

        # Note is finished. Pop it into the results
        elif msg.type == "note_off":
            start_ticks = active_notes.pop(msg.note)   # This also unmarks the note as active
            note = msg.note
            end_ticks = curr_ticks
            results.append((start_ticks, end_ticks, note))

        msgs.pop(0)

    return results

# MIDI links
# http://midi.mathewvp.com/aboutMidi.htm
# http://www.music.mcgill.ca/~ich/classes/mumt306/StandardMIDIfileformat.html


#
# while len(test_msgs) != 0:
#
#     curr_ticks_inc = ticks_inc
#
#     process_next = True
#     while process_next and len(test_msgs) != 0:
#         msg = test_msgs[0]
#
#         # Check if silence/active is satisfactory
#         if msg.type == "note_on":
#             # check silence
#             if silence_ticks >= msg.time:
#                 active_notes[msg.note] = 0    # put it as active
#                 test_msgs.pop(0)              # pop from events
#                 silence_ticks = 0             # accumulated silence = 0
#                 continue
#         elif msg.type == "note_off":
#             # check active
#             if active_notes[msg.note] >= msg.time:
#                 active_notes.pop(msg.note)    # remove from actives
#                 test_msgs.pop(0)              # pop from events
#                 continue
#
#         process_next = False
#
#     # Was that the last event we processed?
#     if len(test_msgs) == 0:
#         break
#
#     # All in-due notes have been processed. Move time forward
#     msg = test_msgs[0]
#
#     # Will be active after "silence_ticks"
#     if msg.type == "note_on":
#         silence_ticks += curr_ticks_inc
#
#     elif msg.type == "note_off":
#         # Count active ticks for all active notes
#         for note in active_notes.keys():
#             active_notes[note] += curr_ticks_inc
#
#     # What's the current status?
#     print(f"{curr_ticks}\t{librosa.midi_to_hz(list(active_notes.keys()))}")
#
#     # Move onto the next tick increment
#     curr_ticks += ticks_inc


