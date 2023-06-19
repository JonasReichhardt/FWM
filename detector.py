#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detects onsets, beats and tempo in WAV files.

For usage information, call with --help.

Author: Jan Schlüter
"""

import sys
from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np
import copy as cp
from scipy.io import wavfile
import librosa
try:
    import tqdm
except ImportError:
    tqdm = None

from agent import Agent
from operator import attrgetter

def opts_parser():
    usage =\
"""Detects onsets, beats and tempo in WAV files.
"""
    parser = ArgumentParser(description=usage)
    parser.add_argument('indir',
            type=str,
            help='Directory of WAV files to process.')
    parser.add_argument('outfile',
            type=str,
            help='Output JSON file to write.')
    parser.add_argument('--plot',
            action='store_true',
            help='If given, plot something for every file processed.')
    return parser


def detect_everything(filename, options):
    """
    Computes some shared features and calls the onset, tempo and beat detectors.
    """
    # read wave file (this is faster than librosa.load)
    sample_rate, signal = wavfile.read(filename)

    # convert from integer to float
    if signal.dtype.kind == 'i':
        signal = signal / np.iinfo(signal.dtype).max

    # convert from stereo to mono (just in case)
    if signal.ndim == 2:
        signal = signal.mean(axis=-1)

    # compute spectrogram with given number of frames per second
    fps = 70
    max_bpm = 200
    min_bpm = 50
    hop_length = sample_rate // fps
    spect = librosa.stft(
            signal, n_fft=2048, hop_length=hop_length, window='hann')

    # only keep the magnitude
    magspect = np.abs(spect)

    # compute a mel spectrogram
    melspect = librosa.feature.melspectrogram(
            S=magspect, sr=sample_rate, n_mels=80, fmin=27.5, fmax=8000)

    # compress magnitudes logarithmically
    melspect = np.log1p(1 + 100 * melspect) 

    # apply super flux
    melspect = super_flux(melspect)

    # compute onset detection function
    odf = onset_detection_function(fps, melspect)

    # detect onsets from the onset detection function
    onsets, onsets_idx, onset_energy = detect_onsets(fps, odf)

    # detect tempo from everything we have
    tempo = detect_tempo(fps, odf, min_bpm, max_bpm)

    # TODO use beat detection to detect tempo again (maybe better result?)

    # detect beats from everything we have (including the tempo)
    beats = detect_beats(fps, onsets_idx, tempo, onset_energy)

    # plot some things for easier debugging, if asked for it
    if options.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, sharex=True)
        plt.subplots_adjust(hspace=0.3)
        plt.suptitle(filename)
        axes[0].set_title('melspect')
        axes[0].imshow(melspect, origin='lower', aspect='auto',
                       extent=(0, melspect.shape[1] / fps,
                               -0.5, melspect.shape[0] - 0.5))
        axes[1].set_title('onsets')
        axes[1].plot(np.arange(len(odf)) / fps, odf)
        for position in onsets:
            axes[1].axvline(position, color='tab:orange')
        axes[2].set_title('beats (tempo: %r)' % list(np.round(tempo, 2)))
        axes[2].plot(np.arange(len(odf)) / fps, odf)
        for position in beats:
            axes[2].axvline(position, color='tab:red')
        plt.show()

    return {'onsets': list(np.round(onsets, 3)),
            'beats': list(np.round(beats, 3)),
            'tempo': list(np.round(tempo, 2))}

def super_flux(melspect):
    super_flux_melspect=cp.deepcopy(melspect)
    
    for i in range(len(melspect)):
        for j in range(len(melspect[i])):
            # apply maximum filter
            # window is from i-1 to i+1 and j-1 to j+1
            local_max = 0
            for x in range(i-1, i+1):
                for y in range(j-1, j+1):
                    # set new max value if current entry in melspect is larger
                    local_max = max(local_max, melspect[min(max(0,x),len(melspect))][min(max(0,y),len(melspect[i]))])

            super_flux_melspect[i][j] = local_max

    return super_flux_melspect  

def onset_detection_function(fps, melspect):
    """
    LSFS
    """
    values = []

    # transpose matrix for easier calculation
    # first dimension is now time instead of frequency
    melspect_transp = np.transpose(melspect)

    for idx in range(1, len(melspect_transp)):
        sum_t = np.sum(melspect_transp[idx])
        sum_th = np.sum(melspect_transp[idx-1])
        sum = sum_t - sum_th
        values.append(max(sum, 0))
            
    return values
        


def detect_onsets(odf_rate, odf):
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """
    
    # magic parameters
    max_lb = 5      #w1
    max_up = 5      #w2
    avg_lb = 6      #w3
    avg_up = 6      #w4
    threshold = 3   #delta    
    distance = 3    #w5

    onset_index = []
    onset_energy = []

    # sliding window over odf array
    n = 0
    while n < len(odf):
        # calculate max in window
        # sliding window from max(0,lower_bound) to min(upper_bound,list_length)
        # to get rid of out of bounds
        detection_list = odf[max(0,n-max_lb):min(n+max_up,len(odf))]
        detection_index = np.array(detection_list,dtype=int).argmax()

        # calculate mean in window
        mean_list = odf[max(0,n-avg_lb):min(n+avg_up,len(odf))]
        lcl_mean = sum(mean_list) / len(mean_list) + threshold

        # local max needs to be larger or equal than local mean
        if detection_list[detection_index] >= lcl_mean and detection_list[detection_index] > 0:
           onset_index.append(max(0,n-max_lb)+detection_index)
           onset_energy.append(detection_list[detection_index])
        
        n+=1

    # filter out items being too close together
    onset_index_filtered = []
    onset_energy_filtered = []
    last = 0
    for i in onset_index:
        if i > last + distance:
            onset_index_filtered.append(i)
            onset_energy_filtered.append((i,onset_energy[onset_index.index(i)]))
            last = i
    
    # return sample in time domain and sample index
    return np.array(onset_index_filtered)/odf_rate, onset_index_filtered, dict(onset_energy_filtered)



def detect_tempo(fps, odf, min_bpm, max_bpm):
    """
    Detect tempo using any of the input representations.
    Returns one tempo or two tempo estimations.
    """    
    tau_range = 200
    # compute autocorrelation
    r = auto_correlation(odf, range(tau_range))
        
    # ignore first n and last x elements to only use possible bpm window
    tau_min = round(fps*60/max_bpm)
    tau_max = round(fps*60/min_bpm)
    r_max = np.argmax(r[tau_min:tau_max]) + tau_min
    
    # calculate tempo estimation candidates
    candidate_window=5 
    candidates = []
    candidates.append(r_max)

    # assume other peaks are at tau*2 (r_max*2) or tau/2 (r_max/2) of our guess
    # we search for local maxima in a window around the assumed peaks
    #   to make sure we use the most suitable value
    if r_max/2 >= tau_min:
        win_lb = round(r_max/2-candidate_window)
        win_ub = round(r_max/2+candidate_window)
        candidates.append(np.argmax(r[win_lb:win_ub]) + win_lb)
    
    if r_max*2 <= tau_max:
        win_lb = r_max*2-candidate_window
        win_ub = r_max*2+candidate_window
        candidates.append(np.argmax(r[win_lb:win_ub]) + win_lb)

    # use only two hypotheses
    if len(candidates) > 2:
        candidate_values = [r[i] for i in candidates]
        del candidates[np.argmin(candidate_values)]

    # convert the tau values back to bpm
    tempo = [fps/c*60 for c in candidates]
    return tempo

def auto_correlation(d, lag_range):
    r = []
    for tau in lag_range:
        r_tau = []
        for t in range(len(d)):
            try:   
                r_tau.append(d[t+tau] * d[t])
            except IndexError:
                # print(str(t+tau))
                None
        r.append(sum(r_tau))

    return r

def detect_beats(fps, onsets_idx, tempo, onset_energy):
    """
    Detect beats using any of the input representations.
    Returns the positions of all beats in seconds.
    """
    
    # generate agents
    agents = create_agents(onsets_idx, tempo, fps, onset_energy, 5)

    # new_agent = agents[1].process()

    # agents predictions
    for idx in onsets_idx:
        new_agents = []
        for agent in agents:
            new_agents = agent.process(idx)

        # append new agents
        if len(new_agents) > 0:
            agents.extend(new_agents)
           
        # prune similar agents
        for agent in cp.deepcopy(agents):
            prune_group = list(
                filter(
                    lambda a: a.tempo_hypothesis * 0.95 <= agent.tempo_hypothesis <= a.tempo_hypothesis * 1.05 
                                 and a.beats[-1][0] * 0.95 <= agent.beats[-1][0] <= a.beats[-1][0] * 1.05, 
                    agents
                ))
            
            # check if prune_group is larger than 1
            # (filter always finds the agent himself)
            if len(prune_group) > 1:
                max_score = max(prune_group, key=attrgetter('score')).score
                prune_agents = list(filter(lambda a: a.score < max_score, prune_group))
                # only keep element with max score
                agents = [a for a in agents if a not in prune_agents]
                None

    best_agent = max(agents, key=attrgetter('score'))
    # todo agent[1] detects all beats but has worse score -> why???
    return [beat[0]/fps for beat in best_agent.beats]

def create_agents(onsets_idx, tempo, fps, onset_energy, n):
    """
    generate agents for events in the first n frames
    """
    agents = []
    inner_interval = 5

    for event in onsets_idx[:n]:
        for t in tempo:
            tempo_hypothesis = int(60/t * fps)
            outer_interval = int(tempo_hypothesis * 0.45)
            agents.append(Agent(event, t, tempo_hypothesis, inner_interval, inner_interval, 
                                outer_interval, outer_interval, onsets_idx, onset_energy, 
                                0, [], tempo[0]))

    return agents

def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()

    # iterate over input directory
    indir = Path(options.indir)
    infiles = list(indir.glob('*.wav'))
    if tqdm is not None:
        infiles = tqdm.tqdm(infiles, desc='File')
    results = {}
    for idx, filename in enumerate(infiles):
        print("Files:", idx, len(infiles))
        results[filename.stem] = detect_everything(filename, options)

    # write output file
    with open(options.outfile, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

