import librosa
import math, numpy as np
import numpy as np
import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)
sys.path.append(ROOT_DIR)

def extract_features_multi_mel(y, sr=44100.0, hop=512, nffts=[1024, 2048, 4096], mel_dim=100):
    featuress = []
    for nfft in nffts:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel_dim, n_fft=nfft, hop_length=hop)  # C2 is 65.4 Hz
        features = librosa.power_to_db(mel, ref=np.max)
        featuress.append(features)
    features = np.stack(featuress, axis=1)
    return features

def extract_features_hybrid(y,sr,hop,mel_dim=12,window_mult=1):
    hop -= hop % 32  #  Chroma CQT only accepts hop lengths that are multiples of 32, so this ensures that condition is met
    window = window_mult * hop # Fast Fourier Transform Window Size is a multiple (default 1) of the hop
    y_harm, y_perc = librosa.effects.hpss(y)
    mels = librosa.feature.melspectrogram(y=y_perc, sr=sr,n_fft=window,hop_length=hop,n_mels=mel_dim, fmax=65.4)  # C2 is 65.4 Hz
    cqts = librosa.feature.chroma_cqt(y=y_harm, sr=sr,hop_length= hop,
                                      norm=np.inf, threshold=0, n_chroma=12,
                                      n_octaves=6, fmin=65.4, cqt_mode='full')
    joint = np.concatenate((mels, cqts), axis=0)
    return joint


def extract_features_mel(y, sr, hop,mel_dim=100):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel_dim, hop_length=hop)  # C2 is 65.4 Hz
    features = librosa.power_to_db(mel, ref=np.max)
    return features

def extract_features_envelope(y, sr, hop,mel_dim=100):
    envelope = librosa.onset.onset_strength(y=y,hop_length=hop, n_mels=mel_dim)
    return envelope

def extract_features_chroma(y,sr, state_times):
    #hop = #int((44100 * 60 * beat_discretization) / bpm) Hop length must be a multiple of 2^6
    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr, C=None, fmin=None,
                                            norm=np.inf, threshold=0.0, tuning=None, n_chroma=12,
                                            n_octaves=7, window=None, bins_per_octave=None, cqt_mode='full')
    # Aggregate chroma features between beat events
    # We'll use the median value of each feature between beat frames
    state_frames = librosa.core.time_to_frames(state_times,sr=sr) # Default hop length of 512
    #TODO: CHANGE THIS TO BECOME LIKE HYBRID IF WE ARE TO EVER USE THIS
    beat_chroma = librosa.util.sync(chromagram, state_frames, aggregate=np.median, pad=True, axis=-1)
    return beat_chroma

def extract_features_mfcc(y,sr,state_times):
    mfcc = librosa.feature.mfcc(y=y, sr=sr) # we can add other specified parameters
    state_frames = librosa.core.time_to_frames(state_times,sr=sr)
    beat_mfcc = librosa.util.sync(mfcc, state_frames, aggregate=np.median, pad=True, axis=-1)
    return beat_mfcc
