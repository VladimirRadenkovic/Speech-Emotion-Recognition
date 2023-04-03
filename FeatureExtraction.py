import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
warnings.filterwarnings('ignore')
import numpy as np
import librosa

df = pd.read_csv('out.csv')
X = df['File_Path']
Y = df['Emotion']
Y = Y.reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42, test_size=0.1, shuffle=True)
y_train = y_train.reset_index(drop=True)
X_train = X_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)

#Data Augmentation

def noise(data,random=False,rate=0.035,threshold=0.075):
    if random:
        rate=np.random.random()*threshold
    noise=rate*np.random.uniform()*np.amax(data)
    augmented_data=data+noise*np.random.normal(size=data.shape[0])
    return augmented_data

def shift(data,rate=1000):
    augmented_data=int(np.random.uniform(low=-5,high=5)*rate)
    augmented_data=np.roll(data,augmented_data)
    return augmented_data

def pitch(data,sr,pitch_factor=0.7,random=False):
    if random:
        pitch_factor=np.random.random() * pitch_factor
    return librosa.effects.pitch_shift(data,sr,pitch_factor)

def stretch(data,rate=0.8):
    return librosa.effects.time_stretch(data,rate)

n_fft = 2048
hop_length = 512

def chunks(data, frame_length, hop_length):
    for i in range(0, len(data), hop_length):
        yield data[i:i+frame_length]

def zeroCrossingRate(data, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(y=data,
                                             frame_length=frame_length,
                                             hop_length=hop_length)
    return np.squeeze(zcr)

def energy(data, frame_length=2048, hop_length=512):
    en = np.array([np.sum(np.power(np.abs(data[hop:hop+frame_length]),2)) for hop in range(0, data.shape[0], hop_length)])
    return en/frame_length

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data[0], frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def entropy_of_energy(data, frame_length=2048, hop_length=512):
    energies = energy(data, frame_length, hop_length)
    energies /= np.sum(energies)

    entropy = 0.0
    entropy -= energies * np.log2(energies)
    return entropy

def spc(data, sr, frame_length=2048, hop_length=512):
    spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(spectral_centroid)

def spc_flux(data):
    isSpectrum = data.ndim == 1
    if isSpectrum:
        data = np.expand_dims(data, axis=1)

    X = np.c_[data[:, 0], data]
    af_Delta_X = np.diff(X, 1, axis=1)
    vsf = np.sqrt((np.power(af_Delta_X, 2).sum(axis=0))) / X.shape[0]

    return np.squeeze(vsf) if isSpectrum else vsf


def spc_rollof(data, sr, frame_length=2048, hop_length=512):
    spcrollof = librosa.feature.spectral_rolloff(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(spcrollof)


def chroma_stft(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    stft = np.abs(librosa.stft(data))
    chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sr)
    return np.squeeze(chroma_stft.T) if not flatten else np.ravel(chroma_stft.T)


def mel_spc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mel = librosa.feature.melspectrogram(y=data, sr=sr)
    return np.squeeze(mel.T) if not flatten else np.ravel(mel.T)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc_feature.T) if not flatten else np.ravel(mfcc_feature.T)

def extract_features(data, sr, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                        zeroCrossingRate(data, frame_length, hop_length),
                        rmse((data, frame_length,hop_length)),
                             mfcc(data, frame_length, hop_length)))
    return result



def get_features(path, duration=2.5, offset=0.6):
    data, sample_rate = librosa.load(path, duration=duration, offset=offset,
                                     sr = 8025)
    # without augmentation

    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data, random=True)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))  # stacking vertically

    # data with pitching
    pitched_data = pitch(data, sample_rate, random=True)
    res3 = extract_features(pitched_data, sample_rate)
    result = np.vstack((result, res3))  # stacking vertically

    # data with pitching and white_noise
    new_data = pitch(data, sample_rate, random=True)
    data_noise_pitch = noise(new_data, random=True)
    res3 = extract_features(data_noise_pitch, sample_rate)
    result = np.vstack((result, res3))  # stacking vertically

    return result

def get_features2(path, duration=2.5, offset=0.6):
    data, sample_rate = librosa.load(path, duration=duration, offset=offset,
                                     sr =8025)
    # without augmentation

    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    return result

df = pd.read_csv('Emotions.csv')
X_train_features, y_train_features = [], []
X_test_features, y_test_features = [], []
X_val_features, y_val_features = [], []

print("Feature processing...")
for path, emotion, ind in zip(X_train, y_train, range(df.File_Path.shape[0])):
    features = get_features(path)
    if ind % 100 == 0:
        print(f"{ind} samples has been processed...")
    for ele in features:
        X_train_features.append(ele)
        # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
        y_train_features.append(emotion)
for path, emotion, ind in zip(X_val, y_val, range(df.File_Path.shape[0])):
    features = get_features2(path)
    if ind % 100 == 0:
        print(f"{ind} samples has been processed...")
    X_val_features.append(features)
        # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
    y_val_features.append(emotion)
for path, emotion, ind in zip(X_test, y_test, range(df.File_Path.shape[0])):
    features = get_features2(path)
    if ind % 100 == 0:
        print(f"{ind} samples has been processed...")
    X_test_features.append(features)
        # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
    y_test_features.append(emotion)

print("Done.")

max_len = max(len(arr) for arr in X_train_features+X_test_features+X_val_features)
X_train_features = [np.pad(el, (0, max_len - len(el)), mode='constant') for el in X_train_features]
X_test_features = [np.pad(el, (0,max_len - len(el)), mode='constant') for el in X_test_features]
X_val_features = [np.pad(el, (0, max_len - len(el)), mode='constant') for el in X_val_features]
scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_test_features = scaler.transform(X_test_features)
X_val_features = scaler.transform(X_val_features)




train_path = "TrainData.csv"
test_path = "TestData.csv"
val_path = "ValData.csv"

train_df = pd.DataFrame(X_train_features)
train_df["labels"] = y_train_features
train_df.to_csv(train_path, index=False)
train_df.head()

test_df = pd.DataFrame(X_test_features)
test_df["labels"] = y_test_features
test_df.to_csv(test_path, index=False)
test_df.head()

val_df = pd.DataFrame(X_val_features)
val_df["labels"] = y_val_features
val_df.to_csv(val_path, index=False)
val_df.head()
