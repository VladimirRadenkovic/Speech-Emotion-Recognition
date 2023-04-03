import random

import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import seaborn as sns

warnings.filterwarnings('ignore')

# Paths to folders with datasets-write yours here

Crema_Path = 'Crema'
Ravdess_Path = 'Ravdess\\audio_speech_actors_01-24'
Savee_Path = 'Savee'
Tess_Path = 'Tess'

# Data from all data sets

# Crema
crema_data = []

# os manipulates with paths from operating system
# listdir gives a list of all paths in directory

# crema format 1001_DFA_ANG_XX.wav, third part is name of emotion:
# options SAD, ANG(angry), DIS(disgust), FEA(fear), HAP(happy), NEU(neutral)
for wav in os.listdir(Crema_Path):
    emotion = wav.split('_')[2]
    if emotion == 'SAD':
        crema_data.append(('sad', Crema_Path + '\\' + wav))
    elif emotion == 'ANG':
        crema_data.append(('angry', Crema_Path + '\\' + wav))
    elif emotion == 'DIS':
        crema_data.append(('disgust', Crema_Path + '\\' + wav))
    elif emotion == 'FEA':
        crema_data.append(('fear', Crema_Path + '\\' + wav))
    elif emotion == 'HAP':
        crema_data.append(('happy', Crema_Path + '\\' + wav))
    elif emotion == 'NEU':
        crema_data.append(('neutral', Crema_Path + '\\' + wav))
    else:
        crema_data.append(('unknown', Crema_Path + '\\' + wav))

Crema_df = pd.DataFrame.from_dict(crema_data)
Crema_df.rename(columns = {0: 'Emotion', 1: 'File_Path'}, inplace = True)
Crema_df.head()
# Ravdess

# this dataset has more directories from different actors and the following emotions:
# 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
# there are two new emotions, calm and surprised- hocemo mi da smatramo calm i neutral istim?
# format of file name is '03-01-01-01-01-01-01.wav' so the third part split by '-' is the id of the emotion
ravdness_data = []

for directory in os.listdir(Ravdess_Path):
    actors_wavs = os.listdir(Ravdess_Path + '\\' + directory )
    for wav in actors_wavs:
        emotion = int(wav.split('-')[2])
        ravdness_data.append((emotion, Ravdess_Path + '\\' + directory + '\\' + wav))
Ravdess_df = pd.DataFrame.from_dict(ravdness_data)
Ravdess_df.rename(columns = {0: 'Emotion', 1: 'File_Path'}, inplace = True)
Ravdess_df['Emotion'].replace(
    {1: 'neutral', 2: 'neutral', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'},
    inplace = True)
Ravdess_df.head()

# Savee

# Savee has the following format of file_name JE_a11.wav where the letters after _ indicate the emotion:
# 'a' = 'anger' 'd' = 'disgust' 'f' = 'fear' 'h' = 'happiness' 'n' = 'neutral' 'sa' = 'sadness' 'su' = 'surprise'
# there are no new emotions
savee_data = []

for wav in os.listdir(Savee_Path):
    temp = wav.split('_')[1].split('.')[0]
    emotion = ''.join(i for i in temp if not i.isdigit())
    if emotion == 'a':
        savee_data.append(('angry', Savee_Path + '/' + wav))
    elif emotion == 'd':
        savee_data.append(('disgust', Savee_Path + '/' + wav))
    elif emotion == 'f':
        savee_data.append(('fear', Savee_Path + '/' + wav))
    elif emotion == 'h':
        savee_data.append(('happy', Savee_Path + '/' + wav))
    elif emotion == 'n':
        savee_data.append(('neutral', Savee_Path + '/' + wav))
    elif emotion == 'sa':
        savee_data.append(('sad', Savee_Path + '/' + wav))
    elif emotion == 'su':
        savee_data.append(('surprise', Savee_Path + '/' + wav))
Savee_df = pd.DataFrame.from_dict(savee_data)
Savee_df.rename(columns = {0: 'Emotion', 1: 'File_Path'}, inplace = True)
Savee_df.head()

# Tess
# Tess has subdirectories that are labeled by emotions

tess_data = []

for directory in os.listdir(Tess_Path):
    emotion = directory.split('_')[1].lower()
    if emotion == 'pleasant':
        emotion = 'surprise'
    wavs = os.listdir(Tess_Path + '\\' + directory )
    for wav in wavs:
        tess_data.append((emotion, Tess_Path + '\\'+ directory +'\\'+ wav))

Tess_df = pd.DataFrame.from_dict(tess_data)
Tess_df.rename(columns = {0: 'Emotion', 1: 'File_Path'}, inplace = True)
Tess_df.head()

main_df=pd.concat([Crema_df,Ravdess_df,Savee_df,Tess_df],axis=0)

emotion_names=main_df['Emotion'].unique()

N=100


main_df.to_csv('Emotions.csv', index=False)

plt.figure(figsize=(12,6))
plt.title('Histogram of Samples')
emotions=sns.countplot(x='Emotion',data=main_df,palette='Set2')
emotions.set_xticklabels(emotions.get_xticklabels(),rotation=45)
plt.show()

#Features from wav files

import librosa

#Splitting dataframe on train val i test data
from sklearn.model_selection import train_test_split

Data_train, Data_test= train_test_split(main_df, random_state = 42, test_size = 0.2, shuffle = True)
Data_train, Data_val= train_test_split(Data_train, random_state = 42, test_size = 0.2, shuffle = True)


#plotting audio files
import librosa.display

def wave_plot(data,sr,emotion,color):
    plt.figure(figsize=(12,5))
    plt.title('Waveplot for audio with {} emotion'.format(emotion), size=17)
    librosa.display.waveshow(y=data,sr=sr,color=color)
    plt.show()

def spectogram(data,sr,emotion):
    audio=librosa.stft(data)
    audio_db=librosa.amplitude_to_db(abs(audio))
    plt.figure(figsize=(12,5))
    plt.title('Spectrogram for audio with {} emotion'.format(emotion), size=17)
    librosa.display.specshow(audio_db,sr=sr,x_axis='time',y_axis='hz')
    plt.show()

import numpy as np

audio_path=[]
colors={'disgust':'#804E2D','happy':'#F19C0E','sad':'#478FB8','neutral':'#4CB847','fear':'#7D55AA','angry':'#C00808','surprise':'#EE00FF'}

for emotion in emotion_names:
    path=np.array(main_df['File_Path'][main_df['Emotion']==emotion])[1]
    data,sample_rate=librosa.load(path)
    wave_plot(data,sample_rate,emotion,colors[emotion])
    spectogram(data,sample_rate,emotion)
    audio_path.append(path)

#IPython.display.Audio(audio_path[0])





