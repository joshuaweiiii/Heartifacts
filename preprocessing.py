import wfdb
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


#-----------Notes--------------
#V = Premature ventricular contraction
#E = Ventricular escape beat
#L = Left bundle branch block beat
#R = Right bundle branch block beat
#F = Fusion of ventricular and normal beat

#N = Normal Beat

#Baseline Drift = small mountains next to giant mountain (heartbeat) in a waveform
#Aliasing = if sampling rate isn't fast enough, high freq waves can seem lower than actually should be
#Nyquist Frequency = half of a signal's sampling rate
    #since our sampling rate is 360Hz, max freq we can record correctly is 180Hz
    #2 samples per cycle needed to tell it's a wave
#Time Domain = shows when things happen
#Frequency Domain = shows what freq are present and how strong they are (FFT)
#High Pass Filter = removes low freq/baseline drift ~ keeps everything above 0.5 Hz
#Band Pass Filter = removes low and high freq ~ keeps everything between 0.5 Hz - 40 Hz

#23367 Heart Attack Beats
#23367 Non Heart Attack Beats

#-----------Functions-----------

def butter_highpass(cutoff, fs, order=4): #Initiating High Pass Filter Function
    nyq = 0.5 * fs # Nyquist frequency
    normal_cutoff = cutoff / nyq  
    b, a = butter(order, normal_cutoff, btype='high', analog=False) 
    return b, a

def highpass_filter(data, cutoff=0.5, fs=360, order=4): #Executing High Pass Filter Function
    b, a = butter_highpass(cutoff, fs, order=order)  
    y = filtfilt(b, a, data)  
    return y

def butter_bandpass(lowcut, highcut, fs, order=4): #Initiating Band Pass Filter Function
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.5, highcut=40, fs=360, order=4): #Executing Band Pass Filter Function
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def plot_fft(beat_window, fs=360): #Fast Fourier Transform Function
    n = len(beat_window)  
    freq = np.fft.rfftfreq(n, d=1/fs)  
    fft_vals = np.fft.rfft(beat_window) 
    fft_power = np.abs(fft_vals)

    plt.figure(figsize=(10, 5))
    plt.plot(freq, fft_power)
    plt.title("Fast Fourier Transform Plot")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Strength of Frequency")
    plt.grid()
    plt.show()

#------------------------Initializing----------------------------------

records_list = [ #all records we want
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234']

ha_labels = ["V", "E", "L", "F", "R"] #heart attack related annotations
nha_labels = ["N"] #normal heart beat

window = 100 #window size plus minus QRS

X = [] #initialize for DF
y = []
record_nums = []

#------------------------Main DF Creation----------------------------------

for record_num in records_list: 
    record_data = wfdb.rdrecord(f"finalDatabase/{record_num}")
    annotation = wfdb.rdann(f"finalDatabase/{record_num}", "atr")

    signals = record_data.p_signal
    mlii = signals[:, 0]

    for sample, symbol in zip(annotation.sample, annotation.symbol): #sample is center of QRS

        if symbol not in ha_labels + nha_labels: #skip beats that aren't what we want
            continue
        if sample - window < 0 or sample + window > len(mlii): #skip beats close to edges
            continue

        beat_window = mlii[sample - window : sample + window] #epoch of 200 samples ~ 555 ms

        beat_window = bandpass_filter(beat_window, lowcut=0.5, highcut=40, fs=360)

        beat_window = (beat_window - np.mean(beat_window)) / np.std(beat_window) #normalize data

        # if len(X) < 5: #fourier transformation of first 5 beat windows
        #     plot_fft(beat_window, fs = 360)

        X.append(beat_window) #append data -> one window per index
        y.append(1 if symbol in ha_labels else 0)
        record_nums.append(record_num)

df = pd.DataFrame(X) #form dataframe
df["Label"] = y
df["Record Number"] = record_num

df_ha = df[df["Label"] == 1] #all heart attack
df_nha = df[df["Label"] == 0] #all non heart attack

df_ha_undersample = df_ha.sample(n = 10000, random_state = 69) #undersample for balancing and less than 100 MB for GitHub
df_nha_undersample = df_nha.sample(n = 10000, random_state = 69) #undersample for balancing

main_df = pd.concat([df_ha_undersample, df_nha_undersample]).sample(frac = 1, random_state = 69) #main df for preprocessing
main_df.to_csv("main_df.csv", index=False)

#------------------------Graph of High Pass Filter----------------------------------

test_record = wfdb.rdrecord(f"finalDatabase/100")
test_annotation = wfdb.rdann(f"finalDatabase/100", "atr")

signals = test_record.p_signal
mlii = signals[:, 0]

# Pick first valid beat in the file
for sample, symbol in zip(test_annotation.sample, test_annotation.symbol):
    if symbol in ha_labels + nha_labels and sample - window >= 0 and sample + window <= len(mlii):
        test_sample = sample
        break

raw_beat = mlii[test_sample - window : test_sample + window]

filtered_beat = highpass_filter(raw_beat, cutoff=0.5, fs=360)

plt.figure(figsize=(12, 6))
plt.plot(raw_beat, label="Raw Data", color='blue')
plt.plot(filtered_beat, label="High Pass Filtered Data", color='red')
plt.title(f"Before vs After High Pass Filter")
plt.xlabel("Sample index")
plt.ylabel("ECG amplitude (mV)")
plt.legend()
plt.grid(True)
plt.show()

#------------------------Graph of Band Pass Filter----------------------------------

test_record = wfdb.rdrecord(f"finalDatabase/100")
test_annotation = wfdb.rdann(f"finalDatabase/100", "atr")

signals = test_record.p_signal
mlii = signals[:, 0]

# Pick first valid beat in the file
for sample, symbol in zip(test_annotation.sample, test_annotation.symbol):
    if symbol in ha_labels + nha_labels and sample - window >= 0 and sample + window <= len(mlii):
        test_sample = sample
        break

raw_beat = mlii[test_sample - window : test_sample + window]

filtered_beat = bandpass_filter(raw_beat, lowcut=0.5, highcut=40, fs=360)

plt.figure(figsize=(12, 6))
plt.plot(raw_beat, label="Raw Data", color='blue')
plt.plot(filtered_beat, label="Band Pass Filtered Data", color='red')
plt.title(f"Before vs After Band Pass Filter")
plt.xlabel("Sample index")
plt.ylabel("ECG amplitude (mV)")
plt.legend()
plt.grid(True)
plt.show()

