import wfdb
import matplotlib.pyplot as plt

#-----------Notes--------------
#.dat is the actual waveform data
#.hea is how to interpret .dat

#Modified Lead II (MLII): main electrode lead used by MIT-BIH team for arrthymia detection

#QRS Complex: big sharp spike on waveform -> main mechanical event of a heartbeat
    #depolarization of the right and left ventricles of the heart -> causes contraction of large ventricular muscles

#-----------End Notes-----------

record = wfdb.rdrecord("finalDatabase/100") #loads data
annotation = wfdb.rdann("finalDatabase/100", "atr") #annotations for each heartbeat

wfdb.plot_wfdb(record=record, annotation=annotation) #PLOT OF FULL RAW FILE

signals = record.p_signal #extracts the signals
mlii = signals[:, 0] #there's two types, only want MLII
fs = record.fs #sampling frequency = 360 Hz

start = 0 #start freq
secs = 5
end = int(secs * fs) #end freq

plt.figure(figsize = (15,5))
plt.plot(mlii[start:end], label = "MLII Signal")
plt.title(f'Modified Lead II - First {secs} Seconds')
plt.xlabel("Sample Number")
plt.ylabel("mV")
plt.show()
