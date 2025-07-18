import numpy as np
import matplotlib
import scipy.signal
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sounddevice as sd
import librosa


# read audio
audio_path = r"C:\Users\ekcgi\PycharmProjects\DSP Assignments\Question 1\Audio Folder\sample-000000.mp3"
audio, Fs = librosa.load(audio_path,sr= None)

# part 1
plt.figure()
plt.plot(audio)
plt.title('Original Signal')
plt.show()
sd.default.samplerate = Fs
sd.play(audio)
sd.wait()

# part 2
number_of_noise_elements = 50 # num of impulse noise
noiseIndex = np.random.choice(len(audio),number_of_noise_elements, replace=False)
noisyAudio = np.copy(audio)
noisyAudioIndex = 1 # selected point amplitudes are 1
plt.figure()
plt.plot(noisyAudio)
plt.title('Noisy Signal')
plt.show()
sd.default.samplerate = Fs
sd.play(noisyAudio)
sd.wait()

# part a
length = 15
stdDev = 3
x = np.linspace(-(length/2), (length/2), length) # Create a 15 point array between -7 and +7
ftr = np.exp(-(x**2)/(2*(stdDev**2))) # source: https://www.geeksforgeeks.org/apply-a-gauss-filter-to-an-image-with-python/ This source is for 2D signals but I've adjusted it for 1D
ftr /= np.sum(ftr) # normalize the filter kernel
filteredSignal_gaussian = np.convolve(noisyAudio,ftr, mode='same')
plt.figure()
plt.plot(filteredSignal_gaussian)
plt.title('Gaussian Filtered Signal')
plt.show()
sd.default.samplerate = Fs
sd.play(filteredSignal_gaussian)
sd.wait()

#part b
filteredSignal_Median = scipy.signal.medfilt(noisyAudio,5)
plt.figure()
plt.plot(filteredSignal_Median)
plt.title('Median Filtered Signal')
plt.show()
sd.default.samplerate = Fs
sd.play(filteredSignal_Median)
sd.wait()