import pyaudio
from asr_engine import VADSegmentRealTime
import wave 

def main():
    FORMAT = pyaudio.paInt16
    SAMPLE_RATE = 16000 # only 8000, 16000 support
    CHANNELS = 1
    CHUNK = 256
    num_samples = 256

    vad = VADSegmentRealTime(sample_rate=SAMPLE_RATE,mode="precise")
    data = []
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("Started Recording")
    for i in range(0, 102400):
        audio_chunk = stream.read(num_samples)
        #print(len(audio_chunk))
        #print(i,len(audio_chunk))
        # get the confidences and add them to the list to plot them later
        data.append(audio_chunk)
        vad.stream_add(audio_chunk)
        # Save the recorded data to a WAV file
    wf = wave.open("rec.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(data))
    wf.close()

main()