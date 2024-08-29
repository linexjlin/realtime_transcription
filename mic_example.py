import pyaudio
from asr_engine import VADSegmentRealTime
import wave
import threading

continue_recording = True

def stop():
    input("Press Enter to stop the recording:")
    global continue_recording
    continue_recording = False

def on_text_change(text):
    print(f"on_text_change:",text)

def on_seg_end(text):
    print(f"on seg end: ",text)

def main():
    global continue_recording
    continue_recording = True

    FORMAT = pyaudio.paInt16
    SAMPLE_RATE = 16000 # only 8000, 16000 support
    CHANNELS = 1
    CHUNK = 512
    num_samples = 512

    vad = VADSegmentRealTime(sample_rate=SAMPLE_RATE,mode="precise",user_seg_interval = 0.8,on_text_change=on_text_change,on_seg_end=on_seg_end)
    data = []
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("Started Recording")
    stop_listener = threading.Thread(target=stop)
    stop_listener.start()
    while continue_recording:
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