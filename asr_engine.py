import numpy as np
import torch
import torchaudio
import matplotlib.pylab as plt
#torchaudio.set_audio_backend("soundfile")
import pyaudio
import math
import threading
import time
import wave
import io
import struct
from transcriptions import send_audio_to_api,groq_send_audio_to_api

class VADSegmentRealTime:
    def __init__(self, sample_rate=8000,voice_confidence=0.80,system_seg_inerval=0.5, user_seg_interval = 1.1):
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=False)
        self.voice_confidence = voice_confidence
        self.sample_rate = sample_rate
        self.CHANNELS = 1
        self.segments = []
        self.chunk = [] #one chunk data
        self.chunks = [] # chunks of audio
        self.confidences = []
        self.events = []
        self.is_speaking = False
        self.is_silent = False
        #self.speaking_cnt = 0
        self.continue_silent_cnt = 0
        self.v_idx = 0
        self.segments_cnt = 0
        
        if sample_rate == 8000:
            self.CHUNK = 256
            self.num_samples = 256
        elif sample_rate == 16000:
            self.CHUNK = 512
            self.num_samples = 512
        else:
            raise ValueError("Unsupported sample rate. Supported rates are 8000 and 16000.")
        self.system_seg = math.floor(system_seg_inerval/(self.CHUNK/self.sample_rate))
        self.user_seg = math.floor(user_seg_interval/(self.CHUNK/self.sample_rate))

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1/32768
        sound = sound.squeeze()  # depends on the use case
        return sound
    
    def pcm_to_wav(self, pcm_chunks):
        audio_data = io.BytesIO()
        FORMAT = pyaudio.paInt16
        with wave.open(audio_data, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
            wf.setframerate(self.sample_rate)
            for chunk in pcm_chunks:
                # Convert the list of integers to a bytes object
                byte_chunk = struct.pack(f'{len(chunk)}h', *chunk)
                wf.writeframes(byte_chunk)
        audio_data.seek(0)
        return audio_data

    def parse_segment_thread(self, seg):
        pcm_chunks = seg["voice_chunks"]
        # Convert bytes to int16 arrays
        int16_chunks = [np.frombuffer(chunk, dtype=np.int16) for chunk in pcm_chunks]
        wav_data = self.pcm_to_wav(int16_chunks)
        ret = send_audio_to_api(wav_data)
        seg["text"] = ret["text"]
        print(ret)
        with open(f"rec_{seg['pos']['start_pos']}-{seg['pos']['end_pos']}.wav", 'wb') as f:
            wav_data.seek(0)
            f.write(wav_data.read())
        self.segments.append(seg)
        print("seg added")

    def add_new_segment(self):
        chunks = self.chunks.copy()
        pos =  {"start_pos": self.events[0]["idx"], "end_pos": self.v_idx,"duration":(self.v_idx-self.events[0]["idx"])*32/1000}
        seg = {"pos":pos, "voice_chunks":chunks, "text":"1"}
        
        self.chunks.clear()
        self.events.clear()

        self.segments_cnt = self.segments_cnt + 1
        # Create a new thread to call parse_segment_thread
        segment_thread = threading.Thread(target=self.parse_segment_thread, args=(seg,))
        segment_thread.start()

    def validate(self, audio_chunk):
        """
        Validate the input audio chunk and return the confidence score.
        
        Args:
            audio_chunk (bytes): The audio data chunk as bytes.
        
        Returns:
            float: The confidence score from the VAD model.
        """
        # Convert the audio chunk to int16 array
        audio_int16 = np.frombuffer(audio_chunk, np.int16)
        
        # Convert int16 array to float32
        audio_float32 = self.int2float(audio_int16)
        
        # Get the confidence score using the model
        confidence = self.model(torch.from_numpy(audio_float32), self.sample_rate).item()
        
        return confidence
    
    def parse_events(self):
        if len(self.events) < self.system_seg:
            return # too short
        
        silent_cnt = 0
        voice_cnt = 0

        for i in range(len(self.events) - 1, -1, -1):
            #print(self.events[i])
            if self.events[i]["event"] == "silent":
                silent_cnt = silent_cnt + 1
            else:
                if voice_cnt == 0 and silent_cnt > self.system_seg:
                    print("found new segment")
                    return self.add_new_segment()
                voice_cnt = voice_cnt + 1

        if voice_cnt == 0 and len(self.events) > self.system_seg: # remove 1 chunk, too long silent
            self.events.pop(0)
            self.chunks.pop(0)
    
    def check_talk_end(self):
         if self.continue_silent_cnt > self.user_seg:
            while self.segments_cnt > len(self.segments):
                time.sleep(0.05)
            print(f"talk finish, segments count:{len(self.segments)}")

            combine_texts = " ".join(seg["text"] for seg in self.segments)
            print("combined:", combine_texts)

            self.segments.clear()
            self.segments_cnt = 0
            self.continue_silent_cnt = 0

    def stream_add(self, audio_frame):
        audio_frame_array = np.frombuffer(audio_frame, dtype=np.int16)
        self.chunk.extend(audio_frame_array)
        while len(self.chunk) >= self.CHUNK:
            full_chunk = np.array(self.chunk[:self.CHUNK], dtype=np.int16)
            self.chunk = self.chunk[self.CHUNK:]
            confidence = self.validate(full_chunk.tobytes())

            if confidence > self.voice_confidence:
                event = {"event": "speaking", "idx": self.v_idx}
                self.continue_silent_cnt = 0
            else:
                if len(self.segments) > 0:
                    self.continue_silent_cnt = self.continue_silent_cnt + 1
                event = {"event": "silent", "idx": self.v_idx}

            self.events.append(event)
            self.chunks.append(full_chunk.tobytes())
            self.parse_events()
            self.check_talk_end()
            self.v_idx = self.v_idx + 1

def main():
    FORMAT = pyaudio.paInt16
    SAMPLE_RATE = 8000
    CHANNELS = 1
    CHUNK = 256
    num_samples = 256

    vad = VADSegmentRealTime(sample_rate=SAMPLE_RATE,)
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

"""

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)

# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound


FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 8000
CHUNK = int(SAMPLE_RATE / 10)
CHUNK = 256
num_samples = 256

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
data = []
voiced_confidences = []

print("Started Recording")
for i in range(0, 1024):
    
    audio_chunk = stream.read(num_samples)
    #print(i,len(audio_chunk))
    
    # in case you want to save the audio later
    data.append(audio_chunk)
    
    audio_int16 = np.frombuffer(audio_chunk, np.int16);

    audio_float32 = int2float(audio_int16)
    
    # get the confidences and add them to the list to plot them later
    new_confidence = model(torch.from_numpy(audio_float32), 8000).item()
    print(i,new_confidence)
    voiced_confidences.append(new_confidence)
    
print("Stopped the recording")

# plot the confidences for the speech
plt.figure(figsize=(20,6))
plt.plot(voiced_confidences)
plt.show()
"""