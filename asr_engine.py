import numpy as np
import torch
import pyaudio
import math
import threading
import time
import wave
import io
import struct
from transcriptions import transcription

class VADSegmentRealTime:
    def __init__(self, sample_rate=8000,voice_confidence=0.80,system_seg_inerval=0.3, user_seg_interval = 0.8, mode="precise", on_text_change=None, on_seg_end=None):
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=False)
        self.on_text_change = on_text_change
        self.on_seg_end = on_seg_end
        self.voice_confidence = voice_confidence
        self.sample_rate = sample_rate
        self.CHANNELS = 1
        self.segments = [] # can be remove in future
        self.segment_voice = []
        self.segment_text =""
        self.segment_duration = 0
        self.texts = []
        self.chunk = [] #one chunk data
        self.chunks = [] # chunks of audio
        self.confidences = []
        self.events = []
        self.is_speaking = False
        self.is_silent = False
        self.mode = mode # saving and precise saving 模式：每个小 segment 只送一次，如果不能 prompt context 效果不好，  precise 精确模式， 整段送。小段分重复送
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
        if self.mode == "saving":
            self.wait_segments_transcripion_complete()
        self.segments_cnt = self.segments_cnt + 1

        previous_text = ""
        if len(self.texts)> 0:
            for i in range(1, min(4, len(self.texts) + 1)):
                previous_text += self.texts[-i]["text"] + " "
            previous_text = previous_text.strip()
            
        pcm_chunks = seg["voice_chunks"]
        self.segment_voice.extend(pcm_chunks)
        self.segment_duration = self.segment_duration + seg["duration"]

        if self.mode == "precise":
            # Convert bytes to int16 arrays
            int16_chunks = [np.frombuffer(chunk, dtype=np.int16) for chunk in self.segment_voice] # whole big segment

            wav_data = self.pcm_to_wav(int16_chunks)
            #print(f"{self.mode}, previous_text: {previous_text}")
            ret = transcription(wav_io=wav_data,prompt=previous_text)
            seg["text"] = ret["text"]
            self.segment_text = ret["text"]
            #print(f"segment text:{self.segment_text} segment duration: {self.segment_duration}")
        else:
            # Convert bytes to int16 arrays
            int16_chunks = [np.frombuffer(chunk, dtype=np.int16) for chunk in pcm_chunks] # whole big segment
            wav_data = self.pcm_to_wav(int16_chunks)
            
            combine_temp_text = " ".join(seg["text"] for seg in self.segments)

            previous_text = previous_text + combine_temp_text
            #print(f"{self.mode}, previous_text: {previous_text}")
            ret = transcription(wav_io=wav_data,prompt=previous_text + combine_temp_text)
            seg["text"] = ret["text"]
            self.segment_text = ret["text"]
            #print(f"segment text:{self.segment_text} segment duration: {self.segment_duration}")

        if self.on_text_change:
            self.on_text_change(seg["text"])
        self.segments.append(seg)
        #print("seg added")

    def add_new_segment(self):
        chunks = self.chunks.copy()
        pos =  {"start_pos": self.events[0]["idx"], "end_pos": self.v_idx}
        seg = {"pos":pos, "duration":(self.v_idx-self.events[0]["idx"])*32/1000,"voice_chunks":chunks, "text":""}
        
        self.chunks.clear()
        self.events.clear()

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
                    #print("found new segment")
                    return self.add_new_segment()
                voice_cnt = voice_cnt + 1

        if voice_cnt == 0 and len(self.events) > self.system_seg: # remove 1 chunk, too long silent
            self.events.pop(0)
            self.chunks.pop(0)

    def wait_segments_transcripion_complete(self):
        while self.segments_cnt > len(self.segments):
            time.sleep(0.05)

    def check_talk_end(self):
         if self.continue_silent_cnt > self.user_seg:
            self.wait_segments_transcripion_complete()

            #print(f"talk finish, segments count:{len(self.segments)}")
            if self.mode == "precise":
                text = self.segment_text
            else:
                text = " ".join(seg["text"] for seg in self.segments)
                
            #combine_texts = " ".join(seg["text"] for seg in self.segments)

            #print(f"segment text: {self.segment_text}")
            start_time = self.segments[0]["pos"]["start_pos"]*32/1000
            end_time = self.segments[-1]["pos"]["end_pos"]*32/1000

            if self.on_seg_end:
                self.on_seg_end({"start_time":start_time, "end_time": end_time, "text": text})

            self.texts.append({"start_time":start_time, "end_time": end_time, "text": text})

            self.segments.clear()
            self.segment_voice.clear()
            self.segments_cnt = 0
            self.segment_duration = 0
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
                self.is_speaking = True
            else:
                self.is_speaking = False
                if len(self.segments) > 0:
                    self.continue_silent_cnt = self.continue_silent_cnt + 1
                event = {"event": "silent", "idx": self.v_idx}

            self.events.append(event)
            self.chunks.append(full_chunk.tobytes())
            self.parse_events()
            self.check_talk_end()
            self.v_idx = self.v_idx + 1

    def is_in_speaking(self):
        if self.is_speaking or self.segments_cnt > 0:
            return True
        return False