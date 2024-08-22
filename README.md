# Realtime transcription

An ASR engine by using free transcription API style like OpenAI's `v1/audio/transcriptions`. 

I have 2 backends tested [siliconflow](https://cloud.siliconflow.cn?referrer=clxhh3xlg0001rx6r4ywpk4m4) and [groq](https://groq.com). 

You can found their API references from below links:

- [siliconflow](https://docs.siliconflow.cn/reference/createaudiotranscriptions-1?referrer=clxhh3xlg0001rx6r4ywpk4m4)

- [groq](https://console.groq.com/docs/speech-text)

## Quick Run 
** set key **
`cp .env.example .env` edit .env set SILICONFLOW_KEY or GROQ_KEY 

`python mic_example.py` 
This will realtime transcribe your audio input from microphone to text.

## Class Usage
```python
class VADSegmentRealTime:
    def __init__(self, sample_rate=8000,voice_confidence=0.80,system_seg_inerval=0.3, user_seg_interval = 0.8, mode="precise", on_text_change=None, on_seg_end=None):
...
```
- sample_rate: sample rate of your audio input

- voice_confidence: confidence threshold for voice activity detection (VAD)

- system_seg_inerval: minimum interval between segments detected by the system

- user_seg_interval: minimum interval between segments that will be returned to the user

- mode: "precise" or "saving", precise mode is more accurate but slower and consumer more tokens, saving mode is faster but less accurate. "precise" is recommended.

- on_text_change: callback function that is triggered when the text of the current segment changes

- on_seg_end: callback function that is triggered when a user defined segment ends

## Credits

- [silero-vad](https://github.com/snakers4/silero-vad) This project using silero-vad for voice detect and segment.