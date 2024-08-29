import requests
import io
import os
import time
from utils import wav_to_mp3
from dotenv import load_dotenv
load_dotenv()

def transcription(wav_io,prompt="",format="mp3"):
    voice_io = wav_io
    if format == "mp3":
        voice_io = wav_to_mp3(wav_io=wav_io)

    if os.getenv("GROQ_KEY"):
        #print("GROQ>>>")
        return groq_api(voice_io=voice_io,prompt=prompt,format=format)
    
    if os.getenv("SILICONFLOW_KEY"):
        return siliconflow_api(voice_io=voice_io,format=format)


def siliconflow_api(voice_io, timeout=3, format="wav"): # mp3
    url = "https://api.siliconflow.cn/v1/audio/transcriptions"

    audio_content = voice_io.read()

    file_type = "audio/wav"
    if format == "mp3":
        file_type = "audio/mpeg"

    files = {
        'model': (None, 'iic/SenseVoiceSmall'),
        'file': ('audio.mp3', audio_content, file_type)
    }

    kauthorization_key = os.getenv("SILICONFLOW_KEY")

    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {kauthorization_key}"
    }

    try:
        start_time = time.time()  # Record the start time
        response = requests.post(url, files=files, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise an error for bad status codes
        end_time = time.time()
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        print(f"siliconflow>> {elapsed_time:.2f} seconds")

        return response.json()  # Return the response as a JSON object
    except requests.exceptions.Timeout:
        print("Request timed out after 1 second.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    
    
def groq_api(voice_io, prompt="",timeout=3,format="wav"):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"

    file_type = "audio/wav"
    if format == "mp3":
        file_type = "audio/mpeg"

    voice_io.seek(0)  # Ensure we're at the start of the BytesIO buffer
    audio_content = voice_io.read()

    files = {
        'model': (None, 'whisper-large-v3'),
        'file': ('audio.wav', audio_content, file_type),
        'temperature': (None, '1'),
        'response_format': (None, 'json'),
        'prompt': (None, prompt),
    }
    authorization_key = os.getenv("GROQ_KEY")
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {authorization_key}"
    }

    try:
        start_time = time.time()  # Record the start time
        response = requests.post(url, files=files, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise an error for bad status codes
        end_time = time.time()
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        print(f"groq>> {elapsed_time:.2f} seconds")
        return response.json()  # Return the response as a JSON object
    except requests.exceptions.Timeout:
        print("Request timed out after 1 second.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


# Example usage
#print(groq_send_audio_to_api('xzs-sample.wav'))