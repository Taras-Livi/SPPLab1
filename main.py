import os
import yt_dlp
import whisper
from google.cloud import speech, storage
from pydub import AudioSegment
import pykakasi
import re
import numpy as np
import soundfile as sf
import subprocess
import torch
import time
import librosa
from datetime import timedelta

# ВАЖЛИВО! Вказати шлях до JSON-файлу з ключем Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "jp-ua-karaoke-subtitles-2e9b708a8ad6.json"

# 🔧 Константи
YOUTUBE_URL = "https://www.youtube.com/watch?v=yI44Sow4iwo"
BUCKET_NAME = "jp-to-ua-audio-sub-bucket"
AUDIO_FILE = "audio.mp3"
WAV_FILE = "audio.wav"
VOCALS_FILE = "vocals.wav"
SAMPLE_RATE = 16000  # Sample rate for Google STT


def download_audio(youtube_url, output_path=AUDIO_FILE):
    """Download audio from YouTube video"""
    print("⬇️ Завантаження аудіо з YouTube...")

    # Remove extension from output_path to avoid duplication
    output_base = os.path.splitext(output_path)[0]

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_base  # Use the base name without extension
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    # The actual output file will have the correct extension added by yt-dlp
    actual_output_path = f"{output_base}.mp3"
    print(f"✅ Аудіо завантажено: {actual_output_path}")
    return actual_output_path


def convert_to_wav(mp3_path=AUDIO_FILE, wav_path=WAV_FILE, sample_rate=SAMPLE_RATE):
    """Convert MP3 to WAV with specific parameters"""
    print(f"🔄 Конвертація у WAV ({sample_rate} Hz)...")
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_channels(1).set_frame_rate(sample_rate)
    audio.export(wav_path, format="wav")
    print(f"✅ Аудіо конвертовано в WAV: {wav_path}")
    return wav_path


def separate_vocals_with_demucs(audio_path=WAV_FILE, output_dir="separated"):
    """Separate vocals from music using Demucs"""
    print("🎤 Відокремлення вокалу від музики за допомогою Demucs...")

    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Install Demucs if it's not installed
    try:
        import demucs
    except ImportError:
        print("📦 Встановлення Demucs...")
        subprocess.run(["pip", "install", "demucs"], check=True)

    # Run Demucs separation
    # Using the latest available model (mdx_extra)
    command = [
        "python", "-m", "demucs.separate",
        "-n", "mdx_extra",  # Latest model as of March 2025
        "--two-stems", "vocals",
        "-o", output_dir,
        audio_path
    ]

    try:
        subprocess.run(command, check=True)
        print("✅ Відокремлення вокалу завершено успішно")

        # Find the vocals file (Demucs creates a specific directory structure)
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        vocals_path = os.path.join(output_dir, "mdx_extra", audio_name, "vocals.wav")

        # Check if file exists and copy it to a known location with resampling
        if os.path.exists(vocals_path):
            # Load and resample to desired sample rate
            y, sr = librosa.load(vocals_path, sr=SAMPLE_RATE)
            sf.write(VOCALS_FILE, y, SAMPLE_RATE)
            print(f"✅ Вокал збережено у {VOCALS_FILE}")
            return VOCALS_FILE
        else:
            print(f"❌ Не вдалося знайти файл вокалу: {vocals_path}")
            return None
    except Exception as e:
        print(f"❌ Помилка під час відокремлення вокалу: {e}")
        return None


def enhance_audio_quality(input_file, output_file="enhanced_vocals.wav",
                          noise_reduction=True,
                          compression=True,
                          normalize=True,
                          highpass_freq=120,
                          lowpass_freq=7500,
                          gain=1.0):
    """
    Enhance audio quality with customizable parameters

    Parameters:
    - input_file: Input audio file path
    - output_file: Output enhanced audio file path
    - noise_reduction: Apply noise reduction (True/False)
    - compression: Apply dynamic range compression (True/False)
    - normalize: Normalize audio levels (True/False)
    - highpass_freq: Highpass filter frequency in Hz
    - lowpass_freq: Lowpass filter frequency in Hz
    - gain: Audio gain multiplier (1.0 = no change)
    """
    print("🔊 Покращення якості аудіо...")

    # Build FFmpeg filter chain based on parameters
    filters = []

    # Add highpass and lowpass filters
    filters.append(f"highpass=f={highpass_freq}")
    filters.append(f"lowpass=f={lowpass_freq}")

    # Add noise reduction if requested
    if noise_reduction:
        filters.append("arnndn=m=./rnnoise-models/bd.rnnn")

    # Add compression if requested
    if compression:
        # Attack time: 10ms, Release: 100ms, Threshold: -25dB, Ratio: 4:1
        filters.append("compand=0.01:0.1:-25/-40|-10/-10|0/-7:6:0:-90:0.2")

    # Add gain adjustment
    if gain != 1.0:
        filters.append(f"volume={gain}")

    # Add normalization if requested
    if normalize:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    # Join all filters
    filter_chain = ",".join(filters)

    # FFmpeg command
    command = [
        'ffmpeg', '-y',
        '-i', input_file,
        '-af', filter_chain,
        '-ar', str(SAMPLE_RATE),
        '-ac', '1',
        output_file
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Аудіо покращено: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"❌ Помилка при обробці аудіо: {e}")
        return input_file


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Upload file to Google Cloud Storage"""
    print(f"☁️ Завантаження {source_file_name} у Google Cloud Storage...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Create bucket if it doesn't exist
    if not bucket.exists():
        bucket.create(location="us")
        print(f"✅ Bucket `{bucket_name}` створено!")

    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(f"✅ Файл {source_file_name} завантажено у gs://{bucket_name}/{destination_blob_name}")
    return f"gs://{bucket_name}/{destination_blob_name}"


def transcribe_audio_google(gcs_uri):
    """Transcribe audio using Google Speech-to-Text with enhanced parameters"""
    print("🎯 Розпізнавання мови через Google Cloud Speech-to-Text...")
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)

    # Enhanced configuration for Japanese song recognition
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="ja-JP",
        # Additional parameters to improve recognition
        audio_channel_count=1,
        enable_automatic_punctuation=True,
        model="latest_long",  # Use the latest model for long audio
        use_enhanced=True,  # Enable enhanced processing
        # Enable word-level timestamps for better subtitles
        enable_word_time_offsets=True,
    )

    # Start long-running recognition operation
    operation = client.long_running_recognize(config=config, audio=audio)
    print("🕒 Обробка аудіо через Google STT, зачекайте...")

    # Wait for completion with timeout
    response = operation.result(timeout=900)  # 15 minutes timeout

    # Process results with timestamps
    full_transcript = ""
    words_with_timestamps = []

    for result in response.results:
        best_alternative = result.alternatives[0]
        full_transcript += best_alternative.transcript + " "

        # Extract word-level timestamps
        for word_info in best_alternative.words:
            word = word_info.word
            start_time = word_info.start_time.total_seconds()
            end_time = word_info.end_time.total_seconds()
            words_with_timestamps.append({
                "word": word,
                "start_time": start_time,
                "end_time": end_time
            })

    print("✅ Google STT розпізнавання завершено")
    return full_transcript.strip(), words_with_timestamps


def transcribe_with_whisper(audio_file, language="ja"):
    """Transcribe audio using OpenAI's Whisper model"""
    print("🤖 Розпізнавання аудіо за допомогою Whisper...")

    # Load the Whisper model (options: tiny, base, small, medium, large)
    # Using medium for better accuracy with songs
    model = whisper.load_model("medium")

    # Run transcription with specific parameters for better song recognition
    result = model.transcribe(
        audio_file,
        language=language,
        verbose=False,
        # Additional parameters to improve song recognition
        initial_prompt="This is a Japanese song lyrics.",
        word_timestamps=True,  # Get word-level timestamps
    )

    # Extract transcript and segments with timestamps
    transcript = result["text"]
    segments = result["segments"]

    # Prepare word-level timestamps for subtitle creation
    words_with_timestamps = []
    for segment in segments:
        if "words" in segment:
            for word_data in segment["words"]:
                words_with_timestamps.append({
                    "word": word_data["word"],
                    "start_time": word_data["start"],
                    "end_time": word_data["end"]
                })

    print("✅ Whisper розпізнавання завершено")
    return transcript, words_with_timestamps


def combine_transcriptions(google_transcript, whisper_transcript):
    """Combine and reconcile transcriptions from multiple sources"""
    print("🔄 Об'єднання результатів транскрипції...")

    # Simple combining approach (can be enhanced with more sophisticated alignment)
    # For now, we'll take the longer transcript as it likely contains more information
    if len(google_transcript) > len(whisper_transcript):
        combined = google_transcript
        print("ℹ️ Використовуємо результат Google STT (довший)")
    else:
        combined = whisper_transcript
        print("ℹ️ Використовуємо результат Whisper (довший)")

    # Here you could implement more sophisticated text alignment algorithms
    # to merge the best parts of both transcriptions

    print(f"✅ Фінальна транскрипція: {combined[:100]}...")
    return combined


def japanese_to_romaji(japanese_text):
    """Convert Japanese text to romaji with proper handling of particles and sokuon"""
    print("🈁 Конвертація японського тексту в romaji...")
    print(f"Текст для конвертації: {japanese_text[:100]}...")

    # First, let's handle the sokuon (っ) correctly
    # We'll mark it with a special token that won't be in the text
    marked_text = japanese_text.replace('っ', '##SOKUON##')

    # Use pykakasi for conversion
    kks = pykakasi.kakasi()
    result = kks.convert(marked_text)

    # Process the results to handle special cases
    romaji_parts = []

    for i, item in enumerate(result):
        orig_word = item['orig']
        romaji_word = item['hepburn']

        # Handle particle は -> wa
        if orig_word == 'は' and (i > 0 or len(result) > 1):
            romaji_word = 'wa'

        # Handle our sokuon marker
        if '##SOKUON##' in romaji_word:
            # Find the character after the sokuon marker
            match = re.search(r'##SOKUON##([bcdfghjklmnpqrstvwxyz])', romaji_word)
            if match:
                consonant = match.group(1)
                # Replace the marker with the doubled consonant
                romaji_word = romaji_word.replace(f'##SOKUON##{consonant}', f'{consonant}{consonant}')
            else:
                # If no consonant follows, just remove the marker
                romaji_word = romaji_word.replace('##SOKUON##', '')

        # Handle long vowel mark (ー)
        if 'ー' in orig_word:
            # Find the vowel before the long mark
            for j in range(len(romaji_word)):
                if j > 0 and romaji_word[j] == '-' and romaji_word[j - 1] in 'aiueo':
                    vowel = romaji_word[j - 1]
                    romaji_word = romaji_word[:j] + vowel + romaji_word[j + 1:]

        romaji_parts.append(romaji_word)

    # Join with spaces
    romaji_text = " ".join(romaji_parts)

    # Special handling for "engrish" (English words in Japanese)
    english_pattern = re.compile(r'[a-zA-Z]+')
    romaji_text = english_pattern.sub(lambda m: m.group(0), romaji_text)

    print(f"✅ Romaji: {romaji_text[:100]}...")
    return romaji_text


# Improved transliteration table from romaji to Ukrainian
ROMAJI_TO_UA = {
    # Basic Japanese syllables
    "ka": "ка", "ki": "кі", "ku": "ку", "ke": "ке", "ko": "ко",
    "ga": "ґа", "gi": "ґі", "gu": "ґу", "ge": "ґе", "go": "ґо",
    "sa": "са", "shi": "ші", "su": "су", "se": "се", "so": "со",
    "za": "дза", "ji": "джі", "zu": "дзу", "ze": "дзе", "zo": "дзо",
    "ta": "та", "chi": "чі", "tsu": "цу", "te": "те", "to": "то",
    "da": "да", "di": "ді", "du": "ду", "de": "де", "do": "до",
    "na": "на", "ni": "ні", "nu": "ну", "ne": "не", "no": "но",
    "ha": "ха", "hi": "хі", "fu": "фу", "he": "хе", "ho": "хо",
    "ba": "ба", "bi": "бі", "bu": "бу", "be": "бе", "bo": "бо",
    "pa": "па", "pi": "пі", "pu": "пу", "pe": "пе", "po": "по",
    "ma": "ма", "mi": "мі", "mu": "му", "me": "ме", "mo": "мо",
    "ya": "я", "yu": "ю", "yo": "йо",
    "ra": "ра", "ri": "рі", "ru": "ру", "re": "ре", "ro": "ро",
    "wa": "ва", "wo": "во", "n": "н",

    # Particle は as "wa"
    "wa": "ва",

    # Additional combinations
    "kya": "кя", "kyu": "кю", "kyo": "кьо",
    "gya": "ґя", "gyu": "ґю", "gyo": "ґьо",
    "sha": "ша", "shu": "шу", "sho": "шьо",
    "ja": "джя", "ju": "джю", "jo": "джьо",
    "cha": "чя", "chu": "чю", "cho": "чьо",
    "nya": "ня", "nyu": "ню", "nyo": "ньо",
    "hya": "хя", "hyu": "хю", "hyo": "хьо",
    "bya": "бя", "byu": "бю", "byo": "бьо",
    "pya": "пя", "pyu": "пю", "pyo": "пьо",
    "mya": "мя", "myu": "мю", "myo": "мьо",
    "rya": "ря", "ryu": "рю", "ryo": "рьо",

    # Double consonants (for sokuon っ)
    "kk": "кк", "ss": "сс", "tt": "тт", "pp": "пп",
    "gg": "ґґ", "zz": "дзз", "dd": "дд", "bb": "бб",
    "mm": "мм", "rr": "рр", "cch": "чч", "ssh": "шш",

    # Long vowels
    "aa": "аа", "ii": "іі", "uu": "уу", "ee": "ее", "oo": "оо",
    "ou": "оу", "ei": "ей",

    # Single vowels
    "a": "а", "i": "і", "u": "у", "e": "е", "o": "о",

    # Punctuation and special characters
    " ": " ", ".": ".", ",": ",", "!": "!", "?": "?", "-": "-",
    "'": "'", "\"": "\"", "(": "(", ")": ")"
}


def romaji_to_ukrainian(romaji_text):
    """Transliterate romaji to Ukrainian using improved mapping"""
    print("🔄 Транслітерація romaji в українську...")

    result = ""
    i = 0
    romaji_text = romaji_text.lower()  # Convert to lowercase for processing

    while i < len(romaji_text):
        # Check for special double consonants first
        if i < len(romaji_text) - 3:
            tetragram = romaji_text[i:i + 4]  # For cases like "cchi"
            if tetragram in ROMAJI_TO_UA:
                result += ROMAJI_TO_UA[tetragram]
                i += 4
                continue

        # Then check trigraphs (like "sha", "chu", etc.)
        if i < len(romaji_text) - 2:
            trigraph = romaji_text[i:i + 3]
            if trigraph in ROMAJI_TO_UA:
                result += ROMAJI_TO_UA[trigraph]
                i += 3
                continue

        # Then check digraphs (including doubled consonants)
        if i < len(romaji_text) - 1:
            digraph = romaji_text[i:i + 2]
            if digraph in ROMAJI_TO_UA:
                result += ROMAJI_TO_UA[digraph]
                i += 2
                continue

        # If no multi-character combination found, check single character
        char = romaji_text[i]
        if char in ROMAJI_TO_UA:
            result += ROMAJI_TO_UA[char]
        else:
            # Check for Latin letters (for "engrish")
            if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
                # Keep Latin letters as is (for English words)
                result += char
            else:
                # Otherwise keep the character as is
                result += char
        i += 1

    print(f"✅ Українська транслітерація: {result[:100]}...")
    return result


def format_time_srt(seconds):
    """Format time in SRT format (HH:MM:SS,mmm)"""
    td = timedelta(seconds=float(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def create_word_groups(words_with_timestamps, group_size=3):
    """Group words together for better subtitle readability"""
    if not words_with_timestamps:
        return []

    groups = []
    current_group = []

    for word_info in words_with_timestamps:
        current_group.append(word_info)

        if len(current_group) >= group_size:
            # Calculate group start and end times
            start_time = current_group[0]["start_time"]
            end_time = current_group[-1]["end_time"]

            # Join words into a phrase
            phrase = " ".join([info["word"] for info in current_group])

            groups.append({
                "text": phrase,
                "start_time": start_time,
                "end_time": end_time
            })

            current_group = []

    # Add any remaining words
    if current_group:
        start_time = current_group[0]["start_time"]
        end_time = current_group[-1]["end_time"]
        phrase = " ".join([info["word"] for info in current_group])

        groups.append({
            "text": phrase,
            "start_time": start_time,
            "end_time": end_time
        })

    return groups


def create_subtitles(japanese_text, words_with_timestamps, output_file_prefix="karaoke_subtitles"):
    """Create subtitles in Japanese, romaji, and Ukrainian"""
    print("📝 Створення субтитрів...")

    # Group words for better readability
    word_groups = create_word_groups(words_with_timestamps, group_size=3)

    # Process each group to generate three versions of subtitles
    japanese_subtitles = []
    romaji_subtitles = []
    ukrainian_subtitles = []

    for i, group in enumerate(word_groups):
        # Get the Japanese text
        jp_text = group["text"]

        # Convert to romaji
        romaji_text = japanese_to_romaji(jp_text)

        # Convert to Ukrainian
        ukrainian_text = romaji_to_ukrainian(romaji_text)

        # Format for SRT
        subtitle_entry = {
            "index": i + 1,
            "start": format_time_srt(group["start_time"]),
            "end": format_time_srt(group["end_time"]),
            "text": jp_text
        }
        japanese_subtitles.append(subtitle_entry)

        # Same timing for romaji
        romaji_entry = subtitle_entry.copy()
        romaji_entry["text"] = romaji_text
        romaji_subtitles.append(romaji_entry)

        # Same timing for Ukrainian
        ukrainian_entry = subtitle_entry.copy()
        ukrainian_entry["text"] = ukrainian_text
        ukrainian_subtitles.append(ukrainian_entry)

    # Write SRT files
    def write_srt(subtitles, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for entry in subtitles:
                f.write(f"{entry['index']}\n")
                f.write(f"{entry['start']} --> {entry['end']}\n")
                f.write(f"{entry['text']}\n\n")

    write_srt(japanese_subtitles, f"{output_file_prefix}_japanese.srt")
    write_srt(romaji_subtitles, f"{output_file_prefix}_romaji.srt")
    write_srt(ukrainian_subtitles, f"{output_file_prefix}_ukrainian.srt")

    print(f"✅ Субтитри збережено у файлах {output_file_prefix}_*.srt")

    # Create a combined trilingual version
    trilingual_subtitles = []
    for i, (jp, ro, ua) in enumerate(zip(japanese_subtitles, romaji_subtitles, ukrainian_subtitles)):
        trilingual_entry = {
            "index": i + 1,
            "start": jp["start"],
            "end": jp["end"],
            "text": f"{jp['text']}\n{ro['text']}\n{ua['text']}"
        }
        trilingual_subtitles.append(trilingual_entry)

    write_srt(trilingual_subtitles, f"{output_file_prefix}_trilingual.srt")
    print(f"✅ Створено тримовні субтитри: {output_file_prefix}_trilingual.srt")

    return {
        "japanese": japanese_subtitles,
        "romaji": romaji_subtitles,
        "ukrainian": ukrainian_subtitles,
        "trilingual": trilingual_subtitles
    }


def analyze_audio_quality(audio_file):
    """Analyze audio quality to help with troubleshooting"""
    print("🔍 Аналіз якості аудіо...")

    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=None)

        # Calculate statistics
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.sqrt(np.mean(y**2))
        db_level = 20 * np.log10(rms) if rms > 0 else -100

        # Calculate signal-to-noise ratio (simple estimation)
        noise_floor = np.mean(np.sort(np.abs(y))[:int(len(y)*0.1)]**2)
        noise_db = 20 * np.log10(np.sqrt(noise_floor)) if noise_floor > 0 else -100
        snr = db_level - noise_db

        print(f"✅ Тривалість аудіо: {duration:.2f} сек")
        print(f"✅ Частота дискретизації: {sr} Hz")
        print(f"✅ Рівень гучності: {db_level:.2f} dB")
        print(f"✅ Приблизне співвідношення сигнал/шум: {snr:.2f} dB")

        return {
            "duration": duration,
            "sample_rate": sr,
            "db_level": db_level,
            "snr": snr
        }
    except Exception as e:
        print(f"❌ Помилка при аналізі аудіо: {e}")
        return None


def main(youtube_url=YOUTUBE_URL, audio_quality_params=None):
    """Main function with parameterized audio quality control"""
    print("🚀 Запуск обробки японської пісні...")

    start_time = time.time()

    # Set default audio quality parameters if not provided
    if audio_quality_params is None:
        audio_quality_params = {
            "noise_reduction": True,
            "compression": True,
            "normalize": True,
            "highpass_freq": 120,
            "lowpass_freq": 7500,
            "gain": 1.5
        }

    # Download audio from YouTube
    mp3_file = download_audio(youtube_url)

    # Convert to WAV
    wav_file = convert_to_wav(mp3_file)

    # Separate vocals from background music
    vocals_file = separate_vocals_with_demucs(wav_file)
    if vocals_file is None:
        print("⚠️ Не вдалося відокремити вокал, використання оригінального аудіо")
        vocals_file = wav_file

    # Analyze original audio quality
    print("📊 Аналіз якості вхідного аудіо:")
    analyze_audio_quality(vocals_file)

    # Enhance audio quality with the specified parameters
    enhanced_file = enhance_audio_quality(
        vocals_file,
        output_file="enhanced_vocals.wav",
        **audio_quality_params
    )

    # Analyze enhanced audio quality
    print("📊 Аналіз якості покращеного аудіо:")
    analyze_audio_quality(enhanced_file)

    # Upload to Google Cloud Storage
    gcs_uri = upload_to_gcs(BUCKET_NAME, enhanced_file, "enhanced_vocals.wav")

    # Transcribe with Google Speech-to-Text
    print("\n1️⃣ Спроба розпізнавання через Google Speech-to-Text")
    google_transcript, google_words = transcribe_audio_google(gcs_uri)

    # Transcribe with Whisper
    print("\n2️⃣ Спроба розпізнавання через OpenAI Whisper")
    whisper_transcript, whisper_words = transcribe_with_whisper(enhanced_file)

    # Combine results from both sources
    combined_transcript = combine_transcriptions(google_transcript, whisper_transcript)

    # Use the words with timestamps from the source that produced the better transcript
    words_with_timestamps = google_words if len(google_transcript) > len(whisper_transcript) else whisper_words

    # Create subtitles in Japanese, romaji, and Ukrainian
    subtitles = create_subtitles(combined_transcript, words_with_timestamps)

    # Print execution time
    execution_time = time.time() - start_time
    print(f"\n✅ Обробка завершена за {execution_time:.2f} секунд")

    # Print summary
    print("\n📋 Підсумок:")
    print(f"- Транскрипція Google STT: {len(google_transcript)} символів")
    print(f"- Транскрипція Whisper: {len(whisper_transcript)} символів")
    print(f"- Фінальна транскрипція: {len(combined_transcript)} символів")
    print(f"- Створено субтитри: {len(subtitles['japanese'])} фраз")

    return {
        "google_transcript": google_transcript,
        "whisper_transcript": whisper_transcript,
        "combined_transcript": combined_transcript,
        "subtitles": subtitles
    }


if __name__ == "__main__":
    # Викликаємо основну функцію з параметрами якості аудіо для експериментів
    # Ці параметри можна змінювати для покращення розпізнавання
    results = main(
        youtube_url=YOUTUBE_URL,
        audio_quality_params={
            "noise_reduction": True,    # Зменшення шуму
            "compression": True,        # Компресія динамічного діапазону
            "normalize": True,          # Нормалізація гучності
            "highpass_freq": 150,       # Частота фільтра високих частот (Гц)
            "lowpass_freq": 7000,       # Частота фільтра низьких частот (Гц)
            "gain": 1.5                 # Підсилення аудіо (1.0 = без змін)
        }
    )

    # Функція для експериментів з параметрами якості аудіо
    def experiment_with_audio_quality():
        print("🧪 Запуск експериментів з параметрами якості аудіо...")

        # Базовий набір параметрів
        base_params = {
            "noise_reduction": True,
            "compression": True,
            "normalize": True,
            "highpass_freq": 150,
            "lowpass_freq": 7000,
            "gain": 1.5
        }

        # Варіанти параметрів для експериментів
        experiments = [
            # Експеримент 1: Базові параметри
            base_params,

            # Експеримент 2: Без шумозаглушення
            {**base_params, "noise_reduction": False},

            # Експеримент 3: Різні частоти фільтрації
            {**base_params, "highpass_freq": 100, "lowpass_freq": 8000},

            # Експеримент 4: Сильніше підсилення
            {**base_params, "gain": 2.0},

            # Експеримент 5: Без компресії
            {**base_params, "compression": False},
        ]

        # Виконання експериментів
        for i, params in enumerate(experiments):
            print(f"\n💡 Експеримент {i+1}:")
            print(f"Параметри: {params}")

            # Створення тимчасового файлу для експерименту
            output_file = f"enhanced_vocals_exp{i+1}.wav"

            # Покращення аудіо з поточними параметрами
            enhanced_file = enhance_audio_quality(
                "vocals.wav",
                output_file=output_file,
                **params
            )

            # Аналіз якості
            analyze_audio_quality(enhanced_file)

            # Додатковий аналіз можна додати тут

        print("\n✅ Експерименти завершено")

    # Відкоментуйте, щоб запустити експерименти:
    # experiment_with_audio_quality()f