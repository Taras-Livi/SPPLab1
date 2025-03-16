import os
import yt_dlp
import whisper
from google.cloud import speech, storage
from pydub import AudioSegment
import pykakasi
import re
import numpy as np
import librosa
import soundfile as sf
from spleeter.separator import Separator

# ВАЖЛИВО! Вказати шлях до JSON-файлу з ключем Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "jp-ua-karaoke-subtitles-2e9b708a8ad6.json"

# 🔧 Константи
YOUTUBE_URL = "https://www.youtube.com/watch?v=WPl10ZrhCtk"
BUCKET_NAME = "jp-to-ua-audio-sub-bucket"
AUDIO_FILE = "audio.wav"
VOCALS_FILE = "vocals.wav"
SAMPLE_RATE = 16000  # Sample rate for Google STT


# Функція для завантаження аудіо з YouTube
def download_audio(youtube_url, output_path="audio.mp3"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    print("✅ Аудіо завантажено")


# Відокремлення вокалу від музики
def separate_vocals(input_audio="audio.mp3", output_dir="separated"):
    print("🎵 Відокремлення вокалу від музики...")

    # Створення директорії для результатів
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ініціалізація розділювача Spleeter
    separator = Separator('spleeter:2stems')

    # Розділення аудіо на вокал та інструментал
    separator.separate_to_file(input_audio, output_dir)

    print("✅ Вокал відокремлено")
    return os.path.join(output_dir, os.path.splitext(os.path.basename(input_audio))[0], "vocals.wav")


# Покращення якості аудіо для розпізнавання
def enhance_audio_for_stt(input_file, output_file=VOCALS_FILE, sr=SAMPLE_RATE,
                          noise_reduce=True, normalize=True):
    print(f"🔊 Покращення якості аудіо для розпізнавання (sample_rate={sr})...")

    # Завантаження аудіо
    y, _ = librosa.load(input_file, sr=sr)

    # Зменшення шуму (простий high-pass filter)
    if noise_reduce:
        # High-pass filter для видалення низькочастотного шуму
        y_filtered = librosa.effects.preemphasis(y, coef=0.97)
        y = y_filtered

    # Нормалізація гучності
    if normalize:
        y = librosa.util.normalize(y)

    # Збереження результату
    sf.write(output_file, y, sr, 'PCM_16')

    print(f"✅ Аудіо оптимізовано для розпізнавання: {output_file}")
    return output_file


# Конвертація в WAV з контролем якості
def convert_to_wav(mp3_path="audio.mp3", wav_path="audio.wav", sample_rate=SAMPLE_RATE,
                   channels=1, bit_depth=16):
    print(f"🔄 Конвертація в WAV: sample_rate={sample_rate}, channels={channels}, bit_depth={bit_depth}")

    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_channels(channels).set_frame_rate(sample_rate)

    # Встановлення bit depth
    if bit_depth == 16:
        audio = audio.set_sample_width(2)  # 2 bytes = 16 bits
    elif bit_depth == 24:
        audio = audio.set_sample_width(3)  # 3 bytes = 24 bits

    audio.export(wav_path, format="wav")
    print(f"✅ Аудіо конвертовано в WAV: {wav_path}")


# Завантаження у GCS
def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Створюємо bucket, якщо він не існує
    if not bucket.exists():
        bucket.create(location="us")
        print(f"✅ Bucket `{bucket_name}` створено!")

    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(f"✅ Файл {source_file_name} завантажено у gs://{bucket_name}/{destination_blob_name}")
    return f"gs://{bucket_name}/{destination_blob_name}"


# Функція розпізнавання мови з покращеними параметрами
def transcribe_audio_gcs(gcs_uri):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)

    # Покращена конфігурація для розпізнавання японської пісні
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="ja-JP",
        # Добавляємо важливі параметри для покращення розпізнавання
        audio_channel_count=1,
        enable_automatic_punctuation=True,
        model="latest_long",  # Використання найновішої моделі для довгих аудіо
        use_enhanced=True,  # Покращена обробка аудіо на стороні Google
        # Додаткові налаштування для пісень
        # Встановлення контексту для розпізнавання пісень
        speech_contexts=[speech_recognition.SpeechContext(
            phrases=["歌詞", "音楽", "歌"],  # Підказки: "lyrics", "music", "song"
        )],
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    print("🕒 Обробка аудіо, зачекайте...")

    response = operation.result(timeout=600)  # Чекаємо до 10 хвилин

    # Збір всіх результатів з часовими мітками
    results = []
    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        transcript = alternative.transcript

        # Додати часові мітки якщо доступні
        if result.result_end_time.seconds > 0 or result.result_end_time.nanos > 0:
            end_time = result.result_end_time.seconds + result.result_end_time.nanos / 1e9
            results.append((transcript.strip(), end_time))
        else:
            results.append((transcript.strip(), None))

    # Об'єднання всіх транскрипцій
    full_transcript = "\n".join([result[0] for result in results])

    # Також повертаємо результати з часовими мітками для майбутнього створення субтитрів
    return full_transcript.strip(), results


# Функція транслітерації японського тексту в romaji з додаванням пробілів
def japanese_to_romaji(japanese_text):
    print("Text before converting to romaji:", japanese_text)

    # Використовуємо новий API для pykakasi
    kks = pykakasi.kakasi()
    result = kks.convert(japanese_text)

    # Збираємо romaji з пробілами між словами
    romaji_parts = []
    for item in result:
        # Використовуємо hepburn romanization
        romaji_parts.append(item['hepburn'])

    # З'єднуємо з пробілами
    romaji_text = " ".join(romaji_parts)

    # Спеціальна обробка для "engrish" (англійських слів у японській)
    english_pattern = re.compile(r'[a-zA-Z]+')
    romaji_text = english_pattern.sub(lambda m: m.group(0), romaji_text)

    print("✅ Romaji з пробілами:", romaji_text)
    return romaji_text


# Покращена таблиця транслітерації romaji → українська з підтримкою багатосимвольних комбінацій
ROMAJI_TO_UA = {
    # Базові японські склади
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

    # Додаткові комбінації
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

    # Подвоєні приголосні
    "kk": "кк", "ss": "сс", "tt": "тт", "pp": "пп",

    # Маленька "tsu" для подвоєних приголосних
    "っ": "っ",

    # Довгі голосні
    "aa": "аа", "ii": "іі", "uu": "уу", "ee": "ее", "oo": "оо",
    "ou": "оу", "ei": "ей",

    # Окремі голосні
    "a": "а", "i": "і", "u": "у", "e": "е", "o": "о",

    # Розділові знаки та спеціальні символи
    " ": " ", ".": ".", ",": ",", "!": "!", "?": "?", "-": "-",
    "'": "'", "\"": "\"", "(": "(", ")": ")"
}


# Функція транслітерації romaji → українська
def romaji_to_ukrainian(romaji_text):
    result = ""
    i = 0
    romaji_text = romaji_text.lower()  # Переводимо текст у нижній регістр для обробки

    while i < len(romaji_text):
        # Спершу перевіряємо тризначні комбінації (для японських приголосних з "ya", "yu", "yo")
        if i < len(romaji_text) - 2:
            trigraph = romaji_text[i:i + 3]
            if trigraph in ROMAJI_TO_UA:
                result += ROMAJI_TO_UA[trigraph]
                i += 3
                continue

        # Потім перевіряємо двозначні комбінації
        if i < len(romaji_text) - 1:
            digraph = romaji_text[i:i + 2]
            if digraph in ROMAJI_TO_UA:
                result += ROMAJI_TO_UA[digraph]
                i += 2
                continue

        # Якщо не знайшли багатосимвольну комбінацію, перевіряємо один символ
        char = romaji_text[i]
        if char in ROMAJI_TO_UA:
            result += ROMAJI_TO_UA[char]
        else:
            # Перевірка на латинські букви (для "engrish")
            if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
                # Якщо це латинська буква, залишаємо її як є (для англійських слів)
                result += char
            else:
                # Інакше залишаємо символ як є
                result += char
        i += 1

    print("✅ Транслітерація:", result)
    return result


# Створення SRT субтитрів із результатів з часовими мітками
def create_srt_subtitles(timed_results, output_file="karaoke_subtitles.srt"):
    print("📝 Створення SRT субтитрів...")

    with open(output_file, "w", encoding="utf-8") as f:
        subtitle_index = 1

        for i, (text, end_time) in enumerate(timed_results):
            if end_time is None:
                continue

            # Примітивна оцінка часу початку (можна покращити)
            start_time = 0 if i == 0 else timed_results[i - 1][1]

            # Форматування часу для SRT (HH:MM:SS,mmm)
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                seconds = seconds % 60
                return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")

            # Запис субтитрів
            f.write(f"{subtitle_index}\n")
            f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
            f.write(f"{text}\n\n")

            subtitle_index += 1

    print(f"✅ Субтитри SRT створені: {output_file}")


# Додаємо додатковий варіант розпізнавання за допомогою Whisper
def transcribe_with_whisper(audio_file, language="ja"):
    print("🔊 Розпізнавання аудіо за допомогою Whisper...")

    # Завантаження моделі Whisper (можна вибрати різні розміри: tiny, base, small, medium, large)
    model = whisper.load_model("medium")

    # Розпізнавання
    result = model.transcribe(audio_file, language=language, verbose=False)

    # Отримання транскрипції та сегментів з часовими мітками
    transcript = result["text"]
    segments = result["segments"]

    # Підготовка результатів з часовими мітками для створення субтитрів
    timed_results = [(segment["text"], segment["end"]) for segment in segments]

    print("✅ Whisper розпізнавання завершено")
    return transcript, timed_results


# Основна функція
def main():
    # Завантаження аудіо з YouTube
    download_audio(YOUTUBE_URL)

    # Відокремлення вокалу від музики
    vocals_file = separate_vocals()

    # Покращення якості аудіо для розпізнавання
    enhanced_vocals = enhance_audio_for_stt(vocals_file)

    # Спроба розпізнавання через Google Cloud STT
    print("\n--- Розпізнавання через Google Cloud Speech-to-Text ---")
    gcs_uri = upload_to_gcs(BUCKET_NAME, enhanced_vocals, "vocals.wav")
    jp_text_google, timed_results_google = transcribe_audio_gcs(gcs_uri)

    # Альтернативне розпізнавання через Whisper
    print("\n--- Розпізнавання через OpenAI Whisper ---")
    jp_text_whisper, timed_results_whisper = transcribe_with_whisper(enhanced_vocals)

    # Порівняння результатів
    print("\n--- Порівняння результатів розпізнавання ---")
    print("Google STT довжина тексту:", len(jp_text_google))
    print("Whisper довжина тексту:", len(jp_text_whisper))

    # Вибір кращого результату (простий підхід - вибрати довший результат)
    if len(jp_text_whisper) > len(jp_text_google) * 1.2:  # Якщо Whisper видав на 20% більше тексту
        print("✅ Вибрано результат Whisper як більш повний")
        jp_text = jp_text_whisper
        timed_results = timed_results_whisper
    else:
        print("✅ Вибрано результат Google STT")
        jp_text = jp_text_google
        timed_results = timed_results_google

    print("\nJapanese text:", jp_text)

    # Транслітерація
    romaji_text = japanese_to_romaji(jp_text)
    ukrainian_text = romaji_to_ukrainian(romaji_text)

    # Створення субтитрів з часовими мітками
    # Оновлюємо результати з транслітерацією
    ukrainian_timed_results = [(romaji_to_ukrainian(japanese_to_romaji(text)), time)
                               for text, time in timed_results]

    # Створення японських субтитрів
    create_srt_subtitles(timed_results, "japanese_subtitles.srt")

    # Створення українських субтитрів
    create_srt_subtitles(ukrainian_timed_results, "ukrainian_subtitles.srt")

    # Збереження повного тексту українською транслітерацією
    with open("karaoke_lyrics.txt", "w", encoding="utf-8") as f:
        f.write(ukrainian_text)

    print("✅ Процес завершено! Створено файли:")
    print("- japanese_subtitles.srt - японські субтитри з часовими мітками")
    print("- ukrainian_subtitles.srt - українські субтитри з часовими мітками")
    print("- karaoke_lyrics.txt - повний текст в українській транслітерації")


if __name__ == "__main__":
    main()