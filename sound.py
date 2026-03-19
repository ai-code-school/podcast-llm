from kokoro import KPipeline
import soundfile as sf
import numpy as np

pipeline = KPipeline(lang_code='a')  # load once


def generate_voice(text: str, voice: str, filename: str):
    generator = pipeline(text, voice=voice)

    audio_chunks = []
    for _, _, audio in generator:
        audio_chunks.append(audio)

    if not audio_chunks:
        raise ValueError("No audio generated")

    audio_data = np.concatenate(audio_chunks)
    sf.write(filename, audio_data, 24000)


# ✅ af_heart works
# ✅ af_sarah works

HOST_VOICE = "af_heart"
GUEST_VOICE = "am_michael"
# ✅ am_adam works
# Testing am_michael
# ✅ am_michael works
# Testing am_eric
# ✅ am_eric works
# Testing am_liam
# ✅ am_liam works - this

if __name__ == "__main__":
    conversation = [
        ("host", "Why does Agile struggle in hierarchical cultures?"),
        ("guest", "Because decision making is centralized and teams are not empowered."),
        ("host", "So is Agile fundamentally incompatible with such environments?")
    ]

    for i, (speaker, text) in enumerate(conversation):
        voice = HOST_VOICE if speaker == "host" else GUEST_VOICE
        generate_voice(text, voice, f"./gen-audio/chunk_{i}.wav")

    # voices = [
    #     "am_adam", "am_michael", "am_john", "am_david", "am_paul",
    #     "am_eric", "am_liam", "am_noah", "am_james", "am_robert"
    # ]
    #
    # voices = [
    #     "af_heart", "af_sarah", "af_emma",
    #     "af_lily", "af_ava", "af_mia"
    # ]
    #
    # for v in voices:
    #     try:
    #         print(f"Testing {v}")
    #         gen = pipeline("This is a voice test.", voice=v)
    #         for _, _, audio in gen:
    #             pass
    #         print(f"✅ {v} works")
    #     except Exception as e:
    #         print(f"❌ {v} failed")