import whisper

def transcribe_audio(file_path):
    # Load the model
    model = whisper.load_model("small")

    # Process the audio file and get the transcription
    result = model.transcribe(file_path)

    # Return the transcription
    return(result["text"])