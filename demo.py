import os

from src.audio_to_text import convert

audio_records_dir = "data/audio_records"
text_transcripts_dir = "data/text_transcripts"

if __name__ == "__main__":
    
    for audio_file_name in os.listdir(audio_records_dir):
        
        audio_file_path = os.path.join(audio_records_dir, audio_file_name)
        text = convert.transcribe_audio(audio_file_path)
        
        text_file_name = audio_file_name.rsplit('.', 1)[0] + ".txt"
        text_file_path = os.path.join(text_transcripts_dir, text_file_name)
        
        with open(text_file_path, 'w') as file:
            file.write(text)
        
        print(f"Text file '{text_file_name}' created/rewritten in '{text_transcripts_dir}'.")
#    
    print("END")
