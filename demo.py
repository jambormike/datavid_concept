import os

from src.audio_to_text.convert import transcribe_audio
from src.text_to_report.generate import llama_wrapper

audio_records_dir = "data/audio_records"
text_transcripts_dir = "data/text_transcripts"
synth_text_transcripts_dir = "data/synth_text_transcripts"
reports_dir = "data/reports"

# Initialize LLM
llm = llama_wrapper()

def transcribe(audio_file_path):
    audio_file_name = os.path.basename(audio_file_path)
    print(f"Transcribing audio file: '{audio_file_name}'")
    
    # Run Whisper
    text = transcribe_audio(audio_file_path)
    
    text_file_name = audio_file_name.rsplit('.', 1)[0] + ".txt"
    text_file_path = os.path.join(text_transcripts_dir, text_file_name)
    with open(text_file_path, 'w') as file:
            file.write(text)
            
    print(f"Text file '{text_file_name}' created/rewritten in '{text_transcripts_dir}'.")
    
    return


def report(llm, text_file_path):
    text_file_name = os.path.basename(text_file_path)
    print(f"Generating report on file '{text_file_name}'.")
    
    # Note that system prompt string has a gap at the beginning.
    if 'cze' in text_file_name:
        system_prompt = " Shrňte tento text jako lékařskou zprávu v českém jazyce."
    else:
        system_prompt = " Summarize this text as a doctor's report."
    
    with open(text_file_path, 'r') as file:
        prompt = file.read()
    
    # Generate report using LLM
    report = llm.invoke(prompt + system_prompt)
    
    report_file_name = text_file_name.rsplit('.', 1)[0] + "_report.txt"
    report_file_path = os.path.join(reports_dir, report_file_name)
    with open(report_file_path, 'w') as file:
        file.write(report)
    
    print(f"Transcript file '{report_file_name}' created rewritten in '{reports_dir}'")
    
    return


if __name__ == "__main__":
    
    for audio_file_name in os.listdir(audio_records_dir):
        if audio_file_name == ".gitkeep":
            continue
        audio_file_path = os.path.join(audio_records_dir, audio_file_name)
        transcribe(audio_file_path)

    for text_file_name in os.listdir(synth_text_transcripts_dir):
        text_file_path = os.path.join(synth_text_transcripts_dir, text_file_name)
        report(llm, text_file_path)
    
    print("END")
