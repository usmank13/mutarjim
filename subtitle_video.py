import argparse
from dataclasses import dataclass
from io import StringIO
import os
import torch
import openai
import pandas as pd
import whisper
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.video.tools.subtitles import SubtitlesClip
from yt_dlp import YoutubeDL
import yaml

# TODO: improve visuals of the font, etc. 
# TODO: support for longer videos

@dataclass
class ProjectPaths:
    experiment_dir: str
    input_video: str = None
    audio: str = None
    transcribed_csv: str = None
    translated_csv: str = None
    refined_csv: str = None
    output_video: str = None
    
    def __post_init__(self):
        if self.input_video is None:
            self.input_video = os.path.join(self.experiment_dir, 'input.mp4')
        if self.audio is None:
            self.audio = os.path.join(self.experiment_dir, 'audio.mp3')
        if self.transcribed_csv is None:
            self.transcribed_csv = os.path.join(self.experiment_dir, 'subs_transcribed.csv')
        if self.translated_csv is None:
            self.translated_csv = os.path.join(self.experiment_dir, 'subs_translated.csv')
        if self.refined_csv is None:
            self.refined_csv = os.path.join(self.experiment_dir, 'subs_auto_edited.csv')
        if self.output_video is None:
            self.output_video = os.path.join(self.experiment_dir, 'output_vid.mp4')

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_arguments():
    parser = argparse.ArgumentParser(description='AutoCaptioning: Subtitle videos automatically')
    parser.add_argument('--config', type=str, 
                        help='Path to the configuration YAML file', default='./cfg.yaml')
    
    # New stage-based entry points
    parser.add_argument('--from-transcript', type=str, choices=['transcribed', 'translated', 'refined'],
                        help='Start from existing transcript stage')
    parser.add_argument('--from-audio', action='store_true',
                        help='Start from existing audio file (skip video download/processing)')
    parser.add_argument('--video', type=str,
                        help='Path to video file (required for --from-transcript)')
    parser.add_argument('--transcript', type=str,
                        help='Path to transcript CSV file (required for --from-transcript)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Validation
    if config.get('download') and not config.get('url'):
        parser.error("URL is required when download is set to true")
    
    if args.from_transcript:
        if not args.video:
            parser.error("--video is required when using --from-transcript")
        if not args.transcript:
            parser.error("--transcript is required when using --from-transcript")
    
    return args, config


def setup_openai_client():
    return openai.Client()

def translate_subtitles(subs_df, openai_client, target_lang='English'):
    prompt = f"""Translate the following Arabic subtitles to {target_lang}. 
    Maintain the timing and structure. Provide natural, fluent translations that preserve the original meaning.
    This is from an Islamic lecture. Note that the transcription may have errors due to similar sounding words.
    Use context to correct those where necessary.
    
    IMPORTANT: Return ONLY a CSV with exactly 3 columns: start,end,text
    - start: start time in seconds
    - end: end time in seconds  
    - text: translated text (if text contains commas, wrap in double quotes)
    
    Example format:
    start,end,text
    0,5,"Hello, world"
    5,10,"How are you?"
    
{subs_df.to_string()}"""
    
    completion = openai_client.chat.completions.create(
        model="gpt-5-mini",
        # temperature=0.1,
        messages=[
            {"role": "system", "content": f"You are a skilled translator specializing in Arabic to {target_lang} translation."},
            {"role": "user", "content": prompt}
        ]
    )
    
    response = completion.choices[0].message.content
    print(response)
    
    # Try to extract just the CSV part if there's extra text
    # lines = response.strip().split('\n')
    # csv_lines = []
    # for line in lines:
    #     if ',' in line and not line.startswith('#'):
    #         csv_lines.append(line)
    
    # csv_content = '\n'.join(csv_lines)
    return response

def fix_subtitles(subs_df, openai_client):
    prompt = f"""Here are English subtitles translated from Arabic.
    There may be minor mistakes or awkward phrasings. Please refine these English subtitles for better coherence and fluency,
    while staying as true to the original meaning as possible. Do not translate back to Arabic. It is from an Islamic lecture.
    Provide the results in the csv format, with nothing else, ensuring all rules for CSV parsing, such as appropriate
    use of escapes, are met.\n\n{subs_df.to_string()}"""
    
    # should we set temperature?
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You are a skilled English language editor, proficient in refining translations from Arabic to English."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content


def create_captioned_vid(vid_path, subs_df, save_dir):

    video = VideoFileClip(vid_path)
    width, height = video.w, video.h
    
    # Define subtitle area dimensions
    subtitle_height = int(height * 0.2)  # 20% of video height for subtitle area
    
    # TODO: maybe set size to a fixed percentage of the image
    def generator(txt):
        return TextClip(
            text=txt,   # Try simpler font name
            font_size=int(max(width, height)/30),  # Increased font size
            stroke_width=2,     # Increased stroke width for better visibility
            color='white',      
            stroke_color='black',
            size=(width, subtitle_height),  # Match background height
            method='caption',
        ).with_opacity(0.95)    # Slight transparency for aesthetics

    
    subs = list(zip(zip(subs_df['start'], subs_df['end']), subs_df['text']))
    subtitles = SubtitlesClip(subs, make_textclip=generator)
    
    # Add a semi-transparent black background behind subtitles for better readability
    # Use the same height as subtitle area and position at exact bottom
    background = ColorClip(size=(width, subtitle_height), 
                          color=(0,0,0)).\
                          with_opacity(0.6).\
                          with_position((0, height - subtitle_height))
    
    # Position subtitles to align with background
    final = CompositeVideoClip([
        video, 
        background, 
        subtitles.with_position((0, height - subtitle_height))
    ])
    final = final.with_duration(video.duration)
    
    # Ensure audio is preserved from original video
    if video.audio is not None:
        final = final.with_audio(video.audio)
    
    output_path = os.path.join(save_dir, 'output_vid.mp4')
    final.write_videofile(output_path, fps=video.fps, remove_temp=True, codec="libx264", audio_codec="aac")
    
    return final

def find_downloaded_file(experiment_dir, base_name, extensions):
    """Find the actual file that was downloaded, regardless of extension or double extensions."""
    # First try the exact base_name with single extensions
    for ext in extensions:
        path = os.path.join(experiment_dir, f'{base_name}{ext}')
        if os.path.exists(path):
            return path
    
    # If not found, look for files starting with base_name
    if os.path.exists(experiment_dir):
        for filename in os.listdir(experiment_dir):
            if filename.startswith(base_name) and any(filename.endswith(ext) for ext in extensions):
                # Prefer files with single extensions (e.g., prefer audio.mp3 over audio.mp3.mp3)
                parts = filename.replace(base_name, '', 1).split('.')
                if len(parts) <= 2:  # base_name + one extension
                    return os.path.join(experiment_dir, filename)
        
        # If still not found, return any file starting with base_name (even with double extensions)
        for filename in os.listdir(experiment_dir):
            if filename.startswith(base_name) and any(filename.endswith(ext) for ext in extensions):
                return os.path.join(experiment_dir, filename)
    
    return None

def find_downloaded_video(experiment_dir, base_name='input'):
    """Find the actual video file that was downloaded, regardless of extension."""
    video_extensions = ['.mp4', '.webm', '.mkv', '.avi']
    return find_downloaded_file(experiment_dir, base_name, video_extensions)

def find_downloaded_audio(experiment_dir, base_name='audio'):
    """Find the actual audio file that was downloaded, regardless of extension."""
    audio_extensions = ['.mp3', '.m4a', '.wav', '.opus', '.ogg']
    return find_downloaded_file(experiment_dir, base_name, audio_extensions)

def download_youtube_video(url, combined_opts, aud_opts):
    # Download combined video+audio (yt-dlp handles merging automatically)
    with YoutubeDL(combined_opts) as ydl:
        ydl.download([url])
    
    # Download separate audio for transcription
    with YoutubeDL(aud_opts) as ydl:
        ydl.download([url])

def process_local_video(input_path, output_video_path, output_audio_path):
    video = VideoFileClip(input_path)
    # Preserve audio when writing video file
    video.write_videofile(output_video_path, audio_codec="aac")
    video.audio.write_audiofile(output_audio_path)

# using the local model
def transcribe_audio(audio_path, model_type, lang):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model(model_type, device=device)
    return model.transcribe(audio_path, task='transcribe', language=lang)

# using the api
def transcribe_api(openai_client, audio_path):
    audio_file = open(audio_path, "rb")
    transcription = openai_client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file,
        response_format='verbose_json'
    )
    # Convert the API response to a dictionary format similar to the local model output
    result = {
        "text": transcription.text,
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            }
            for segment in transcription.segments
        ]
    }
    
    return result

def create_subtitles_df(result):
    return pd.DataFrame({
        'start': [int(segment['start']) for segment in result['segments']],
        'end': [int(segment['end']) for segment in result['segments']],
        'text': [segment['text'] for segment in result['segments']]
    })

def _format_timedelta(seconds, sep=','):
    """Format seconds (int/float) into HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT)."""
    total_ms = int(round(seconds * 1000))
    hrs, remainder = divmod(total_ms, 3_600_000)
    mins, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}{sep}{ms:03d}"


def export_subtitles(subs_df, format, output_path):
    if format == 'srt':
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, row in subs_df.iterrows():
                f.write(f"{i+1}\n")
                f.write(f"{_format_timedelta(row['start'], ',')} --> {_format_timedelta(row['end'], ',')}\n")
                f.write(f"{row['text']}\n\n")
    elif format == 'vtt':
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for i, row in subs_df.iterrows():
                f.write(f"{_format_timedelta(row['start'], '.')} --> {_format_timedelta(row['end'], '.')}\n")
                f.write(f"{row['text']}\n\n")
    else:
        raise ValueError(f"Unsupported subtitle format: {format}")

def load_transcript_csv(csv_path):
    return pd.read_csv(csv_path, index_col=0)

def run_from_transcript(paths, config, stage='transcribed'):
    openai_client = setup_openai_client()
    
    # Load transcript from specified stage
    if stage == 'transcribed':
        subs_df = load_transcript_csv(paths.transcribed_csv)
        # Continue with translation
        translated_subs = translate_subtitles(subs_df, openai_client, config.get('target_language', 'English'))
        subs_df = pd.read_csv(StringIO(translated_subs))
        subs_df.to_csv(paths.translated_csv)
        
        # Optional refinement
        if config.get('llm_refine'):
            fixed_subs = fix_subtitles(subs_df, openai_client)
            subs_df = pd.read_csv(StringIO(fixed_subs))
            subs_df.to_csv(paths.refined_csv)
            
    elif stage == 'translated':
        subs_df = load_transcript_csv(paths.translated_csv)
        # Optional refinement only
        if config.get('llm_refine'):
            fixed_subs = fix_subtitles(subs_df, openai_client)
            subs_df = pd.read_csv(StringIO(fixed_subs))
            subs_df.to_csv(paths.refined_csv)
            
    elif stage == 'refined':
        subs_df = load_transcript_csv(paths.refined_csv)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    # Generate final output
    if config.get('output_format') == 'mp4':
        create_captioned_vid(paths.input_video, subs_df, paths.experiment_dir)
    else:
        output_file = os.path.join(paths.experiment_dir, f'output.{config["output_format"]}')
        export_subtitles(subs_df, config['output_format'], output_file)

def run_from_audio(paths, config):
    openai_client = setup_openai_client()
    
    # Transcribe audio
    if config.get('use_api'):
        result = transcribe_api(openai_client, paths.audio)
    else:
        result = transcribe_audio(paths.audio, config.get('model_type'), config.get('source_language'))
    
    subs_df = create_subtitles_df(result)
    subs_df.to_csv(paths.transcribed_csv)
    
    # Continue with rest of pipeline
    run_from_transcript(paths, config, stage='transcribed')

def run_full_pipeline(config):
    # make the experiment directory
    experiment_dir = f'experiments/{config["name"]}'
    os.makedirs(experiment_dir, exist_ok=True)
    paths = ProjectPaths(experiment_dir)
    
    if config['download']:
        # Let yt-dlp merge best video + best audio automatically
        # Use outtmpl without extension to avoid double extensions
        video_base = os.path.join(experiment_dir, 'input')
        audio_base = os.path.join(experiment_dir, 'audio')
        
        combined_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': video_base + '.%(ext)s',
            'merge_output_format': 'mp4',  # Force merge into mp4
        }
        aud_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_base + '.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }]
        }
        
        download_youtube_video(config['url'], combined_opts, aud_opts)
        
        # Find the actual files that were downloaded
        actual_video = find_downloaded_video(experiment_dir, 'input')
        actual_audio = find_downloaded_audio(experiment_dir, 'audio')
        
        if actual_video:
            paths.input_video = actual_video
            print(f"Found downloaded video: {actual_video}")
        else:
            raise FileNotFoundError(f"Could not find downloaded video in {experiment_dir}")
        
        if actual_audio:
            paths.audio = actual_audio
            print(f"Found downloaded audio: {actual_audio}")
        else:
            raise FileNotFoundError(f"Could not find downloaded audio in {experiment_dir}")
    else:
        assert config['input_file'], "Input file is required when download is set to false"
        process_local_video(config['input_file'], paths.input_video, paths.audio)
    
    # Continue from audio
    run_from_audio(paths, config)

def main():
    args, config = parse_arguments()
    
    if args.from_transcript:
        # Setup paths for existing files
        experiment_dir = os.path.dirname(args.transcript)
        paths = ProjectPaths(experiment_dir)
        paths.input_video = args.video
        
        # Set transcript path based on stage
        if args.from_transcript == 'transcribed':
            paths.transcribed_csv = args.transcript
        elif args.from_transcript == 'translated':
            paths.translated_csv = args.transcript
        elif args.from_transcript == 'refined':
            paths.refined_csv = args.transcript
            
        run_from_transcript(paths, config, stage=args.from_transcript)
        
    elif args.from_audio:
        experiment_dir = f'experiments/{config["name"]}'
        os.makedirs(experiment_dir, exist_ok=True)
        paths = ProjectPaths(experiment_dir)
        run_from_audio(paths, config)
        
    else:
        # Full pipeline (existing behavior)
        run_full_pipeline(config)

if __name__ == '__main__':
    main()