import argparse
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
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_arguments():
    parser = argparse.ArgumentParser(description='AutoCaptioning: Subtitle videos automatically')
    parser.add_argument('--config', type=str, 
                        help='Path to the configuration YAML file', default='./cfg.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if config.get('download') and not config.get('url'):
        parser.error("URL is required when download is set to true")
    
    return config


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

def generate_subtitle_clip(txt, font_size, stroke_width):
    text_clip = TextClip(
        font='Arial-Bold',
        text=txt,  
        font_size=font_size,
        color='white', 
        stroke_color='black', 
        stroke_width=stroke_width,
        text_align='center',
        method='caption'
    ).with_duration(10)
    
    text_clip = text_clip.with_position('center')
    text_clip_w, text_clip_h = text_clip.size
    
    background_clip = ColorClip(size=(text_clip_w + 20, text_clip_h + 10), color=(0, 0, 0), duration=text_clip.duration).with_opacity(0.8)
    
    composite_clip = CompositeVideoClip([background_clip, text_clip.with_position('center')], size=(text_clip_w + 20, text_clip_h + 10))
    
    return composite_clip.with_position('center')

def create_captioned_vid(vid_path, subs_df, save_dir):
    # TODO: this is broken in general, perhaps not compatible with the version
    # of moviepy we are now using (latest)
    video = VideoFileClip(vid_path)
    width, height = video.w, video.h
    
    def generator(txt):
        return TextClip(
            text=txt,   # Try simpler font name
            font_size=int(width/30),  # Increased font size
            stroke_width=2,     # Increased stroke width for better visibility
            color='white',      
            stroke_color='black',
            size=(width, int(height*.35)),  # Slightly larger text area
            method='caption',
        ).with_opacity(0.95)    # Slight transparency for aesthetics

    
    subs = list(zip(zip(subs_df['start'], subs_df['end']), subs_df['text']))
    subtitles = SubtitlesClip(subs, make_textclip=generator)
    
    # Add a semi-transparent black background behind subtitles for better readability
    background = ColorClip(size=(int(width), int(height*.15)), 
                          color=(0,0,0)).\
                          with_opacity(0.6).\
                          with_position(('center', 'bottom'))
    
    final = CompositeVideoClip([video, background, subtitles.with_position(('center','bottom'))])
    final = final.with_duration(video.duration)
    
    output_path = os.path.join(save_dir, 'output_vid.mp4')
    final.write_videofile(output_path, fps=video.fps, remove_temp=True, codec="libx264", audio_codec="aac")
    
    return final

def download_youtube_video(url, aud_opts, vid_opts):
    with YoutubeDL(aud_opts) as ydl:
        ydl.download([url])
    with YoutubeDL(vid_opts) as ydl:
        ydl.download([url])

def process_local_video(input_path, output_video_path, output_audio_path):
    video = VideoFileClip(input_path)
    video.write_videofile(output_video_path)
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
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
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

def export_subtitles(subs_df, format, output_path):
    if format == 'srt':
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, row in subs_df.iterrows():
                f.write(f"{i+1}\n")
                f.write(f"{pd.to_timedelta(row['start'], unit='s').strftime('%H:%M:%S,%f')[:-3]} --> ")
                f.write(f"{pd.to_timedelta(row['end'], unit='s').strftime('%H:%M:%S,%f')[:-3]}\n")
                f.write(f"{row['text']}\n\n")
    elif format == 'vtt':
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for i, row in subs_df.iterrows():
                f.write(f"{pd.to_timedelta(row['start'], unit='s').strftime('%H:%M:%S.%f')[:-3]} --> ")
                f.write(f"{pd.to_timedelta(row['end'], unit='s').strftime('%H:%M:%S.%f')[:-3]}\n")
                f.write(f"{row['text']}\n\n")
    else:
        raise ValueError(f"Unsupported subtitle format: {format}")

def subtitle_video(args):
    # make the experiment directory
    experiment_dir = f'experiments/{args['name']}'
    os.makedirs(experiment_dir, exist_ok=True)
    
    # get the file paths
    input_file = os.path.join(experiment_dir, 'input.mp4')
    audio_file = os.path.join(experiment_dir, 'audio.mp3')
    output_file = os.path.join(experiment_dir, f'output.{args['output_format']}')
    
    aud_opts = {'format': 'mp3/bestaudio/best', 'outtmpl': audio_file}
    vid_opts = {'format': 'mp4/bestvideo/best', 'outtmpl': input_file}
    
    if args['download']:
        download_youtube_video(args['url'], aud_opts, vid_opts)
    else:
        assert args['input_file'], "Input file is required when download is set to false"
        process_local_video(args['input_file'], input_file, audio_file)
    
    openai_client = setup_openai_client()
    
    if args['use_api']: # new, using the api
        result = transcribe_api(openai_client, audio_file)
    else: # what we already had
        result = transcribe_audio(audio_file, args['model_type'], args['source_language'])
    subs_df = create_subtitles_df(result)
    subs_df.to_csv(os.path.join(experiment_dir, 'subs_transcribed.csv'))
    
    # Translate transcribed subtitles using LLM
    translated_subs = translate_subtitles(subs_df, openai_client, args.get('target_language', 'English'))
    subs_df = pd.read_csv(StringIO(translated_subs))
    subs_df.to_csv(os.path.join(experiment_dir, 'subs_translated.csv'))
    
    if args['llm_refine']: # 
        fixed_subs = fix_subtitles(subs_df, openai_client)
        subs_df = pd.read_csv(StringIO(fixed_subs))
        subs_df.to_csv(os.path.join(experiment_dir, 'subs_auto_edited.csv'))
    
    if args['output_format'] == 'mp4':
        create_captioned_vid(input_file, subs_df, experiment_dir)
    else:
        export_subtitles(subs_df, args.output_format, output_file)

def main():
    args = parse_arguments()
    subtitle_video(args)

if __name__ == '__main__':
    main()