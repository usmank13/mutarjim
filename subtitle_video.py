import argparse
from io import StringIO
import os

import openai
import pandas as pd
import whisper
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.video.tools.subtitles import SubtitlesClip
from yt_dlp import YoutubeDL

# TODO: Use openai speech to text api instead of local whisper (add the option)
# TODO: improve visuals of the font, etc. 
# TODO: support for longer videos


def parse_arguments():
    parser = argparse.ArgumentParser(description='AutoCaptioning: Subtitle videos automatically')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument('--name', type=str, required=True,
                        help='Name of directory to store files in experiments folder')

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--download', action='store_true',
                             help='Download video from YouTube')
    input_group.add_argument('--input_file', type=str,
                             help='Path to local input video file')
    parser.add_argument('--use_api', action='store_true',)
    parser.add_argument('--url', type=str,
                        help='URL of YouTube video (required if --download is used)')
    parser.add_argument('--model_type', type=str, default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model type (default: base)')
    parser.add_argument('--source_language', type=str, default='arabic',
                        help='Source language of the video (default: arabic)')
    parser.add_argument('--llm_refine', action='store_true',
                        help='Refine subtitles using LLM')
    parser.add_argument('--output_format', type=str, default='mp4',
                        choices=['mp4', 'srt', 'vtt'],
                        help='Output format (default: mp4)')

    args = parser.parse_args()

    # Input validation
    if args.download and not args.url:
        parser.error("--url is required when --download is set")
    if not args.download and not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Local video file '{args.input_file}' not found.")

    return args


def setup_openai_client():
    try:
        return openai.Client()
    except Exception as e:
        raise ConnectionError(f"Failed to set up OpenAI client: {e}")


def fix_subtitles(subs_df, openai_client):
    try:
        prompt = f"""Here are English subtitles translated from Arabic.
        There may be minor mistakes or awkward phrasings. Please refine these English subtitles for better coherence and fluency,
        while staying as true to the original meaning as possible. Do not translate back to Arabic.
        Provide the results in the csv format, with nothing else, ensuring all rules for CSV parsing, such as appropriate
        use of escapes, are met.\n\n{subs_df.to_string()}"""

        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a skilled English language editor, proficient in refining translations from Arabic to English."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise ValueError(f"Error refining subtitles using OpenAI: {e}")


def generate_subtitle_clip(txt, font_size, stroke_width):
    try:
        # Text designing
        text_clip = TextClip(
            txt,
            font='Arial-Bold',
            fontsize=font_size,
            color='white',  # White text
            stroke_color='black',  # black stroke
            stroke_width=stroke_width,
            align='center',
            method='caption'
        ).set_duration(10)

        # padding + background changes
        text_clip = text_clip.set_position('center')
        text_clip_w, text_clip_h = text_clip.size

        # background is partially transparent
        background_clip = ColorClip(size=(text_clip_w + 40, text_clip_h + 20), color=(0, 0, 0),
                                    duration=text_clip.duration).set_opacity(0.6)

        # putting text and background together
        composite_clip = CompositeVideoClip([background_clip, text_clip.set_position('center')],
                                            size=(text_clip_w + 40, text_clip_h + 20))

        return composite_clip.set_position('center')
    except Exception as e:
        raise RuntimeError(f"Error generating subtitle clip: {e}")


def create_captioned_vid(vid_path, subs_df, save_dir):
    try:
        video = VideoFileClip(vid_path)
        width, height = video.w, video.h

        font_size = width // 35
        stroke_width = 2

        generator = lambda txt: TextClip(
            txt,
            font='Arial-Bold',
            fontsize=font_size,
            stroke_width=stroke_width,
            color='white',
            stroke_color='black',
            method='caption'
        ).set_position(('center', 'bottom'), relative=True).margin(
            bottom=20)

        # subtitle data handling
        subs = list(zip(zip(subs_df['start'], subs_df['end']), subs_df['text']))
        subtitles = SubtitlesClip(subs, generator)

        # composite video and subtitles
        final = CompositeVideoClip([video, subtitles.set_position(('center', 'bottom'))])
        final = final.set_duration(video.duration)

        # output file saving
        output_path = os.path.join(save_dir, 'output_vid.mp4')
        final.write_videofile(output_path, fps=video.fps, remove_temp=True, codec="libx264", audio_codec="aac")

        return final
    except Exception as e:
        raise RuntimeError(f"Error creating captioned video: {e}")


def download_youtube_video(url, aud_opts, vid_opts):
    try:
        with YoutubeDL(aud_opts) as ydl:
            ydl.download([url])
        with YoutubeDL(vid_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise ValueError(f"Error downloading video from {url}: {e}")


def process_local_video(input_path, output_video_path, output_audio_path):
    try:
        video = VideoFileClip(input_path)
        video.write_videofile(output_video_path)
        video.audio.write_audiofile(output_audio_path)
    except Exception as e:
        raise RuntimeError(f"Error processing local video '{input_path}': {e}")


# using the local model
def transcribe_audio(audio_path, model_type, lang):
    try:
        model = whisper.load_model(model_type)
        return model.transcribe(audio_path, task='translate', language=lang)
    except Exception as e:
        raise RuntimeError(f"Error during audio transcription: {e}")


# using the api
def transcribe_api(openai_client, audio_path):
    audio_file = open(audio_path, "rb")
    transcription = openai_client.audio.translations.create(
        model="whisper-1", 
        file=audio_file
    )
    return transcription.text

def create_subtitles_df(result):
    try:
        return pd.DataFrame({
            'start': [int(segment['start']) for segment in result['segments']],
            'end': [int(segment['end']) for segment in result['segments']],
            'text': [segment['text'] for segment in result['segments']]
        })
    except Exception as e:
        raise ValueError(f"Error creating subtitles DataFrame: {e}")


def export_subtitles(subs_df, format, output_path):
    try:
        if format == 'srt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, row in subs_df.iterrows():
                    f.write(f"{i + 1}\n")
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
    except IOError as e:
        raise IOError(f"Error writing subtitles to {output_path}: {e}")


def subtitle_video(args):
    # make the experiment directory
    experiment_dir = f'experiments/{args.name}'
    os.makedirs(experiment_dir, exist_ok=True)
    # get the file paths
    input_file = os.path.join(experiment_dir, 'input.mp4')
    audio_file = os.path.join(experiment_dir, 'audio.mp3')
    output_file = os.path.join(experiment_dir, f'output.{args.output_format}')

    aud_opts = {'format': 'mp3/bestaudio/best', 'outtmpl': audio_file}
    vid_opts = {'format': 'mp4/bestvideo/best', 'outtmpl': input_file}

    try:
        if args.download:
            download_youtube_video(args.url, aud_opts, vid_opts)
        else:
            process_local_video(args.input_file, input_file, audio_file)
    
        openai_client = setup_openai_client()
        if args.use_api: # new, using the api
          result = transcribe_api(openai_client, audio_file)
          # we know there's a problem here
          print(result) # this will tell us more about the format in which the api gives us the result
        else: # what we already had
          result = transcribe_audio(audio_file, args.model_type, args.source_language)
          subs_df = create_subtitles_df(result)
          subs_df.to_csv(os.path.join(experiment_dir, 'subs.csv'))

          if args.llm_refine:
              openai_client = setup_openai_client()
              fixed_subs = fix_subtitles(subs_df, openai_client)
              subs_df = pd.read_csv(StringIO(fixed_subs))
              subs_df.to_csv(os.path.join(experiment_dir, 'subs_auto_edited.csv'))

          if args.output_format == 'mp4':
              create_captioned_vid(input_file, subs_df, experiment_dir)
          else:
              export_subtitles(subs_df, args.output_format, output_file)
    except Exception as e:
        print(f"Error processing subtitles: {e}")


def main():
    args = parse_arguments()
    subtitle_video(args)


if __name__ == '__main__':
    main()