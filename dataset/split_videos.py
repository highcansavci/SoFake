from moviepy.editor import VideoFileClip
from time import sleep
import glob
import os


if __name__ == "__main__":
    spoof_video_path = r"data/spoof/*.mp4"

    spoof_videos = glob.glob(spoof_video_path)
    for full_video in spoof_videos:
        current_duration = VideoFileClip(full_video).duration
        divide_into_count = current_duration // 10 + 1
        single_duration = current_duration / divide_into_count
        import string
        import random
        N = 7

        while current_duration > single_duration:
            clip = VideoFileClip(full_video).subclip(current_duration - single_duration, current_duration)
            current_duration -= single_duration
            res = ''.join(random.choices(string.ascii_letters, k=N))
            current_video = f"{res}.mp4"
            clip.to_videofile(current_video, codec="libx264", temp_audiofile='temp-audio.m4a', remove_temp=True,
                              audio_codec='aac')

        os.remove(full_video)
