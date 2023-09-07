def prepare_dl_files():
    out1 = open('data/spoof/download_deepfake.sh', 'w')
    out2 = open('data/no-spoof/download_non_deepfake.sh', 'w')
    file_fake = open('zf_dataset_deepfake.txt', 'r')
    file_real = open('zf_dataset_non_deepfake.txt', 'r')
    youtube_deepfake_list = []
    youtube_real_list = []

    while 1:
        line = file_fake.readline().strip()
        if line == '':
            break
        if 'youtube' in line:
            youtube_deepfake_list.append(line)

    while 1:
        line = file_real.readline().strip()
        if line == '':
            break
        if 'youtube' in line:
            youtube_real_list.append(line)

    youtube_list = sorted(list(set(youtube_deepfake_list)))
    print('Videos to download from youtube: {}'.format(len(youtube_list)))
    for i in range(len(youtube_list)):
        you = youtube_list[i]
        out1.write(
            '"yt-dlp" --download-archive downloaded.txt --id -f "bestvideo[ext=mp4]" {}\n'.format(you))

    youtube_list = sorted(list(set(youtube_real_list)))
    print('Videos to download from youtube: {}'.format(len(youtube_list)))
    for i in range(len(youtube_list)):
        you = youtube_list[i]
        out2.write(
            '"yt-dlp" --download-archive downloaded.txt --id -f "bestvideo[ext=mp4]" {}\n'.format(you))

    out1.close()
    out2.close()


def prepare_extractor_files():
    out1 = open('extract_frames.bat', 'w')
    file_fake = open('zf_dataset_deepfake.txt', 'r')
    file_real = open('zf_dataset_non_deepfake.txt', 'r')

    fake_files = []
    while 1:
        line = file_fake.readline().strip()
        if line == '':
            break
        if 'youtube' in line:
            current_id = line.split('=')[-1]
        else:
            arr = line.split('-')
            fn = 'fake/{}_00_{}_00_{}.mp4'.format(current_id, arr[0].replace(':', '_'), arr[1].replace(':', '_'))
            fake_files.append(fn)
            out1.write('ffmpeg -n -i {}.mp4 -ss {} -to {} -c:v libx264 -preset veryslow -crf 15 {}\n'.
                       format(current_id, arr[0], arr[1], fn))
    print('Fake files: {}'.format(len(fake_files)))

    real_files = []
    while 1:
        line = file_real.readline().strip()
        if line == '':
            break
        if 'youtube' in line:
            current_id = line.split('=')[-1]
        else:
            arr = line.split('-')
            fn = 'real/{}_00_{}_00_{}.mp4'.format(current_id, arr[0].replace(':', '_'), arr[1].replace(':', '_'))
            real_files.append(fn)
            out1.write('ffmpeg -n -i {}.mp4 -ss {} -to {} -c:v libx264 -preset veryslow -crf 15 {}\n'.
                       format(current_id, arr[0], arr[1], fn))
    print('Real files: {}'.format(len(real_files)))

    out1.close()


if __name__ == "__main__":
    prepare_dl_files()
