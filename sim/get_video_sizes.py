import os


TOTAL_VIDEO_CHUNCK = 49
BITRATE_LEVELS = 6
dir_path = os.path.dirname(os.path.realpath(__file__))
VIDEO_PATH = dir_path + '/../video_server/'
VIDEO_FOLDER = 'video'

# assume videos are in ../video_servers/video[1, 2, 3, 4, 5]
# the quality at video5 is the lowest and video1 is the highest


for bitrate in range(BITRATE_LEVELS):
	with open('video_size_' + str(bitrate), 'wb') as f:
		for chunk_num in range(1, TOTAL_VIDEO_CHUNCK+1):
			video_chunk_path = VIDEO_PATH + \
							   VIDEO_FOLDER + \
							   str(BITRATE_LEVELS - bitrate) + \
							   '/' + \
							   str(chunk_num) + \
							   '.m4s'
			chunk_size = os.path.getsize(video_chunk_path)
			f.write(str(chunk_size) + '\n')
