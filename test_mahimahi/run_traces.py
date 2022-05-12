import sys
import os
import subprocess
import numpy as np



RUN_SCRIPT = 'run_video.py'
RANDOM_SEED = 42
RUN_TIME = 350 #320  # sec
MM_DELAY = 40   # millisec
MAX_RETRIES = 3
#DEBUG
# RUN_TIME = 50

DEBUG = True


def log(val):
	if DEBUG:
		print("[run_traces] " + str(val))


def main():

	log("main")
	log("args: {}".format(sys.argv))
	trace_path = sys.argv[1]
	abr_algo = sys.argv[2]
	process_id = sys.argv[3]
	ip = sys.argv[4]
	nn_model = sys.argv[5]

	sleep_vec = range(1, 10)  # random sleep second

	files = os.listdir(trace_path)
	# print(files)
	num_files = len(files)
	for i, f in enumerate(files):
		retry = 0
		while True:

			np.random.shuffle(sleep_vec)
			sleep_time = sleep_vec[int(process_id)]

			# running python not from env
			command = 'mm-delay ' + str(MM_DELAY) + \
					' mm-link 12mbps ' + trace_path + f + ' ' +\
					  '/usr/bin/python ' + RUN_SCRIPT + ' ' + ip + ' ' +\
					  abr_algo + ' ' + str(RUN_TIME) + ' ' +\
					  process_id + ' ' + f + ' ' + str(sleep_time) + ' ' + nn_model

			# command = 'mm-delay ' + str(MM_DELAY) + \
			# 		' mm-link 12mbps ' + trace_path + f + ' ' +\
			# 		  'python ' + RUN_SCRIPT + ' ' + ip + ' ' +\
			# 		  abr_algo + ' ' + str(RUN_TIME) + ' ' +\
			# 		  process_id + ' ' + f + ' ' + str(sleep_time)

# mm-delay 40 mm-link 12mbps ../cooked_traces/trace_8748_http---www.amazon.com /usr/bin/python run_video.py 132.68.60.153 RL 320 7 trace_8748_http---www.amazon.com 6

			# print("[run_traces] " + str(command))
			log("file {} / {}".format(i,num_files))

			log(command)

			# TODO: need pipes for loging out, err
			proc = subprocess.Popen(command,
					  stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

			# proc = subprocess.Popen(command, shell=True)


			(out, err) = proc.communicate()
			log("----output-----")
			log(out)

			log("---end of output---")
			log("---err---")
			log(err)
			log("---end of err---")
			# print("[run_traces] " + str(out))
			if out == 'done\n':
				log("recived done in out")
				break
			else:
				retry += 1
				if retry < MAX_RETRIES:
					break
				log("write to chrome_retry_log")
				with open('./chrome_retry_log', 'ab') as log1:
					log1.write(abr_algo + '_' + f + '\n')
					log1.write(out + '\n')
					# hagay
					log1.write(err + '\n')
					log1.flush()



if __name__ == '__main__':
	main()
