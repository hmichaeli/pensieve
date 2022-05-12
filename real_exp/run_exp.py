import sys
import os
import subprocess
import numpy as np


RUN_SCRIPT = 'run_video.py'
RANDOM_SEED = 42
RUN_TIME = 280  # sec
# ABR_ALGO = ['fastMPC', 'robustMPC', 'BOLA', 'RL']
ABR_ALGO = [ 'RL']

REPEAT_TIME = 10
#
# def print1(val):
# 	print("[run_exp] ".format(val))

def main():
	print("[run_exp] ABR_ALGO: " + str(ABR_ALGO))
	np.random.seed(RANDOM_SEED)

	with open('./chrome_retry_log', 'wb') as log:
		log.write('chrome retry log\n')
		log.flush()

		for rt in xrange(REPEAT_TIME):
			np.random.shuffle(ABR_ALGO)
			for abr_algo in ABR_ALGO:

				while True:

					script = 'python ' + RUN_SCRIPT + ' ' + \
							  abr_algo + ' ' + str(RUN_TIME) + ' ' + str(rt)
					print("[run_exp] script: " + script)
					proc = subprocess.Popen(script,
							  stdout=subprocess.PIPE, 
							  stderr=subprocess.PIPE, 
							  shell=True)

					(out, err) = proc.communicate()
					print("[run_exp] command out: ")
					print(out)
					if out == 'done\n':
						break
					else:
						log.write(abr_algo + '_' + str(rt) + '\n')
						log.write(out + '\n')
						log.flush()



if __name__ == '__main__':
	main()
