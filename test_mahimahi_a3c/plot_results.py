import os
import numpy as np
import matplotlib.pyplot as plt
import json

# RESULTS_FOLDER = './results/'
NUM_BINS = 100
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 40 # 64 # try to fix bug in line
VIDEO_BIT_RATE = [350, 600, 1000, 2000, 3000]
COLOR_MAP = plt.cm.jet #nipy_spectral, Set1,Paired 
SIM_DP = 'sim_dp'
# SCHEMES = ['BB', 'RB', 'FIXED', 'FESTIVE', 'BOLA', 'RL',  'sim_rl', SIM_DP]

RESULTS_FOLDER = './results/'
PLOTS_FOLDER = './plots/'
# SCHEMES = ['RL']
SCHEMES = ['a3c_orig', 'a3c1', 'a3c2', 'a3c3', 'pretrained']

def get_schemes():
	with open(os.path.join(RESULTS_FOLDER, 'models_dict.json')) as f:
		models_dict = json.load(f)
		schemes = models_dict.keys()
		return schemes, models_dict

def main():
	# SCHEMES, models_dict = get_schemes()
	print("schemes: \n{}".format(SCHEMES))
	time_all = {}
	bit_rate_all = {}
	buff_all = {}
	bw_all = {}
	raw_reward_all = {}

	for scheme in SCHEMES:
		time_all[scheme] = {}
		raw_reward_all[scheme] = {}
		bit_rate_all[scheme] = {}
		buff_all[scheme] = {}
		bw_all[scheme] = {}

	log_files = os.listdir(RESULTS_FOLDER)

	for f in log_files:
		if 'log' not in f:
			print("{} is not a log file - remove from list".format(f))
			log_files.remove(f)

		elif os.stat(os.path.join(RESULTS_FOLDER, f)).st_size == 0:
			print("{}is empty - remove from list".format(f))
			log_files.remove(f)



	print("parse all log files")
	broken_files = []
	for log_file in log_files:
		try:
			time_ms = []
			bit_rate = []
			buff = []
			bw = []
			reward = []

			print log_file

			with open(RESULTS_FOLDER + log_file, 'rb') as f:
				if SIM_DP in log_file:
					for line in f:
						parse = line.split()
						if len(parse) == 1:
							reward = float(parse[0])
						elif len(parse) >= 6:
							time_ms.append(float(parse[3]))
							bit_rate.append(VIDEO_BIT_RATE[int(parse[6])])
							buff.append(float(parse[4]))
							bw.append(float(parse[5]))

				else:
					for line in f:
						parse = line.split()
						if len(parse) <= 1:
							break
						time_ms.append(float(parse[0]))
						bit_rate.append(int(parse[1]))
						buff.append(float(parse[2]))
						bw.append(float(parse[4]) / float(parse[5]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
						reward.append(float(parse[6]))

			if SIM_DP in log_file:
				time_ms = time_ms[::-1]
				bit_rate = bit_rate[::-1]
				buff = buff[::-1]
				bw = bw[::-1]

			time_ms = np.array(time_ms)
			time_ms -= time_ms[0]

			# print log_file

			for scheme in SCHEMES:
				if scheme in log_file:
					time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
					bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
					buff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = buff
					bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
					raw_reward_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = reward
					break
		except:
			broken_files.append(log_file)

	if len(broken_files) != 0:
		print("broken files " , broken_files)
		exit()
	print("done parsing log files")
	# ---- ---- ---- ----
	# Reward records
	# ---- ---- ---- ----
		
	log_file_all = []
	reward_all = {}
	for scheme in SCHEMES:
		reward_all[scheme] = []
	print("time all {}".format(time_all))
	for l in time_all[SCHEMES[0]]:
		schemes_check = True
		for scheme in SCHEMES:
			if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
				schemes_check = False
				break
		if schemes_check:
			log_file_all.append(l)
			for scheme in SCHEMES:
				if scheme == SIM_DP:
					reward_all[scheme].append(raw_reward_all[scheme][l])
				else:
					reward_all[scheme].append(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN]))

	mean_rewards = {}
	for scheme in SCHEMES:
		print("scheme: {} rewards_len: {}".format(scheme, len(reward_all[scheme])))
		mean_rewards[scheme] = np.mean(reward_all[scheme])
	print("mean rewards per scheme:\n {}".format(mean_rewards))

	print("show total reward graph: ")
	fig = plt.figure()
	ax = fig.add_subplot(111)

	for scheme in SCHEMES:
		ax.plot(reward_all[scheme])
	
	SCHEMES_REW = []
	for scheme in SCHEMES:
		SCHEMES_REW.append(scheme + ': ' + str(mean_rewards[scheme]))

	colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	for i,j in enumerate(ax.lines):
		j.set_color(colors[i])

	ax.legend(SCHEMES_REW, loc=4)
	
	plt.ylabel('total reward')
	plt.xlabel('trace index')
	plt.show()
	# plt.savefig(os.path.join(PLOTS_FOLDER, 'rewards.png'))

	# ---- ---- ---- ----
	# CDF 
	# ---- ---- ---- ----
	print("show CDF graph: ")
	fig = plt.figure()
	ax = fig.add_subplot(111)

	for scheme in SCHEMES:
		values, base = np.histogram(reward_all[scheme], bins=NUM_BINS)
		cumulative = np.cumsum(values)
		ax.plot(base[:-1], cumulative)	

	colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	for i,j in enumerate(ax.lines):
		j.set_color(colors[i])	

	ax.legend(SCHEMES_REW, loc=2)
	
	plt.ylabel('CDF')
	plt.xlabel('total reward')
	plt.show()
	# plt.savefig(os.path.join(PLOTS_FOLDER, 'CDF.png'))

	# ---- ---- ---- ----
	# check each trace
	# ---- ---- ---- ----

	for l in time_all[SCHEMES[0]]:
		schemes_check = True
		for scheme in SCHEMES:
			if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
				schemes_check = False
				break
		if schemes_check:
			fig = plt.figure()

			ax = fig.add_subplot(311)
			for scheme in SCHEMES:
				ax.plot(time_all[scheme][l][:VIDEO_LEN], bit_rate_all[scheme][l][:VIDEO_LEN])
			colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
			for i,j in enumerate(ax.lines):
				j.set_color(colors[i])
			plt.title(l)
			plt.ylabel('bit rate selection (kbps)')

			ax = fig.add_subplot(312)
			for scheme in SCHEMES:
				ax.plot(time_all[scheme][l][:VIDEO_LEN], buff_all[scheme][l][:VIDEO_LEN])
			colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
			for i,j in enumerate(ax.lines):
				j.set_color(colors[i])
			plt.ylabel('buffer size (sec)')

			ax = fig.add_subplot(313)
			for scheme in SCHEMES:
				ax.plot(time_all[scheme][l][:VIDEO_LEN], bw_all[scheme][l][:VIDEO_LEN])
			colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
			for i,j in enumerate(ax.lines):
				j.set_color(colors[i])
			plt.ylabel('bandwidth (mbps)')
			plt.xlabel('time (sec)')

			SCHEMES_REW = []
			for scheme in SCHEMES:
				if scheme == SIM_DP:
					SCHEMES_REW.append(scheme + ': ' + str(raw_reward_all[scheme][l]))
				else:
					SCHEMES_REW.append(scheme + ': ' + str(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN])))

			ax.legend(SCHEMES_REW, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(np.ceil(len(SCHEMES) / 2.0)))
			plt.show()


if __name__ == '__main__':
	main()