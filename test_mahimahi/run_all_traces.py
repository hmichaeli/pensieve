import os
import time
import json
import urllib
import subprocess


def log(val):
	print("[run_all_traces] " + str(val))


# TRACE_PATH = '../cooked_traces/'

# mahimahi traces
TRACE_PATH = '/home/haggai/projects/networkML/data/mahimahi_all_traces/test/'

# list of RL models to test (NN checkpoints)
MODELS_DIR = '/home/haggai/projects/networkML/pensieve/test_mahimahi/nn_models/'

server_offset = 8 # will use RL-# , # starting offset

MODELS = []
files = os.listdir(MODELS_DIR)
for f in files:
	suffix = '.ckpt.meta'
	if suffix in f:
		prefix = f.split(suffix)[0]
		MODELS.append(prefix)

# check all checkpoint files exist:
for m in MODELS:
	count = 0
	for f in files:
		if m in f:
			count += 1
	if count != 3:
		MODELS.remove(m)

# DEBUG
#MODELS = [MODELS[0],MODELS[1]]

if len(MODELS) > 8:
	print("UP to 8 models!!")
	exit(1)



print("TRACE_PATH: " + TRACE_PATH)
with open('./chrome_retry_log', 'wb') as f:
	f.write('chrome retry log\n')
print("models: {}".format(MODELS))

os.system('sudo sysctl -w net.ipv4.ip_forward=1')
ip_data = json.loads(urllib.urlopen("http://ip.jsontest.com/").read())
ip = str(ip_data['ip'])

# ip = 'localhost'

command_list = []
proc_list = []
MODELS_DICT = {}
for i, nn_model in enumerate(MODELS):
	#TODO: add json that maps RL-# to ckpt
	serv_id = server_offset + i
	ABR_ALGO = 'RL-' + str(serv_id)
	MODELS_DICT[ABR_ALGO] = MODELS[i]
	PROCESS_ID = i
	nn_model_path = os.path.join(MODELS_DIR, nn_model + '.ckpt')
	command_RL = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip + ' ' + nn_model_path
	log("running commands")
	log(command_RL)
	command_list.append(command_RL)
	# proc_RL = subprocess.Popen(command_RL, stdout=subprocess.PIPE, shell=True)
	proc_RL = subprocess.Popen(command_RL, shell=True)
	proc_list.append(proc_RL)
	time.sleep(0.1)

import json
with open('./results/models_dict.json', 'w') as fp:
    json.dump(MODELS_DICT, fp)
#TODO: check if proceesses can run in parallel
log("waiting for process to finish")
for i in range(len(MODELS)):
	proc_list[i].wait()
	log("done")

#
# ABR_ALGO = 'BB'
# PROCESS_ID = 0
# command_BB = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip
#
# ABR_ALGO = 'RB'
# PROCESS_ID = 1
# command_RB = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip
#
# ABR_ALGO = 'FIXED'
# PROCESS_ID = 2
# command_FIXED = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip
#
# ABR_ALGO = 'FESTIVE'
# PROCESS_ID = 3
# command_FESTIVE = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip
#
# ABR_ALGO = 'BOLA'
# PROCESS_ID = 4
# command_BOLA = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip
#
# ABR_ALGO = 'fastMPC'
# PROCESS_ID = 5
# command_fastMPC = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip
#
# ABR_ALGO = 'robustMPC'
# PROCESS_ID = 6
# command_robustMPC = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip
#
# ABR_ALGO = 'RL'
# PROCESS_ID = 7
# command_RL = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

# log(command_BB)
# proc_BB = subprocess.Popen(command_BB, stdout=subprocess.PIPE, shell=True)
# time.sleep(0.1)
# log(command_RB)
# proc_RB = subprocess.Popen(command_RB, stdout=subprocess.PIPE, shell=True)
# time.sleep(0.1)
# log(command_FIXED)
# proc_FIXED = subprocess.Popen(command_FIXED, stdout=subprocess.PIPE, shell=True)
# time.sleep(0.1)
#
# log(command_FESTIVE)
# proc_FESTIVE = subprocess.Popen(command_FESTIVE, stdout=subprocess.PIPE, shell=True)
# time.sleep(0.1)
# log(command_BOLA)
# proc_BOLA = subprocess.Popen(command_BOLA, stdout=subprocess.PIPE, shell=True)
# time.sleep(0.1)
#
# log(command_fastMPC)
# proc_fastMPC = subprocess.Popen(command_fastMPC, stdout=subprocess.PIPE, shell=True)
# time.sleep(0.1)
# log(command_robustMPC)
# proc_robustMPC = subprocess.Popen(command_robustMPC, stdout=subprocess.PIPE, shell=True)
# time.sleep(0.1)

# log(command_RL)
# # proc_RL = subprocess.Popen(command_RL, stdout=subprocess.PIPE, shell=True)
# proc_RL = subprocess.Popen(command_RL,  shell=True)
# time.sleep(0.1)
#
# log("waiting for process to finish")
# # proc_BB.wait()
# # proc_RB.wait()
# # proc_FIXED.wait()
# # proc_FESTIVE.wait()
# # proc_BOLA.wait()
# # proc_fastMPC.wait()
# # proc_robustMPC.wait()
# proc_RL.wait()
# log("done")
