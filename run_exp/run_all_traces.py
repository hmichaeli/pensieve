import os
import time
import json
import urllib
import subprocess


def log(val):
	print("[run_all_traces] " + str(val))


# TRACE_PATH = '../cooked_traces/'

TRACE_PATH = '/home/haggai/projects/networkML/data/mahimi_cooked_traces/fcc/test/'
print("TRACE_PATH: " + TRACE_PATH)
with open('./chrome_retry_log', 'wb') as f:
	f.write('chrome retry log\n')


os.system('sudo sysctl -w net.ipv4.ip_forward=1')

ip_data = json.loads(urllib.urlopen("http://ip.jsontest.com/").read())
ip = str(ip_data['ip'])

# ip = 'localhost'

ABR_ALGO = 'BB'
PROCESS_ID = 0
command_BB = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

ABR_ALGO = 'RB'
PROCESS_ID = 1
command_RB = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

ABR_ALGO = 'FIXED'
PROCESS_ID = 2
command_FIXED = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

ABR_ALGO = 'FESTIVE'
PROCESS_ID = 3
command_FESTIVE = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

ABR_ALGO = 'BOLA'
PROCESS_ID = 4
command_BOLA = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

ABR_ALGO = 'fastMPC'
PROCESS_ID = 5
command_fastMPC = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

ABR_ALGO = 'robustMPC'
PROCESS_ID = 6
command_robustMPC = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

ABR_ALGO = 'RL'
PROCESS_ID = 7
command_RL = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

log("running commands")
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

log(command_RL)
# proc_RL = subprocess.Popen(command_RL, stdout=subprocess.PIPE, shell=True)
proc_RL = subprocess.Popen(command_RL,  shell=True)
time.sleep(0.1)

log("waiting for process to finish")
# proc_BB.wait()
# proc_RB.wait()
# proc_FIXED.wait()
# proc_FESTIVE.wait()
# proc_BOLA.wait()
# proc_fastMPC.wait()
# proc_robustMPC.wait()
proc_RL.wait()
log("done")
