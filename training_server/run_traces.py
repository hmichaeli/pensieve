import os
import logging
import numpy as np
import argparse
import multiprocessing as mp
# os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
# tf.enable_eager_execution()
import time
import json
import importlib
import Queue
import time
import random
# import env
#import a3c   # change to dynamic import
global a3c
# import load_trace

from datetime import datetime
# import sys
import subprocess
import urllib
import pickle

RUN_SCRIPT = 'run_video.py'
RANDOM_SEED = 42
RUN_TIME = 350 # 320 # 320  # sec
MM_DELAY = 40   # millisec

#DEBUG
# RUN_TIME = 50 # 320 # 320  # sec


DEBUG = True


def log(val):
	if DEBUG:
		print("[run_traces] " + str(val))


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]



TRACE_PATH = '/home/haggai/projects/networkML/data/mahimahi_all_traces/train/'
NN_MODEL = './init_model/pretrain_linear_reward.ckpt'

global latest_ckpt


def central_agent(net_params_queues, exp_queues, args):
	# tf.enable_eager_execution()
	# global latest_ckpt
	# assert len(net_params_queues) == args.NUM_AGENTS
	# assert len(exp_queues) == args.NUM_AGENTS

	logging.basicConfig(filename=args.LOG_FILE + '_central',
						filemode='w',
						level=logging.INFO)

	with tf.Session() as sess, open(args.LOG_FILE + '_test', 'wb') as test_log_file:

		actor = a3c.ActorNetwork(sess,
								 state_dim=[args.S_INFO, args.S_LEN], action_dim=args.A_DIM,
								 learning_rate=args.ACTOR_LR_RATE)
		critic = a3c.CriticNetwork(sess,
								   state_dim=[args.S_INFO, args.S_LEN],
								   learning_rate=args.CRITIC_LR_RATE)

		summary_ops, summary_vars = a3c.build_summaries()

		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter(args.SUMMARY_DIR, sess.graph)  # training monitor
		saver = tf.train.Saver()  # save neural net parameters

		# restore neural net parameters
		# nn_model = args.RESTORE_NN_MODEL


		nn_model = NN_MODEL

		if nn_model is not None:  # nn_model is the path to file
			print("[central-agent] restoring model " + str(nn_model))
			saver.restore(sess, nn_model)
			epoch = args.RESTORE_EPOCH
			print("Model restored.")
		else:
			print("[central-agent] RESTORE_NN_MODEL = None, starting from scratch")
			epoch = 0

		print("[central-agent]epoch = " + str(epoch))
		# assemble experiences from agents, compute the gradients
		while True:
			# synchronize the network parameters of work agent
			actor_net_params = actor.get_network_params()
			critic_net_params = critic.get_network_params()

			# log1("[central agent] wait for agents")
			# for i in range(args.NUM_AGENTS):
			#     net_params_queues[i].put([actor_net_params, critic_net_params])
				# Note: this is synchronous version of the parallel training,
				# which is easier to understand and probe. The framework can be
				# fairly easily modified to support asynchronous training.
				# Some practices of asynchronous training (lock-free SGD at
				# its core) are nicely explained in the following two papers:
				# https://arxiv.org/abs/1602.01783
				# https://arxiv.org/abs/1106.5730
			# log1("[central agent] got data from all agents")
			# record average reward and td loss change
			# in the experiences from the agents
			total_batch_len = 0.0
			total_reward = 0.0
			total_td_loss = 0.0
			total_entropy = 0.0
			total_agents = 0.0

			# assemble experiences from the agents
			actor_gradient_batch = []
			critic_gradient_batch = []

			log("[central agent] get from queue")
			# s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()
			all_train_data = exp_queues[0].get()
			log("[central agent] get from queue - done")
			log("extracting train data:")
			for train_data in all_train_data: # need to get 1 tuple
				# log(type(all_train_data))
				# log(type(train_data))
				# log(train_data)
				s_batch, a_batch, r_batch, terminal, info = train_data
			# for i in range(1):#args.NUM_AGENTS):
				# log("[central agent] get from queue")
				# # s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()
				# all_train_data = exp_queues[i].get()
				# log("[central agent] get from queue - done")
				# log("train_data shapes")
				#
				# print(s_batch)
				# print(a_batch)
				# print(r_batch)
				# log("s: {}".format(s_batch.shape))
				# log("a: {}".format(a_batch.shape))
				# log("r: {}".format(r_batch.shape))

				actor_gradient, critic_gradient, td_batch = \
					a3c.compute_gradients(
						s_batch = np.stack(s_batch, axis=0),
						a_batch = np.vstack(a_batch),
						r_batch = np.vstack(r_batch),
						terminal=terminal, actor=actor, critic=critic)

				actor_gradient_batch.append(actor_gradient)
				critic_gradient_batch.append(critic_gradient)

				total_reward += np.sum(r_batch)
				total_td_loss += np.sum(td_batch)
				total_batch_len += len(r_batch)
				total_agents += 1.0
				total_entropy += np.sum(info['entropy'])


			if total_agents == 0:
				# this should not happen
				log("WARNING - this should not happen")
				log("got empty data - skipping updating net")
				continue

			# compute aggregated gradient

			# Hagay - try removed - because some agents can fail
			# assert args.NUM_AGENTS == len(actor_gradient_batch)
			# assert len(actor_gradient_batch) == len(critic_gradient_batch)


			# assembled_actor_gradient = actor_gradient_batch[0]
			# assembled_critic_gradient = critic_gradient_batch[0]
			# for i in range(len(actor_gradient_batch) - 1):
			#     for j in range(len(assembled_actor_gradient)):
			#             assembled_actor_gradient[j] += actor_gradient_batch[i][j]
			#             assembled_critic_gradient[j] += critic_gradient_batch[i][j]
			# actor.apply_gradients(assembled_actor_gradient)
			# critic.apply_gradients(assembled_critic_gradient)
			for i in range(len(actor_gradient_batch)):
				actor.apply_gradients(actor_gradient_batch[i])
				critic.apply_gradients(critic_gradient_batch[i])

			# log training information
			epoch += 1
			avg_reward = total_reward  / total_agents
			avg_td_loss = total_td_loss / total_batch_len
			avg_entropy = total_entropy / total_batch_len

			logging.info('Epoch: ' + str(epoch) +
						 ' TD_loss: ' + str(avg_td_loss) +
						 ' Avg_reward: ' + str(avg_reward) +
						 ' Avg_entropy: ' + str(avg_entropy))

			summary_str = sess.run(summary_ops, feed_dict={
				summary_vars[0]: avg_td_loss,
				summary_vars[1]: avg_reward,
				summary_vars[2]: avg_entropy
			})

			writer.add_summary(summary_str, epoch)
			writer.flush()
			# debug
			args.MODEL_SAVE_INTERVAL = 20
			###
			# if epoch % args.MODEL_SAVE_INTERVAL == 0:
			if True: # save checkpoint every iteration
				# Save the neural net parameters to disk.
				save_path = saver.save(sess, args.SUMMARY_DIR + "/nn_model_ep_" +
									   str(epoch) + ".ckpt")
				latest_ckpt = save_path
				log("[central_agent] latest ckpt: " + latest_ckpt)
				logging.info("Model saved in file: " + save_path)
				log("[central_agent] put latest ckpt in queue")
				net_params_queues[0].put(latest_ckpt)
				log("[central_agent] put latest ckpt in queue - done")

				# print("[central agent] test model epoch: " + str(epoch))
				# testing(epoch,
				#     args.SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt",
				#     test_log_file, args)
				# print("[central agent] test model - done.")
				# TODO: add backup
				if epoch % 50 == 0: #True: #((epoch < 10000 and epoch % 1000 == 0) or (epoch % args.backup_freq == 0)) and args.backup_freq != -1:
					print("[central_agent] saving backup of epoch {} in {}".format(str(epoch), args.backup_path))
					print("[central_agent] " + 'cp -r ' + save_path + '*  \"' + args.backup_path + '\"')
					os.system('cp -r ' + save_path + '*  \"' + args.backup_path + '\"')



def start_central_agent(args):
	global a3c
	print("[main] dynamic import a3c: {}".format(args.A3C))
	a3c = importlib.import_module(args.A3C)

	# args_json = json.dumps(args)
	timestr = time.strftime("%Y-%m-%d--%H-%M-%S")
	args.backup_path = os.path.join(args.backup_path, timestr + '/')
	log("[main] create saved models dir: " + args.backup_path)
	os.makedirs(args.backup_path)

	# parser.add_argument("--SUMMARY_DIR", default='/home/haggai/projects/networkML/data/results')
	# parser.add_argument("--LOG_FILE", default='/home/haggai/projects/networkML/data/results/log')
	# parser.add_argument("--TEST_LOG_FOLDER", default='/home/haggai/projects/networkML/data/test_results/')
	args.SUMMARY_DIR = os.path.join(args.SUMMARY_DIR, timestr + '/results')
	args.LOG_FILE = os.path.join(args.LOG_FILE, timestr + '/results/log')
	args.TEST_LOG_FOLDER = os.path.join(args.TEST_LOG_FOLDER, timestr + '/test_results/')
	print("[main] running with arguments:")
	argparse_dict = vars(args)
	print(str(argparse_dict))
	print("[main] save args in file")
	# TODO: save file in backup dir with args
	with open(args.backup_path + 'run_args.json', 'w') as file:
		json.dump(argparse_dict, file)
	np.random.seed(int(args.RANDOM_SEED))
	assert len(args.VIDEO_BIT_RATE) == args.A_DIM

	# create result directory
	if not os.path.exists(args.SUMMARY_DIR):
		os.makedirs(args.SUMMARY_DIR)

	# inter-process communication queues
	net_params_queues = []
	exp_queues = []
	print("[main] create queues")
	for i in range(args.NUM_AGENTS):
		net_params_queues.append(mp.Queue(1))
		exp_queues.append(mp.Queue(1))

	# create a coordinator and multiple agent processes
	# (note: threading is not desirable due to python GIL)
	coordinator = mp.Process(target=central_agent,
							 args=(net_params_queues, exp_queues, args))
	print("[main] start central_agent")
	coordinator.start()

	return exp_queues, net_params_queues



def main(args):
	log("main")
	log("TRACE_PATH: " + TRACE_PATH)
	with open('./chrome_retry_log', 'wb') as f:
		f.write('chrome retry log\n')

	os.system('sudo sysctl -w net.ipv4.ip_forward=1')

	ip_data = json.loads(urllib.urlopen("http://ip.jsontest.com/").read())
	IP = str(ip_data['ip'])
	PROC_ID = 1

	trace_path = TRACE_PATH
	# abr_algo = 'RL'
	# process_id = PROC_ID
	ip = IP

	sleep_vec = range(1, 10)  # random sleep second

	files = os.listdir(trace_path)
	# log("files list:\n{}".format(files))

	log("start central agent")
	exp_queues, net_params_queues = start_central_agent(args)


	EPOCHS = 100
	global latest_ckpt
	latest_ckpt = NN_MODEL
	# BATCH_SIZE = 16 # number of RL servers in /html/www/...
	# debug
	BATCH_SIZE = 15
	skip_ckpt_update = True
	for ep in range(EPOCHS):
		random.shuffle(files)
		# try:
		# 	ckpt = net_params_queues[0].get(block=False)
		# 	latest_ckpt = ckpt
		# except Queue.Empty:
		# 	# use older ckpt
		# 	pass

		for b, x in enumerate(batch(files, BATCH_SIZE)):
			log("Batch: {} Iteration: {}".format(ep, b))
			now = datetime.now()  # current date and time
			date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
			log("Time: " + date_time)

			if len(x) < BATCH_SIZE:
				log("skip last batch - npt full")
				break
			if not skip_ckpt_update:
				t1 = time.time()
				log("wait to params")
				ckpt = net_params_queues[0].get(block=True)
				latest_ckpt = ckpt
				elapsed = time.time() - t1
				log("waiting time to ckpt: {}".format(elapsed))
			else:
				log("skipped ckpt update")
				skip_ckpt_update = False

			t2 = time.time()
			print (x)
			command_list = []
			proc_list = []
			#debug
			for j, f in enumerate(x):

				sleep_time = j
				process_id = j
				abr_algo = 'RL-' + str(j)
				command = 'mm-delay ' + str(MM_DELAY) + \
						  ' mm-link 12mbps ' + trace_path + f + ' ' + \
						  '/usr/bin/python ' + RUN_SCRIPT + ' ' + ip + ' ' + \
						  abr_algo + ' ' + str(RUN_TIME) + ' ' + \
						  str(process_id) + ' ' + f + ' ' + str(sleep_time) + \
						  ' ' + latest_ckpt
				command_list.append(command)
				log(command)
				proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

				# TODO: need pipes for loging out, err
				# proc = subprocess.Popen(command,
				# 		  stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

				# proc = subprocess.Popen(command, shell=True)

				proc_list.append(proc)

			succeed_proc = []
			#debug
			for j, f in enumerate(x):

				(out, err) = proc_list[j].communicate()

				# (out, err) = proc_list[j].communicate()

				log("----proc {} output-----".format(j))
				log(out)
				log("---end of output---")
				# print("[run_traces] " + str(out))
				if out == 'done\n':
					log("proc {} recived done in out".format(j))
					succeed_proc.append(j)
					# break
				else:
					log("write to chrome_retry_log")
					with open('./chrome_retry_log', 'ab') as log1:
						log1.write(abr_algo + '_' + f + '\n')
						log1.write(out + '\n')
						# hagay
						log1.write(err + '\n')
						log1.flush()

				# while True:
				#
				# 	# np.random.shuffle(sleep_vec)
				# 	# sleep_time = sleep_vec[int(process_id)]
				#
				#
				#
				#
				#
				#
				# 	# print("[run_traces] " + str(command))
				# 	log(command)
				#
				# 	# TODO: need pipes for loging out, err
				# 	proc = subprocess.Popen(command,
				# 			  stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
				#
				# 	# proc = subprocess.Popen(command, shell=True)
				#
				#
				# 	(out, err) = proc.communicate()
				# 	log("----output-----")
				# 	log(out)
				# 	log("---end of output---")
				# 	# print("[run_traces] " + str(out))
				# 	if out == 'done\n':
				# 		log("recived done in out")
				# 		break
				# 	else:
				# 		log("write to chrome_retry_log")
				# 		with open('./chrome_retry_log', 'ab') as log1:
				# 			log1.write(abr_algo + '_' + f + '\n')
				# 			log1.write(out + '\n')
				# 			# hagay
				# 			log1.write(err + '\n')
				# 			log1.flush()


				# parse data files
			log("succeeded proc: {}".format(succeed_proc))
			elapsed = time.time() - t2
			log("waiting time for all workers: {}".format(elapsed))
			all_data = []
			log("open all succesful workers train data")

			for j in range(BATCH_SIZE):
				if j in succeed_proc:
					filename = './train_data_' + str(j)
					log("load {}".format(filename))
					infile = open(filename, 'rb')
					train_data = pickle.load(infile)
					infile.close()
					# print(train_data)
					# TODO: check if need to append here
					s_batch, a_batch, r_batch, terminal, info = train_data
					if len(s_batch) != 0:
						log("add train_date_{} to all_data".format(j))
						# all_data = all_data + train_data
						all_data.append(train_data)
					else:
						log("WARNING: ./train_data_{} is invalid".format(j))
				else:
					log("{} is unsuccessful proc".format(j))

				log("all data len: {}".format(len(all_data)))

			log("done parsing train data")

			if len(all_data) == 0:
				log("WARNING: ALL DATA is empy")
				log("skipping ckpt update")
				skip_ckpt_update = True
				continue

			log("put all_train_date in queue")
			exp_queues[0].put(all_data)
			log("put all_train_data in queue - done.")
			# for j in range(BATCH_SIZE):
			#
			# 	filename = './train_data_' + str(j)
			# 	infile = open(filename, 'rb')
			# 	train_data = pickle.load(infile)
			# 	infile.close()
			# 	# print(train_data)
			# 	# TODO: check if need to append here
			# 	s_batch, a_batch, r_batch, terminal, info = train_data
			# 	if len(s_batch) != 0:
			# 		log("put train_date_{} in queue".format(j))
			# 		exp_queues[0].put(train_data)
			# 		log("put train_date_{} in queue - done.".format(j))
			# 	else:
			# 		log("WARNING: ./train_data_{} is invalid".format(j))
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--python_command", type=str, default='python')
	parser.add_argument("--backup_freq", type=int, default=10000, help="epoch freq to backup nn_model")
	parser.add_argument("--backup_path", type=str, default='/home/haggai/projects/networkML/data/server_trained_models/',
						help="path to save backups")
	parser.add_argument("--pensieve_path", type=str, default='/home/haggai/projects/networkML/',
						help="path to pensieve dir including (networkML)")
	parser.add_argument("--S_INFO",
						default=6)  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
	parser.add_argument("--S_LEN", default=8)  # take how many frames in the past
	parser.add_argument("--A_DIM", default=6)
	parser.add_argument("--ACTOR_LR_RATE", default=0.0001)
	parser.add_argument("--CRITIC_LR_RATE", default=0.001)
	parser.add_argument("--NUM_AGENTS", type=int, default=1, help="multiprocessing number of agents") # must be 1 at the moment
	parser.add_argument("--TRAIN_SEQ_LEN", default=100)  # take as a train batch
	parser.add_argument("--MODEL_SAVE_INTERVAL", default=100)
	parser.add_argument("--VIDEO_BIT_RATE", default=[300, 750, 1200, 1850, 2850, 4300])  # Kbps
	parser.add_argument("--HD_REWARD", default=[1, 2, 3, 12, 15, 20])
	parser.add_argument("--BUFFER_NORM_FACTOR", default=10.0)
	parser.add_argument("--CHUNK_TIL_VIDEO_END_CAP", default=48.0)
	parser.add_argument("--M_IN_K", default=1000.0)
	parser.add_argument("--REBUF_PENALTY", default=4.3)  # 1 sec rebuffering -> 3 Mbps
	parser.add_argument("--SMOOTH_PENALTY", default=1)
	parser.add_argument("--DEFAULT_QUALITY", default=1)  # default video quality without agent
	parser.add_argument("--RANDOM_SEED", default=42, type=int)
	parser.add_argument("--RAND_RANGE", default=1000)

	# parser.add_argument("--SUMMARY_DIR", default='/home/haggai/projects/networkML/data/results')
	# parser.add_argument("--LOG_FILE", default='/home/haggai/projects/networkML/data/results/log')
	# parser.add_argument("--TEST_LOG_FOLDER", default='/home/haggai/projects/networkML/data/test_results/')
	#
	parser.add_argument("--SUMMARY_DIR", default='/home/haggai/projects/networkML/data/')
	parser.add_argument("--LOG_FILE", default='/home/haggai/projects/networkML/data/')
	parser.add_argument("--TEST_LOG_FOLDER", default='/home/haggai/projects/networkML/data/')

	# parser.add_argument("--TRAIN_TRACES", default='/home/haggai/projects/networkML/data/paper_samples/train/',
	# 					help="relative to networkML/pensieve/sim")
	# parser.add_argument("--VALIDATE_TRACES", default='/home/haggai/projects/networkML/data/paper_samples/validation/',
	# 					help="relative to networkML/pensieve/sim")

	# NN_MODEL = './results/pretrain_linear_reward.ckpt'
	parser.add_argument("--RESTORE_NN_MODEL", default=None)
	parser.add_argument("--RESTORE_EPOCH", type=int, default=0, help="restore model epoch")
	parser.add_argument("--INSPECT_DATA_USAGE", type=bool, default=True, help="restore model epoch")
	parser.add_argument("--A3C", type=str, default='a3c', help="a3c version / file name")

	args = parser.parse_args()
	# args.TRAIN_TRACES   = args.pensieve_path + '/pensieve/sim/' + args.TRAIN_TRACES
	# args.VALIDATE_TRACES   = args.pensieve_path + '/pensieve/sim/' + args.VALIDATE_TRACES
	return args


if __name__ == '__main__':
	args = get_args()
	main(args)
