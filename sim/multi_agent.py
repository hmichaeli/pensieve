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

import env
#import a3c   # change to dynamic import
global a3c
import load_trace



# S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
# S_LEN = 8  # take how many frames in the past
# A_DIM = 6
# ACTOR_LR_RATE = 0.0001
# CRITIC_LR_RATE = 0.001
# NUM_AGENTS = 16
# TRAIN_SEQ_LEN = 100  # take as a train batch
# MODEL_SAVE_INTERVAL = 100
# VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
# HD_REWARD = [1, 2, 3, 12, 15, 20]
# BUFFER_NORM_FACTOR = 10.0
# CHUNK_TIL_VIDEO_END_CAP = 48.0
# M_IN_K = 1000.0
# REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
# SMOOTH_PENALTY = 1
# DEFAULT_QUALITY = 1  # default video quality without agent
# RANDOM_SEED = 42
# RAND_RANGE = 1000
#
# dir_path = os.path.dirname(os.path.realpath(__file__))
# SUMMARY_DIR = './results'
# LOG_FILE = './results/log'
# TEST_LOG_FOLDER = './test_results/'
# TRAIN_TRACES = dir_path+'/train_sim_traces/'
# # NN_MODEL = './results/pretrain_linear_reward.ckpt'
# NN_MODEL = None
DEBUG = True
DEEP_DEBUG = False

def log(val):
    if DEBUG:
        print(val)

def log1(val):
    if DEEP_DEBUG:
        print(val)

def testing(epoch, nn_model, log_file, args):
    # tf.enable_eager_execution()

    # clean up the test results folder
    os.system('rm -r ' + args.TEST_LOG_FOLDER)
    os.system('mkdir ' + args.TEST_LOG_FOLDER)
    
    # run test script
    print("[testing] " + args.python_command + ' ' + args.pensieve_path + '/pensieve/sim/rl_test.py ' + '--TEST_NN_MODEL ' + nn_model +
              ' --pensieve_path \"' + args.pensieve_path + '\"' + " --TEST_TRACES " + args.VALIDATE_TRACES +
          ' --LOG_FILE ' + args.TEST_LOG_FOLDER + ' --A3C ' + args.A3C)
    os.system(args.python_command + ' ' + args.pensieve_path + '/pensieve/sim/rl_test.py ' + '--TEST_NN_MODEL ' + nn_model +
              ' --pensieve_path \"' + args.pensieve_path + '\"'+ " --TEST_TRACES " + args.VALIDATE_TRACES +
              ' --LOG_FILE ' + args.TEST_LOG_FOLDER+ ' --A3C ' + args.A3C)
    
    # append test performance to the log
    rewards = []
    test_log_files = os.listdir(args.TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward = []
        with open(args.TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.sum(reward[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()


def central_agent(net_params_queues, exp_queues, args):
    # tf.enable_eager_execution()

    assert len(net_params_queues) == args.NUM_AGENTS
    assert len(exp_queues) == args.NUM_AGENTS

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
        nn_model = args.RESTORE_NN_MODEL
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
            log1("[central agent] wait for agents")
            for i in range(args.NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])
                # Note: this is synchronous version of the parallel training,
                # which is easier to understand and probe. The framework can be
                # fairly easily modified to support asynchronous training.
                # Some practices of asynchronous training (lock-free SGD at
                # its core) are nicely explained in the following two papers:
                # https://arxiv.org/abs/1602.01783
                # https://arxiv.org/abs/1106.5730
            log1("[central agent] got data from all agents")
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

            for i in range(args.NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])

            # compute aggregated gradient
            assert args.NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
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

            if epoch % args.MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, args.SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                logging.info("Model saved in file: " + save_path)
                print("[central agent] test model epoch: " + str(epoch))
                testing(epoch, 
                    args.SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt",
                    test_log_file, args)
                print("[central agent] test model - done.")
                # TODO: add backup
                if ((epoch < 10000 and epoch % 1000 == 0) or (epoch % args.backup_freq == 0)) and args.backup_freq != -1:
                    print("[central_agent] saving backup of epoch {} in {}".format(str(epoch), args.backup_path))
                    print("[central_agent] " + 'cp -r ' + save_path + '*  \"' + args.backup_path + '\"')
                    os.system('cp -r ' + save_path + '*  \"' + args.backup_path + '\"')


def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue, args):
    # tf.enable_eager_execution()



    print("[agent{}]start".format(str(agent_id)))
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=(agent_id * args.RANDOM_SEED))

    with tf.Session() as sess, open(args.LOG_FILE + '_agent_' + str(agent_id), 'wb') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[args.S_INFO, args.S_LEN], action_dim=args.A_DIM,
                                 learning_rate=args.ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[args.S_INFO, args.S_LEN],
                                   learning_rate=args.CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        last_bit_rate = args.DEFAULT_QUALITY
        bit_rate = args.DEFAULT_QUALITY

        action_vec = np.zeros(args.A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((args.S_INFO, args.S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        time_stamp = 0
        loop_i = 0
        while True:  # experience video streaming forever
            log1("[agent{}] loop start".format(str(agent_id)))

            loop_i += 1
            # the action is from the last decision
            # this is to make the framework similar to the real
            log1("[agent{}] get video chunk".format(str(agent_id)))
            #TODO: check why stucks here sometimes
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # -- linear reward --
            # reward is video quality - rebuffer penalty - smoothness
            log1("[agent{}] get reward".format(str(agent_id)))
            reward = args.VIDEO_BIT_RATE[bit_rate] / args.M_IN_K \
                     - args.REBUF_PENALTY * rebuf \
                     - args.SMOOTH_PENALTY * np.abs(args.VIDEO_BIT_RATE[bit_rate] -
                                               args.VIDEO_BIT_RATE[last_bit_rate]) / args.M_IN_K

            # -- log scale reward --
            # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[-1]))
            # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[-1]))

            # reward = log_bit_rate \
            #          - REBUF_PENALTY * rebuf \
            #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

            # -- HD reward --
            # reward = HD_REWARD[bit_rate] \
            #          - REBUF_PENALTY * rebuf \
            #          - SMOOTH_PENALTY * np.abs(HD_REWARD[bit_rate] - HD_REWARD[last_bit_rate])

            r_batch.append(reward)

            last_bit_rate = bit_rate

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((args.S_INFO, args.S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            log1("[agent{}] create state".format(str(agent_id)))

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = args.VIDEO_BIT_RATE[bit_rate] / float(np.max(args.VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / args.BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / args.M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / args.M_IN_K / args.BUFFER_NORM_FACTOR  # 10 sec
            state[4, :args.A_DIM] = np.array(next_video_chunk_sizes) / args.M_IN_K / args.M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, args.CHUNK_TIL_VIDEO_END_CAP) / float(args.CHUNK_TIL_VIDEO_END_CAP)

            # compute action probability vector

            log1("[agent{}] predict".format(str(agent_id)))

            action_prob = actor.predict(np.reshape(state, (1, args.S_INFO, args.S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, args.RAND_RANGE) / float(args.RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # log time_stamp, bit_rate, buffer_size, reward
            log1("[agent{}] write to file".format(str(agent_id)))
            log_file.write(str(time_stamp) + '\t' +
                           str(args.VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # report experience to the coordinator
            if len(r_batch) >= args.TRAIN_SEQ_LEN or end_of_video:
                log1("[agent{}] write results in queue".format(str(agent_id)))
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                log_file.write('\n')  # so that in the log we know where video ends

            # store the state and action into batches
            if end_of_video:
                log1("[agent{}] end of video".format(str(agent_id)))

                last_bit_rate = args.DEFAULT_QUALITY
                bit_rate = args.DEFAULT_QUALITY  # use the default action here

                action_vec = np.zeros(args.A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((args.S_INFO, args.S_LEN)))
                a_batch.append(action_vec)

            else:
                s_batch.append(state)

                action_vec = np.zeros(args.A_DIM)
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)

            # if args.INSPECT_DATA_USAGE and loop_i % 1000000 == 0:
            #     hist = net_env.trace_histogram
            #     print("[agent {}] histogram max: {} histogram avg: {} ".format(agent_id, hist.max(), (hist.sum() / len(hist))))
            #     print("[agent {}] save traces histogram as ".format(agent_id) + args.LOG_FILE + '_agent_' + str(agent_id) + 'data_usage.csv')
            #     np.savetxt(args.LOG_FILE + '_agent_' + str(agent_id) + 'data_usage.csv', hist, delimiter=',')

                # # save numpy array as csv file
                # from numpy import asarray
                # from numpy import savetxt
                # # define data
                # data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
                # # save to csv file
                # savetxt('data.csv', data, delimiter=',')


def main(args):

    print("main")
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
    args.SUMMARY_DIR = os.path.join(args.SUMMARY_DIR,  timestr + '/results')
    args.LOG_FILE = os.path.join(args.LOG_FILE,  timestr + '/results/log')
    args.TEST_LOG_FOLDER = os.path.join(args.TEST_LOG_FOLDER,  timestr + '/test_results/')
    print("[main] running with arguments:")
    argparse_dict = vars(args)
    print(str(argparse_dict))
    print("[main] save args in file")
    #TODO: save file in backup dir with args
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
    print("[main] loading traces")
    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(args.TRAIN_TRACES)
    print("[main] creating {} workers".format(args.NUM_AGENTS))
    agents = []
    for i in range(args.NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw,
                                       net_params_queues[i],
                                       exp_queues[i], args)))
    for i in range(args.NUM_AGENTS):
        print("[main] start agent "+str(i))
        agents[i].start()

    # wait unit training is done
    coordinator.join()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_command", type=str, default='python')
    parser.add_argument("--backup_freq", type=int, default=10000,  help="epoch freq to backup nn_model")
    parser.add_argument("--backup_path", type=str, default='/home/haggai/projects/networkML/data/models/', help="path to save backups")
    parser.add_argument("--pensieve_path", type=str, default='/home/haggai/projects/networkML/', help="path to pensieve dir including (networkML)" )
    parser.add_argument("--S_INFO", default=6)  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
    parser.add_argument("--S_LEN", default=8)  # take how many frames in the past
    parser.add_argument("--A_DIM", default=6)
    parser.add_argument("--ACTOR_LR_RATE", default=0.0001)
    parser.add_argument("--CRITIC_LR_RATE", default=0.001)
    parser.add_argument("--NUM_AGENTS", type=int, default=16, help="multiprocessing number of agents")
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

    parser.add_argument("--TRAIN_TRACES", default='/home/haggai/projects/networkML/data/paper_samples/train/', help="relative to networkML/pensieve/sim")
    parser.add_argument("--VALIDATE_TRACES", default='/home/haggai/projects/networkML/data/paper_samples/validation/', help="relative to networkML/pensieve/sim")

    # NN_MODEL = './results/pretrain_linear_reward.ckpt'
    parser.add_argument("--RESTORE_NN_MODEL", default=None)
    parser.add_argument("--RESTORE_EPOCH", type=int, default=0, help="restore model epoch")
    parser.add_argument("--INSPECT_DATA_USAGE", type=bool, default=True, help="restore model epoch")
    parser.add_argument("--A3C", type=str, default='a3c', help="a3c version / file name")


    args = parser.parse_args()
    args.TRAIN_TRACES #= args.pensieve_path + '/pensieve/sim/' + args.TRAIN_TRACES
    args.VALIDATE_TRACES #= args.pensieve_path + '/pensieve/sim/' + args.VALIDATE_TRACES
    return args


# def init_gpu_test():
#     import timeit
#    # tf.enable_eager_execution()
#     device_name = tf.test.gpu_device_name()
#     print("device_name = " + str(device_name))
#     # if device_name != '/device:GPU:0':
#     #     print(
#     #         '\n\nThis error most likely means that this notebook is not '
#     #         'configured to use a GPU.  Change this in Notebook Settings via the '
#     #         'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
#     #     raise SystemError('GPU device not found')
#
#     def cpu():
#         with tf.device('/cpu:0'):
#             random_image_cpu = tf.random.normal((100, 100, 100, 3))
#             net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
#             return tf.math.reduce_sum(net_cpu)
#
#     def gpu():
#         with tf.device('/device:GPU:0'):
#             random_image_gpu = tf.random.normal((100, 100, 100, 3))
#             net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
#             return tf.math.reduce_sum(net_gpu)
#
#     # We run each op once to warm up; see: https://stackoverflow.com/a/45067900
#     cpu()
#     gpu()
#
#     # Run the op several times.
#     print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
#           '(batch x height x width x channel). Sum of ten runs.')
#     print('CPU (s):')
#     cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
#     print(cpu_time)
#     print('GPU (s):')
#     gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
#     print(gpu_time)
#     print('GPU speedup over CPU: {}x'.format(int(cpu_time / gpu_time)))



if __name__ == '__main__':
    args = get_args()
    # init_gpu_test()
    main(args)
