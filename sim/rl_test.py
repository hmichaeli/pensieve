import os
import sys
import argparse
import importlib
#os.environ['CUDA_VISIBLE_DEVICES']=''

import numpy as np
import tensorflow as tf
import load_trace
#import a3c  # dynamic import
import fixed_env as env
#tf.enable_eager_execution()

# S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
# S_LEN = 8  # take how many frames in the past
# A_DIM = 6
# ACTOR_LR_RATE = 0.0001
# CRITIC_LR_RATE = 0.001
# VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
# BUFFER_NORM_FACTOR = 10.0
# CHUNK_TIL_VIDEO_END_CAP = 48.0
# M_IN_K = 1000.0
# REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
# SMOOTH_PENALTY = 1
# DEFAULT_QUALITY = 1  # default video quality without agent
# RANDOM_SEED = 42
# RAND_RANGE = 1000
# dir_path = os.path.dirname(os.path.realpath(__file__))
# LOG_FILE = './test_results/log_sim_rl'
# TEST_TRACES = dir_path+'/test_sim_traces/'
# # log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = sys.argv[1]


def main(args):
    print("[main] dynamic import a3c: {}".format(args.A3C))
    a3c = importlib.import_module(args.A3C)

    np.random.seed(args.RANDOM_SEED)

    assert len(args.VIDEO_BIT_RATE) == args.A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(args.TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = args.LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'wb')

    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[args.S_INFO, args.S_LEN], action_dim=args.A_DIM,
                                 learning_rate=args.ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[args.S_INFO, args.S_LEN],
                                   learning_rate=args.CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if args.TEST_NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, args.TEST_NN_MODEL)
            print("Testing model restored.")

        time_stamp = 0

        last_bit_rate = args.DEFAULT_QUALITY
        bit_rate = args.DEFAULT_QUALITY

        action_vec = np.zeros(args.A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((args.S_INFO, args.S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smoothness
            reward = args.VIDEO_BIT_RATE[bit_rate] / args.M_IN_K \
                     - args.REBUF_PENALTY * rebuf \
                     - args.SMOOTH_PENALTY * np.abs(args.VIDEO_BIT_RATE[bit_rate] -
                                               args.VIDEO_BIT_RATE[last_bit_rate]) / args.M_IN_K

            r_batch.append(reward)

            last_bit_rate = bit_rate

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp / args.M_IN_K) + '\t' +
                           str(args.VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((args.S_INFO, args.S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = args.VIDEO_BIT_RATE[bit_rate] / float(np.max(args.VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / args.BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / args.M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / args.M_IN_K / args.BUFFER_NORM_FACTOR  # 10 sec
            state[4, :args.A_DIM] = np.array(next_video_chunk_sizes) / args.M_IN_K / args.M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, args.CHUNK_TIL_VIDEO_END_CAP) / float(args.CHUNK_TIL_VIDEO_END_CAP)

            action_prob = actor.predict(np.reshape(state, (1, args.S_INFO, args.S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, args.RAND_RANGE) / float(args.RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            s_batch.append(state)

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            if end_of_video:
                log_file.write('\n')
                log_file.close()

                last_bit_rate = args.DEFAULT_QUALITY
                bit_rate = args.DEFAULT_QUALITY  # use the default action here

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(args.A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((args.S_INFO, args.S_LEN)))
                a_batch.append(action_vec)
                entropy_record = []

                video_count += 1

                if video_count >= len(all_file_names):
                    break

                log_path = args.LOG_FILE + '_' + all_file_names[net_env.trace_idx]
                log_file = open(log_path, 'wb')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_command", type=str, default='/mnt/c/projects/networkML/venv/bin/python')
    parser.add_argument("--backup_freq", type=int, default=-1, help="epoch freq to backup nn_model")
    parser.add_argument("--backup_path", type=str, default='none', help="path to save backups")
    parser.add_argument("--pensieve_path", type=str, default='/mnt/c/projects/networkML/',
                        help="path to pensieve dir including (networkML)")
    parser.add_argument("--S_INFO",
                        default=6)  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
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
    parser.add_argument("--RANDOM_SEED", default=42)
    parser.add_argument("--RAND_RANGE", default=1000)
    parser.add_argument("--A3C", type=str, default='a3c1', help="a3c version / file name")

    # parser.add_argument("--SUMMARY_DIR", default='./results')
    # parser.add_argument("--TEST_LOG_FOLDER", default='./test_results/')
    # parser.add_argument("--TRAIN_TRACES", default='/train_sim_traces/', help="relative to networkML")
    # NN_MODEL = './results/pretrain_linear_reward.ckpt'
    # parser.add_argument("--NN_MODEL", default=None)
    parser.add_argument("--TEST_NN_MODEL", default=None, help="nn_model to test")
    parser.add_argument("--LOG_FILE", default='/home/haggai/projects/networkML/data/test_results/')#'./test_results/log_sim_rl')
    parser.add_argument("--TEST_TRACES", default='/test_sim_traces/', help="relative to networkML")

    args = parser.parse_args()
    # args.TRAIN_TRACES = args.pensieve_path + '/pensieve/sim/' + args.TRAIN_TRACES
    # args.TEST_TRACES = args.pensieve_path + '/pensieve/sim/' + args.TEST_TRACES

    return args



# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward



if __name__ == '__main__':
    args = get_args()
    main(args)
