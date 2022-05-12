import os
import subprocess

MODELS_DIR = '/home/haggai/projects/networkML/pensieve/test_a3c_sim/nn_models/'
# MODELS = ['a3c', 'a3c1', 'a3c2', 'a3c3', 'pretrained']
# A3C = ['a3c', 'a3c1', 'a3c2', 'a3c3', 'a3c']
MODELS = [ 'a3c', 'a3c1','a3c2', 'a3c3', 'pretrained']
A3C = ['a3c', 'a3c1','a3c2', 'a3c3', 'a3c']
log_name = [ 'a3c_orig', 'a3c1','a3c2', 'a3c3', 'pretrained']
# files = os.listdir(MODELS_DIR)
# for f in files:
# 	suffix = '.ckpt.meta'
# 	if suffix in f:
# 		prefix = f.split(suffix)[0]
# 		MODELS.append(prefix)


for model, a3c, log in zip(MODELS, A3C, log_name):
    mod_path = os.path.join(MODELS_DIR, model)
    command = 'python rl_test_a3c.py ' + ' ' + mod_path + ' ' + a3c + ' ' + log
    print(command)
    proc = subprocess.Popen(command,  shell=True)
    proc.wait()
