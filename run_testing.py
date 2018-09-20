###################################################
#
#   Script to execute the prediction
#
##################################################

import os, sys
from configparser import ConfigParser


#config file to read from
config = ConfigParser()
config.read('configuration.txt')

#===========================================
#name of the experiment
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('testing settings', 'nohup')   #std output on log file?


#create a folder for the results
result_dir = name_experiment
print ("\n1. Create directory for the results (if not already existing)")
if os.path.exists(result_dir):
    print ("\nDir already exists")
else:
    os.system('mkdir ' + result_dir)
    print ("\nDir created")


# finally run the prediction
if nohup:
    print ("\n2. Run the prediction on GPU  with nohup")
    os.system(' nohup python -u ./src/retinaNN_predict.py > ' +'./'+name_experiment+'/'+name_experiment+'_prediction.nohup')
else:
    print ("\n2. Run the prediction on GPU (no nohup)")
    os.system(' python ./src/retinaNN_predict.py')
