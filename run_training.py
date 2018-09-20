###################################################
#
#   Script to launch the training
#
##################################################

import os, sys
from configparser import ConfigParser


#config file to read from
config = ConfigParser()
config.read('configuration.txt')
#===========================================

name_experiment = config.get('experiment name', 'name')   #Name of experiment
nohup = config.getboolean('training settings', 'nohup')   #No Heads Up (Sends to log file if true)

#create a folder for the results
result_dir = name_experiment
print ("\n1. Create directory for the results (if not already existing)")
if os.path.exists(result_dir):
    print ("\nDir already exists")
else:
    os.system('mkdir ' +result_dir)
    print ("\nDir created")


print ("copy the configuration file in the results folder")
os.system('cp configuration.txt ./' +name_experiment+'/'+name_experiment+'_configuration.txt')

# run the experiment
if nohup:
    print ("\n2. Run the training on GPU with nohup")
    os.system(' nohup python -u ./src/retinaNN_training.py > ' +'./'+name_experiment+'/'+name_experiment+'_training.nohup')
else:
    print ("\n2. Run the training on GPU (no nohup)")
    os.system(' python ./src/retinaNN_training.py')

#Prediction/testing is run with a different script
