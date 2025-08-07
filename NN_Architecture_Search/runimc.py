#!/usr/bin/python
# -*-coding:utf-8-*-
import torch
import sys
import os
import math
import argparse
import numpy as np
import torch
import collections
import configparser
import time
from importlib import import_module
from MNSIM.Interface.interface import *
from MNSIM.Accuracy_Model.Weight_update import weight_update
from MNSIM.Mapping_Model.Behavior_mapping import behavior_mapping
from MNSIM.Mapping_Model.Tile_connection_graph import TCG
from MNSIM.Latency_Model.Model_latency import Model_latency
from MNSIM.Area_Model.Model_Area import Model_area
from MNSIM.Power_Model.Model_inference_power import Model_inference_power
from MNSIM.Energy_Model.Model_energy import Model_energy

def add_model (choice, kchoice, cand):
    code_r=open('MNSIM/Interface/network.py','r+')
    #params1=params.copy()
    #choice1 = [i for i in choice]
    choice1=[]
    kchoice1=[]
    for num in range(len(choice)):
        if choice[num] != 3:
            choice1.append(choice[num])
            kchoice1.append(kchoice[num])
    print(choice1)
    print(kchoice1)
    kchoice1 = [32*pow(2,i) for i in kchoice1]
    kchoice1.insert(0, 3)
    layers = len(choice1)
    i=0
    lb=7 #lines per block
    data= code_r.readlines()
    vgg=0
    for line in data:
        i+=1
        #if 'assert cate in [' in line:
        #    data[i-1]='    assert cate in [\''+name+'\']\n'
        if '#add_new_model_network' in line:
            data.insert(i, '    elif cate.startswith(\''+ str(cand) +'\'):\n')
            for j in range(layers):
                if choice1[j] == 0:
                    vgg += 1
                    data.insert(i+j*lb+1,'        \n')
                    data.insert(i+j*lb+2,'        layer_config_list.append({\'type\': \'conv\', \'in_channels\': '+ str(kchoice1[j]) +', \'out_channels\': ' + str(kchoice1[j+1]) + ', \'kernel_size\': 3, \'padding\': 1, \'stride\': 1})\n')
                    data.insert(i+j*lb+3,'        layer_config_list.append({\'type\': \'relu\'})\n')
                    data.insert(i+j*lb+4,'        layer_config_list.append({\'type\': \'conv\', \'in_channels\': ' + str(kchoice1[j+1]) + ', \'out_channels\': ' + str(kchoice1[j+1]) + ', \'kernel_size\': 3, \'padding\': 1, \'stride\': 1})\n')
                    data.insert(i+j*lb+5,'        layer_config_list.append({\'type\': \'relu\'})\n')
                    data.insert(i+j*lb+6,'        layer_config_list.append({\'type\': \'pooling\', \'mode\': \'MAX\', \'kernel_size\': 2, \'stride\': 2})\n')
                    data.insert(i+j*lb+7,'        \n')
                    #data.insert(i+j*lb+8,'        \n')
                elif choice1[j] == 1:
                    if j==0:
                        data.insert(i+j*lb+1,'        layer_config_list.append({\'type\': \'pooling\', \'mode\': \'MAX\', \'kernel_size\': 1, \'stride\': 1})\n')
                    else:
                        data.insert(i+j*lb+1,'        \n')
                    data.insert(i+j*lb+2,'        layer_config_list.append({\'type\': \'conv\', \'in_channels\': '+ str(kchoice1[j]) +', \'out_channels\': ' + str(kchoice1[j+1]) + ', \'kernel_size\': 3, \'padding\': 1, \'stride\': 1})\n')
                    data.insert(i+j*lb+3,'        layer_config_list.append({\'type\': \'relu\'})\n')
                    data.insert(i+j*lb+4,'        layer_config_list.append({\'type\': \'conv\', \'in_channels\': ' + str(kchoice1[j+1]) + ', \'out_channels\': ' + str(kchoice1[j+1]) + ', \'kernel_size\': 3, \'padding\': 1, \'stride\': 1})\n')
                    data.insert(i+j*lb+5,'        layer_config_list.append({\'type\': \'conv\', \'in_channels\': '+ str(kchoice1[j]) +', \'out_channels\': ' + str(kchoice1[j+1]) + ', \'kernel_size\': 1, \'padding\': 0, \'stride\': 1, \'input_index\': [-4]})\n')
                    data.insert(i+j*lb+6,'        layer_config_list.append({\'type\': \'element_sum\', \'input_index\': [-1, -2]})\n')
                    data.insert(i+j*lb+7,'        layer_config_list.append({\'type\': \'relu\'})\n')
                    #data.insert(i+j*lb+7,'        layer_config_list.append({\'type\': \'pooling\', \'mode\': \'MAX\', \'kernel_size\': 2, \'stride\': 2})\n')
                    #data.insert(i+j*lb+8,'        \n')
                elif choice1[j] == 2:
                    data.insert(i+j*lb+1,'        \n')
                    data.insert(i+j*lb+2,'        layer_config_list.append({\'type\': \'conv\', \'in_channels\': '+ str(kchoice1[j]) +', \'out_channels\': ' + str(kchoice1[j+1]) + ', \'kernel_size\': 3, \'padding\': 1, \'stride\': 1})\n')
                    data.insert(i+j*lb+3,'        layer_config_list.append({\'type\': \'relu\'})\n')
                    data.insert(i+j*lb+4,'        layer_config_list.append({\'type\': \'conv\', \'in_channels\': ' + str(kchoice1[j+1]) + ', \'out_channels\': ' + str(kchoice1[j+1]) + ', \'kernel_size\': 3, \'padding\': 1, \'stride\': 1})\n')
                    data.insert(i+j*lb+5,'        layer_config_list.append({\'type\': \'relu\'})\n')
                    data.insert(i+j*lb+6,'        \n')
                    data.insert(i+j*lb+7,'        \n')
                    #data.insert(i+j*lb+8,'        \n')
                else:
                    continue
                    #kchoice1[j] = kchoice1[j-1]
            #fc_shape = int(32/(2**vgg))
            '''
            for k in range(4-vgg):
                data.insert(i+layers*lb+k,'        layer_config_list.append({\'type\': \'pooling\', \'mode\': \'MAX\', \'kernel_size\': 2, \'stride\': 2})\n')
            '''
            data.insert(i+layers*lb+1,'        layer_config_list.append({\'type\': \'pooling\', \'mode\': \'ADA\', \'kernel_size\': 2, \'stride\': 2})\n')
            fc_input = 2*2*kchoice1[layers]
            vgg = 3
            data.insert(i+layers*lb+4-vgg+1,'        layer_config_list.append({\'type\': \'view\'})\n')
            #data.insert(i+(params['num_layer'])*lb+2,'        layer_config_list.append({\'type\': \'dropout\'})\n')
            data.insert(i+layers*lb+4-vgg+2,'        layer_config_list.append({\'type\': \'fc\', \'in_features\': '+ str(fc_input) +', \'out_features\': 200})\n')
            data.insert(i+layers*lb+4-vgg+3,'        layer_config_list.append({\'type\': \'relu\'})\n')
            data.insert(i+layers*lb+4-vgg+4,'        layer_config_list.append({\'type\': \'dropout\'})\n')
            data.insert(i+layers*lb+4-vgg+5,'        layer_config_list.append({\'type\': \'fc\', \'in_features\': 200, \'out_features\': 50})\n')
            data.insert(i+layers*lb+4-vgg+6,'        layer_config_list.append({\'type\': \'relu\'})\n')
            data.insert(i+layers*lb+4-vgg+7,'        layer_config_list.append({\'type\': \'dropout\'})\n')
            data.insert(i+layers*lb+4-vgg+8,'        layer_config_list.append({\'type\': \'fc\', \'in_features\': 50, \'out_features\': 10})\n')
    print('Network added to MNSIM')
            
    code_r.seek(0)
    code_r.truncate()
    code_r.writelines(data)
    code_r.close()
   
def run_mnsim (name):
    home_path = os.getcwd()
    # print(home_path)
    SimConfig_path = os.path.join(home_path, "SimConfig.ini")
    __TestInterface = TrainTestInterface(network_module=name, dataset_module='MNSIM.Interface.cifar10',  
        SimConfig_path=SimConfig_path, weights_file=None, device=2)
    structure_file = __TestInterface.get_structure()
    TCG_mapping = TCG(structure_file, SimConfig_path)
    if not (False):
        #hardware_modeling_start_time = time.time()
        __latency = Model_latency(NetStruct=structure_file, SimConfig_path=SimConfig_path, TCG_mapping=TCG_mapping)
        if not (False):
            __latency.calculate_model_latency(mode=1)
            # __latency.calculate_model_latency_nopipe()
        else:
            __latency.calculate_model_latency_nopipe()
        #hardware_modeling_end_time = time.time()
        #print("========================Latency Results=================================")
        final_latency= __latency.model_latency_output(not (False), not (False))
        
        #__area = Model_area(NetStruct=structure_file, SimConfig_path=args.hardware_description, TCG_mapping=TCG_mapping)
        #print("========================Area Results=================================")
        #__area.model_area_output(not (args.disable_module_output), not (args.disable_layer_output))
        
        __power = Model_inference_power(NetStruct=structure_file, SimConfig_path=SimConfig_path, TCG_mapping=TCG_mapping)
        #print("========================Power Results=================================")
        #__power.model_power_output(not (False), not (True))
        
        __energy = Model_energy(NetStruct=structure_file, SimConfig_path=SimConfig_path,
                                TCG_mapping=TCG_mapping,
                                model_latency=__latency, model_power=__power)
        
        #print("========================Energy Results=================================")
        final_energy=__energy.model_energy_output(not (False), not (True))
        return final_latency, final_energy
