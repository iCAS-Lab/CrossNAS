
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

def update_conf(xbar, adc, dac):
    conf=open('SimConfig.ini', "r+")
    i=0
    data= conf.readlines()
    for line in data:
        i+=1
        if 'Xbar_Size' in line:
            data[i-1]='Xbar_Size = ' + str(xbar) + ',' + str(xbar)+'\n'
        elif 'Subarray_Size' in line:
            data[i-1]='Subarray_Size = ' + str(xbar) +'\n'
        elif 'DAC_Choice' in line:
            data[i-1]='DAC_Choice = ' + str(dac) +'\n'
        elif 'ADC_Choice' in line:
            data[i-1]='ADC_Choice = ' + str(adc) +'\n'
    conf.seek(0)
    conf.truncate()
    conf.writelines(data)
    conf.close()

def add_model (cand, qcand):
    kernel_list = [32, 64, 128]
    weight_list = [5, 7, 9]
    activation_list = [5, 7, 9]
    choice = [x//3 for x in cand]
    kchoice = [x%3 for x in cand]
    choice1=[]
    kchoice1=[]
    for num in range(len(choice)):
        if choice[num] != 3:
            choice1.append(choice[num])
            kchoice1.append(kchoice[num])
    print("choice = ", choice1)
    print("kchoice = ", kchoice1)
    wchoice = [x//3 for x in qcand]
    achoice = [x%3 for x in qcand]
    print("whoice = ", wchoice)
    print("achoice = ", achoice)
    kchoice1 = [kernel_list[i] for i in kchoice1]
    kchoice1.insert(0, 3)
    wchoice = [weight_list[i] for i in wchoice]
    achoice = [activation_list[i] for i in achoice]
    layers = len(choice1)
    code_r=open('MNSIM/Interface/network.py','r+')
    i=0
    lb=20 #lines per block
    data= code_r.readlines()
    vgg=0
    for line in data:
        i+=1
        if '#add_new_model_network' in line:
            data.insert(i, '    elif cate.startswith(\''+ str(qcand) +'\'):\n')
            for j in range(layers):
                if choice1[j] == 0:
                    vgg += 1
                    data.insert(i+j*lb+1,'        \n')
                    data.insert(i+j*lb+2,'        \n')
                    data.insert(i+j*lb+3,'        layer_config_list.append({\'type\': \'conv\', \'in_channels\': '+ str(kchoice1[j]) +', \'out_channels\': ' + str(kchoice1[j+1]) + ', \'kernel_size\': 3, \'padding\': 1, \'stride\': 1})\n')
                    data.insert(i+j*lb+4,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+5,'        layer_config_list.append({\'type\': \'bn\', \'features\': ' + str(kchoice1[j+1]) + '})\n')
                    data.insert(i+j*lb+6,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+7,'        layer_config_list.append({\'type\': \'relu\'})\n')
                    data.insert(i+j*lb+8,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+9,'        layer_config_list.append({\'type\': \'conv\', \'in_channels\': ' + str(kchoice1[j+1]) + ', \'out_channels\': ' + str(kchoice1[j+1]) + ', \'kernel_size\': 3, \'padding\': 1, \'stride\': 1})\n')
                    data.insert(i+j*lb+10,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+11,'        layer_config_list.append({\'type\': \'bn\', \'features\': ' + str(kchoice1[j+1]) + '})\n')
                    data.insert(i+j*lb+12,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+13,'        layer_config_list.append({\'type\': \'relu\'})\n')
                    data.insert(i+j*lb+14,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+15,'        layer_config_list.append({\'type\': \'pooling\', \'mode\': \'MAX\', \'kernel_size\': 2, \'stride\': 2})\n')
                    data.insert(i+j*lb+16,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+17,'        \n')
                    data.insert(i+j*lb+18,'        \n')
                    data.insert(i+j*lb+19,'        \n')
                    data.insert(i+j*lb+20,'        \n')
                    #data.insert(i+j*lb+8,'        \n')
                elif choice1[j] == 1: 
                    data.insert(i+j*lb+1,'        \n')
                    data.insert(i+j*lb+2,'        \n')
                    data.insert(i+j*lb+3,'        layer_config_list.append({\'type\': \'conv\', \'in_channels\': '+ str(kchoice1[j]) +', \'out_channels\': ' + str(kchoice1[j+1]) + ', \'kernel_size\': 3, \'padding\': 1, \'stride\': 1})\n')
                    data.insert(i+j*lb+4,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+5,'        layer_config_list.append({\'type\': \'bn\', \'features\': ' + str(kchoice1[j+1]) + '})\n')
                    data.insert(i+j*lb+6,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+7,'        layer_config_list.append({\'type\': \'relu\'})\n')
                    data.insert(i+j*lb+8,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+9,'        layer_config_list.append({\'type\': \'conv\', \'in_channels\': ' + str(kchoice1[j+1]) + ', \'out_channels\': ' + str(kchoice1[j+1]) + ', \'kernel_size\': 3, \'padding\': 1, \'stride\': 1})\n')
                    data.insert(i+j*lb+10,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+11,'        layer_config_list.append({\'type\': \'bn\', \'features\': ' + str(kchoice1[j+1]) + '})\n')
                    data.insert(i+j*lb+12,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+13,'        layer_config_list.append({\'type\': \'relu\'})\n')
                    data.insert(i+j*lb+14,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+15,'        \n')
                    data.insert(i+j*lb+16,'        \n')
                    data.insert(i+j*lb+17,'        \n')
                    data.insert(i+j*lb+18,'        \n')
                    data.insert(i+j*lb+19,'        \n')
                    data.insert(i+j*lb+20,'        \n')
                    #data.insert(i+j*lb+8,'        \n')
                elif choice1[j] == 2:
                    if j==0:
                        data.insert(i+j*lb+1,'        layer_config_list.append({\'type\': \'pooling\', \'mode\': \'MAX\', \'kernel_size\': 1, \'stride\': 1})\n')
                        data.insert(i+j*lb+2,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    else:
                        data.insert(i+j*lb+1,'        \n')
                        data.insert(i+j*lb+2,'        \n')
                    data.insert(i+j*lb+3,'        layer_config_list.append({\'type\': \'conv\', \'in_channels\': '+ str(kchoice1[j]) +', \'out_channels\': ' + str(kchoice1[j+1]) + ', \'kernel_size\': 3, \'padding\': 1, \'stride\': 2})\n')
                    data.insert(i+j*lb+4,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+5,'        layer_config_list.append({\'type\': \'bn\', \'features\': ' + str(kchoice1[j+1]) + '})\n')
                    data.insert(i+j*lb+6,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+7,'        layer_config_list.append({\'type\': \'relu\'})\n')
                    data.insert(i+j*lb+8,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+9,'        layer_config_list.append({\'type\': \'conv\', \'in_channels\': ' + str(kchoice1[j+1]) + ', \'out_channels\': ' + str(kchoice1[j+1]) + ', \'kernel_size\': 3, \'padding\': 1, \'stride\': 1})\n')
                    data.insert(i+j*lb+10,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+11,'        layer_config_list.append({\'type\': \'bn\', \'features\': ' + str(kchoice1[j+1]) + '})\n')
                    data.insert(i+j*lb+12,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+13,'        layer_config_list.append({\'type\': \'conv\', \'in_channels\': '+ str(kchoice1[j]) +', \'out_channels\': ' + str(kchoice1[j+1]) + ', \'kernel_size\': 1, \'padding\': 0, \'stride\': 2, \'input_index\': [-6]})\n')
                    data.insert(i+j*lb+14,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+15,'        layer_config_list.append({\'type\': \'bn\', \'features\': ' + str(kchoice1[j+1]) + '})\n')
                    data.insert(i+j*lb+16,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+17,'        layer_config_list.append({\'type\': \'element_sum\', \'input_index\': [-1, -3]})\n')
                    data.insert(i+j*lb+18,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                    data.insert(i+j*lb+19,'        layer_config_list.append({\'type\': \'relu\'})\n')
                    data.insert(i+j*lb+20,'        quantize_config_list.append({\'weight_bit\': '+ str(wchoice[j]) + ', \'activation_bit\': '+ str(achoice[j]) + ', \'point_shift\': -2})\n')
                else:
                    continue
            
            data.insert(i+layers*lb+1,'        layer_config_list.append({\'type\': \'pooling\', \'mode\': \'ADA\', \'kernel_size\': 2, \'stride\': 2})\n')
            data.insert(i+layers*lb+2,'        quantize_config_list.append({\'weight_bit\': '+ str(9) + ', \'activation_bit\': '+ str(9) + ', \'point_shift\': -2})\n')
            fc_input = 2*2*kchoice1[layers]
            vgg = 2
            data.insert(i+layers*lb+4-vgg+1,'        layer_config_list.append({\'type\': \'view\'})\n')
            data.insert(i+layers*lb+4-vgg+2,'        quantize_config_list.append({\'weight_bit\': '+ str(9) + ', \'activation_bit\': '+ str(9) + ', \'point_shift\': -2})\n')
            data.insert(i+layers*lb+4-vgg+3,'        layer_config_list.append({\'type\': \'fc\', \'in_features\': '+ str(fc_input) +', \'out_features\': 100})\n')
            data.insert(i+layers*lb+4-vgg+4,'        quantize_config_list.append({\'weight_bit\': '+ str(9) + ', \'activation_bit\': '+ str(9) + ', \'point_shift\': -2})\n')
    print('Network added to MNSIM')
    code_r.seek(0)
    code_r.truncate()
    code_r.writelines(data)
    code_r.close()
   
def run_mnsim (name):
    home_path = os.getcwd()
    # print(home_path)
    SimConfig_path = os.path.join(home_path, "SimConfig.ini")
    weights_file_path = os.path.join(home_path, "MNSIM/Interface/zoo/cifar100_[4, 2, 8, 5]_qbest_params.pth")#cifar100_[5, 2, 8, 7, 5]_qmix_params.pth")
    print(SimConfig_path)
     
    parser = argparse.ArgumentParser(description='MNSIM example')
    parser.add_argument("-AutoDelete", "--file_auto_delete", default=True,
        help="Whether delete the unnecessary files automatically")
    parser.add_argument("-HWdes", "--hardware_description", default=SimConfig_path,
        help="Hardware description file location & name, default:/MNSIM_Python/SimConfig.ini")
    parser.add_argument("-Weights", "--weights", default=weights_file_path,
        help="NN model weights file location & name, default:/MNSIM_Python/cifar10_vgg8_params.pth")
    parser.add_argument("-NN", "--NN", default='SimpleCNN',
        help="NN model description (name), default: vgg8")
    parser.add_argument("-DisHW", "--disable_hardware_modeling", action='store_true', default=False,
        help="Disable hardware modeling, default: false")
    parser.add_argument("-DisAccu", "--disable_accuracy_simulation", action='store_true', default=False,
        help="Disable accuracy simulation, default: false")
    parser.add_argument("-SAF", "--enable_SAF", action='store_true', default=False,
        help="Enable simulate SAF, default: false")
    parser.add_argument("-Var", "--enable_variation", action='store_true', default=False,
        help="Enable simulate variation, default: false")
    parser.add_argument("-Rratio", "--enable_R_ratio", action='store_true', default=False,
        help="Enable simulate the effect of R ratio, default: false")
    parser.add_argument("-FixRange", "--enable_fixed_Qrange", action='store_true', default=False,
        help="Enable fixed quantization range (max value), default: false")
    parser.add_argument("-DisPipe", "--disable_inner_pipeline", action='store_true', default=False,
        help="Disable inner layer pipeline in latency modeling, default: false")
    parser.add_argument("-D", "--device", default=0,
        help="Determine hardware device (CPU or GPU-id) for simulation, default: CPU")
    parser.add_argument("-DisModOut", "--disable_module_output", action='store_true', default=True,
        help="Disable module simulation results output, default: false")
    parser.add_argument("-DisLayOut", "--disable_layer_output", action='store_true', default=True,
        help="Disable layer-wise simulation results output, default: false")
    args = parser.parse_args()
    print("Hardware description file location:", args.hardware_description)
    print("Software model file location:", args.weights)
    print("Whether perform hardware simulation:", not (args.disable_hardware_modeling))
    print("Whether perform accuracy simulation:", not (args.disable_accuracy_simulation))
    print("Whether consider SAFs:", args.enable_SAF)
    print("Whether consider variations:", args.enable_variation)
    if args.enable_fixed_Qrange:
        print("Quantization range: fixed range (depends on the maximum value)")
    else:
        print("Quantization range: dynamic range (depends on the data distribution)")
    
    mapping_start_time = time.time()
    
    __TestInterface = TrainTestInterface(network_module=name, dataset_module='MNSIM.Interface.cifar100',  
        SimConfig_path=SimConfig_path, weights_file=args.weights, device=1)
    structure_file = __TestInterface.get_structure()
    TCG_mapping = TCG(structure_file, SimConfig_path)
    # print(TCG_mapping.max_inbuf_size)
    # print(TCG_mapping.max_outbuf_size)
    mapping_end_time = time.time()
    if not (args.disable_hardware_modeling):
        hardware_modeling_start_time = time.time()
        __latency = Model_latency(NetStruct=structure_file, SimConfig_path=SimConfig_path, TCG_mapping=TCG_mapping)
        if not (args.disable_inner_pipeline):
            __latency.calculate_model_latency(mode=1)
            # __latency.calculate_model_latency_nopipe()
        else:
            __latency.calculate_model_latency_nopipe()
        hardware_modeling_end_time = time.time()
        #print("========================Latency Results=================================")
        final_latency= __latency.model_latency_output(not (args.disable_module_output), not (args.disable_layer_output))
        
        __area = Model_area(NetStruct=structure_file, SimConfig_path=args.hardware_description, TCG_mapping=TCG_mapping)
        #print("========================Area Results=================================")
        final_area = __area.model_area_output(not (args.disable_module_output), not (args.disable_layer_output))
        
        __power = Model_inference_power(NetStruct=structure_file, SimConfig_path=SimConfig_path, TCG_mapping=TCG_mapping)
        #print("========================Power Results=================================")
        __power.model_power_output(not (args.disable_module_output), not (args.disable_layer_output))
        
        __energy = Model_energy(NetStruct=structure_file, SimConfig_path=SimConfig_path,
                                TCG_mapping=TCG_mapping,
                                model_latency=__latency, model_power=__power)
        
        #print("========================Energy Results=================================")
        final_energy=__energy.model_energy_output(not (args.disable_module_output), not (args.disable_layer_output))
        #return final_latency, final_energy
    
    try:
        print("Loading pretrained model...")
        __TestInterface.net.load_change_weights(torch.load(os.path.join(os.path.dirname(__file__), f'MNSIM/Interface/zoo99/{__TestInterface.network_module}_params.pth'), map_location=__TestInterface.device))
        print("Loaded")
    except:
        print("Training...")
        __TestInterface.train_net(0.0008, 200)

    if not (args.disable_accuracy_simulation):
        print("======================================")
        print("Accuracy simulation will take a few minutes on GPU")
        accuracy_modeling_start_time = time.time()
        weight = __TestInterface.get_net_bits()
        weight_2 = weight_update(args.hardware_description, weight,
                                 is_Variation=args.enable_variation, is_SAF=args.enable_SAF, is_Rratio=args.enable_R_ratio)
        if not False: #(args.enable_fixed_Qrange):
            print("Original accuracy:", __TestInterface.origin_evaluate(method='FIX_TRAIN', adc_action='SCALE'))
            final_accuracy=__TestInterface.set_net_bits_evaluate(weight_2, adc_action='SCALE')
            print("PIM-based computing accuracy:", final_accuracy)
        else:
            print("Original accuracy:", __TestInterface.origin_evaluate(method='FIX_TRAIN', adc_action='FIX'))
            print("PIM-based computing accuracy:", __TestInterface.set_net_bits_evaluate(weight_2, adc_action='FIX'))
        accuracy_modeling_end_time = time.time()

    mapping_time = mapping_end_time - mapping_start_time
    
    
    print("Mapping time:", mapping_time)
    if not (args.disable_hardware_modeling):
        hardware_modeling_time = hardware_modeling_end_time - hardware_modeling_start_time
        print("Hardware modeling time:", hardware_modeling_time)
    else:
        hardware_modeling_time = 0
    if not (args.disable_accuracy_simulation):
        accuracy_modeling_time = accuracy_modeling_end_time - accuracy_modeling_start_time
        print("Accuracy modeling time:", accuracy_modeling_time)
    else:
        accuracy_modeling_time = 0
    print("Total simulation time:", mapping_time+hardware_modeling_time+accuracy_modeling_time)
    return final_accuracy, final_latency, final_energy, final_area
    
def get_results(cand):
    L = 4
    xbar_list = [32, 64, 128, 256]
    dac_list = [1, 2]
    xbar = cand[L]%4
    dac = cand[L]//4
    update_conf(xbar_list[xbar], cand[L+1]+1, dac_list[dac])
    print(str(list(cand[:L])))
    accuracy,latency,energy,area = run_mnsim(str(cand[:L]))
    return accuracy, latency, energy, area
