'''
TODO:
- epochs (lines)
- iteration
- grid
- 0.25 y-axis epochscale

'''
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import argparse
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
parser = argparse.ArgumentParser()
parser.add_argument("-P", "--path", help="path to log file", required=True)
args = vars(parser.parse_args())
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
if not os.path.exists(output_dir):
  os.mkdir(output_dir)
load_path_split = args['path'].split('/')
exp_name = load_path_split[1]
output_path = os.path.join(output_dir, exp_name)
if not os.path.exists(output_path):
  os.mkdir(output_path)
column_list = ['val_type', 'epoch', 'it_n', 'it_max', 'g_rec', 'g_tsf', 'g_style', 'g_face', 'g_adv', 'g_mask', 'g_mask_smooth', 'd_real', 'd_fake', 'd_real_loss', 'd_fake_loss']
val_cols = ['g_rec', 'g_tsf', 'g_style', 'g_face', 'g_adv', 'g_mask', 'g_mask_smooth', 'd_real', 'd_fake', 'd_real_loss', 'd_fake_loss']
y_min , y_max, y_diff = (-2, 6, 0.5)
def get_value(row):
    # loss values
    values = [float(re.search(r':(?:\+|\-)?\d+\.\d+\n', x).group(0)[1:-1]) for x in row[1:-1]]
    val_dict = dict(zip(val_cols, values))
    ret_dict = {
        'val_type': re.search(r'\(\w+,', row[0]).group(0)[1:-1],
        'epoch': int(re.search(r'epoch: \d+', row[0]).group(0)[7:]),
    }
    # combining
    ret_dict.update(val_dict)
    return ret_dict
# reading the loss file
data_buffer = []
with open(args['path'],'r') as f:
    data_buffer.append(f.readlines())
# remove the training header line
data = list(filter(lambda x : len(x)<70, data_buffer[0]))
data = np.array_split(data, len(data)/11)
# extract relevant values from
clean_data = [get_value(i) for i in data]
# convert to dataframe
data_df = pd.DataFrame(clean_data)
# remove columns - not to plot
keep_cols = val_cols
keep_cols.remove('g_style')
data_splits = {
    'Validation': data_df[data_df.val_type=='V'],
    'Training': data_df[data_df.val_type!='V']
}
for name, df in data_splits.items():
    df = df.reset_index()
    df['e_comp'] = df['epoch'].diff()
    epoch_lines = df[df.e_comp!=0].index
    for i in keep_cols:
        fig, ax = plt.subplots()
        ax.plot(df.index, df.loc[:,i], label=i, linewidth=0.75)
        ax.legend(loc='upper right', fontsize=6)
        # x axis
        ax.xaxis.set_label_text('# of epochs')
        ax.set_xticks(epoch_lines)
        ax.set_xticklabels(df.epoch[epoch_lines], fontsize=6)
        y_tick_index = np.arange(y_min, y_max, y_diff)
        ax.set_ylim([min(y_tick_index), max(y_tick_index)])
        ax.yaxis.set_label_text('Loss value')
        ax.set_yticks(y_tick_index)
        ax.set_title(exp_name + '-' + name + ' Loss - ' + i)
        plt.savefig(output_path+'/' + exp_name+'_'+name+'_'+i+'_loss.png', dpi=300)

for name, df in data_splits.items():
    df = df.reset_index()
    df['e_comp'] = df['epoch'].diff()
    epoch_lines = df[df.e_comp!=0].index 
    ax.xaxis.set_label_text('# of epochs')
    ax.set_xticks(epoch_lines)  
    ax.set_xticklabels(df.epoch[epoch_lines], fontsize=6)
    fig, ax = plt.subplots()
    for i in keep_cols:
        ax.plot(df.index, df.loc[:,i], label=i, linewidth=0.75)
    ax.legend(loc='upper right', fontsize=6)
    # x axis 
    ax.xaxis.set_label_text('# of epochs')
    ax.set_xticks(epoch_lines)
    ax.set_xticklabels(df.epoch[epoch_lines], fontsize=6)
    y_tick_index = np.arange(y_min, y_max, y_diff)
    ax.set_ylim([min(y_tick_index), max(y_tick_index)])
    ax.yaxis.set_label_text('Loss value')
    ax.set_yticks(y_tick_index)
    ax.set_title(exp_name + ' - ' + name + ' Loss')
    plt.savefig(output_path+'/'+exp_name+'_'+name+'_all_loss.png', dpi=300)
