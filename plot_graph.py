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
#
parser = argparse.ArgumentParser()
parser.add_argument("-P", "--path", help="path to log file", required=True)
parser.add_argument("-T", "--tag", help="tag for output files", required=True)
#
args = vars(parser.parse_args())
#
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
if not os.path.exists(output_dir):
  os.mkdir(output_dir)
#
output_path = os.path.join(output_dir, args['tag'])
if not os.path.exists(output_path):
  os.mkdir(output_path)
#
column_list = ['val_type', 'epoch', 'it_n', 'it_max', 'g_rec', 'g_tsf', 'g_style', 'g_face', 'g_adv', 'g_mask', 'g_mask_smooth', 'd_real', 'd_fake']
val_cols = ['g_rec', 'g_tsf', 'g_style', 'g_face', 'g_adv', 'g_mask', 'g_mask_smooth', 'd_real', 'd_fake']  
#
y_min , y_max, y_diff = (-2, 6, 0.5)
#
def get_value(row):
    # loss values
    values = [float(re.search(r':(?:\+|\-)?\d+\.\d+\n', x).group(0)[1:-1]) for x in row[1:-1]]  
    val_dict = dict(zip(val_cols, values))
    # iteration variables
    # it_n = re.search(r'it: \d+/\d+', row[0])
    # it_max = re.search(r'it: \d+/\d+', row[0])
    ret_dict = {
        'val_type': re.search(r'\(\w+,', row[0]).group(0)[1:-1],
        'epoch': int(re.search(r'epoch: \d+', row[0]).group(0)[7:]), 
        # 'it_n': int(it_n.group(0).split('/')[0][4:]) if it_n is not None else np.NaN,
        # 'it_max': int(it_max.group(0).split('/')[1]) if it_max is not None else np.NaN
    }
    # combining
    ret_dict.update(val_dict)
    # ret_dict['it_actual'] = (ret_dict['epoch']-1)*ret_dict['it_max']+ret_dict['it_n']
    # date = re.search(r'\d+/\d+/\d+', i).group(0)
    # time = re.search(r'\d+:\d+:\d+', i).group(0)
    return ret_dict
#
#
# reading the loss file
data_buffer = []
with open(args['path'],'r') as f:
# with open('./loss_log2.txt','r') as f: 
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
#
data_splits = {
    'Validation': data_df[data_df.val_type=='V'],
    'Training': data_df[data_df.val_type!='V']
}
# 
for name, df in data_splits.items():
    df = df.reset_index()
    # if name=='Training':
        # df['it_actual'] = df.it_actual.astype('int32').astype('str') 
    df['e_comp'] = df['epoch'].diff()
    epoch_lines = df[df.e_comp!=0].index 
    for i in keep_cols:
        fig, ax = plt.subplots()
        # flag = True
        # # epoch lines
        # for xv in epoch_lines:
        #     if flag:
        #         ax.axvline(xv, color='g', linestyle='dotted', linewidth=1, label='epoch')
        #         flag = False
        #     else:
        #         ax.axvline(xv, color='g', linestyle='dotted', linewidth=1)
        #         flag = False
        # plot
        ax.plot(df.index, df.loc[:,i], label=i, linewidth=0.75)
        ax.legend(loc='upper right', fontsize=6)
        # x axis 
        ax.xaxis.set_label_text('# of epochs')
        ax.set_xticks(epoch_lines)
        ax.set_xticklabels(df.epoch[epoch_lines], fontsize=6)
        # if name=='Training':
            # x_tick_index = np.arange(0, len(df), 30)     
            # ax.xaxis.set_label_text('# of terations')
        #     ax.set_xticks(df.index.values[x_tick_index])
        #     ax.set_xticklabels(df.it_actual.values[x_tick_index], fontsize=9)
        # else :
        #     ax.get_xaxis().set_visible(False)
        #y axis
        # y_tick_index = np.arange(int(min(df[i])-1), int(max(df[i])+1), 0.75)
        y_tick_index = np.arange(y_min, y_max, y_diff)
        ax.set_ylim([min(y_tick_index), max(y_tick_index)])
        ax.yaxis.set_label_text('Loss value')
        ax.set_yticks(y_tick_index)
        #
        ax.set_title(args['tag'] + '-' + name + ' Loss - ' + i)
        #
        plt.savefig(output_path+'/' + args['tag']+'_'+name+'_'+i+'_loss.png', dpi=300)

for name, df in data_splits.items():
    df = df.reset_index()
    # if name=='Training':
        # df['it_actual'] = df.it_actual.astype('int32').astype('str') 
    df['e_comp'] = df['epoch'].diff()
    epoch_lines = df[df.e_comp!=0].index 
    ax.xaxis.set_label_text('# of epochs')
    ax.set_xticks(epoch_lines)  
    ax.set_xticklabels(df.epoch[epoch_lines], fontsize=6)
    fig, ax = plt.subplots()
    # flag = True
    # epoch lines
    # for xv in epoch_lines:
    #     if flag:
    #         ax.axvline(xv, color='g', linestyle='dotted', linewidth=1, label='epoch')
    #         flag = False
    #     else:
    #         ax.axvline(xv, color='g', linestyle='dotted', linewidth=1)
    #         flag = False
    for i in keep_cols:
        # plot
        ax.plot(df.index, df.loc[:,i], label=i, linewidth=0.75)
        #
    ax.legend(loc='upper right', fontsize=6)
    # x axis 
    ax.xaxis.set_label_text('# of epochs')
    ax.set_xticks(epoch_lines)
    ax.set_xticklabels(df.epoch[epoch_lines], fontsize=6)
    # if name=='Training':
    #     x_tick_index = np.arange(0, len(df), 25)     
    #     ax.xaxis.set_label_text('# of terations')
    #     ax.set_xticks(df.index.values[x_tick_index])
    #     ax.set_xticklabels(df.it_actual.values[x_tick_index], fontsize=9)
    # else :
    #     ax.get_xaxis().set_visible(False)
    #y axis
    # y_tick_index = np.arange(int(df[keep_cols].min(0).min(0)-1), int(df[keep_cols].max(0).max(0)+1), 0.75)
    y_tick_index = np.arange(y_min, y_max, y_diff)
    ax.set_ylim([min(y_tick_index), max(y_tick_index)])
    ax.yaxis.set_label_text('Loss value')
    ax.set_yticks(y_tick_index)
    #
    ax.set_title(args['tag'] + ' - ' + name + ' Loss')
    #
    plt.savefig(output_path+'/'+args['tag']+'_'+name+'_all_loss.png', dpi=300)






