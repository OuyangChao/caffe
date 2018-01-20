"""
  Description:
      Parse log generated by Caffe.
  Usage:
      python plot_curve.py --logs mnist.log --out training-curve.png
  Author:
      Chao Ouyang
  Date:
      2018-01-16
"""
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse

res = [re.compile('Iteration (\d+).*?loss.*?Train net output #0: loss = ([.e\-\d]+)'),
       re.compile('Iteration (\d+), Testing net.*?Test net output #1: loss = ([.e\-\d]+)'),
       re.compile('Iteration (\d+), Testing net.*?Test net output #0: accuracy = ([.\d]+)')]


def plot_acc(log_name, color):
    title_name = log_name.split('/')[-1].replace('.log', '')
    train_name = title_name + '-train'
    val_name = title_name + '-val'

    with open(log_name) as f:
        lines = f.readlines()

    lines = "".join(lines)
    lines = lines.replace('\n', '')
    train_loss = res[0].findall(lines)
    val_loss = res[1].findall(lines)
    val_acc = res[2].findall(lines)

    x_train = [int(iter[0]) for iter in train_loss]
    y_train_loss = [float(iter[1]) for iter in train_loss]
    x_val = [int(iter[0]) for iter in val_loss]
    y_val_loss = [float(iter[1]) for iter in val_loss]
    y_val_acc = [float(iter[1]) for iter in val_acc]

    # loss
    y_step = [0.0, 1.000001, 0.05]
    fig,ax1 = plt.subplots(figsize=(14, 8))
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.plot(x_train, y_train_loss, linestyle='--', color=color[0], linewidth=2, label=train_name+'-loss')
    ax1.plot(x_val, y_val_loss, linestyle='-', color=color[1], linewidth=2, label=val_name+'-loss')
    ax1.plot(x_val, y_val_acc, linestyle='-', color=color[2], linewidth=2, label=val_name+'-acc')
    ax1.legend(loc='best')
    ax1.set_xticks(np.arange(0, x_train[-1]+1, x_train[-1]//20))
    ax1.set_yticks(np.arange(y_step[0], y_step[1], y_step[2]))
    ax1.set_xlim([0, x_train[-1]])
    ax1.set_ylim([y_step[0], y_step[1]])
    ax1.grid(True)
    ax1.set_title(title_name)
    # acc
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ax2.set_yticks(np.arange(y_step[0], y_step[1], y_step[2]))


def main():
    color = [('r', 'g', 'b'), ('c', 'm', 'y'), ('k', 'w', 'r')]
    log_files = [i for i in args.logs.split(',')]
    color = color[:len(log_files)]
    for c in range(len(log_files)):
        plot_acc(log_files[c], color[c])
    if (args.out != ''):
        plt.savefig(args.out)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parses log file generated by Caffe')
    parser.add_argument('--logs', type=str, default="resnet-50.log",
                        help='the path of log file, --logs=resnet-50.log,resnet-101.log')
    parser.add_argument('--out', type=str, default="", help='the name of output curve ')
    args = parser.parse_args()
    main()
