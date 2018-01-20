#coding=utf-8
"""
  Description:
      Visualize feature maps.
  Usage:
      python visualize.py test.png --model_def lenet.protxt --model_weights lenet.caffemodel
  Author:
      Chao Ouyang
  Date:
      2018-01-19
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import caffe


def vis_square(data):
    # 输入的数据为一个ndarray，尺寸可以为(n, height, width)或者是 (n, height, width, 3)
    # 前者即为n个灰度图像的数据，后者为n个rgb图像的数据
    # 在一个sqrt(n) by sqrt(n)的格子中，显示每一幅图像
    # 对输入的图像进行normlization
    data = (data - data.min()) / (data.max() - data.min())
    # 强制性地使输入的图像个数为平方数，不足平方数时，手动添加几幅
    n = int(np.ceil(np.sqrt(data.shape[0])))
    # 每幅小图像之间加入小空隙
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
                           + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # 将所有输入的data图像平复在一个ndarray-data中（tile the filters into an image）
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    # data的一个小例子,e.g., (3,120,120)
    # 即，这里的data是一个2d 或者 3d 的ndarray
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    # 显示data所对应的图像
    plt.imshow(data)
    plt.axis('off')
    plt.show()


def main():
    if args.cpu:
        caffe.set_mode_cpu()
        print('CPU mode')
    else:
        caffe.set_device(0)
        caffe.set_mode_gpu()
        print('GPU mode')
    
    net = caffe.Net(args.model_def,      # 定义模型结构
                    args.model_weights,  # 预训练的网络
                    caffe.TEST)          # 测试模式
    # transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer = caffe.io.Transformer({'data': (1,1,28,28)})
    transformer.set_transpose('data', (2,0,1))  # 变换image矩阵，把channel放到最后一维
    # transformer.set_raw_scale('data', 255)      # 从[0,1]rescale到[0,255]
    # transformer.set_channel_swap('data', (2,1,0))  # 调整 channels from RGB to BGR
    image = caffe.io.load_image(args.image, color=False)
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    net.forward()
    for layer_name, blob in net.blobs.iteritems():
        print(layer_name + ', ' + str(blob.data.shape))
    for layer_name, param in net.params.iteritems():
        print(layer_name + ', ' + str(param[0].data.shape) + str(param[1].data.shape))
    # filters = net.params['conv1'][0].data
    # vis_square(filters.transpose(0, 2, 3, 1).reshape(32,9,9))
    vis_square(net.blobs['data'].data[0, :36])
    feat = net.blobs['st_output'].data[0, :36]
    vis_square(feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize feature maps')
    parser.add_argument('image', type=str, help='Image to be visualized')
    parser.add_argument('--model_def', type=str,
                        default="examples/mnist_tests/ST_CNN/ST_CNN_deploy.prototxt",
                        help='Model definetion file')
    parser.add_argument('--model_weights', type=str,
                        default="examples/mnist_tests/ST_CNN/ST_CNN_iter_50000.caffemodel",
                        help='Trained model weights file')
    parser.add_argument('--cpu', action='store_true', help='Cpu mode')
    args = parser.parse_args()
    main()
