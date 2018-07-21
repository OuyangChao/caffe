/*************************************************************************
  > File Name: test.cpp
  > Description:
  > Author: ouyangchao
  > Mail: ouyangchao16@gmail.com
  > Created Time:   2018-04-17 10:35:08
  > Last modified:  2018-04-17 10:35:09
 ************************************************************************/

#include <iostream>
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

using namespace std;
using namespace caffe;

typedef double Dtype;

Dtype data[] = {
    2, 3, 4, 5, 6, 3, 2,
    5, 7, 8, 5, 3, 2, 2,
    7, 8, 3, 1, 2, 4, 2,
    6, 4, 3, 8, 2, 5, 9,
    1, 2, 6, 4, 2, 8, 4,
    9, 1, 3, 2, 4, 2, 6,
    3, 5, 2, 5, 3, 2, 5
};

Dtype weight[] = {
    1, 2, 1,
    4, 2, 1,
    4, 2, 2
};

void print_blob(const Blob<Dtype>& a)
{
    for (int u = 0; u < a.shape()[0]; ++u)
	{
		for (int v = 0; v < a.shape()[1]; ++v)
		{
			for (int h = 0; h < a.shape()[2]; ++h)
			{
				for (int w = 0; w < a.shape()[3]; ++w)
				{
					cout <<  a.data_at(u, v, h, w) << " ";
				}
                cout << endl;
			}
            cout << "==========================" << endl;
		}
        cout << "************************" << endl;
	}
}

void test_conv()
{
    // 卷积层参数
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_kernel_size(3);
    convolution_param->add_stride(1);
    convolution_param->set_num_output(1);
    convolution_param->set_bias_term(false);
    shared_ptr<Layer<Dtype> > conv_layer(
        new ConvolutionLayer<Dtype>(layer_param));
    conv_layer->blobs().resize(1);
    conv_layer->blobs()[0].reset(new Blob<Dtype>(1, 1, 3, 3));
    conv_layer->blobs()[0]->set_cpu_data(weight);

    // 输入定义
    Blob<Dtype> input;
	vector<int> shape(4);
	shape[0] = 1; // num
	shape[1] = 1; // channels
	shape[2] = 7; // height
	shape[3] = 7; // width
	input.Reshape(shape);
    input.set_cpu_data(data);
    vector<Blob<Dtype>*> vec_bottom;
    vec_bottom.push_back(&input);
    print_blob(input); // print input blob

    // 输出定义
    Blob<Dtype> output;
    vector<Blob<Dtype>*> vec_top;
    vec_top.push_back(&output);

    // 前向运算
    conv_layer->SetUp(vec_bottom, vec_top);
    conv_layer->Forward(vec_bottom, vec_top);
    print_blob(*vec_top[0]); // print output blob
}


void test_softmax()
{
    LayerParameter layer_param;
    SoftmaxLayer<Dtype> layer(layer_param);

    // 输入定义
    Blob<Dtype> input;
	vector<int> shape(4);
	shape[0] = 1; // num
	shape[1] = 7; // channels
	shape[2] = 1; // height
	shape[3] = 7; // width
	input.Reshape(shape);
    input.set_cpu_data(data);
    vector<Blob<Dtype>*> vec_bottom;
    vec_bottom.push_back(&input);
    print_blob(input); // print input blob

    // 输出定义
    Blob<Dtype> output;
    vector<Blob<Dtype>*> vec_top;
    vec_top.push_back(&output);

    // 前向运算
    layer.SetUp(vec_bottom, vec_top);
    layer.Forward(vec_bottom, vec_top);
    print_blob(*vec_top[0]); // print output blob

    // Test sum
    for (int i = 0; i < vec_bottom[0]->num(); ++i) {
        for (int k = 0; k < vec_bottom[0]->height(); ++k) {
            for (int l = 0; l < vec_bottom[0]->width(); ++l) {
                Dtype sum = 0;
                for (int j = 0; j < vec_top[0]->channels(); ++j) {
                    sum += vec_top[0]->data_at(i, j, k, l);
                }
                // Test exact values
                Dtype scale = 0;
                for (int j = 0; j < vec_bottom[0]->channels(); ++j) {
                    scale += exp(vec_bottom[0]->data_at(i, j, k, l));
                }
                cout << scale << endl;
                for (int j = 0; j < vec_bottom[0]->channels(); ++j) {
                    cout << "1: " << vec_top[0]->data_at(i, j, k, l) + 1e-4 << endl;
                    cout << "2: " << exp(vec_bottom[0]->data_at(i, j, k, l)) / scale << endl;
                }
            }
        }
    }
}


int main()
{
    //test_conv();
    //test_softmax();

    return 0;
}