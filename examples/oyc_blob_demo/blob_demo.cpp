/*************************************************************************
>	File Name: blob_demo.cpp
>	Description: 主程序，Blob使用的demo
>	Author: Ouyang Chao
>	Created Time:   2017-05-22
>	Last modified:  2017-05-23
************************************************************************/

#define OPEN_BLOB_DEMO

#ifdef OPEN_BLOB_DEMO


#include <iostream>
#include <vector>
#include "caffe/blob.hpp"     // Blob头文件
#include "caffe/util/io.hpp"  // 将Blob保存到磁盘或从磁盘载入内存，需包含这个头文件

using namespace std;
using namespace caffe;

int main()
{
	Blob<float> a;
	cout << "Size: " << a.shape_string() << endl;
	//a.Reshape(1, 2, 3, 4); // Deprecated
	vector<int> shape(4);
	shape[0] = 1; // num
	shape[1] = 2; // channels
	shape[2] = 3; // height
	shape[3] = 4; // width
	a.Reshape(shape);
	cout << "Size: " << a.shape_string() << endl;

	float *p = a.mutable_cpu_data();
	for (int i = 0; i < a.count(); ++i)
	{
		p[i] = i;
	}
	for (int u = 0; u < a.shape()[0]; ++u)
	{
		for (int v = 0; v < a.shape()[1]; ++v)
		{
			for (int h = 0; h < a.shape()[2]; ++h)
			{
				for (int w = 0; w < a.shape()[3]; ++w)
				{
					cout << "a[" << u << "][" << v << "][" << h << "][" << w << "] = " << a.data_at(u, v, h, w) << endl;
				}
			}
		}
	}

	cout << "===============================" << endl;

	float *q = a.mutable_cpu_diff();
	for (int i = 0; i < a.count(); ++i)
	{
		q[i] = a.count() - 1 - i; // 将diff初始化为23, 22, 21, ...
	}

	a.Update(); // 执行Update操作，将diff与data融合（data = data - diff），这也是CNN权值更新步骤的最终实施者
	for (int u = 0; u < a.shape()[0]; ++u)
	{
		for (int v = 0; v < a.shape()[1]; ++v)
		{
			for (int h = 0; h < a.shape()[2]; ++h)
			{
				for (int w = 0; w < a.shape()[3]; ++w)
				{
					cout << "a[" << u << "][" << v << "][" << h << "][" << w << "] = " << a.data_at(u, v, h, w) << endl;
				}
			}
		}
	}

	// Blob支持计算所有元素绝对值之和（L1-范数），平方和（L2-范数）
	cout << "ASUM = " << a.asum_data() << endl;
	cout << "SUMSQ = " << a.sumsq_data() << endl;


	BlobProto bp; // 构造一个BlobProto对象
	a.ToProto(&bp, true); // 将a序列化，连同diff（默认不带）
	WriteProtoToBinaryFile(bp, "temp/a.blob"); // 写入磁盘文件
	BlobProto bp2;
	ReadProtoFromBinaryFileOrDie("temp/a.blob", &bp2);
	Blob<float> b;
	b.FromProto(bp2, true); // 从序列化对象bp2中克隆b（连同形状）
	for (int u = 0; u < a.shape()[0]; ++u)
	{
		for (int v = 0; v < a.shape()[1]; ++v)
		{
			for (int h = 0; h < a.shape()[2]; ++h)
			{
				for (int w = 0; w < a.shape()[3]; ++w)
				{
					cout << "b[" << u << "][" << v << "][" << h << "][" << w << "] = " << b.data_at(u, v, h, w) << endl;
				}
			}
		}
	}

	return 0;
}

#endif  // OPEN_BLOB_DEMO