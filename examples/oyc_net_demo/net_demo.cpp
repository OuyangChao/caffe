/*************************************************************************
>	File Name: net_demo.cpp
>	Description: 主程序，Net使用的demo
>	Author: Ouyang Chao
>	Created Time:   2017-05-25
>	Last modified:  2017-05-25
************************************************************************/

#define OPEN_NET_DEMO

#ifdef OPEN_NET_DEMO

#include <iostream>
#include <string>
#include <vector>
#include <caffe/net.hpp>

using namespace std;
using namespace caffe;

int main()
{
	string proto("models/bvlc_reference_caffenet/deploy.prototxt");
	Net<float> nn(proto, caffe::TEST);
	vector<string> bn = nn.blob_names(); // 获取Net中所有的Blob对象名
	for (int i = 0; i < bn.size(); ++i)
	{
		cout << "Blob #" << i << " : " << bn[i] << endl;
	}

	return 0;
}


#endif // OPEN_NET_DEMO