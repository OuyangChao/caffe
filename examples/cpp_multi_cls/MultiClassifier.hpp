#ifndef MULTICLASSFIER_H
#define MULTICLASSFIER_H

#include <string>
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include <opencv2/core/core.hpp>
#include <cstdio>
#include <algorithm>

using cv::Mat;
using std::string;
using std::vector;

namespace caffe{

	template<typename Dtype>
	class MultiClassifier{
	public:
		explicit MultiClassifier(const string&param_file, const string& weights_file);
		Dtype test(vector<Mat> &images, vector<int> &labels, int iter_num);
		inline shared_ptr<Net<Dtype> >net(){ return net_; }
		void predict(vector<Mat> &images, vector<vector<Dtype> > &labels);
		void predict(vector<Dtype> &data, vector<int> *labels, int num);
		void extract_feature(vector<Mat> &images, vector<vector<Dtype> > *out);
	protected:
		shared_ptr<Net<Dtype> >net_;
		MemoryDataLayer<Dtype>*m_layer_;
		int batch_size_;
		int channels_;
		int height_;
		int width_;
		DISABLE_COPY_AND_ASSIGN(MultiClassifier);
	};

    template<typename Dtype>
	MultiClassifier<Dtype>::MultiClassifier(const string&param_file, const string &weights_file) :net_()
	{
#ifdef CPU_ONLY
		Caffe::set_mode(Caffe::CPU);
#else
		Caffe::set_mode(Caffe::GPU);
#endif
		net_.reset(new Net<Dtype>(param_file, TEST));
		net_->CopyTrainedLayersFrom(weights_file);
		m_layer_ = (MemoryDataLayer<Dtype>*)net_->layers()[0].get();
		batch_size_ = m_layer_->batch_size();
		channels_ = m_layer_->channels();
		height_ = m_layer_->height();
		width_ = m_layer_->width();
	}

	template<typename Dtype>
	Dtype MultiClassifier<Dtype>::test(vector<Mat>&images, vector<int>&labels, int iter_num)
	{
		m_layer_->AddMatVector(images, labels);
		int iterations = iter_num;
		vector<Blob<Dtype>*>bottom_vec;
		vector<int> test_score_output_id;
		vector<Dtype> test_score;
		Dtype loss = 0;
		for (int i = 0; i < iterations; ++i) {
			Dtype iter_loss;
			const vector<Blob<Dtype>*>& result =
				net_->Forward(bottom_vec, &iter_loss);
			loss += iter_loss;
			int idx = 0;
			for (int j = 0; j < result.size(); ++j) {
				const Dtype* result_vec = result[j]->cpu_data();
				for (int k = 0; k < result[j]->count(); ++k, ++idx) {
					const Dtype score = result_vec[k];
					if (i == 0) {
						test_score.push_back(score);
						test_score_output_id.push_back(j);
					}
					else {
						test_score[idx] += score;
					}
					const std::string& output_name = net_->blob_names()[
						net_->output_blob_indices()[j]];
						LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
				}
			}
		}
		loss /= iterations;
		LOG(INFO) << "Loss: " << loss;
		return loss;
	}

	template <typename Dtype>
	void MultiClassifier<Dtype>::predict(vector<Mat> &images, vector<vector<Dtype> >&labels)
	{
		int original_length = images.size();
		if (original_length == 0)
			return;
		int valid_length = original_length / batch_size_ * batch_size_;
		if (original_length != valid_length)
		{
			valid_length += batch_size_;
			for (int i = original_length; i < valid_length; i++)
			{
				images.push_back(images[0].clone());
			}
		}
		vector<int>valid_labels;
		valid_labels.resize(valid_length, 0);
		m_layer_->AddMatVector(images, valid_labels);
		vector<Blob<Dtype>* > bottom_vec;
		for (int i = 0; i < valid_length / batch_size_; i++)
		{
			const vector<Blob<Dtype>*>& result = net_->Forward(bottom_vec);
			const Dtype * result_vec = result[1]->cpu_data();
			int nums = result[1]->shape()[0];
			int channels = result[1]->shape()[1];
			for (int k = 0; k < nums; k++)
			{
				vector<Dtype> temp;
				for (int j = 0; j < channels; j++)
				{
					temp.push_back(result_vec[k*channels + j]);
				}
				labels.push_back(temp);
			}
		}
		if (original_length != valid_length)
		{
			images.erase(images.begin() + original_length, images.end());
			labels.erase(labels.begin() + original_length, labels.end());
		}
	}
	
	template <typename Dtype>
	void MultiClassifier<Dtype>::extract_feature(vector<Mat> &images, vector<vector<Dtype> > *out)
	{
		int original_length = images.size();
		if (original_length == 0)
			return;
		int valid_length = original_length / batch_size_ * batch_size_;
		if (original_length != valid_length)
		{
			valid_length += batch_size_;
			for (int i = original_length; i < valid_length; i++)
			{
				images.push_back(images[0].clone());
			}
		}
		vector<int> valid_labels;
		valid_labels.resize(valid_length, 0);
		m_layer_->AddMatVector(images, valid_labels);
		vector<Blob<Dtype>* > bottom_vec;
		out->clear();
		for (int i = 0; i < valid_length / batch_size_; i++)
		{
			const vector<Blob<Dtype>*>& result = net_->Forward(bottom_vec);
			const Dtype * result_vec = result[0]->cpu_data();
			const int dim = result[0]->count(1);
			for (int j = 0; j < result[0]->num(); j++)
			{
				const Dtype * ptr = result_vec + j * dim;
				vector<Dtype> one_;
				for (int k = 0; k < dim; ++k)
					one_.push_back(ptr[k]);
				out->push_back(one_);
			}
		}
		if (original_length != valid_length)
		{
			images.erase(images.begin() + original_length, images.end());
			out->erase(out->begin() + original_length, out->end());
		}
	}
	INSTANTIATE_CLASS(MultiClassifier);
}
#endif