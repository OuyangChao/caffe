#include "MultiClassifier.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
#include <string>

#ifdef linux
#include <unistd.h>
#include <dirent.h>
#endif
#ifdef WIN32
#include <direct.h>
#include <io.h>
#endif

using namespace caffe;
using std::string;
typedef std::pair<string, float> Prediction;


void getFiles(std::string path, std::vector<std::string>& files)
{
#ifdef WIN32
	long hFile = 0;
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
#endif // WIN32

#ifdef linux
	DIR *dir;
	struct dirent *ptr;

	if ((dir = opendir(path.c_str())) == NULL)
	{
		perror("Open dir error...");
		exit(1);
	}

	while ((ptr = readdir(dir)) != NULL)
	{
		if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    ///current dir OR parrent dir
			continue;
		else if (ptr->d_type == 8)     // file
			files.push_back(path + ptr->d_name);
		else if (ptr->d_type == 10)    // link file
			continue;
		else if (ptr->d_type == 4)     // dir
		{
			files.push_back(path + ptr->d_name);
		}
	}
	closedir(dir);
#endif // linux
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " deploy.prototxt network.caffemodel"
                  << " labels.txt image/ output.txt" << std::endl;
        return 1;
    }

    ::google::InitGoogleLogging(argv[0]);
    // caffe的准备工作
    clock_t start, finish;
    double elapsedime;
    start = clock();

    string model_file = argv[1];
    string trained_file = argv[2];
    string label_file = argv[3];
    string test_dir = argv[4];
    string out_file = argv[5];
    // load lables;
    vector<string> labels_;   // 存放所有标签名称
    vector<int> labels_idx_;  // 存放分类结果标签索引
    vector<cv::Mat> input;
    vector<vector<float> > labels;  // 存放分类结果，每张图片，每个类别的概率
    std::ifstream flabels(label_file.c_str());
    CHECK(flabels) << "Unable to open labels file " << label_file;
    string line;
    while (std::getline(flabels, line))
        labels_.push_back(string(line));

    // init Multiclassifier
    MultiClassifier<float> classifier(model_file, trained_file);
    vector<string> files;
    getFiles(test_dir, files);
    for (int i = 0; i < files.size(); i++)
    {
        Mat reimg;
        Mat img = cv::imread(files[i], -1);
        cv::resize(img, reimg, cv::Size(28, 28));
        input.push_back(reimg);
    }
    std::cout << "loaded images" << std::endl;

    classifier.predict(input, labels);

    for (int i = 0; i < labels.size(); i++)
    {
        std::vector<float>::iterator max_e = std::max_element(labels[i].begin(), labels[i].end());
        labels_idx_.push_back(std::distance(labels[i].begin(), max_e));
    }

    std::ofstream out(out_file.c_str());
    for (int i = 0; i < labels_idx_.size(); i++)
    {
        int idx = labels_idx_[i];
        out << files[i] << ": " << labels_[idx] << ", " << labels[i][idx] << std::endl;
    }
    out.close();

    finish = clock();
    elapsedime = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "elapsed: " << elapsedime << " seconds" << std::endl;

    return 0;
}
