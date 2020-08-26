#pragma once
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ximgproc.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>
#include <vector>
using namespace std;
using namespace cv;

class PreProcessing
{
public:
	PreProcessing();
	~PreProcessing();
	
	void readTargetConfiRef(fstream &target_txtfile, const Mat &reference, Mat &target, Mat &target_dilate, Mat structure_element, Mat &reference_gray, Mat &confidence);

	/* 抽稀深度补全实验；数据预处理 */
	void readTargetConfiRef(Mat &target_thinout, const Mat &reference, Mat &target_dilate, Mat structure_element, Mat &reference_gray, Mat &confidence);

	/* 抽稀GT图 */
	void thinoutGT(const Mat &GT, float delete_percent_perDim, Mat &GT_thinout);

	/* 视差图中生成大面积的空白 */
	void emptyArea( int x, int y, int width, int height, Mat &output_target, Mat &output_confi);

	/* 读取pfm格式的文件，并转换为mat格式 */
	cv::Mat loadPFM(const std::string filePath); 

	/* 存储mat格式为pfm格式 */
	bool savePFM(const cv::Mat image, const std::string filePath);

	/* 视差图转深度图 */
	void disp2Depth(cv::Mat dispMap, cv::Mat &depthMap, float fx, float baseline)
	{
		int height = dispMap.rows;
		int width = dispMap.cols;
		depthMap.create(height, width, CV_16U);
		// 自行更改格式
		short* dispData = (short*)dispMap.data;
		ushort* depthData = (ushort*)depthMap.data;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int id = i*width + j;
				if (!dispData[id])  continue;  
				depthData[id] = ushort(fx*baseline / dispData[id]);
			}
		}
	}

	/* 深度图转视差图 */
	void depth2Disp(cv::Mat depthMap, cv::Mat &dispMap,float fx,float baseline)
	{
		int height = depthMap.rows;
		int width = depthMap.cols;

		for (int i = 0; i < height; i++)
		{
			float* depthData = depthMap.ptr<float>(i);
			float* dispData = dispMap.ptr<float>(i);
			for (int j = 0; j < width; j++)
			{
				if (!depthData[j])  continue;  
				dispData[j] = fx*baseline / depthData[j];
			}
		}
	}

	/* 如果有膨胀，则生成对应于膨胀图的置信图 */
	void generateConfi(const Mat &target, const Mat &target_dense, Mat &confi) {
		// 1. 检测target中为0且target_dense中也为0的地方，confi中设置为0
		// 2. 检测target中不为0的地方，confi中设置为1
		int rows = target.rows;
		int cols = target.cols;

		for (int i = 0; i < rows; ++i) {
			const float* ptr = target.ptr<float>(i);
			const float* ptr_dense = target_dense.ptr<float>(i);
			float* ptr_confi = confi.ptr<float>(i);
			for (int j = 0; j < cols; ++j) {
				if (ptr[j] <= 0 && ptr_dense[j] <= 0)  ptr_confi[j] = 0;
				if (ptr[j] > 0) ptr_confi[j] = 1;
			}
		}
	}
	
	/* 读取txt文件并转为mat格式 */
	void txt2Mat32F(fstream &txtfile, int rows, int cols, Mat &target) {
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				txtfile >> target.at<float>(i, j);
			}
		}
		return;
	}
};

