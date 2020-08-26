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

	/* ��ϡ��Ȳ�ȫʵ�飻����Ԥ���� */
	void readTargetConfiRef(Mat &target_thinout, const Mat &reference, Mat &target_dilate, Mat structure_element, Mat &reference_gray, Mat &confidence);

	/* ��ϡGTͼ */
	void thinoutGT(const Mat &GT, float delete_percent_perDim, Mat &GT_thinout);

	/* �Ӳ�ͼ�����ɴ�����Ŀհ� */
	void emptyArea( int x, int y, int width, int height, Mat &output_target, Mat &output_confi);

	/* ��ȡpfm��ʽ���ļ�����ת��Ϊmat��ʽ */
	cv::Mat loadPFM(const std::string filePath); 

	/* �洢mat��ʽΪpfm��ʽ */
	bool savePFM(const cv::Mat image, const std::string filePath);

	/* �Ӳ�ͼת���ͼ */
	void disp2Depth(cv::Mat dispMap, cv::Mat &depthMap, float fx, float baseline)
	{
		int height = dispMap.rows;
		int width = dispMap.cols;
		depthMap.create(height, width, CV_16U);
		// ���и��ĸ�ʽ
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

	/* ���ͼת�Ӳ�ͼ */
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

	/* ��������ͣ������ɶ�Ӧ������ͼ������ͼ */
	void generateConfi(const Mat &target, const Mat &target_dense, Mat &confi) {
		// 1. ���target��Ϊ0��target_dense��ҲΪ0�ĵط���confi������Ϊ0
		// 2. ���target�в�Ϊ0�ĵط���confi������Ϊ1
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
	
	/* ��ȡtxt�ļ���תΪmat��ʽ */
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

