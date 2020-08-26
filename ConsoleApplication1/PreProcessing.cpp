#include "PreProcessing.h"

PreProcessing::PreProcessing()
{
}

PreProcessing::~PreProcessing()
{
}

/* tof数据实验数据预处理；读取给定的target的txt文件　*/
void PreProcessing::readTargetConfiRef(fstream &target_txtfile, const Mat &reference, Mat &target, Mat &target_dilate, Mat structure_element, Mat &reference_gray, Mat &confidence) {
	int rows = reference.rows;
	int cols = reference.cols;

	// 1. 读取txt，膨胀target
	txt2Mat32F(target_txtfile, rows, cols, target);
	dilate(target, target_dilate, structure_element); //膨胀

	// 2. 预处理reference，BGR ---> Gray
	cv::cvtColor(reference, reference_gray, cv::COLOR_BGR2GRAY);

	// 3. 得到confidence
	confidence = 0.6*Mat::ones(rows, cols, CV_32FC1);
	generateConfi(target, target_dilate, confidence);
};

/* 抽稀深度补全实验数据预处理；读取mat格式的target文件　*/
void PreProcessing::readTargetConfiRef(Mat &target_thinout, const Mat &reference, Mat &target_dilate, Mat structure_element, Mat &reference_gray, Mat &confidence) {
	int rows = reference.rows;
	int cols = reference.cols;

	// 1. 膨胀target
	dilate(target_thinout, target_dilate, structure_element); //膨胀

	// 2. 预处理reference，BGR ---> Gray
	cv::cvtColor(reference, reference_gray, cv::COLOR_BGR2GRAY);

	// 3. 得到confidence
	confidence = 0.6*Mat::ones(rows, cols, CV_32FC1);
	generateConfi(target_thinout, target_dilate, confidence);
};

/* 抽稀GT；delete_percent_perDim:删除的点数比例 */
void PreProcessing::thinoutGT(const Mat &GT, float delete_percent, Mat &GT_thinout) {
	int rows = GT.rows;
	int cols = GT.cols;
	int new_rows = int(rows * delete_percent);
	int new_cols = int(cols * delete_percent);

	vector<int> index_cols;
	vector<int> index_rows;

	for (int i = 0; i < rows;++i) index_rows.push_back(i);
	for (int j = 0; j < cols;++j) index_cols.push_back(j);
	random_shuffle(index_cols.begin(), index_cols.end());

	for (auto i : index_rows) {
		float *ptr = GT_thinout.ptr<float>(i);
		random_shuffle(index_cols.begin(), index_cols.end());
		std::vector<int>::const_iterator first1 = index_cols.begin();
		std::vector<int>::const_iterator last1 = index_cols.begin() + int(delete_percent*cols);
		std::vector<int> cut_index_cols(first1, last1);
		for (auto j : cut_index_cols) {
			ptr[j] = 0;
		}
	}
}

/* 读取pfm格式的文件并将其转换为mat格式 */
Mat PreProcessing::loadPFM(const string filePath)
{
	//Open binary file
	ifstream file(filePath.c_str(), ios::in | ios::binary);

	Mat imagePFM;

	//If file correctly openened
	if (file)
	{
		//Read the type of file plus the 0x0a UNIX return character at the end
		char type[3];
		file.read(type, 3 * sizeof(char));

		//Read the width and height
		unsigned int width(0), height(0);
		file >> width >> height;

		//Read the 0x0a UNIX return character at the end
		char endOfLine;
		file.read(&endOfLine, sizeof(char));

		int numberOfComponents(0);
		//The type gets the number of color channels
		if (type[1] == 'F')
		{
			imagePFM = Mat(height, width, CV_32FC3);
			numberOfComponents = 3;
		}
		else if (type[1] == 'f')
		{
			imagePFM = Mat(height, width, CV_32FC1);
			numberOfComponents = 1;
		}

		//TODO Read correctly depending on the endianness
		//Read the endianness plus the 0x0a UNIX return character at the end
		//Byte Order contains -1.0 or 1.0
		char byteOrder[4];
		file.read(byteOrder, 4 * sizeof(char));

		//Find the last line return 0x0a before the pixels of the image
		char findReturn = ' ';
		while (findReturn != 0x0a)
		{
			file.read(&findReturn, sizeof(char));
		}

		//Read each RGB colors as 3 floats and store it in the image.
		float *color = new float[numberOfComponents];
		for (unsigned int i = 0; i<height; ++i)
		{
			for (unsigned int j = 0; j<width; ++j)
			{
				file.read((char*)color, numberOfComponents * sizeof(float));

				//In the PFM format the image is upside down
				if (numberOfComponents == 3)
				{
					//OpenCV stores the color as BGR
					imagePFM.at<Vec3f>(height - 1 - i, j) = Vec3f(color[2], color[1], color[0]);
				}
				else if (numberOfComponents == 1)
				{
					//OpenCV stores the color as BGR
					imagePFM.at<float>(height - 1 - i, j) = color[0];
				}
			}
		}

		delete[] color;

		//Close file
		file.close();
	}
	else
	{
		cerr << "Could not open the file : " << filePath << endl;
	}

	// 将imagePFM中的最大值改为0
	for (int i = 0;i < imagePFM.rows;++i) {
		float* ptr = imagePFM.ptr<float>(i);
		for (int j = 0; j < imagePFM.cols;++j) {
			if (ptr[j] >= FLT_MAX) ptr[j] = 0;
		}
	}

	return imagePFM;
}

/* 存储mat文件为pfm文件 */
bool PreProcessing::savePFM(const cv::Mat image, const std::string filePath)
{
	//Open the file as binary!
	ofstream imageFile(filePath.c_str(), ios::out | ios::trunc | ios::binary);

	if (imageFile)
	{
		int width(image.cols), height(image.rows);
		int numberOfComponents(image.channels());

		//Write the type of the PFM file and ends by a line return
		char type[3];
		type[0] = 'P';
		type[2] = 0x0a;

		if (numberOfComponents == 3)
		{
			type[1] = 'F';
		}
		else if (numberOfComponents == 1)
		{
			type[1] = 'f';
		}

		imageFile << type[0] << type[1] << type[2];

		//Write the width and height and ends by a line return
		imageFile << width << " " << height << type[2];

		//Assumes little endian storage and ends with a line return 0x0a
		//Stores the type
		char byteOrder[10];
		byteOrder[0] = '-'; byteOrder[1] = '1'; byteOrder[2] = '.'; byteOrder[3] = '0';
		byteOrder[4] = '0'; byteOrder[5] = '0'; byteOrder[6] = '0'; byteOrder[7] = '0';
		byteOrder[8] = '0'; byteOrder[9] = 0x0a;

		for (int i = 0; i<10; ++i)
		{
			imageFile << byteOrder[i];
		}

		//Store the floating points RGB color upside down, left to right
		float* buffer = new float[numberOfComponents];

		for (int i = 0; i<height; ++i)
		{
			for (int j = 0; j<width; ++j)
			{
				if (numberOfComponents == 1)
				{
					buffer[0] = image.at<float>(height - 1 - i, j);
				}
				else
				{
					Vec3f color = image.at<Vec3f>(height - 1 - i, j);

					//OpenCV stores as BGR
					buffer[0] = color.val[2];
					buffer[1] = color.val[1];
					buffer[2] = color.val[0];
				}

				//Write the values
				imageFile.write((char *)buffer, numberOfComponents * sizeof(float));

			}
		}

		delete[] buffer;

		imageFile.close();
	}
	else
	{
		cerr << "Could not open the file : " << filePath << endl;
		return false;
	}

	return true;
}

/* 大面积空白 */
void PreProcessing::emptyArea( int row_idx, int col_idx, int width, int height, Mat &output_target, Mat &output_confi) {
	for (int i = row_idx;i < row_idx + height; ++i) {

		float* ptr_output_target = output_target.ptr<float>(i);
		float* ptr_output_confi = output_confi.ptr<float>(i);

		for (int j = col_idx; j < col_idx + width; ++j) {
			ptr_output_target[j] = 0;
			ptr_output_confi[j] = 0;
		}
	}
}
