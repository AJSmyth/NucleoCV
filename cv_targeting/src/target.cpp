#include <stdio.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>
#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "teleop_interfaces/msg/pitch_yaw_motor.hpp"

const float CONF_THRESH = 0.2, NMS_THRESH = 0.2, TGT_THRESH = 0.5, FOV = 72;

using namespace std::chrono;

int main(int argc, char * argv[]) {
	////setup ROS
	rclcpp::init(argc, argv);
	auto node = std::make_shared<rclcpp::Node>("vision");
	auto publisher = node->create_publisher<teleop_interfaces::msg::PitchYawMotor>("pitch_yaw_motor", 10);
	
	teleop_interfaces::msg::PitchYawMotor msg;
	msg.pitch = 0;
	msg.yaw = 0;
	msg.motors = false;
	publisher->publish(msg);
	
	////setup dnn
	//the yolo dnn code is adapated from (source: https://github.com/spmallick/learnopencv/blob/master/ObjectDetection-YOLO/object_detection_yolo.cpp)
	std::vector<std::string> classes {"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
	
	//initialize the net
	cv::dnn::Net net = cv::dnn::readNetFromDarknet("/ROS2/cv_targeting/src/yolov3-tiny.cfg", "/ROS2/cv_targeting/src/yolov3-tiny.weights");
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	
	//get class names 
	std::vector<int> outLayers = net.getUnconnectedOutLayers(); std::vector<std::string> ln = net.getLayerNames();
	ln.resize(outLayers.size());
	for (size_t i = 0; i < outLayers.size(); ++i) ln[i] = ln[outLayers[i] - 1];

	cv::Mat img, blob;
	cv::VideoCapture cap(0);
	cap.set(3, 640);
	cap.set(4, 480);
	cap >> img;

	float pxDegX = FOV / img.cols;
	float pxDegY = FOV / img.rows;
	float degSmall = (180.0f - FOV) / 2.0f; 
	std::vector<cv::Rect> bboxes;
	std::vector<float> scores;
	std::vector<int> classIDs;
	std::vector<int> indices;

	while (rclcpp::ok()) {
		long t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

		//set input and run forward pass
		cv::dnn::blobFromImage(img, blob, 1/255.0, cv::Size(608, 608), cv::Scalar(0,0,0), true, false);
		net.setInput(blob);
		std::vector<cv::Mat> out;
		net.forward(out, ln);	

		double conf;
		cv::Point classIdx;	
		bboxes.clear();
		scores.clear();
		classIDs.clear();
		indices.clear();

		for (int s = 0; s < out.size(); ++s) {
			float* data = (float*)out[s].data;	
			for (int row = 0; row < out[s].rows; ++row, data += out[s].cols) {
				cv::Mat classScores = out[s].row(row).colRange(5, out[s].cols);
				minMaxLoc(classScores, nullptr, &conf, nullptr, &classIdx);
				if (conf > CONF_THRESH) {
					int centerX = (int)(data[0] * img.cols);
					int centerY = (int)(data[1] * img.rows);
					int width = (int)(data[2] * img.cols);
					int height = (int)(data[3] * img.rows);
					int x = centerX - width / 2;
					int y = centerY - height / 2;
			
					bboxes.push_back(cv::Rect(x, y, width, height));
					scores.push_back(conf);
					classIDs.push_back(classIdx.x);
				}
			}
		}

		//Perform NMS
		cv::dnn::NMSBoxes(bboxes, scores, CONF_THRESH, NMS_THRESH, indices);
		long now = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
		float fps = (now - t1 > 0) ? 1000.0f / (now - t1) : 0;

		for (std::vector<int>::iterator it = indices.begin(); it < indices.end(); it++) {
			cv::Rect box = bboxes[*it];
			std::string className = classes[classIDs[*it]];
			if (className == "person") {
				int centerX = box.x + (box.width / 2);
				int offX = std::abs(centerX - (img.cols / 2));
				int halfW = (box.width / 2);
				float deg = 180 - degSmall - (pxDegX * centerX);
				float error = (float)((halfW - offX)) / halfW;

				RCLCPP_INFO(node->get_logger(),"[%f FPS] Tracking person at %f deg.\n", fps, deg);
				if (error >= 0) RCLCPP_INFO(node->get_logger(),"Locking... %f\%\n",  error * 100);
				msg.pitch = (int)(deg);
				publisher->publish(msg);		
				break;
			}
		}
		cap >> img;
		rclcpp::spin_some(node);
	}
}

