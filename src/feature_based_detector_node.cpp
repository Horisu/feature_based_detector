#include <ros/ros.h>
#include <geometry_msgs/PolygonStamped.h>
#include <sensor_msgs/Image.h>

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <iomanip>

#define MY_DETECTOR cv::AKAZE

// feature extractor
cv::Ptr<MY_DETECTOR> detector_;
cv::Ptr<cv::DescriptorMatcher> matcher_;
cv::Mat ref_desc_;

// flags
bool active_;
bool reset_;

// object info
std::vector<cv::KeyPoint> ref_kp_;
std::vector<cv::Point2f> bb_;

// ros
ros::Publisher pub_;

// parameters
const double akaze_thresh = 1e-4; // AKAZE detection threshold set to locate about 1000 keypoints
const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
const int bb_min_inliers = 10; // Minimal number of inliers to draw bounding box
const int stats_update_period = 10; // On-screen statistics are updated every 10 frames

std::vector<cv::Point2f> Points(std::vector<cv::KeyPoint> keypoints)
{
  std::vector<cv::Point2f> res;
  for(unsigned i = 0; i < keypoints.size(); i++) {
    res.push_back(keypoints[i].pt);
  }
  return res;
}

void bbCallback(const geometry_msgs::PolygonStamped::ConstPtr &_msg){
  ROS_INFO("reset bb start");
  // reset the bounding box
  active_ = false;
  reset_ = false;

  bb_.reserve(4);
  bb_.clear();
  bb_.push_back(cv::Point2f(static_cast<float>(_msg->polygon.points[0].x), static_cast<float>(_msg->polygon.points[0].y)));
  bb_.push_back(cv::Point2f(static_cast<float>(_msg->polygon.points[1].x), static_cast<float>(_msg->polygon.points[0].y)));
  bb_.push_back(cv::Point2f(static_cast<float>(_msg->polygon.points[1].x), static_cast<float>(_msg->polygon.points[1].y)));
  bb_.push_back(cv::Point2f(static_cast<float>(_msg->polygon.points[0].x), static_cast<float>(_msg->polygon.points[1].y)));
  reset_ = true;
  active_ = true;
  ROS_INFO("bb x: %f to %f  y: %f to %f", bb_[0].x, bb_[2].x, bb_[0].y, bb_[2].y);
  ROS_INFO("reset bb finished");
};

void stopCallback(){
  active_ = false;
};

void resetKeyPoints(cv::Mat &_image){
  ref_kp_.clear();

  cv::Point *ptMask = new cv::Point[bb_.size()];
  const cv::Point* ptContain = { &ptMask[0] };
  int iSize = static_cast<int>(bb_.size());
  for (size_t i=0; i<bb_.size(); i++) {
    ptMask[i].x = static_cast<int>(bb_[i].x);
    ptMask[i].y = static_cast<int>(bb_[i].y);
  }
  cv::Mat matMask = cv::Mat::zeros(_image.size(), CV_8UC1);
  cv::fillPoly(matMask, &ptContain, &iSize, 1, cv::Scalar::all(255));
  detector_->detectAndCompute(_image, matMask, ref_kp_, ref_desc_);
  delete[] ptMask;
};

void process(cv::Mat &_image) {
  std::vector<cv::KeyPoint> kp;
  cv::Mat desc;
  detector_->detectAndCompute(_image, cv::noArray(), kp, desc);
  cv::Mat result = _image.clone();

  std::vector< std::vector<cv::DMatch> > matches;
  std::vector<cv::KeyPoint> matched1, matched2;
  matcher_->knnMatch(ref_desc_, desc, matches, 2);
  for(unsigned i = 0; i < matches.size(); i++) {
    if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
      matched1.push_back(ref_kp_[matches[i][0].queryIdx]);
      matched2.push_back(kp[matches[i][0].trainIdx]);
    }
  }

  cv::Mat inlier_mask, homography;
  std::vector<cv::KeyPoint> inliers1, inliers2;
  std::vector<cv::DMatch> inlier_matches;
  if(matched1.size() >= 4) {
    homography = findHomography(Points(matched1), Points(matched2),
                                cv::RANSAC, ransac_thresh, inlier_mask);
  }
  if(matched1.size() < 4 || homography.empty()) {
    return;
  }
  for(unsigned i = 0; i < matched1.size(); i++) {
    if(inlier_mask.at<uchar>(i)) {
      int new_i = static_cast<int>(inliers1.size());
      inliers1.push_back(matched1[i]);
      inliers2.push_back(matched2[i]);
      inlier_matches.push_back(cv::DMatch(new_i, new_i, 0));
    }
  }

  std::vector<cv::Point2f> new_bb;
  perspectiveTransform(bb_, new_bb, homography);
  if((int)inliers1.size() >= bb_min_inliers) {
    // object found
    ROS_INFO("found bb x: %f to %f  y: %f to %f", new_bb[0].x, new_bb[2].x, new_bb[0].y, new_bb[2].y);
    cv::rectangle(result, new_bb[0], new_bb[2], cv::Scalar(0, 0, 255));
  }

  return;
      
};

void imageCallback(const sensor_msgs::Image::ConstPtr& _image_msg){
  if (!active_) return;

  // convert ros_msg -> cv::Mat
  cv::Mat image = cv::Mat(_image_msg->height, _image_msg->width, CV_8UC3);
  int count = 0;
  for (int i = 0; i < _image_msg->height; ++i) {
    for (int j = 0; j < _image_msg->width; ++j) {
      image.at<cv::Vec3b>(i,j)[0] = _image_msg->data[count + 2];//b
      image.at<cv::Vec3b>(i,j)[1] = _image_msg->data[count + 1];//g
      image.at<cv::Vec3b>(i,j)[2] = _image_msg->data[count];//r
      count += 3;
    }
  }
  if (reset_) {
    ROS_INFO("reset key points");
    // reset the object's KeyPoints
    resetKeyPoints(image);
    reset_ = false;
  }
  
  process(image);
};

int main(int argc, char **argv){
  ros::init(argc, argv, "feature_based_detector");
  ros::NodeHandle nh;
  pub_ = nh.advertise<geometry_msgs::PolygonStamped>("/object/detected", 100);

  detector_ = MY_DETECTOR::create();
  matcher_ = cv::DescriptorMatcher::create("BruteForce-Hamming");

  active_ = false;
  reset_ = false;

  ros::Subscriber  bb_sub_ = nh.subscribe("input_bb", 3, bbCallback);
  ros::Subscriber image_sub_ = nh.subscribe("input", 3, imageCallback);

  ros::spin();
  return 0;
}
