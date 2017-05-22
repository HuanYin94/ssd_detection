#include "ros/ros.h"
#include "ros/console.h"

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"

#include <cv_bridge/cv_bridge.h>  // needed in ros sensor -> cv::Mat
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>

#include <boost/bind.hpp>

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>
#include <map>

//added
#include <boost/shared_ptr.hpp>
#include "pointmatcher_ros/get_params_from_server.h" //getParam

//libPM
#include "pointmatcher_ros/point_cloud.h"
#include "pointmatcher/PointMatcher.h"

typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;
using namespace std;

int main(int argc, char **argv)
{
    cv::Mat cameraMat;
    cv::Mat distCoeffsMat;
    cv::Mat rotaionVector;
    cv::Mat translationMat;

    ros::init(argc, argv, "projection");
    ros::NodeHandle n;

    string picName = getParam<string>("picName", ".");
    string vtkName = getParam<string>("vtkName", ".");
    string cameraFileName = getParam<string>("cameraFileName", ".");

    cv::Mat rotationMat;
    cv::FileStorage fs(cameraFileName.c_str(), cv::FileStorage::READ);
    fs["projection_matrix"] >> cameraMat;  //Not the camera matrix
  //  fs["distortion_coefficients"] >> distCoeffsMat;
    fs["calib_rotation_matrix"] >> rotationMat;
    fs["calib_translation_vector"] >> translationMat;

    //After Rectified
    distCoeffsMat = cv::Mat::zeros(1, 5, CV_32FC1);
    cameraMat = cameraMat(cv::Rect(0,0,3,3));

    cv::Rodrigues(rotationMat, rotaionVector);

    cv::Mat image = cv::imread(picName);

    DP cloud = DP::load(vtkName);

    int numLaserPoints = cloud.features.cols();


    cv::Mat laserXYZ(numLaserPoints, 3, CV_64F);

    for(int i = 0; i < numLaserPoints; i++)
    {
        laserXYZ.at<double>(i, 0) = cloud.features(0, i);
        laserXYZ.at<double>(i, 1) = cloud.features(1, i);
        laserXYZ.at<double>(i, 2) = cloud.features(2, i);
    }

    cv::Mat imageUV;

    cv::projectPoints(laserXYZ,
                      rotaionVector,
                      translationMat,
                      cameraMat,
                      distCoeffsMat,
                      imageUV);

    for(int i = 0; i < numLaserPoints; i++)
    {

        //filter double projected
        if(cloud.features(0, i) < 0)
            continue;

        // to draw
        CvPoint point;
        point.x = int(imageUV.at<double>(i, 0));
        point.y = int(imageUV.at<double>(i, 1));

        cv::circle(image, point, 1.50, cv::Scalar(0, 255, 0));

    }

    cv::imshow("imageShow", image);
    cv::waitKey(5);

    cv::imwrite("/home/yh/projectedImage.png", image);

    ros::Publisher cloudPub;
    cloudPub = n.advertise<sensor_msgs::PointCloud2>("Clouds", 2, true);

    ros::Rate r(1);
    while(ros::ok())
    {

        cloudPub.publish(PointMatcher_ros::pointMatcherCloudToRosMsg<float>(cloud, "velodyne", ros::Time::now()));
        r.sleep();
    }

    return 0;
}
