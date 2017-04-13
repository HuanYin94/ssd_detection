/*
 * How to build caffe with strong& robust cmake?
 * I want to use libcaffe.so outside caffe.dir
 *
 * KENG!
 *
 * So, some notices & problems:
 * 1. MKL or ATLAS.
 *    Intel MKL is faster, but ATLAS is apt-get install.
 *
 * 2. To use libcaffe.so, cmake_build is needed.
 *
 * 3. write_json is noted, ortherwise upgrate g++.
 *    so, detection_output is disabled?
 *
 * 4. locate GPU / CPU-only carefully.
 *
 * 5. batch_size problem exists:  error == cudaSuccess (2 vs. 0)  out of memory??
 *    Use the 600 * 180 instead
 *
 * 6. cv_bridge -> opencv2/3?
 *    fuck opencv !
 *
 * 7. finally change the opencv of caffe's cmakeLists
 *    :(
 *    3.0 -> 2.4.7
 *
 * 8. C++ getInstance() used
 *
 *    //important
 *    Detector* Detector::m_pInstance;
 *
 * 9. memcpy(cameraPara.data(), cameraMatrix.data, 3*3*sizeof(double));
 *    from eigen to cv , or orther
 *
 * 10. finally use cv::Mat all the time, fuck!
 *
 * 11. There is sth. wrong on the wiki of cv::projectPoints, tthe input must be N*3 not 3*N !!!
 *
 * 12. Section 2 is not syncronized, Section3 for it
 *
 * 13. approximate_time is use, not the exact time (time_synchronizer)
 *     and the function cannot be used in Class CaffeNet?!
 *
 * 14. Stop
 *
 * 15. Thx to Tang Li and Li Dongxuan's help
 *
 * --by Yin Huan in ZJU
 * */
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

#define detcThreshold 0.6
#define sectionNum 20   //1-20: Cars    21-40： Cyclists     41-60: Pedistrains
#define carLength 3.0
#define cycLength 2.0
#define pedLength 1.0

using namespace std;
using namespace caffe;

typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.3,
    "Only store detections with score higher than the threshold.");

class Detector
{
public:


    std::vector<vector<float> > Detect(const cv::Mat& img);

    //YH
    static Detector * GetInstance()
    {
        return m_pInstance;
    }

    //YH
    static Detector * initialized(const string& model_file,
                       const string& weights_file,
                       const string& mean_file,
                       const string& mean_value)
    {
        if (m_pInstance == NULL)
            m_pInstance = new Detector(model_file,
                                   weights_file,
                                   mean_file,
                                   mean_value);

        return m_pInstance;
    }

private:

    Detector(const string& model_file,
           const string& weights_file,
           const string& mean_file,
           const string& mean_value);

    //dan li, YH
    static Detector *m_pInstance;

    void SetMean(const string& mean_file, const string& mean_value);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);


private:
    boost::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
};

Detector* Detector::m_pInstance;

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value)
{
//#ifdef CPU_ONLY
//    Caffe::set_mode(Caffe::CPU);
//#else
    Caffe::set_mode(Caffe::GPU);
//#endif

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img)
{
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* result_blob = net_->output_blobs()[0];
    const float* result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<float> > detections;
    for (int k = 0; k < num_det; ++k)
    {
        if (result[0] == -1)
        {
            // Skip invalid detection.
            result += 7;
            continue;
        }
        vector<float> detection(result, result + 7);
        detections.push_back(detection);
        result += 7;
    }
    return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value)
{
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}




///CAMERA CLASS : OUT THE DETECTOR
#if 0
///SECTION 1
///kitti benchmark detection, frame by frame, vision only
class CaffeNet {
public:
    CaffeNet(ros::NodeHandle &n);
private:
    ros::NodeHandle& n;
    const string deployFileName;
    const string caffeModelFileName;
    const string picturesFileName;
    ros::Subscriber imageSub;

    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
protected:

};

CaffeNet::CaffeNet(ros::NodeHandle& n):
  n(n),
  deployFileName(getParam<string>("deployFileName", ".")),
  caffeModelFileName(getParam<string>("caffeModelFileName", ".")),
  picturesFileName(getParam<string>("picturesFileName", "."))
{

    FLAGS_alsologtostderr = 1;

    const string& model_file = deployFileName;
    const string& weights_file = caffeModelFileName;
    const string& mean_file = FLAGS_mean_file;
    const string& mean_value = FLAGS_mean_value;
    const string& file_type = FLAGS_file_type;
    const string& out_file = FLAGS_out_file;
    const float confidence_threshold = FLAGS_confidence_threshold;

    // Initialize the network.
    Detector detector(model_file, weights_file, mean_file, mean_value);

    // Set the output mode.
    std::streambuf* buf = std::cout.rdbuf();
    std::ofstream outfile;
    if (!out_file.empty()) {
      outfile.open(out_file.c_str());
      if (outfile.good()) {
        buf = outfile.rdbuf();
      }
    }
    std::ostream out(buf);

    // Process image one by one.
    std::ifstream infile(picturesFileName);
    std::string file;

    //window set up
    cv::namedWindow("imageShow");

    while (infile >> file)
    {
        if (file_type == "image")
        {
          cv::Mat img = cv::imread(file, -1);
          CHECK(!img.empty()) << "Unable to decode image " << file;
          std::vector<vector<float> > detections = detector.Detect(img);

          /* Print the detection results. */
          for (int i = 0; i < detections.size(); ++i)
          {
            const vector<float>& d = detections[i];
            // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
            CHECK_EQ(d.size(), 7);
            const float score = d[2];
            if (score >= confidence_threshold)
            {
              out << file << " ";
              out << static_cast<int>(d[1]) << " ";
              out << score << " ";
              out << static_cast<int>(d[3] * img.cols) << " ";
              out << static_cast<int>(d[4] * img.rows) << " ";
              out << static_cast<int>(d[5] * img.cols) << " ";
              out << static_cast<int>(d[6] * img.rows) << std::endl;



              //draw the rectangele
              int lx = d[3] * img.cols;
              int ly = d[4] * img.rows;
              int rx = d[5] * img.cols;
              int ry = d[6] * img.rows;
              cv::rectangle( img, cvPoint(lx, ly), cvPoint(rx, ry), cvScalar(0, 0, 255), 2, 4, 0 );

            }
          }


          cv::imshow("imageShow", img);
          cv::waitKey(1000);
        }
    }


}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ssd_detection");
    ros::NodeHandle n;
    CaffeNet CaffeNet(n);

    ros::spin();

    return 0;
}

#endif

#if 1
///SECTION 3
///USING SYNC, DISABLE CLASS
///
///

//No Class, Set Outside
cv::Mat cameraMat;
cv::Mat distCoeffsMat;
cv::Mat rotaionVector;
cv::Mat translationMat;
ros::Publisher stampCloudPub;


void imageLaserCallback(const sensor_msgs::ImageConstPtr &img, const sensor_msgs::PointCloud2ConstPtr &pcl)
{

    double t1 = ros::Time::now().toSec();

  // Solve all of perception here...

    vector<vector<float>> detectionResults;

    cv::namedWindow("imageShow");

    cv_bridge::CvImageConstPtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(img, "bgr8");

    cv::Mat image = cv_ptr->image;

//    cout<<image.cols<<"  "<<image.rows<<endl;

    CHECK(!image.empty()) << "Unable to decode image!!! ";
    Detector inDetector = *Detector::GetInstance();
    std::vector<vector<float> > detections = inDetector.Detect(image);

    /* Print the detection results. */
    for (int i = 0; i < detections.size(); ++i)
    {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        const float score = d[2];
        if (score >= detcThreshold)
        {

          //draw the rectangele
          int lx = d[3] * image.cols;
          int ly = d[4] * image.rows;
          int rx = d[5] * image.cols;
          int ry = d[6] * image.rows;

          //B-G-R
          if(d[1] == 1) //car red
            cv::rectangle( image, cvPoint(lx, ly), cvPoint(rx, ry), cvScalar(0, 0, 255), 3, 4, 0 );
          else if(d[1] == 2) //Cyclist blue
            cv::rectangle( image, cvPoint(lx, ly), cvPoint(rx, ry), cvScalar(255, 0, 0), 3, 4, 0 );
          else if(d[1] == 3) //Pedestrian green
            cv::rectangle( image, cvPoint(lx, ly), cvPoint(rx, ry), cvScalar(0, 255, 0), 3, 4, 0 );
          else if(d[1] == 4) //Truck
            cv::rectangle( image, cvPoint(lx, ly), cvPoint(rx, ry), cvScalar(65, 255, 255), 3, 4, 0 );
          else if(d[1] == 5) //Van
            cv::rectangle( image, cvPoint(lx, ly), cvPoint(rx, ry), cvScalar(255, 65, 255), 3, 4, 0 );

          //push into results vector
          detectionResults.push_back(detections[i]);

        }
    }

    ///LASER
    DP cloud = PointMatcher_ros::rosMsgToPointMatcherCloud<float>(*pcl);

    //Publish to Lv
    DP obstacleCloud(cloud.createSimilarEmpty());
    int oCcount = 0;

    //record sth
    vector<int> stampedOstacle;
    stampedOstacle.push_back(-1);

    //filter the background point clouds
    map<int, float> obstacleNearestDis;

    //count no.
    int car = 0;
    int pedestrain = 0;
    int cyclist = 0;

    int numLaserPoints = cloud.features.cols();

    cloud.addDescriptor("stamp_detection", PM::Matrix::Zero(1, numLaserPoints));
    //also needed in the new point cloud
    obstacleCloud.addDescriptor("stamp_detection", PM::Matrix::Zero(1, numLaserPoints));

    int stampRow = cloud.getDescriptorStartingRow("stamp_detection");

    for(int i = 0; i < numLaserPoints; i++)
    {
        Eigen::Vector3f inputXYZ = cloud.features.col(i).head(3);

        // filter the points behind the robot, x < 0
        if(inputXYZ(0) < 0)
            continue;

        cv::Mat laserXYZ(1, 3, CV_64F);//DataType<float>::type);
        laserXYZ.at<double>(0,0) = inputXYZ(0);
        laserXYZ.at<double>(0,1) = inputXYZ(1);
        laserXYZ.at<double>(0,2) = inputXYZ(2);

        cv::Mat imageUV;

        cv::projectPoints(laserXYZ,
                          rotaionVector,
                          translationMat,
                          cameraMat,
                          distCoeffsMat,
                          imageUV);

        for(int j = 0; j < detectionResults.size(); j++)
        {
            const vector<float>& d = detectionResults.at(j);

            int lx = d[3] * image.cols;
            int ly = d[4] * image.rows;
            int rx = d[5] * image.cols;
            int ry = d[6] * image.rows;

            if(imageUV.at<double>(0,0) > lx && imageUV.at<double>(0,0) < rx &&
               imageUV.at<double>(0,1) > ly && imageUV.at<double>(0,1) < ry)
            {
                vector<int>::iterator it;

                //find out if a new obstacle is coming
                it = find(stampedOstacle.begin(),
                             stampedOstacle.end(),
                             j);

                //No. of Obstacle
                int obstacleNo;

                if(it != stampedOstacle.end())
                {
                    //has
                    //car || truck || van
                    if(d[1] == 1 || d[1] == 4 || d[1] == 5)
                         obstacleNo = car + sectionNum*0;   // :), to show
                    //Cyclist
                    else if(d[1] == 2)
                         obstacleNo = cyclist + sectionNum*1;
                    //Pedestrain
                    else if(d[1] == 3)
                         obstacleNo = pedestrain + sectionNum*2;

                    //set the nearest range distance
                    if(obstacleNearestDis[obstacleNo] > inputXYZ.norm())
                        obstacleNearestDis[obstacleNo] = inputXYZ.norm();

                }
                else
                {
                    //not found
                    stampedOstacle.push_back(j);

                    //car || truck || van
                    if(d[1] == 1 || d[1] == 4 || d[1] == 5)
                    {
                        car++;
                        obstacleNo = car + sectionNum*0;   // :), to show
                    }
                    //Cyclist
                    else if(d[1] == 2)
                    {
                        cyclist++;
                        obstacleNo = cyclist + sectionNum*1;
                    }
                    //Pedestrain
                    else if(d[1] == 3)
                    {
                        pedestrain++;
                        obstacleNo = pedestrain + sectionNum*2;
                    }

                    //set the nearest range distance
                    obstacleNearestDis[obstacleNo] = inputXYZ.norm();

                }

                cloud.descriptors(stampRow, i) = obstacleNo;

                obstacleCloud.setColFrom(oCcount, cloud, i);
                oCcount++;
                break;
            }

        }

        CvPoint point;
        point.x = int(imageUV.at<double>(0,0));
        point.y = int(imageUV.at<double>(0,1));

        cv::circle(image, point, 0.5, cv::Scalar(0, 255, 0));

    }

    obstacleCloud.conservativeResize(oCcount);

    DP obstacleFilterCloud(obstacleCloud.createSimilarEmpty());

#if 1
    //filer the background points according to the nearest range
    {
        for(int i = 0; i < obstacleCloud.features.cols(); i++)
        {
            Eigen::Vector3f inputXYZ = obstacleCloud.features.col(i).head(3);

            int obstacleNo = obstacleCloud.descriptors(stampRow, i);
            float length = 99;

            if(obstacleNo >= 1 && obstacleNo <= 20)
                length = carLength;
            else if(obstacleNo >= 21 && obstacleNo <= 40)
                length = cycLength;
            else if(obstacleNo >= 41)
                length = pedLength;

            if(inputXYZ.norm() > (obstacleNearestDis[obstacleNo] + length))
                obstacleCloud.descriptors(stampRow, i) = -99;   // the point needs to be deleted
            else
                continue;
        }

        obstacleFilterCloud.addDescriptor("stamp_detection", PM::Matrix::Zero(1, obstacleCloud.features.cols()));

        int count = 0;

        for(int i = 0; i < obstacleCloud.features.cols(); i++)
        {
            if(obstacleCloud.descriptors(stampRow, i) > 0)
            {
                obstacleFilterCloud.setColFrom(count, obstacleCloud, i);
                count++;
            }
        }

        obstacleFilterCloud.conservativeResize(count);

    }

#endif

    //Publish & Show
    stampCloudPub.publish(PointMatcher_ros::pointMatcherCloudToRosMsg<float>(obstacleFilterCloud, "velodyne", ros::Time::now()));

    cv::imshow("imageShow", image);
    cv::waitKey(5);

    double t2 = ros::Time::now().toSec();

//    cout<<"Cost Time:  "<<t2-t1<<endl;

//    sleep(5*1000);
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "ssd_detection");
  ros::NodeHandle n;

  ///Initializaion Network
  FLAGS_alsologtostderr = 1;

  string deployFileName = getParam<string>("deployFileName", ".");
  string caffeModelFileName = getParam<string>("caffeModelFileName", ".");
  string cameraFileName = getParam<string>("cameraFileName", ".");

  const string& model_file = deployFileName;
  const string& weights_file = caffeModelFileName;
  const string& mean_file = FLAGS_mean_file;
  const string& mean_value = FLAGS_mean_value;

  // Initialize the network.
  Detector *detector = Detector::initialized(model_file,
                                             weights_file,
                                             mean_file,
                                             mean_value);
  ///Params
  //calibration & Reference & projection
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

  ///Publish Cloud
  stampCloudPub = n.advertise<sensor_msgs::PointCloud2>("stamped_pointClouds", 2, true);

  ///Synchronize
  message_filters::Subscriber<sensor_msgs::Image> imageSub(n, "/camera/left/image_raw", 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> laserSub(n, "/velodyne_points", 1);

  //approximate
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), imageSub, laserSub);

  //exact Time only?
//  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::PointCloud2> sync(imageSub, laserSub, 10);

  sync.registerCallback(boost::bind(&imageLaserCallback,  _1, _2));

  ros::spin();
  return 0;
}

#endif
