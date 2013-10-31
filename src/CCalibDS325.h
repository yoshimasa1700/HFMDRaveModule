#include <opencv2/opencv.hpp>
#include <string>

#define MAX_DEPTH 1000
#define MIN_DEPTH 0

class CCalibDS325{
 public:
  CCalibDS325(){
/* cv::namedWindow("a");
   cv::namedWindow("b");*/
}
  ~CCalibDS325(){
//cv::destroyAllWindows();
}

int loadParameters(std::string inName, std::string exName);
int calib(cv::Mat &colorSrc, cv::Mat &depthSrc, cv::Mat &colorDest, cv::Mat &depthDest);

private:
cv::Mat cameraMatrix[2], distCoeffs[2], 
  R, T, R1, R2, P1, P2, Q, F, 
  rmap[2][2];
  
cv::Rect validRoi[2];

cv::Mat canvas;
double sf;
int w, h;
};
