#include "CCalibDS325.h"

int CCalibDS325::loadParameters(std::string inName, std::string exName){
  // save intrinsic parameters
  cv::FileStorage fs(inName.c_str() , CV_STORAGE_READ);
  if( fs.isOpened() )
    {
      //fs << "M1" << cameraMatrixS[0] << "D1" << distCoeffsS[0] <<
      //	  "M2" << cameraMatrixS[1] << "D2" << distCoeffsS[1];
      fs["M1"] >> cameraMatrix[0];
      fs["D1"] >> distCoeffs[0];
      fs["M2"] >> cameraMatrix[1];
      fs["D2"] >> distCoeffs[1];
      
      fs.release();
    }
  else
    {
      std::cout << "Error: can not save the intrinsic parameters\n";
      return -1;
    }

  // stereoRectify(cameraMatrixS[0], distCoeffsS[0],
  //               cameraMatrixS[1], distCoeffsS[1],
  //               rgb[0].size(), R, T, R1, R2, P1, P2, Q,
  //               cv::CALIB_ZERO_DISPARITY, 1, rgb[0].size(), &validRoi[0], &validRoi[1]);

  fs.open(exName.c_str() , CV_STORAGE_READ);
  if( fs.isOpened() )
    {
      //fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
      fs["R"] >> R;
      fs["T"] >> T;
      fs["R1"] >> R1;
      fs["R2"] >> R2;
      fs["P1"] >> P1;
      fs["P2"] >> P2;
      fs["Q"] >> Q;
      fs.release();
    }
  else
    {
      std::cout << "Error: can not save the intrinsic parameters\n";
      return -1;
    }
  return 0;
}

int CCalibDS325::calib(cv::Mat &colorSrc, cv::Mat &depthSrc, cv::Mat &colorDest, cv::Mat &depthDest){
  
  std::cout << depthSrc.size() << std::endl;
  cv::Mat scaledDepth(640, 480, CV_16UC1);
  cv::Mat maxDist = cv::Mat::ones(depthSrc.rows * 2, depthSrc.cols * 2, CV_16U) * MAX_DEPTH;
  cv::Mat minDist = cv::Mat::ones(depthSrc.rows * 2, depthSrc.cols * 2, CV_16U) * MIN_DEPTH;

  cv::resize(depthSrc, scaledDepth, cv::Size(), 2.0, 2.0);
  cv::min(scaledDepth, maxDist, scaledDepth);

  scaledDepth -= minDist;
  //scaledDepth.convertTo(scaledDepth, CV_16UC1, 255.0 / (MAX_DEPTH - MIN_DEPTH));
 
  //colorSrc.copyTo(colorDest);
  //scaledDepth.copyTo(depthDest);
  // std::cout << "koreka" << std::endl;

  // cv::imshow("a", colorDest);
  // cv::imshow("b", depthDest);

  // cv::waitKey(0);

  // return 0;
  

  

    

  // cv::resize(scaledDepth, scaledDepth, cv::Size(), 2.0,2.0);

  cv::initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, colorSrc.size(), CV_16SC2, rmap[0][0], rmap[0][1]);
  cv::initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R1, P2, colorSrc.size(), CV_16SC2, rmap[1][0], rmap[1][1]);

  sf = 600./MAX(colorSrc.size().width, colorSrc.size().height);
  w = cvRound(colorSrc.size().width*sf);
  h = cvRound(colorSrc.size().height*sf);
  canvas.create(h, w*2, CV_8UC3);

  {
    cv::Mat img = colorSrc.clone(), rimg, cimg;
    //cv::remap(img, rimg, rmap[k][0], rmap[k][1], CV_INTER_LINEAR);
    undistort(img, cimg, cameraMatrix[0], distCoeffs[0]);
    cimg.copyTo(colorDest);

    //cv::cvtColor(cimg, cimg, CV_GRAY2BGR);
    //cv::Rect roi(0,0,w,h);
    //std::cout << roi << std::endl;

    cv::Mat canvasPart = canvas(cv::Rect(0, 0, w,h));
    cv::resize(cimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);
    cv::Rect vroi(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
  		  cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));
    cv::rectangle(canvasPart, vroi, cv::Scalar(0,0,255), 3, 8);

  }
  {
    cv::Mat img = scaledDepth.clone(), rimg, cimg;
    //cv::remap(img, img, rmap[k][0], rmap[k][1], CV_INTER_LINEAR);
    undistort(img, rimg, cameraMatrix[1], distCoeffs[1]);
    //rimg.copyTo(depthDest);
    
    

    cvtColor(rimg, cimg, CV_GRAY2BGR);

    std::cout << "kokoke?" << std::endl;
    cv::Mat depthPart = cimg(cv::Rect(40, 43,498,498 / 4 * 3));
    cv::Mat rescale;
    cv::resize(depthPart, rescale, cv::Size(640, 480));

    rescale.copyTo(depthDest);
    cvtColor(depthDest, depthDest,CV_RGB2GRAY,1);
    //    depthDest.convertTo(depthDest,1.0,CV_16UC1);
    std::cout << depthDest.type() << " " << CV_64FC1	<< std::endl;
   

    // cv::Mat canvasPart = canvas(cv::Rect(w * 1, 0, w, h));
    // cv::resize(depthPart, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);

    // cv::Rect vroi(cvRound(validRoi[1].x*sf), cvRound(validRoi[1].y*sf),
    // 		  cvRound(validRoi[1].width*sf), cvRound(validRoi[1].height*sf));
    // cv::rectangle(canvasPart, vroi, cv::Scalar(0,0,255), 3, 8);
	     
  }


  return 0;
}
