#include <openrave/openrave.h>
#include <openrave/plugin.h>
#include <boost/bind.hpp>

#include <HFMD_core/CRForest.h>
#include <HFMD_core/util.h>
#include <HFMD_core/CDataset.h>


#include <stdio.h>
#include <vector>
#include <exception>

#include "DepthSense.hxx"
#include "CCalibDS325.h"

using namespace DepthSense;
using namespace std;
using namespace OpenRAVE;

Context g_context;
DepthNode g_dnode;
ColorNode g_cnode;
AudioNode g_anode;
uint32_t g_aFrames = 0;
uint32_t g_cFrames = 0;
uint32_t g_dFrames = 0;
bool g_bDeviceFound = false;
ProjectionHelper* g_pProjHelper = NULL;
StereoCameraParameters g_scp;

int imageNum = 0;

cv::Mat g_depth,g_color;

CRForest *g_forest;

CCalibDS325 *g_calib;

/*----------------------------------------------------------------------------*/
// New color sample event handler
void onNewColorSample(ColorNode node, ColorNode::NewSampleReceivedData data){
  printf("C#%u: %d\n",g_cFrames,data.colorMap.size());
  memcpy(g_color.data, data.colorMap, data.colorMap.size());

  //    int key = cv::waitKey(10);
  
  RAVELOG_INFO("kokoyade\n");
  CTestDataset seqImg;

  cv::Mat scaledDepth;

  g_calib->calib(g_color, g_depth, g_color, scaledDepth);
  RAVELOG_INFO("kokoyade\n");

  seqImg.img.push_back(&g_color);
  seqImg.img.push_back(&scaledDepth);

  CDetectionResult detectR;

  detectR = g_forest->detection(seqImg);

  // cv::circle(g_color, detectR.detectedClass[0].centerPoint, 10, cv::Scalar(0,0,255),5);

  // std::cout << g_depth.size() << std::endl;

  // if(key == 't'){

  //   stringstream ss_c, ss_d;
  //   ss_c << "color_" << imageNum << ".png";
  //   ss_d << "depth_" << imageNum << ".png";

  //   cv::imwrite(ss_c.str(), g_color);
  //   cv::imwrite(ss_d.str(), g_depth);

  //   imageNum++;
  // }else if(key == 'q'){
  //   g_context.quit();    
  // }

  cv::Mat showDepth;
  //scaledDepth.convertTo(showDepth, CV_8UC1, 255.0 / (MAX_DEPTH));

  //cv::circle(showDepth, cv::Point(320, 240), 5, cv::Scalar(0,0,0), 5);


  cv::imshow("color", g_color);
  //cv::imshow("depth", showDepth);
  

  //key = cv::waitKey(1);

  g_cFrames++;
}

/*----------------------------------------------------------------------------*/
// New depth sample event handler
void onNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data)
{
  printf("Z#%u: %d\n",g_dFrames,data.vertices.size());
  g_dFrames++;

  printf("Z2#: %d\n", g_depth.cols * g_depth.rows * g_depth.channels());
 
  memcpy(g_depth.data, data.depthMap, data.depthMap.size()*2);
}

/*----------------------------------------------------------------------------*/
void configureDepthNode()
{
  g_dnode.newSampleReceivedEvent().connect(&onNewDepthSample);

  DepthNode::Configuration config = g_dnode.getConfiguration();
  config.frameFormat = FRAME_FORMAT_QVGA;
  config.framerate = 30;
  config.mode = DepthNode::CAMERA_MODE_CLOSE_MODE;
  config.saturation = true;

  g_dnode.setEnableVertices(true);
  g_dnode.setEnableDepthMap(true);

  try 
    {
      g_context.requestControl(g_dnode,0);

      g_dnode.setConfiguration(config);
    }
  catch (ArgumentException& e)
    {
      printf("Argument Exception: %s\n",e.what());
    }
  catch (UnauthorizedAccessException& e)
    {
      printf("Unauthorized Access Exception: %s\n",e.what());
    }
  catch (IOException& e)
    {
      printf("IO Exception: %s\n",e.what());
    }
  catch (InvalidOperationException& e)
    {
      printf("Invalid Operation Exception: %s\n",e.what());
    }
  catch (ConfigurationException& e)
    {
      printf("Configuration Exception: %s\n",e.what());
    }
  catch (StreamingException& e)
    {
      printf("Streaming Exception: %s\n",e.what());
    }
  catch (TimeoutException&)
    {
      printf("TimeoutException\n");
    }

}

/*----------------------------------------------------------------------------*/
void configureColorNode()
{
  // connect new color sample handler
  g_cnode.newSampleReceivedEvent().connect(&onNewColorSample);

  ColorNode::Configuration config = g_cnode.getConfiguration();
  //config.frameFormat = FRAME_FORMAT_WXGA_H;
  config.frameFormat = FRAME_FORMAT_VGA;
  config.compression = COMPRESSION_TYPE_MJPEG;
  config.powerLineFrequency = POWER_LINE_FREQUENCY_50HZ;
  config.framerate = 30;

  g_cnode.setEnableColorMap(true);

  try 
    {
      g_context.requestControl(g_cnode,0);

      g_cnode.setConfiguration(config);
    }
  catch (ArgumentException& e)
    {
      printf("Argument Exception: %s\n",e.what());
    }
  catch (UnauthorizedAccessException& e)
    {
      printf("Unauthorized Access Exception: %s\n",e.what());
    }
  catch (IOException& e)
    {
      printf("IO Exception: %s\n",e.what());
    }
  catch (InvalidOperationException& e)
    {
      printf("Invalid Operation Exception: %s\n",e.what());
    }
  catch (ConfigurationException& e)
    {
      printf("Configuration Exception: %s\n",e.what());
    }
  catch (StreamingException& e)
    {
      printf("Streaming Exception: %s\n",e.what());
    }
  catch (TimeoutException&)
    {
      printf("TimeoutException\n");
    }
}

/*----------------------------------------------------------------------------*/
void configureNode(Node node)
{
  if ((node.is<DepthNode>())&&(!g_dnode.isSet()))
    {
      g_dnode = node.as<DepthNode>();
      configureDepthNode();
      g_context.registerNode(node);
    }

  if ((node.is<ColorNode>())&&(!g_cnode.isSet()))
    {
      g_cnode = node.as<ColorNode>();
      configureColorNode();
      g_context.registerNode(node);
    }

  cv::namedWindow("color");
  cv::namedWindow("depth");

  cv::namedWindow("vote");

  //  cv::namedWindow("test");

}

/*----------------------------------------------------------------------------*/
void onNodeConnected(Device device, Device::NodeAddedData data)
{
  configureNode(data.node);
}

/*----------------------------------------------------------------------------*/
void onNodeDisconnected(Device device, Device::NodeRemovedData data)
{
  // if (data.node.is<AudioNode>() && (data.node.as<AudioNode>() == g_anode))
  //     g_anode.unset();
  if (data.node.is<ColorNode>() && (data.node.as<ColorNode>() == g_cnode))
    g_cnode.unset();
  if (data.node.is<DepthNode>() && (data.node.as<DepthNode>() == g_dnode))
    g_dnode.unset();
  printf("Node disconnected\n");
}

/*----------------------------------------------------------------------------*/
void onDeviceConnected(Context context, Context::DeviceAddedData data)
{
  if (!g_bDeviceFound)
    {
      data.device.nodeAddedEvent().connect(&onNodeConnected);
      data.device.nodeRemovedEvent().connect(&onNodeDisconnected);
      g_bDeviceFound = true;
    }

  cv::destroyAllWindows();
}

/*----------------------------------------------------------------------------*/
void onDeviceDisconnected(Context context, Context::DeviceRemovedData data)
{
  g_bDeviceFound = false;
  printf("Device disconnected\n");
}

/*----------------------------------------------------------------------------*/
namespace hfmdraveplugin {

  class CHFMDRaveModule : public ModuleBase
  {
  public:
    CHFMDRaveModule(EnvironmentBasePtr penv) : ModuleBase(penv)
    {

      std::system("echo $PWD");
      RAVELOG_INFO("softkinetic");

      __description = "hough forest multi object detection for openrave";
      RegisterCommand("numbodies",boost::bind(&CHFMDRaveModule::NumBodies,this,_1,_2),"returns bodies");
      RegisterCommand("load",boost::bind(&CHFMDRaveModule::Load,this,_1,_2),"softkinetic");
    }
    virtual ~CHFMDRaveModule() {
    }

    void Destroy() {RAVELOG_INFO("module unloaded from environment\n");}

    int main(const string& cmd)
    {
      RAVELOG_INFO("module initialized cmd; %s\n", cmd.c_str());
      return 0;
    }

    bool NumBodies(ostream& sout, istream& sinput)
    {
      vector<KinBodyPtr> vbodies;
      GetEnv()->GetBodies(vbodies);
      sout << vbodies.size();     // publish the results
      return true;
    }

    bool Load(ostream& sout, istream& sinput)
    {
      RAVELOG_INFO("softkinetic");
      RAVELOG_INFO("testes\n");
      CConfig		conf;	 // setting
      std::vector<CDataset> dataSet; // training data name list and grand truth

      //read argument
      //check argument
      //conf.loadConfig("/home/masahiko/HFMD/config.xml");

      conf.loadConfig("./config.xml");
      conf.demoMode = 1;

      g_forest = NULL;
      g_forest = new CRForest(conf);

      g_forest->loadForest();

      g_calib = new CCalibDS325;

      //g_calib->loadParameters("/home/masahiko/HFMD/intrinsics.yml", "/home/masahiko/HFMD/extrinsics.yml");
      g_calib->loadParameters("./intrinsics.yml", "./extrinsics.yml");

      g_context = Context::create("localhost");
      g_context.deviceAddedEvent().connect(&onDeviceConnected);
      g_context.deviceRemovedEvent().connect(&onDeviceDisconnected);
      // Get the list of currently connected devices
      vector<Device> da = g_context.getDevices();

      g_color = cv::Mat(480, 640, CV_8UC3);
      g_depth = cv::Mat(240, 320, CV_16UC1);
  
      // We are only interested in the first device
      if (da.size() >= 1)
	{
	  g_bDeviceFound = true;
	  da[0].nodeAddedEvent().connect(&onNodeConnected);
	  da[0].nodeRemovedEvent().connect(&onNodeDisconnected);
	  vector<Node> na = da[0].getNodes();
	  printf("Found %u nodes\n",na.size());
	  for (int n = 0; n < (int)na.size();n++){configureNode(na[n]);}
	}
      g_context.startNodes();
      g_context.run();
      g_context.stopNodes();
      if (g_cnode.isSet()) g_context.unregisterNode(g_cnode);
      if (g_dnode.isSet()) g_context.unregisterNode(g_dnode);
      if (g_anode.isSet()) g_context.unregisterNode(g_anode);
      if (g_pProjHelper)
        delete g_pProjHelper;

      if (g_forest)
	delete g_forest;

      if (g_calib)
	delete g_calib;

      return true;
    }

  };

} // end namespace cppexamp

InterfaceBasePtr CreateInterfaceValidated(InterfaceType type, const std::string& interfacename, std::istream& sinput, EnvironmentBasePtr penv)
{
  if( type == PT_Module && interfacename == "hfmdravemodule" ) {
    return InterfaceBasePtr(new hfmdraveplugin::CHFMDRaveModule(penv));
  }
  return InterfaceBasePtr();
}

void GetPluginAttributesValidated(PLUGININFO& info)
{
  info.interfacenames[PT_Module].push_back("HFMDRaveModule");
}

OPENRAVE_PLUGIN_API void DestroyPlugin()
{
  RAVELOG_INFO("destroying plugin\n");
}
