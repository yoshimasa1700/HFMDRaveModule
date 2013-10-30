#include <openrave/openrave.h>
#include <openrave/plugin.h>
#include <boost/bind.hpp>


#include <stdio.h>
#include <vector>
#include <exception>

#include "DepthSense.hxx"

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

// New audio sample event handler
void onNewAudioSample(AudioNode node, AudioNode::NewSampleReceivedData data)
{
    printf("A#%u: %d\n",g_aFrames,data.audioData.size());
    g_aFrames++;
}
// New color sample event handler
void onNewColorSample(ColorNode node, ColorNode::NewSampleReceivedData data)
{
    printf("C#%u: %d\n",g_cFrames,data.colorMap.size());
    g_cFrames++;
}
// New depth sample event handler
void onNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data)
{
    printf("Z#%u: %d\n",g_dFrames,data.vertices.size());
    // Project some 3D points in the Color Frame
    if (!g_pProjHelper)
    {
        g_pProjHelper = new ProjectionHelper (data.stereoCameraParameters);
        g_scp = data.stereoCameraParameters;
    }
    else if (g_scp != data.stereoCameraParameters)
    {
        g_pProjHelper->setStereoCameraParameters(data.stereoCameraParameters);
        g_scp = data.stereoCameraParameters;
    }
    int32_t w, h;
    FrameFormat_toResolution(data.captureConfiguration.frameFormat,&w,&h);
    int cx = w/2;
    int cy = h/2;

    Vertex p3DPoints[4];

    p3DPoints[0] = data.vertices[(cy-h/4)*w+cx-w/4];
    p3DPoints[1] = data.vertices[(cy-h/4)*w+cx+w/4];
    p3DPoints[2] = data.vertices[(cy+h/4)*w+cx+w/4];
    p3DPoints[3] = data.vertices[(cy+h/4)*w+cx-w/4];
    
    Point2D p2DPoints[4];
    g_pProjHelper->get2DCoordinates ( p3DPoints, p2DPoints, 4, CAMERA_PLANE_COLOR);
    g_dFrames++;
    // Quit the main loop after 200 depth frames received
    if (g_dFrames == 200)
        g_context.quit();
}
/*----------------------------------------------------------------------------*/
void configureAudioNode()
{
    g_anode.newSampleReceivedEvent().connect(&onNewAudioSample);

    AudioNode::Configuration config = g_anode.getConfiguration();
    config.sampleRate = 44100;
    try 
    {
        g_context.requestControl(g_anode,0);
        g_anode.setConfiguration(config);
        g_anode.setInputMixerLevel(0.5f);
    }
    catch (ArgumentException& e)
    {printf("Argument Exception: %s\n",e.what());}
    catch (UnauthorizedAccessException& e)
    {printf("Unauthorized Access Exception: %s\n",e.what());}
    catch (ConfigurationException& e)
    {printf("Configuration Exception: %s\n",e.what());}
    catch (StreamingException& e)
    {printf("Streaming Exception: %s\n",e.what());}
    catch (TimeoutException&)
    {printf("TimeoutException\n");}
}
/*----------------------------------------------------------------------------*/
void configureDepthNode()
{
    g_dnode.newSampleReceivedEvent().connect(&onNewDepthSample);
    DepthNode::Configuration config = g_dnode.getConfiguration();
    config.frameFormat = FRAME_FORMAT_QVGA;
    config.framerate = 25;
    config.mode = DepthNode::CAMERA_MODE_CLOSE_MODE;
    config.saturation = true;
    g_dnode.setEnableVertices(true);
    try 
    {
        g_context.requestControl(g_dnode,0);
        g_dnode.setConfiguration(config);
    }
    catch (ArgumentException& e)
    {printf("Argument Exception: %s\n",e.what());}
    catch (UnauthorizedAccessException& e)
    {printf("Unauthorized Access Exception: %s\n",e.what());}
    catch (IOException& e)
    {printf("IO Exception: %s\n",e.what());}
    catch (InvalidOperationException& e)
    {printf("Invalid Operation Exception: %s\n",e.what());}
    catch (ConfigurationException& e)
    {printf("Configuration Exception: %s\n",e.what());}
    catch (StreamingException& e)
    {printf("Streaming Exception: %s\n",e.what());}
    catch (TimeoutException&)
    {printf("TimeoutException\n");}
}
/*----------------------------------------------------------------------------*/
void configureColorNode()
{
    // connect new color sample handler
    g_cnode.newSampleReceivedEvent().connect(&onNewColorSample);
    ColorNode::Configuration config = g_cnode.getConfiguration();
    config.frameFormat = FRAME_FORMAT_VGA;
    config.compression = COMPRESSION_TYPE_MJPEG;
    config.powerLineFrequency = POWER_LINE_FREQUENCY_50HZ;
    config.framerate = 25;
    g_cnode.setEnableColorMap(true);
    try 
    {
        g_context.requestControl(g_cnode,0);
        g_cnode.setConfiguration(config);
    }
    catch (ArgumentException& e)
    {printf("Argument Exception: %s\n",e.what());}
    catch (UnauthorizedAccessException& e)
    {printf("Unauthorized Access Exception: %s\n",e.what());}
    catch (IOException& e)
    {printf("IO Exception: %s\n",e.what());}
    catch (InvalidOperationException& e)
    {printf("Invalid Operation Exception: %s\n",e.what());}
    catch (ConfigurationException& e)
    {printf("Configuration Exception: %s\n",e.what());}
    catch (StreamingException& e)
    {printf("Streaming Exception: %s\n",e.what());}
    catch (TimeoutException&)
    {printf("TimeoutException\n");}
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
    if ((node.is<AudioNode>())&&(!g_anode.isSet()))
    {
        g_anode = node.as<AudioNode>();
        configureAudioNode();
        g_context.registerNode(node);
    }
}
/*----------------------------------------------------------------------------*/
void onNodeConnected(Device device, Device::NodeAddedData data)
{configureNode(data.node);}
/*----------------------------------------------------------------------------*/
void onNodeDisconnected(Device device, Device::NodeRemovedData data)
{
    if (data.node.is<AudioNode>() && (data.node.as<AudioNode>() == g_anode))
        g_anode.unset();
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
}
/*----------------------------------------------------------------------------*/
void onDeviceDisconnected(Context context, Context::DeviceRemovedData data)
{
    g_bDeviceFound = false;
    printf("Device disconnected\n");
}

/*----------------------------------------------------------------------------*/
namespace cppexamp {

class CDemoModule : public ModuleBase
{
public:
    CDemoModule(EnvironmentBasePtr penv) : ModuleBase(penv)
    {
        __description = "A very simple plugin.";
        RegisterCommand("numbodies",boost::bind(&CDemoModule::NumBodies,this,_1,_2),"returns bodies");
        RegisterCommand("load",boost::bind(&CDemoModule::Load,this,_1,_2),"softkinetic");
    }
    virtual ~CDemoModule() {
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
    g_context = Context::create("localhost");
    g_context.deviceAddedEvent().connect(&onDeviceConnected);
    g_context.deviceRemovedEvent().connect(&onDeviceDisconnected);
    // Get the list of currently connected devices
    vector<Device> da = g_context.getDevices();
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

    return true;
    }

};

} // end namespace cppexamp

InterfaceBasePtr CreateInterfaceValidated(InterfaceType type, const std::string& interfacename, std::istream& sinput, EnvironmentBasePtr penv)
{
    if( type == PT_Module && interfacename == "cdemomodule" ) {
        return InterfaceBasePtr(new cppexamp::CDemoModule(penv));
    }
    return InterfaceBasePtr();
}

void GetPluginAttributesValidated(PLUGININFO& info)
{
    info.interfacenames[PT_Module].push_back("CDemoModule");
}

OPENRAVE_PLUGIN_API void DestroyPlugin()
{
    RAVELOG_INFO("destroying plugin\n");
}
