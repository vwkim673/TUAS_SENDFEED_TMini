/*
TUAS Endfeed 3D (T-Mini) Sensor Processor.

South Endfeed TMini Processing Application.

*/

#define _CRT_SECURE_NO_WARNINGS
#define BOOST_ALL_NO_LIB
#define _USE_MATH_DEFINES
#define WIN32_LEAN_AND_MEAN

#pragma warning(disable: 4819 4996 4477 6258 4100)
#ifndef BOOST_USE_WINDOWS_H
#define BOOST_USE_WINDOWS_H
#endif

#define NOMINMAX
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <fstream>
#include <iterator>
#include <bitset>
#include <chrono>
#include <future>
#include <thread>
#include <filesystem>
#include <direct.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <condition_variable>
#include <atomic>
#include <signal.h>
#include <queue>
#include <ctime>
#include <iomanip>

#ifndef _WINDOW_SOCKET
#define _WINDOW_SOCKET
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>
#include <synchapi.h>
#endif

#include "VisionaryControl.h"
#include "CoLaParameterReader.h"
#include "CoLaParameterWriter.h"
#include "VisionaryTMiniData.h"    // Header specific for the Time of Flight data
#include "VisionaryDataStream.h"
#include "PointXYZ.h"
#include "PointCloudPlyWriter.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/videoio.hpp>

#include "miniINI.h"

#include <onnxruntime_cxx_api.h>

#include <boost/filesystem.hpp>

#ifndef PCL_COMMON_H
#define PCL_COMMON_H
#include <pcl/pcl_base.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>


#include <pcl/console/parse.h>
#include <pcl/common/random.h>
#include <pcl/common/transforms.h>
#include <pcl/common/geometry.h>
#include <pcl/common/intersections.h>
#include <pcl/common/angles.h>
#include <pcl/common/centroid.h>
#include <pcl/common/norms.h>
#include <pcl/common/time.h>
#include <pcl/common/distances.h>

#include <boost/thread.hpp>

// filter include
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

// segmentation include
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

// region growing
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>

//sample consensus for table calibrations
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_line.h>

#endif

#include <open3d/Open3D.h>

#include "GlobalVar.h"
#include "DataStack.h"
//#include "VA.h"

#include "TrtRunner.h"
#include "YOLO11.hpp"
#include "ChassisRoi.h"

#pragma comment(lib, "Dbghelp.lib")
#include <DbgHelp.h>

#define DEFAULT_RECVLEN 50
#define DEFAULT_SENDLEN 50

#define ENDFEED true

// Use the same default thresholds as Ultralytics CLI
const float confThreshold = 0.5f;  // Match Ultralytics default confidence threshold
const float iouThreshold = 0.45f;   // Match Ultralytics default IoU threshold

/*
* Update: 2025.08.19
* Update Log:
*	Added CLPS OK (chassis-container separation detection logic)
*/

const std::string program_version = "1.3";
const std::vector<std::string> class_names = { "hole", "cone", "land", "igh", "ighs", "cstf", "csts", "cstb", "cstsl" };

void logMessage(const std::string& message);

struct bbx
{
	int index = -1;
	std::string label = "";
	float x = 0;
	float y = 0;
	float center_x = 0;
	float center_y = 0;
	float w = 0;
	float h = 0;
	float prob = 0;
};

class DetectionAverager {
public:

	void update(const bbx& input) {
		count_++;

		avg_.x += (input.x - avg_.x) / count_;
		avg_.y += (input.y - avg_.y) / count_;
		avg_.w += (input.w - avg_.w) / count_;
		avg_.h += (input.h - avg_.h) / count_;
		avg_.center_x += (input.center_x - avg_.center_x) / count_;
		avg_.center_y += (input.center_y - avg_.center_y) / count_;
		avg_.prob += (input.prob - avg_.prob) / count_;
	}
	void set() {
		set_ = true;
	}

	bbx get_average() const {
		return avg_;
	}
	size_t get_count() const {
		return count_;
	}

	bool get_set() const {
		return set_;
	}

	void reset()
	{
		count_ = 0;
		avg_ = {};
		set_ = false;
	}

private:
	size_t count_;
	bbx avg_;

	bool set_;
};

class PCAAverager {
public:

	void update(const pcl::PointXYZ& input) {
		++count_;
		float k = 1.0f / static_cast<float>(count_);

		avg_.x += (input.x - avg_.x) * k;
		avg_.y += (input.y - avg_.y) * k;
		avg_.z += (input.z - avg_.z) * k;

	}
	void set() {
		set_ = true;
	}

	pcl::PointXYZ get_average() const {
		return avg_;
	}
	size_t get_count() const {
		return count_;
	}

	bool get_set() const
	{
		return set_;
	}

	void reset()
	{
		count_ = 0;
		avg_ = {};
		set_ = false;
	}

private:
	size_t count_ = 0;
	pcl::PointXYZ avg_{ 0.0f, 0.0f, 0.0f };
	bool set_;
};

//Robust per-axis position latch for the 3D deviation OUTPUT multiplex (mount cone /
//offload hole & cone). Collects up to N=10 valid samples, then MAD-filters outliers
//(drop |x-median| > 3*1.4826*MAD) and freezes the median of the survivors. Once
//locked() the value() never changes until reset(). Invalid (-10000) samples ignored.
class MedianLatch3D {
public:
	void push(const pcl::PointXYZ& p) {
		if (locked_) return;
		//Reject the -10000 sentinel AND non-finite (NaN/inf): point-cloud centroids can be
		//NaN when a region hits invalid depth. A NaN passes "!= -10000" and, once 10 samples
		//collect, makes std::sort in finalize() violate strict-weak-ordering -> the debug STL
		//aborts ("Fatal program exit requested"). Guard here so only clean samples enter.
		if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) return;
		if (p.x == -10000 || p.y == -10000 || p.z == -10000) return;
		xs_.push_back(p.x); ys_.push_back(p.y); zs_.push_back(p.z);
		if (xs_.size() >= N_) finalize();
	}
	bool locked() const { return locked_; }
	pcl::PointXYZ value() const { return val_; }
	void reset() {
		xs_.clear(); ys_.clear(); zs_.clear();
		locked_ = false;
		val_ = pcl::PointXYZ(-10000, -10000, -10000);
	}
private:
	static float median_(std::vector<float> a) {
		if (a.empty()) return 0.0f;
		std::sort(a.begin(), a.end());
		size_t n = a.size();
		return (n % 2) ? a[n / 2] : 0.5f * (a[n / 2 - 1] + a[n / 2]);
	}
	static float madMedian_(const std::vector<float>& v) {
		float m = median_(v);
		std::vector<float> dev; dev.reserve(v.size());
		for (float x : v) dev.push_back(std::abs(x - m));
		float mad = median_(dev);
		if (mad <= 1e-6f) return m;					//no spread -> plain median
		float thr = 3.0f * 1.4826f * mad;			//1.4826 = normal-consistency scale
		std::vector<float> keep;
		for (float x : v) if (std::abs(x - m) <= thr) keep.push_back(x);
		return keep.empty() ? m : median_(keep);
	}
	void finalize() {
		val_.x = madMedian_(xs_);
		val_.y = madMedian_(ys_);
		val_.z = madMedian_(zs_);
		locked_ = true;
	}
	static const size_t N_ = 10;
	std::vector<float> xs_, ys_, zs_;
	bool locked_ = false;
	pcl::PointXYZ val_{ -10000, -10000, -10000 };
};

struct IDEAL_POS
{
	bbx CONE;
	bbx LANDED;
	bbx GUIDE;

	pcl::PointXYZ CONE_PCA;
	pcl::PointXYZ LANDED_PCA;
	pcl::PointXYZ GUIDE_PCA;
};

struct MAX_MIN_RANGE
{
	int MAX;
	int MIN;
};

struct PRESET_POS
{
	bbx F_CONE_4X_F; //45ft - 45, 20 ; 40ft - 40
	bbx F_CONE_4X_R; //45ft - 40
	bbx F_CONE_20_F; //40ft - 20
	bbx F_CONE_20_R; //45ft - 20

	bbx F_IGH_4X_F; //45ft - 45, 20 ; 40ft - 40
	bbx F_IGH_4X_R; //45ft - 40
	bbx F_IGH_20_F; //40ft - 20
	bbx F_IGH_20_R; //45ft - 20

	bbx F_CST_4X_F; //45ft - 45, 20 ; 40ft - 40
	bbx F_CST_4X_R; //45ft - 40
	bbx F_CST_20_F; //40ft - 20
	bbx F_CST_20_R; //45ft - 20

	bbx R_CONE_4X_F; //40ft 
	bbx R_CONE_4X_R; //45ft
	bbx R_CONE_20_F; //45ft/40ft chassis - 20

	bbx R_IGH_4X_F; //40ft 
	bbx R_IGH_4X_R; //45ft
	bbx R_IGH_20_F; //45ft/40ft chassis - 20

	bbx R_CST_4X_F; //40ft 
	bbx R_CST_4X_R; //45ft
	bbx R_CST_20_F; //45ft/40ft chassis - 20

	MAX_MIN_RANGE F_XT_4XF_RNG;
	MAX_MIN_RANGE F_XT_4XR_RNG;
	MAX_MIN_RANGE F_XT_20F_RNG;
	MAX_MIN_RANGE F_XT_20R_RNG;

	MAX_MIN_RANGE F_IGH_4XF_RNG;
	MAX_MIN_RANGE F_IGH_4XR_RNG;
	MAX_MIN_RANGE F_IGH_20F_RNG;
	MAX_MIN_RANGE F_IGH_20R_RNG;

	MAX_MIN_RANGE F_CST_4XF_RNG;
	MAX_MIN_RANGE F_CST_4XR_RNG;
	MAX_MIN_RANGE F_CST_20F_RNG;
	MAX_MIN_RANGE F_CST_20R_RNG;

	MAX_MIN_RANGE R_XT_4XF_RNG;
	MAX_MIN_RANGE R_XT_4XR_RNG;
	MAX_MIN_RANGE R_XT_20F_RNG;

	MAX_MIN_RANGE R_IGH_4XF_RNG;
	MAX_MIN_RANGE R_IGH_4XR_RNG;
	MAX_MIN_RANGE R_IGH_20F_RNG;

	MAX_MIN_RANGE R_CST_4XF_RNG;
	MAX_MIN_RANGE R_CST_4XR_RNG;
	MAX_MIN_RANGE R_CST_20F_RNG;

	//Per-config CONE HEIGHT (world Y, mm) for FORCE_ROI_MODE region extraction. The
	//CHS20 wheel is nearer-Z than the cone, so the cone height must be a PRESET (a
	//runtime nearest-Z estimate would lock onto the wheel). Optional 3rd value of each
	//PRESET_*_3D range key (MAX;MIN;CONE_Y); defaults below are GT body-centre means.
	int F_XT_4XF_CY = -218; int F_XT_4XR_CY = -148; int F_XT_20F_CY = -274; int F_XT_20R_CY = -274;
	int R_XT_4XF_CY = -438; int R_XT_4XR_CY = -323; int R_XT_20F_CY = -435;
	//4XR values re-derived 2026-07 from the new Center-CHS45-SPRD40 GT samples
	//(FS cluster H mean -148, RS cluster H mean -323; opposite-cone outlier job excluded).
};

struct JobInfo
{
	bool isMountCycle = false;
	bool isOffloadCycle = false;

	std::string jobPos = "";
	std::string chassisType = "";

	int chassisLen = 0;
	int jobSize = 0;

	bbx presetBox;
	MAX_MIN_RANGE presetRangeX;

	void set(bool mountCycle, bool offloadCycle,
		std::string job_pos, std::string chassis_type, int chassis_len, int job_size, bbx preset_box, MAX_MIN_RANGE preset_range)
	{
		isMountCycle = mountCycle;
		isOffloadCycle = offloadCycle;
		jobPos = job_pos;
		chassisType = chassis_type;
		chassisLen = chassis_len;
		jobSize = job_size;
		presetBox = preset_box;
		presetRangeX = preset_range;
	}

	void print_jobInfo()
	{
		logMessage("Job Info - MountCycle: " + std::to_string(isMountCycle) +
			", OffloadCycle: " + std::to_string(isOffloadCycle) +
			", JobPos: " + jobPos +
			", ChassisType: " + chassisType +
			", ChassisLen: " + std::to_string(chassisLen) +
			", JobSize: " + std::to_string(jobSize));
		logMessage("Job Info - Preset: (" + std::to_string(presetBox.x) + "," + std::to_string(presetBox.y) + "," + std::to_string(presetBox.w) + "," + std::to_string(presetBox.h) + ")");
		logMessage("Job Info - Preset X (Min, Max): " + std::to_string(presetRangeX.MIN) + " , " + std::to_string(presetRangeX.MAX));
	}
};

struct ProcessResults
{
	//Chassis Type
	std::string VA_Chassis_Type = "";

	bool bDetected_XT = false;
	bool bDetected_CST = false;
	bool bDetected_IGH = false;

	bool bDetected_Container = false;
	bool bDetected_Chassis = false;

	//Target Hole
	bbx VA_Target_Container;
	bbx VA_Target_Chassis;

	//Deviations
	int VA_Devout_X = 0;
	int VA_Devout_Y = 0;

	//VA Results
	bool VA_isLandout = false;
	bool VA_isCLPS = false;
	bool VA_isCLPS_OK = false;

	//3D Results
	pcl::PointXYZ PCA_Target_Container;
	pcl::PointXYZ PCA_Target_Chassis;
	pcl::PointXYZ PCA_Target_Guide;

	//Deviations
	int PCA_Devout_X = 0;
	int PCA_Devout_Y = 0;
	int PCA_Devout_Z = 0;

	//3D Results
	bool PCA_isLandout = false;
	bool PCA_isCLPS = false;
	bool PCA_isCLPS_OK = false;

	bool bLandOutDetected = false;
	bool bClpsDetected = false;
	bool bClpsOkDetected = false;
};


std::string app_path;

#pragma region INI Variables

std::string sensor_ip;
std::string sensor_port;

std::string SOCKET_IP = "127.0.0.1";
int SOCKET_PORT;

const int sendLen = 50;
const int recvLen = 50;

bool MODE_DEBUG = true;

std::string DEBUG_SAMPLE_JOB = "";

bool DEBUG_WITH_FILES = false;
std::string DEBUG_PATH;
std::string DEBUG_SENSOR_POSITION = "FRONT";

bool DEBUG_BATCH_JOB = false;
std::string DEBUG_BATCH_ROOT_DIR = "";
std::string DEBUG_BATCH_SAVE_DIR = "";
//Reverse-replay: process Mount jobs BACKWARDS and treat them as OFFLOAD jobs.
//A mount recording played in reverse is an offload (landed -> hoist -> separation ->
//cone-only) -- the only sequence data containing the separation phase, since real
//offload recordings all end while still landed. Used to exercise/validate the
//CLPS / CLPS-OK logic offline. seoho.ini [DEBUG] ENABLE_MOUNT_REVERSE=1.
bool ENABLE_MOUNT_REVERSE = false;

bool DEBUG_CONVERT_PCL_RANGE = false;

bool ENALBE_RESTART_APPLICATION = false;
bool ENABLE_SAVE_LOG_BIT = false;
bool ENABLE_PRESET_VISUAL_ON_STREAM = false;
bool ENABLE_IMAGE_LOGGING_ONLY = false;

std::string CURRENT_SENSOR_POSITION = "FRONT";
bool DIRECTION_FRONT_IS_RIGHT = true; //current sample. TODO: need to read from INI.

int PROB_LIMIT = 50;

std::string onnx_model_path;

double TILT_ANGLE_DEG = 0.0;

int SENSOR_HEIGHT = 0;

MAX_MIN_RANGE HEIGHT_MMR;
MAX_MIN_RANGE TROLLEY_MMR;

int HEIGHT_RANGE_MAX = 200;
int HEIGHT_RANGE_MIN = -400;
int TROLLEY_RANGE_MAX = 2000;
int TROLLEY_RANGE_MIN = 1500;

//FORCE_ROI_MODE: VA-independent chassis localization backup (test path).
//When set, the chassis 3D body comes from the ROI tracker (ChassisRoi.h)
//instead of the VA-box mapping; VA still runs for A/B comparison logging.
bool FORCE_ROI_MODE = false;
//ROI_FALLBACK: VA-PRIMARY, ROI-FALLBACK (distinct from FORCE_ROI_MODE's always-ROI).
//VA runs normally; the ROI cone (ChassisRoi.h) replaces _cone_pos ONLY when VA missed
//the cone or (mount) mis-picked it (VA cone world-X outside the job's preset window).
bool ROI_FALLBACK = false;
int ROI_LOCK_K = 3;			//consecutive agreeing frames to lock baseline
int ROI_LOCK_TOL = 30;		//acquire agreement tolerance (mm)
int ROI_MOVE_TOL = 50;		//lateral move vs baseline => chassis moved (mm)
int ROI_REVERIFY_N = 10;	//broad re-verification every N frames
int ROI_CONE_MIN_H = 50;	//cone/guide summit above deck, min (mm)
int ROI_CONE_MAX_H = 250;	//cone/guide summit above deck, max (mm)
int ROI_MIN_POINTS = 50;	//min points to accept the cone/guide body
//Wide ROI CROP for the close-range XT Center 40chs/20job (F-app, 20F preset). The two
//front/back cones (split along crop p.x = Gantry) sit close, and the Center sensor is
//near, so the tight target-centred crop drops the target on a small chassis shift and
//admits the neighbour. When set (MAX;MIN along p.x), fillRoiParams uses this wider crop so
//BOTH cones are captured -> coneColumns splits them -> the fixed side-rule picks the target
//end. Unset (-10000) = use presetRangeX. The VA-validation window stays presetRangeX(tight),
//so widening the crop does NOT weaken the mis-pick check for this config.
int ROI_WIDE_20F_MAX = -10000;
int ROI_WIDE_20F_MIN = -10000;
ChassisRoiTracker chassisRoiTracker;
ClpsOffload3D clpsOffload3D;	//3D CLPS / CLPS-OK (offload), reverse-replay validated

PRESET_POS presetPos;

int LDO_NEAR_X_THRESHOLD = 50;
int LDO_NEAR_Y_THRESHOLD = 50;

int LDO_FAR_X_THRESHOLD = 50;
int LDO_FAR_Y_THRESHOLD = 50;

int LDO_GANTRY_THRESHOLD = 50;
int LDO_TROLLEY_THRESHOLD = 50;

int LDO_NCOUNT = 10;

int CLPS_NEAR_X_THRESHOLD = 50;
int CLPS_NEAR_Y_THRESHOLD = 50;

int CLPS_NCOUNT = 10;

//For aRMG.
std::string IP_20F = "127.0.0.1";
std::string IP_20R = "127.0.0.1";
std::string IP_4XF = "127.0.0.1";
std::string IP_4XR = "127.0.0.1";

//For Endfeed.
std::vector<std::string> FRONT_IP_ADDRESSES = {};
std::vector<std::string> CENTER_IP_ADDRESSES = {};
std::vector<std::string> REAR_IP_ADDRESSES = {};

#pragma endregion

SOCKET connectSocket_ = NULL;

char* sendBuffer_;
char sendBuf_[DEFAULT_SENDLEN] = {};

char* recvBuffer_;
char recvBuf_[DEFAULT_RECVLEN] = {};

bool blnDataSave;
std::string SaveDir;
std::string SaveImageDir;
std::string SaveDepthDir;

/*
* IO List Variables
*/

//PLC_Crane ---> T-Mini Application
unsigned short heartbeat_, prev_heartbeat_;
short Crane_ID;
short Lane_Number;

//Job information 1,2
bool Offload_Cycle = false, Mount_Cycle = false;
bool Job_Pos_B = false, Job_Pos_S = false, Job_Pos_C = false;
bool SPRD_TWL_Locked = false, SPRD_TWL_Unlocked = false, SPRD_Landed = false;

bool SPRD_20ft = false, SPRD_40ft = false, SPRD_45ft = false;
bool Hoist_Stopped = false, Trolley_Stopped = false, Gantry_Stopped = false, Above_Safe_Height = false;

bool NewJobPulse = false;

bool CHS_20ft = false, CHS_40ft = false, CHS_45ft = false;
bool CHS_XT = false, CHS_CST = false, CHS_IGH = false;

//Commands
bool enable_stream = false, enable_process = false, enable_logging = false, enable_save_logs = false;
bool Connected_Sensor_Reboot = false, Application_Shutdown = false;

//T-Mini Application ---> PLC_Crane
bool model_loaded = false, model_initialized = false;
bool sensor_connected = false, sensor_fault = false;
bool enabled_stream = false, enabled_process = false, enabled_logging = false;

short Detected_Chs_X_VA = 0, Detected_Chs_Y_VA = 0, Detected_Chs_Prob = 0;
short Detected_Cntr_X_VA = 0, Detected_Cntr_Y_VA = 0, Detected_Cntr_Prob = 0;

//VA Process Results
bool Detected_XT = false, Detected_CST = false, Detected_IGH = false;
bool Detected_Cntr = false, Detected_Chs = false;

short Detected_Chs_X_3D = 0, Detected_Chs_Y_3D = 0, Detected_Chs_Z_3D = 0;
short Detected_Cntr_X_3D = 0, Detected_Cntr_Y_3D = 0, Detected_Cntr_Z_3D = 0;

bool Landout_Detected_VA = false, CLPS_Detected_VA = false, CLPS_OK_Detected_VA = false;

//3D Process Results
bool Landout_Detected_3D = false, CLPS_Detected_3D = false, CLPS_OK_Detected_3D = false;

short Dev_X_VA = 0, Dev_Y_VA = 0;
short Dev_X_3D = 0, Dev_Y_3D = 0, Dev_Z_3D = 0;

/*
* Process Variables
*/

std::string current_lane_ip = "";
std::string current_lane_camera = "";

std::string saveName;
std::string saveDirName;

std::string job_name;
std::string job_result_folder_name;
//images, pointcloud, and .log files
std::string job_result_log_file_name;

std::chrono::system_clock::time_point sensor_last_attempted_time = std::chrono::system_clock::now();
std::chrono::system_clock::time_point sensor_last_connected_time = std::chrono::system_clock::now();

YOLO11Detector yolo_detector;
TrtRunner trt_detector;

bbx LDO_intrim;
bbx LDO_Base;
int LDO_Base_Count = 0;
bool LDO_Base_Set = false;

int LDO_Current_Count = 0;
int CLPS_Current_Count = 0;
int CLPS_OK_Current_Count = 0;

bbx CLPS_intrim;
bbx CLPS_Base;
int CLPS_Base_Count = 0;
bool CLPS_Base_Set = false;

bool landout_detected_pca = false;
int landout_current_count_pca = 0;
bool clps_detected_pca = false;
int clps_current_count_pca = 0;
bool clps_ok_detected_pca = false;
int clps_ok_current_count_pca = 0;

bool clps_detected_pca_only = false;
bool clps_ok_detected_pca_only = false;
int clps_detected_pca_only_count = 0;
int clps_ok_detected_pca_only_count = 0;

JobInfo current_job_info;
ProcessResults current_process_results;

bool Logging_Enable = false;
bool Front_Logging = false;
bool Rear_Logging = false;

bool trigger_log_delete = false;
bool trigger_terminate = false;

int tCntr_x, tCntr_y, tCntr_prob;
int tCone_x, tCone_y, tCone_prob;
int chassis_detected_type;

bool landout_detected = false;
bool clps_detected = false;
bool clps_ok_detected = false;

int target_lane_number = 1;

int devOut_x, devOut_y;
int devOut_LDO_x, devOut_LDO_y;
//2025.03.11
int devOut_x_mm, devOut_y_mm;
int devOut_pca_x, devOut_pca_y, devOut_pca_z;

bool bDetected_hole = false, bDetected_cone = false;

//combined output.
bool bCLPS_Detected_Out = false;
bool bCLPS_OK_Detected_Out = false;

DetectionAverager AVG_CONE;
DetectionAverager AVG_HOLE;

PCAAverager AVG_CONE_PCA;

//3D deviation OUTPUT multiplex (XT/IGH). The value written to Dev_*_3D (PLC bytes
//36-41) is phase-multiplexed: POSITION phase -> the robust latched reference; DEVIATION
//phase -> hole(realtime) - reference(latched). Mount ref = cone(XT)/guide(IGH) latched
//from job start; offload latches the hole from the start and starts the cone latch at
//the VA cone-idx1+hole separation. Detection (landout/CLPS) still runs on the realtime
//deviation; this only shapes the output bytes. Reset per job in reset_jobVariables().
MedianLatch3D DEV3D_RefLatch;			//chassis reference (cone/guide)
MedianLatch3D DEV3D_HoleLatch;			//offload phase-1 hole
int  DEV3D_mount_hole_count = 0;		//cumulative (not continuous) VA hole frames
bool DEV3D_mount_dev_phase = false;		//sticky: cone position -> deviation
bool DEV3D_offload_dev_phase = false;	//sticky: hole position -> deviation

//Validated VA cone INIT (mount, ROI_FALLBACK). Once the VA cone is seen IN-RANGE (inside
//the job's preset window) we latch it: 3D position (robust median) + last-good 2D box.
//On a later VA failure we reuse this (chassis stationary until load transfer) BEFORE
//resorting to ROI. An out-of-range mis-pick never enters the latch, so systematically
//mis-picked jobs correctly fall through to ROI. Reset per job.
MedianLatch3D VA_ConeInit3D;			//validated init cone WORLD position
bbx  VA_ConeInit2D;						//validated init cone 2D box (last in-range)
bool VA_ConeInit2D_Set = false;

bbx Offload_Hole_Base;
bbx Offload_Hole_intrim;
int Offload_Hole_Base_Count = 0;
bool Offload_Hole_Base_Set = false;

pcl::PointXYZ PCA_intrim;
pcl::PointXYZ PCA_Base;
int PCA_Base_Count = 0;
bool PCA_Base_Set = false;


//Debug
std::string DEBUG_IMG_PATH = "";
std::string DEBUG_PLY_PATH = "";
std::vector<std::string> DEBUG_IMG_FILES;
std::vector<std::string> DEBUG_PLY_FILES;
int DEBUG_CURRENT_INDEX = 0;
int DEBUG_MAX_INDEX = 0;

LONG WINAPI MyUnhandledExceptionFilter(EXCEPTION_POINTERS* ExceptionInfo) {
	HANDLE hFile = CreateFile(L"crash.dmp", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (hFile != INVALID_HANDLE_VALUE) {
		MINIDUMP_EXCEPTION_INFORMATION mdei;
		mdei.ThreadId = GetCurrentThreadId();
		mdei.ExceptionPointers = ExceptionInfo;
		mdei.ClientPointers = FALSE;

		MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(),
			hFile, MiniDumpNormal, &mdei, nullptr, nullptr);

		CloseHandle(hFile);
	}

	return EXCEPTION_EXECUTE_HANDLER;
}

//job result logging related
bool save_trigger_by_landed = false;
bool save_trigger_by_TWL_Locked = false;

std::chrono::system_clock::time_point last_frame_get_time = std::chrono::system_clock::now();
bool blnGetNewFrame = false;
bool blnFrameFault = false;

std::atomic<bool> socket_running(true);

/*
 * Mutex and Conditional Variables
 *
 */

std::mutex mutex_tmini_ctrl;
std::condition_variable cond_tmini_ctrl;
//this one is for running/terminating thread.
std::atomic<bool> tmini_ctrl_running(true);
bool tmini_ctrl_flag = false;

std::mutex mutex_tmini_data_stream;
std::condition_variable cond_tmini_data_stream;
//this one is for running/terminating thread.
std::atomic<bool> tmini_data_stream_running(true);
bool tmini_data_stream_flag = false;

std::mutex mutex_processing;
std::condition_variable cond_processing;
//this one is for running/terminating thread.
std::atomic<bool> proc_running(true);

std::mutex mutex_logging;
std::condition_variable cond_logging;
//this one is for running/terminating thread.
std::atomic<bool> logging_running(true);

/*
 * Log Timer Variables
 */

std::queue<std::string> logQueue;
std::mutex queueMutex;
std::condition_variable cvLog;
std::atomic<bool> log_running(true);

std::mutex logMutex; // Mutex to protect log file writing
std::string currentLogFileName; // To track the current log file

std::queue<std::string> jobLogQueue;
std::mutex jlQueueMutex;
std::condition_variable cvJobLog;
std::atomic<bool> job_log_running(true);

std::mutex jobLogMutex; // Mutex to protect log file writing
std::string currentJobLogFileName; // To track the current log file

std::mutex jobLogEnableMutex;
std::condition_variable cond_jobLogEnable;

std::string appID = "";
std::string appName = "TMini";

// Generate Visionary instance
auto pDataHandler = std::make_shared<VisionaryTMiniData>();
VisionaryDataStream dataStream(pDataHandler);
VisionaryControl visionaryControl;

//Save Update 2026.03.04
struct SaveTask
{
	cv::Mat oImage;
	cv::Mat resImage;
	pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud; //pointer not value
	bool isJobLog;
	std::string msg;
	std::chrono::system_clock::time_point frameTime;

	SaveTask() = default;

	// Constructor matching the usage: SaveTask(img, res_img, pcPointCloud, true, std::move(msg), ts);
	SaveTask(const cv::Mat& oImg,
		const cv::Mat& rImg,
		const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cld,
		bool jobLog,
		std::string&& message,
		std::chrono::system_clock::time_point time)
		: oImage(oImg.clone()),
		resImage(rImg.clone()),
		cloud(cld),
		isJobLog(jobLog),
		msg(std::move(message)),
		frameTime(time)
	{
	}
};

class ThreadSafeQueue {
private:
	std::queue<SaveTask> q;
	mutable std::mutex mtx;
	std::condition_variable cv;
	bool stopping = false;

public:
	void push(SaveTask&& task) {
		{
			std::lock_guard<std::mutex> lock(mtx);
			q.push(std::move(task));
		}
		cv.notify_one();
	}

	// Blocks until an item exists OR stop() is called.
	bool pop(SaveTask& out) {
		std::unique_lock<std::mutex> lock(mtx);
		cv.wait(lock, [&] { return stopping || !q.empty(); });
		if (stopping && q.empty()) return false;
		out = std::move(q.front());
		q.pop();
		return true;
	}

	// Optional: batch drain
	bool try_pop(SaveTask& out) {
		std::lock_guard<std::mutex> lock(mtx);
		if (q.empty()) return false;
		out = std::move(q.front());
		q.pop();
		return true;
	}

	void stop() {
		{
			std::lock_guard<std::mutex> lock(mtx);
			stopping = true;
		}
		cv.notify_all();
	}

	size_t size() const {
		std::lock_guard<std::mutex> lock(mtx);
		return q.size();
	}

	void clear() {
		std::lock_guard<std::mutex> lock(mtx);
		while (!q.empty()) q.pop();
	}
};


/*
class ThreadSafeQueue
{
private:
	std::queue <std::tuple<cv::Mat, cv::Mat, pcl::PointCloud<pcl::PointXYZ>, bool, std::string, std::chrono::system_clock::time_point>> queue;
	std::mutex mtx;
	std::condition_variable cv;

public:
	void push(std::tuple<cv::Mat, cv::Mat, pcl::PointCloud<pcl::PointXYZ>, bool, std::string, std::chrono::system_clock::time_point> value) {
		std::lock_guard<std::mutex> lock(mtx);
		queue.push(value);
		cv.notify_one(); // Notify the consumer
	}

	// Pop element from the queue (blocks if empty)
	std::tuple<cv::Mat, cv::Mat, pcl::PointCloud<pcl::PointXYZ>, bool, std::string, std::chrono::system_clock::time_point> pop() {
		std::unique_lock<std::mutex> lock(mtx);
		cv.wait(lock, [this] { return !queue.empty(); }); // Wait until queue is non-empty
		std::tuple<cv::Mat, cv::Mat, pcl::PointCloud<pcl::PointXYZ>, bool, std::string, std::chrono::system_clock::time_point> value = queue.front();
		queue.pop();
		return value;
	}

	int GetQueueLen()
	{
		std::lock_guard<std::mutex> lock(mtx);
		return queue.size();
	}

	void clear() {
		std::lock_guard<std::mutex> lock(mtx);
		while (!queue.empty()) {
			queue.pop();
		}
	}
};
*/

ThreadSafeQueue tsq;

DataStack dataStack;

bool processFlag = false;
bool saveFlag = false;

bool jobLogFlag = false;


void my_handler(int s)
{

	logging_running.store(false);
	proc_running.store(false);	
	socket_running.store(false);

	logMessage("Caught signal " + std::to_string(s));
	logMessage("Terminating tmini_process application triggered by Ctrl+C");

	log_running.store(false);

	exit(1);
}

/*
* Utility Functions
*/

std::vector<std::string> split(const std::string& str, char delimiter) {
	std::vector<std::string> tokens;
	std::stringstream ss(str);
	std::string token;

	while (std::getline(ss, token, delimiter)) {
		tokens.push_back(token);
	}
	return tokens;
}
auto print_time() -> std::string
{
	auto now = std::chrono::system_clock::now();

	// Convert to time_t to use with std::localtime
	std::time_t current_time = std::chrono::system_clock::to_time_t(now);

	// Convert to tm struct (local time)
	std::tm local_time = *std::localtime(&current_time);

	// Get the milliseconds part
	auto duration = now.time_since_epoch();
	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000;

	std::ostringstream oss;
	oss << std::put_time(&local_time, "%Y%m%d_%H%M%S") << "_" << std::setw(3) << std::setfill('0') << milliseconds;

	// Get the formatted time as a string
	std::string formatted_time = oss.str();

	// Output formatted current time
	return formatted_time;
}
//Check if directory exists
auto dirExists(const char* const path) -> int
{
	struct stat info;

	int statRC = stat(path, &info);
	if (statRC != 0)
	{
		if (errno == ENOENT) { return 0; } // something along the path does not exist
		if (errno == ENOTDIR) { return 0; } // something in path prefix is not a dir
		return -1;
	}

	return (info.st_mode & S_IFDIR) ? 1 : 0;
}
//Check if directory exists, if not create one (returns true if created).
auto createDirectory_ifexists(std::string path) -> bool
{
	auto status = false;
	auto dirE = dirExists(path.c_str());
	if (dirE == 0)
	{
		auto ret = _mkdir(path.c_str());
		if (ret == -1) {
			logMessage("Error in creating directory!");
		}
		status = true;
	}
	return status;
}
// Get the current date as a string in the format YYYY-MM-DD
std::string getCurrentDate() {
	auto now = std::chrono::system_clock::now();
	std::time_t current_time = std::chrono::system_clock::to_time_t(now);
	std::tm local_time = *std::localtime(&current_time);
	std::ostringstream oss;
	oss << std::put_time(&local_time, "%Y%m%d");
	return oss.str();
}

template <typename Duration>
auto print_time(tm t, Duration fraction) -> std::string
{
	using namespace std::chrono;
	char val[256];
	std::sprintf(val, "%04u%02u%02u_%02u%02u%02u_%03u", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec, static_cast<unsigned>(fraction / milliseconds(1)));
	std::string s(val);
	return s;
}
//Represent current datetime as filename
auto time_as_name() -> std::string
{
	//name is specific to be time. yyyyMMdd_hhMMss_msec
	std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
	std::chrono::system_clock::duration tp = now.time_since_epoch();
	tp -= std::chrono::duration_cast<std::chrono::seconds>(tp);
	time_t tt = std::chrono::system_clock::to_time_t(now);

	std::string t = print_time(*localtime(&tt), tp);
	return t;
}
auto time_as_name(std::chrono::system_clock::time_point ts) -> std::string
{
	//name is specific to be time. yyyyMMdd_hhMMss_msec
	//std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
	auto now = ts;
	std::chrono::system_clock::duration tp = now.time_since_epoch();
	tp -= std::chrono::duration_cast<std::chrono::seconds>(tp);
	time_t tt = std::chrono::system_clock::to_time_t(now);

	std::string t = print_time(*localtime(&tt), tp);
	return t;
}
auto time_up_to_millseconds(std::chrono::system_clock::time_point time_data) -> std::string {
	std::chrono::system_clock::duration tp = time_data.time_since_epoch();
	tp -= std::chrono::duration_cast<std::chrono::seconds>(tp);
	time_t tt = std::chrono::system_clock::to_time_t(time_data);

	std::string t = print_time(*localtime(&tt), tp);
	return t;
}

std::wstring s2ws(const std::string& s) //string -> LPCWSTR needs first conversion to wstring then to LPCWSTR.
{
	int len;
	int slength = (int)s.length() + 1;
	len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
	wchar_t* buf = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
	std::wstring r(buf);
	delete[] buf;
	return r;
}
template <size_t N1, size_t N2>
std::bitset <N1 + N2> concat(const std::bitset <N1>& b1, const std::bitset <N2>& b2) {
	std::string s1 = b1.to_string();
	std::string s2 = b2.to_string();
	return std::bitset <N1 + N2>(s1 + s2);
}
template<std::size_t B>
long bitset_to_long(const std::bitset<B>& b) {
	struct { long x : B; } s;
	return s.x = b.to_ulong();
}

std::string Utf16ToUtf8(const std::wstring& w) {
	int len = WideCharToMultiByte(CP_UTF8, 0, w.data(), w.size(), nullptr, 0, nullptr, nullptr);
	std::string s(len, '\0');
	WideCharToMultiByte(CP_UTF8, 0, w.data(), w.size(), &s[0], len, nullptr, nullptr);
	return s;
}
namespace fs = std::filesystem;
std::string getParent(const std::string& pathStr) {
	fs::path p(pathStr);
	return p.parent_path().string();
}


// Function to write log entries to file
void logWriterThread() {
	std::ofstream logFile;

	createDirectory_ifexists(app_path + "/Log");

	while (log_running || !logQueue.empty()) {
		try
		{
			std::unique_lock<std::mutex> lock(queueMutex);
			cvLog.wait(lock, [] { return !logQueue.empty() || !log_running; });

			// Get the current date and check if it has changed
			std::string newLogFileName = app_path + "/Log/" + appName + "_log_" + getCurrentDate() + ".log";

			// If the date has changed, close the current log file and open a new one
			if (newLogFileName != currentLogFileName) {
				if (logFile.is_open()) {
					logFile.close(); // Close the previous file
				}
				logFile.open(newLogFileName, std::ios::out | std::ios::app); // Open new log file
				currentLogFileName = newLogFileName; // Update the current log file name
			}

			// Write all logs in the queue to the current file
			while (!logQueue.empty()) {
				logFile << logQueue.front() << std::endl;
				logQueue.pop();
			}
			lock.unlock();
		}
		catch (std::exception& ex)
		{
			std::cout << "Error in log thread : " << std::string(ex.what());
			//return false;
		}
		catch (...)
		{
			std::cout << "Unknown error in log thread";
			//return false;
		}
	}

	// Close the log file at the end
	if (logFile.is_open()) {
		logFile.close();
	}
}
void jobLogWriterThread() {
	std::ofstream logFile;

	while (job_log_running || !jobLogQueue.empty()) {
		std::unique_lock<std::mutex> lock(jlQueueMutex);
		cvJobLog.wait(lock, [] { return !jobLogQueue.empty() || !job_log_running; });

		// Get the current date and check if it has changed
		std::string newLogFileName = job_result_log_file_name;

		// If the date has changed, close the current log file and open a new one
		if (newLogFileName != currentJobLogFileName)
		{
			logMessage("New filename: " + newLogFileName);
			if (logFile.is_open()) {
				logFile.close(); // Close the previous file
			}
			logFile.open(newLogFileName, std::ios::out | std::ios::app); // Open new log file
			currentJobLogFileName = newLogFileName; // Update the current log file name
		}

		// Write all logs in the queue to the current file
		while (!jobLogQueue.empty()) {
			logFile << jobLogQueue.front() << std::endl;
			jobLogQueue.pop();
		}
		lock.unlock();
	}

	// Close the log file at the end
	if (logFile.is_open()) {
		logFile.close();
	}
}

// Function to log messages (called by other threads)
void logMessage(const std::string& message) {

	try
	{
		//std::cout << "Logging working " << message << "\n";
		std::stringstream ss;
		auto now = std::chrono::system_clock::now();
		auto time_since_epoch = now.time_since_epoch();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_epoch).count() % 1000;

		std::time_t current_time = std::chrono::system_clock::to_time_t(now);
		std::tm local_time = *std::localtime(&current_time);

		ss << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << "." << std::setw(3) << std::setfill('0') << ms << " - " << message;

		{
			std::lock_guard<std::mutex> lock(queueMutex);
			logQueue.push(ss.str());
		}
		printf((ss.str() + "\n").c_str());
		cvLog.notify_one();
	}
	catch (std::exception& ex)
	{
		std::cout << "Error in log message : " << std::string(ex.what());
		//return false;
	}
}
void jobLogMessage(const std::string& message) {
	std::stringstream ss;
	auto now = std::chrono::system_clock::now();
	auto time_since_epoch = now.time_since_epoch();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_epoch).count() % 1000;

	std::time_t current_time = std::chrono::system_clock::to_time_t(now);
	std::tm local_time = *std::localtime(&current_time);

	ss << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << "." << std::setw(3) << std::setfill('0') << ms << " - " << message;

	{
		std::lock_guard<std::mutex> lock(jlQueueMutex);
		jobLogQueue.push(ss.str());
	}
	//printf((ss.str() + "\n").c_str());
	cvJobLog.notify_one();
}

/*
* INI Parsing Functions
*/

static inline void remove_all_whitespace(std::string& s) {
	s.erase(std::remove_if(s.begin(), s.end(),
		[](unsigned char c) { return std::isspace(c); }),
		s.end());
}

static inline std::vector<std::string> split_semicolon(const std::string& s) {
	std::vector<std::string> out;
	size_t start = 0;
	while (start <= s.size()) {
		size_t pos = s.find(';', start);
		if (pos == std::string::npos) pos = s.size();
		out.emplace_back(s.substr(start, pos - start));
		start = pos + 1;
	}
	// optional: drop trailing empty if string ends with ';'
	if (!out.empty() && out.back().empty()) out.pop_back();
	return out;
}

auto parseAppName(string path) -> bool
{
	try
	{
		mINI::INIFile inireader(path + "/INI/seoho.ini");
		mINI::INIStructure inidata;

		inireader.read(inidata);

		appID = inidata.get("ID").get("APP");
		return true;
	}
	catch (std::exception& ex)
	{
		std::cout << "Error in parsing INI file : " << std::string(ex.what());
		return false;
	}
}
auto parseINI(string path) -> bool
{
	try
	{
		mINI::INIFile inireader(path + "/INI/seoho.ini");
		mINI::INIStructure inidata;

		inireader.read(inidata);

		appID = inidata.get("ID").get("APP");
		if (appID.find("FRONT") != std::string::npos)
		{
			CURRENT_SENSOR_POSITION = "FRONT";
			logMessage("Current application is for T-Mini Front Sensor!");
		}
		else
		{
			CURRENT_SENSOR_POSITION = "REAR";
			logMessage("Current application is for T-Mini Rear Sensor!");
		}

		SOCKET_IP = inidata.get("SOCKET").get("IP");
		SOCKET_PORT = std::stoi(inidata.get("SOCKET").get("PORT"));
		logMessage("Socket Port: " + SOCKET_IP + " , " + std::to_string(SOCKET_PORT));
		
		PROB_LIMIT = std::stoi(inidata.get("SYSTEM").get("PROB_LIMIT"));
		ENABLE_SAVE_LOG_BIT = inidata.get("SYSTEM").get("ENABLE_SAVE_LOG_BIT") == "1" ? true : false;
		ENALBE_RESTART_APPLICATION = inidata.get("SYSTEM").get("ENABLE_RESTART_APPLICATION") == "1" ? true : false;
		ENABLE_PRESET_VISUAL_ON_STREAM = inidata.get("SYSTEM").get("ENABLE_PRESET_VISUAL_ON_STREAM") == "1" ? true : false;
		ENABLE_IMAGE_LOGGING_ONLY = inidata.get("SYSTEM").get("ENABLE_IMAGE_LOGGING_ONLY") == "1" ? true : false;

		onnx_model_path = inidata.get("MODEL").get("ONNX_PATH");
		logMessage("onnx model path: " + onnx_model_path);

		MODE_DEBUG = inidata.get("DEBUG").get("MODE_DEBUG") == "1" ? true : false;
		if (MODE_DEBUG) logMessage("Debug mode enabled!");

		DEBUG_SAMPLE_JOB = inidata.get("DEBUG").get("DEBUG_SAMPLE_JOB");

		DEBUG_CONVERT_PCL_RANGE = inidata.get("DEBUG").get("DEBUG_CONVERT_PCL_RANGE") == "1" ? true : false;

		DEBUG_SENSOR_POSITION = inidata.get("DEBUG").get("DEBUG_SENSOR_POSITION");
		
		DEBUG_WITH_FILES = inidata.get("DEBUG").get("DEBUG_WITH_FILES") == "1" ? true : false;

		DEBUG_PATH = inidata.get("DEBUG").get("DEBUG_PATH");

		DEBUG_BATCH_JOB = inidata.get("DEBUG").get("DEBUG_BATCH_JOB") == "1" ? true : false;
		if (DEBUG_BATCH_JOB) logMessage("Debug mode in batch mode!");
		DEBUG_BATCH_ROOT_DIR = inidata.get("DEBUG").get("DEBUG_BATCH_ROOT_DIR");
		DEBUG_BATCH_SAVE_DIR = inidata.get("DEBUG").get("DEBUG_BATCH_SAVE_DIR");

		//Optional: reverse-replay Mount jobs as OFFLOAD (CLPS/CLPS-OK offline testing).
		if (inidata.get("DEBUG").has("ENABLE_MOUNT_REVERSE"))
			ENABLE_MOUNT_REVERSE = inidata.get("DEBUG").get("ENABLE_MOUNT_REVERSE") == "1";
		if (ENABLE_MOUNT_REVERSE) logMessage("[DEBUG] ENABLE_MOUNT_REVERSE: Mount jobs will be replayed BACKWARDS as OFFLOAD");

		if (MODE_DEBUG || DEBUG_WITH_FILES || DEBUG_BATCH_JOB) logMessage("Debug Sensor Pos: " + DEBUG_SENSOR_POSITION);

		logMessage("INI Configuration Complete!");
	}
	catch (std::exception& e)
	{
		logMessage("Error in parsing INI file : " + std::string(e.what()));
		return false;
	}
	return true;
}
void parseProcessINI(string path)
{
	mINI::INIFile file(path + "/INI/process.ini");
	mINI::INIStructure ini;

	if (file.read(ini))
	{
		//Optional per-config CONE_Y = 3rd field of the XT_* range keys (MAX;MIN;CONE_Y).
		//MUST be throw-safe: legacy INIs carry a free-text comment as the 3rd field, and
		//parseProcessINI runs inside one big try -- an uncaught stoi would silently abort
		//ALL remaining preset parsing.
		auto parseCY = [](const std::vector<std::string>& parts, int& dst)
		{
			if ((int)parts.size() <= 2) return;
			try { dst = std::stoi(parts.at(2)); }
			catch (...) {}	//non-numeric 3rd field (comment) -> keep default
		};

		try
		{

			TILT_ANGLE_DEG = std::stod(ini.get("SETTINGS").get("TILT_ANGLE_DEG"));

			SENSOR_HEIGHT = std::stoi(ini.get("SETTINGS").get("SENSOR_HEIGHT"));

			HEIGHT_RANGE_MAX = std::stoi(ini.get("SETTINGS").get("HEIGHT_RANGE_MAX"));
			HEIGHT_RANGE_MIN = std::stoi(ini.get("SETTINGS").get("HEIGHT_RANGE_MIN"));

			HEIGHT_MMR.MAX = HEIGHT_RANGE_MAX;
			HEIGHT_MMR.MIN = HEIGHT_RANGE_MIN;

			TROLLEY_RANGE_MAX = std::stoi(ini.get("SETTINGS").get("TROLLEY_RANGE_MAX"));
			TROLLEY_RANGE_MIN = std::stoi(ini.get("SETTINGS").get("TROLLEY_RANGE_MIN"));

			TROLLEY_MMR.MAX = TROLLEY_RANGE_MAX;
			TROLLEY_MMR.MIN = TROLLEY_RANGE_MIN;

			//FORCE_ROI_MODE keys are optional; defaults apply when absent.
			if (ini.get("SETTINGS").has("FORCE_ROI_MODE")) FORCE_ROI_MODE = (std::stoi(ini.get("SETTINGS").get("FORCE_ROI_MODE")) != 0);
			if (ini.get("SETTINGS").has("ROI_FALLBACK")) ROI_FALLBACK = (std::stoi(ini.get("SETTINGS").get("ROI_FALLBACK")) != 0);
			if (ini.get("SETTINGS").has("ROI_WIDE_20F_MAX")) ROI_WIDE_20F_MAX = std::stoi(ini.get("SETTINGS").get("ROI_WIDE_20F_MAX"));
			if (ini.get("SETTINGS").has("ROI_WIDE_20F_MIN")) ROI_WIDE_20F_MIN = std::stoi(ini.get("SETTINGS").get("ROI_WIDE_20F_MIN"));
			if (ini.get("SETTINGS").has("ROI_LOCK_K")) ROI_LOCK_K = std::stoi(ini.get("SETTINGS").get("ROI_LOCK_K"));
			if (ini.get("SETTINGS").has("ROI_LOCK_TOL")) ROI_LOCK_TOL = std::stoi(ini.get("SETTINGS").get("ROI_LOCK_TOL"));
			if (ini.get("SETTINGS").has("ROI_MOVE_TOL")) ROI_MOVE_TOL = std::stoi(ini.get("SETTINGS").get("ROI_MOVE_TOL"));
			if (ini.get("SETTINGS").has("ROI_REVERIFY_N")) ROI_REVERIFY_N = std::stoi(ini.get("SETTINGS").get("ROI_REVERIFY_N"));
			if (ini.get("SETTINGS").has("ROI_CONE_MIN_H")) ROI_CONE_MIN_H = std::stoi(ini.get("SETTINGS").get("ROI_CONE_MIN_H"));
			if (ini.get("SETTINGS").has("ROI_CONE_MAX_H")) ROI_CONE_MAX_H = std::stoi(ini.get("SETTINGS").get("ROI_CONE_MAX_H"));
			if (ini.get("SETTINGS").has("ROI_MIN_POINTS")) ROI_MIN_POINTS = std::stoi(ini.get("SETTINGS").get("ROI_MIN_POINTS"));
			logMessage("[INI] FORCE_ROI_MODE: " + std::to_string(FORCE_ROI_MODE)
				+ " (LOCK_K:" + std::to_string(ROI_LOCK_K)
				+ " LOCK_TOL:" + std::to_string(ROI_LOCK_TOL)
				+ " MOVE_TOL:" + std::to_string(ROI_MOVE_TOL)
				+ " REVERIFY_N:" + std::to_string(ROI_REVERIFY_N) + ")");

			//PCA - 3D -- FRONT
			//XT
			auto tVal = ini.get("PRESET_FRONT_3D").get("XT_4XF");
			remove_all_whitespace(tVal);
			auto tParts = split_semicolon(tVal);

			presetPos.F_XT_4XF_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.F_XT_4XF_RNG.MIN = std::stoi(tParts.at(1));
			parseCY(tParts, presetPos.F_XT_4XF_CY);

			tVal = ini.get("PRESET_FRONT_3D").get("XT_4XR");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.F_XT_4XR_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.F_XT_4XR_RNG.MIN = std::stoi(tParts.at(1));
			parseCY(tParts, presetPos.F_XT_4XR_CY);

			tVal = ini.get("PRESET_FRONT_3D").get("XT_20F");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.F_XT_20F_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.F_XT_20F_RNG.MIN = std::stoi(tParts.at(1));
			parseCY(tParts, presetPos.F_XT_20F_CY);

			tVal = ini.get("PRESET_FRONT_3D").get("XT_20R");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.F_XT_20R_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.F_XT_20R_RNG.MIN = std::stoi(tParts.at(1));
			parseCY(tParts, presetPos.F_XT_20R_CY);

			//IGH
			tVal = ini.get("PRESET_FRONT_3D").get("IGH_4XF");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.F_IGH_4XF_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.F_IGH_4XF_RNG.MIN = std::stoi(tParts.at(1));

			tVal = ini.get("PRESET_FRONT_3D").get("IGH_4XR");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.F_IGH_4XR_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.F_IGH_4XR_RNG.MIN = std::stoi(tParts.at(1));

			tVal = ini.get("PRESET_FRONT_3D").get("IGH_20F");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.F_IGH_20F_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.F_IGH_20F_RNG.MIN = std::stoi(tParts.at(1));

			tVal = ini.get("PRESET_FRONT_3D").get("IGH_20R");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.F_IGH_20R_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.F_IGH_20R_RNG.MIN = std::stoi(tParts.at(1));

			//CST
			tVal = ini.get("PRESET_FRONT_3D").get("CST_4XF");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.F_CST_4XF_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.F_CST_4XF_RNG.MIN = std::stoi(tParts.at(1));

			tVal = ini.get("PRESET_FRONT_3D").get("CST_4XR");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.F_CST_4XR_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.F_CST_4XR_RNG.MIN = std::stoi(tParts.at(1));

			tVal = ini.get("PRESET_FRONT_3D").get("CST_20F");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.F_CST_20F_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.F_CST_20F_RNG.MIN = std::stoi(tParts.at(1));

			tVal = ini.get("PRESET_FRONT_3D").get("CST_20R");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.F_CST_20R_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.F_CST_20R_RNG.MIN = std::stoi(tParts.at(1));

			//REAR
			//XT
			tVal = ini.get("PRESET_REAR_3D").get("XT_4XF");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.R_XT_4XF_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.R_XT_4XF_RNG.MIN = std::stoi(tParts.at(1));
			parseCY(tParts, presetPos.R_XT_4XF_CY);

			tVal = ini.get("PRESET_REAR_3D").get("XT_4XR");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.R_XT_4XR_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.R_XT_4XR_RNG.MIN = std::stoi(tParts.at(1));
			parseCY(tParts, presetPos.R_XT_4XR_CY);

			tVal = ini.get("PRESET_REAR_3D").get("XT_20F");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.R_XT_20F_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.R_XT_20F_RNG.MIN = std::stoi(tParts.at(1));
			parseCY(tParts, presetPos.R_XT_20F_CY);

			//IGH
			tVal = ini.get("PRESET_REAR_3D").get("IGH_4XF");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.R_IGH_4XF_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.R_IGH_4XF_RNG.MIN = std::stoi(tParts.at(1));

			tVal = ini.get("PRESET_REAR_3D").get("IGH_4XR");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.R_IGH_4XR_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.R_IGH_4XR_RNG.MIN = std::stoi(tParts.at(1));

			tVal = ini.get("PRESET_REAR_3D").get("IGH_20F");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.R_IGH_20F_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.R_IGH_20F_RNG.MIN = std::stoi(tParts.at(1));

			//CST
			tVal = ini.get("PRESET_REAR_3D").get("CST_4XF");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.R_CST_4XF_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.R_CST_4XF_RNG.MIN = std::stoi(tParts.at(1));

			tVal = ini.get("PRESET_REAR_3D").get("CST_4XR");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.R_CST_4XR_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.R_CST_4XR_RNG.MIN = std::stoi(tParts.at(1));

			tVal = ini.get("PRESET_REAR_3D").get("CST_20F");
			remove_all_whitespace(tVal);
			tParts = split_semicolon(tVal);

			presetPos.R_CST_20F_RNG.MAX = std::stoi(tParts.at(0));
			presetPos.R_CST_20F_RNG.MIN = std::stoi(tParts.at(1));


			//Preset Load
#pragma region FRONT - XT Parse
			auto tempVal = ini.get("PRESET_FRONT").get("XT_4XF");
			remove_all_whitespace(tempVal);
			auto parts = split_semicolon(tempVal);

			presetPos.F_CONE_4X_F.x = std::stof(parts.at(0));
			presetPos.F_CONE_4X_F.y = std::stof(parts.at(1));
			presetPos.F_CONE_4X_F.w = std::stof(parts.at(2));
			presetPos.F_CONE_4X_F.h = std::stof(parts.at(3));

			logMessage("[INI] XT_4XF - X:" + std::to_string(presetPos.F_CONE_4X_F.x) + " , Y: " + std::to_string(presetPos.F_CONE_4X_F.y) + " , W: " + std::to_string(presetPos.F_CONE_4X_F.w) + " , H: " + std::to_string(presetPos.F_CONE_4X_F.h));

			tempVal = ini.get("PRESET_FRONT").get("XT_4XR");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.F_CONE_4X_R.x = std::stof(parts.at(0));
			presetPos.F_CONE_4X_R.y = std::stof(parts.at(1));
			presetPos.F_CONE_4X_R.w = std::stof(parts.at(2));
			presetPos.F_CONE_4X_R.h = std::stof(parts.at(3));

			tempVal = ini.get("PRESET_FRONT").get("XT_20F");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.F_CONE_20_F.x = std::stof(parts.at(0));
			presetPos.F_CONE_20_F.y = std::stof(parts.at(1));
			presetPos.F_CONE_20_F.w = std::stof(parts.at(2));
			presetPos.F_CONE_20_F.h = std::stof(parts.at(3));

			tempVal = ini.get("PRESET_FRONT").get("XT_20R");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.F_CONE_20_R.x = std::stof(parts.at(0));
			presetPos.F_CONE_20_R.y = std::stof(parts.at(1));
			presetPos.F_CONE_20_R.w = std::stof(parts.at(2));
			presetPos.F_CONE_20_R.h = std::stof(parts.at(3));
#pragma endregion
#pragma region FRONT - IGH Parse
			tempVal = ini.get("PRESET_FRONT").get("IGH_4XF");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.F_IGH_4X_F.x = std::stof(parts.at(0));
			presetPos.F_IGH_4X_F.y = std::stof(parts.at(1));
			presetPos.F_IGH_4X_F.w = std::stof(parts.at(2));
			presetPos.F_IGH_4X_F.h = std::stof(parts.at(3));

			tempVal = ini.get("PRESET_FRONT").get("IGH_4XR");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.F_IGH_4X_R.x = std::stof(parts.at(0));
			presetPos.F_IGH_4X_R.y = std::stof(parts.at(1));
			presetPos.F_IGH_4X_R.w = std::stof(parts.at(2));
			presetPos.F_IGH_4X_R.h = std::stof(parts.at(3));

			tempVal = ini.get("PRESET_FRONT").get("IGH_20F");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.F_IGH_20_F.x = std::stof(parts.at(0));
			presetPos.F_IGH_20_F.y = std::stof(parts.at(1));
			presetPos.F_IGH_20_F.w = std::stof(parts.at(2));
			presetPos.F_IGH_20_F.h = std::stof(parts.at(3));

			tempVal = ini.get("PRESET_FRONT").get("IGH_20R");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.F_IGH_20_R.x = std::stof(parts.at(0));
			presetPos.F_IGH_20_R.y = std::stof(parts.at(1));
			presetPos.F_IGH_20_R.w = std::stof(parts.at(2));
			presetPos.F_IGH_20_R.h = std::stof(parts.at(3));
#pragma endregion
#pragma region FRONT - CST Parse
			tempVal = ini.get("PRESET_FRONT").get("CST_4XF");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.F_CST_4X_F.x = std::stof(parts.at(0));
			presetPos.F_CST_4X_F.y = std::stof(parts.at(1));
			presetPos.F_CST_4X_F.w = std::stof(parts.at(2));
			presetPos.F_CST_4X_F.h = std::stof(parts.at(3));

			tempVal = ini.get("PRESET_FRONT").get("CST_4XR");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.F_CST_4X_R.x = std::stof(parts.at(0));
			presetPos.F_CST_4X_R.y = std::stof(parts.at(1));
			presetPos.F_CST_4X_R.w = std::stof(parts.at(2));
			presetPos.F_CST_4X_R.h = std::stof(parts.at(3));

			tempVal = ini.get("PRESET_FRONT").get("CST_20F");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.F_CST_20_F.x = std::stof(parts.at(0));
			presetPos.F_CST_20_F.y = std::stof(parts.at(1));
			presetPos.F_CST_20_F.w = std::stof(parts.at(2));
			presetPos.F_CST_20_F.h = std::stof(parts.at(3));

			tempVal = ini.get("PRESET_FRONT").get("CST_20R");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.F_CST_20_R.x = std::stof(parts.at(0));
			presetPos.F_CST_20_R.y = std::stof(parts.at(1));
			presetPos.F_CST_20_R.w = std::stof(parts.at(2));
			presetPos.F_CST_20_R.h = std::stof(parts.at(3));
#pragma endregion

			//REAR PRESET
#pragma region REAR - XT Parse
			tempVal = ini.get("PRESET_REAR").get("XT_4XF");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.R_CONE_4X_F.x = std::stof(parts.at(0));
			presetPos.R_CONE_4X_F.y = std::stof(parts.at(1));
			presetPos.R_CONE_4X_F.w = std::stof(parts.at(2));
			presetPos.R_CONE_4X_F.h = std::stof(parts.at(3));

			tempVal = ini.get("PRESET_REAR").get("XT_4XR");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.R_CONE_4X_R.x = std::stof(parts.at(0));
			presetPos.R_CONE_4X_R.y = std::stof(parts.at(1));
			presetPos.R_CONE_4X_R.w = std::stof(parts.at(2));
			presetPos.R_CONE_4X_R.h = std::stof(parts.at(3));

			tempVal = ini.get("PRESET_REAR").get("XT_20F");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.R_CONE_20_F.x = std::stof(parts.at(0));
			presetPos.R_CONE_20_F.y = std::stof(parts.at(1));
			presetPos.R_CONE_20_F.w = std::stof(parts.at(2));
			presetPos.R_CONE_20_F.h = std::stof(parts.at(3));

#pragma endregion
#pragma region REAR - IGH Parse
			tempVal = ini.get("PRESET_REAR").get("IGH_4XF");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.R_IGH_4X_F.x = std::stof(parts.at(0));
			presetPos.R_IGH_4X_F.y = std::stof(parts.at(1));
			presetPos.R_IGH_4X_F.w = std::stof(parts.at(2));
			presetPos.R_IGH_4X_F.h = std::stof(parts.at(3));

			tempVal = ini.get("PRESET_REAR").get("IGH_4XR");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.R_IGH_4X_R.x = std::stof(parts.at(0));
			presetPos.R_IGH_4X_R.y = std::stof(parts.at(1));
			presetPos.R_IGH_4X_R.w = std::stof(parts.at(2));
			presetPos.R_IGH_4X_R.h = std::stof(parts.at(3));

			tempVal = ini.get("PRESET_REAR").get("IGH_20F");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.R_IGH_20_F.x = std::stof(parts.at(0));
			presetPos.R_IGH_20_F.y = std::stof(parts.at(1));
			presetPos.R_IGH_20_F.w = std::stof(parts.at(2));
			presetPos.R_IGH_20_F.h = std::stof(parts.at(3));

#pragma endregion
#pragma region REAR - CST Parse
			tempVal = ini.get("PRESET_REAR").get("CST_4XF");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.R_CST_4X_F.x = std::stof(parts.at(0));
			presetPos.R_CST_4X_F.y = std::stof(parts.at(1));
			presetPos.R_CST_4X_F.w = std::stof(parts.at(2));
			presetPos.R_CST_4X_F.h = std::stof(parts.at(3));

			tempVal = ini.get("PRESET_REAR").get("CST_4XR");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.R_CST_4X_R.x = std::stof(parts.at(0));
			presetPos.R_CST_4X_R.y = std::stof(parts.at(1));
			presetPos.R_CST_4X_R.w = std::stof(parts.at(2));
			presetPos.R_CST_4X_R.h = std::stof(parts.at(3));

			tempVal = ini.get("PRESET_REAR").get("CST_20F");
			remove_all_whitespace(tempVal);
			parts = split_semicolon(tempVal);

			presetPos.R_CST_20_F.x = std::stof(parts.at(0));
			presetPos.R_CST_20_F.y = std::stof(parts.at(1));
			presetPos.R_CST_20_F.w = std::stof(parts.at(2));
			presetPos.R_CST_20_F.h = std::stof(parts.at(3));

#pragma endregion

			//Landout-VA
			//LDO_NEAR_X_THRESHOLD = std::stoi(ini.get("LANDOUT").get("NEAR_X_THRESHOLD"));
			//LDO_NEAR_Y_THRESHOLD = std::stoi(ini.get("LANDOUT").get("NEAR_Y_THRESHOLD"));

			//LDO_FAR_X_THRESHOLD = std::stoi(ini.get("LANDOUT").get("FAR_X_THRESHOLD"));
			//LDO_FAR_Y_THRESHOLD = std::stoi(ini.get("LANDOUT").get("FAR_Y_THRESHOLD"));

			//LDO_GANTRY_THRESHOLD = std::stoi(ini.get("LANDOUT").get("GANTRY_THRESHOLD_PCA"));
			//LDO_TROLLEY_THRESHOLD = std::stoi(ini.get("LANDOUT").get("TROLLEY_THRESHOLD_PCA"));

			LDO_NCOUNT = std::stoi(ini.get("LANDOUT").get("NCOUNT"));

			//CLPS-VA
			//CLPS_NEAR_X_THRESHOLD = std::stoi(ini.get("CLPS").get("NEAR_X_THRESHOLD"));
			//CLPS_NEAR_Y_THRESHOLD = std::stoi(ini.get("CLPS").get("NEAR_Y_THRESHOLD"));

			CLPS_NCOUNT = std::stoi(ini.get("CLPS").get("NCOUNT"));

			logMessage("PROCESS INI Configuration Complete!");
		}
		catch (std::exception& ex)
		{
			logMessage("Error: " + std::string(ex.what()) + " from reading INI/process.ini");
		}
		catch (...)
		{
			logMessage("Unknown error reading from INI/process.ini");
		}
	}
	else
	{
		logMessage("Failed to read INI/process.ini file");
	}
}
void parseIPINI(string path)
{
	mINI::INIFile file(path + "/INI/IP_LIST.ini");
	mINI::INIStructure ini;
	if (file.read(ini))
	{
		try
		{
			for (int i = 0; i < 4; i++)
			{
				auto ip = ini.get("FRONT_IP").get("LANE" + std::to_string(i + 1));
				if (ip != "") {
					FRONT_IP_ADDRESSES.push_back(ip);
				}

				ip = ini.get("Center_IP").get("LANE" + std::to_string(i + 1));
				if (ip != "") {
					CENTER_IP_ADDRESSES.push_back(ip);
				}

				ip = ini.get("REAR_IP").get("LANE" + std::to_string(i + 1));
				if (ip != "") {
					REAR_IP_ADDRESSES.push_back(ip);
				}
			}
			
			logMessage("IP INI Configuration Complete!");
		}
		catch (std::exception& ex)
		{
			logMessage("Error: " + std::string(ex.what()) + " from reading INI/IP_LIST.ini");
		}
		catch (...)
		{
			logMessage("Unknown error reading from INI/IP_LIST.ini");
		}
	}
	else
	{
		logMessage("Failed to read INI/IP_LIST.ini file");
	}
}

void reset_recvVariables()
{
	Offload_Cycle = false; Mount_Cycle = false;
	Job_Pos_B = false; Job_Pos_S = false; Job_Pos_C = false;
	SPRD_TWL_Locked = false; SPRD_TWL_Unlocked = false; SPRD_Landed = false;
	SPRD_20ft = false; SPRD_40ft = false; SPRD_45ft = false;

	Hoist_Stopped = false; Trolley_Stopped = false; Gantry_Stopped = false; Above_Safe_Height = false;
	NewJobPulse = false;

	CHS_20ft = false; CHS_40ft = false; CHS_45ft = false;
	CHS_XT = false; CHS_CST = false; CHS_IGH = false;

	enable_stream = false; enable_process = false; enable_logging = false; enable_save_logs = false;
	Connected_Sensor_Reboot = false; Application_Shutdown = false;

	logMessage("Resetting received variables from PLC");
}
void reset_sendVariables()
{
	model_loaded = false; model_initialized = false;
	sensor_connected = false; sensor_fault = false;
	enabled_stream = false; enabled_process = false; enabled_logging = false;

	logMessage("Resetting send variables to PLC");
}
void reset_jobVariables()
{
	current_job_info = JobInfo();
	current_process_results = ProcessResults();

	chassisRoiTracker.reset();
	clpsOffload3D.reset();

	//3D deviation output multiplex state
	DEV3D_RefLatch.reset();
	DEV3D_HoleLatch.reset();
	DEV3D_mount_hole_count = 0;
	DEV3D_mount_dev_phase = false;
	DEV3D_offload_dev_phase = false;

	VA_ConeInit3D.reset();
	VA_ConeInit2D = bbx();
	VA_ConeInit2D_Set = false;

	//VA
	Detected_XT = false; Detected_CST = false; Detected_IGH = false;
	Detected_Cntr = false; Detected_Chs = false;
	Landout_Detected_VA = false; CLPS_Detected_VA = false; CLPS_OK_Detected_VA = false;

	Detected_Chs_X_VA = 0; Detected_Chs_Y_VA = 0; Detected_Chs_Prob = 0;
	Detected_Cntr_X_VA = 0; Detected_Cntr_Y_VA = 0; Detected_Cntr_Prob = 0;

	//3D
	Detected_Chs_X_3D = 0; Detected_Chs_Y_3D = 0; Detected_Chs_Z_3D = 0;
	Detected_Cntr_X_3D = 0; Detected_Cntr_Y_3D = 0; Detected_Cntr_Z_3D = 0;

	Landout_Detected_3D = false; CLPS_Detected_3D = false; CLPS_OK_Detected_3D = false;

	//Deviations
	Dev_X_VA = 0; Dev_Y_VA = 0;
	Dev_X_3D = 0; Dev_Y_3D = 0; Dev_Z_3D = 0;

	LDO_intrim = bbx(); LDO_Base = bbx();
	int LDO_Base_Count = 0;
	bool LDO_Base_Set = false;

	int LDO_Current_Count = 0;
	int CLPS_Current_Count = 0;
	int CLPS_OK_Current_Count = 0;

	bbx CLPS_intrim;
	bbx CLPS_Base;
	int CLPS_Base_Count = 0;
	bool CLPS_Base_Set = false;

	bool landout_detected_pca = false;
	int landout_current_count_pca = 0;
	bool clps_detected_pca = false;
	int clps_current_count_pca = 0;
	bool clps_ok_detected_pca = false;
	int clps_ok_current_count_pca = 0;


	job_name = "";

	logMessage("Resetting job variables to PLC");
}
void old_reset_jobVariables()
{
	tCntr_x = -10000; tCntr_y = -10000; tCntr_prob = -10000;
	tCone_x = -10000; tCone_y = -10000; tCone_prob = -10000;
	Detected_XT = false; Detected_CST = false;

	devOut_x = -10000; devOut_y = -10000;
	devOut_LDO_x = -10000; devOut_LDO_y = -10000;
	devOut_pca_x = -10000; devOut_pca_y = -10000; devOut_pca_z = -10000;

	LDO_Current_Count = 0; CLPS_Current_Count = 0;

	landout_detected_pca = false;
	landout_current_count_pca = 0;
	clps_detected_pca = false;
	clps_current_count_pca = 0;
	clps_ok_detected_pca = false;
	clps_ok_current_count_pca = 0;

	clps_detected_pca_only = false;
	clps_ok_detected_pca_only = false;
	clps_detected_pca_only_count = 0;
	clps_ok_detected_pca_only_count = 0;

	bDetected_hole = false;
	bDetected_cone = false;

	AVG_CONE.reset();
	AVG_HOLE.reset();
	AVG_CONE_PCA.reset();

	PCA_intrim.x = 0; PCA_intrim.y = 0; PCA_intrim.z = 0;
	PCA_Base.x = 0; PCA_Base.y = 0; PCA_Base.z = 0;
	PCA_Base_Count = 0; PCA_Base_Set = false;

	Offload_Hole_Base = bbx();
	Offload_Hole_intrim = bbx();
	Offload_Hole_Base_Count = 0; Offload_Hole_Base_Set = false;

	LDO_intrim = bbx();
	LDO_Base = bbx();
	LDO_Base_Count = 0; LDO_Base_Set = false;

	CLPS_intrim = bbx();
	CLPS_Base = bbx();

	CLPS_Base_Count = 0; CLPS_Base_Set = false;

	landout_detected = false;
	clps_detected = false;
	clps_ok_detected = false;

	bCLPS_Detected_Out = false;
	bCLPS_OK_Detected_Out = false;
}

std::vector<std::string> getAllFiles(boost::filesystem::path const& root, std::string const& ext)
{
	std::vector<boost::filesystem::path> paths;
	std::vector<std::string> full_paths;

	if (boost::filesystem::exists(root) && boost::filesystem::is_directory(root))
	{
		for (auto const& entry : boost::filesystem::recursive_directory_iterator(root))
		{
			if (boost::filesystem::is_regular_file(entry) && entry.path().extension() == ext)
			{
				paths.emplace_back(entry.path().filename());
				full_paths.push_back(entry.path().string());
			}
		}
	}
	//Filenames start with the timestamp, so lexicographic = chronological. The directory
	//iterator's order is filesystem-dependent -- an explicit sort makes the frame order
	//(and ENABLE_MOUNT_REVERSE's reversal) deterministic.
	std::sort(full_paths.begin(), full_paths.end());
	return full_paths;
}

std::vector<std::string> ListSubDirectories(const std::string& directoryPath) {
	std::vector<std::string> subDirs;

	boost::filesystem::path dir(directoryPath);
	if (boost::filesystem::exists(dir) && boost::filesystem::is_directory(dir)) {
		// Iterate through the directory
		for (boost::filesystem::directory_entry& entry : boost::filesystem::directory_iterator(dir)) {
			if (boost::filesystem::is_directory(entry.status())) {
				subDirs.push_back(entry.path().string());
			}
		}
	}
	else {
		std::cerr << "Provided path is not a valid directory: " << directoryPath << std::endl;
	}

	return subDirs;
}

void selectTargetSensor()
{
	if (CURRENT_SENSOR_POSITION == "FRONT")
	{
		if (Job_Pos_S) //Use Center Sensor.
		{
			current_lane_ip = CENTER_IP_ADDRESSES[Lane_Number - 1];
			current_lane_camera = "CS" + std::to_string(Lane_Number);
			logMessage("Sensor ip from Front App: " + current_lane_ip);
		}
		else //Use Front Sensor.
		{
			current_lane_ip = FRONT_IP_ADDRESSES[Lane_Number - 1];
			current_lane_camera = "FS" + std::to_string(Lane_Number);
			logMessage("Sensor ip from Front App: " + current_lane_ip);
		}
	}
	else
	{
		if (Job_Pos_B) //Use Center Sensor
		{
			current_lane_ip = CENTER_IP_ADDRESSES[Lane_Number - 1];
			current_lane_camera = "CS" + std::to_string(Lane_Number);
			logMessage("Sensor ip from Rear App: " + current_lane_ip);
		}
		else //Use Rear Sensor.
		{
			current_lane_ip = REAR_IP_ADDRESSES[Lane_Number - 1];
			current_lane_camera = "RS" + std::to_string(Lane_Number);
			logMessage("Sensor ip from Rear App: " + current_lane_ip);
		}

	}
}

auto makeCommand(char* sendbuf) -> void {
	try
	{
		//Client Echo
		//send heartbeat
		//Server Heartbeat increment
		//heartbeat_ += 1;
		//if (heartbeat_ > 65535) heartbeat_ = 0;
		
		//std::cout << "Sending heartbeat: " << std::to_string(heartbeat_) << std::endl;
		std::bitset<16>heartbeat_bit(heartbeat_);
		const std::string temp_str = heartbeat_bit.to_string();
		std::bitset<8>first_half(temp_str.substr(0, temp_str.size() / 2));
		std::bitset<8>second_half(temp_str.substr(temp_str.size() / 2, temp_str.size()));

		sendbuf[0] = static_cast<char>(first_half.to_ulong());
		sendbuf[1] = static_cast<char>(second_half.to_ulong());

		std::bitset<8>op_byte_2(0);
		if (model_loaded) op_byte_2.set(0);
		if (model_initialized) op_byte_2.set(1);
		if (sensor_connected) op_byte_2.set(2);
		if (sensor_fault) op_byte_2.set(3);
		if (enabled_stream) op_byte_2.set(4);
		if (enabled_process) op_byte_2.set(5);
		if (enabled_logging) op_byte_2.set(6);
		sendbuf[2] = static_cast<char>(op_byte_2.to_ulong());

		//Detected_Chs_X_VA
		{
			std::bitset<16>short_bitArr(current_process_results.VA_Target_Chassis.x);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[4] = static_cast<char>(first_half.to_ulong());
			sendbuf[5] = static_cast<char>(second_half.to_ulong());
		}
		//Detected_Chs_Y_VA
		{
			std::bitset<16>short_bitArr(current_process_results.VA_Target_Chassis.y);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[6] = static_cast<char>(first_half.to_ulong());
			sendbuf[7] = static_cast<char>(second_half.to_ulong());
		}
		//Detected_Chs_Prob
		{
			std::bitset<16>short_bitArr(current_process_results.VA_Target_Chassis.prob);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[8] = static_cast<char>(first_half.to_ulong());
			sendbuf[9] = static_cast<char>(second_half.to_ulong());
		}
		//Detected_Cntr_X_VA
		{
			std::bitset<16>short_bitArr(current_process_results.VA_Target_Container.x);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[10] = static_cast<char>(first_half.to_ulong());
			sendbuf[11] = static_cast<char>(second_half.to_ulong());
		}
		//Detected_Cntr_Y_VA
		{
			std::bitset<16>short_bitArr(current_process_results.VA_Target_Container.y);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[12] = static_cast<char>(first_half.to_ulong());
			sendbuf[13] = static_cast<char>(second_half.to_ulong());
		}
		//Detected_Cntr_Prob
		{
			std::bitset<16>short_bitArr(current_process_results.VA_Target_Container.prob);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[14] = static_cast<char>(first_half.to_ulong());
			sendbuf[15] = static_cast<char>(second_half.to_ulong());
		}

		//VA Process Results
		std::bitset<8>op_byte_16(0);
		if (current_process_results.bDetected_XT) op_byte_16.set(0);
		if (current_process_results.bDetected_CST) op_byte_16.set(1);
		if (current_process_results.bDetected_IGH) op_byte_16.set(2);
		if (current_process_results.bDetected_Container) op_byte_16.set(3);
		if (current_process_results.bDetected_Chassis) op_byte_16.set(4);
		sendbuf[16] = static_cast<char>(op_byte_16.to_ulong());

		std::bitset<8>op_byte_17(0);
		//2026-07 fix: byte17 is THE combined output byte. It used to mix VA-only bits
		//(0,1) with a PCA-only bit (2), ignoring the combined b* results entirely --
		//the documented makeCommand bug. Per-pipeline values remain on byte30 (PCA).
		if (current_process_results.bLandOutDetected) op_byte_17.set(0);
		if (current_process_results.bClpsDetected) op_byte_17.set(1);
		if (current_process_results.bClpsOkDetected) op_byte_17.set(2);
		sendbuf[17] = static_cast<char>(op_byte_17.to_ulong());

		//Detected_Chs_X_3D
		{
			std::bitset<16>short_bitArr(current_process_results.PCA_Target_Chassis.x);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[18] = static_cast<char>(first_half.to_ulong());
			sendbuf[19] = static_cast<char>(second_half.to_ulong());
		}
		//Detected_Chs_Y_3D
		{
			std::bitset<16>short_bitArr(current_process_results.PCA_Target_Chassis.y);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[20] = static_cast<char>(first_half.to_ulong());
			sendbuf[21] = static_cast<char>(second_half.to_ulong());
		}
		//Detected_Chs_Z_3D
		{
			std::bitset<16>short_bitArr(current_process_results.PCA_Target_Chassis.z);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[22] = static_cast<char>(first_half.to_ulong());
			sendbuf[23] = static_cast<char>(second_half.to_ulong());
		}

		//Detected_Cntr_X_3D
		{
			std::bitset<16>short_bitArr(current_process_results.PCA_Target_Container.x);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[24] = static_cast<char>(first_half.to_ulong());
			sendbuf[25] = static_cast<char>(second_half.to_ulong());
		}
		//Detected_Cntr_Y_3D
		{
			std::bitset<16>short_bitArr(current_process_results.PCA_Target_Container.y);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[26] = static_cast<char>(first_half.to_ulong());
			sendbuf[27] = static_cast<char>(second_half.to_ulong());
		}
		//Detected_Cntr_Z_3D
		{
			std::bitset<16>short_bitArr(current_process_results.PCA_Target_Container.z);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[28] = static_cast<char>(first_half.to_ulong());
			sendbuf[29] = static_cast<char>(second_half.to_ulong());
		}

		//3D Process Results
		std::bitset<8>op_byte_30(0);
		if (current_process_results.PCA_isLandout) op_byte_30.set(0);
		if (current_process_results.PCA_isCLPS) op_byte_30.set(1);
		if (current_process_results.PCA_isCLPS_OK) op_byte_30.set(2);
		sendbuf[30] = static_cast<char>(op_byte_30.to_ulong());

		//Dev_X_VA
		{
			std::bitset<16>short_bitArr(current_process_results.VA_Devout_X);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[32] = static_cast<char>(first_half.to_ulong());
			sendbuf[33] = static_cast<char>(second_half.to_ulong());
		}
		//Dev_Y_VA
		{
			std::bitset<16>short_bitArr(current_process_results.VA_Devout_Y);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[34] = static_cast<char>(first_half.to_ulong());
			sendbuf[35] = static_cast<char>(second_half.to_ulong());
		}
		//Dev_X_3D
		{
			std::bitset<16>short_bitArr(current_process_results.PCA_Devout_X);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[36] = static_cast<char>(first_half.to_ulong());
			sendbuf[37] = static_cast<char>(second_half.to_ulong());
		}
		//Dev_Y_3D
		{
			std::bitset<16>short_bitArr(current_process_results.PCA_Devout_Y);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[38] = static_cast<char>(first_half.to_ulong());
			sendbuf[39] = static_cast<char>(second_half.to_ulong());
		}
		//Dev_Z_3D
		{
			std::bitset<16>short_bitArr(current_process_results.PCA_Devout_Z);
			const std::string temp_str = short_bitArr.to_string();
			std::bitset<8>first_half(temp_str.substr(0, 8));
			std::bitset<8>second_half(temp_str.substr(8, 16));
			sendbuf[40] = static_cast<char>(first_half.to_ulong());
			sendbuf[41] = static_cast<char>(second_half.to_ulong());
		}		
	}
	catch (std::exception& ex)
	{
		logMessage("Error in making command : " + std::string(ex.what()));
	}
	catch (...)
	{
		logMessage("Unknown error in making command");
	}
}

std::string makeJobFolderName()
{
	std::string job_info = print_time() + std::string("_");

	//BayNumber first
	job_info.append("B" + std::to_string(Lane_Number) + "_");

	if (Offload_Cycle) job_info.append("Offload_");
	else if (Mount_Cycle) job_info.append("Mount_");
	else job_info.append("None_");

	if (Job_Pos_B) job_info.append("Big_");
	else if (Job_Pos_S) job_info.append("Small_");
	else if (Job_Pos_C) job_info.append("Center_");
	else job_info.append("None_");

	if (CHS_XT) job_info.append("XT_");
	else if (CHS_CST) job_info.append("CST_");
	else if (CHS_IGH) job_info.append("IGH_");
	else job_info.append("None_");

	if (CHS_20ft) job_info.append("CHS20_");
	else if (CHS_40ft) job_info.append("CHS40_");
	else if (CHS_45ft) job_info.append("CHS45_");
	else job_info.append("None_");

	if (SPRD_20ft) job_info.append("SPRD20_");
	else if (SPRD_40ft) job_info.append("SPRD40_");
	else if (SPRD_45ft) job_info.append("SPRD45_");
	else job_info.append("None_");

	job_info.append(CURRENT_SENSOR_POSITION);

	//CraneID add.
	job_info.append("_" + std::to_string(Crane_ID));

	return job_info;
}

void parseCommand(char recvbuf[])
{
	try
	{
		char* recvBuf = &recvbuf[0];
		//heartbeat section
		{
			std::bitset<8> byte0(recvBuf[0]);
			std::bitset<8> byte1(recvBuf[1]);
			std::bitset<16> temp_heartbeat_ = concat(byte0, byte1);
			heartbeat_ = temp_heartbeat_.to_ulong();
			//std::cout << "Received heartbeat: " << std::to_string(heartbeat_) << std::endl;
		}

		{
			std::bitset<8> byte2(recvBuf[2]);
			std::bitset<8> byte3(recvBuf[3]);
			std::bitset<16> temp_CraneID = concat(byte2, byte3);
			Crane_ID = temp_CraneID.to_ulong();
		}


		{
			std::bitset<8> byte8(recvBuf[8]);
			std::bitset<8> byte9(recvBuf[9]);
			std::bitset<16> temp_Lane_Number = concat(byte8, byte9);
			Lane_Number = temp_Lane_Number.to_ulong();
		}


		{
			std::bitset<8> byte4(recvBuf[4]);
			Offload_Cycle = (bool)byte4[0];
			Mount_Cycle = (bool)byte4[1];

			Job_Pos_B = (bool)byte4[2];
			Job_Pos_S = (bool)byte4[3];
			Job_Pos_C = (bool)byte4[4];

			bool temp_twl_l = (bool)byte4[5];
			if (temp_twl_l != SPRD_TWL_Locked)
			{
				if (enabled_process)
				{
					save_trigger_by_TWL_Locked = true;
					jobLogMessage("Save Triggered by SPRD TWL Locked");
				}
			}
			SPRD_TWL_Locked = temp_twl_l;

			SPRD_TWL_Unlocked = (bool)byte4[6];
			bool temp = (bool)byte4[7];
			if (temp != SPRD_Landed)
			{
				if (enabled_process)
				{
					save_trigger_by_landed = true;
					jobLogMessage("Save Triggered by SPRD Landed");
				}
			}
			SPRD_Landed = temp;

			std::bitset<8> byte5(recvBuf[5]);
			SPRD_20ft = (bool)byte5[0];
			SPRD_40ft = (bool)byte5[1];
			SPRD_45ft = (bool)byte5[2];

			Hoist_Stopped = (bool)byte5[3];
			Trolley_Stopped = (bool)byte5[4];
			Gantry_Stopped = (bool)byte5[5];
			Above_Safe_Height = (bool)byte5[6];


			std::bitset<8> byte6(recvBuf[6]);
			CHS_20ft = (bool)byte6[0];
			CHS_40ft = (bool)byte6[1];
			CHS_45ft = (bool)byte6[2];

			CHS_XT = (bool)byte6[3];
			CHS_CST = (bool)byte6[4];
			CHS_IGH = (bool)byte6[5];
			
			bool tmp_newjobpulse = (bool)byte5[7];
			if (tmp_newjobpulse && tmp_newjobpulse != NewJobPulse)
			{
				reset_jobVariables();
				
				//1. Select target sensor
				selectTargetSensor();
				//2. Job Folder Name -- Logging
				job_name = makeJobFolderName();
				logMessage("Job: " + job_name);
				//3. Update Job Info

				std::string job_pos = "Center";
				if (Job_Pos_B) job_pos = "Big";
				else if (Job_Pos_S) job_pos = "Small";

				std::string chassis_type = "XT";
				if (CHS_CST) chassis_type = "CST";
				else if (CHS_IGH) chassis_type = "IGH";

				int chassis_len = 40;
				if (CHS_20ft) chassis_len = 20;
				else if (CHS_45ft) chassis_len = 45;

				int job_size = 40;
				if (SPRD_20ft) job_size = 20;
				else if (SPRD_45ft) job_size = 45;

				bbx presetTarget;
				MAX_MIN_RANGE presetTargetX;
				//PRESET SELECT
				if (CURRENT_SENSOR_POSITION == "FRONT")
				{
					if (chassis_type == "XT")
					{
						if (chassis_len == 45 && job_size == 40) { presetTarget = presetPos.F_CONE_4X_R; presetTargetX = presetPos.F_XT_4XR_RNG; }
						else if (chassis_len == 45 && job_pos == "Small") { presetTarget = presetPos.F_CONE_20_R; presetTargetX = presetPos.F_XT_20R_RNG; }
						else if (chassis_len == 40 && job_pos == "Small") { presetTarget = presetPos.F_CONE_20_F; presetTargetX = presetPos.F_XT_20F_RNG; }
						else
						{
							presetTarget = presetPos.F_CONE_4X_F;
							presetTargetX = presetPos.F_XT_4XF_RNG;
						}
					}
					else if (chassis_type == "IGH")
					{
						if (chassis_len == 45 && job_size == 40) { presetTarget = presetPos.F_IGH_4X_R; presetTargetX = presetPos.F_IGH_4XR_RNG; }
						else if (chassis_len == 45 && job_pos == "Small") { presetTarget = presetPos.F_IGH_20_R; presetTargetX = presetPos.F_IGH_20R_RNG;}
						else if (chassis_len == 40 && job_pos == "Small") { presetTarget = presetPos.F_IGH_20_F; presetTargetX = presetPos.F_IGH_20F_RNG;}
						else
						{
							presetTarget = presetPos.F_IGH_4X_F;
							presetTargetX = presetPos.F_IGH_4XF_RNG;
						}
					}
					else if (chassis_type == "CST")
					{
						if (chassis_len == 45 && job_size == 40) { presetTarget = presetPos.F_CST_4X_R; presetTargetX = presetPos.F_CST_4XR_RNG; }
						else if (chassis_len == 45 && job_pos == "Small") { presetTarget = presetPos.F_CST_20_R; presetTargetX = presetPos.F_CST_20R_RNG; }
						else if (chassis_len == 40 && job_pos == "Small") { presetTarget = presetPos.F_CST_20_F; presetTargetX = presetPos.F_CST_20F_RNG; }
						else
						{
							presetTarget = presetPos.F_CST_4X_F;
							presetTargetX = presetPos.F_CST_4XF_RNG;
						}
					}
				}
				else
				{
					if (chassis_type == "XT")
					{
						if (job_pos == "Big") { presetTarget = presetPos.R_CONE_20_F; presetTargetX = presetPos.R_XT_20F_RNG; }
						else if (chassis_len == 40) { presetTarget = presetPos.R_CONE_4X_F; presetTargetX = presetPos.R_XT_4XF_RNG; }
						else { presetTarget = presetPos.R_CONE_4X_R; presetTargetX = presetPos.R_XT_4XR_RNG; }
					}
					else if (chassis_type == "IGH")
					{
						if (job_pos == "Big") { presetTarget = presetPos.R_IGH_20_F; presetTargetX = presetPos.R_IGH_20F_RNG; }
						else if (chassis_len == 40) { presetTarget = presetPos.R_IGH_4X_F; presetTargetX = presetPos.R_IGH_4XF_RNG; }
						else { presetTarget = presetPos.R_IGH_4X_R; presetTargetX = presetPos.R_IGH_4XR_RNG; }
					}
					else if (chassis_type == "CST")
					{
						if (job_pos == "Big") { presetTarget = presetPos.R_CST_20_F; presetTargetX = presetPos.R_CST_20F_RNG; }
						else if (chassis_len == 40) { presetTarget = presetPos.R_CST_4X_F; presetTargetX = presetPos.R_CST_4XF_RNG; }
						else { presetTarget = presetPos.R_CST_4X_R; presetTargetX = presetPos.R_CST_4XR_RNG; }
					}
				}

				current_job_info.set(Mount_Cycle, Offload_Cycle,
					job_pos, chassis_type, chassis_len, job_size, presetTarget, presetTargetX);

				logMessage("New Job Pulse Detected!");
			}
			NewJobPulse = tmp_newjobpulse;

		}
		
		{
			std::bitset<8> byte18(recvBuf[18]);
			enable_stream = (bool)byte18[0];
			
			bool tempVal = (bool)byte18[1];
			if (tempVal && tempVal != enable_process)
			{
				//rising edge.
				logMessage("Enable Process Detected!");

				//streaming thread will be triggered from tmini_ctrl thread.
				
				//Create necessary job result folders and specify names.
				job_result_folder_name = std::string("JOB_LOG/");
				createDirectory_ifexists(job_result_folder_name);
				job_result_folder_name += std::string("Lane") + std::to_string(target_lane_number) + "/";
				createDirectory_ifexists(job_result_folder_name);

				
				job_result_folder_name.append(job_name);
				job_result_folder_name.append("_JOBLOG");
				createDirectory_ifexists(job_result_folder_name);

				//images, pointcloud
				//createDirectory_ifexists(job_result_folder_name + "/Image");
				//createDirectory_ifexists(job_result_folder_name + "/Depth");
				//createDirectory_ifexists(job_result_folder_name + "/Image_RGB");

				//job info log.
				//job_result_log_file_name = job_result_folder_name + "/" + job_name + "_job_info.log";
				
				//allow job result log thread to resume?
				//logMessage(job_result_log_file_name);
				//jobLogMessage("Testing if Job Log is created on each job start.");
				
				enabled_process = true;

				std::unique_lock<std::mutex> lock(mutex_tmini_ctrl);
				tmini_ctrl_flag = true; //wake tmini ctrl thread.
				
				cond_tmini_ctrl.notify_one();

				proc_running.store(true);
				//Wake the processing thread.
				processFlag = true;
				cond_processing.notify_one();
				
				trigger_log_delete = false;
				trigger_terminate = false;

				logMessage("Process Enabled with job name: " + job_name);
			}
			else if (!tempVal && tempVal != enable_process)
			{
				logMessage("Disable Process Detected!");
				
				processFlag = false;

				tmini_ctrl_flag = false;
				//streaming off.
				tmini_data_stream_flag = false;

				enabled_process = false;
				
				if (!enable_save_logs && ENABLE_SAVE_LOG_BIT)
				{
					logMessage("Delete Log Triggered.");
					trigger_log_delete = true;
					//need to delete "saved" logs.
				}

				trigger_terminate = true;

				//wake data save thread
				//saveFlag = true;
				//cond_logging.notify_one();
			}
			enable_process = tempVal;

			//sensor re-connection check.
			if (enable_process && false)
			{
				if (!sensor_connected) //comm fault.
				{
					auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - sensor_last_attempted_time).count();
					if (ts > 10000) //10 seconds
					{
						logMessage("Sensor Connection Fault Detected! Attempting to reconnect...");
						sensor_connected = false;
						sensor_fault = false;
						{
							std::unique_lock<std::mutex> lock(mutex_tmini_ctrl);
							tmini_ctrl_flag = true; //wake tmini ctrl thread.
						}
						cond_tmini_ctrl.notify_one();
					}
				}

				if (sensor_fault)
				{
					if (sensor_connected)
					{
						//Disconnect from current sensor and try to reconnect.
						; 
					}
				}

			}

			if (ENABLE_SAVE_LOG_BIT && trigger_log_delete && !enable_logging)
			{
				logMessage("Deleting save logs as per command.");

				//stop any files on queue
				if (tsq.size() > 0)
				{
					tsq.clear();
					logMessage("Cleared Thread Safe Queue.");
				}

				//delete folder recursively.
				//saveDirName
				try
				{
					if (std::filesystem::exists(saveDirName)) {
						std::filesystem::remove_all(saveDirName);  // Recursively removes the directory and its contents
						logMessage("Directory removed: " + saveDirName);
					}
					else
					{
						logMessage("Directory does not exist: " + saveDirName);
					}
				}
				catch (const std::filesystem::filesystem_error& e) {
					logMessage("Filesystem error: " + std::string(e.what()));
				}

				trigger_log_delete = false;
			}

			tempVal = (bool)byte18[2];
			if (tempVal && tempVal != enable_logging)
			{
				//rising edge.
				logMessage("Enable Logging Detected!");

				saveDirName = std::string("SAVE/");
				//saveDirName += std::string("Lane") + std::to_string(target_lane_number) + "/";
				createDirectory_ifexists(saveDirName);

				//auto job_info = makeJobFolderName();
				saveDirName.append(job_name);
				createDirectory_ifexists(saveDirName);
				createDirectory_ifexists(saveDirName + "/Image");
				createDirectory_ifexists(saveDirName + "/Depth");
				
				enabled_logging = true;

				//wake data save thread
				saveFlag = true;
				cond_logging.notify_one();

				enable_logging = true;

			}
			else if (!tempVal && tempVal != enable_logging)
			{
				logMessage("Disable Logging Detected!");
				//tmini_ctrl_flag = false;
				//streaming off.
				//tmini_data_stream_flag = false;

				saveFlag = false;
				enable_logging = false;
				enabled_logging = false;
			}

			enable_save_logs = (bool)byte18[4];

			std::bitset<8> byte19(recvBuf[19]);
			
			bool tVal = (bool)byte19[0];
			if (tVal && tVal != Connected_Sensor_Reboot)
			{
				logMessage("Sensor Reboot Command Detected!");
			}
			Connected_Sensor_Reboot = tVal;
			
			tVal = (bool)byte19[1];
			if (tVal && tVal != Application_Shutdown)
			{
				logMessage("Application Shutdown Command Detected!");

				if (ENALBE_RESTART_APPLICATION)
				{
					if (trigger_terminate && tsq.size() == 0)
					{
						logMessage("Triggered Application Termination!");

						//Nothing to save.	
						proc_running.store(false);
						cond_processing.notify_one();

						logging_running.store(false);
						cond_logging.notify_one();

						tmini_ctrl_running.store(false);
						cond_tmini_ctrl.notify_one();

						tmini_data_stream_running.store(false);
						cond_tmini_data_stream.notify_one();

						socket_running.store(false);

						logMessage("Threads to be terminated.");
					}
				}

			}
			Application_Shutdown = tVal;

		}
		
		return;
	}
	catch (std::exception& ex)
	{
		logMessage("Error: " + std::string(ex.what()) + " from parsing command.");
		return;
	}
	catch (...)
	{
		logMessage("Unknown error from parsing command.");
		return;
	}
}

#pragma region TensorRT - 10.48

struct LetterboxInfo {
	float scale = 1.f;
	int pad_x = 0;
	int pad_y = 0;
	int new_w = 0;
	int new_h = 0;
};

// Letterbox to (dst_w, dst_h), pads with 114 like YOLO
static cv::Mat letterbox(const cv::Mat& src, int dst_w, int dst_h, LetterboxInfo& info) {
	int src_w = src.cols, src_h = src.rows;
	float r = std::min((float)dst_w / src_w, (float)dst_h / src_h);

	info.scale = r;
	info.new_w = (int)std::round(src_w * r);
	info.new_h = (int)std::round(src_h * r);

	cv::Mat resized;
	cv::resize(src, resized, cv::Size(info.new_w, info.new_h), 0, 0, cv::INTER_LINEAR);

	int pad_w = dst_w - info.new_w;
	int pad_h = dst_h - info.new_h;

	info.pad_x = pad_w / 2;
	info.pad_y = pad_h / 2;

	cv::Mat out(dst_h, dst_w, src.type(), cv::Scalar(114, 114, 114));
	resized.copyTo(out(cv::Rect(info.pad_x, info.pad_y, info.new_w, info.new_h)));
	return out;
}

// BGR cv::Mat -> RGB float32 NCHW normalized
static void matToCHW(const cv::Mat& img_bgr_letterboxed, std::vector<float>& chw) {
	cv::Mat rgb;
	cv::cvtColor(img_bgr_letterboxed, rgb, cv::COLOR_BGR2RGB);

	cv::Mat f;
	rgb.convertTo(f, CV_32F, 1.0 / 255.0);

	int H = f.rows, W = f.cols;
	chw.resize(3ull * H * W);

	// HWC -> CHW
	for (int y = 0; y < H; ++y) {
		const cv::Vec3f* row = f.ptr<cv::Vec3f>(y);
		for (int x = 0; x < W; ++x) {
			const cv::Vec3f& p = row[x]; // RGB
			chw[0ull * H * W + (size_t)y * W + x] = p[0];
			chw[1ull * H * W + (size_t)y * W + x] = p[1];
			chw[2ull * H * W + (size_t)y * W + x] = p[2];
		}
	}
}

struct RectInfo {
	cv::Rect2f box;   // x,y,w,h in original image coords
	int class_id = -1;
	float score = 0.f;
};

static float IoU(const cv::Rect2f& a, const cv::Rect2f& b) {
	float interArea = (a & b).area();
	float unionArea = a.area() + b.area() - interArea;
	return unionArea > 0.f ? interArea / unionArea : 0.f;
}

static std::vector<RectInfo> nms(const std::vector<RectInfo>& dets, float iouThresh) {
	std::vector<int> idx(dets.size());
	std::iota(idx.begin(), idx.end(), 0);

	std::sort(idx.begin(), idx.end(), [&](int i, int j) {
		return dets[i].score > dets[j].score;
		});

	std::vector<RectInfo> out;
	std::vector<bool> suppressed(dets.size(), false);

	for (size_t _i = 0; _i < idx.size(); ++_i) {
		int i = idx[_i];
		if (suppressed[i]) continue;
		out.push_back(dets[i]);

		for (size_t _j = _i + 1; _j < idx.size(); ++_j) {
			int j = idx[_j];
			if (suppressed[j]) continue;

			// class-aware NMS (usually desired for YOLO)
			if (dets[i].class_id != dets[j].class_id) continue;

			if (IoU(dets[i].box, dets[j].box) > iouThresh) {
				suppressed[j] = true;
			}
		}
	}
	return out;
}

static std::vector<RectInfo> decodeYoloV8_1xCxN(
	const float* out,          // float pointer to output0
	int C, int N,              // C=8, N=8400
	int numClasses,            // 4 in your case
	float confThresh,
	const LetterboxInfo& lb,
	int origW, int origH
) {
	std::vector<RectInfo> dets;
	dets.reserve(256);

	// out is arranged as [C, N] for your dims [1,C,N]
	// index: out[c*N + i]
	for (int i = 0; i < N; ++i) {
		float x = out[0 * N + i];
		float y = out[1 * N + i];
		float w = out[2 * N + i];
		float h = out[3 * N + i];

		// class score = max over classes
		int bestCls = -1;
		float bestScore = 0.f;
		for (int c = 0; c < numClasses; ++c) {
			float s = out[(4 + c) * N + i];
			if (s > bestScore) {
				bestScore = s;
				bestCls = c;
			}
		}

		if (bestScore < confThresh) continue;

		// xywh -> x1y1x2y2 in letterboxed input coords
		float x1 = x - w * 0.5f;
		float y1 = y - h * 0.5f;
		float x2 = x + w * 0.5f;
		float y2 = y + h * 0.5f;

		// Undo letterbox: (coord - pad) / scale
		x1 = (x1 - lb.pad_x) / lb.scale;
		y1 = (y1 - lb.pad_y) / lb.scale;
		x2 = (x2 - lb.pad_x) / lb.scale;
		y2 = (y2 - lb.pad_y) / lb.scale;

		// Clip to original image
		x1 = std::max(0.f, std::min(x1, (float)(origW - 1)));
		y1 = std::max(0.f, std::min(y1, (float)(origH - 1)));
		x2 = std::max(0.f, std::min(x2, (float)(origW - 1)));
		y2 = std::max(0.f, std::min(y2, (float)(origH - 1)));

		float bw = x2 - x1;
		float bh = y2 - y1;
		if (bw <= 1.f || bh <= 1.f) continue;

		RectInfo d;
		d.box = cv::Rect2f(x1, y1, bw, bh);
		d.class_id = bestCls;
		d.score = bestScore;
		dets.push_back(d);
	}

	return dets;
}

#pragma endregion

auto controlConnected = false;
auto dataConnected = false;

struct cloud_data
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr pc_ptr = NULL;
	int pointSize = 0;

	std::string chassisType = "";

	pcl::PointXYZ center = pcl::PointXYZ(0, 0, 0);

	int max_x = 0, min_x = 0;
	int max_y = 0, min_y = 0;
	int max_z = 0, min_z = 0;

	int len_x = 0, len_y = 0, len_z = 0;
	int len_x_min_z = 0;

	int max_x_at_max_y = 0;
	int min_x_at_max_y = 0;

	pcl::PointXYZ max_x_at_min_z = pcl::PointXYZ(0, 0, 0);
	pcl::PointXYZ min_x_at_min_z = pcl::PointXYZ(0, 0, 0);

	int midBreakPoint = 0;

	bool valid = false;
};

/*
* Point Cloud Operations
*/

pcl::PointCloud<pcl::PointXYZ>::Ptr singlePassThrough(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	float minX, float maxX,
	float minY, float maxY,
	float minZ, float maxZ,
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcResult)
{
	pcResult->clear();
	for (const auto& point : pcInput->points)
	{
		if (point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY && point.z >= minZ && point.z <= maxZ)
			pcResult->points.push_back(point);
	}

	pcResult->width = pcResult->points.size();
	pcResult->height = 1;

	return pcResult;
}

auto get_max_min_xyz(const pcl::PointCloud<pcl::PointXYZ>::Ptr input, int& minX, int& maxX, int& minY, int& maxY, int& minZ, int& maxZ) -> void
{
	int local_maxZ = -10000;
	int local_minZ = 10000;
	int local_maxX = -10000;
	int local_minX = 10000;
	int local_maxY = -10000;
	int local_minY = 10000;
	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).z;
		if (val > local_maxZ)
			local_maxZ = val;
		if (val < local_minZ)
			local_minZ = val;

		auto valX = (*it).x;
		if (valX > local_maxX)
			local_maxX = valX;
		if (valX < local_minX)
			local_minX = valX;

		auto valY = (*it).y;
		if (valY > local_maxY)
			local_maxY = valY;
		if (valY < local_minY)
			local_minY = valY;
	}

	maxZ = local_maxZ;
	minZ = local_minZ;
	maxX = local_maxX;
	minX = local_minX;
	maxY = local_maxY;
	minY = local_minY;
}
/**
 * @brief Find valid min or max value for a specified axis in a single pass
 *
 * @param input The input point cloud
 * @param axis Which axis to process ('x', 'y', or 'z')
 * @param find_max If true, finds max value; if false, finds min value
 * @param window_size Size of the sliding window for density detection (in mm)
 * @param min_points Minimum number of points required to consider a window valid
 * @param outlier_min Lower bound for outlier rejection
 * @param outlier_max Upper bound for outlier rejection
 * @return int The valid min or max value for the specified axis
 */
auto get_valid_axis_value(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
	char axis = 'y',
	bool find_max = false,
	float window_size = 40.0f,
	int min_points = 30,
	float outlier_min = -10000.0f,
	float outlier_max = 10000.0f) -> int
{
	if (input->empty()) return -1;

	// 1. Collect axis values in a single pass
	std::vector<float> values;
	values.reserve(input->size());
	float global_min = outlier_max;
	float global_max = outlier_min;

	// Single pass through the point cloud
	for (const auto& point : input->points)
	{
		float value;
		// Select the appropriate coordinate based on axis parameter
		switch (axis) {
		case 'x': value = point.x; break;
		case 'z': value = point.z; break;
		case 'y':
		default: value = point.y; break;
		}

		if (value > outlier_min && value < outlier_max)
		{
			values.push_back(value);
			if (value < global_min) global_min = value;
			if (value > global_max) global_max = value;
		}
	}

	if (values.empty()) return -2;

	// 2. Sort values for efficient window processing
	std::sort(values.begin(), values.end());

	// Default return value based on what we're looking for
	int default_return = find_max ? static_cast<int>(global_max) : static_cast<int>(global_min);
	//logMessage("Global Max: " + std::to_string(global_max) + " , Global Min: " + std::to_string(global_min) + " , values.size: " + std::to_string(values.size()));

	// 3. Use sliding window to find dense region
	if (find_max) {
		// Find MAXIMUM: start from the end and work backwards
		size_t window_start = values.size() - 1;
		size_t window_end = values.size() - 1;

		while (window_start > 0) {
			// Expand window until it exceeds window_size
			while (window_start > 0 &&
				values[window_end] - values[window_start] <= window_size) {
				window_start--;
			}

			// Adjust window_start if we went too far
			if (values[window_end] - values[window_start] > window_size) {
				window_start++;
			}

			// Check if current window has enough points
			if (window_end - window_start + 1 >= min_points) {
				// Calculate average of points in this window
				float sum = 0.0f;
				for (size_t i = window_start; i <= window_end; ++i) {
					sum += values[i];
				}
				return static_cast<int>(sum / (window_end - window_start + 1));
			}

			// Move window end backward
			if (window_end > 0) window_end--;
		}
	}
	else {
		// Find MINIMUM: start from the beginning and work forwards
		size_t window_start = 0;
		size_t window_end = 0;

		while (window_end < values.size()) {
			// Expand window until it exceeds window_size
			while (window_end < values.size() &&
				values[window_end] - values[window_start] <= window_size) {
				window_end++;
			}

			// Check if current window has enough points
			if (window_end - window_start >= min_points) {
				// Calculate average of points in this window
				float sum = 0.0f;
				for (size_t i = window_start; i < window_end; ++i) {
					sum += values[i];
					//logMessage("i: " + std::to_string(i) + " value: " + std::to_string(values[i]));
				}
				//logMessage("Sum: " + std::to_string(sum) + " , Window End: " + std::to_string(window_end) + " , Window Start: " + std::to_string(window_start));
				return static_cast<int>(sum / (window_end - window_start));
			}

			// Move window start forward
			window_start++;
		}
	}

	// Fallback to global min/max if no valid window found
	return default_return;
}



bool pc_passThrough(pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	float minV, float maxV, std::string axis, pcl::PointCloud<pcl::PointXYZ>::Ptr pcResult) {

	pcl::PassThrough<pcl::PointXYZ> pass;
	auto proc = false;
	auto minVal = min(minV, maxV);
	auto maxVal = max(minV, maxV);

	if (abs(maxVal - minVal) > DBL_EPSILON)
	{
		pass.setInputCloud(pcInput);
		pass.setFilterFieldName(axis);
		pass.setFilterLimits(minVal, maxVal); //min, max
		pass.filter(*pcResult);
		proc = true;
	}

	return proc;
}

bool pc_passThrough(
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	float minX, float maxX,
	float minY, float maxY,
	float minZ, float maxZ,
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcResult)
{
	pcl::PassThrough<pcl::PointXYZ> pass;
	pcl::PointCloud<pcl::PointXYZ>::Ptr retX(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr retY(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr retZ(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr stepInput(new pcl::PointCloud<pcl::PointXYZ>);

	double minVal = 0.0;
	double maxVal = 0.0;
	bool proc = false;

	stepInput = pcInput;
	minVal = min(minX, maxX);
	maxVal = max(minX, maxX);
	if (abs(maxVal - minVal) > DBL_EPSILON)
	{
		pass.setInputCloud(stepInput);
		pass.setFilterFieldName("x");
		pass.setFilterLimits(minVal, maxVal); //min, max
		pass.filter(*retX);
		proc = true;
	}
	stepInput = proc ? retX : stepInput; proc = false;

	minVal = min(minY, maxY);
	maxVal = max(minY, maxY);
	if (abs(maxVal - minVal) > DBL_EPSILON)
	{
		pass.setInputCloud(stepInput);
		pass.setFilterFieldName("y");
		pass.setFilterLimits(minVal, maxVal); //min, max
		pass.filter(*retY);
		proc = true;
	}
	stepInput = proc ? retY : stepInput; proc = false;


	minVal = min(minZ, maxZ);
	maxVal = max(minZ, maxZ);
	if (abs(maxVal - minVal) > DBL_EPSILON)
	{
		pass.setInputCloud(stepInput);
		pass.setFilterFieldName("z");
		pass.setFilterLimits(minVal, maxVal); //min, max
		pass.filter(*retZ);
		proc = true;
	}
	stepInput = proc ? retZ : stepInput; proc = false;

	//마지막은 없으면 없는데로 넣어줘야 한다.
	*pcResult = *stepInput;//

	return true;
}

bool pc_VoxelDown(
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	float leafX, float leafY, float leafZ,
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcResult)
{
	if (pcInput->empty())
		return false;

	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(pcInput);
	sor.setLeafSize(leafX, leafY, leafZ);// 0.01 : 1cm
	sor.filter(*pcResult);

	return true;
}

bool pc_NoiseFilter(
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	int MeanK_neighberCount,
	float StddevMulThresh,
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcResult)
{
	if (pcInput->empty())
		return false;

	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud(pcInput);
	sor.setMeanK(MeanK_neighberCount); //이웃한 점의 수
	sor.setStddevMulThresh(StddevMulThresh); //노이즈 표준편차 임계값
	sor.filter(*pcResult);

	//sor.setNegative(true);
	//sor.filter(*pcOutlier);

	return true;
}

bool pc_RadiusFilter(
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	float radiusSearch,
	float minNeighbors,
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcResult
)
{
	if (pcInput->empty())
		return false;

	pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
	// build the filter
	outrem.setInputCloud(pcInput);
	outrem.setRadiusSearch(radiusSearch);
	outrem.setMinNeighborsInRadius(minNeighbors);
	outrem.setKeepOrganized(true);
	// apply filter
	outrem.filter(*pcResult);

	return true;
}

bool pc_CubeExtractFilter(
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	Eigen::Vector3f vCubeCenter, float lengthX, float lengthY, float lengthZ,
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcResult)
{
	if (pcInput->size() <= 0)
		return false;

	float min_x = vCubeCenter.x() - lengthX;
	float max_x = vCubeCenter.x() + lengthX;

	float min_y = vCubeCenter.y() - lengthY;
	float max_y = vCubeCenter.y() + lengthY;

	float min_z = vCubeCenter.z() - lengthZ;
	float max_z = vCubeCenter.z() + lengthZ;

	return pc_passThrough(
		pcInput,
		min_x, max_x,
		min_y, max_y,
		min_z, max_z, pcResult);
}

bool pc_CountComp(const pcl::PointCloud<pcl::PointXYZ>::Ptr lhs, const pcl::PointCloud<pcl::PointXYZ>::Ptr rhs) {

	return lhs->points.size() < rhs->points.size();
}

bool pc_Clustering(
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	int sizeMin, int sizeMax, float fTolerance,
	std::list<pcl::PointCloud<pcl::PointXYZ>::Ptr>* listOut)
{
	if (pcInput->size() <= 0)
		return false;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_filtered(new pcl::PointCloud < pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_out(new pcl::PointCloud < pcl::PointXYZ>());

	cloud_filtered->resize(pcInput->size());

	pcl::copyPointCloud(*pcInput, *cloud_filtered);


	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud_filtered);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setInputCloud(cloud_filtered);
	ec.setClusterTolerance(fTolerance); //0.02 - 2cm
	ec.setMinClusterSize(sizeMin); // 100 - 100
	ec.setMaxClusterSize(sizeMax); // 25000 - 25000
	ec.setSearchMethod(tree);
	ec.extract(cluster_indices);

	int sum = 0;
	int k = 0;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
		{
			int idx = *pit;
			cloud_cluster->points.push_back(cloud_filtered->points[idx]); //*
		}
		//OutputDebugString("\n");
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;


		listOut->push_back(cloud_cluster);

		//sum += cloud_cluster->points.size();
	}
	listOut->sort(pc_CountComp);
	return true;
}

struct greater_pointXYZ_x
{
	inline bool operator() (const pcl::PointXYZ left, const pcl::PointXYZ right)
	{
		return (left.x > right.x);
	}
};
struct greater_pointXYZ_z
{
	inline bool operator() (const pcl::PointXYZ left, const pcl::PointXYZ right)
	{
		return (left.z > right.z);
	}
};

pcl::PointCloud<pcl::PointXYZ>::Ptr open3d_NoiseFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	int MeanK_neighberCount,
	float StddevMulThresh)
{
	//2023-04-18: pcl PointCloud to open3d pointcloud convert -> filter -> revert.
	try
	{
		auto pt_cloud = open3d::geometry::PointCloud();
		auto ply_pointer = std::make_shared<open3d::geometry::PointCloud>(pt_cloud);

		auto d_points = std::vector<Eigen::Vector3d>();

		auto pc_data = pcInput->points;
		for (int i = 0; i < pc_data.size(); i++)
		{
			Eigen::Vector3d temp(pc_data.at(i).x, pc_data.at(i).y, pc_data.at(i).z);
			d_points.push_back(temp);
		}

		ply_pointer->points_ = d_points;

		//auto time_start = std::chrono::system_clock::now();
		auto nf_result = ply_pointer->RemoveStatisticalOutliers(MeanK_neighberCount, StddevMulThresh);
		//auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - time_start).count();
		//leave_a_message("open3d statistical outlier filter: " + std::to_string(dur) + "ms", std::chrono::system_clock::now());

		auto new_d_points = std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>>();
		for (int i = 0; i < std::get<0>(nf_result)->points_.size(); i++)
		{
			pcl::PointXYZ temp(std::get<0>(nf_result)->points_.at(i).x(), std::get<0>(nf_result)->points_.at(i).y(), std::get<0>(nf_result)->points_.at(i).z());
			new_d_points.push_back(temp);
		}

		pcl::PointCloud<pcl::PointXYZ>::Ptr pcResult(new pcl::PointCloud<pcl::PointXYZ>);
		pcResult->points = new_d_points;

		return pcResult;
	}
	catch (...)
	{
		logMessage("Exception occurred during open3d noise filter");
		return NULL;
	}
}

auto get_center_of_cloud(const pcl::PointCloud<pcl::PointXYZ> input, int& max_x, int& min_x, int& max_y, int& min_y, int& max_z, int& min_z) -> pcl::PointXYZ
{
	//Get max min for each axis
	auto loc_max_x = -100000;
	auto loc_min_x = 100000;

	auto loc_max_y = -100000;
	auto loc_min_y = 100000;

	auto loc_max_z = -100000;
	auto loc_min_z = 100000;

	for (auto it = input.points.begin(); it < input.points.end(); ++it)
	{
		auto val_x = (*it).x;
		auto val_y = (*it).y;
		auto val_z = (*it).z;

		if (val_x > loc_max_x)
			loc_max_x = val_x;
		if (val_x < loc_min_x)
			loc_min_x = val_x;

		if (val_y > loc_max_y)
			loc_max_y = val_y;
		if (val_y < loc_min_y)
			loc_min_y = val_y;

		if (val_z > loc_max_z)
			loc_max_z = val_z;
		if (val_z < loc_min_z)
			loc_min_z = val_z;
	}
	max_x = loc_max_x;
	min_x = loc_min_x;
	max_y = loc_max_y;
	min_y = loc_min_y;
	max_z = loc_max_z;
	min_z = loc_min_z;
	return pcl::PointXYZ((max_x - min_x) / 2 + min_x, (max_y - min_y) / 2 + min_y, (max_z - min_z) / 2 + min_z);
}

auto average_x(pcl::PointCloud<pcl::PointXYZ>::Ptr input) -> float
{
	float sum = 0;
	int count = 0;
	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).x;
		sum += val;
	}
	return sum / input->points.size();
}
auto get_max_min_x(pcl::PointCloud<pcl::PointXYZ>::Ptr input, int& maxX, int& minX) -> void
{
	int local_maxX = -10000;
	int local_minX = 10000;
	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).x;
		if (val > local_maxX) local_maxX = val;
		if (val < local_minX) local_minX = val;
	}

	maxX = local_maxX;
	minX = local_minX;
}
auto get_valid_max_x(pcl::PointCloud<pcl::PointXYZ>::Ptr input) -> int
{
	int local_maxX = -10000;
	int local_minX = 10000;

	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).x;
		if (val > local_maxX && val < 10000) local_maxX = val;
		if (val < local_minX && val > -10000) local_minX = val;
	}

	//slice through x to find first valid slice for min_z.
	for (int cx = local_maxX; cx > local_minX ; cx -= 10)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
		auto val = pc_passThrough(input, cx - 40, cx, "x", slice_);
		//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
		if (val)
		{
			if (slice_->points.size() > 30)
			{
				//get average val.
				auto avg_max_x = average_x(slice_);
				return (int)avg_max_x;

			}
		}
	}

	return local_maxX;
}
auto get_valid_min_x(pcl::PointCloud<pcl::PointXYZ>::Ptr input) -> int
{
	int local_maxX = -10000;
	int local_minX = 10000;

	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).x;
		if (val > local_maxX && val < 10000) local_maxX = val;
		if (val < local_minX && val > -10000) local_minX = val;
	}

	//slice through x to find first valid slice for min_z.
	for (int cx = local_minX; cx < local_maxX; cx += 10)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
		auto val = pc_passThrough(input, cx, cx + 40, "x", slice_);
		//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
		if (val)
		{
			if (slice_->points.size() > 30)
			{
				//get average val.
				auto avg_min_x = average_x(slice_);
				return (int)avg_min_x;

			}
		}
	}

	return local_minX;
}

auto get_max_min_y(pcl::PointCloud<pcl::PointXYZ>::Ptr input, int& maxY, int& minY) -> void
{
	int local_maxY = -10000;
	int local_minY = 10000;
	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).y;
		if (val > local_maxY) local_maxY = val;
		if (val < local_minY) local_minY = val;
	}

	maxY = local_maxY;
	minY = local_minY;
}

auto average_z(pcl::PointCloud<pcl::PointXYZ>::Ptr input) -> float
{
	float sum = 0;
	int count = 0;
	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).z;
		sum += val;
	}
	return sum / input->points.size();
}
auto get_valid_max_z(pcl::PointCloud<pcl::PointXYZ>::Ptr input) -> int
{
	int local_maxZ = -10000;
	int local_minZ = 10000;

	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).z;
		if (val > local_maxZ && val < 10000) local_maxZ = val;
		if (val < local_minZ && val > -10000) local_minZ = val;
	}

	//slice through x to find first valid slice for min_z.
	for (int cz = local_maxZ; cz > local_minZ; cz -= 10)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
		auto val = pc_passThrough(input, cz - 40, cz, "z", slice_);
		//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
		if (val)
		{
			if (slice_->points.size() > 30)
			{
				//get average val.
				auto avg_max_z = average_z(slice_);
				return (int)avg_max_z;

			}
		}
	}

	return local_maxZ;
}
auto get_valid_min_z(pcl::PointCloud<pcl::PointXYZ>::Ptr input) -> int
{
	int local_maxZ = -10000;
	int local_minZ = 10000;

	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).z;
		if (val > local_maxZ && val < 10000) local_maxZ = val;
		if (val < local_minZ && val > -10000) local_minZ = val;
	}

	//slice through x to find first valid slice for min_z.
	for (int cz = local_minZ; cz < local_maxZ; cz += 10)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
		auto val = pc_passThrough(input, cz, cz + 40, "z", slice_);
		//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
		if (val)
		{
			if (slice_->points.size() > 30)
			{
				//get average val.
				auto avg_min_z = average_z(slice_);
				return (int)avg_min_z;

			}
		}
	}

	//at this point, just return the local min z.
	return local_minZ;


}
auto get_max_min_z(pcl::PointCloud<pcl::PointXYZ>::Ptr input, int& maxZ, int& minZ) -> void
{
	int local_maxZ = -10000;
	int local_minZ = 10000;
	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).z;
		if (val > local_maxZ) local_maxZ = val;
		if (val < local_minZ) local_minZ = val;
	}

	maxZ = local_maxZ;
	minZ = local_minZ;
}

pcl::PointXYZ pixel_to_depth(pcl::PointCloud<pcl::PointXYZ>::Ptr input, int x, int y)
{
	//512, 424

	//int x = 206;
	//int y = 296;
	pcl::PointXYZ ref_data(0, 0, 0);
	int index = y * 512 + x;
	if (y == 0) index = x;
	//int index = y * 640 + x;
	if (index > 217088 || index < 0)
	{
		index = 0;
		return ref_data;
	}

	try {
		//leave_a_message("mapping index = " + std::to_string(index), std::chrono::system_clock::now());
		if (input->points.size() > index) {
			ref_data = input->points[index];
		}
		return ref_data;
	}
	catch (...)
	{
		return ref_data;
	}
}

auto map_box_to_pc(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, bbx detection) -> pcl::PointCloud<pcl::PointXYZ>::Ptr
{
	auto mark = detection.label;
	auto conf = detection.prob;
	auto x = detection.x;
	auto y = detection.y;
	auto w = detection.w;
	auto h = detection.h;

	int x1 = detection.x;
	int y1 = detection.y;
	int x2 = x1 + w;
	int y2 = y1 + h;

	//auto mp_start = std::chrono::system_clock::now();
	//offset for index.
	//y1 -= 50;
	//y2 -= 20;

	if (y1 < 0) y1 = 0;
	if (y2 < 0) y2 = 0;

	pcl::PointCloud<pcl::PointXYZ>::Ptr res_cloud = std::make_shared<pcl::PointCloud <pcl::PointXYZ>>();
	std::vector<pcl::PointXYZ> temp_points = {};
	//res_cloud->points.resize(size_t(w * h));
	for (int j = y1; j <= y2; j++)
	{
		for (int i = x1; i <= x2; i++)
		{
			pcl::PointXYZ ref_point = pixel_to_depth(pointCloud, i, j);

			if ((ref_point.x != 0 && ref_point.y != 0 && ref_point.z != 0) ||
				((int)ref_point.x != 23 && (int)ref_point.y != 4 && (int)ref_point.z != -1))
			{
				//basic range.
				//if (ref_point.z > 1000 && ref_point.z < 3000)
				//{
				res_cloud->points.push_back(ref_point);
				//}//temp_points.push_back(ref_point);
			}
		}
	}

	//logMessage("map size: " + std::to_string(res_cloud->points.size()));

	//auto nf_start = std::chrono::system_clock::now();
	pcl::PointCloud<pcl::PointXYZ>::Ptr nf_cloud = std::make_shared<pcl::PointCloud <pcl::PointXYZ>>();
	nf_cloud = open3d_NoiseFilter(res_cloud, 30, 1.0);
	//auto nf_end = std::chrono::system_clock::now();
	//logMessage("o3d noise filter: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(nf_end - nf_start).count()) + "ms");

	return nf_cloud;
}

auto map_box_to_pc_noisefiltertest(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, bbx detection, int x_limit, int z_limit, std::string save_path = "") -> pcl::PointCloud<pcl::PointXYZ>::Ptr
{
	auto mark = detection.label;
	auto conf = detection.prob;
	auto x = detection.x;
	auto y = detection.y;
	auto w = detection.w;
	auto h = detection.h;

	int x1 = detection.x;
	int y1 = detection.y;
	int x2 = x1 + w;
	int y2 = y1 + h;

	//auto mp_start = std::chrono::system_clock::now();
	//offset for index.
	//y1 -= 50;
	//y2 -= 20;

	if (y1 < 0) y1 = 0;
	if (y2 < 0) y2 = 0;

	pcl::PointCloud<pcl::PointXYZ>::Ptr res_cloud(new pcl::PointCloud <pcl::PointXYZ>);
	std::vector<pcl::PointXYZ> temp_points = {};
	//res_cloud->points.resize(size_t(w * h));
	for (int j = y1; j <= y2; j++)
	{
		for (int i = x1; i <= x2; i++)
		{
			pcl::PointXYZ ref_point = pixel_to_depth(pointCloud, i, j);

			if ((ref_point.x != 0 && ref_point.y != 0 && ref_point.z != 0) ||
				((int)ref_point.x != 23 && (int)ref_point.y != 4 && (int)ref_point.z != -1))
			{

				//logMessage("point x: " + std::to_string(ref_point.x) + ", y: " + std::to_string(ref_point.y) + ", z:" + std::to_string(ref_point.z));

				if (x_limit > 0)
				{
					if (ref_point.x < x_limit && ref_point.z < z_limit)
					{
						{
							res_cloud->points.push_back(ref_point);
						}
					}
				}
				else if (x_limit < 0)
				{
					if (ref_point.x > x_limit && ref_point.z < z_limit)
					{
						{
							res_cloud->points.push_back(ref_point);
						}
					}
				}
				else if (x_limit == 0 && z_limit != 0)
				{
					if (ref_point.z < z_limit) res_cloud->points.push_back(ref_point);
				}
				else if (x_limit == 0 && z_limit == 0)
				{
					res_cloud->points.push_back(ref_point);
				}

			}
		}
	}
	if (save_path != "") pcl::io::savePCDFileBinaryCompressed(save_path + "_no_filter" + std::to_string(res_cloud->points.size()) + ".pcd", *res_cloud);

	//logMessage("map size: " + std::to_string(res_cloud->points.size()));
	pcl::PointCloud<pcl::PointXYZ>::Ptr nf_out(new pcl::PointCloud <pcl::PointXYZ>);
	int max_points = 0;
	int nf_index = -1;
	//nf1: 30, 0.1
	//pcl::PointCloud<pcl::PointXYZ>::Ptr nf_cloud1(new pcl::PointCloud <pcl::PointXYZ>);
	//nf_cloud1 = open3d_NoiseFilter(res_cloud, 30, 1.0);
	//if (save_path != "") pcl::io::savePCDFileBinaryCompressed(save_path + "_nf30_1.0_pc_" + std::to_string(nf_cloud1->points.size()) + ".pcd", *nf_cloud1);


	//if (save_path != "") pcl::io::savePCDFileBinaryCompressed(save_path + "_nf_out_i_" + std::to_string(nf_index) + ".pcd", *nf_out);

	return res_cloud;// nf_out;
}

auto get_closest_xz(pcl::PointCloud<pcl::PointXYZ>::Ptr input, pcl::PointXYZ refPT) -> pcl::PointXYZ
{
	auto min = 10000;
	pcl::PointXYZ closest(0.0f, 0.0f, 0.0f);
	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = sqrt(pow((*it).x - refPT.x, 2) + pow((*it).z - refPT.z, 2));
		if (val < min)
		{
			min = val;
			closest = (*it);
		}
	}
	return closest;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr makePCL_PointCloud(std::vector<visionary::PointXYZ> input)
{
	try
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud <pcl::PointXYZ>);

		pointCloud->points.resize(input.size());

		std::transform(input.begin(), input.end(), pointCloud->points.begin(),
			[](visionary::PointXYZ val) { return pcl::PointXYZ(val.x * 1000, val.y * 1000, val.z * 1000); });

		return pointCloud;
	}
	catch (std::exception& ex)
	{
		logMessage("std exception occurred in pcl pointer making from T-Mini data: " + std::string(ex.what()));
		return NULL;
	}
}

//for early logging files 
pcl::PointCloud<pcl::PointXYZ>::Ptr makePCL_PointCloud(pcl::PointCloud<pcl::PointXYZ> input, bool convert = false)
{
	//auto dt_now = std::chrono::system_clock::now();
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ >);

	pointCloud->points.resize(input.size());

	if (convert)
	{
		std::transform(input.points.begin(), input.points.end(), pointCloud->points.begin(),
			[](pcl::PointXYZ val) { return pcl::PointXYZ(val.x * 1000, val.y * 1000, val.z * 1000); });
	}
	else
	{
		std::transform(input.points.begin(), input.points.end(), pointCloud->points.begin(),
			[](pcl::PointXYZ val) { return pcl::PointXYZ(val.x, val.y, val.z); });
	}
	//auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - dt_now).count();
	//std::cout << "make dur ms: " << std::to_string(dur) << std::endl;

	//pcl::io::savePCDFileBinaryCompressed("temp.pcd", *pointCloud);
	return pointCloud;
}



//PointCloud :: Hole Detection
pcl::PointXYZ pc_hole_detection(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool isLeftSide, std::string save_path)
{
	try
	{
		pcl::PointXYZ hole_position = pcl::PointXYZ(-1, -1, -1);

		//target position is most left-lower corner point on the cornercastings.

		//main extraction region is gantry side of the cornercastings.
		//once trolley moves (-), trolley side of the cornercastings may not be visible.

		//attempt1: detect center of the hole. then use offset.
		//Use slice width of 40mm.
		//get min, max Z.
		int maxZ, minZ;
		get_max_min_z(pcInput, ref(maxZ), ref(minZ));
		int maxX, minX;
		get_max_min_x(pcInput, ref(maxX), ref(minX));

		int slice_width = 50;
		int inc = (maxZ - minZ) / slice_width;
		
		pcl::PointCloud<pcl::PointXYZ>::Ptr mostPoints_z_slice = std::make_shared<pcl::PointCloud <pcl::PointXYZ>>();
		int max_points_z = 0;
		
		//logMessage("minz: " + std::to_string(minZ) + " maxz: " + std::to_string(maxZ));

		try
		{
			for (int current_z = minZ; current_z < (maxZ - slice_width); current_z += 5)
			{
				//logMessage("current_z: " + std::to_string(current_z) + " , to z + slice: " + std::to_string(current_z + slice_width));
				pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
				auto val = pc_passThrough(pcInput, current_z, current_z + slice_width, "z", slice_);
				if (max_points_z < slice_->points.size())
				{
					pcl::copyPointCloud(*slice_, *mostPoints_z_slice);
					max_points_z = slice_->points.size();
					//logMessage("pointcount: " + std::to_string(max_points_z));
				}	
			}
		}
		catch (exception& ex)
		{
			logMessage(ex.what());
		}
		//logMessage("mostPoints z: " + std::to_string(mostPoints_z_slice->points.size()));
		if (mostPoints_z_slice->points.size() > 0)
		{
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_mostPointsZ_" + std::to_string(mostPoints_z_slice->points.size()) + ".pcd", *mostPoints_z_slice);

			//slice through y,
			int maxY, minY;
			get_max_min_y(mostPoints_z_slice, ref(maxY), ref(minY));
			logMessage("miny: " + std::to_string(minY) + " maxy: " + std::to_string(maxY));

			int mpz_maxZ, mpz_minZ;
			get_max_min_z(mostPoints_z_slice, ref(mpz_maxZ), ref(mpz_minZ));
			logMessage("minz: " + std::to_string(mpz_minZ) + " maxz: " + std::to_string(mpz_maxZ));

			int slice_y_width = 20;
			int y_inc = (maxY - minY) / slice_y_width;

			std::vector<int>hole_center_by_y = {};
			
			int y_index = 0;
			for (auto current_y = minY; current_y <= maxY; current_y += slice_y_width, y_index++)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr slice_y(new pcl::PointCloud<pcl::PointXYZ>);
				pc_passThrough(mostPoints_z_slice, current_y, current_y + slice_y_width, "y", slice_y);

				//slice through x,
				if (slice_y->points.size() > 0)
				{
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_mpz_slice_y_" + std::to_string(y_index) + ".pcd", *slice_y);

					int maxX, minX;
					get_max_min_x(slice_y, ref(maxX), ref(minX));
					int lenX = maxX - minX;
					//15% front and back should not be considered.
					

					int slice_x_width = 5;
					int x_inc = (maxX - minX) / slice_x_width;
					int minX_threshold = std::ceil(x_inc * 0.15);
					int maxX_threshold = std::floor(x_inc * 0.85);

					int hole_start_index = -1;
					int hole_end_index = -1;

					int current_idx = 0;
					int consecutive_count = 0;
					int consecutive_flag = false;
					for (auto current_x = minX; current_x <= maxX; current_x += 5)
					{
						pcl::PointCloud<pcl::PointXYZ>::Ptr slice_x(new pcl::PointCloud<pcl::PointXYZ>);
						pc_passThrough(slice_y, current_x, current_x + slice_x_width, "x", slice_x);

						//but needs to be consecutive.
						//look for non point index.
						if (hole_start_index == -1 && slice_x->points.size() == 0 && current_idx >= minX_threshold)
						{
							hole_start_index = current_idx;
						}
						else if (hole_start_index != -1 && slice_x->points.size() == 0 && hole_end_index == -1)
						{
							consecutive_count++;
						}
						//then check for index with points.
						else if (hole_start_index != -1 && slice_x->points.size() > 0 && current_idx <= maxX_threshold)
						{
							if (consecutive_count > 4)
							{
								//valid end of the hole.
								hole_end_index = current_idx;
								break;
							}
							else
							{
								//insufficient. reset start of the hole.
								hole_start_index = -1;
								consecutive_count = 0;
							}

						}

						current_idx++;
					}

					if (hole_start_index != -1 && hole_end_index != -1)
					{
						int offset_index = (hole_end_index - hole_start_index) / 2 + hole_start_index;
						int hole_center = minX + slice_x_width * offset_index;

						hole_center_by_y.push_back(hole_center);
					}
				}
			}
			
			//average out the center of the gap = center of the hole.
			if (hole_center_by_y.size() > 0)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr hole_center_points(new pcl::PointCloud<pcl::PointXYZ>);
				for (auto& hc_x : hole_center_by_y)
				{
					hole_center_points->points.push_back(pcl::PointXYZ(hc_x, minY, minZ));
				}
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_center_points_count_" + std::to_string(hole_center_by_y.size()) + ".pcd", *hole_center_points);

				double hole_center = static_cast<double>(std::accumulate(hole_center_by_y.begin(), hole_center_by_y.end(), 0)) / hole_center_by_y.size();

				//x: hole_center - cornercastings width / 2. = 162/2 = 81mm.
				//y: min y?
				//z: min z?
				if (isLeftSide)
				{
					hole_position.x = hole_center + 81;
				}
				else
				{
					hole_position.x = hole_center - 81;
				}

				hole_position.y = minY;
				hole_position.z = mpz_minZ;

			}
			else
			{
				logMessage("Hole Center algorithm failed.");
			}
		}
		else
		{
			logMessage("Hole has no valid z slice!");
		}



		//slice through z direction to get most points slice.
		//pcl::PointCloud<pcl::PointXYZ>::Ptr most_points_slice
		
		
		//slice through x direction to get most points slice.


		//slice through y direction on most points z slice and average out x,z position while using the lowest y position.


		return hole_position;

	}
	catch (exception& ex)
	{
		logMessage(std::string("Exception in hole detection: ") + ex.what());
	}
	catch (...)
	{
		logMessage("Unknown error from hole detection");
	}
}

pcl::PointXYZ pc_hole_detection_naive(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool target_on_right, std::string save_path)
{
	pcl::PointXYZ hole_position(-10000, 10000, -10000);
	try
	{
		//X-Z directional limit for bigger ROI.
		int local_min_x, local_max_x;
		int local_min_y, local_max_y;
		int local_min_z, local_max_z;

		pcl::PointCloud<pcl::PointXYZ>::Ptr init_hole_filter_x(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_hole_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		local_min_z = get_valid_axis_value(pcInput, 'z', false);
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pre_init_hole_" + std::to_string(local_min_z) + ".pcd", *pcInput);
		if (!target_on_right)
		{
			//max x
			local_max_x = get_valid_axis_value(pcInput, 'x', true);
			singlePassThrough(pcInput, local_max_x - 200, local_max_x + 100, -10000, 10000, -10000, 10000, init_hole_filter_x);
			singlePassThrough(init_hole_filter_x, -10000, 10000, -10000, 10000, local_min_z - 100, local_min_z + 200, init_hole_filtered);
		}
		else
		{
			//min x
			local_min_x = get_valid_axis_value(pcInput, 'x', false);
			singlePassThrough(pcInput, local_min_x - 100, local_min_x + 200, -10000, 10000, -10000, 10000, init_hole_filter_x);
			singlePassThrough(init_hole_filter_x, -10000, 10000, -10000, 10000, local_min_z - 100, local_min_z + 200, init_hole_filtered);
		}
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_init_hole_filtered.pcd", *init_hole_filtered);

		//Get Y slices :: 20 height slices.
		//Cornercastings height = usually 120mm.

		int y_slice_height = 20;

		//int minY, maxY;
		//get_max_min_y(init_hole_filtered, ref(maxY), ref(minY));
		get_max_min_xyz(init_hole_filtered, ref(local_min_x), ref(local_max_x), ref(local_min_y), ref(local_max_y), ref(local_min_z), ref(local_max_z));

		std::vector<pcl::PointXYZ> target_points = {};

		int i = 0;
		pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr sliceIntrim(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr mpz_slice(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr mpx_slice(new pcl::PointCloud<pcl::PointXYZ>);

		for (int cy = local_min_y; cy < local_max_y; cy += y_slice_height)
		{
			singlePassThrough(init_hole_filtered, -10000, 10000, cy, cy + y_slice_height, -10000, 10000, slice_);
			//auto val = pc_passThrough(init_hole_filtered, cy, cy + y_slice_height, "y", slice_);
			//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (slice_->points.size() > 0)
			{
				//Get most points MinX and most points MinZ Slices

				int slice_width = 20;

				int l_minZ, l_maxZ, l_minX, l_maxX, l_minY, l_maxY;
				get_max_min_xyz(slice_, ref(l_minX), ref(l_maxX), ref(l_minY), ref(l_maxY), ref(l_minZ), ref(l_maxZ));

				for (int cx = l_minX; cx < l_maxX; cx += slice_width)
				{
					singlePassThrough(slice_, cx, cx + slice_width, -10000, 10000, -10000, 10000, sliceIntrim);
					if (sliceIntrim->points.size() > mpx_slice->points.size())
					{
						pcl::copyPointCloud(*sliceIntrim, *mpx_slice);
					}
				}

				for (int cz = l_minZ; cz < l_maxZ; cz += slice_width)
				{
					singlePassThrough(slice_, -10000, 10000, -10000, 10000, cz, cz + slice_width, sliceIntrim);
					if (sliceIntrim->points.size() > mpz_slice->points.size())
					{
						pcl::copyPointCloud(*sliceIntrim, *mpz_slice);
					}
				}
				
				//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_i_" + std::to_string(i) + "_mpx_" + std::to_string(mpx_slice->points.size()) + ".pcd", *mpx_slice);
				//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_i_" + std::to_string(i) + "_mpz_" + std::to_string(mpz_slice->points.size()) + ".pcd", *mpz_slice);

				//from minX slice, get minZ.
				int mpx_minX = -10000, mpx_maxX = -10000;
				int mpx_minY = -10000, mpx_maxY = -10000;
				int mpx_minZ = -10000, mpx_maxZ = -10000;

				if (mpx_slice->points.size() > 0)
				{
					get_max_min_xyz(mpx_slice, ref(mpx_minX), ref(mpx_maxX), ref(mpx_minY), ref(mpx_maxY), ref(mpx_minZ), ref(mpx_maxZ));
				}

				//from minZ slice, get minX.
				int mpz_minX = -10000, mpz_maxX = -10000;
				int mpz_minY = -10000, mpz_maxY = -10000;
				int mpz_minZ = -10000, mpz_maxZ = -10000;

				if (mpz_slice->points.size() > 0)
				{
					get_max_min_xyz(mpz_slice, ref(mpz_minX), ref(mpz_maxX), ref(mpz_minY), ref(mpz_maxY), ref(mpz_minZ), ref(mpz_maxZ));
				}
				//then get closest "actual" point to X,Z values.
				if (!target_on_right)
				{
					if (mpx_minZ != -10000 && mpz_maxX != -10000)
					{
						pcl::PointXYZ refPoint(mpz_maxX, 0, mpx_minZ);

						//Get Closest Point.
						auto targetPt = get_closest_xz(init_hole_filtered, refPoint);
						targetPt.y = mpz_minY;

						pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
						tPt->points.push_back(targetPt);
						//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_targetPt_i_" + std::to_string(i) + ".pcd", *tPt);

						target_points.push_back(targetPt);
					}
				}
				else
				{
					if (mpx_minZ != -10000 && mpz_minX != -10000)
					{
						pcl::PointXYZ refPoint(mpz_minX, 0, mpx_minZ);

						//Get Closest Point.
						auto targetPt = get_closest_xz(init_hole_filtered, refPoint);
						targetPt.y = mpz_minY;

						pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
						tPt->points.push_back(targetPt);
						//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_targetPt_i_" + std::to_string(i) + ".pcd", *tPt);

						target_points.push_back(targetPt);
					}
				}

			}

			i++;

			mpx_slice.reset(new pcl::PointCloud<pcl::PointXYZ>);
			mpz_slice.reset(new pcl::PointCloud<pcl::PointXYZ>);
		}

		//based on target points.
		//average x and z while get min y.
		float avg_x = 0.0f, avg_z = 0.0f;
		for (auto& it : target_points)
		{
			if (it.y < hole_position.y) hole_position.y = it.y;
			avg_x += it.x;
			avg_z += it.z;
		}
		avg_x = avg_x / target_points.size();
		avg_z = avg_z / target_points.size();

		hole_position.x = avg_x;
		hole_position.z = avg_z;
	}

	catch (std::exception& ex)
	{
		logMessage("[PCA_HOLE] " + std::string(ex.what()));
		return pcl::PointXYZ(-10000, -10000, -10000);
	}
	catch (...)
	{
		logMessage("[PCA_HOLE] Unknown Exception!");
		return pcl::PointXYZ(-10000, -10000, -10000);
	}

	return hole_position;
}

//PointCloud :: Cone Detection (XT)?
pcl::PointXYZ pc_cone_detection_naive(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool target_on_right, std::string save_path)
{
	pcl::PointXYZ cone_position(-10000, -10000, -10000);
	try
	{
		//X-Z directional limit for bigger ROI.
		int local_min_x, local_max_x;
		int local_min_y, local_max_y;
		int local_min_z, local_max_z;
		
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filter_z(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		local_min_z = get_valid_axis_value(pcInput, 'z', false, 40.0f, 30, TROLLEY_RANGE_MIN, TROLLEY_RANGE_MAX);
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pre_init_cone_" + std::to_string(local_min_z) + ".pcd", *pcInput);
		if (!target_on_right)
		{
			singlePassThrough(pcInput, -10000, 10000, -10000, 10000, local_min_z - 100, local_min_z + 300, init_cone_filter_z);
			local_max_x = get_valid_axis_value(init_cone_filter_z, 'x', true);
			singlePassThrough(init_cone_filter_z, local_max_x - 300, local_max_x + 100, -10000, 10000, -10000, 10000, init_cone_filtered);
		}
		else
		{
			singlePassThrough(pcInput, -10000, 10000, -10000, 10000, local_min_z - 100, local_min_z + 300, init_cone_filter_z);
			local_min_x = get_valid_axis_value(init_cone_filter_z, 'x', false);
			singlePassThrough(init_cone_filter_z, local_min_x - 100, local_min_x + 300, -10000, 10000, -10000, 10000, init_cone_filtered);
		}
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_init_cone_filtered.pcd", *init_cone_filtered);

		get_max_min_xyz(init_cone_filtered, ref(local_min_x), ref(local_max_x), ref(local_min_y), ref(local_max_y), ref(local_min_z), ref(local_max_z));

		//slice through y direction with slice height = 20.

		//Get most points y slice.
		//In the process of finding most points y slice, check for increase in point count then decrease in point count?

		//from most points y slice, slice through x,z for most points and look for target point.

		int y_slice_height = 30;

		int i = 0;
		int most_points_i = -1;
		pcl::PointCloud<pcl::PointXYZ>::Ptr mpy_slice(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);


		for (int cy = local_max_y; cy > local_min_y; cy -= y_slice_height)
		{
			singlePassThrough(init_cone_filtered, -10000, 10000, cy - y_slice_height, cy, -10000, 10000, slice_);
			//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (slice_->points.size() > 0)
			{
				if (slice_->points.size() > mpy_slice->points.size())
				{
					pcl::copyPointCloud(*slice_, *mpy_slice);
					most_points_i = i;
				}
			}
			i++;
		}

		if (most_points_i != -1)
		{
			int slice_max_y = local_max_y - y_slice_height * most_points_i;
			int slice_min_y = slice_max_y - y_slice_height - 15;
			slice_max_y += 15;

			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			singlePassThrough(init_cone_filtered, -10000, 10000, slice_min_y, slice_max_y, -10000, 10000, slice_);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_extended_slice_y_pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (slice_->points.size() > mpy_slice->points.size())
			{
				pcl::copyPointCloud(*slice_, *mpy_slice);
			}
		}
		
		if (mpy_slice->points.size() > 0)
		{
			int slice_width = 20;
			pcl::PointCloud<pcl::PointXYZ>::Ptr mpz_slice(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr mpx_slice(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr sliceIntrim(new pcl::PointCloud<pcl::PointXYZ>);

			int minZ, maxZ, minX, maxX, minY, maxY;
			get_max_min_xyz(mpy_slice, ref(minX), ref(maxX), ref(minY), ref(maxY), ref(minZ), ref(maxZ));
			
			for (int cx = minX; cx < maxX; cx += slice_width)
			{
				singlePassThrough(mpy_slice, cx, cx + slice_width, -10000, 10000, -10000, 10000, sliceIntrim);
				if (sliceIntrim->points.size() > mpx_slice->points.size())
				{
					pcl::copyPointCloud(*sliceIntrim, *mpx_slice);
				}
			}

			for (int cz = minZ; cz < maxZ; cz += slice_width)
			{
				singlePassThrough(mpy_slice, -10000, 10000, -10000, 10000, cz, cz + slice_width, sliceIntrim);
				if (sliceIntrim->points.size() > mpz_slice->points.size())
				{
					pcl::copyPointCloud(*sliceIntrim, *mpz_slice);
				}
			}

			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_mpx_" + std::to_string(mpx_slice->points.size()) + ".pcd", *mpx_slice);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_mpz_" + std::to_string(mpz_slice->points.size()) + ".pcd", *mpz_slice);

			//from minX slice, get minZ.
			int mpx_minX = -10000, mpx_maxX = -10000;
			int mpx_minZ = -10000, mpx_maxZ = -10000;
			int mpx_minY = -10000, mpx_maxY = -10000;
			if (mpx_slice->points.size() > 0)
			{
				get_max_min_xyz(mpx_slice, ref(mpx_minX), ref(mpx_maxX), ref(mpx_minY), ref(mpx_maxY), ref(mpx_minZ), ref(mpx_maxZ));
			}

			//from minZ slice, get minX.
			int mpz_minX = -10000, mpz_maxX = -10000;
			int mpz_minZ = -10000, mpz_maxZ = -10000;
			int mpz_minY = -10000, mpz_maxY = -10000;
			if (mpz_slice->points.size() > 0)
			{
				get_max_min_xyz(mpz_slice, ref(mpz_minX), ref(mpz_maxX), ref(mpz_minY), ref(mpz_maxY), ref(mpz_minZ), ref(mpz_maxZ));
			}
			//then get closest "actual" point to X,Z values.
			if (!target_on_right)
			{
				if (mpx_minZ != -10000 && mpz_maxX != -10000)
				{
					pcl::PointXYZ refPoint(mpz_maxX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);
					targetPt.y = mpz_minY;
					
					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt.pcd", *tPt);

					cone_position = targetPt;
				}
				else if (mpx_minZ != -10000 && mpz_maxX == -10000)
				{
					//use mpx slice to get minZ and maxX.
					pcl::PointXYZ refPoint(mpx_maxX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);
					targetPt.y = mpx_minY;

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt_mpx.pcd", *tPt);

					cone_position = targetPt;
				}
				else if (mpx_minZ == -10000 && mpz_maxX != -10000)
				{
					//use mpx slice to get minZ and maxX.
					pcl::PointXYZ refPoint(mpz_maxX, 0, mpz_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);
					targetPt.y = mpz_minY;

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt_mpz.pcd", *tPt);

					cone_position = targetPt;
				}
			}
			else
			{
				if (mpx_minZ != -10000 && mpz_minX != -10000)
				{
					pcl::PointXYZ refPoint(mpz_minX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);
					targetPt.y = mpz_minY;

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt.pcd", *tPt);

					cone_position = targetPt;
				}
				else if (mpx_minZ != -10000 && mpz_minX == -10000)
				{
					//use mpx slice to get minZ and maxX.
					pcl::PointXYZ refPoint(mpx_minX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);
					targetPt.y = mpx_minY;

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt_mpx.pcd", *tPt);

					cone_position = targetPt;
				}
				else if (mpx_minZ == -10000 && mpz_minX != -10000)
				{
					//use mpx slice to get minZ and maxX.
					pcl::PointXYZ refPoint(mpz_minX, 0, mpz_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);
					targetPt.y = mpz_minY;

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt_mpz.pcd", *tPt);

					cone_position = targetPt;
				}
			}
		}
	}
	catch (std::exception& ex)
	{
		logMessage("[PCA_CONE] " + std::string(ex.what()));
		return pcl::PointXYZ(-10000, -10000, -10000);
	}
	catch (...)
	{
		logMessage("[PCA_CONE] Unknown Exception!");
		return pcl::PointXYZ(-10000, -10000, -10000);
	}

	return cone_position;
}

pcl::PointXYZ pc_cone_detection_preset(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool target_on_right, MAX_MIN_RANGE presetRange, std::string save_path)
{
	pcl::PointXYZ cone_position(-10000, -10000, -10000);
	try
	{
		//X-Z directional limit for bigger ROI.
		int local_min_x, local_max_x;
		int local_min_y, local_max_y;
		int local_min_z, local_max_z;

		//1. Preset Range로 먼저 Extract.
		pcl::PointCloud<pcl::PointXYZ>::Ptr preset_cone_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		singlePassThrough(pcInput, presetRange.MIN, presetRange.MAX, HEIGHT_RANGE_MIN, HEIGHT_RANGE_MAX, TROLLEY_RANGE_MIN, TROLLEY_RANGE_MAX, preset_cone_filtered);
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pre_preset_cone_" + std::to_string(local_min_z) + ".pcd", *pcInput);
		if (save_path != std::string(""))
		{
			if (preset_cone_filtered->points.size() > 0) pcl::io::savePCDFileBinaryCompressed(save_path + "_preset_cone.pcd", *preset_cone_filtered);
		}

		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filter_z(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		local_min_z = get_valid_axis_value(pcInput, 'z', false);
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pre_init_cone_" + std::to_string(local_min_z) + ".pcd", *pcInput);
		if (!target_on_right)
		{
			singlePassThrough(pcInput, -10000, 10000, -10000, 10000, local_min_z - 100, local_min_z + 300, init_cone_filter_z);
			local_max_x = get_valid_axis_value(init_cone_filter_z, 'x', true);
			singlePassThrough(init_cone_filter_z, local_max_x - 300, local_max_x + 100, -10000, 10000, -10000, 10000, init_cone_filtered);
		}
		else
		{
			singlePassThrough(pcInput, -10000, 10000, -10000, 10000, local_min_z - 100, local_min_z + 300, init_cone_filter_z);
			local_min_x = get_valid_axis_value(init_cone_filter_z, 'x', false);
			singlePassThrough(init_cone_filter_z, local_min_x - 100, local_min_x + 300, -10000, 10000, -10000, 10000, init_cone_filtered);
		}
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_init_cone_filtered.pcd", *init_cone_filtered);

		get_max_min_xyz(init_cone_filtered, ref(local_min_x), ref(local_max_x), ref(local_min_y), ref(local_max_y), ref(local_min_z), ref(local_max_z));

		//slice through y direction with slice height = 20.

		//Get most points y slice.
		//In the process of finding most points y slice, check for increase in point count then decrease in point count?

		//from most points y slice, slice through x,z for most points and look for target point.

		int y_slice_height = 30;

		int i = 0;
		int most_points_i = -1;
		pcl::PointCloud<pcl::PointXYZ>::Ptr mpy_slice(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);


		for (int cy = local_max_y; cy > local_min_y; cy -= y_slice_height)
		{
			singlePassThrough(init_cone_filtered, -10000, 10000, cy - y_slice_height, cy, -10000, 10000, slice_);
			//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (slice_->points.size() > 0)
			{
				if (slice_->points.size() > mpy_slice->points.size())
				{
					pcl::copyPointCloud(*slice_, *mpy_slice);
					most_points_i = i;
				}
			}
			i++;
		}

		if (most_points_i != -1)
		{
			int slice_max_y = local_max_y - y_slice_height * most_points_i;
			int slice_min_y = slice_max_y - y_slice_height - 15;
			slice_max_y += 15;

			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			singlePassThrough(init_cone_filtered, -10000, 10000, slice_min_y, slice_max_y, -10000, 10000, slice_);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_extended_slice_y_pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (slice_->points.size() > mpy_slice->points.size())
			{
				pcl::copyPointCloud(*slice_, *mpy_slice);
			}
		}

		if (mpy_slice->points.size() > 0)
		{
			int slice_width = 20;
			pcl::PointCloud<pcl::PointXYZ>::Ptr mpz_slice(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr mpx_slice(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr sliceIntrim(new pcl::PointCloud<pcl::PointXYZ>);

			int minZ, maxZ, minX, maxX, minY, maxY;
			get_max_min_xyz(mpy_slice, ref(minX), ref(maxX), ref(minY), ref(maxY), ref(minZ), ref(maxZ));

			for (int cx = minX; cx < maxX; cx += slice_width)
			{
				singlePassThrough(mpy_slice, cx, cx + slice_width, -10000, 10000, -10000, 10000, sliceIntrim);
				if (sliceIntrim->points.size() > mpx_slice->points.size())
				{
					pcl::copyPointCloud(*sliceIntrim, *mpx_slice);
				}
			}

			for (int cz = minZ; cz < maxZ; cz += slice_width)
			{
				singlePassThrough(mpy_slice, -10000, 10000, -10000, 10000, cz, cz + slice_width, sliceIntrim);
				if (sliceIntrim->points.size() > mpz_slice->points.size())
				{
					pcl::copyPointCloud(*sliceIntrim, *mpz_slice);
				}
			}

			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_mpx_" + std::to_string(mpx_slice->points.size()) + ".pcd", *mpx_slice);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_mpz_" + std::to_string(mpz_slice->points.size()) + ".pcd", *mpz_slice);

			//from minX slice, get minZ.
			int mpx_minX = -10000, mpx_maxX = -10000;
			int mpx_minZ = -10000, mpx_maxZ = -10000;
			int mpx_minY = -10000, mpx_maxY = -10000;
			if (mpx_slice->points.size() > 0)
			{
				get_max_min_xyz(mpx_slice, ref(mpx_minX), ref(mpx_maxX), ref(mpx_minY), ref(mpx_maxY), ref(mpx_minZ), ref(mpx_maxZ));
			}

			//from minZ slice, get minX.
			int mpz_minX = -10000, mpz_maxX = -10000;
			int mpz_minZ = -10000, mpz_maxZ = -10000;
			int mpz_minY = -10000, mpz_maxY = -10000;
			if (mpz_slice->points.size() > 0)
			{
				get_max_min_xyz(mpz_slice, ref(mpz_minX), ref(mpz_maxX), ref(mpz_minY), ref(mpz_maxY), ref(mpz_minZ), ref(mpz_maxZ));
			}
			//then get closest "actual" point to X,Z values.
			if (!target_on_right)
			{
				if (mpx_minZ != -10000 && mpz_maxX != -10000)
				{
					pcl::PointXYZ refPoint(mpz_maxX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);
					targetPt.y = mpz_minY;

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt.pcd", *tPt);

					cone_position = targetPt;
				}
				else if (mpx_minZ != -10000 && mpz_maxX == -10000)
				{
					//use mpx slice to get minZ and maxX.
					pcl::PointXYZ refPoint(mpx_maxX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);
					targetPt.y = mpx_minY;

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt_mpx.pcd", *tPt);

					cone_position = targetPt;
				}
				else if (mpx_minZ == -10000 && mpz_maxX != -10000)
				{
					//use mpx slice to get minZ and maxX.
					pcl::PointXYZ refPoint(mpz_maxX, 0, mpz_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);
					targetPt.y = mpz_minY;

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt_mpz.pcd", *tPt);

					cone_position = targetPt;
				}
			}
			else
			{
				if (mpx_minZ != -10000 && mpz_minX != -10000)
				{
					pcl::PointXYZ refPoint(mpz_minX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);
					targetPt.y = mpz_minY;

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt.pcd", *tPt);

					cone_position = targetPt;
				}
				else if (mpx_minZ != -10000 && mpz_minX == -10000)
				{
					//use mpx slice to get minZ and maxX.
					pcl::PointXYZ refPoint(mpx_minX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);
					targetPt.y = mpx_minY;

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt_mpx.pcd", *tPt);

					cone_position = targetPt;
				}
				else if (mpx_minZ == -10000 && mpz_minX != -10000)
				{
					//use mpx slice to get minZ and maxX.
					pcl::PointXYZ refPoint(mpz_minX, 0, mpz_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);
					targetPt.y = mpz_minY;

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt_mpz.pcd", *tPt);

					cone_position = targetPt;
				}
			}
		}
	}
	catch (std::exception& ex)
	{
		logMessage("[PCA_CONE] " + std::string(ex.what()));
		return pcl::PointXYZ(-10000, -10000, -10000);
	}
	catch (...)
	{
		logMessage("[PCA_CONE] Unknown Exception!");
		return pcl::PointXYZ(-10000, -10000, -10000);
	}

	return cone_position;
}

//IGH
pcl::PointXYZ pc_IGH_detection_naive(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool target_on_right, std::string save_path)
{
	pcl::PointXYZ cone_position(-10000, -10000, -10000);
	try
	{
		//X-Z directional limit for bigger ROI.
		int local_min_x, local_max_x;
		int local_min_z, local_max_Z;

		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filter_z(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		local_min_z = get_valid_min_z(pcInput);
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pre_init_igh_" + std::to_string(local_min_z) + ".pcd", *pcInput);
		if (!target_on_right)
		{
			auto val = pc_passThrough(pcInput, local_min_z - 100, local_min_z + 400, "z", init_cone_filter_z);
			//max x
			local_max_x = get_valid_max_x(init_cone_filter_z);
			val = pc_passThrough(init_cone_filter_z, local_max_x - 400, local_max_x + 100, "x", init_cone_filtered);
		}
		else
		{
			auto val = pc_passThrough(pcInput, local_min_z - 100, local_min_z + 400, "z", init_cone_filter_z);
			//min x
			local_min_x = get_valid_min_x(pcInput);
			val = pc_passThrough(init_cone_filter_z, local_min_x - 100, local_min_x + 400, "x", init_cone_filtered);

		}
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_init_igh_filtered.pcd", *init_cone_filtered);

		get_max_min_x(init_cone_filtered, ref(local_max_x), ref(local_min_x));
		get_max_min_z(init_cone_filtered, ref(local_max_Z), ref(local_min_z));

		//1. Slice through z first.
		
		int z_slice_depth = 40;
		int z_slice_inc = 20;

		pcl::PointCloud<pcl::PointXYZ>::Ptr mostPoints_z_slice(new pcl::PointCloud<pcl::PointXYZ>);
		int z = 0;
		int most_points_z_i = -1;
		for (int cz = local_min_z; cz < local_max_Z; cz += z_slice_inc)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(init_cone_filtered, cz, cz + z_slice_depth, "z", slice_);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_igh_slice_z_i_" + std::to_string(z) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (val)
			{
				if (slice_->points.size() > mostPoints_z_slice->points.size())
				{
					pcl::copyPointCloud(*slice_, *mostPoints_z_slice);
					most_points_z_i = z;

					if (slice_->points.size() > 300)
					{
						//early stop.
						break;
					}
				}
			}
			z++;
		}

		//Get CenterX of most points z slice.
		int mpz_minX = -10000, mpz_maxX = -10000;
		int mpz_minY = -10000, mpz_maxY = -10000;
		int mpz_minZ = -10000, mpz_maxZ = -10000;

		int mpz_cx = -10000;
		if (mostPoints_z_slice->points.size() > 0)
		{
			//get_max_min_x(mostPoints_z_slice, ref(mpz_maxX), ref(mpz_minX));
			auto cpt = get_center_of_cloud(*mostPoints_z_slice, ref(mpz_maxX), ref(mpz_minX), ref(mpz_maxY), ref(mpz_minY), ref(mpz_maxZ), ref(mpz_minZ));
			mpz_cx = (mpz_maxX + mpz_minX) / 2 ;
		}
		
		//then slice through x but only up to CenterX of z slice.
		int x_slice_depth = 50;
		int x_slice_inc = 25;

		pcl::PointCloud<pcl::PointXYZ>::Ptr mostPoints_x_slice(new pcl::PointCloud<pcl::PointXYZ>);
		int x = 0;
		int most_points_x_i = -1;

		if (!target_on_right)
		{
			int min_x = mpz_cx;
			if (mpz_cx == -10000) min_x = local_min_x;
			for (int cx = local_max_x; cx > min_x; cx -= x_slice_inc)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
				auto val = pc_passThrough(init_cone_filtered, cx - x_slice_depth, cx, "x", slice_);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_igh_slice_x_i_" + std::to_string(x) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
				if (val)
				{
					if (slice_->points.size() > mostPoints_x_slice->points.size())
					{
						pcl::copyPointCloud(*slice_, *mostPoints_x_slice);
						most_points_x_i = x;

						if (slice_->points.size() > 300)
						{
							//early stop.
							break;
						}
					}
				}
				x++;
			}
		}
		else
		{
			int max_x = mpz_cx;
			if (mpz_cx == -10000) max_x = local_max_x;
			for (int cx = local_min_x; cx < max_x; cx += x_slice_inc)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
				auto val = pc_passThrough(init_cone_filtered, cx, cx + x_slice_depth, "x", slice_);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_igh_slice_x_i_" + std::to_string(x) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
				if (val)
				{
					if (slice_->points.size() > mostPoints_x_slice->points.size())
					{
						pcl::copyPointCloud(*slice_, *mostPoints_x_slice);
						most_points_x_i = x;

						if (slice_->points.size() > 300)
						{
							//early stop.
							break;
						}
					}
				}
				x++;
			}
		}

		//x_slice -> max or min x.
		int mpx_minX = -10000, mpx_maxX = -10000;
		//if (mostPoints_x_slice->points.size()) get_max_min_x(mostPoints_x_slice, ref(mpx_maxX), ref(mpx_minX));
		int mpx_minZ = -10000, mpx_maxZ = -10000;
		//if (mostPoints_x_slice->points.size()) get_max_min_z(mostPoints_x_slice, ref(mpx_maxZ), ref(mpx_minZ));
		int mpx_minY = -10000, mpx_maxY = -10000;
		//if (mostPoints_x_slice->points.size()) get_max_min_y(mostPoints_x_slice, ref(mpx_maxY), ref(mpx_minY));

		if (mostPoints_x_slice->points.size() > 0) get_center_of_cloud(*mostPoints_x_slice, ref(mpx_maxX), ref(mpx_minX), ref(mpx_maxY), ref(mpx_minY), ref(mpx_maxZ), ref(mpx_minZ));

		//then get closest "actual" point to X,Z values.
		if (!target_on_right)
		{
			if (mostPoints_x_slice->points.size() > 0 && mostPoints_z_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpx_maxX, mpz_maxY, mpz_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_igh_targetPt_xz.pcd", *tPt);

				cone_position = refPoint;
			}
			else if (mostPoints_z_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpz_maxX, mpz_maxY, mpz_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_igh_targetPt_z.pcd", *tPt);

				cone_position = refPoint;
			}
			else if (mostPoints_x_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpx_maxX, mpx_maxY, mpx_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_igh_targetPt_x.pcd", *tPt);

				cone_position = refPoint;
			}
		}
		else
		{
			if (mostPoints_x_slice->points.size() > 0 && mostPoints_z_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpx_minX, mpz_maxY, mpz_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_igh_targetPt_xz.pcd", *tPt);

				cone_position = refPoint;
			}
			else if (mostPoints_z_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpz_minX, mpz_maxY, mpz_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_igh_targetPt_z.pcd", *tPt);

				cone_position = refPoint;
			}
			else if (mostPoints_x_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpx_minX, mpx_maxY, mpx_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_igh_targetPt_x.pcd", *tPt);

				cone_position = refPoint;
			}
		}
	}
	catch (std::exception& ex)
	{
		logMessage("[PCA_IGH] " + std::string(ex.what()));
		return pcl::PointXYZ(-10000, -10000, -10000);
	}
	catch (...)
	{
		logMessage("[PCA_IGH] Unknown Exception!");
		return pcl::PointXYZ(-10000, -10000, -10000);
	}

	return cone_position;
}
pcl::PointXYZ pc_IGHS_detection_naive(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool target_on_right, std::string save_path)
{
	pcl::PointXYZ cone_position(-10000, -10000, -10000);
	try
	{
		//X-Z directional limit for bigger ROI.
		int local_min_x, local_max_x;
		int local_min_z, local_max_Z;

		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filter_z(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		local_min_z = get_valid_min_z(pcInput);
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pre_init_ighs_" + std::to_string(local_min_z) + ".pcd", *pcInput);
		if (!target_on_right)
		{
			auto val = pc_passThrough(pcInput, local_min_z - 100, local_min_z + 300, "z", init_cone_filter_z);
			//max x
			local_max_x = get_valid_max_x(init_cone_filter_z);
			val = pc_passThrough(init_cone_filter_z, local_max_x - 300, local_max_x + 100, "x", init_cone_filtered);
		}
		else
		{
			auto val = pc_passThrough(pcInput, local_min_z - 100, local_min_z + 300, "z", init_cone_filter_z);
			//min x
			local_min_x = get_valid_min_x(pcInput);
			val = pc_passThrough(init_cone_filter_z, local_min_x - 100, local_min_x + 300, "x", init_cone_filtered);

		}
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_init_ighs_filtered.pcd", *init_cone_filtered);

		get_max_min_x(init_cone_filtered, ref(local_max_x), ref(local_min_x));
		get_max_min_z(init_cone_filtered, ref(local_max_Z), ref(local_min_z));

		//1. Slice through z first.

		int z_slice_depth = 40;
		int z_slice_inc = 20;

		pcl::PointCloud<pcl::PointXYZ>::Ptr mostPoints_z_slice(new pcl::PointCloud<pcl::PointXYZ>);
		int z = 0;
		int most_points_z_i = -1;
		for (int cz = local_min_z; cz < local_max_Z; cz += z_slice_inc)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(init_cone_filtered, cz, cz + z_slice_depth, "z", slice_);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_ighs_slice_z_i_" + std::to_string(z) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (val)
			{
				if (slice_->points.size() > mostPoints_z_slice->points.size())
				{
					pcl::copyPointCloud(*slice_, *mostPoints_z_slice);
					most_points_z_i = z;

					if (slice_->points.size() > 300)
					{
						//early stop.
						break;
					}
				}
			}
			z++;
		}

		//Get CenterX of most points z slice.
		int mpz_minX = -10000, mpz_maxX = -10000;
		int mpz_minY = -10000, mpz_maxY = -10000;
		int mpz_minZ = -10000, mpz_maxZ = -10000;

		int mpz_cx = -10000;
		if (mostPoints_z_slice->points.size() > 0)
		{
			//get_max_min_x(mostPoints_z_slice, ref(mpz_maxX), ref(mpz_minX));
			auto cpt = get_center_of_cloud(*mostPoints_z_slice, ref(mpz_maxX), ref(mpz_minX), ref(mpz_maxY), ref(mpz_minY), ref(mpz_maxZ), ref(mpz_minZ));
			mpz_cx = (mpz_maxX + mpz_minX) / 2;
		}

		//then slice through x but only up to CenterX of z slice.
		int x_slice_depth = 50;
		int x_slice_inc = 25;

		pcl::PointCloud<pcl::PointXYZ>::Ptr mostPoints_x_slice(new pcl::PointCloud<pcl::PointXYZ>);
		int x = 0;
		int most_points_x_i = -1;

		if (!target_on_right)
		{
			int min_x = mpz_cx;
			if (mpz_cx == -10000) min_x = local_min_x;
			for (int cx = local_max_x; cx > min_x; cx -= x_slice_inc)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
				auto val = pc_passThrough(init_cone_filtered, cx - x_slice_depth, cx, "x", slice_);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_ighs_slice_x_i_" + std::to_string(x) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
				if (val)
				{
					if (slice_->points.size() > mostPoints_x_slice->points.size())
					{
						pcl::copyPointCloud(*slice_, *mostPoints_x_slice);
						most_points_x_i = x;

						if (slice_->points.size() > 300)
						{
							//early stop.
							break;
						}
					}
				}
				x++;
			}
		}
		else
		{
			int max_x = mpz_cx;
			if (mpz_cx == -10000) max_x = local_max_x;
			for (int cx = local_min_x; cx < max_x; cx += x_slice_inc)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
				auto val = pc_passThrough(init_cone_filtered, cx, cx + x_slice_depth, "x", slice_);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_ighs_slice_x_i_" + std::to_string(x) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
				if (val)
				{
					if (slice_->points.size() > mostPoints_x_slice->points.size())
					{
						pcl::copyPointCloud(*slice_, *mostPoints_x_slice);
						most_points_x_i = x;

						if (slice_->points.size() > 300)
						{
							//early stop.
							break;
						}
					}
				}
				x++;
			}
		}

		//x_slice -> max or min x.
		int mpx_minX = -10000, mpx_maxX = -10000;
		//if (mostPoints_x_slice->points.size()) get_max_min_x(mostPoints_x_slice, ref(mpx_maxX), ref(mpx_minX));
		int mpx_minZ = -10000, mpx_maxZ = -10000;
		//if (mostPoints_x_slice->points.size()) get_max_min_z(mostPoints_x_slice, ref(mpx_maxZ), ref(mpx_minZ));
		int mpx_minY = -10000, mpx_maxY = -10000;
		//if (mostPoints_x_slice->points.size()) get_max_min_y(mostPoints_x_slice, ref(mpx_maxY), ref(mpx_minY));

		if (mostPoints_x_slice->points.size() > 0) get_center_of_cloud(*mostPoints_x_slice, ref(mpx_maxX), ref(mpx_minX), ref(mpx_maxY), ref(mpx_minY), ref(mpx_maxZ), ref(mpx_minZ));

		//then get closest "actual" point to X,Z values.
		if (!target_on_right)
		{
			if (mostPoints_x_slice->points.size() > 0 && mostPoints_z_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpx_maxX, mpz_maxY, mpz_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_ighs_targetPt_xz.pcd", *tPt);

				cone_position = refPoint;
			}
			else if (mostPoints_z_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpz_maxX, mpz_maxY, mpz_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_ighs_targetPt_z.pcd", *tPt);

				cone_position = refPoint;
			}
			else if (mostPoints_x_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpx_maxX, mpx_maxY, mpx_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_ighs_targetPt_x.pcd", *tPt);

				cone_position = refPoint;
			}
		}
		else
		{
			if (mostPoints_x_slice->points.size() > 0 && mostPoints_z_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpx_minX, mpz_maxY, mpz_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_ighs_targetPt_xz.pcd", *tPt);

				cone_position = refPoint;
			}
			else if (mostPoints_z_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpz_minX, mpz_maxY, mpz_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_ighs_targetPt_z.pcd", *tPt);

				cone_position = refPoint;
			}
			else if (mostPoints_x_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpx_minX, mpx_maxY, mpx_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_ighs_targetPt_x.pcd", *tPt);

				cone_position = refPoint;
			}
		}
	}
	catch (std::exception& ex)
	{
		logMessage("[PCA_IGHS] " + std::string(ex.what()));
		return pcl::PointXYZ(-10000, -10000, -10000);
	}
	catch (...)
	{
		logMessage("[PCA_IGHS] Unknown Exception!");
		return pcl::PointXYZ(-10000, -10000, -10000);
	}

	return cone_position;
}

//PointCloud :: Landed Detection
pcl::PointXYZ pc_landed_detection_naive(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool target_on_right, std::string save_path)
{
	pcl::PointXYZ cone_position(-10000, -10000, -10000);
	try
	{
		int local_min_x, local_max_x;
		int local_min_z, local_max_Z;

		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filter_z(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		local_min_z = get_valid_min_z(pcInput);
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pre_init_land_" + std::to_string(local_min_z) + ".pcd", *pcInput);
		if (!target_on_right)
		{
			auto val = pc_passThrough(pcInput, local_min_z - 100, local_min_z + 300, "z", init_cone_filter_z);
			//max x
			local_max_x = get_valid_max_x(init_cone_filter_z);
			val = pc_passThrough(init_cone_filter_z, local_max_x - 300, local_max_x + 100, "x", init_cone_filtered);


		}
		else
		{
			auto val = pc_passThrough(pcInput, local_min_z - 100, local_min_z + 300, "z", init_cone_filter_z);
			//min x
			local_min_x = get_valid_min_x(pcInput);
			val = pc_passThrough(init_cone_filter_z, local_min_x - 100, local_min_x + 300, "x", init_cone_filtered);

		}
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_init_land_filtered.pcd", *init_cone_filtered);
		//Get mpx, mpz slices
		int slice_width = 20;
		pcl::PointCloud<pcl::PointXYZ>::Ptr mpz_slice(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr mpx_slice(new pcl::PointCloud<pcl::PointXYZ>);

		int minZ, maxZ, minX, maxX, minY, maxY;
		get_max_min_x(init_cone_filtered, ref(maxX), ref(minX));
		get_max_min_z(init_cone_filtered, ref(maxZ), ref(minZ));
		get_max_min_y(init_cone_filtered, ref(maxY), ref(minY));

		for (int cx = minX; cx < maxX; cx += slice_width)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr sliceX_(new pcl::PointCloud<pcl::PointXYZ>);
			auto valX = pc_passThrough(init_cone_filtered, cx, cx + slice_width, "x", sliceX_);
			if (sliceX_->points.size() > mpx_slice->points.size())
			{
				pcl::copyPointCloud(*sliceX_, *mpx_slice);
			}
		}

		for (int cz = minZ; cz < maxZ; cz += slice_width)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr sliceZ_(new pcl::PointCloud<pcl::PointXYZ>);
			auto valZ = pc_passThrough(init_cone_filtered, cz, cz + slice_width, "z", sliceZ_);
			if (sliceZ_->points.size() > mpz_slice->points.size())
			{
				pcl::copyPointCloud(*sliceZ_, *mpz_slice);
			}
		}

		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_mpx_" + std::to_string(mpx_slice->points.size()) + ".pcd", *mpx_slice);
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_mpz_" + std::to_string(mpz_slice->points.size()) + ".pcd", *mpz_slice);

		//from minX slice, get minZ.
		int mpx_minX = -10000, mpx_maxX = -10000;
		int mpx_minZ = -10000, mpx_maxZ = -10000;
		if (mpx_slice->points.size() > 0)
		{
			get_max_min_x(mpx_slice, ref(mpx_maxX), ref(mpx_minX));
			get_max_min_z(mpx_slice, ref(mpx_maxZ), ref(mpx_minZ));
		}

		//from minZ slice, get minX.
		int mpz_minX = -10000, mpz_maxX = -10000;
		int mpz_minZ = -10000, mpz_maxZ = -10000;
		if (mpz_slice->points.size() > 0)
		{
			get_max_min_x(mpz_slice, ref(mpz_maxX), ref(mpz_minX));
			get_max_min_z(mpz_slice, ref(mpz_maxZ), ref(mpz_minZ));
		}
		//then get closest "actual" point to X,Z values.
		if (!target_on_right)
		{
			if (mpx_minZ != -10000 && mpz_maxX != -10000)
			{
				pcl::PointXYZ refPoint(mpz_maxX, 0, mpx_minZ);

				//Get Closest Point.
				auto targetPt = get_closest_xz(init_cone_filtered, refPoint);
				targetPt.y = maxY;

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(targetPt);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_targetPt.pcd", *tPt);

				cone_position = targetPt;
			}
			else if (mpx_minZ != -10000 && mpz_maxX == -10000)
			{
				//use mpx slice to get minZ and maxX.
				pcl::PointXYZ refPoint(mpx_maxX, 0, mpx_minZ);

				//Get Closest Point.
				auto targetPt = get_closest_xz(init_cone_filtered, refPoint);
				targetPt.y = maxY;

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(targetPt);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_targetPt_mpx.pcd", *tPt);

				cone_position = targetPt;
			}
			else if (mpx_minZ == -10000 && mpz_maxX != -10000)
			{
				//use mpx slice to get minZ and maxX.
				pcl::PointXYZ refPoint(mpz_maxX, 0, mpz_minZ);

				//Get Closest Point.
				auto targetPt = get_closest_xz(init_cone_filtered, refPoint);
				targetPt.y = maxY;

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(targetPt);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_targetPt_mpz.pcd", *tPt);

				cone_position = targetPt;
			}
		}
		else
		{
			if (mpx_minZ != -10000 && mpz_minX != -10000)
			{
				pcl::PointXYZ refPoint(mpz_minX, 0, mpx_minZ);

				//Get Closest Point.
				auto targetPt = get_closest_xz(init_cone_filtered, refPoint);
				targetPt.y = maxY;

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(targetPt);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_targetPt.pcd", *tPt);

				cone_position = targetPt;
			}
			else if (mpx_minZ != -10000 && mpz_minX == -10000)
			{
				//use mpx slice to get minZ and maxX.
				pcl::PointXYZ refPoint(mpx_minX, 0, mpx_minZ);

				//Get Closest Point.
				auto targetPt = get_closest_xz(init_cone_filtered, refPoint);
				targetPt.y = maxY;

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(targetPt);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_targetPt_mpx.pcd", *tPt);

				cone_position = targetPt;
			}
			else if (mpx_minZ == -10000 && mpz_minX != -10000)
			{
				//use mpx slice to get minZ and maxX.
				pcl::PointXYZ refPoint(mpz_minX, 0, mpz_minZ);

				//Get Closest Point.
				auto targetPt = get_closest_xz(init_cone_filtered, refPoint);
				targetPt.y = maxY;

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(targetPt);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_targetPt_mpz.pcd", *tPt);

				cone_position = targetPt;
			}
		}
	}
	catch (std::exception& ex)
	{
		logMessage("[PCA_LANDED] " + std::string(ex.what()));
		return pcl::PointXYZ(-10000, -10000, -10000);
	}
	catch (...)
	{
		logMessage("[PCA_LANDED] Unknown Exception!");
		return pcl::PointXYZ(-10000, -10000, -10000);
	}

	return cone_position;
}

pcl::PointXYZ pc_CSTS_detection_naive(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool target_on_right, std::string save_path)
{
	pcl::PointXYZ cone_position(-10000, -10000, -10000);
	try
	{
		//X-Z directional limit for bigger ROI.
		int local_min_x, local_max_x;
		int local_min_z, local_max_Z;

		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filter_z(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		local_min_z = get_valid_min_z(pcInput);
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pre_init_CSTS_" + std::to_string(local_min_z) + ".pcd", *pcInput);
		if (!target_on_right)
		{
			auto val = pc_passThrough(pcInput, local_min_z - 100, local_min_z + 300, "z", init_cone_filter_z);
			//max x
			local_max_x = get_valid_max_x(init_cone_filter_z);
			val = pc_passThrough(init_cone_filter_z, local_max_x - 300, local_max_x + 100, "x", init_cone_filtered);
		}
		else
		{
			auto val = pc_passThrough(pcInput, local_min_z - 100, local_min_z + 300, "z", init_cone_filter_z);
			//min x
			local_min_x = get_valid_min_x(pcInput);
			val = pc_passThrough(init_cone_filter_z, local_min_x - 100, local_min_x + 300, "x", init_cone_filtered);

		}
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_init_CSTS_filtered.pcd", *init_cone_filtered);

		get_max_min_x(init_cone_filtered, ref(local_max_x), ref(local_min_x));
		get_max_min_z(init_cone_filtered, ref(local_max_Z), ref(local_min_z));

		//1. Slice through z first.

		int z_slice_depth = 40;
		int z_slice_inc = 20;

		pcl::PointCloud<pcl::PointXYZ>::Ptr mostPoints_z_slice(new pcl::PointCloud<pcl::PointXYZ>);
		int z = 0;
		int most_points_z_i = -1;
		for (int cz = local_min_z; cz < local_max_Z; cz += z_slice_inc)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(init_cone_filtered, cz, cz + z_slice_depth, "z", slice_);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_CSTS_slice_z_i_" + std::to_string(z) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (val)
			{
				if (slice_->points.size() > mostPoints_z_slice->points.size())
				{
					pcl::copyPointCloud(*slice_, *mostPoints_z_slice);
					most_points_z_i = z;

					if (slice_->points.size() > 300)
					{
						//early stop.
						break;
					}
				}
			}
			z++;
		}

		//Get CenterX of most points z slice.
		int mpz_minX = -10000, mpz_maxX = -10000;
		int mpz_minY = -10000, mpz_maxY = -10000;
		int mpz_minZ = -10000, mpz_maxZ = -10000;

		int mpz_cx = -10000;
		if (mostPoints_z_slice->points.size() > 0)
		{
			//get_max_min_x(mostPoints_z_slice, ref(mpz_maxX), ref(mpz_minX));
			auto cpt = get_center_of_cloud(*mostPoints_z_slice, ref(mpz_maxX), ref(mpz_minX), ref(mpz_maxY), ref(mpz_minY), ref(mpz_maxZ), ref(mpz_minZ));
			mpz_cx = (mpz_maxX + mpz_minX) / 2;
		}

		//then slice through x but only up to CenterX of z slice.
		int x_slice_depth = 50;
		int x_slice_inc = 25;

		pcl::PointCloud<pcl::PointXYZ>::Ptr mostPoints_x_slice(new pcl::PointCloud<pcl::PointXYZ>);
		int x = 0;
		int most_points_x_i = -1;

		if (!target_on_right)
		{
			int min_x = mpz_cx;
			if (mpz_cx == -10000) min_x = local_min_x;
			for (int cx = local_max_x; cx > min_x; cx -= x_slice_inc)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
				auto val = pc_passThrough(init_cone_filtered, cx - x_slice_depth, cx, "x", slice_);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_CSTS_slice_x_i_" + std::to_string(x) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
				if (val)
				{
					if (slice_->points.size() > mostPoints_x_slice->points.size())
					{
						pcl::copyPointCloud(*slice_, *mostPoints_x_slice);
						most_points_x_i = x;

						if (slice_->points.size() > 300)
						{
							//early stop.
							break;
						}
					}
				}
				x++;
			}
		}
		else
		{
			int max_x = mpz_cx;
			if (mpz_cx == -10000) max_x = local_max_x;
			for (int cx = local_min_x; cx < max_x; cx += x_slice_inc)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
				auto val = pc_passThrough(init_cone_filtered, cx, cx + x_slice_depth, "x", slice_);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_CSTS_slice_x_i_" + std::to_string(x) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
				if (val)
				{
					if (slice_->points.size() > mostPoints_x_slice->points.size())
					{
						pcl::copyPointCloud(*slice_, *mostPoints_x_slice);
						most_points_x_i = x;

						if (slice_->points.size() > 300)
						{
							//early stop.
							break;
						}
					}
				}
				x++;
			}
		}

		//x_slice -> max or min x.
		int mpx_minX = -10000, mpx_maxX = -10000;
		//if (mostPoints_x_slice->points.size()) get_max_min_x(mostPoints_x_slice, ref(mpx_maxX), ref(mpx_minX));
		int mpx_minZ = -10000, mpx_maxZ = -10000;
		//if (mostPoints_x_slice->points.size()) get_max_min_z(mostPoints_x_slice, ref(mpx_maxZ), ref(mpx_minZ));
		int mpx_minY = -10000, mpx_maxY = -10000;
		//if (mostPoints_x_slice->points.size()) get_max_min_y(mostPoints_x_slice, ref(mpx_maxY), ref(mpx_minY));

		if (mostPoints_x_slice->points.size() > 0) get_center_of_cloud(*mostPoints_x_slice, ref(mpx_maxX), ref(mpx_minX), ref(mpx_maxY), ref(mpx_minY), ref(mpx_maxZ), ref(mpx_minZ));

		//then get closest "actual" point to X,Z values.
		if (!target_on_right)
		{
			if (mostPoints_x_slice->points.size() > 0 && mostPoints_z_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpx_maxX, mpz_maxY, mpz_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_CSTS_targetPt_xz.pcd", *tPt);

				cone_position = refPoint;
			}
			else if (mostPoints_z_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpz_maxX, mpz_maxY, mpz_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_CSTS_targetPt_z.pcd", *tPt);

				cone_position = refPoint;
			}
			else if (mostPoints_x_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpx_maxX, mpx_maxY, mpx_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_CSTS_targetPt_x.pcd", *tPt);

				cone_position = refPoint;
			}
		}
		else
		{
			if (mostPoints_x_slice->points.size() > 0 && mostPoints_z_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpx_minX, mpz_maxY, mpz_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_CSTS_targetPt_xz.pcd", *tPt);

				cone_position = refPoint;
			}
			else if (mostPoints_z_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpz_minX, mpz_maxY, mpz_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_CSTS_targetPt_z.pcd", *tPt);

				cone_position = refPoint;
			}
			else if (mostPoints_x_slice->points.size() > 0)
			{
				pcl::PointXYZ refPoint(mpx_minX, mpx_maxY, mpx_minZ);

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(refPoint);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_CSTS_targetPt_x.pcd", *tPt);

				cone_position = refPoint;
			}
		}
	}
	catch (std::exception& ex)
	{
		logMessage("[PCA_CSTS] " + std::string(ex.what()));
		return pcl::PointXYZ(-10000, -10000, -10000);
	}
	catch (...)
	{
		logMessage("[PCA_CSTS] Unknown Exception!");
		return pcl::PointXYZ(-10000, -10000, -10000);
	}

	return cone_position;
}

pcl::PointXYZ pc_guide_detection_naive(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool target_on_right, std::string save_path)
{
	pcl::PointXYZ cone_position(10000, 10000, 10000);
	try
	{
		//X-Z directional limit for bigger ROI.
		int local_min_x, local_max_x;
		int local_min_z, local_max_Z;

		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filter_z(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		local_min_z = get_valid_min_z(pcInput);
		local_min_z = get_valid_min_z(pcInput);
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pre_init_guide_" + std::to_string(local_min_z) + ".pcd", *pcInput);
		if (!target_on_right)
		{
			auto val = pc_passThrough(pcInput, local_min_z - 100, local_min_z + 300, "z", init_cone_filter_z);
			//max x
			local_max_x = get_valid_max_x(init_cone_filter_z);
			val = pc_passThrough(init_cone_filter_z, local_max_x - 300, local_max_x + 100, "x", init_cone_filtered);
		}
		else
		{
			auto val = pc_passThrough(pcInput, local_min_z - 100, local_min_z + 300, "z", init_cone_filter_z);
			//min x
			local_min_x = get_valid_min_x(pcInput);
			val = pc_passThrough(init_cone_filter_z, local_min_x - 100, local_min_x + 300, "x", init_cone_filtered);

		}
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_init_guide_filtered.pcd", *init_cone_filtered);


		get_max_min_x(init_cone_filtered, ref(local_max_x), ref(local_min_x));
		get_max_min_z(init_cone_filtered, ref(local_max_Z), ref(local_min_z));


		//slice through y direction with slice height = 20.

		//Get most points y slice.
		//In the process of finding most points y slice, check for increase in point count then decrease in point count?

		//from most points y slice, slice through x,z for most points and look for target point.

		int y_slice_height = 30;

		int minY, maxY;
		get_max_min_y(init_cone_filtered, ref(maxY), ref(minY));

		pcl::PointCloud<pcl::PointXYZ>::Ptr mpy_slice(new pcl::PointCloud<pcl::PointXYZ>);

		int i = 0;
		int most_points_i = -1;
		for (int cy = maxY; cy > minY; cy -= y_slice_height)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(init_cone_filtered, cy - y_slice_height, cy, "y", slice_);
			//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (val)
			{
				if (slice_->points.size() > mpy_slice->points.size())
				{
					pcl::copyPointCloud(*slice_, *mpy_slice);
					most_points_i = i;
				}
			}
			i++;
		}

		if (most_points_i != -1)
		{
			int slice_max_y = maxY - y_slice_height * most_points_i;
			int slice_min_y = slice_max_y - y_slice_height - 15;
			slice_max_y += 15;

			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(init_cone_filtered, slice_min_y, slice_max_y, "y", slice_);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_guide_extended_slice_y_pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (slice_->points.size() > mpy_slice->points.size())
			{
				pcl::copyPointCloud(*slice_, *mpy_slice);
			}
		}

		if (mpy_slice->points.size() > 0)
		{
			int slice_width = 20;
			pcl::PointCloud<pcl::PointXYZ>::Ptr mpz_slice(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr mpx_slice(new pcl::PointCloud<pcl::PointXYZ>);

			int minZ, maxZ, minX, maxX;
			get_max_min_x(mpy_slice, ref(maxX), ref(minX));
			get_max_min_z(mpy_slice, ref(maxZ), ref(minZ));

			for (int cx = minX; cx < maxX; cx += slice_width)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr sliceX_(new pcl::PointCloud<pcl::PointXYZ>);
				auto valX = pc_passThrough(mpy_slice, cx, cx + slice_width, "x", sliceX_);
				if (sliceX_->points.size() > mpx_slice->points.size())
				{
					pcl::copyPointCloud(*sliceX_, *mpx_slice);
				}
			}

			for (int cz = minZ; cz < maxZ; cz += slice_width)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr sliceZ_(new pcl::PointCloud<pcl::PointXYZ>);
				auto valZ = pc_passThrough(mpy_slice, cz, cz + slice_width, "z", sliceZ_);
				if (sliceZ_->points.size() > mpz_slice->points.size())
				{
					pcl::copyPointCloud(*sliceZ_, *mpz_slice);
				}
			}

			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_guide_mpx_" + std::to_string(mpx_slice->points.size()) + ".pcd", *mpx_slice);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_guide_mpz_" + std::to_string(mpz_slice->points.size()) + ".pcd", *mpz_slice);

			//from minX slice, get minZ.
			int mpx_minX = -10000, mpx_maxX = -10000;
			int mpx_minZ = -10000, mpx_maxZ = -10000;
			if (mpx_slice->points.size() > 0)
			{
				get_max_min_x(mpx_slice, ref(mpx_maxX), ref(mpx_minX));
				get_max_min_z(mpx_slice, ref(mpx_maxZ), ref(mpx_minZ));
			}

			//from minZ slice, get minX.
			int mpz_minX = -10000, mpz_maxX = -10000;
			int mpz_minZ = -10000, mpz_maxZ = -10000;
			if (mpz_slice->points.size() > 0)
			{
				get_max_min_x(mpz_slice, ref(mpz_maxX), ref(mpz_minX));
				get_max_min_z(mpz_slice, ref(mpz_maxZ), ref(mpz_minZ));
			}
			//then get closest "actual" point to X,Z values.
			if (!target_on_right)
			{
				if (mpx_minZ != -10000 && mpz_maxX != -10000)
				{
					pcl::PointXYZ refPoint(mpz_maxX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_guide_targetPt.pcd", *tPt);

					cone_position = targetPt;
				}
				else if (mpx_minZ != -10000 && mpz_maxX == -10000)
				{
					//use mpx slice to get minZ and maxX.
					pcl::PointXYZ refPoint(mpx_maxX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_guide_targetPt_mpx.pcd", *tPt);

					cone_position = targetPt;
				}
				else if (mpx_minZ == -10000 && mpz_maxX != -10000)
				{
					//use mpx slice to get minZ and maxX.
					pcl::PointXYZ refPoint(mpz_maxX, 0, mpz_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_guide_targetPt_mpz.pcd", *tPt);

					cone_position = targetPt;
				}
			}
			else
			{
				if (mpx_minZ != -10000 && mpz_minX != -10000)
				{
					pcl::PointXYZ refPoint(mpz_minX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_guide_targetPt.pcd", *tPt);

					cone_position = targetPt;
				}
				else if (mpx_minZ != -10000 && mpz_minX == -10000)
				{
					//use mpx slice to get minZ and maxX.
					pcl::PointXYZ refPoint(mpx_minX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_guide_targetPt_mpx.pcd", *tPt);

					cone_position = targetPt;
				}
				else if (mpx_minZ == -10000 && mpz_minX != -10000)
				{
					//use mpx slice to get minZ and maxX.
					pcl::PointXYZ refPoint(mpz_minX, 0, mpz_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_guide_targetPt_mpz.pcd", *tPt);

					cone_position = targetPt;
				}
			}
		}
	}
	catch (std::exception& ex)
	{
		logMessage("[PCA_GUIDE] " + std::string(ex.what()));
		return pcl::PointXYZ(-10000, -10000, -10000);
	}
	catch (...)
	{
		logMessage("[PCA_GUIDE] Unknown Exception!");
		return pcl::PointXYZ(-10000, -10000, -10000);
	}

	return cone_position;
}


//PointCloud :: Guide Detection
static inline void drawDottedLine(cv::Mat& img, cv::Point p1, cv::Point p2,
	const cv::Scalar& color, int radius = 2, int gap = 8)
{
	cv::LineIterator it(img, p1, p2, 8);
	for (int i = 0; i < it.count; i += gap) {
		cv::circle(img, it.pos(), radius, color, cv::FILLED, cv::LINE_AA);
	}
}

static inline void drawDottedRect(cv::Mat& img, const cv::Rect& r,
	const cv::Scalar& color, int radius = 2, int gap = 8)
{
	cv::Point tl = r.tl();
	cv::Point tr(r.x + r.width, r.y);
	cv::Point br(r.x + r.width, r.y + r.height);
	cv::Point bl(r.x, r.y + r.height);

	drawDottedLine(img, tl, tr, color, radius, gap);
	drawDottedLine(img, tr, br, color, radius, gap);
	drawDottedLine(img, br, bl, color, radius, gap);
	drawDottedLine(img, bl, tl, color, radius, gap);
}

cv::Mat drawOnImage(cv::Mat input, std::vector<rectangle_info> det_results)
{
	cv::Mat result_image = input.clone();

	for (auto& it : det_results)
	{
		bbx box;
		box.index = it.ClassInfo;
		box.label = class_names[box.index];

		box.x = (int)it.X;
		box.y = (int)it.Y;

		box.center_x = (int)it.Center_X;
		box.center_y = (int)it.Center_Y;

		box.w = (int)it.W;
		box.h = (int)it.H;

		box.prob = (int)it.Prob;

		cv::Scalar clr(0, 0, 255);
		if (box.index == 1) clr = cv::Scalar(0, 255, 0);
		else if (box.index == 2) clr = cv::Scalar(255, 0, 0);
		else if (box.index == 3) clr = cv::Scalar(128, 128, 0);

		cv::rectangle(result_image, cv::Rect(box.x, box.y, box.w, box.h), clr);

		//put-text
		std::string label = box.label + "," + std::to_string(box.prob) + "%";
		cv::putText(result_image, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 2);
	}

	return result_image;
}
cv::Mat drawOnImage(cv::Mat input, std::vector<Detection> det_results)
{
	cv::Mat result_image = input.clone();

	for (auto& it : det_results)
	{
		bbx box;
		box.index = it.classId;
		box.label = class_names[box.index];

		box.x = (int)it.box.x;
		box.y = (int)it.box.y;

		box.center_x = (int)(it.box.x + it.box.width / 2);
		box.center_y = (int)(it.box.y + it.box.height / 2);

		box.w = (int)it.box.width;
		box.h = (int)it.box.height;

		box.prob = (int)(it.conf * 100);

		cv::Scalar clr(0, 0, 255);
		bool target_check = true;
		if (!target_check)
		{
			if (box.index == 1) clr = cv::Scalar(0, 255, 0);
			else if (box.index == 2) clr = cv::Scalar(255, 0, 0);
			else if (box.index == 3) clr = cv::Scalar(128, 128, 0);
		}
		else
		{
			//black.
			clr = cv::Scalar(0, 0, 0);

		}
		cv::rectangle(result_image, cv::Rect(box.x, box.y, box.w, box.h), clr);


		if (!target_check)
		{
			//put-text
			std::string label = box.label + "," + std::to_string(round(box.prob)) + "%";
			cv::putText(result_image, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 2);
		}
		else
		{
			std::string label = box.label + "," + std::to_string(round(box.prob)) + "%";
			cv::putText(result_image, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 1);
		}
	}

	return result_image;
}
cv::Mat drawOnImage(cv::Mat input, std::vector<RectInfo> det_results)
{
	cv::Mat result_image = input.clone();

	for (auto& it : det_results)
	{
		bbx box;
		box.index = it.class_id;
		box.label = class_names[box.index];

		box.x = (int)it.box.x;
		box.y = (int)it.box.y;

		box.center_x = (int)(it.box.x + it.box.width / 2);
		box.center_y = (int)(it.box.y + it.box.height / 2);

		box.w = (int)it.box.width;
		box.h = (int)it.box.height;

		box.prob = (int)(it.score * 100);

		cv::Scalar clr(0, 0, 255);
		bool target_check = true;
		if (!target_check)
		{
			if (box.index == 1) clr = cv::Scalar(0, 255, 0);
			else if (box.index == 2) clr = cv::Scalar(255, 0, 0);
			else if (box.index == 3) clr = cv::Scalar(128, 128, 0);
		}
		else
		{
			//black.
			clr = cv::Scalar(0, 0, 0);

		}
		cv::rectangle(result_image, cv::Rect(box.x, box.y, box.w, box.h), clr);


		if (!target_check)
		{
			//put-text
			std::string label = box.label + "," + std::to_string(round(box.prob)) + "%";
			cv::putText(result_image, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 2);
		}
		else
		{
			std::string label = box.label + "," + std::to_string(round(box.prob)) + "%";
			cv::putText(result_image, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 1);
		}
	}

	return result_image;
}
bool save_to_drive(bool isJobLog, cv::Mat oImage, cv::Mat resImage, pcl::PointCloud<pcl::PointXYZ> pointCloud, std::string msg, std::chrono::system_clock::time_point frameTime)
{
	try {
		if (isJobLog)
		{
			auto SaveOImageDir = job_result_folder_name + "/Image/original";
			createDirectory_ifexists(SaveOImageDir);
			auto SaveResImageDir = job_result_folder_name + "/Image/result";
			createDirectory_ifexists(SaveResImageDir);
			auto SaveDepthDir = job_result_folder_name + "/Depth";

			auto timeNow = time_as_name(frameTime);

			//msg: {TWL/TWUL}_{LANDED/LANDOFF}

			std::string pos = current_lane_camera;// "RL";
			//if (CURRENT_SENSOR_POSITION == "REAR_RIGHT") pos = "RR";

			if (!oImage.empty()) cv::imwrite(SaveOImageDir + "/" + timeNow + "_" + msg + "_TMini_" + pos + "_oImage.jpg", oImage);
			if (!resImage.empty()) cv::imwrite(SaveResImageDir + "/" + timeNow + "_" + msg + "_TMini_" + pos + "_resImage.jpg", resImage);

			std::string plyFilePath = SaveDepthDir + "/" + timeNow + "_" + msg + "_TMini_" + pos + "_Depth.pcd";
			if (pointCloud.points.size() > 0) pcl::io::savePCDFileBinaryCompressed(plyFilePath.c_str(), pointCloud);	
		}
		else
		{
			auto SaveImageDir = saveDirName + "/Image";
			auto SaveDepthDir = saveDirName + "/Depth";

			auto timeNow = time_as_name(frameTime);

			std::string pos = current_lane_camera;

			if (!oImage.empty()) cv::imwrite(SaveImageDir + "/" + timeNow + "_TMini_" + pos + "_Image.jpg", oImage);
			//-----------------------------------------------
			// Write point cloud to PLY

			std::string plyFilePath = SaveDepthDir + "/" + timeNow + "_TMini_" + pos + "_Depth.pcd";
			if (pointCloud.points.size() > 0) pcl::io::savePCDFileBinaryCompressed(plyFilePath.c_str(), pointCloud);
			
			//std::printf("Writing frame to %s\n", plyFilePath);
			//PointCloudPlyWriter::WriteFormatPLY(plyFilePath.c_str(), pointCloud, intensityMap, true);
			//std::printf("Finished writing frame to %s\n", plyFilePath);
		}

		return true;
	}
	catch (...)
	{
		logMessage("[save_to_drive] Exception occurred.");
		return false;
	}
}

/*
void data_save_thread_deprecated()
{
	logMessage("Data Logging Thread Activated");
	try
	{
		while (logging_running.load())
		{
			//Safe blocking op.
			std::unique_lock<std::mutex> lock(mutex_logging);
			bool res = cond_logging.wait_for(lock,
				std::chrono::seconds(3600),
				[]() { return saveFlag; });

			//purposely woken up.
			int proc_count = 5;
			if (res)
			{
				while (enable_logging || tsq.GetQueueLen() > 0)
				{
					try
					{
						if (tsq.GetQueueLen() > 0)
						{
							//in blocked state until item is placed in queue.
							auto dataTup = tsq.pop();

							cv::Mat oImage; cv::Mat resImage; pcl::PointCloud<pcl::PointXYZ> pointCloud;
							bool isJobLog = false; std::string msg;

							oImage = std::get<0>(dataTup);
							resImage = std::get<1>(dataTup);
							pointCloud = std::get<2>(dataTup);
							isJobLog = std::get<3>(dataTup);
							msg = std::get<4>(dataTup);
							auto frameTime = std::get<5>(dataTup);

							bool save_res = save_to_drive(isJobLog, oImage, resImage, pointCloud, msg, frameTime);
							if (!save_res) logMessage("Failed to save data to drive!");
							else logMessage("[Data-Save-Thread] Data saved successfully for " + msg + " at " + time_as_name(frameTime));
							//do not hug the process.
							std::this_thread::sleep_for(std::chrono::milliseconds(25));
						}
					}
					catch (std::exception& ex)
					{
						logMessage("[Data-Save-Thread] " + std::string(ex.what()));
						std::this_thread::sleep_for(std::chrono::milliseconds(10));
					}
					catch (...)
					{
						logMessage("[Data-Save-Thread] Unknown Exception!");
						std::this_thread::sleep_for(std::chrono::milliseconds(10));
					}
				}
			}
		}

		logMessage("Data Logging Thread Deactivated");
	}
	catch (std::exception& ex)
	{
		logMessage("[Data-Save-Thread] " + std::string(ex.what()));
	}
	catch (...)
	{
		logMessage("[Data-Save-Thread] Unknown Exception!");
	}	
}
*/

void data_save_thread()
{
	logMessage("Data Logging Thread Activated");

	SaveTask task;
	while (logging_running.load())
	{
		// Blocks when empty; wakes on push(); exits cleanly on tsq.stop()
		if (!tsq.pop(task)) break;

		try
		{
			// cloud is a pointer; dereference only when calling save
			if (!task.cloud) {
				logMessage("[Data-Save-Thread] Null cloud pointer, skipping.");
				continue;
			}

			bool ok = save_to_drive(
				task.isJobLog,
				task.oImage,
				task.resImage,
				*task.cloud,
				task.msg,
				task.frameTime
			);

			if (!ok)
				logMessage("[Data-Save-Thread] Failed to save data to drive!");
			// Consider rate-limiting success logs if high FPS
			// else logMessage("[Data-Save-Thread] Saved: " + task.msg);
		}
		catch (const std::exception& ex)
		{
			logMessage(std::string("[Data-Save-Thread] ") + ex.what());
		}
		catch (...)
		{
			logMessage("[Data-Save-Thread] Unknown Exception!");
		}

		// Optional: batch-drain a few queued items to reduce lock wakeups
		for (int i = 0; i < 5 && logging_running.load(); ++i) {
			SaveTask t2;
			if (!tsq.try_pop(t2)) break;

			try {
				if (!t2.cloud) continue;
				bool ok = save_to_drive(t2.isJobLog, t2.oImage, t2.resImage, *t2.cloud, t2.msg, t2.frameTime);
				if (!ok) logMessage("[Data-Save-Thread] Failed to save data to drive!");
			}
			catch (...) {
				logMessage("[Data-Save-Thread] Exception while batch saving.");
			}
		}
	}

	logMessage("Data Logging Thread Deactivated");
}

auto tmini_setup() -> bool
{
	bool status = false;
	try
	{
		logMessage("TMini Setup Start!");
		using namespace visionary;

		//Get IP from INI
		std::string deviceIpAddr = current_lane_ip;
		unsigned short deviceBlobCtrlPort = 2114;
		unsigned cnt = 100u;

		controlConnected = false;
		dataConnected = false;

		logMessage("Connecting to sensor " + current_lane_camera + " : " + current_lane_ip);

		//-----------------------------------------------
		// Connect to devices data stream 
		if (!dataStream.open(deviceIpAddr, htons(deviceBlobCtrlPort)))
		{
			logMessage("Failed to open data stream connection to device.");
			return false;   // connection failed
		}

		logMessage("Data Stream opened.");
		dataConnected = true;

		//-----------------------------------------------
		// Connect to devices control channel
		if (!visionaryControl.open(VisionaryControl::ProtocolType::COLA_2, deviceIpAddr, 5000/*ms*/))
		{
			logMessage("Failed to open control connection to device.");
			return false;   // connection failed
		}

		logMessage("Control stream opened.");
		controlConnected = true;

		//-----------------------------------------------
		// read Device Ident
		logMessage("DeviceIdent: " + visionaryControl.getDeviceIdent());
		if (!visionaryControl.logout())
		{
			logMessage("Failed to logout");
		}
		status = true;
	}
	catch (exception ex)
	{
		return false;
	}
	catch (...)
	{
		return false;
	}

	return status;
}
int connect_timeout = 1000; //ms

void thread_tmini_control()
{
	//server that controls the connection to specific "ip" address of t mini depending on lane number.
	try
	{
		while (tmini_ctrl_running.load())
		{

			std::unique_lock<std::mutex> lock(mutex_tmini_ctrl);
			bool res = cond_tmini_ctrl.wait_for(lock,
				std::chrono::seconds(3600),
				[]() { return tmini_ctrl_flag; });

			if (tmini_ctrl_flag)
			{
				//sensor_last_attempted_time = std::chrono::system_clock::now();
				while (!sensor_connected && tmini_ctrl_flag)
				{
					if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - sensor_last_attempted_time).count() > connect_timeout)
					{
						auto status = tmini_setup();
						sensor_last_attempted_time = std::chrono::system_clock::now();
						if (status)
						{
							sensor_last_connected_time = std::chrono::system_clock::now();
							sensor_connected = true;
							sensor_fault = false;

							tmini_ctrl_flag = false; //dont enter this loop again.

							//allow streaming to start.
							tmini_data_stream_flag = true;
							cond_tmini_data_stream.notify_one();

							logMessage("TMini Setup Success!");
						}
						else
						{

							sensor_connected = false;
							sensor_fault = true;

							logMessage("TMini Setup Failed!");
						}
					}
					else
					{
						//wait before next attempt.
						std::this_thread::sleep_for(std::chrono::milliseconds(500));
					}
				}
			}
		}

		logMessage("Terminating tmini control thread");

	}
	catch (std::exception& ex)
	{
		logMessage("[TMini-Control] " + std::string(ex.what()));
	}
	catch (...)
	{
		logMessage("[TMini-Control] Unknown Exception!");
	}
}


pcl::PointCloud<pcl::PointXYZ>::Ptr demo_ply(new pcl::PointCloud<pcl::PointXYZ>);
cv::Mat demo_img;

void printMatType(const cv::Mat& mat)
{
	int type = mat.type(); // Get the type of the Mat

	// Get the number of channels
	int channels = mat.channels();

	// Print the type based on the channels and depth
	std::string outMsg = "Type: ";

	switch (CV_MAT_DEPTH(type)) {
	case CV_8U: outMsg += "CV_8U"; break;
	case CV_8S: outMsg += "CV_8S"; break;
	case CV_16U: outMsg += "CV_16U"; break;
	case CV_16S: outMsg += "CV_16S"; break;
	case CV_32S: outMsg += "CV_32S"; break;
	case CV_32F: outMsg += "CV_32F"; break;
	case CV_64F: outMsg += "CV_64F"; break;
	default: outMsg += "Unknown depth"; break;
	}

	// Print the number of channels
	if (channels == 1) {
		outMsg += "C1";  // Single channel
	}
	else if (channels == 3) {
		outMsg += "C3";  // Three channels (e.g., RGB)
	}
	else if (channels == 4) {
		outMsg += "C4";  // Four channels (e.g., RGBA)
	}
	else {
		outMsg += "C" + channels;  // General case
	}

	logMessage(outMsg);

}

cv::VideoCapture tryOpenStream(const std::string& url, bool& success) {
	cv::VideoCapture cap(url, cv::CAP_FFMPEG);
	success = cap.isOpened();
	return cap;
}

auto tmini_data_stream() -> void
{
	try
	{
		if (visionaryControl.stopAcquisition())
			logMessage("Data Acquisition Stopped");
		//cv::namedWindow("T Mini", cv::WINDOW_AUTOSIZE);
		
		if (tmini_data_stream_flag) logMessage("Data Streaming Ready & Starting!");
		while (tmini_data_stream_flag)
		{
			auto waitDur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - last_frame_get_time).count();
			if (waitDur_ms > 150)
			{
				blnGetNewFrame = false;
				bool stepComplete = visionaryControl.stepAcquisition();
				if (stepComplete)
				{
					if (dataStream.getNextFrame())
					{
						blnGetNewFrame = true;
						last_frame_get_time = std::chrono::system_clock::now();

						//no wait.	
						bool isFrameGrabbed = false;					
						//std::printf("Frame received through step called, frame #%d, timestamp: %u \n", pDataHandler->getFrameNum(), pDataHandler->getTimestampMS());					
						{
							//-----------------------------------------------
							// Convert data to a point cloud
							std::vector<PointXYZ> pointCloud;
							pDataHandler->generatePointCloud(pointCloud);
							pDataHandler->transformPointCloud(pointCloud);

							auto intensityMap = pDataHandler->getIntensityMap();

							auto iW = pDataHandler->getWidth();
							auto iH = pDataHandler->getHeight();
							auto gImg = cv::Mat(pDataHandler->getHeight(), pDataHandler->getWidth(), CV_16UC1, intensityMap.data());
							cv::Mat im3; // I want im3 to be the CV_16UC1 of im2
							gImg.convertTo(im3, CV_8UC1);

							cv::Mat grayToClr;
							cv::cvtColor(im3, grayToClr, cv::COLOR_GRAY2BGR);

							auto pclPointCloud = makePCL_PointCloud(pointCloud);
							if (pclPointCloud == NULL)
							{
								logMessage("[TMini-Data-Stream] Failed to create PCL Point Cloud from data handler. -- nullptr!");
								continue;
							}

							if (MODE_DEBUG)
							{
								//replace image with another image.
								logMessage("Current index: " + std::to_string(DEBUG_CURRENT_INDEX) + " Max index: " + std::to_string(DEBUG_MAX_INDEX));
								auto imgFile = cv::imread(DEBUG_IMG_FILES.at(DEBUG_CURRENT_INDEX), 0);
								auto plyFile = DEBUG_PLY_FILES.at(DEBUG_CURRENT_INDEX);

								pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>);
								if (pcl::io::loadPLYFile<pcl::PointXYZ>(plyFile, *pointCloud) == -1)
								{
									std::cout << std::string("Couldn't read file by PLY ") << std::endl;
									//return false;
								}

								imgFile.convertTo(im3, CV_8UC1);
								//printMatType(imgFile);
								//printMatType(im3);
								auto pclPointCloud_temp = makePCL_PointCloud(*pointCloud, DEBUG_CONVERT_PCL_RANGE);
								pcl::copyPointCloud(*pclPointCloud_temp, *pclPointCloud);

								if (DEBUG_CURRENT_INDEX < DEBUG_MAX_INDEX - 1) DEBUG_CURRENT_INDEX++;
								else DEBUG_CURRENT_INDEX = 0;
							}

							if (true)
							{
								//logMessage("Data stack updating stack with new data...");
								auto res = dataStack.Update_Stack(grayToClr, *pclPointCloud, last_frame_get_time);
								//if (res) logMessage("Data Stack Updated Successfully!");
								//else logMessage("Data Stack Update Failed!");

								if (enable_stream)
								{

									if (ENABLE_PRESET_VISUAL_ON_STREAM)
									{
										auto presetBox = current_job_info.presetBox;
										auto line = std::string(std::to_string(presetBox.x) + "," + std::to_string(presetBox.y) + "," + std::to_string(presetBox.w) + "," + std::to_string(presetBox.h));
										
										cv::putText(grayToClr, line, cv::Point(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
										drawDottedRect(grayToClr, cv::Rect(presetBox.x, presetBox.y, presetBox.w, presetBox.h), cv::Scalar(0, 0, 255), 1, 10);
									}

									cv::imshow(CURRENT_SENSOR_POSITION + " T Mini", grayToClr);
									cv::waitKey(1);
								}
								else
								{
									cv::destroyAllWindows();
								}
							}
						}
					}
					else
					{
						logMessage("Data Acqusition Failed!");
						std::this_thread::sleep_for(std::chrono::milliseconds(30));
					}
				}
				else
				{
					logMessage("Data Step Failed!");
					std::this_thread::sleep_for(std::chrono::milliseconds(30));
				}
			}
			else
			{
				if (waitDur_ms > 50)
				{
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
				}
				else
				{
					std::this_thread::sleep_for(std::chrono::milliseconds(50 - waitDur_ms));
				}
			}
		}

		cv::destroyAllWindows();
		visionaryControl.close();
		dataStream.close();

		logMessage("[TMini-Data-Stream] Data Stream Closed!");
	}
	catch (std::exception& ex)
	{
		logMessage("[TMini-Data-Stream] " + std::string(ex.what()));
		cv::destroyAllWindows();
		visionaryControl.close();
		dataStream.close();
	}
	catch (...)
	{
		logMessage("[TMini-Data-Stream] Unknown Exception!");
		cv::destroyAllWindows();
		visionaryControl.close();
		dataStream.close();
	}
}
void thread_tmini_data_stream()
{
	try
	{
		while (tmini_data_stream_running.load())
		{
			std::unique_lock<std::mutex> lock(mutex_tmini_data_stream);
			bool res = cond_tmini_data_stream.wait_for(lock,
				std::chrono::seconds(3600),
				[]() { return tmini_data_stream_flag; });

			if (tmini_data_stream_flag)
			{
				proc_running.store(true);

				tmini_data_stream();
		
				//deflag sensor related bits.
				sensor_connected = false;
				sensor_fault = false;

				logMessage("Sensor Disconnected!");
			}
		}
	}
	catch (std::exception& ex)
	{
		logMessage("[TMini-Data-Stream] " + std::string(ex.what()));
	}
	catch (...)
	{
		logMessage("[TMini-Data-Stream] Unknown Exception!");
	}
}

Eigen::Matrix4f getTiltMatrix(double angle)
{
	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	double rad_angle = M_PI / 180 * angle;
	double cos_val = cos(rad_angle);
	double sin_val = sin(rad_angle);

	transform(1, 1) = cos_val;
	transform(1, 2) = -sin_val;
	transform(2, 1) = sin_val;
	transform(2, 2) = cos_val;

	return transform;
}
Eigen::Matrix4f getSkewMatrix(double angle)
{
	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	double rad_angle = M_PI / 180 * angle;
	double cos_val = cos(rad_angle);
	double sin_val = sin(rad_angle);

	transform(0, 0) = cos_val;
	transform(0, 1) = -sin_val;
	transform(1, 0) = sin_val;
	transform(1, 1) = cos_val;

	return transform;
}
Eigen::Matrix4f getPanMatrix(double angle)
{
	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	double rad_angle = M_PI / 180 * angle;
	double cos_val = cos(rad_angle);
	double sin_val = sin(rad_angle);

	transform(0, 0) = cos_val;
	transform(0, 2) = sin_val;
	transform(2, 0) = -sin_val;
	transform(2, 2) = cos_val;

	return transform;
}
/*
Last Updated: 2020-08-24

Commulative matrix multiplication for point cloud transformation.
Transformation order: skew (z axis) - tilt (x axis) - pan (y axis).
order of execution is reveresed for column matrix multiplication.
*/
pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud_transform(pcl::PointCloud<pcl::PointXYZ>::Ptr input, double pan, double tilt, double skew)//, pcl::PointXYZ center)
{
	//Get Transform matrix given above angles
	Eigen::Matrix4f panMatrix = getPanMatrix(pan);
	Eigen::Matrix4f tiltMatrix = getTiltMatrix(tilt);
	Eigen::Matrix4f skewMatrix = getSkewMatrix(skew);
	Eigen::Matrix4f compressedMatrix =  panMatrix* tiltMatrix* skewMatrix;

	pcl::PointCloud<pcl::PointXYZ>::Ptr nan_out(new pcl::PointCloud<pcl::PointXYZ>);
	std::vector<int> idx;
	pcl::removeNaNFromPointCloud(*input, *nan_out, idx);

	auto has_nan = [](const pcl::PointCloud<pcl::PointXYZ>& c) {
		for (const auto& p : c.points) if (!pcl::isFinite(p)) return true; return false;
		};
	//std::cerr << "before: " << has_nan(*nan_out) << "\n";

	auto finite = [](const pcl::PointXYZ& p) {
		return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)
			&& std::abs(p.x) < 1e6f && std::abs(p.y) < 1e6f && std::abs(p.z) < 1e6f;
		};
	pcl::PointCloud<pcl::PointXYZ>::Ptr cleaned(new pcl::PointCloud<pcl::PointXYZ>);
	cleaned->points.reserve(nan_out->points.size());
	for (const auto& p : nan_out->points) if (finite(p)) cleaned->points.push_back(p);
	cleaned->width = (uint32_t)cleaned->points.size(); cleaned->height = 1; cleaned->is_dense = true;
	nan_out.swap(cleaned);

	pcl::PointCloud<pcl::PointXYZ>::Ptr transform_output(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cleaned, *transform_output, compressedMatrix, true);

	return transform_output;
}

void sanitizePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	for (auto& p : cloud->points)
	{
		if (!std::isfinite(p.x)) p.x = 0.0f;
		if (!std::isfinite(p.y)) p.y = 0.0f;
		if (!std::isfinite(p.z)) p.z = 0.0f;

		// optional clamp for crazy large magnitudes
		if (std::abs(p.x) > 1e6f) p.x = 0.0f;
		if (std::abs(p.y) > 1e6f) p.y = 0.0f;
		if (std::abs(p.z) > 1e6f) p.z = 0.0f;
	}

	cloud->is_dense = true;   // mark as no NaNs
}
auto fix_layout = [](pcl::PointCloud<pcl::PointXYZ>::Ptr& c) {
	const std::size_t n = c->points.size();
	if (n == 0) { c->width = 0; c->height = 0; c->is_dense = true; return; }

	// If organized layout is unknown or inconsistent, make it unorganized.
	if (c->width == 0 || c->height == 0 || std::size_t(c->width) * std::size_t(c->height) != n) {
		c->width = static_cast<uint32_t>(n);
		c->height = 1;
	}
	c->is_dense = true; // we sanitized NaN/Inf -> 0
};
pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud_tilt_transform(pcl::PointCloud<pcl::PointXYZ>::Ptr input, double tilt)//, pcl::PointXYZ center)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);

	float rad = tilt * float(M_PI) / 180.0f;
	//logMessage("rad: " + std::to_string(rad));
	Eigen::Affine3f A = Eigen::Affine3f::Identity();

	// Example: tilt about X
	A.rotate(Eigen::AngleAxisf(rad, Eigen::Vector3f::UnitX()));
	// Optional: A.translation() << tx, ty, tz;

	sanitizePointCloud(input);
	fix_layout(input);
	/*
	pcl::PointCloud<pcl::PointXYZ>::Ptr nan_out(new pcl::PointCloud<pcl::PointXYZ>);
	std::vector<int> idx;
	pcl::removeNaNFromPointCloud(*input, *nan_out, idx);

	auto has_nan = [](const pcl::PointCloud<pcl::PointXYZ>& c) {
		for (const auto& p : c.points) if (!pcl::isFinite(p)) return true; return false;
		};
	//std::cerr << "before: " << has_nan(*nan_out) << "\n";

	auto finite = [](const pcl::PointXYZ& p) {
		return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)
			&& std::abs(p.x) < 1e6f && std::abs(p.y) < 1e6f && std::abs(p.z) < 1e6f;
		};
	pcl::PointCloud<pcl::PointXYZ>::Ptr cleaned(new pcl::PointCloud<pcl::PointXYZ>);
	cleaned->points.reserve(nan_out->points.size());
	for (const auto& p : nan_out->points)
	{
		if (finite(p)) cleaned->points.push_back(p);
	}
	cleaned->width = (uint32_t)cleaned->points.size(); cleaned->height = 1; cleaned->is_dense = true;
	nan_out.swap(cleaned);
	*/
	//std::cerr << "after: " << has_nan(*nan_out) << "\n";

	//for (int i = 0; i < 30; i++)
	//{
	//	std::cout << nan_out->points.at(i) << std::endl;
	//}

	Eigen::Matrix4f T = A.matrix();

	//logMessage("lets see");
	//std::cout << T << std::endl;

	pcl::transformPointCloud(*input, *cloud_out, T);

	//logMessage("yes");

	return cloud_out;
}


//modular functions
//Inference.
/*
bool onnx_inference(VA& detector, std::string SENSOR_POSITION, cv::Mat image, cv::Mat& res_image, std::vector<rectangle_info>& det_results, std::vector<std::vector<bbx>>& det_sorted_objects, int& det_count, std::string save_path = "", bool jobLog = false)
{
	try
	{
		bool status_return = true;
		if (image.empty())
		{
			logMessage("[ONNX-Inference] Image is empty!");
			return false;
		}
		det_sorted_objects = std::vector<std::vector<bbx>>(4, std::vector<bbx>(0));

		//Image Inference
		det_results = {};
		detector.onnx_inference(image, ref(det_results));
		
		det_count = 0;
		for (auto& it : det_results)
		{
			bbx box;
			box.label = it.ClassInfo;

			box.x = (int)it.X;
			box.y = (int)it.Y;

			box.center_x = (int)it.Center_X;
			box.center_y = (int)it.Center_Y;

			box.w = (int)it.W;
			box.h = (int)it.H;

			box.prob = (int)it.Prob;

			if (box.prob < PROB_LIMIT)
			{
				logMessage("Get rid of this object: " + std::to_string(box.label) + " at prob: " + std::to_string(box.prob));
				continue;
			}

			//Choose side depending on job lane.
			if (SENSOR_POSITION == "REAR_LEFT")
			{
				if (box.center_x < 256)
				{
					logMessage("Get rid of this object RL");
					continue;
				}
			}
			else if (SENSOR_POSITION == "REAR_RIGHT")
			{
				if (box.center_x > 256)
				{
					logMessage("Get rid of this object RR");
					continue;
				}
			}

			det_count++;
			det_sorted_objects.at(box.label).push_back(box);	

			if (jobLog) jobLogMessage("Det Object: " + std::to_string(box.label) + "," + std::to_string(box.x) + "," + std::to_string(box.y) + "," + std::to_string(box.w) + "," + std::to_string(box.h) + "," + std::to_string(box.prob));	
		}
		if (save_path != std::string(""))
		{
			res_image = drawOnImage(image, det_results);
			//cv::imwrite(save_path + "_result.jpg", res_img);
		}

		if (det_count == 0) status_return = false;

	}
	catch (std::exception& ex)
	{
		logMessage("[ONNX-Inference] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[ONNX-Inference] Unknown Exception!");
		return false;
	}
}
*/
bool yolo_inference(YOLO11Detector& detector, cv::Mat image, cv::Mat& res_image, std::vector<rectangle_info>& det_results, std::vector<std::vector<bbx>>& det_sorted_objects, int& det_count, std::string save_path = "", bool debugMode = false, bool jobLog = false)
{
	try
	{
		bool status_return = true;
		if (image.empty())
		{
			logMessage("[ONNX-Inference] Image is empty!");
			return false;
		}
		det_sorted_objects = std::vector<std::vector<bbx>>(class_names.size(), std::vector<bbx>(0));

		//Image Inference
		det_results = {};

		// Perform detection with the updated thresholds
		std::vector<Detection> detections = detector.detect(image, confThreshold, iouThreshold);

		det_count = 0;
		for (auto& it : detections)
		{
			if (it.conf * 100 < PROB_LIMIT)
			{
				//logMessage("Get rid of this object: " + std::to_string(it.classId) + " at prob: " + std::to_string(it.conf * 100));
				continue;
			}

			bbx box;
			box.index = it.classId;
			box.label = class_names[box.index];

			box.x = (int)it.box.x;
			box.y = (int)it.box.y;

			box.center_x = (int)(it.box.x + it.box.width / 2);
			box.center_y = (int)(it.box.y + it.box.height / 2);

			box.w = (int)it.box.width;
			box.h = (int)it.box.height;

			box.prob = (int)(it.conf * 100);

			det_count++;
			det_sorted_objects.at(box.index).push_back(box);

			if (jobLog) jobLogMessage("Det Object: " + std::to_string(box.index) + "," + box.label + "," + std::to_string(box.x) + "," + std::to_string(box.y) + "," + std::to_string(box.w) + "," + std::to_string(box.h) + "," + std::to_string(box.prob));
			if (debugMode) logMessage("Det Object: " + std::to_string(box.index) + "," + box.label + "," + std::to_string(box.x) + "," + std::to_string(box.y) + "," + std::to_string(box.w) + "," + std::to_string(box.h) + "," + std::to_string(box.prob));
		}
		
		if (save_path != std::string("")) res_image = drawOnImage(image, detections);
		
		if (det_count == 0) status_return = false;
	}
	catch (std::exception& ex)
	{
		logMessage("[ONNX-Inference] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[ONNX-Inference] Unknown Exception!");
		return false;
	}
}

bool trt_inference(TrtRunner& detector, cv::Mat image, cv::Mat& res_image, std::vector<rectangle_info>& det_results, std::vector<std::vector<bbx>>& det_sorted_objects, int& det_count, std::string save_path = "", bool debugMode = false, bool jobLog = false)
{
	try
	{
		bool status_return = true;
		if (image.empty())
		{
			logMessage("[ONNX-Inference] Image is empty!");
			return false;
		}
		det_sorted_objects = std::vector<std::vector<bbx>>(class_names.size(), std::vector<bbx>(0));

		//Image Inference
		det_results = {};
		
		auto start = std::chrono::high_resolution_clock::now();
		// Perform detection with the updated thresholds
		LetterboxInfo lb;
		cv::Mat in = letterbox(image, 640, 640, lb);
		std::vector<float> inputCHW;
		matToCHW(in, inputCHW);

		size_t inputBytes = inputCHW.size() * sizeof(float);

		if (!detector.infer("images", inputCHW.data(), inputBytes)) {
			logMessage("Trt Inference failed here");
		}

		auto outs = detector.outputs();
		logMessage("Got %zu output tensors " + std::to_string(outs.size()));

		const auto& out0 = outs.at(0);

		// out0.host is bytes; convert to float pointer
		const float* pred = reinterpret_cast<const float*>(out0.host.data());

		// dims: [1,8,8400]
		int C = out0.dims.d[1];   // 8
		int N = out0.dims.d[2];   // 8400
		int numClasses = C - 4;   // 4

		//float confThresh = 0.25f;
		//float iouThresh = 0.45f;

		std::vector<RectInfo> dets = decodeYoloV8_1xCxN(
			pred, C, N, numClasses, confThreshold,
			lb, image.cols, image.rows
		);

		std::vector<RectInfo> finalDets = nms(dets, iouThreshold);

		auto stop_ts = std::chrono::high_resolution_clock::now();
		logMessage("Trt Inference time: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(stop_ts - start).count()));

		det_count = 0;
		for (auto& it : finalDets)
		{
			if (it.score * 100 < PROB_LIMIT)
			{
				//logMessage("Get rid of this object: " + std::to_string(it.classId) + " at prob: " + std::to_string(it.conf * 100));
				continue;
			}

			bbx box;
			box.index = it.class_id;
			box.label = class_names[box.index];

			box.x = (int)it.box.x;
			box.y = (int)it.box.y;

			box.center_x = (int)(it.box.x + it.box.width / 2);
			box.center_y = (int)(it.box.y + it.box.height / 2);

			box.w = (int)it.box.width;
			box.h = (int)it.box.height;

			box.prob = (int)(it.score * 100);

			det_count++;
			det_sorted_objects.at(box.index).push_back(box);

			if (jobLog) jobLogMessage("Det Object: " + std::to_string(box.index) + "," + box.label + "," + std::to_string(box.x) + "," + std::to_string(box.y) + "," + std::to_string(box.w) + "," + std::to_string(box.h) + "," + std::to_string(box.prob));
			if (debugMode) logMessage("Det Object: " + std::to_string(box.index) + "," + box.label + "," + std::to_string(box.x) + "," + std::to_string(box.y) + "," + std::to_string(box.w) + "," + std::to_string(box.h) + "," + std::to_string(box.prob));
		}

		if (save_path != std::string("")) res_image = drawOnImage(image, finalDets);

		if (det_count == 0) status_return = false;
	}
	catch (std::exception& ex)
	{
		logMessage("[Trt-Inference] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Trt-Inference] Unknown Exception!");
		return false;
	}
}

std::string chassis_type_selection_va(std::vector<std::vector<bbx>>& det_sorted_objects, bool jobLog = false)
{
	try
	{
		std::string chassis_type = "";

		//cone + land 
		int xt_counter = det_sorted_objects[1].size() + det_sorted_objects[2].size();
		//igh + ighs
		int igh_counter = det_sorted_objects[3].size() + det_sorted_objects[4].size();
		//cstf + csts + cstb + cstsl
		int cst_counter = det_sorted_objects[5].size() + det_sorted_objects[6].size() + det_sorted_objects[7].size() + det_sorted_objects[8].size();

		if (xt_counter > 0 && (xt_counter > igh_counter && xt_counter > cst_counter))
		{
			chassis_type = "XT";
			logMessage("Chassis Type: XT");
			if (jobLog) jobLogMessage("Chassis Type: XT");
		}
		else if (igh_counter > 0 && (igh_counter > xt_counter && igh_counter > cst_counter))
		{
			chassis_type = "IGH";
			logMessage("Chassis Type: IGH");
			if (jobLog) jobLogMessage("Chassis Type: IGH");
		}
		else if (cst_counter > 0 && (cst_counter > xt_counter && cst_counter > igh_counter))
		{
			chassis_type = "CST";
			logMessage("Chassis Type: CST");
			if (jobLog) jobLogMessage("Chassis Type: CST");
		}
		else
		{
			chassis_type = "unknown";
			logMessage("Chassis Type: Unknown");
			if (jobLog) jobLogMessage("Chassis Type: Unknown");
		}

		return chassis_type;
	}
	catch (std::exception& ex)
	{
		logMessage("[Chassis-Type-Selection-VA] " + std::string(ex.what()));
		if (jobLog) jobLogMessage("[Chassis-Type-Selection-VA] " + std::string(ex.what()));
		return "";
	}
	catch (...)
	{
		logMessage("[Chassis-Type-Selection-VA] Unknown Exception!");
		if (jobLog) jobLogMessage("[Chassis-Type-Selection-VA] Unknown Exception!");
		return "";
	}
}
bool target_selections_va(bool isFrontSensor, std::string cam_pos, std::string jobPos, std::string chassisType, int chassis_size, int sprd_size, std::vector<std::vector<bbx>> det_sorted_objects, bbx& target_hole, bbx& target_cone, bbx& target_guide, bool& detected_hole, bool& detected_cone, bool jobLog = false)
{
	try
	{
		//Current assumptions.
		if (cam_pos == "Rear") //Front is Right.
		{
			if (det_sorted_objects[0].size() > 0)
			{
				std::vector<bbx> detected_holes = det_sorted_objects[0];
				//ascending order.
				std::sort(detected_holes.begin(), detected_holes.end(), [](const bbx& a, const bbx& b) {
					return a.center_x < b.center_x;  // Comparison based on the 'center_x' field
					});

				//Get Most Left. (min x)
				target_hole = detected_holes[0];
			}

			logMessage("TargetSelect ChassisType:" + chassisType);
			
			if (chassisType == "XT")
			{
				// Combine vectors using std::copy
				std::vector<bbx> detected_cones;
				detected_cones.reserve(det_sorted_objects[1].size() + det_sorted_objects[2].size());  // Reserve space to avoid multiple allocations
				detected_cones.insert(detected_cones.end(), det_sorted_objects[1].begin(), det_sorted_objects[1].end());  // Insert first vector
				std::copy(det_sorted_objects[2].begin(), det_sorted_objects[2].end(), std::back_inserter(detected_cones));  // Copy second vector
				
				std::sort(detected_cones.begin(), detected_cones.end(), [](const bbx& a, const bbx& b) {
					return a.center_x < b.center_x;  // Comparison based on the 'center_x' field
					});

				if (detected_cones.size() > 0)
					target_cone = detected_cones[0];

			}
			else if (chassisType == "IGH")
			{
				// Combine vectors using std::copy
				std::vector<bbx> detected_cones;
				detected_cones.reserve(det_sorted_objects[3].size() + det_sorted_objects[4].size());  // Reserve space to avoid multiple allocations
				detected_cones.insert(detected_cones.end(), det_sorted_objects[3].begin(), det_sorted_objects[3].end());  // Insert first vector
				std::copy(det_sorted_objects[4].begin(), det_sorted_objects[4].end(), std::back_inserter(detected_cones));  // Copy second vector

				std::sort(detected_cones.begin(), detected_cones.end(), [](const bbx& a, const bbx& b) {
					return a.center_x < b.center_x;  // Comparison based on the 'center_x' field
					});


				//Get Most Left. (min x)
				if (detected_cones.size() > 0)
					target_cone = detected_cones[0];

				
			}
			else if (chassisType == "CST")
			{
				//CSTF on target cone, CSTS/CSTSL on target guide
				// Combine vectors using std::copy
				std::vector<bbx> detected_cones;
				detected_cones.reserve(det_sorted_objects[6].size() + det_sorted_objects[8].size());  // Reserve space to avoid multiple allocations
				detected_cones.insert(detected_cones.end(), det_sorted_objects[6].begin(), det_sorted_objects[6].end());  // Insert first vector
				std::copy(det_sorted_objects[8].begin(), det_sorted_objects[8].end(), std::back_inserter(detected_cones));  // Copy second vector

				std::sort(detected_cones.begin(), detected_cones.end(), [](const bbx& a, const bbx& b) {
					return a.center_x < b.center_x;  // Comparison based on the 'center_x' field
					});

				//Guide: rear.
				//Cone: side
				if (det_sorted_objects[7].size() > 0)
				{
					target_cone = det_sorted_objects[7][0];
				}

				//get most left.
				if (detected_cones.size() > 0)
				{
					target_guide = detected_cones[0];
				}

			}
			else
			{
				logMessage("Unknown Chassis Type for Target Selection!");
			}
		}
		else //Front is Left. {Front, Center} --Split due to sensor direction.
		{
			if (det_sorted_objects[0].size() > 0)
			{
				std::vector<bbx> detected_holes = det_sorted_objects[0];

				//CenterClose (XT 40ft, Center sensor): a 40ft chassis carrying 20ft containers
				//has TWO positions, so two container holes can appear. Drop holes that are
				//COUPLED (2D-box overlap) to a LAND(idx2) box -- an already-landed NEIGHBOUR
				//container -- so the target stays the FREE (descending) container. If ALL holes
				//are coupled (e.g. the target itself just landed) keep them so landing isn't broken.
				if (chassisType == "XT" && chassis_size == 40 && cam_pos == "Center" && det_sorted_objects[2].size() > 0)
				{
					std::vector<bbx> freeHoles;
					for (const auto& h : detected_holes)
					{
						bool coupled = false;
						for (const auto& l : det_sorted_objects[2])
							if (h.x < l.x + l.w && h.x + h.w > l.x && h.y < l.y + l.h && h.y + h.h > l.y) { coupled = true; break; }
						if (!coupled) freeHoles.push_back(h);
					}
					if (!freeHoles.empty() && freeHoles.size() < detected_holes.size())
					{
						logMessage("[HoleFilter] dropped " + std::to_string(detected_holes.size() - freeHoles.size()) + " LAND-coupled hole(s), " + std::to_string(freeHoles.size()) + " free");
						if (jobLog) jobLogMessage("[HoleFilter] dropped LAND-coupled hole(s)");
						detected_holes = freeHoles;
					}
				}

				//ascending order.
				std::sort(detected_holes.begin(), detected_holes.end(), [](const bbx& a, const bbx& b) {
					return a.center_x < b.center_x;  // Comparison based on the 'center_x' field
					});

				if (cam_pos == "Front")
				{
					if (chassis_size == 45 && sprd_size == 40)
					{
						//Get Most Right. (max x)
						target_hole = detected_holes[detected_holes.size() - 1];
					}
					else
					{
						//Get Most Left. (min x)
						target_hole = detected_holes[0];
					}
				}
				else //Cam position is Center; 
				{
					if (isFrontSensor) //only when jobPos is small. (otherwise Front App will be assigned to Front sensor).
					{
						//Get Most Right. (max x)
						target_hole = detected_holes[detected_holes.size() - 1];
					}
					else
					{
						if (chassis_size == 20 || jobPos == "Big")
						{
							//Get Most Left. (min x)
							target_hole = detected_holes[0];
						}
						else
						{
							//Get Most Right. (max x)
							target_hole = detected_holes[detected_holes.size() - 1];
						}
						
					}
				}
			}

			logMessage("TargetSelect ChassisType:" + chassisType);

			if (chassisType == "XT")
			{
				// Combine vectors using std::copy
				std::vector<bbx> detected_cones;
				detected_cones.reserve(det_sorted_objects[1].size() + det_sorted_objects[2].size());  // Reserve space to avoid multiple allocations
				detected_cones.insert(detected_cones.end(), det_sorted_objects[1].begin(), det_sorted_objects[1].end());  // Insert first vector
				std::copy(det_sorted_objects[2].begin(), det_sorted_objects[2].end(), std::back_inserter(detected_cones));  // Copy second vector

				logMessage(std::to_string(detected_cones.size()) + " cones detected for XT chassis.");

				std::sort(detected_cones.begin(), detected_cones.end(), [](const bbx& a, const bbx& b) {
					return a.center_x < b.center_x;  // Comparison based on the 'center_x' field
					});

				if (cam_pos == "Front")
				{
					if (chassis_size == 45 && sprd_size == 40)
					{
						//Get Most Right. (max x)
						if (detected_cones.size() > 0)
							target_cone = detected_cones[detected_cones.size() - 1];
					}
					else
					{
						//Get Most Left. (min x)
						if (detected_cones.size() > 0)
							target_cone = detected_cones[0];
					}
				}
				else //Center
				{
					if (isFrontSensor)
					{
						if (detected_cones.size() > 0)
							target_cone = detected_cones[detected_cones.size() - 1];
					}
					else
					{
						if (detected_cones.size() > 0)
						{
							if (chassis_size == 20 || jobPos == "Big")
							{
								//Get Most Left. (min x)
								target_cone = detected_cones[0];
							}
							else
							{
								//Get Most Right. (max x)
								target_cone = detected_cones[detected_cones.size() - 1];
							}
						}
					}
				}
			}
			else if (chassisType == "IGH")
			{
				// Combine vectors using std::copy
				std::vector<bbx> detected_cones;
				detected_cones.reserve(det_sorted_objects[3].size() + det_sorted_objects[4].size());  // Reserve space to avoid multiple allocations
				detected_cones.insert(detected_cones.end(), det_sorted_objects[3].begin(), det_sorted_objects[3].end());  // Insert first vector
				std::copy(det_sorted_objects[4].begin(), det_sorted_objects[4].end(), std::back_inserter(detected_cones));  // Copy second vector

				std::sort(detected_cones.begin(), detected_cones.end(), [](const bbx& a, const bbx& b) {
					return a.center_x < b.center_x;  // Comparison based on the 'center_x' field
					});

				if (cam_pos == "Front")
				{
					if (chassis_size == 45 && sprd_size == 40)
					{
						//Get Most Right. (max x)
						if (detected_cones.size() > 0)
							target_cone = detected_cones[detected_cones.size() - 1];
					}
					else
					{
						//Get Most Left. (min x)
						if (detected_cones.size() > 0)
							target_cone = detected_cones[0];
					}
				}
				else //Center
				{
					if (isFrontSensor)
					{
						if (detected_cones.size() > 0)
							target_cone = detected_cones[detected_cones.size() - 1];
					}
					else
					{
						if (detected_cones.size() > 0)
						{
							if (chassis_size == 20 || jobPos == "Big")
							{
								//Get Most Left. (min x)
								target_cone = detected_cones[0];
							}
							else
							{
								//Get Most Right. (max x)
								target_cone = detected_cones[detected_cones.size() - 1];
							}
						}
					}
				}
			}
			else if (chassisType == "CST")
			{
				//CSTF on target cone, CSTS/CSTSL on target guide
				// Combine vectors using std::copy
				std::vector<bbx> detected_cones;
				detected_cones.reserve(det_sorted_objects[6].size() + det_sorted_objects[8].size());  // Reserve space to avoid multiple allocations
				detected_cones.insert(detected_cones.end(), det_sorted_objects[6].begin(), det_sorted_objects[6].end());  // Insert first vector
				std::copy(det_sorted_objects[8].begin(), det_sorted_objects[8].end(), std::back_inserter(detected_cones));  // Copy second vector

				std::sort(detected_cones.begin(), detected_cones.end(), [](const bbx& a, const bbx& b) {
					return a.center_x < b.center_x;  // Comparison based on the 'center_x' field
					});


				if (cam_pos == "Front")
				{
					if (det_sorted_objects[5].size() > 0)
					{
						target_cone = det_sorted_objects[5][0];
					}

					//get most left.
					if (detected_cones.size() > 0)
					{
						target_guide = detected_cones[0];
					}
				}
				else //cam_pos == "Center"
				{
					if (jobPos == "Big")
					{
						//Get left
						//get most left.
						if (detected_cones.size() > 0)
						{
							target_guide = detected_cones[0];
						}
					}
					else if (jobPos == "Small")
					{
						//get most right.
						if (detected_cones.size() > 0)
						{
							target_guide = detected_cones[detected_cones.size() - 1];
						}
					}
				}
			}
			else
			{
				logMessage("Unknown Chassis Type for Target Selection!");
			}
		}

		if (target_hole.prob > 0)
		{
			bDetected_hole = true;
			detected_hole = true;
			logMessage("Detected Hole at: " + std::to_string(target_hole.center_x) + "," + std::to_string(target_hole.center_y) + " with prob: " + std::to_string(target_hole.prob));
			if (jobLog) jobLogMessage("Detected Hole at: " + std::to_string(target_hole.center_x) + "," + std::to_string(target_hole.center_y) + " with prob: " + std::to_string(target_hole.prob));
		}
		else bDetected_hole = false;
		if (target_cone.prob > 0)
		{
			bDetected_cone = true;
			detected_cone = true;
			logMessage("Detected Cone at: " + std::to_string(target_cone.center_x) + "," + std::to_string(target_cone.center_y) + " with prob: " + std::to_string(target_cone.prob));
			if (jobLog) jobLogMessage("Detected Cone at: " + std::to_string(target_cone.center_x) + "," + std::to_string(target_cone.center_y) + " with prob: " + std::to_string(target_cone.prob));
		}
		else bDetected_cone = false;

		return true;
	}
	catch (std::exception& ex)
	{
		logMessage("[Target-Selections-VA] " + std::string(ex.what()));
		if (jobLog) jobLogMessage("[Target-Selections-VA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Target-Selections-VA] Unknown Exception!");
		if (jobLog) jobLogMessage("[Target-Selections-VA] Unknown Exception!");
		return false;
	}
}

bool pre_land_chassis_position_va(bool isMount, bool isOffload, bbx target_cone, bbx target_hole, bool jobLog = false)
{
	try
	{
		if (target_cone.prob > 0)
		{
			if (isMount)
			{
				if (AVG_CONE.get_count() < LDO_NCOUNT)
				{
					if (target_cone.index == 1 && target_cone.prob >= 70) AVG_CONE.update(target_cone);				
				}
				else
				{
					if (!AVG_CONE.get_set())
					{
						logMessage("Average Cone: (X: " + std::to_string(AVG_CONE.get_average().center_x) + " Y: " + std::to_string(AVG_CONE.get_average().center_y) + "),  prob: " + std::to_string(AVG_CONE.get_average().prob));
						if (jobLog) jobLogMessage("Average Cone: (X: " + std::to_string(AVG_CONE.get_average().center_x) + " Y: " + std::to_string(AVG_CONE.get_average().center_y) + "),  prob: " + std::to_string(AVG_CONE.get_average().prob));

						AVG_CONE.set();
					}
				}

			}
			else if (isOffload)
			{
				if (AVG_CONE.get_count() < CLPS_NCOUNT)
				{
					if (target_cone.prob >= 70) AVG_CONE.update(target_cone);					
				}
				else
				{
					if (!AVG_CONE.get_set())
					{
						logMessage("Average Cone: (X: " + std::to_string(AVG_CONE.get_average().center_x) + " Y: " + std::to_string(AVG_CONE.get_average().center_y) + "),  prob: " + std::to_string(AVG_CONE.get_average().prob));
						if (jobLog) jobLogMessage("Average Cone: (X: " + std::to_string(AVG_CONE.get_average().center_x) + " Y: " + std::to_string(AVG_CONE.get_average().center_y) + "),  prob: " + std::to_string(AVG_CONE.get_average().prob));
						AVG_CONE.set();
					}
				}
			}
		}

		if (target_hole.prob > 0)
		{
			if (isOffload)
			{
				if (AVG_HOLE.get_count() < CLPS_NCOUNT)
				{
					if (target_hole.prob > 70) AVG_HOLE.update(target_hole);					
				}
				else
				{
					if (!AVG_HOLE.get_set())
					{
						logMessage("Average Hole: (X: " + std::to_string(AVG_HOLE.get_average().center_x) + " Y: " + std::to_string(AVG_HOLE.get_average().center_y) + "),  prob: " + std::to_string(AVG_HOLE.get_average().prob));
						if (jobLog) jobLogMessage("Average Hole: (X: " + std::to_string(AVG_HOLE.get_average().center_x) + " Y: " + std::to_string(AVG_HOLE.get_average().center_y) + "),  prob: " + std::to_string(AVG_HOLE.get_average().prob));
						AVG_HOLE.set();
					}
				}
			}
		}

		return true;
	}
	catch (std::exception& ex)
	{
		logMessage("[Pre-Land-Chassis-Position-VA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Pre-Land-Chassis-Position-VA] Unknown Exception!");
		return false;
	}
}

bool deviation_output_va(bool isFrontSensor, std::string cam_pos, std::string ChassisType, bbx target_hole, bbx target_cone, bbx target_guide, int& tCntr_x, int& tCntr_y, int& tCntr_prob, int& tCone_x, int& tCone_y, int& tCone_prob, int& devOut_x, int& devOut_y, int& devOut_x_mm, int& devOut_y_mm, bool jobLog = false)
{
	try
	{
		//If target hole is present, deviation between target cone and hole.
		if (target_hole.prob > 0)
		{
			if (cam_pos == "Center" && !isFrontSensor) tCntr_x = target_hole.x + target_hole.w;
			else tCntr_x = target_hole.x;

			tCntr_y = target_hole.y + target_hole.h;
			tCntr_prob = target_hole.prob;
		}

		logMessage("DevOut ChassisType:" + ChassisType);
		if (ChassisType == "XT" || ChassisType == "IGH")
		{
			if (target_cone.prob > 0)
			{
				if (cam_pos == "Center" && !isFrontSensor) tCone_x = target_cone.x + target_cone.w;
				else tCone_x = target_cone.x;

				tCone_y = target_cone.center_y;
				tCone_prob = target_cone.prob;

				//if hole present,
				if (target_hole.prob > 0)
				{
					devOut_x = tCntr_x - tCone_x;
					if (cam_pos == "Rear") devOut_x *= -1; //Invert for rear camera.
					devOut_y = tCntr_y - tCone_y;
				}
			}
		}
		else if (ChassisType == "CST")
		{
			if (target_cone.prob > 0)
			{
				if (cam_pos == "Center" && !isFrontSensor) tCone_x = target_cone.x + target_cone.w;
				else tCone_x = target_cone.x;

				tCone_y = target_cone.y;
				tCone_prob = target_cone.prob;

				//if hole present,
				if (target_hole.prob > 0)
				{
					devOut_x = tCntr_x - tCone_x;
					if (cam_pos == "Rear") devOut_x *= -1; //Invert for rear camera.
					devOut_y = tCntr_y - tCone_y;
				}
			}
			else if (target_guide.prob > 0)
			{
				tCone_x = target_guide.x + target_guide.w;	
				tCone_y = target_guide.center_y;
				tCone_prob = target_guide.prob;

				//if hole present,
				if (target_hole.prob > 0)
				{
					devOut_x = tCntr_x - tCone_x;
					if (cam_pos == "Rear") devOut_x *= -1; //Invert for rear camera.
					devOut_y = tCntr_y - tCone_y;
				}
			}
		}
		
		//2025.03.11
		if (devOut_x != -10000 && devOut_y != -10000)
		{
			//Apply mm per pixel.
			//5mm per pixel.

			devOut_x_mm = devOut_x * 5;
			devOut_y_mm = devOut_y * 5;

		}
		logMessage("VA Dev Out: (X: " + std::to_string(devOut_x) + " Y: " + std::to_string(devOut_y) + ")");
		if (jobLog) jobLogMessage("VA Dev Out: (X: " + std::to_string(devOut_x) + " Y: " + std::to_string(devOut_y) + ")");
		logMessage("VA Dev Out in mm: (X: " + std::to_string(devOut_x_mm) + " Y: " + std::to_string(devOut_y_mm) + ")");
		if (jobLog) jobLogMessage("VA Dev Out in mm (X: " + std::to_string(devOut_x_mm) + " Y: " + std::to_string(devOut_y_mm) + ")");

		return true;
	}
	catch (std::exception& ex)
	{
		logMessage("[Deviation-Output-VA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Deviation-Output-VA] Unknown Exception!");
		return false;
	}
}

bool debug_landed_trigger(bool isFrontSensor, std::string ChassisType, bbx target_hole, bbx target_cone, bbx target_guide)
{
	try
	{
		bool debug_landed_trigger = false;

		if (ChassisType == "XT")
		{
			if (target_cone.prob > 0)
			{
				if (target_hole.prob > 0)
				{
					if (target_hole.y + target_hole.h - target_cone.y > 0)
					{
						debug_landed_trigger = true;
					}
					else if (target_hole.y + target_hole.h < target_cone.y)
					{
						debug_landed_trigger = false;
					}
				}
				else if (target_cone.index == 2) //landed detected
				{
					debug_landed_trigger = true;
				}
				else if (target_cone.index == 1)
				{
					debug_landed_trigger = false;
				}
			}
		}

		else if (ChassisType == "IGH")
		{
			if (target_cone.prob > 0)
			{
				if (target_hole.prob > 0)
				{
					if (target_hole.y + target_hole.h - target_cone.y > 0)
					{
						debug_landed_trigger = true;
					}
					else if (target_hole.y + target_hole.h < target_cone.y)
					{
						debug_landed_trigger = false;
					}
				}
			}
		}

		else if (ChassisType == "CST")
		{
			if (target_cone.prob > 0)
			{
				if (target_hole.prob > 0)
				{
					if (target_hole.y + target_hole.h - target_cone.y > 0)
					{
						debug_landed_trigger = true;
					}
					else if (target_hole.y + target_hole.h < target_cone.y)
					{
						debug_landed_trigger = false;
					}
				}
			}
			else if (target_guide.prob > 0)
			{
				if (target_hole.prob > 0)
				{
					if (target_hole.y + target_hole.h - target_guide.y > 0)
					{
						debug_landed_trigger = true;
					}
					else if (target_hole.y + target_hole.h < target_guide.y)
					{
						debug_landed_trigger = false;
					}
				}
			}
		}

		return debug_landed_trigger;
	}
	catch (std::exception& ex)
	{
		logMessage("[Debug-Landed-Trigger] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Debug-Landed-Trigger] Unknown Exception!");
		return false;
	}
}

bool landOut_detected_va(bool isMount, std::string ChassisType, bbx target_hole, bbx target_cone, bbx target_guide, bool sprd_landed, bool detected_hole, int& devX, int& devY, bool jobLog = false)
{
	try
	{
		bool ldo_result = false;

		if (isMount)
		{
			//based on landed bit
			if (ChassisType == "XT" || ChassisType == "IGH")
			{
				if (sprd_landed)
				{
					if (target_hole.prob > 0 && target_cone.prob > 0)
					{
						devX = abs(target_cone.center_x - target_hole.center_x);
						if (devX > target_cone.w / 2)
						{
							if (LDO_Current_Count < LDO_NCOUNT) LDO_Current_Count++;

							if (LDO_Current_Count >= LDO_NCOUNT)
							{
								if (!landout_detected)
								{
									logMessage("Landout detected by hole exceeding cone!");
									if (jobLog) jobLogMessage("Landout detected with hole exceeding cone!");
								}
								landout_detected = true;
							}
						}
						else
						{
							if (LDO_Current_Count > 0) LDO_Current_Count--;
							if (LDO_Current_Count == LDO_NCOUNT)
							{
								landout_detected = false;
							}
						}
					}
				}
				else
				{
					LDO_Current_Count = 0;
					if (landout_detected) 
					{
						logMessage("Landout Detection Released by SPRD LAND bit OFF");
						if (jobLog) jobLogMessage("Landout Detection Released by SPRD LAND bit OFF");
					}
					landout_detected = false;
				}
			}
			else if (ChassisType == "CST")
			{
				if (sprd_landed)
				{
					if (target_hole.prob > 0 && target_cone.prob > 0)
					{
						if (target_cone.label == "cstf")
						{
							if (target_hole.x + target_hole.w > target_cone.x)
							{
								if (LDO_Current_Count < LDO_NCOUNT) LDO_Current_Count++;
								if (LDO_Current_Count >= LDO_NCOUNT)
								{
									if (!landout_detected)
									{
										logMessage("Landout detected with exceeding cstf!");
										if (jobLog) jobLogMessage("Landout detected with exceeding cstf!");
									}
									landout_detected = true;
								}
							}
							else
							{
								if (LDO_Current_Count > 0) LDO_Current_Count--;
								if (LDO_Current_Count == LDO_NCOUNT)
								{
									landout_detected = false;
								}
							}
						}
						else if (target_cone.label == "cstb")
						{
							if (target_hole.x < target_cone.x + target_cone.w)
							{
								if (LDO_Current_Count < LDO_NCOUNT) LDO_Current_Count++;
								if (LDO_Current_Count >= LDO_NCOUNT)
								{
									if (!landout_detected)
									{
										logMessage("Landout detected with exceeding cstb!");
										if (jobLog) jobLogMessage("Landout detected with exceeding cstb!");
									}
									landout_detected = true;
								}
							}
							else
							{
								if (LDO_Current_Count > 0) LDO_Current_Count--;
								if (LDO_Current_Count == LDO_NCOUNT)
								{
									landout_detected = false;
								}
							}
						}
					}
				}
			}
			else if (detected_hole)
			{
				//only cornercastings is detected.

				//Confidence in inference in general.
				if (target_hole.prob >= 85)
				{
					if (LDO_Current_Count < LDO_NCOUNT) LDO_Current_Count++;

					if (LDO_Current_Count >= LDO_NCOUNT)
					{
						if (!landout_detected) 
						{
							logMessage("Landout detected with HOLE only object!");
							if (jobLog) jobLogMessage("Landout detected with HOLE only object!");
						}
						landout_detected = true;
					}
				}
			}
		}

		return landout_detected;
	}
	catch (std::exception& ex)
	{
		logMessage("[LandOut-Detected-VA] " + std::string(ex.what()));
		if (jobLog) jobLogMessage("[LandOut-Detected-VA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[LandOut-Detected-VA] Unknown Exception!");
		if (jobLog) jobLogMessage("[LandOut-Detected-VA] Unknown Exception!");
		return false;
	}
}

bool clps_detection_va(std::string ChassisType, bbx target_hole, bbx target_cone, bbx target_guide, bool sprd_landed, bool& clps_detected, bool& clps_ok_detected, bool jobLog = false)
{
	try
	{
		if (ChassisType == "CST")
		{
			//CLPS detection not applicable for CST chassis.
			if (!sprd_landed)
			{
				if (CLPS_OK_Current_Count < CLPS_NCOUNT) CLPS_OK_Current_Count++;
				if (CLPS_OK_Current_Count >= CLPS_NCOUNT)
				{
					if (!clps_ok_detected)
					{
						logMessage("CLPS-OK Detected for CST by SPRD LAND bit OFF");
						if (jobLog) jobLogMessage("CLPS-OK Detected for CST by SPRD LAND bit OFF");
					}
					clps_ok_detected = true;
				}
			}
		}
		else if (ChassisType == "IGH" || ChassisType == "XT")
		{
			auto CLPS_Base = AVG_CONE.get_average();
			auto diff_y = std::abs(CLPS_Base.center_y - target_cone.center_y);
			{
				if (diff_y > CLPS_NEAR_Y_THRESHOLD)
				{
					if (!clps_detected)
					{
						if (CLPS_Current_Count < CLPS_NCOUNT) CLPS_Current_Count++;

						if (CLPS_Current_Count >= CLPS_NCOUNT)
						{
							if (!clps_detected)
							{
								logMessage("CLPS Detected! by cone diff_y: " + std::to_string(diff_y));
								if (jobLog) jobLogMessage("CLPS Detected! by cone diff_y: " + std::to_string(diff_y));
							}
							clps_detected = true;
						}
					}
				}
				else
				{
					if (clps_detected)
					{
						if (CLPS_Current_Count > 0) CLPS_Current_Count--;
						if (CLPS_Current_Count == 0)
						{
							clps_detected = false;
						}
					}
				}
			}

			if (ChassisType == "XT")
			{
				//2026-07: instant-fire REMOVED. Reverse-replay proved the ungated single-frame
				//rule asserts CLPS-OK while LANDED (partner cone seen as idx1: 4 jobs / 98
				//frames) -- the intolerable FP. Both paths are now gated by landed==OFF and
				//debounced with CLPS_NCOUNT; landed re-ON releases (re-land observed in data).
				if (sprd_landed)
				{
					CLPS_OK_Current_Count = 0;
					if (clps_ok_detected)
					{
						logMessage("VA CLPS-OK Released! by Landed bit ON");
						if (jobLog) jobLogMessage("VA CLPS-OK Released! by Landed bit ON");
					}
					clps_ok_detected = false;
				}
				else if (target_cone.prob > 70 && target_cone.index == 1)
				{
					//2026-07 rework: the target HOLE is now REQUIRED. VA CLPS-OK asserts ONLY
					//on the certain two-object separation -- the unloaded chassis cone (idx1)
					//AND the container hole BOTH detected, with the hole above the cone
					//(cone.center_y - hole.center_y > 20px). The cone-only default-true was
					//removed: it asserted on the idx1 partner-cone flap while landed (the
					//intolerable FP). The cone-only END state (hole gone once the container
					//leaves frame) is intentionally NOT asserted here -- the 3D ClpsOffload3D
					//height machine covers it, so the combined output still fires there.
					bool sep = (target_hole.prob > 70)
						&& ((target_cone.center_y - target_hole.center_y) > 20);
					if (sep)
					{
						if (CLPS_OK_Current_Count < CLPS_NCOUNT) CLPS_OK_Current_Count++;
						if (CLPS_OK_Current_Count >= CLPS_NCOUNT)
						{
							if (!clps_ok_detected)
							{
								logMessage("VA CLPS-OK Detected! (cone idx1 + hole, Y-sep) dy: " + std::to_string(target_cone.center_y - target_hole.center_y));
								if (jobLog) jobLogMessage("VA CLPS-OK Detected! (cone idx1 + hole, Y-sep) dy: " + std::to_string(target_cone.center_y - target_hole.center_y));
							}
							clps_ok_detected = true;
						}
					}
					else if (CLPS_OK_Current_Count > 0) CLPS_OK_Current_Count--;
				}
				else if (target_cone.prob > 70 && target_cone.index == 2)
				{
					CLPS_OK_Current_Count = 0;
					if (clps_ok_detected)
					{
						logMessage("VA CLPS-OK Released! by Landed class");
						if (jobLog) jobLogMessage("VA CLPS-OK Released! by Landed class");
					}
					clps_ok_detected = false;
				}
			}

			else if (ChassisType == "IGH")
			{
				//2026-07: same gate + debounce as XT (was instant, no release).
				if (sprd_landed)
				{
					CLPS_OK_Current_Count = 0;
					if (clps_ok_detected)
					{
						logMessage("VA CLPS-OK Released! by Landed bit ON");
						if (jobLog) jobLogMessage("VA CLPS-OK Released! by Landed bit ON");
					}
					clps_ok_detected = false;
				}
				else if (target_hole.prob > 70 && target_cone.prob > 70)
				{
					auto diff_y = target_cone.center_y - target_hole.center_y;
					if (diff_y > 20)
					{
						if (CLPS_OK_Current_Count < CLPS_NCOUNT) CLPS_OK_Current_Count++;
						if (CLPS_OK_Current_Count >= CLPS_NCOUNT)
						{
							if (!clps_ok_detected)
							{
								logMessage("VA CLPS-OK Detected! (gated+debounced) by Cone and Hole, diff_y: " + std::to_string(diff_y));
								if (jobLog) jobLogMessage("VA CLPS-OK Detected! (gated+debounced) by Cone and Hole, diff_y: " + std::to_string(diff_y));
							}
							clps_ok_detected = true;
						}
					}
					else if (CLPS_OK_Current_Count > 0) CLPS_OK_Current_Count--;
				}
			}

		}
	}
	catch (std::exception& ex)
	{
		logMessage("[CLPS-Detection-VA] " + std::string(ex.what()));
		if (jobLog) jobLogMessage("[CLPS-Detection-VA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[CLPS-Detection-VA] Unknown Exception!");
		if (jobLog) jobLogMessage("[CLPS-Detection-VA] Unknown Exception!");
		return false;
	}
}


bool target_pointcloud_extraction(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, bbx target_hole, bbx target_cone, bbx target_guide, bool sprd_landed, pcl::PointCloud<pcl::PointXYZ>::Ptr& pcHole, pcl::PointCloud<pcl::PointXYZ>::Ptr& pcCone, pcl::PointCloud<pcl::PointXYZ>::Ptr& pcGuide, std::string save_path = "", bool jobLog = false)
{
	try
	{
		if (target_hole.prob > 0)
		{
			std::string bPath = (save_path != std::string("")) ? save_path + "_Hole" : "";
			//logMessage(std::to_string(target_hole.x) + " " + std::to_string(target_hole.y) + " , " + std::to_string(x_limit) + " , " + std::to_string(z_limit));
			pcHole = map_box_to_pc_noisefiltertest(pointCloud, target_hole, 0, 0, bPath);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pcHole.pcd", *pcHole);
		}
		if (target_cone.prob > 0)
		{
			if (target_cone.index == 1)
			{
				std::string bPath = (save_path != std::string("")) ? save_path + "_Cone" : "";
				pcCone = map_box_to_pc_noisefiltertest(pointCloud, target_cone, 0, 0, bPath);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pcCone.pcd", *pcCone);
			}
			else if (target_cone.index == 2)
			{
				//modify detected landed area.
				if (target_hole.prob > 0)
				{
					std::string bPath = (save_path != std::string("")) ? save_path + "_PreLanded" : "";
					auto pcPreCone = map_box_to_pc_noisefiltertest(pointCloud, target_cone, 0, 0, bPath);
					if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_pcPreLanded.pcd", *pcCone);

					bbx modified_landed = target_cone;
					auto delta_y = target_hole.y + target_hole.h - target_cone.y;
					modified_landed.y = target_hole.y + target_hole.h;
					modified_landed.h -= delta_y;

					std::string bPath2 = (save_path != std::string("")) ? save_path + "_Landed" : "";
					pcCone = map_box_to_pc_noisefiltertest(pointCloud, modified_landed, 0, 0, bPath2);
					if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_pcLanded.pcd", *pcCone);
				}
				else
				{
					std::string bPath = (save_path != std::string("")) ? save_path + "_PreLanded" : "";
					auto pcPreCone = map_box_to_pc_noisefiltertest(pointCloud, target_cone, 0, 0, bPath);
					if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_pcPreLanded.pcd", *pcCone);

					//preset y down?
					int preset_y_hole = 20;
					bbx modified_landed = target_cone;
					modified_landed.y = target_cone.y + preset_y_hole;
					modified_landed.h = target_cone.h - preset_y_hole;
					std::string bPath2 = (save_path != std::string("")) ? save_path + "_Landed" : "";
					pcCone = map_box_to_pc_noisefiltertest(pointCloud, modified_landed, 0, 0, bPath2);
					if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_pcLanded_preset.pcd", *pcCone);
				}
			}
			else if (target_cone.index == 3 || target_cone.index == 4)
			{
				std::string bPath = (save_path != std::string("")) ? save_path + "_IGH" : "";
				pcCone = map_box_to_pc_noisefiltertest(pointCloud, target_cone, 0, 0, bPath);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pcIGH.pcd", *pcCone);
			}
			else if (target_cone.index == 5 || target_cone.index == 6 || target_cone.index == 7)
			{
				std::string bPath = (save_path != std::string("")) ? save_path + "_CST" : "";
				pcCone = map_box_to_pc_noisefiltertest(pointCloud, target_cone, 0, 0, bPath);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pcCST.pcd", *pcCone);
			}
		}	
		if (target_guide.prob > 0)
		{
			std::string bPath = (save_path != std::string("")) ? save_path + "_Guide" : "";
			pcGuide = map_box_to_pc_noisefiltertest(pointCloud, target_guide, 0, 0, bPath);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pcGuide.pcd", *pcGuide);
		}

		if (jobLog) jobLogMessage("pcHole count: " + std::to_string(pcHole->points.size()));
		if (jobLog) jobLogMessage("pcCone count: " + std::to_string(pcCone->points.size()));
		if (jobLog) jobLogMessage("pcGuide count: " + std::to_string(pcGuide->points.size()));

		return true;
	}
	catch (std::exception& ex)
	{
		logMessage("[Target-PointCloud-Extraction] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Target-PointCloud-Extraction] Unknown Exception!");
		return false;
	}
}

bool target_pointcloud_extraction_preset(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, MAX_MIN_RANGE GRange, MAX_MIN_RANGE HRange, MAX_MIN_RANGE TRange, bool sprd_landed, bool usePreset_CC, bbx target_hole, pcl::PointCloud<pcl::PointXYZ>::Ptr& pcHole, pcl::PointCloud<pcl::PointXYZ>::Ptr& pcCone, pcl::PointCloud<pcl::PointXYZ>::Ptr& pcGuide, std::string save_path = "", bool jobLog = false)
{
	try
	{
		//Step1: PassThrough via ini set ROI.
		pcl::PointCloud<pcl::PointXYZ>::Ptr pt_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		singlePassThrough(pointCloud, GRange.MIN, GRange.MAX, HRange.MIN, HRange.MAX, TRange.MIN, TRange.MAX, pt_cloud);
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_preset_pt.pcd", *pt_cloud);

		//Step2: Voxelize pointcloud
		pcl::PointCloud<pcl::PointXYZ>::Ptr vx_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pc_VoxelDown(pt_cloud, 10.f, 10.f, 10.f, vx_cloud);
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_preset_vx.pcd", *vx_cloud);

		//Step3: Clustering pointcloud
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
		tree->setInputCloud(vx_cloud);

		std::vector<pcl::PointIndices> cluster_indices;
		pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
		ec.setClusterTolerance(20.f);   // mm
		ec.setMinClusterSize(100);
		ec.setMaxClusterSize(25000);
		ec.setSearchMethod(tree);
		ec.setInputCloud(vx_cloud);
		ec.extract(cluster_indices);

		int extra_range = 10, i = 0;
		std::vector<pcl::PointCloud<pcl::PointXYZ>> extracted_clusters;
		for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
			for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
			{
				cloud_cluster->points.push_back(vx_cloud->points[*pit]);
			}
			//extracted_clusters.push_back(*cloud_cluster);

			int c_max_x = 0, c_min_x = 0, c_max_y = 0, c_min_y = 0, c_max_z = 0, c_min_z = 0;
			auto center = get_center_of_cloud(*cloud_cluster, c_max_x, c_min_x, c_max_y, c_min_y, c_max_z, c_min_z);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_extrated(new pcl::PointCloud<pcl::PointXYZ>);
			singlePassThrough(pt_cloud, c_min_x - extra_range, c_max_x + extra_range, c_min_y, c_max_y, c_min_z - extra_range, c_max_z + extra_range, cloud_extrated);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_preset_cluster_" + std::to_string(i) + ".pcd", *cloud_extrated);

			extracted_clusters.push_back(*cloud_extrated);
			i++;
		}

		//Step4: Consider biggest two only.
		if (extracted_clusters.size() >= 2)
		{
			auto ec1 = extracted_clusters.at(0);
			auto ec2 = extracted_clusters.at(1);

			pcl::PointCloud<pcl::PointXYZ>::Ptr pcClust1(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr pcClust2(new pcl::PointCloud<pcl::PointXYZ>);

			*pcClust1 = ec1;
			*pcClust2 = ec2;
			
			int ec1_max_y, ec1_min_y;
			get_max_min_y(pcClust1, ec1_max_y, ec1_min_y);
			int ec1_center_y = ec1_min_y + (ec1_max_y - ec1_min_y) / 2;
			int ec2_max_y, ec2_min_y;
			get_max_min_y(pcClust2, ec2_max_y, ec2_min_y);
			int ec2_center_y = ec2_min_y + (ec2_max_y - ec2_min_y) / 2;

			if (ec1_center_y > ec2_center_y)
			{
				pcl::copyPointCloud(ec1, *pcHole);
				pcl::copyPointCloud(ec2, *pcCone);
			}
			else
			{
				pcl::copyPointCloud(ec2, *pcHole);
				pcl::copyPointCloud(ec1, *pcCone);
			}

			if (usePreset_CC) if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pcHole.pcd", *pcHole);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pcCone.pcd", *pcCone);
		}
		else if (extracted_clusters.size() == 1)
		{
			pcl::copyPointCloud(extracted_clusters.at(0), *pcCone);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pcCone.pcd", *pcCone);
		}

		if (!usePreset_CC)
		{
			std::string bPath = (save_path != std::string("")) ? save_path + "_Hole" : "";
			//logMessage(std::to_string(target_hole.x) + " " + std::to_string(target_hole.y) + " , " + std::to_string(x_limit) + " , " + std::to_string(z_limit));
			pcHole = map_box_to_pc_noisefiltertest(pointCloud, target_hole, 0, 0, bPath);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pcHole.pcd", *pcHole);
		}

		return true;
	}
	catch (std::exception& ex)
	{
		logMessage("[Target-PointCloud-Extraction] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Target-PointCloud-Extraction] Unknown Exception!");
		return false;
	}
}



bool position_detection_pca(JobInfo jobInfo, bool usePreset, bool isFrontSensor, bool target_on_right, std::string ChassisType, bbx target_hole, bbx target_cone, bbx target_guide, pcl::PointCloud<pcl::PointXYZ>::Ptr pcHole, pcl::PointCloud<pcl::PointXYZ>::Ptr pcCone, pcl::PointCloud<pcl::PointXYZ>::Ptr pcGuide, pcl::PointXYZ& hole_pos, pcl::PointXYZ& cone_pos, pcl::PointXYZ& guide_pos, std::string save_path = "", bool jobLog = false)
{
	try
	{
		if (pcHole->points.size() > 0)
		{
			hole_pos = pc_hole_detection_naive(pcHole, target_on_right, save_path);
			logMessage("Detected hole position: (T:" + std::to_string(hole_pos.x) + " , H:" + std::to_string(hole_pos.y) + " , G:" + std::to_string(hole_pos.z) + ")");
			if (jobLog) jobLogMessage("Detected hole position: (T:" + std::to_string(hole_pos.x) + " , H:" + std::to_string(hole_pos.y) + " , G:" + std::to_string(hole_pos.z) + ")");

			if (hole_pos.x != -10000)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr holeCloud(new pcl::PointCloud <pcl::PointXYZ>);
				holeCloud->points.push_back(hole_pos);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_pos.pcd", *holeCloud);
			}
		}

		if (pcCone->points.size() > 0)
		{
			if (target_cone.index == 1)
			{
				if (usePreset)
				{
					//preset용 logic -- can be XT or landed.
					cone_pos = pc_cone_detection_preset(pcCone, target_on_right, jobInfo.presetRangeX, save_path);
				}
				else
				{
					cone_pos = pc_cone_detection_naive(pcCone, target_on_right, save_path);
				}
				logMessage("Detected cone position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");
				if (jobLog) jobLogMessage("Detected cone position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");

				if (cone_pos.x != -10000 && save_path != std::string(""))
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr coneCloud(new pcl::PointCloud <pcl::PointXYZ>);
					coneCloud->points.push_back(cone_pos);
					pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_pos.pcd", *coneCloud);
				}
			}
			else if (target_cone.index == 2)
			{
				cone_pos = pc_landed_detection_naive(pcCone, target_on_right, save_path);
				logMessage("Detected landed position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");
				if (jobLog) jobLogMessage("Detected landed position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");

				if (cone_pos.x != -10000)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr coneCloud(new pcl::PointCloud <pcl::PointXYZ>);
					coneCloud->points.push_back(cone_pos);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_pos.pcd", *coneCloud);
				}
			}
			//IGH
			else if (target_cone.index == 3)
			{
				cone_pos = pc_IGH_detection_naive(pcCone, target_on_right, save_path);
				logMessage("Detected igh position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");
				if (jobLog) jobLogMessage("Detected igh position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");

				if (cone_pos.x != -10000)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr coneCloud(new pcl::PointCloud <pcl::PointXYZ>);
					coneCloud->points.push_back(cone_pos);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_igh_pos.pcd", *coneCloud);
				}
			}
			//IGHS
			else if (target_cone.index == 4)
			{
				cone_pos = pc_IGHS_detection_naive(pcCone, target_on_right, save_path);
				logMessage("Detected ighs position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");
				if (jobLog) jobLogMessage("Detected ighs position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");

				if (cone_pos.x != -10000)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr coneCloud(new pcl::PointCloud <pcl::PointXYZ>);
					coneCloud->points.push_back(cone_pos);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_ighs_pos.pcd", *coneCloud);
				}
			}
			//CSTF
			else if (target_cone.index == 5 && false)
			{
				cone_pos = pc_cone_detection_naive(pcCone, isFrontSensor, save_path);
				logMessage("Detected cone position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");
				if (jobLog) jobLogMessage("Detected cone position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");

				if (cone_pos.x != -10000)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr coneCloud(new pcl::PointCloud <pcl::PointXYZ>);
					coneCloud->points.push_back(cone_pos);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cstf_pos.pcd", *coneCloud);
				}
			}
			//CSTB
			else if (target_cone.index == 7 && false)
			{
				cone_pos = pc_cone_detection_naive(pcCone, target_on_right, save_path);
				logMessage("Detected cone position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");
				if (jobLog) jobLogMessage("Detected cone position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");

				if (cone_pos.x != -10000)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr coneCloud(new pcl::PointCloud <pcl::PointXYZ>);
					coneCloud->points.push_back(cone_pos);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cstb_pos.pcd", *coneCloud);
				}
			}
		}

		//CSTS, CSTSL
		if (pcGuide->points.size() > 0)
		{
			guide_pos = pc_CSTS_detection_naive(pcGuide, target_on_right, save_path);
			logMessage("Detected guide position: (T:" + std::to_string(guide_pos.x) + " , H:" + std::to_string(guide_pos.y) + " , G:" + std::to_string(guide_pos.z) + ")");
			if (jobLog) jobLogMessage("Detected guide position: (T:" + std::to_string(guide_pos.x) + " , H:" + std::to_string(guide_pos.y) + " , G:" + std::to_string(guide_pos.z) + ")");

			if (guide_pos.x != -10000)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr guideCloud(new pcl::PointCloud <pcl::PointXYZ>);
				guideCloud->points.push_back(guide_pos);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_guide_pos.pcd", *guideCloud);
			}
		}


		return true;
	}
	catch (std::exception& ex)
	{
		logMessage("[Position-Detection-PCA] " + std::string(ex.what()));
		if (jobLog) jobLogMessage("[Position-Detection-PCA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Position-Detection-PCA] Unknown Exception!");
		if (jobLog) jobLogMessage("[Position-Detection-PCA] Unknown Exception!");
		return false;
	}
}

//2025.08.06
//CLPS Detection using only 3D Point Cloud data.
//input: 3d base point cloud
//output: Separated or combined.
bool CLPS_Detection_PCA_Only(pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud, bool isLeftSensor, int& devY, std::string savePath = std::string(""))
{
	auto t_start = std::chrono::high_resolution_clock::now();
	bool blnSeparated = false;
	try
	{
		if (Offload_Cycle)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr pcPassThrough(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr pcVoxel(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr pcNoiseFilter(new pcl::PointCloud<pcl::PointXYZ>);

			//step1: Extract ROI based on preset
			int x_min_limit = 450;
			int x_max_limit = 450 + 200 + 1200;
			if (isLeftSensor)
			{
				x_min_limit = -1 * x_max_limit; //invert x limit for right sensor.
				x_max_limit = -450;
			}
			int y_min_limit = -1200;
			int y_max_limit = 500;

			int z_min_limit = 1500;
			int z_max_limit = 4000;

			pc_passThrough(pointcloud, x_min_limit, x_max_limit, y_min_limit, y_max_limit, z_min_limit, z_max_limit, pcPassThrough);
			if (savePath != std::string("") && pcPassThrough->points.size() > 0) pcl::io::savePCDFileBinaryCompressed(savePath + "_pcPassThrough.pcd", *pcPassThrough);

			//step2: Noise filter
			pc_NoiseFilter(pcPassThrough, 30, 0.7, pcNoiseFilter);
			if (savePath != std::string("") && pcNoiseFilter->points.size() > 0) pcl::io::savePCDFileBinaryCompressed(savePath + "_pcNoiseFilter.pcd", *pcNoiseFilter);

			//step3: Voxel Down Filter
			pc_VoxelDown(pcNoiseFilter, 50.0f, 50.0f, 50.0f, pcVoxel);
			if (savePath != std::string("") && pcVoxel->points.size() > 0) pcl::io::savePCDFileBinaryCompressed(savePath + "_pcVoxel.pcd", *pcVoxel);

			//step4: clustering
#pragma region Clustering
//auto clustering_time = std::chrono::high_resolution_clock::now();

			pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
			tree->setInputCloud(pcVoxel);

			std::vector<pcl::PointIndices> cluster_indices;
			pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
			ec.setInputCloud(pcVoxel);
			//ec.setClusterTolerance(12);
			ec.setClusterTolerance(100); //100mm Euclidean distance (between two points) tolerance for clustering.
			//ec.setMinClusterSize(1000);
			ec.setMinClusterSize(200);
			ec.setMaxClusterSize(20000);
			ec.setSearchMethod(tree);
			ec.extract(cluster_indices);

			int i = 0;
			int extra_range = 10;
			std::vector<pcl::PointCloud<pcl::PointXYZ>> extracted_clusters;
			std::vector<cloud_data> extracted_clusters_data;
			for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
				for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
				{
					cloud_cluster->points.push_back(pcVoxel->points[*pit]);
				}
				//extracted_clusters.push_back(*cloud_cluster);

				//auto cloud_extract_time = std::chrono::high_resolution_clock::now();
				int c_max_x = 0, c_min_x = 0, c_max_y = 0, c_min_y = 0, c_max_z = 0, c_min_z = 0;
				auto center = get_center_of_cloud(*cloud_cluster, c_max_x, c_min_x, c_max_y, c_min_y, c_max_z, c_min_z);

				//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_extrated(new pcl::PointCloud<pcl::PointXYZ>);
				//pc_passThrough(pcPassThrough, c_min_x - extra_range, c_max_x + extra_range, c_min_y, c_max_y, c_min_z - extra_range, c_max_z + extra_range, cloud_extrated);

				cloud_data cloud_data_extracted;
				cloud_data_extracted.pc_ptr = cloud_cluster;
				cloud_data_extracted.max_x = c_max_x;
				cloud_data_extracted.min_x = c_min_x;
				cloud_data_extracted.max_y = c_max_y;
				cloud_data_extracted.min_y = c_min_y;
				cloud_data_extracted.max_z = c_max_z;
				cloud_data_extracted.min_z = c_min_z;

				extracted_clusters_data.push_back(cloud_data_extracted);
				extracted_clusters.push_back(*cloud_cluster);

				//auto cloud_extract_Endtime = std::chrono::high_resolution_clock::now();
				//auto cloud_extract_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cloud_extract_Endtime - cloud_extract_time);
				//leave_a_message(log_heading + "Cloud extract algorithm= " + std::to_string(cloud_extract_duration.count()) + "msec", std::chrono::system_clock::now());

				if (savePath != std::string(""))pcl::io::savePCDFileBinaryCompressed(savePath + "_cluster_" + std::to_string(i) + ".pcd", *cloud_cluster);
				i++;
			}

			//auto clustering_Endtime = std::chrono::high_resolution_clock::now();
			//auto clustering_duration = std::chrono::duration_cast<std::chrono::milliseconds>(clustering_Endtime - clustering_time);
			//logMessage("Clustering algorithm= " + std::to_string(clustering_duration.count()) + "msec");
#pragma endregion

//Step5: Get Min H, Max H from biggest two clusters and check dev Y.
			if (extracted_clusters_data.size() > 0)
			{
				if (extracted_clusters_data.size() == 1)
				{
					//empty chassis or loaded chassis.
					if (extracted_clusters_data.at(0).max_y > 0)
					{
						//cluster data above sensor height -> loaded chassis.
					}
					else
					{
						//cluster data below sensor height -> empty chassis.
					}
				}
				else if (extracted_clusters_data.size() >= 2)
				{
					//take biggest two clusters.
					std::vector<cloud_data> two_biggest_clusters{};
					two_biggest_clusters.push_back(extracted_clusters_data.at(0));
					two_biggest_clusters.push_back(extracted_clusters_data.at(1));

					std::sort(two_biggest_clusters.begin(), two_biggest_clusters.end(), [](const cloud_data& a, const cloud_data& b) {
						return a.max_y > b.max_y; // Sort by max_y (height)
						});

					auto container_cluster = two_biggest_clusters.at(0);
					auto chassis_cluster = two_biggest_clusters.at(1);

					if (savePath != std::string(""))pcl::io::savePCDFileBinaryCompressed(savePath + "_container_clust.pcd", *container_cluster.pc_ptr);
					if (savePath != std::string(""))pcl::io::savePCDFileBinaryCompressed(savePath + "_chassis_clust.pcd", *chassis_cluster.pc_ptr);

					devY = static_cast<int>(container_cluster.min_y - chassis_cluster.max_y);
					if (devY > 200)
					{
						logMessage("CLPS-OK-PCA-ONLY: Chassis separated");
						if (clps_ok_detected_pca_only_count >= CLPS_NCOUNT)
						{
							if (!clps_ok_detected_pca_only)
							{
								logMessage("CLPS-OK Detected by PCA-Only");
								clps_ok_detected_pca_only = true;
							}
						}
						else
						{
							if (clps_ok_detected_pca_only_count < 20) clps_ok_detected_pca_only_count++;
						}
						blnSeparated = true;
					}
					else
					{
						if (clps_ok_detected_pca_only_count == 0)
						{
							if (clps_ok_detected_pca_only)
							{
								;
								//logMessage("CLPS-OK Detection Released by PCA-Only");
								//clps_ok_detected_pca_only = false;
							}
						}
						else
						{
							//if (clps_ok_detected_pca_only_count > 0) clps_ok_detected_pca_only_count--;
						}

						blnSeparated = false;
					}
				}
			}

			auto t_stop = std::chrono::high_resolution_clock::now();
			auto t_dur = std::chrono::duration_cast<std::chrono::milliseconds>(t_stop - t_start);
			logMessage("PCA-ONLY CLPS Proc Time= " + std::to_string(t_dur.count()) + "msec");
		}
		return blnSeparated;
	}
	catch (std::exception& ex)
	{
		logMessage("[CLPS-Detection-PCA-Only] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[CLPS-Detection-PCA-Only] Unknown Exception!");
		return false;
	}
}

bool CLPS_Detection_PCA(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, bool isLeftSensor, bbx target_hole, pcl::PointXYZ hole_pos, int g_data_limit, int t_data_limit, bool detected_hole, bool sprd_landed, int& pcUnderHole, int& clps_y_level, std::string save_path = "", bool jobLog = false)
{
	try
	{
		if (Offload_Cycle)
		{
			//Hole detected, but not chassis.
			if (detected_hole)
			{
				//if hole lifted above certain level.
				clps_y_level = 414 - (414 / 3);
				if (AVG_HOLE.get_count() >= CLPS_NCOUNT)
				{
					auto bb = AVG_HOLE.get_average();
					clps_y_level = bb.y - bb.h * 2.5;
				}
				if (jobLog) jobLogMessage("PCA-CLPS Y Level Threshold: " + std::to_string(clps_y_level));
				if (target_hole.y < clps_y_level)
				{
					//extract region beneath the detected hole.
					bbx targetRegion;
					
					targetRegion.x = target_hole.x;
					targetRegion.y = target_hole.y + target_hole.h;
					targetRegion.w = target_hole.w;
					targetRegion.h = target_hole.h * 2;
					
					if (isLeftSensor)
					{
						targetRegion.w *= 2;
					}
					else
					{
						targetRegion.x -= targetRegion.w;
						targetRegion.w *= 2;
					}
					
					targetRegion.center_x = targetRegion.x + targetRegion.w / 2;
					targetRegion.center_y = targetRegion.y + targetRegion.h / 2;

					//then extract.
					pcl::PointCloud<pcl::PointXYZ>::Ptr pcHoleBeneath(new pcl::PointCloud<pcl::PointXYZ>);
					pcHoleBeneath = map_box_to_pc(pointCloud, targetRegion);
					//if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_pcHoleBeneath.pcd", *pcHoleBeneath);

					//Do a pre filter 
					pcl::PointCloud<pcl::PointXYZ>::Ptr pcFilterZ(new pcl::PointCloud <pcl::PointXYZ>);
					pc_passThrough(pcHoleBeneath, 0, g_data_limit, "z", pcFilterZ);
					pcl::PointCloud<pcl::PointXYZ>::Ptr pcTargetBase(new pcl::PointCloud <pcl::PointXYZ>);
					pc_passThrough(pcFilterZ, 0, t_data_limit, "x", pcTargetBase);

					//Added filter
					if (hole_pos.x > -10000 && hole_pos.x != 0)
					{
						pc_passThrough(pcTargetBase, hole_pos.z - 100, hole_pos.z + 200, "z", pcTargetBase);
						pc_passThrough(pcTargetBase, hole_pos.x - 200, hole_pos.x + 200, "x", pcTargetBase);
					}

					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pcHoleBeneath_Target_" + std::to_string(pcTargetBase->points.size()) + ".pcd", *pcTargetBase);

					pcUnderHole = pcTargetBase->points.size();

					if (pcTargetBase->points.size() > 700)
					{
						if (clps_current_count_pca < CLPS_NCOUNT) clps_current_count_pca++;
						else
						{
							if (!clps_detected_pca)
							{
								logMessage("CLPS Detected by PCA");
								if (jobLog) jobLogMessage("CLPS Detected by PCA");
								clps_detected_pca = true;
							}
						}

						if (clps_ok_current_count_pca > 0)
						{
							clps_ok_current_count_pca--;
						}
						
						if (clps_ok_current_count_pca == 0)
						{
							if (clps_ok_detected_pca) clps_ok_detected_pca = false;
						}
						
					}
					else
					{
						if (clps_ok_current_count_pca < CLPS_NCOUNT) clps_ok_current_count_pca++;
						else
						{
							if (!clps_ok_detected_pca)
							{
								logMessage("CLPS-OK Detected by PCA");
								if (jobLog) jobLogMessage("CLPS-OK Detected by PCA");
								clps_ok_detected_pca = true;
							}
						}

						if (clps_current_count_pca > 0) clps_current_count_pca--;
						if (clps_current_count_pca == 0)
						{
							if (clps_detected_pca) clps_detected_pca = false;
						}
					}
				}
				else
				{
					///below the threshold height.
					if (clps_current_count_pca > 0) clps_current_count_pca--;
					if (clps_current_count_pca == 0)
					{
						if (clps_detected_pca) clps_detected_pca = false;
					}
					if (clps_ok_current_count_pca > 0) clps_ok_current_count_pca = 0;
					if (clps_ok_detected_pca) clps_ok_detected_pca = false;
				}
			}
			else if (sprd_landed)
			{
				//release both clps, clps-ok.
				if (clps_current_count_pca > 0) clps_current_count_pca = 0;
				if (clps_detected_pca) clps_detected_pca = false;
				if (clps_ok_current_count_pca > 0)
				{
					clps_ok_current_count_pca--;
				}
				if (clps_ok_current_count_pca == 0)
				{
					if (clps_ok_detected_pca) clps_ok_detected_pca = false;
				}
			}
		}

		return true;
	}
	catch (std::exception& ex)
	{
		logMessage("[CLPS-Detection-PCA] " + std::string(ex.what()));
		if (jobLog) jobLogMessage("[CLPS-Detection-PCA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[CLPS-Detection-PCA] Unknown Exception!");
		if (jobLog) jobLogMessage("[CLPS-Detection-PCA] Unknown Exception!");
		return false;
	}
}

bool Pre_Land_Chassis_Position_PCA(bbx target_cone, pcl::PointXYZ cone_pos, pcl::PointCloud<pcl::PointXYZ>::Ptr pcCone, bool jobLog = false)
{
	try
	{
		if (target_cone.prob > 0)
		{
			//initial chassis position.
			if (AVG_CONE_PCA.get_count() < LDO_NCOUNT)
			{
				if (target_cone.prob >= 75 && pcCone->points.size() > 0)
				{
					if (cone_pos.x != -10000)
					{
						AVG_CONE_PCA.update(cone_pos);
					}
				}
			}
			else
			{
				if (!AVG_CONE_PCA.get_set())
				{
					auto bb = AVG_CONE_PCA.get_average();
					logMessage("Pre-Land Chassis Position PCA: (T: " + std::to_string(bb.x) + " , H: " + std::to_string(bb.y) + " , G: " + std::to_string(bb.z));
					if (jobLog) jobLogMessage("Pre-Land Chassis Position PCA: (T: " + std::to_string(bb.x) + " , H: " + std::to_string(bb.y) + " , G: " + std::to_string(bb.z));

					AVG_CONE_PCA.set();
				}
			}
		}
	}
	catch (std::exception& ex)
	{
		logMessage("[Pre-Land-Chassis-Position-PCA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Pre-Land-Chassis-Position-PCA] Unknown Exception!");
		return false;
	}
}

bool deviation_output_pca(bool isFrontSensor, bool Target_on_Right, bool isMount, bool isOffload, std::string ChassisType, pcl::PointXYZ hole_pos, pcl::PointXYZ cone_pos, pcl::PointXYZ guide_pos, bool sprd_landed, int& devOut_pca_x, int& devOut_pca_y, int& devOut_pca_z, bool& landout_detected, bool& clps_detected, bool& clps_ok_detected, bool jobLog = false)
{
	try
	{
		if (hole_pos.x != -10000 && cone_pos.x != -10000)
		{
			//based on TUAS-3DSP.
			int x_offset = 14;
			int z_offset = 26;

			//Hole - Cone
			if (!Target_on_Right) devOut_pca_x = cone_pos.x - (hole_pos.x - x_offset);
			else devOut_pca_x = cone_pos.x - (hole_pos.x + x_offset);
			
			devOut_pca_y = cone_pos.y - hole_pos.y;
			devOut_pca_z = cone_pos.z - (hole_pos.z + z_offset);


			int landout_threshold = 50;
			if (ChassisType == "CST") landout_threshold = 300;

			if (isMount)
			{
				if (sprd_landed || devOut_pca_y > -50)
				{
					if (std::abs(devOut_pca_x) > landout_threshold || std::abs(devOut_pca_z) > landout_threshold)
					{
						if (landout_current_count_pca < LDO_NCOUNT)
						{
							landout_current_count_pca++;
						}
						if (landout_current_count_pca >= LDO_NCOUNT)
						{
							if (!landout_detected)
							{
								logMessage("Landout detected by PCA!");
								if (jobLog) jobLogMessage("Landout detected by PCA!");
							}
							landout_detected = true;
						}
					}
					else if (sprd_landed && devOut_pca_y < -50)
					{
						//Detected Cone is above detected hole.
						if (landout_current_count_pca < LDO_NCOUNT)
						{
							landout_current_count_pca++;
						}
						if (landout_current_count_pca >= LDO_NCOUNT)
						{
							if (!landout_detected)
							{
								logMessage("Landout detected by PCA - DevY: " + std::to_string(devOut_pca_y) + "!");
								if (jobLog) jobLogMessage("Landout detected by PCA - DevY: " + std::to_string(devOut_pca_y) + "!");
							}
							landout_detected = true;
						}
					}
					else
					{
						if (landout_current_count_pca > 0) landout_current_count_pca--;
						if (landout_current_count_pca == 0)
						{
							landout_detected = false;
						}
					}
				}			
				else
				{
					if (landout_current_count_pca > 0) landout_current_count_pca--;
					if (landout_current_count_pca == 0)
					{
						landout_detected = false;
					}
				}
			}
			else if (isOffload)
			{
				if (devOut_pca_y < -300)
				{
					if (clps_ok_current_count_pca < CLPS_NCOUNT)
					{
						clps_ok_current_count_pca++;
					}
					if (clps_ok_current_count_pca >= CLPS_NCOUNT)
					{
						if (!clps_ok_detected)
						{
							logMessage("CLPS-OK Detected by PCA!");
							if (jobLog) jobLogMessage("CLPS-OK Detected by PCA!");
						}
						clps_ok_detected = true;
					}
				}
			}
		}
		else if (hole_pos.x != -10000 && guide_pos.x != -10000)
		{
			//based on TUAS-3DSP.
			int x_offset = 14;
			int z_offset = 26;

			//Hole - Cone
			if (!Target_on_Right) devOut_pca_x = cone_pos.x - (hole_pos.x - x_offset);
			else devOut_pca_x = cone_pos.x - (hole_pos.x + x_offset);

			devOut_pca_y = cone_pos.y - hole_pos.y;
			devOut_pca_z = cone_pos.z - (hole_pos.z + z_offset);

			int landout_threshold = 50;
			if (ChassisType == "CST") landout_threshold = 300;

			if (isMount)
			{
				if (sprd_landed || devOut_pca_y > -50)
				{
					if (std::abs(devOut_pca_z) > landout_threshold)//if (std::abs(devOut_pca_x) > landout_threshold || std::abs(devOut_pca_z) > landout_threshold)
					{
						if (landout_current_count_pca < LDO_NCOUNT)
						{
							landout_current_count_pca++;
						}
						if (landout_current_count_pca >= LDO_NCOUNT)
						{
							if (!landout_detected)
							{
								logMessage("Landout detected by PCA!");
								if (jobLog) jobLogMessage("Landout detected by PCA!");
							}
							landout_detected = true;
						}
					}
					else if (sprd_landed && devOut_pca_y < -50)
					{
						//Detected Cone is above detected hole.
						if (landout_current_count_pca < LDO_NCOUNT)
						{
							landout_current_count_pca++;
						}
						if (landout_current_count_pca >= LDO_NCOUNT)
						{
							if (!landout_detected)
							{
								logMessage("Landout detected by PCA - DevY: " + std::to_string(devOut_pca_y) + "!");
								if (jobLog) jobLogMessage("Landout detected by PCA - DevY: " + std::to_string(devOut_pca_y) + "!");
							}
							landout_detected = true;
						}
					}
					else
					{
						if (landout_current_count_pca > 0) landout_current_count_pca--;
						if (landout_current_count_pca == 0)
						{
							landout_detected = false;
						}
					}
				}
				else
				{
					if (landout_current_count_pca > 0) landout_current_count_pca--;
					if (landout_current_count_pca == 0)
					{
						landout_detected = false;
					}
				}
			}
			else if (isOffload)
			{
				if (devOut_pca_y < -300)
				{
					if (clps_ok_current_count_pca < CLPS_NCOUNT)
					{
						clps_ok_current_count_pca++;
					}
					if (clps_ok_current_count_pca >= CLPS_NCOUNT)
					{
						if (!clps_ok_detected)
						{
							logMessage("CLPS-OK Detected by PCA!");
							if (jobLog) jobLogMessage("CLPS-OK Detected by PCA!");
						}
						clps_ok_detected = true;
					}
				}
			}
		}

		logMessage("PCA Dev Out: (T: " + std::to_string(devOut_pca_x) + " H: " + std::to_string(devOut_pca_y) + " G: " + std::to_string(devOut_pca_z) + ")");
		if (jobLog) jobLogMessage("PCA Dev Out: (T: " + std::to_string(devOut_pca_x) + " H: " + std::to_string(devOut_pca_y) + " G: " + std::to_string(devOut_pca_z) + ")");
		logMessage("PCA LandOut Detected: " + std::to_string(landout_detected_pca));
		if (jobLog) jobLogMessage("PCA LandOut Detected PCA: " + std::to_string(landout_detected_pca));

		return true;
	}
	catch (std::exception& ex)
	{
		logMessage("[Deviation-Output-PCA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Deviation-Output-PCA] Unknown Exception!");
		return false;
	}
}

//Cone/guide side selection for the ROI backup path (site rule, 2026-06).
//Left/right runs along the GANTRY axis: G goes -(left) to +(right) in the
//camera view. FS/CS: -G is toward the chassis front; RS: +G is toward the
//chassis front (sensor on the other side of the lane).
// - Front sensor: 45ft chassis with 40ft container job -> right cone;
//   any other job -> left (or only) cone.
// - Center sensor: Big job -> left cone; Small job -> right cone.
// - Rear sensor: not specified yet; defaults to left.
bool roi_cone_on_right(std::string cam_pos, JobInfo jobInfo)
{
	if (cam_pos == "Center") return (jobInfo.jobPos == "Small");
	if (cam_pos == "Front") return (jobInfo.chassisLen == 45 && jobInfo.jobSize == 40);
	return false;
}

//2025.11.26 Endfeed용으로 수정.
//Build the per-job ROI params (preset world window + per-config XT cone height).
//Shared by FORCE_ROI_MODE and the ROI_FALLBACK path so the cone-height table lives once.
void fillRoiParams(const JobInfo& jobInfo, ChassisRoiParams& roiPrm)
{
	roiPrm.xMin = (float)jobInfo.presetRangeX.MIN;
	roiPrm.xMax = (float)jobInfo.presetRangeX.MAX;
	roiPrm.yMin = (float)HEIGHT_MMR.MIN;
	roiPrm.yMax = (float)HEIGHT_MMR.MAX;
	roiPrm.zMin = (float)TROLLEY_MMR.MIN;
	roiPrm.zMax = (float)TROLLEY_MMR.MAX;
	roiPrm.lockK = ROI_LOCK_K;
	roiPrm.lockTol = (float)ROI_LOCK_TOL;
	roiPrm.moveTol = (float)ROI_MOVE_TOL;

	//Per-config cone HEIGHT (world Y) -- the CHS20 wheel is nearer-Z than the cone, so a
	//runtime estimate would pick the wheel. XT only; IGH/CST leave coneY unset (-10000).
	if (jobInfo.chassisType == "XT")
	{
		if (CURRENT_SENSOR_POSITION == "FRONT")
		{
			if (jobInfo.chassisLen == 45 && jobInfo.jobSize == 40)         roiPrm.coneY = (float)presetPos.F_XT_4XR_CY;
			else if (jobInfo.chassisLen == 45 && jobInfo.jobPos == "Small") roiPrm.coneY = (float)presetPos.F_XT_20R_CY;
			else if (jobInfo.chassisLen == 40 && jobInfo.jobPos == "Small") roiPrm.coneY = (float)presetPos.F_XT_20F_CY;
			else                                                           roiPrm.coneY = (float)presetPos.F_XT_4XF_CY;
		}
		else
		{
			if (jobInfo.jobPos == "Big")       roiPrm.coneY = (float)presetPos.R_XT_20F_CY;
			else if (jobInfo.chassisLen == 40) roiPrm.coneY = (float)presetPos.R_XT_4XF_CY;
			else                               roiPrm.coneY = (float)presetPos.R_XT_4XR_CY;
		}
	}
	else if (jobInfo.chassisType == "IGH")
	{
		//IGH crop is SEPARATE from XT -- own X/Y/Z, NOT the shared HEIGHT_MMR/TROLLEY_MMR or the
		//colliding IGH_4XF RNG key. Sized from the GT spread (roi_igh_eval / igh_spread analysis):
		//the CS(REAR) guides span Z~1222..1842 and X wide (merged ~480mm), larger than XT's fixed
		//window. TODO: lift these to dedicated IGH INI keys.
		roiPrm.yMin = -600.f; roiPrm.yMax = 100.f;			//Y (Height): guide + targets (up to ~56)
		if (CURRENT_SENSOR_POSITION == "FRONT")				//FS sensor
		{
			roiPrm.xMin = 500.f;  roiPrm.xMax = 1150.f;		//X (Gantry): GT 796..927 + margin
			roiPrm.zMin = 1400.f; roiPrm.zMax = 2050.f;		//Z (Trolley): GT 1521..1776 + margin
			roiPrm.coneY = (jobInfo.chassisLen == 45) ? -7.f : -4.f;	//guide TOP height
		}
		else												//CS sensor (REAR app, Center)
		{
			roiPrm.xMin = 350.f;  roiPrm.xMax = 1300.f;		//X: GT 512..1035 (merged whole)
			roiPrm.zMin = 1150.f; roiPrm.zMax = 2050.f;		//Z: GT 1222..1842 (near-Z front included)
			roiPrm.coneY = (jobInfo.chassisLen == 45) ? -66.f : -77.f;
			//MERGED long guide (CHS40, ~480mm): widen the region xspan so the WHOLE guide is
			//passed on (default 300mm crops both ends). cx is already the merged centre.
			if (jobInfo.chassisLen == 40) roiPrm.xspan = 600.f;
		}
	}

	//WIDE CROP override -- close-range XT Center 40chs/20job (F app -> Center sensor, 20F).
	//Widen ONLY the crop p.x window (the Gantry / front-back axis the two cones split along)
	//so both cones are always captured; coneColumns + the fixed side-rule then pick the
	//target end. presetRangeX (used for VA mis-pick validation) is left untouched.
	if (jobInfo.chassisType == "XT" && CURRENT_SENSOR_POSITION == "FRONT"
		&& jobInfo.chassisLen == 40 && jobInfo.jobPos == "Small"
		&& ROI_WIDE_20F_MAX != -10000 && ROI_WIDE_20F_MIN != -10000)
	{
		roiPrm.xMin = (float)ROI_WIDE_20F_MIN;
		roiPrm.xMax = (float)ROI_WIDE_20F_MAX;
	}
}

//IGH target = the GUIDE's TOP-LEFT corner. Reuses the XT near-face corner detector for the
//lateral/depth (x,z), then OVERRIDES the height to the GUIDE TOP (dense-max Y at the corner
//column over the SOR'd region) instead of the XT plate-corner height (~90mm below the guide
//top). Port of tools/roi_igh_eval.py:target_ightop (dT<=9 on the 3 single-guide configs).
//A MERGED long guide (CS 40chs) is one continuous structure, so `region` is the WHOLE guide
//and the corner is the whole-guide corner (site targeting offset still TBD).
//`region` = the extracted guide+plate region (chassis_roi_extract_region output).
pcl::PointXYZ igh_target_from_region(const pcl::PointCloud<pcl::PointXYZ>::Ptr& region, bool target_on_right)
{
	pcl::PointXYZ inv(-10000.f, -10000.f, -10000.f);
	if (!region || (int)region->points.size() < 30) return inv;
	auto clean = chassis_roi_sor(region);
	if ((int)clean->points.size() < 30) clean = region;
	pcl::PointCloud<pcl::PointXYZ>::Ptr face = chassis_roi_near_face(clean, 80.f);	//upper-half + near-Z
	pcl::PointXYZ nc = pc_cone_detection_naive(face, target_on_right, "");			//= new_target corner
	if (nc.x == -10000.f) return inv;
	chassis_roi_sparse_edge_rescue(face, target_on_right, nc);
	//Y -> GUIDE TOP: dense-max Y in the corner X-column over the (SOR'd) region.
	std::vector<float> colY;
	for (const auto& p : clean->points)
		if (std::fabs(p.x - nc.x) <= 60.f) colY.push_back(p.y);
	if ((int)colY.size() >= 10) nc.y = chassis_roi_detail::denseExtreme(colY, 40.f, 10, true);
	return nc;
}

//Localize the chassis cone from the preset ROI (VA-independent), returning a WORLD point
//(-10000 sentinel on failure). Self-contained: does NOT touch the VA _pcCone/_cone_pos.
//XT uses the near-Z-face refinement (same chain as FORCE_ROI); IGH uses pc_IGH_detection.
pcl::PointXYZ computeRoiConePos(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, bool target_on_right,
	const JobInfo& jobInfo, const std::string& savePath)
{
	pcl::PointXYZ result(-10000.f, -10000.f, -10000.f);
	try
	{
		ChassisRoiParams prm; fillRoiParams(jobInfo, prm);
		pcl::PointXYZ coarse(-10000.f, -10000.f, -10000.f);
		auto crop = chassis_roi_crop(pointCloud, prm.xMin, prm.xMax, prm.yMin, prm.yMax, prm.zMin, prm.zMax, savePath, "ROI_FB");
		auto region = chassis_roi_extract_region(crop, target_on_right, prm, coarse, savePath, "ROI_FB");
		if (!region || region->points.size() == 0) return result;

		if (jobInfo.chassisType == "IGH")
		{
			result = igh_target_from_region(region, target_on_right);	//guide top-left corner
		}
		else //XT
		{
			auto clean = chassis_roi_sor(region);
			auto face = chassis_roi_near_face(clean, 80.f);
			pcl::PointXYZ nc = pc_cone_detection_naive(face, target_on_right, "");
			if (nc.x != -10000.f)
			{
				chassis_roi_sparse_edge_rescue(face, target_on_right, nc);
				float ch = chassis_roi_corner_height(face, nc.x, nc.z);
				if (ch != -10000.f) nc.y = ch;
				result = nc;
			}
		}
	}
	catch (...) {}
	return result;
}

ProcessResults ProcessLogic(bool isFrontSensor, JobInfo jobInfo, const cv::Mat image, pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, std::string savePath, bool jobLog, bool debugMode, std::string& logLines)
{
	ProcessResults procResults;

	jobInfo.print_jobInfo();

	std::string cam_pos = "Front";
	if (current_lane_camera.find("CS") != std::string::npos)
		cam_pos = "Center";
	else if (current_lane_camera.find("RS") != std::string::npos)
		cam_pos = "Rear";

	bool target_on_right = (cam_pos == "Center" && (jobInfo.jobPos == "Big" || jobInfo.chassisLen == 20));
	logMessage("Target on right: " + std::to_string(target_on_right) + " , job pos:" + jobInfo.jobPos + " , ChassisLen:" + std::to_string(jobInfo.chassisLen));

	cv::Mat res_image = image.clone();

	//VA Inference
	int det_count = 0;
	std::vector<rectangle_info> det_results;
	std::vector<std::vector<bbx>> det_sorted_objects(class_names.size(), std::vector<bbx>(0));
	//auto inference_status = trt_inference(trt_detector, image, ref(res_image), ref(det_results), ref(det_sorted_objects), ref(det_count), savePath);
	auto inference_status = yolo_inference(yolo_detector, image, ref(res_image), ref(det_results), ref(det_sorted_objects), ref(det_count), savePath);
	//Intrim variables
	bool usePreset = false;
	bool usePreset_CC = false;

	bool _detected_hole = false, _detected_cone = false;
	std::string _detected_chassis_type = "";

	bbx _target_hole, _target_cone, _target_guide;
	int _cntrX = -10000, _cntrY = -10000, _cntrProb = 0;
	int _coneX = -10000, _coneY = -10000, _coneProb = 0;

	int _dvOutX = -10000, _dvOutY = -10000;
	int _dvOutX_mm = -10000, _dvOutY_mm = -10000;	

	bool _sprd_landed_trigger = false;
	bool _eff_landed = false;	//PLC landed bit in live, visual trigger in debug/batch

	bool _landout_detected = false;
	bool _clps_detected = false, _clps_ok_detected = false;

	//For now; PCA only with VA-Mapping.
	pcl::PointCloud<pcl::PointXYZ>::Ptr _pcHole(new pcl::PointCloud <pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr _pcCone(new pcl::PointCloud <pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr _pcGuide(new pcl::PointCloud <pcl::PointXYZ>);

	pcl::PointXYZ _hole_pos(-10000, -10000, -10000);
	pcl::PointXYZ _cone_pos(-10000, -10000, -10000);
	pcl::PointXYZ _guide_pos(-10000, -10000, -10000);

	int _dvOut_pca_x = -10000, _dvOut_pca_y = -10000, _dvOut_pca_z = -10000;
	bool _landout_detected_pca = false, _clps_detected_pca = false, _clps_ok_detected_pca = false;

	//Combined Results.
	bool _Out_landout_detected = false;
	bool _Out_clps_detected = false;
	bool _Out_clps_ok_detected = false;

	if (det_count == 0)
	{
		logMessage("No acceptable inference results are avaialbe from VA detector.");
		_detected_chassis_type = "unknown";
	}
	else
	{
#pragma region VA Processing
		_detected_chassis_type = chassis_type_selection_va(det_sorted_objects, jobLog);
		if (_detected_chassis_type != "")
		{
			logMessage("Detected Chassis Type from VA: " + _detected_chassis_type);
			if (jobLog) jobLogMessage("Detected Chassis Type from VA: " + _detected_chassis_type);
		}

		target_selections_va(isFrontSensor, cam_pos, jobInfo.jobPos, _detected_chassis_type, jobInfo.chassisLen, jobInfo.jobSize, det_sorted_objects, ref(_target_hole), ref(_target_cone), ref(_target_guide), ref(_detected_hole), ref(_detected_cone), false);

		if (_target_hole.prob > 0)
		{
			//Hole: Red.
			cv::Scalar clr = cv::Scalar(0, 0, 255);
			cv::rectangle(res_image, cv::Rect(_target_hole.x, _target_hole.y, _target_hole.w, _target_hole.h), clr);

			//put-text
			std::string label = _target_hole.label + "," + std::to_string((int)_target_hole.prob) + "%";
			cv::putText(res_image, label, cv::Point(_target_hole.x, _target_hole.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 2);
		}
		if (_target_cone.prob > 0)
		{
			//Cone: Green.
			cv::Scalar clr = cv::Scalar(0, 255, 0);
			cv::rectangle(res_image, cv::Rect(_target_cone.x, _target_cone.y, _target_cone.w, _target_cone.h), clr);

			//put-text
			std::string label = _target_cone.label + "," + std::to_string((int)_target_cone.prob) + "%";
			cv::putText(res_image, label, cv::Point(_target_cone.x, _target_cone.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 2);
		}
		if (_target_guide.prob > 0)
		{
			//Side: Blue.
			cv::Scalar clr = cv::Scalar(255, 0, 0);
			cv::rectangle(res_image, cv::Rect(_target_guide.x, _target_guide.y, _target_guide.w, _target_guide.h), clr);

			//put-text
			std::string label = _target_guide.label + "," + std::to_string((int)_target_guide.prob) + "%";
			cv::putText(res_image, label, cv::Point(_target_guide.x, _target_guide.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 2);
		}

		pre_land_chassis_position_va(jobInfo.isMountCycle, jobInfo.isOffloadCycle, _target_cone, _target_hole);

		_sprd_landed_trigger = debug_landed_trigger(isFrontSensor, _detected_chassis_type, _target_hole, _target_cone, _target_guide);
		//Effective landed = the RELIABLE PLC bit in live (aggregates all 4 spreader
		//corners); the visual trigger only in debug/batch where no PLC is connected.
		//The PLC bit was parsed but UNUSED before 2026-07 (visual-only gating).
		_eff_landed = (MODE_DEBUG || DEBUG_BATCH_JOB || DEBUG_WITH_FILES) ? _sprd_landed_trigger : SPRD_Landed;

		deviation_output_va(isFrontSensor, cam_pos, _detected_chassis_type, _target_hole, _target_cone, _target_guide, ref(_cntrX), ref(_cntrY), ref(_cntrProb), ref(_coneX), ref(_coneY), ref(_coneProb), ref(_dvOutX), ref(_dvOutY), ref(_dvOutX_mm), ref(_dvOutY_mm), jobLog);

#pragma region landout detection
		if (jobInfo.isMountCycle)
		{
			_landout_detected = landOut_detected_va(jobInfo.isMountCycle, _detected_chassis_type, _target_hole, _target_cone, _target_guide, _sprd_landed_trigger, _detected_hole, ref(devOut_LDO_x), ref(devOut_LDO_y));
		}
#pragma endregion

#pragma region clps detection
		//_eff_landed (PLC bit in live): gates the VA CLPS-OK paths and drives the CST
		//rule -- site premise: CST has no locks, PLC landed OFF alone may assert CLPS-OK.
		if (jobInfo.isOffloadCycle) clps_detection_va(_detected_chassis_type, _target_hole, _target_cone, _target_guide, _eff_landed, ref(_clps_detected), ref(_clps_ok_detected), jobLog);
#pragma endregion


#pragma endregion

		logLines += std::to_string(_target_hole.x) + ";" + std::to_string(_target_hole.y) + ";" + std::to_string(_target_hole.w) + ";" + std::to_string(_target_hole.h) + ";" + std::to_string(_target_hole.prob) + ";";
		logLines += std::to_string(_target_cone.index) + ";" + std::to_string(_target_cone.x) + ";" + std::to_string(_target_cone.y) + ";" + std::to_string(_target_cone.w) + ";" + std::to_string(_target_cone.h) + ";" + std::to_string(_target_cone.prob) + ";";
		logLines += std::to_string(_target_guide.x) + ";" + std::to_string(_target_guide.y) + ";" + std::to_string(_target_guide.w) + ";" + std::to_string(_target_guide.h) + ";" + std::to_string(_target_guide.prob) + ";";

		logLines += std::to_string(_dvOutX) + ";" + std::to_string(_dvOutY) + ";";
		logLines += std::to_string(_dvOutX_mm) + ";" + std::to_string(_dvOutY_mm) + ";";
		//2025.03.11
		logLines += std::to_string(_sprd_landed_trigger) + ";" + std::to_string(_landout_detected) + ";" + std::to_string(_clps_detected) + ";" + std::to_string(_clps_ok_detected) + ";";

		if (debugMode)
		{
			if (_sprd_landed_trigger) cv::putText(res_image, std::string("LANDED:TRUE"), cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
			else cv::putText(res_image, std::string("LANDED:FALSE"), cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

			if (jobInfo.isMountCycle)
			{
				//Landout by logic
				if (_landout_detected) cv::putText(res_image, std::string("LANDOUT_VA:TRUE"), cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
				else cv::putText(res_image, std::string("LANDOUT_VA:FALSE"), cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

				cv::putText(res_image, std::string("LDO-VA Dev: ") + std::to_string(devOut_LDO_x) + "," + std::to_string(devOut_LDO_y), cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
			}
			else if (jobInfo.isOffloadCycle)
			{
				//CLPS - VA
				if (_clps_detected) cv::putText(res_image, std::string("CLPS_VA:TRUE"), cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
				else cv::putText(res_image, std::string("CLPS_VA:FALSE"), cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

				//CLPS-OK - VA
				if (_clps_ok_detected) cv::putText(res_image, std::string("CLPS-OK_VA:TRUE"), cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
				else cv::putText(res_image, std::string("CLPS-OK_VA:FALSE"), cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
			}
		}
	
	}

	if (_target_hole.prob == 0)
	{
		usePreset_CC = true;
	}

	//FORCE_ROI_MODE: preset-ROI crop -> existing detector -> tracker + A/B.
	ChassisRoiParams roiPrm;
	bool _ab_valid = false;
	int _ab_dT = -10000, _ab_dH = -10000, _ab_dG = -10000;

#pragma region FORCE_ROI_MODE - setup (test/backup path)
	bbx _va_chassis_ab;	//VA chassis detection kept aside for A/B comparison.
	if (FORCE_ROI_MODE)
	{
		//Chassis type comes from the job data; VA may be absent entirely here.
		if (_detected_chassis_type == "" || _detected_chassis_type == "unknown")
			_detected_chassis_type = jobInfo.chassisType;

		_va_chassis_ab = (_target_guide.prob > _target_cone.prob) ? _target_guide : _target_cone;

		fillRoiParams(jobInfo, roiPrm);
	}
#pragma endregion

	if (!FORCE_ROI_MODE && _target_cone.prob == 0)
	{
		usePreset = true;
		_target_cone = jobInfo.presetBox;
		_target_cone.prob = 50;

		if (jobInfo.chassisType == "XT") _target_cone.index = 1;
		else if (jobInfo.chassisType == "IGH") _target_cone.index = 3;
		else if (jobInfo.chassisType == "CST") 
		{
			_target_guide = _target_cone;
			_target_cone = bbx();
			_target_guide.index = 6;
		}

		logMessage("Using preset VA.");

		//Cone: Pink.
		cv::Scalar clr = cv::Scalar(203, 192, 255);
		if (_target_cone.prob >= 50)
		{
			cv::rectangle(res_image, cv::Rect(_target_cone.x, _target_cone.y, _target_cone.w, _target_cone.h), clr);

			//put-text
			std::string label = _target_cone.label + "," + std::to_string((int)_target_cone.prob) + "%";
			cv::putText(res_image, label, cv::Point(_target_cone.x, _target_cone.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 2);
		}
		else if (_target_guide.prob >= 50)
		{
			cv::rectangle(res_image, cv::Rect(_target_guide.x, _target_guide.y, _target_guide.w, _target_guide.h), clr);

			//put-text
			std::string label = _target_guide.label + "," + std::to_string((int)_target_guide.prob) + "%";
			cv::putText(res_image, label, cv::Point(_target_guide.x, _target_guide.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 2);
		}
	}

	//A/B debug: keep VA's own mapped cone target (via the full detector chain)
	//to compare against the ROI-cropped result. Computed here before the ROI
	//crop overwrites _target_cone/_pcCone.
	pcl::PointXYZ _ab_va_cone(-10000.f, -10000.f, -10000.f);
	if (FORCE_ROI_MODE && _va_chassis_ab.prob >= 70 && jobInfo.chassisType != "CST")
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr pcVa(new pcl::PointCloud<pcl::PointXYZ>);
		*pcVa = *map_box_to_pc_noisefiltertest(pointCloud, _va_chassis_ab, 0, 0, "");
		if (savePath != std::string("") && pcVa->points.size() > 0) pcl::io::savePCDFileBinaryCompressed(savePath + "_AB_va_body.pcd", *pcVa);
		if (pcVa->points.size() > 0)
		{
			bbx vaCone = _va_chassis_ab;
			if (vaCone.index < 1) vaCone.index = (jobInfo.chassisType == "IGH") ? 3 : 1;
			//Same detector the ROI path uses, on the VA box crop -> apples to apples.
			if (vaCone.index == 1) _ab_va_cone = pc_cone_detection_naive(pcVa, target_on_right, "");
			else if (vaCone.index == 3) _ab_va_cone = pc_IGH_detection_naive(pcVa, target_on_right, "");
			else _ab_va_cone = pc_cone_detection_naive(pcVa, target_on_right, "");

			//Save VA's cone POINT too (for the 3-way GT/ROI/VA visual overlay).
			if (savePath != std::string("") && _ab_va_cone.x != -10000.f)
			{
				pcl::PointCloud<pcl::PointXYZ> vc; vc.points.push_back(_ab_va_cone);
				vc.width = 1; vc.height = 1; vc.is_dense = true;
				pcl::io::savePCDFileBinaryCompressed(savePath + "_AB_va_conePos.pcd", vc);
			}
		}
	}

#pragma region Target pointcloud extraction via VA-Based
	pcl::PointXYZ _roi_cone_pos(-10000.f, -10000.f, -10000.f);
	if (FORCE_ROI_MODE)
	{
		//Hole stays VA-based (container/cornercasting phase is the next stage).
		if (_target_hole.prob > 0)
		{
			std::string bPath = (savePath != std::string("")) ? savePath + "_Hole" : "";
			_pcHole = map_box_to_pc_noisefiltertest(pointCloud, _target_hole, 0, 0, bPath);
			if (savePath != std::string("") && _pcHole->points.size() > 0) pcl::io::savePCDFileBinaryCompressed(savePath + "_pcHole.pcd", *_pcHole);
		}

		//Crop the preset world ROI (cone-height Y band + lane-bordered trolley),
		//then EXTRACT the target cone+plate REGION (per-column near-Z candidate ->
		//side-rule -> tight lateral/depth/height box). This region REPLACES the VA
		//mapping: it is fed as _pcCone (or _pcGuide for CST) so position_detection_pca
		//localizes it with the same pc_*_detection chain VA uses.
		std::string roiTag = chassisRoiTracker.locked() ? "ROI_TRK" : "ROI_ACQ";
		auto roiCrop = chassis_roi_crop(pointCloud, roiPrm.xMin, roiPrm.xMax, roiPrm.yMin, roiPrm.yMax, roiPrm.zMin, roiPrm.zMax, savePath, roiTag);
		//ROI region stays FULL (cone+plate; body may be included) -- the targeting logic
		//below, not the region, is what excludes the back body.
		pcl::PointCloud<pcl::PointXYZ>::Ptr _roiRegion = chassis_roi_extract_region(roiCrop, target_on_right, roiPrm, _roi_cone_pos, savePath, roiTag);
		if (jobInfo.chassisType == "CST") _pcGuide = _roiRegion;
		else                              _pcCone = _roiRegion;

		//Synthetic target class so deviation/logging downstream behave.
		if (jobInfo.chassisType == "CST")
		{
			_target_cone = bbx();
			_target_guide = bbx();
			_target_guide.index = 6;
			_target_guide.label = "csts";
			_target_guide.prob = 50;
		}
		else
		{
			_target_guide = bbx();
			_target_cone = bbx();
			_target_cone.index = (jobInfo.chassisType == "IGH") ? 3 : 1;
			_target_cone.label = (jobInfo.chassisType == "IGH") ? "igh" : "cone";
			_target_cone.prob = 50;
		}
	}
	else if (usePreset)
	{
		target_pointcloud_extraction_preset(pointCloud, jobInfo.presetRangeX, HEIGHT_MMR, TROLLEY_MMR, _sprd_landed_trigger, usePreset_CC, _target_hole, ref(_pcHole), ref(_pcCone), ref(_pcGuide), savePath);
	}
	else
	{
		target_pointcloud_extraction(pointCloud, _target_hole, _target_cone, _target_guide, _sprd_landed_trigger, ref(_pcHole), ref(_pcCone), ref(_pcGuide), savePath);
	}
#pragma endregion

#pragma region PCA - Hole, Cone detection
	position_detection_pca(jobInfo, usePreset, isFrontSensor, target_on_right, _detected_chassis_type, _target_hole, _target_cone, _target_guide, _pcHole, _pcCone, _pcGuide, ref(_hole_pos), ref(_cone_pos), ref(_guide_pos), savePath);
#pragma endregion

#pragma region ROI FALLBACK + validated VA-init reuse (VA-primary) -- XT/IGH
	//Priority: current VA cone (in-range) -> validated VA INIT -> ROI (last resort). The
	//init latch accepts ONLY in-range VA cones, so a systematic mis-pick (45/40 grabbing the
	//FRONT cone) never poisons it -> such jobs correctly fall to ROI, while a job that
	//started well but lost VA (occlusion as the container descends) reuses the validated
	//init. `usePreset` = the earlier 2D-preset fallback fired (VA cone absent). _cone_pos is
	//the chassis reference for BOTH XT and IGH. Init reuse is MOUNT-only; offload = MISS->ROI.
	int  _coneSrc = 0;	//0=VA current, 1=VA init(3D pos), 2=VA init(2D box remap), 3=ROI
	if (ROI_FALLBACK && !FORCE_ROI_MODE
		&& (_detected_chassis_type == "XT" || _detected_chassis_type == "IGH"))
	{
		if (jobInfo.isMountCycle)
		{
			bool vaPresent = (!usePreset && _target_cone.prob > 0 && _cone_pos.x != -10000.f);
			bool vaInRange = vaPresent
				&& _cone_pos.x >= (float)jobInfo.presetRangeX.MIN
				&& _cone_pos.x <= (float)jobInfo.presetRangeX.MAX;

			if (vaInRange)
			{
				//Trustworthy current VA cone -> use it AND feed the validated init latches.
				VA_ConeInit2D = _target_cone; VA_ConeInit2D_Set = true;
				VA_ConeInit3D.push(_cone_pos);
			}
			else
			{
				bool wasMispick = vaPresent;	//present but out-of-range (a miss => !vaPresent)

				//1) Validated VA init, preferred over ROI.
				if (VA_ConeInit2D_Set)
				{
					if (VA_ConeInit3D.locked())
					{
						_cone_pos = VA_ConeInit3D.value();	//occlusion-proof (chassis stationary)
						_coneSrc = 1;
					}
					else
					{
						//No robust 3D yet -> re-extract the region from the latched 2D box on
						//the CURRENT cloud and run the same cone-position logic.
						auto pcInit = map_box_to_pc_noisefiltertest(pointCloud, VA_ConeInit2D, 0, 0, "");
						pcl::PointXYZ ip(-10000.f, -10000.f, -10000.f);
						if (pcInit->points.size() > 0)
							ip = (_detected_chassis_type == "IGH")
								? pc_IGH_detection_naive(pcInit, target_on_right, "")
								: pc_cone_detection_naive(pcInit, target_on_right, "");
						if (ip.x != -10000.f) { _cone_pos = ip; _coneSrc = 2; }
					}

					if (_coneSrc != 0)
					{
						//Treat as detected: restore the validated VA 2D box (REAL coords, not a
						//synthetic marker) so bDetected_Chassis + VA reporting reflect it.
						_target_cone = VA_ConeInit2D;
						if (wasMispick) { _landout_detected = false; LDO_Current_Count = 0; }
						logMessage(std::string("[VA_INIT] current VA cone ") + (wasMispick ? "out-of-range" : "missed")
							+ " -> reuse validated init (" + (_coneSrc == 1 ? "3D pos" : "2D-box remap")
							+ ") T:" + std::to_string((int)_cone_pos.x));
						if (jobLog) jobLogMessage("[VA_INIT] reuse validated init cone");
					}
				}

				//2) ROI last resort (no validated init, or the 2D re-extract failed).
				if (_coneSrc == 0)
				{
					pcl::PointXYZ rc = computeRoiConePos(pointCloud, target_on_right, jobInfo, savePath);
					if (rc.x != -10000.f)
					{
						_cone_pos = rc; _coneSrc = 3;
						if (wasMispick) { _landout_detected = false; LDO_Current_Count = 0; }
						_target_cone = bbx();
						_target_cone.index = (_detected_chassis_type == "IGH") ? 3 : 1;
						_target_cone.label = (_detected_chassis_type == "IGH") ? "igh" : "cone";
						_target_cone.prob = 50;
						logMessage(std::string("[ROI_FB] no validated VA init -> ROI cone T:")
							+ std::to_string((int)rc.x) + " (va " + (wasMispick ? "out-of-range" : "missed") + ")");
						if (jobLog) jobLogMessage("[ROI_FB] ROI cone used (no VA init)");
					}
				}
			}
		}
		else if (jobInfo.isOffloadCycle)
		{
			//Offload: MISS -> ROI only (no init reuse; landed idx2 vs unloaded idx1 makes the
			//validated-init definition unreliable here).
			bool missed = usePreset || (_target_cone.prob == 0) || (_cone_pos.x == -10000.f);
			if (missed)
			{
				pcl::PointXYZ rc = computeRoiConePos(pointCloud, target_on_right, jobInfo, savePath);
				if (rc.x != -10000.f)
				{
					_cone_pos = rc; _coneSrc = 3;
					_target_cone = bbx();
					_target_cone.index = (_detected_chassis_type == "IGH") ? 3 : 1;
					_target_cone.label = (_detected_chassis_type == "IGH") ? "igh" : "cone";
					_target_cone.prob = 50;
					logMessage(std::string("[ROI_FB] offload VA cone missed -> ROI cone T:") + std::to_string((int)rc.x));
				}
			}
		}
	}
#pragma endregion

#pragma region FORCE_ROI_MODE - tracker update + A/B vs VA-mapped cone
	pcl::PointXYZ _roi_detected(-10000.f, -10000.f, -10000.f);
	if (FORCE_ROI_MODE)
	{
		//NEW targeting (XT): position_detection_pca above ran the cone detector on the FULL
		//region, whose densest-slice can latch onto the BACK BODY (target drifts deep/low).
		//Re-run the SAME detector on the NEAR-Z FACE (thin front slab) so the back body can't
		//be selected, then set the plate-corner height locally. (The full-region result is
		//still what position_detection_pca produced -- kept as OLD for the offline 3-way sheet.)
		if (jobInfo.chassisType == "XT" && _pcCone->points.size() > 0)
		{
			//Isolated-flyer removal first: floating scatter above the cone otherwise
			//inflates max-Y and pushes the upper-half mid-Y up (shaving the plate).
			auto _roiClean = chassis_roi_sor(_pcCone);
			auto _roiFace = chassis_roi_near_face(_roiClean, 80.f);
			pcl::PointXYZ nc = pc_cone_detection_naive(_roiFace, target_on_right, "");
			if (nc.x != -10000.f)
			{
				//Sparse-face rescue: on thin faces the detector's dense X stage stops short
				//of the true edge column -- conditionally move the corner there (x,z only).
				chassis_roi_sparse_edge_rescue(_roiFace, target_on_right, nc);
				float ch = chassis_roi_corner_height(_roiFace, nc.x, nc.z);
				if (ch != -10000.f) nc.y = ch;
				_cone_pos = nc;

				//Debug artifacts (batch): the face fed to the detector + the NEW target point.
				if (savePath != std::string(""))
				{
					try
					{
						if (_roiFace->points.size() > 0)
							pcl::io::savePCDFileBinaryCompressed(savePath + "_ROI_face.pcd", *_roiFace);
						pcl::PointCloud<pcl::PointXYZ> tc; tc.points.push_back(nc);
						tc.width = 1; tc.height = 1; tc.is_dense = true;
						pcl::io::savePCDFileBinaryCompressed(savePath + "_ROI_target.pcd", tc);
					}
					catch (...) {}
				}
			}
		}

		//IGH: guide top-left corner refine (guide-top height) on the ROI-extracted region,
		//replacing the pc_IGH_detection_naive result. Same rule as computeRoiConePos.
		if (jobInfo.chassisType == "IGH" && _pcCone->points.size() > 0)
		{
			pcl::PointXYZ ig = igh_target_from_region(_pcCone, target_on_right);
			if (ig.x != -10000.f)
			{
				_cone_pos = ig;
				if (savePath != std::string(""))
				{
					try {
						//_Region = the guide+plate region we pass on (matches the Python dump
						//naming so it can be compared vs GT in the same tool); _tgt = target point.
						if (_pcCone->points.size() > 0)
							pcl::io::savePCDFileBinaryCompressed(savePath + "_Region.pcd", *_pcCone);
						pcl::PointCloud<pcl::PointXYZ> tc; tc.points.push_back(ig);
						tc.width = 1; tc.height = 1; tc.is_dense = true;
						pcl::io::savePCDFileBinaryCompressed(savePath + "_ROI_target.pcd", tc);
					}
					catch (...) {}
				}
			}
		}

		//Position now comes from position_detection_pca run on the ROI-extracted region
		//(set into _pcCone/_pcGuide above) -- no separate override. Feed THAT to the tracker.
		_roi_detected = (jobInfo.chassisType == "CST") ? _guide_pos : _cone_pos;

		chassisRoiTracker.update(_roi_detected, _sprd_landed_trigger, roiPrm,
			[jobLog](const std::string& m) { logMessage(m); if (jobLog) jobLogMessage(m); });

		//A/B: ROI-detected cone vs VA-mapped cone (same detector, both valid).
		bool roiOk = (_roi_detected.x != -10000.f);
		bool vaOk = (_ab_va_cone.x != -10000.f);
		if (roiOk && vaOk)
		{
			_ab_dT = (int)(_roi_detected.x - _ab_va_cone.x);
			_ab_dH = (int)(_roi_detected.y - _ab_va_cone.y);
			_ab_dG = (int)(_roi_detected.z - _ab_va_cone.z);
			_ab_valid = true;
			logMessage("[ROI_AB] ROI-vs-VA cone delta (T:" + std::to_string(_ab_dT)
				+ " H:" + std::to_string(_ab_dH) + " G:" + std::to_string(_ab_dG)
				+ ")  ROI(T:" + std::to_string((int)_roi_detected.x) + " G:" + std::to_string((int)_roi_detected.z)
				+ ") VA(T:" + std::to_string((int)_ab_va_cone.x) + " G:" + std::to_string((int)_ab_va_cone.z) + ")");
			if (jobLog) jobLogMessage("[ROI_AB] ROI-vs-VA cone delta (T:" + std::to_string(_ab_dT)
				+ " H:" + std::to_string(_ab_dH) + " G:" + std::to_string(_ab_dG) + ")");
		}
	}
#pragma endregion

#pragma region dev_out_in pca -- landout/clps included.
	deviation_output_pca(isFrontSensor, target_on_right, jobInfo.isMountCycle, jobInfo.isOffloadCycle, _detected_chassis_type, _hole_pos, _cone_pos, _guide_pos, _sprd_landed_trigger, ref(_dvOut_pca_x), ref(_dvOut_pca_y), ref(_dvOut_pca_z), ref(_landout_detected_pca), ref(_clps_detected_pca), ref(_clps_ok_detected_pca));
#pragma endregion

#pragma region CLPS 3D (offload, XT/IGH) -- reverse-replay validated 2026-07
	//Primary CLPS/CLPS-OK from the 3D target height: works with VA blind (18% of
	//replayed jobs), immune to the partner-cone idx1 flap, and guards the
	//"hoist paused with chassis dangling" case via the landedH+margin check.
	bool _clps_3d = false, _clps_ok_3d = false;
	if (jobInfo.isOffloadCycle && jobInfo.chassisType != "CST")
	{
		//twistlock-locked arm condition: PLC bit in live; debug/batch has no PLC.
		bool twl = (MODE_DEBUG || DEBUG_BATCH_JOB || DEBUG_WITH_FILES) ? true : SPRD_TWL_Locked;
		bool prevOk = clpsOffload3D.clpsOk(), prevCl = clpsOffload3D.clps();
		clpsOffload3D.update(_eff_landed, twl, _cone_pos.y);
		_clps_ok_3d = clpsOffload3D.clpsOk();
		_clps_3d = clpsOffload3D.clps();
		if (_clps_ok_3d && !prevOk)
		{
			logMessage("[CLPS3D] CLPS-OK Detected! H stable at " + std::to_string((int)_cone_pos.y)
				+ " (landedH " + std::to_string((int)clpsOffload3D.landedH()) + ")");
			if (jobLog) jobLogMessage("[CLPS3D] CLPS-OK Detected! H stable at " + std::to_string((int)_cone_pos.y));
		}
		if (_clps_3d && !prevCl)
		{
			logMessage("[CLPS3D] CLPS Detected! H " + std::to_string((int)_cone_pos.y)
				+ " above landedH " + std::to_string((int)clpsOffload3D.landedH()) + " -> chassis lift suspected");
			if (jobLog) jobLogMessage("[CLPS3D] CLPS Detected! chassis lift suspected");
		}
	}
#pragma endregion

	//Landout final output: VA ONLY for now (PCA landout intentionally NOT combined into the
	//output -- _landout_detected_pca still computed & logged for diagnostics, just not sent).
	_Out_landout_detected = _landout_detected;	//was: (_landout_detected || _landout_detected_pca)
	_Out_clps_detected = (_clps_detected || _clps_detected_pca || _clps_3d);
	_Out_clps_ok_detected = (_clps_ok_detected || _clps_ok_detected_pca || _clps_ok_3d);

#pragma region 3D deviation OUTPUT multiplex (XT/IGH) -- position vs deviation
	//What we SEND to the PLC on Dev_*_3D is phase-multiplexed. Detection above still ran
	//on the realtime deviation (_dvOut_pca_*); this only shapes the output bytes.
	//  MOUNT:   POSITION = latched chassis ref (cone/guide) until the hole is confirmed
	//           in 3 cumulative VA frames -> then DEVIATION = hole(realtime) - ref(latched).
	//  OFFLOAD: POSITION = latched hole (from job start) until the VA cone-idx1+hole
	//           separation fires -> then DEVIATION with the cone latched from separation.
	//The PLC infers position-vs-deviation from the existing VA detection bits (byte16)
	//plus its own Landed bit -- no new protocol bits (offload phase-1 is landed, so the
	//PLC's Landed gate keeps it from reading the latched hole position as a deviation).
	//Non-XT/IGH (CST) keeps the realtime deviation unchanged.
	int _dev3d_out_x = _dvOut_pca_x, _dev3d_out_y = _dvOut_pca_y, _dev3d_out_z = _dvOut_pca_z;
	bool _dev3d_is_position = false;
	if (_detected_chassis_type == "XT" || _detected_chassis_type == "IGH")
	{
		const int x_offset = 14, z_offset = 26;						//legacy TUAS-3DSP offsets
		//Chassis reference: position_detection_pca writes BOTH XT (idx1) and IGH (idx3) into
		//_cone_pos (only CST uses _guide_pos, which is out of scope here). So _cone_pos for both.
		pcl::PointXYZ chRef = _cone_pos;
		const pcl::PointXYZ INVALID_PT(-10000, -10000, -10000);

		auto finite3 = [](const pcl::PointXYZ& q)
		{
			return std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z)
				&& q.x != -10000 && q.y != -10000 && q.z != -10000;
		};
		auto devFromLatchedRef = [&](const pcl::PointXYZ& c)
		{
			if (finite3(_hole_pos) && finite3(c))
			{
				_dev3d_out_x = (int)(c.x - (target_on_right ? (_hole_pos.x + x_offset) : (_hole_pos.x - x_offset)));
				_dev3d_out_y = (int)(c.y - _hole_pos.y);
				_dev3d_out_z = (int)(c.z - (_hole_pos.z + z_offset));
			}
			else { _dev3d_out_x = _dev3d_out_y = _dev3d_out_z = -10000; }
		};

		if (jobInfo.isMountCycle)
		{
			DEV3D_RefLatch.push(chRef);								//latch chassis ref from job start
			if (!DEV3D_mount_dev_phase && _target_hole.prob > 70)
			{
				DEV3D_mount_hole_count++;							//cumulative, not continuous
				if (DEV3D_mount_hole_count >= 3) DEV3D_mount_dev_phase = true;
			}
			if (!DEV3D_mount_dev_phase)
			{
				pcl::PointXYZ p = DEV3D_RefLatch.value();		//sentinel (-10000) until locked
				_dev3d_out_x = (int)p.x; _dev3d_out_y = (int)p.y; _dev3d_out_z = (int)p.z;
				_dev3d_is_position = true;
			}
			else
			{
				devFromLatchedRef(DEV3D_RefLatch.locked() ? DEV3D_RefLatch.value() : INVALID_PT);
			}
		}
		else if (jobInfo.isOffloadCycle)
		{
			DEV3D_HoleLatch.push(_hole_pos);						//latch hole from job start
			if (_clps_ok_detected) DEV3D_offload_dev_phase = true;	//VA cone-idx1+hole separation (sticky)
			if (!DEV3D_offload_dev_phase)
			{
				pcl::PointXYZ p = DEV3D_HoleLatch.value();
				_dev3d_out_x = (int)p.x; _dev3d_out_y = (int)p.y; _dev3d_out_z = (int)p.z;
				_dev3d_is_position = true;
			}
			else
			{
				DEV3D_RefLatch.push(chRef);							//cone latch STARTS at separation
				devFromLatchedRef(DEV3D_RefLatch.locked() ? DEV3D_RefLatch.value() : chRef);
			}
		}
	}
#pragma endregion


	logLines += std::to_string(_pcHole->points.size()) + ";" + std::to_string(_pcCone->points.size()) + ";" + std::to_string(_pcGuide->points.size()) + ";";
	logLines += std::to_string(_hole_pos.x) + ";" + std::to_string(_hole_pos.y) + ";" + std::to_string(_hole_pos.z) + ";";

	logLines += std::to_string(_cone_pos.x) + ";" + std::to_string(_cone_pos.y) + ";" + std::to_string(_cone_pos.z) + ";";

	logLines += std::to_string(_dvOut_pca_x) + ";" + std::to_string(_dvOut_pca_y) + ";" + std::to_string(_dvOut_pca_z) + ";";
	logLines += std::to_string(_landout_detected_pca) + ";" + std::to_string(_clps_detected_pca) + ";" + std::to_string(_clps_ok_detected_pca) + ";";

	logLines += std::to_string(0) + ";" + std::to_string(LDO_Current_Count) + ";" + std::to_string(landout_current_count_pca) + ";";
	logLines += std::to_string(CLPS_Current_Count) + ";" + std::to_string(clps_current_count_pca) + ";";
	logLines += std::to_string(clps_ok_current_count_pca) + ";" + std::to_string(0) + ";";

	logLines += std::to_string(false) + ";" + std::to_string(0) + ";";

	logLines += std::to_string(_Out_clps_detected) + ";" + std::to_string(_Out_clps_ok_detected);

	if (FORCE_ROI_MODE)
	{
		//Extra columns (header extended in the batch thread when FORCE_ROI_MODE).
		//ROI_T/H/G = ROI-detected cone (via the existing detector on the crop).
		logLines += ";" + chassisRoiTracker.stateString() + ";";
		logLines += std::to_string((int)_roi_detected.x) + ";" + std::to_string((int)_roi_detected.y) + ";" + std::to_string((int)_roi_detected.z) + ";";
		logLines += std::to_string((int)_pcCone->points.size()) + ";";
		logLines += std::to_string(_ab_valid) + ";" + std::to_string(_ab_dT) + ";" + std::to_string(_ab_dH) + ";" + std::to_string(_ab_dG);
	}

	//CLPS 3D diagnostics (appended LAST so earlier column indices stay stable):
	//EffLanded = the landed used for gating (PLC in live / visual in debug);
	//CLPS_3D / CLPSOK_3D = the INDIVIDUAL 3D verdicts (CLPS_OUT/CLPS_OK_OUT above are
	//the combined VA||PCA||3D); LandedH_3D = learned loaded-phase H baseline.
	logLines += ";" + std::to_string(_eff_landed) + ";" + std::to_string(_clps_3d) + ";" + std::to_string(_clps_ok_3d) + ";";
	logLines += std::to_string((int)clpsOffload3D.landedH());

	//3D deviation output multiplex (appended last): phase 0=n/a(CST) 1=position 2=deviation,
	//then the multiplexed value actually sent to the PLC on Dev_*_3D.
	int _dev3d_phase = (_detected_chassis_type == "XT" || _detected_chassis_type == "IGH") ? (_dev3d_is_position ? 1 : 2) : 0;
	logLines += ";" + std::to_string(_dev3d_phase) + ";" + std::to_string(_dev3d_out_x) + ";" + std::to_string(_dev3d_out_y) + ";" + std::to_string(_dev3d_out_z);
	//Cone source: 0=VA current, 1=VA init(3D pos), 2=VA init(2D-box remap), 3=ROI fallback.
	logLines += ";" + std::to_string(_coneSrc);

	//===== process-result overlays (skim on debug images) =====
	//flag helper: red when TRUE, grey when FALSE.
	auto putFlag = [&](const std::string& name, bool v, int y)
	{
		cv::putText(res_image, name + (v ? ":TRUE" : ":FALSE"), cv::Point(0, y),
			cv::FONT_HERSHEY_SIMPLEX, 0.8, v ? cv::Scalar(0, 0, 255) : cv::Scalar(180, 180, 180), 2);
	};
	//3D deviation OUTPUT actually sent on Dev_*_3D -- BOTH cycles. POS = latched position
	//phase, DEV = deviation phase (hole-latched ref), N/A = non XT/IGH (realtime dev).
	{
		std::string ph = (_dev3d_phase == 1) ? "POS" : (_dev3d_phase == 2) ? "DEV" : "N/A";
		cv::putText(res_image, std::string("3D-") + ph + ": " + std::to_string(_dev3d_out_x) + ","
			+ std::to_string(_dev3d_out_y) + "," + std::to_string(_dev3d_out_z),
			cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 255), 2);
	}
	//Cone source when NOT the current-frame VA detection -- orange.
	//1=validated VA init (3D pos), 2=validated VA init (2D-box remap), 3=ROI fallback.
	if (_coneSrc != 0)
	{
		std::string src = (_coneSrc == 1) ? "VAinit-3D" : (_coneSrc == 2) ? "VAinit-box" : "ROI";
		cv::putText(res_image, std::string("CONE_SRC:") + src, cv::Point(0, 80),
			cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 165, 255), 2);
	}

	if (jobInfo.isOffloadCycle)
	{
		//CLPS-OK - VA
		if (clps_detected_pca) cv::putText(res_image, std::string("CLPS_PCA:TRUE"), cv::Point(0, 110), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
		else cv::putText(res_image, std::string("CLPS_PCA:FALSE"), cv::Point(0, 110), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

		if (clps_ok_detected_pca) cv::putText(res_image, std::string("CLPS-OK_PCA:TRUE"), cv::Point(0, 140), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
		else cv::putText(res_image, std::string("CLPS-OK_PCA:FALSE"), cv::Point(0, 140), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

		//combined outputs actually sent to the PLC (byte17) + per-pipeline breakdown.
		putFlag("CLPS", _Out_clps_detected, 200);
		putFlag("CLPSOK", _Out_clps_ok_detected, 230);
		cv::putText(res_image, std::string("C va/pca/3d:") + (_clps_detected ? "1" : "0") + (_clps_detected_pca ? "1" : "0") + (_clps_3d ? "1" : "0")
			+ "  OK:" + (_clps_ok_detected ? "1" : "0") + (_clps_ok_detected_pca ? "1" : "0") + (_clps_ok_3d ? "1" : "0"),
			cv::Point(0, 260), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

		/*
		if (clps_ok_detected_pca_only) cv::putText(res_image, std::string("CLPS-SEP_PCA:TRUE : ") + std::to_string(pca_dev_y), cv::Point(0, 170), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
		else cv::putText(res_image, std::string("CLPS-SEP_PCA:FALSE : ") + std::to_string(pca_dev_y), cv::Point(0, 170), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
			
		//draw line here.
		if (clps_y_level > 0) cv::line(res_image, cv::Point(0, clps_y_level), cv::Point(512, clps_y_level), cv::Scalar(0, 0, 255));
		cv::putText(res_image, std::string("PCunder: ") + std::to_string(refPCUnderHole), cv::Point(0, 230), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
		*/
	}
	else if (jobInfo.isMountCycle)
	{
		cv::putText(res_image, std::string("DEV_PCA: ") + std::to_string(_dvOut_pca_x) + " y: " + std::to_string(_dvOut_pca_y) + " z: " + std::to_string(_dvOut_pca_z), cv::Point(0, 140), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

		if (landout_detected_pca) cv::putText(res_image, std::string("LANDOUT_PCA:TRUE"), cv::Point(0, 170), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
		else cv::putText(res_image, std::string("LANDOUT_PCA:FALSE"), cv::Point(0, 170), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

		//combined landout actually sent to the PLC (byte17) + per-pipeline breakdown.
		putFlag("LANDOUT", _Out_landout_detected, 200);
		cv::putText(res_image, std::string("LDO va/pca:") + (_landout_detected ? "1" : "0") + (_landout_detected_pca ? "1" : "0"),
			cv::Point(0, 230), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
	}
	

	if (debugMode)
	{
		if (savePath != "")
		{
			std::string debug_img_path = savePath + "_debug_result.jpg";
			logMessage(debug_img_path);
			cv::imwrite(debug_img_path, res_image);
		}
	}

	//va process results update.
	procResults.VA_Chassis_Type = _detected_chassis_type;
	if (_detected_chassis_type == "XT") procResults.bDetected_XT = true;
	else if (_detected_chassis_type == "CST") procResults.bDetected_CST = true;
	else if (_detected_chassis_type == "IGH") procResults.bDetected_IGH = true;
	
	procResults.VA_Target_Chassis = _target_cone;
	procResults.VA_Target_Container = _target_hole;
	
	if (_target_cone.prob > 0 || _target_guide.prob > 0) procResults.bDetected_Chassis = true;
	if (_target_hole.prob > 0) procResults.bDetected_Container = true;
	
	procResults.VA_Devout_X = _dvOutX;
	procResults.VA_Devout_Y = _dvOutY;
	procResults.VA_isLandout = _landout_detected;
	procResults.VA_isCLPS = _clps_detected;
	procResults.VA_isCLPS_OK = _clps_ok_detected;

	//pca process results update.
	if (_detected_chassis_type == "CST")
		procResults.PCA_Target_Chassis = _guide_pos;
	else
		procResults.PCA_Target_Chassis = _cone_pos;

	procResults.PCA_Target_Container = _hole_pos;
	procResults.PCA_Target_Guide = _guide_pos;
	//Multiplexed output (position in phase 1 / deviation in phase 2 for XT/IGH; realtime
	//deviation otherwise). Realtime deviation stays in _dvOut_pca_* (res.txt + detection).
	procResults.PCA_Devout_X = _dev3d_out_x;
	procResults.PCA_Devout_Y = _dev3d_out_y;
	procResults.PCA_Devout_Z = _dev3d_out_z;
	procResults.PCA_isLandout = _landout_detected_pca;
	procResults.PCA_isCLPS = _clps_detected_pca;
	procResults.PCA_isCLPS_OK = _clps_ok_detected_pca;

	procResults.bLandOutDetected = _Out_landout_detected;
	procResults.bClpsDetected = _Out_clps_detected;
	procResults.bClpsOkDetected = _Out_clps_ok_detected;

	return procResults;
}


void processingThread()
{
	try
	{
		//__try
		//{
			logMessage("Processing Thread Activated!");

			while (proc_running.load())
			{
				std::unique_lock<std::mutex> lock(mutex_processing);
				bool res = cond_processing.wait_for(lock,
					std::chrono::seconds(3600),
					[]() { return processFlag; });

				if (res)
				{
					logMessage("Enabled process! Current Position : " + CURRENT_SENSOR_POSITION);
					//jobLogMessage("Enabled process! Current Position : " + CURRENT_SENSOR_POSITION);

					current_job_info.print_jobInfo();

					//data log file.
					auto logHeader = std::string("Filename;Hole_x;Hole_y;Hole_w;Hole_h;Hole_prob;XT_index;XT_x;XT_y;XT_w;XT_h;XT_prob;Guide_x;Guide_y;Guide_w;Guide_y;Guide_prob;devout_x;devout_y;devout_x_mm;devout_y_mm;Landed_Trigger;Landout_Detected;CLPS_Detected;CLPS_OK;Hole_pc;Cone_pc;Guide_pc;hole_x;hole_y;hole_z;cone_x;cone_y;cone_z;devout_pca_x;devout_pca_y;devout_pca_z;landout_pca;clps_pca;clpsOk_pca;pcUnderHole;LDO_VA_Count;LDO_PCA_Count;CLPS_VA_Count;CLPS_PCA_Count;CLPSOK_PCA_Count;CLPS_Y_LEVEL;CLPS_PCA_ONLY_SEP;SEP_DEV_Y;CLPS_OUT;CLPS_OK_OUT");
					//Keep IDENTICAL to the batch header (OfflineDebugBatchProcessingThread) so the
					//live res.txt columns line up with the log_lines data (ProcessLogic appends all
					//of these). FORCE_ROI columns are conditional; the rest are always present.
					if (FORCE_ROI_MODE) logHeader += ";ROI_State;ROI_T;ROI_H;ROI_G;ROI_CropPts;AB_Valid;AB_dT;AB_dH;AB_dG";
					logHeader += ";EffLanded;CLPS_3D;CLPSOK_3D;LandedH_3D";
					logHeader += ";Dev3D_Phase;Dev3D_OutX;Dev3D_OutY;Dev3D_OutZ;ConeSrc";

					ofstream Simfile;
					std::string sim_result_txt = job_result_folder_name + "/" + job_name + "_res.txt";
					Simfile.open(sim_result_txt.c_str(), ios::out | ios::app);
					if (!Simfile.is_open())
					{
						logMessage("Failed to open file?");
					}
					if (Simfile.is_open())
					{
						Simfile << logHeader + "\n";
						Simfile.close();
					}

					int frame_index = 0;

					//printf("Process starting.\n");
					auto check_interval = 200;
					auto last_dt_checked = std::chrono::system_clock::now();

					bool dataStackReady = false;

					logMessage("Process Setup Complete!");
					while (enable_process)
					{
						auto get_dt = std::chrono::system_clock::now();
						auto wait_dur = std::chrono::duration_cast<std::chrono::milliseconds>(get_dt - last_dt_checked).count();
						if (wait_dur >= check_interval)
						{
							if (dataStack.Stack_Ready())
							{
								dataStackReady = true;
								frame_index++;
								std::string log_lines = std::to_string(frame_index) + ";";

								last_dt_checked = get_dt;

								cv::Mat img;
								pcl::PointCloud<pcl::PointXYZ>::Ptr pcPointCloud(new pcl::PointCloud<pcl::PointXYZ>);
								pcl::PointCloud<pcl::PointXYZ> pointCloud;
								std::chrono::system_clock::time_point last_ts, ts;

								//logMessage("Retrieving stack data...");
								auto r_status = dataStack.Retrieve_Stack(ref(img), ref(pointCloud), ref(ts));
								//if (r_status) logMessage("Stack data retrieved successfully!");
								if (!r_status)
								{
									logMessage("Failed to retrieve stack data!");
									std::this_thread::sleep_for(std::chrono::milliseconds(30)); //single frame capture time.
									continue;
								}

								if (r_status && last_ts < ts)
								{
									last_ts = ts;

									//Convert visionary::pointcloud to pcl::pointcloud
									if (pcPointCloud == NULL)
									{
										pcPointCloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
										logMessage("Created new pcl::PointCloud<pcl::PointXYZ> instance.");
									}

									pcPointCloud->points.resize(pointCloud.size());
									std::copy(pointCloud.begin(), pointCloud.end(), pcPointCloud->points.begin());

									auto res_img = img.clone();

									auto t_now = std::chrono::system_clock::now();
										
									if (save_trigger_by_landed || save_trigger_by_TWL_Locked)
									{
										//Draw and push to queue.
										//res_img = drawOnImage(res_img, det_results);

										std::string msg = save_trigger_by_TWL_Locked ? "TWL" : "TWUL";
										std::string msg2 = save_trigger_by_landed ? "LANDED" : "LANDOFF";
										msg = msg + std::string("_") + msg2;

										SaveTask dataTask(img, res_img, pcPointCloud, true, std::move(msg), ts);
										tsq.push(std::move(dataTask));

										if (save_trigger_by_landed) save_trigger_by_landed = false;
										if (save_trigger_by_TWL_Locked) save_trigger_by_TWL_Locked = false;
									}
									else if (enable_logging)
									{
										SaveTask dataTask(img, cv::Mat(), pcPointCloud, false, std::string{}, last_frame_get_time);
										tsq.push(std::move(dataTask));
									}

									if (!ENABLE_IMAGE_LOGGING_ONLY)
									{
										//Process here.
										bool isFrontSensor = (CURRENT_SENSOR_POSITION == "FRONT") ? true : false;
										auto pre_pointCloud = makePCL_PointCloud(*pcPointCloud, DEBUG_CONVERT_PCL_RANGE);

										auto pointCloud = pointCloud_tilt_transform(pre_pointCloud, TILT_ANGLE_DEG);
										//pcl::io::savePCDFileBinaryCompressed(save_path + "_base_converted.pcd", *pointCloud);

										current_process_results = ProcessLogic(isFrontSensor, current_job_info, img, pointCloud, std::string(""), false, false, ref(log_lines));

										Simfile.open(sim_result_txt.c_str(), ios::out | ios::app);
										if (!Simfile.is_open())
										{
											logMessage("Failed to open file?");
										}
										if (Simfile.is_open())
										{
											Simfile << log_lines + "\n";
											Simfile.close();
										}
									}
								}	

								std::this_thread::sleep_for(std::chrono::milliseconds(10));
							}
							else
							{
								if (dataStackReady)
								{
									dataStackReady = false;
									logMessage("Data Stack is NOT ready for processing. Waiting for data...");
								}
								std::this_thread::sleep_for(std::chrono::milliseconds(100));
							}
						}
						else
						{
							auto sleep_time = check_interval - wait_dur;
							std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
						}
					}

					logMessage("Process thread is back to blocked state...");
				}
				else
				{
					//Timeout triggered awake.
					logMessage("Timeout triggered awake, returning to wait_for...");
				}
			}

			logMessage("Prcessing Thread Deactivated!");
		//}
		//__except (EXCEPTION_EXECUTE_HANDLER)
		//{
		//	logMessage("Processing thread exception SEH (Access Violation, Stack Overflow caught!");
		//}
	}
	catch (std::exception& ex)
	{
		logMessage("Processing thread exception: " + std::string(ex.what()));
	}
	catch (...)
	{
		logMessage("Processing thread has been terminated due to an unknown exception!");
	}
}

void OfflineDebugBatchProcessingThread()
{
	logMessage("Debug Batch Job on " + DEBUG_BATCH_ROOT_DIR + " to be processed and saved to " + DEBUG_BATCH_SAVE_DIR);

	//Get list of sub directories given ROOT DIR.
	auto jobDirectories = ListSubDirectories(DEBUG_BATCH_ROOT_DIR);

	bool savePLY = true;

	auto logHeader_offload_batch = std::string("Total_Job;OK3;OK-VA,OK-PCA;OK-VA,OK-PCA-ONLY;OK-PCA,OK-PCA-ONLY;OK-VA;OK-PCA;OK-PCA-ONLY;OK-N;CLPS-VA,CLPS-PCA;CLPS-VA;CLPS-PCA;CLPS-N");

	//Per job: directory name, then results.
	//For batch: total jobs and each count.
	int total_jobs = 0, OK_VA = 0, OK_VA_OK_PCA = 0, OK_VA_OK_PCA_ONLY = 0, OK3 = 0, OK_PCA = 0, OK_PCA_OK_PCA_ONLY = 0, OK_PCA_ONLY = 0, OK_N = 0;
	int CLPS_VA = 0, CLPS_VA_CLPS_PCA = 0, CLPS_PCA = 0, CLPS_N = 0;

	ofstream Simfile_batch;
	std::string sim_batch_result_txt = DEBUG_BATCH_ROOT_DIR + "/Offload_batch_results.txt";
	Simfile_batch.open(sim_batch_result_txt.c_str(), ios::out | ios::app);
	if (!Simfile_batch.is_open())
	{
		logMessage("Failed to open file?");
	}
	if (Simfile_batch.is_open())
	{
		Simfile_batch << logHeader_offload_batch + "\n";
		Simfile_batch.close();
	}

	//Work list: with ENABLE_MOUNT_REVERSE every Mount job is processed TWICE --
	//pass 0 = forward (normal Mount), pass 1 = frames reversed and treated as OFFLOAD
	//(the only sequence data with the separation phase). Results are split into
	//Mount_Result/ and Offload_Result/ under DEBUG_BATCH_SAVE_DIR.
	std::vector<std::pair<std::string, bool>> workItems;
	for (const auto& d : jobDirectories)
	{
		workItems.push_back(std::make_pair(d, false));
		if (ENABLE_MOUNT_REVERSE) workItems.push_back(std::make_pair(d, true));
	}

	for (const auto& workItem : workItems)
	{
		const std::string& jobDir = workItem.first;
		const bool reversePass = workItem.second;

		reset_recvVariables();
		reset_jobVariables();

		logMessage("Processing: " + jobDir + (reversePass ? "  [REVERSE pass]" : ""));

		//load .jpg, .pcd files for processing
		auto image_filePath = jobDir + "/" + std::string("Image");
		auto depth_filePath = jobDir + "/" + std::string("Depth");

		std::filesystem::path pathObj = std::filesystem::path(jobDir).lexically_normal();
		std::string lastDirectory = pathObj.filename().string();

		//Only process current-site recordings. Accepted: FS*/CS* prefixed
		//folders (new naming) or folders with the _FRONT_/_REAR_ sensor token
		//(2026 era). Excluded: *TMINI_LEFT/RIGHT* (other-site/older era) and
		//anything else (e.g. nested result folders).
		{
			std::string upperName = lastDirectory;
			std::transform(upperName.begin(), upperName.end(), upperName.begin(), ::toupper);

			bool isOldEra = (upperName.find("TMINI") != std::string::npos);
			bool hasSitePrefix = (upperName.rfind("FS", 0) == 0) || (upperName.rfind("CS", 0) == 0);
			bool hasSensorToken = (upperName.find("_FRONT_") != std::string::npos) || (upperName.find("_REAR_") != std::string::npos);

			if (isOldEra || (!hasSitePrefix && !hasSensorToken))
			{
				logMessage("Skipping non-site folder: " + lastDirectory);
				continue;
			}
		}

		//The reverse pass only applies to Mount jobs (reversing a real Offload job
		//would fabricate a mount) -- skip its reverse work item.
		bool nameIsMount = (lastDirectory.find("Mount") != std::string::npos);
		if (reversePass && !nameIsMount) continue;

		bool loadedPLY = false;
		auto image_files = getAllFiles(image_filePath, ".jpg");
		auto depth_files = getAllFiles(depth_filePath, ".pcd");
		if (depth_files.size() == 0)
		{
			depth_files = getAllFiles(depth_filePath, ".ply");
			loadedPLY = true;
		}

		createDirectory_ifexists(DEBUG_BATCH_SAVE_DIR);
		std::string cycleRoot = DEBUG_BATCH_SAVE_DIR;
		if (false)//ENABLE_MOUNT_REVERSE)
		{
			//Split results by the EFFECTIVE cycle: forward Mount -> Mount_Result/,
			//reversed Mount (and native Offload jobs) -> Offload_Result/.
			cycleRoot += std::string("/") + ((nameIsMount && !reversePass) ? "Mount_Result" : "Offload_Result");
			createDirectory_ifexists(cycleRoot);
		}
		auto save_file_path = cycleRoot + "/" + lastDirectory;
		auto folder_created = createDirectory_ifexists(save_file_path);

		if (!folder_created)
		{
			logMessage("Skipping already existing folder: " + save_file_path);
			continue;
		}

		logMessage("Saving to " + save_file_path);

		auto logHeader = std::string("Filename;Hole_x;Hole_y;Hole_w;Hole_h;Hole_prob;XT_index;XT_x;XT_y;XT_w;XT_h;XT_prob;Guide_x;Guide_y;Guide_w;Guide_y;Guide_prob;devout_x;devout_y;devout_x_mm;devout_y_mm;Landed_Trigger;Landout_Detected;CLPS_Detected;CLPS_OK;Hole_pc;Cone_pc;Guide_pc;hole_x;hole_y;hole_z;cone_x;cone_y;cone_z;devout_pca_x;devout_pca_y;devout_pca_z;landout_pca;clps_pca;clpsOk_pca;pcUnderHole;LDO_VA_Count;LDO_PCA_Count;CLPS_VA_Count;CLPS_PCA_Count;CLPSOK_PCA_Count;CLPS_Y_LEVEL;CLPS_PCA_ONLY_SEP;SEP_DEV_Y;CLPS_OUT;CLPS_OK_OUT");
		if (FORCE_ROI_MODE) logHeader += ";ROI_State;ROI_T;ROI_H;ROI_G;ROI_CropPts;AB_Valid;AB_dT;AB_dH;AB_dG";
		logHeader += ";EffLanded;CLPS_3D;CLPSOK_3D;LandedH_3D";
		logHeader += ";Dev3D_Phase;Dev3D_OutX;Dev3D_OutY;Dev3D_OutZ;ConeSrc";

		ofstream Simfile;
		std::string sim_result_txt = save_file_path + "/" + lastDirectory + "_res.txt";
		Simfile.open(sim_result_txt.c_str(), ios::out | ios::app);
		if (!Simfile.is_open())
		{
			logMessage("Failed to open file?");
		}
		if (Simfile.is_open())
		{
			Simfile << logHeader + "\n";
			Simfile.close();
		}

		//Get job information from the folder name. Tokens are located relative
		//to the Mount/Offload token, so both old ({ts}_{Bn}_...) and
		//site-prefixed (FS_/CS_...) folder names parse correctly.
		auto info_temp = split(lastDirectory, '_');
		int cycleIdx = -1;
		for (int t = 0; t < (int)info_temp.size(); t++)
		{
			if (info_temp[t] == "Mount" || info_temp[t] == "Offload") { cycleIdx = t; break; }
		}

		if (cycleIdx >= 0)
		{
			if (info_temp[cycleIdx] == "Mount")
			{
				Mount_Cycle = true;
				Offload_Cycle = false;
				logMessage("Mount Cycle Detected");
			}
			else
			{
				Offload_Cycle = true;
				Mount_Cycle = false;
				logMessage("Offload Cycle Detected");
			}
		}

		//Reverse pass: play the Mount job BACKWARDS and process it as an OFFLOAD
		//(landed -> hoist -> separation -> cone-only). Real offload recordings all end
		//while landed, so this is the only sequence data with the separation phase --
		//used to exercise CLPS / CLPS-OK offline. Image and depth lists are
		//index-paired, so both are reversed together.
		if (reversePass && Mount_Cycle)
		{
			Offload_Cycle = true;
			Mount_Cycle = false;
			std::reverse(image_files.begin(), image_files.end());
			std::reverse(depth_files.begin(), depth_files.end());
			logMessage("[MOUNT-REVERSE] Mount job replayed BACKWARDS as OFFLOAD: " + lastDirectory);
		}

		std::string job_pos = "Center";
		std::string chs_type = "XT";
		int chs_size = 40;
		int sprd_size = 40;

		if (cycleIdx >= 0 && (int)info_temp.size() > cycleIdx + 4)
		{
			job_pos = (info_temp[cycleIdx + 1] != "None") ? info_temp[cycleIdx + 1] : "Center";

			chs_type = (info_temp[cycleIdx + 2] != "None") ? info_temp[cycleIdx + 2] : "XT";


			if (info_temp[cycleIdx + 3] == "CHS45") chs_size = 45;
			else if (info_temp[cycleIdx + 3] == "CHS20") chs_size = 20;

			if (info_temp[cycleIdx + 4] == "SPRD45") sprd_size = 45;
			else if (info_temp[cycleIdx + 4] == "SPRD20" || info_temp[cycleIdx + 4] == "SRPD20") sprd_size = 20;
		}

		bbx presetTarget;
		MAX_MIN_RANGE presetTargetX;
		//PRESET SELECT
		if (DEBUG_SENSOR_POSITION == "FRONT")
		{
			if (chs_type == "XT")
			{
				if (chs_size == 45 && sprd_size == 40) { presetTarget = presetPos.F_CONE_4X_R; presetTargetX = presetPos.F_XT_4XR_RNG; }
				else if (chs_size == 45 && job_pos == "Small") { presetTarget = presetPos.F_CONE_20_R;  presetTargetX = presetPos.F_XT_20R_RNG;}
				else if (chs_size == 40 && job_pos == "Small") { presetTarget = presetPos.F_CONE_20_F;  presetTargetX = presetPos.F_XT_20F_RNG;}
				else
				{
					presetTarget = presetPos.F_CONE_4X_F;
					presetTargetX = presetPos.F_XT_4XF_RNG;
				}
			}
			else if (chs_type == "IGH")
			{
				if (chs_size == 45 && sprd_size == 40) { presetTarget = presetPos.F_IGH_4X_R; presetTargetX = presetPos.F_IGH_4XR_RNG; }
				else if (chs_size == 45 && job_pos == "Small") { presetTarget = presetPos.F_IGH_20_R;  presetTargetX = presetPos.F_IGH_20R_RNG; }
				else if (chs_size == 40 && job_pos == "Small") { presetTarget = presetPos.F_IGH_20_F;  presetTargetX = presetPos.F_IGH_20F_RNG; }
				else
				{
					presetTarget = presetPos.F_IGH_4X_F;
					presetTargetX = presetPos.F_IGH_4XF_RNG;
				}
			}
			else if (chs_type == "CST")
			{
				if (chs_size == 45 && sprd_size == 40) { presetTarget = presetPos.F_CST_4X_R; presetTargetX = presetPos.F_CST_4XR_RNG; }
				else if (chs_size == 45 && job_pos == "Small") { presetTarget = presetPos.F_CST_20_R;  presetTargetX = presetPos.F_CST_20R_RNG; }
				else if (chs_size == 40 && job_pos == "Small") { presetTarget = presetPos.F_CST_20_F;  presetTargetX = presetPos.F_CST_20F_RNG; }
				else
				{
					presetTarget = presetPos.F_CST_4X_F;
					presetTargetX = presetPos.F_CST_4XF_RNG;
				}
			}
		}
		else
		{
			if (chs_type == "XT")
			{
				if (job_pos == "Big") { presetTarget = presetPos.R_CONE_20_F; presetTargetX = presetPos.R_XT_20F_RNG; }
				else if (chs_size == 40) { presetTarget = presetPos.R_CONE_4X_F; presetTargetX = presetPos.R_XT_4XF_RNG;}
				else { presetTarget = presetPos.R_CONE_4X_R; presetTargetX = presetPos.R_XT_4XR_RNG;}
			}
			else if (chs_type == "IGH")
			{
				if (job_pos == "Big") { presetTarget = presetPos.R_IGH_20_F; presetTargetX = presetPos.R_IGH_20F_RNG; }
				else if (chs_size == 40) { presetTarget = presetPos.R_IGH_4X_F; presetTargetX = presetPos.R_IGH_4XF_RNG; }
				else { presetTarget = presetPos.R_IGH_4X_R; presetTargetX = presetPos.R_IGH_4XR_RNG; }
			}
			else if (chs_type == "CST")
			{
				if (job_pos == "Big") { presetTarget = presetPos.R_CST_20_F; presetTargetX = presetPos.R_CST_20F_RNG; }
				else if (chs_size == 40) { presetTarget = presetPos.R_CST_4X_F; presetTargetX = presetPos.R_CST_4XF_RNG; }
				else { presetTarget = presetPos.R_CST_4X_R; presetTargetX = presetPos.R_CST_4XR_RNG; }
			}
		}


		JobInfo currentJobInfo;
		currentJobInfo.set(Mount_Cycle, Offload_Cycle, job_pos, chs_type, chs_size, sprd_size, presetTarget, presetTargetX);


		IDEAL_POS IP;
		int g_data_limit = 5000;
		int t_data_limit = 5000;

		bool isFrontSensor = (DEBUG_SENSOR_POSITION == "FRONT");

		//set current_lane_camera;
		current_lane_camera = "FS";
		if (isFrontSensor)
		{
			if (job_pos == "Small") current_lane_camera = "CS";
		}
		else
		{
			if (job_pos == "Big") current_lane_camera = "CS";
			else current_lane_camera = "RS";
		}

		//2025.03.11
		bool debug_landed_trigger = false;
		//every 10 files.
		int procSkip = 5;

		for (int i = 0; i < image_files.size(); i++)
		{
			if (i % procSkip != 0) continue;

			logMessage("Processing Cycle Start!");

			std::filesystem::path pathObj(image_files.at(i));
			// Get the filename with extension
			std::string filename = pathObj.stem().string();

			std::string log_lines = filename + ";";

			//folder for each file
			auto save_current_file_path = save_file_path + "/" + filename;
			createDirectory_ifexists(save_current_file_path);

			std::string save_path = save_current_file_path + "/" + filename;
			
			//load image
			cv::Mat image = cv::imread(image_files.at(i));
			cv::Mat res_image;
			if (image.empty())
			{
				logMessage("Couldn't read file by JPG " + image_files.at(i));
				continue;
			}

			pcl::PointCloud<pcl::PointXYZ>::Ptr pcData = std::make_shared<pcl::PointCloud <pcl::PointXYZ>>();
			if (loadedPLY)
			{
				//load ply
				if (pcl::io::loadPLYFile<pcl::PointXYZ>(depth_files.at(i), *pcData) == -1)
				{
					std::cout << std::string("Couldn't read file by PLY ") << std::endl;
					//return false;
				}
			}
			else
			{
				if (pcl::io::loadPCDFile<pcl::PointXYZ>(depth_files.at(i), *pcData) == -1)
				{
					std::cout << std::string("Couldn't read file by PCD ") << std::endl;
					//return false;
				}
			}
			auto pre_pointCloud = makePCL_PointCloud(*pcData, DEBUG_CONVERT_PCL_RANGE);

			auto pointCloud = pointCloud_tilt_transform(pre_pointCloud, TILT_ANGLE_DEG);
			pcl::io::savePCDFileBinaryCompressed(save_path + "_base_converted.pcd", *pointCloud);
			
			ProcessResults procRes = ProcessLogic(isFrontSensor, currentJobInfo, image, pointCloud, save_path, false, true, ref(log_lines));
			
			Simfile.open(sim_result_txt.c_str(), ios::out | ios::app);
			if (!Simfile.is_open())
			{
				logMessage("Failed to open file?");
			}
			if (Simfile.is_open())
			{
				Simfile << log_lines + "\n";
				Simfile.close();
			}
		}

		total_jobs++;

		//offload batch test
		bool bOK_VA = false, bOK_VA_OK_PCA = false, bOK_VA_OK_PCA_ONLY = false, bOK3 = false, bOK_PCA = false, bOK_PCA_OK_PCA_ONLY = false, bOK_PCA_ONLY = false, bOK_N = false;
		bool bCLPS_VA = false, bCLPS_VA_CLPS_PCA = false, bCLPS_PCA = false, bCLPS_N = false;

		//CLPS results
		if (clps_detected && clps_detected_pca) { bCLPS_VA_CLPS_PCA = true; CLPS_VA_CLPS_PCA++; }
		else if (clps_detected) { bCLPS_VA = true; CLPS_VA++; }
		else if (clps_detected_pca) { bCLPS_PCA = true; CLPS_PCA++; }
		else { bCLPS_N = true; CLPS_N++;  }

		//CLPS-OK results
		if (clps_ok_detected && clps_ok_detected_pca && clps_ok_detected_pca_only) { bOK3 = true; OK3++; }
		if (clps_ok_detected && clps_ok_detected_pca) { bOK_VA_OK_PCA = true; OK_VA_OK_PCA++; }
		if (clps_ok_detected && clps_ok_detected_pca_only) { bOK_VA_OK_PCA_ONLY = true; OK_VA_OK_PCA_ONLY++; }
		if (clps_ok_detected_pca && clps_ok_detected_pca_only) { bOK_PCA_OK_PCA_ONLY = true; OK_PCA_OK_PCA_ONLY++; }
		
		if (clps_ok_detected) { bOK_VA = true; OK_VA++; }
		if (clps_ok_detected_pca) { bOK_PCA = true; OK_PCA++; }
		if (clps_ok_detected_pca_only) { bOK_PCA_ONLY = true; OK_PCA_ONLY++; }
		if (!clps_ok_detected && !clps_ok_detected_pca && !clps_ok_detected_pca_only) { bOK_N = true; OK_N++; }

		//auto logHeader_offload_batch = std::string("Total_Job;OK3;OK-VA,OK-PCA;OK-VA,OK-PCA-ONLY;OK-PCA,OK-PCA-ONLY;OK-VA;OK-PCA;OK-PCA-ONLY;OK-N;CLPS-VA,CLPS-PCA;CLPS-VA;CLPS-PCA;CLPS-N");

		std::string batch_line = lastDirectory + ";" + std::to_string(bOK3) + ";" + std::to_string(bOK_VA_OK_PCA) + ";" + std::to_string(bOK_VA_OK_PCA_ONLY) + ";" + std::to_string(bOK_PCA_OK_PCA_ONLY) + ";"
			+ std::to_string(bOK_VA) + ";" + std::to_string(bOK_PCA) + ";" + std::to_string(bOK_PCA_ONLY) + ";" + std::to_string(bOK_N);
		batch_line += ";" + std::to_string(bCLPS_VA_CLPS_PCA) + ";" + std::to_string(bCLPS_VA) + ";" + std::to_string(bCLPS_PCA) + ";" + std::to_string(bCLPS_N);

		//ofstream Simfile_batch;
		//std::string sim_batch_result_txt = DEBUG_BATCH_ROOT_DIR + "/Offload_batch_results.txt";
		Simfile_batch.open(sim_batch_result_txt.c_str(), ios::out | ios::app);
		if (!Simfile_batch.is_open())
		{
			logMessage("Failed to open file?");
		}
		if (Simfile_batch.is_open())
		{
			Simfile_batch << batch_line + "\n";
			Simfile_batch.close();
		}

		//break;
	}

	std::string batch_line = std::to_string(total_jobs) + ";" + std::to_string(OK3) + ";" + std::to_string(OK_VA_OK_PCA) + ";" + std::to_string(OK_VA_OK_PCA_ONLY) + ";" + std::to_string(OK_PCA_OK_PCA_ONLY) + ";"
		+ std::to_string(OK_VA) + ";" + std::to_string(OK_PCA) + ";" + std::to_string(OK_PCA_ONLY) + ";" + std::to_string(OK_N);
	batch_line += ";" + std::to_string(CLPS_VA_CLPS_PCA) + ";" + std::to_string(CLPS_VA) + ";" + std::to_string(CLPS_PCA) + ";" + std::to_string(CLPS_N);

	//ofstream Simfile_batch;
	//std::string sim_batch_result_txt = DEBUG_BATCH_ROOT_DIR + "/Offload_batch_results.txt";
	Simfile_batch.open(sim_batch_result_txt.c_str(), ios::out | ios::app);
	if (!Simfile_batch.is_open())
	{
		logMessage("Failed to open file?");
	}
	if (Simfile_batch.is_open())
	{
		Simfile_batch << batch_line + "\n";
		Simfile_batch.close();
	}
}

void handleClient(SOCKET clientSocket) {
	logMessage("Client connected. Sending initial message...");

	char buffer[DEFAULT_RECVLEN] = { 0 };
	while (socket_running.load()) 
	{
		int bytesReceived = recv(clientSocket, buffer, sizeof(buffer), 0);
		if (bytesReceived <= 0) {
			logMessage("Client disconnected. Waiting for new client...");
			break;
		}
		//buffer[bytesReceived] = '\0';

		parseCommand(buffer);

		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		
		char sendbuf[DEFAULT_SENDLEN] = {};
		makeCommand(ref(sendbuf));

		int bytesSent = send(clientSocket, sendbuf, DEFAULT_SENDLEN, 0);
		if (bytesSent <= 0)
		{
			logMessage("Sending data failed!");
			break;
		}
	}
	closesocket(clientSocket);
}
//2025.02.06 Updated socket server
int start_server() {
	WSADATA wsaData;
	SOCKET serverSocket;
	struct sockaddr_in serverAddr;
	int addrlen = sizeof(serverAddr);

	// Initialize Winsock
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
		std::cerr << "WSAStartup failed!\n";
		logMessage("WSAStartup failed!");
		return 1;
	}

	// Create socket
	serverSocket = socket(AF_INET, SOCK_STREAM, 0);
	if (serverSocket == INVALID_SOCKET) {
		std::cerr << "Socket creation failed!\n";
		logMessage("Socket creation failed!");
		WSACleanup();
		return 1;
	}

	BOOL on = TRUE;
	if (setsockopt(serverSocket, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
		reinterpret_cast<const char*>(&on), sizeof(on)) == SOCKET_ERROR) {
		std::cerr << "Socket reuse port failed!\n";
		logMessage("Socket reuse port failed!");
		WSACleanup();
		return 1;
	}

	// Bind socket
	serverAddr.sin_family = AF_INET;
	serverAddr.sin_addr.s_addr = INADDR_ANY;
	serverAddr.sin_port = htons(SOCKET_PORT);

	if (::bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
		std::cerr << "Bind failed!\n";
		logMessage("Bind failed!");
		closesocket(serverSocket);
		WSACleanup();
		return 1;
	}

	// Start listening
	if (listen(serverSocket, SOMAXCONN) == SOCKET_ERROR) {
		std::cerr << "Listen failed!\n";
		logMessage("Listen failed");
		closesocket(serverSocket);
		WSACleanup();
		return 1;
	}

	logMessage("Server listening on port " + std::to_string(SOCKET_PORT) + "...");

	while (socket_running.load()) 
	{
		try
		{
			struct sockaddr_in clientAddr;
			int clientSize = sizeof(clientAddr);
			SOCKET clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &clientSize);

			if (clientSocket == INVALID_SOCKET)
			{
				logMessage("Accept failed!");
				std::cerr << "Accept failed!\n";
				continue;
			}

			// Handle client in a separate thread
			std::thread clientThread(handleClient, clientSocket);

			clientThread.join();  // Wait for the thread to finish

			//clientThread.detach();  // Let it run independently
			logMessage("Waiting for new client...");
		}
		catch (std::exception& ex)
		{
			logMessage("[SERVER] " + std::string(ex.what()));
		}
		catch (...)
		{
			logMessage("[SERVER] Unknown Exception!");
		}
	}

	logMessage("Terminating Socket Server");
	closesocket(serverSocket);
	WSACleanup();
	return 0;
}

void socketClient(const std::string& server_ip, int server_port)
{
	WSADATA wsaData;
	WSAStartup(MAKEWORD(2, 2), &wsaData);  // Init Winsock

	while (socket_running) {
		SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
		if (sock == INVALID_SOCKET) {
			logMessage("Socket creation failed");
			std::this_thread::sleep_for(std::chrono::seconds(2));
			continue;
		}

		sockaddr_in server_addr{};
		server_addr.sin_family = AF_INET;
		server_addr.sin_port = htons(server_port);
		inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr);

		logMessage("Attempting to connect to " + server_ip + ":" + std::to_string(server_port) + "...");
		if (connect(sock, (sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
			logMessage("Connection failed: " + std::to_string(WSAGetLastError()));
			closesocket(sock);
			std::this_thread::sleep_for(std::chrono::seconds(2));
			continue;
		}

		logMessage("Connected to server!");

		//initial send?
		char sendbuf_init[DEFAULT_SENDLEN] = { 0 };
		//makeCommand(ref(sendbuf));
		int bytesSent_init = send(sock, sendbuf_init, DEFAULT_SENDLEN, 0);

		// Sample loop: Read from server
		char buffer[DEFAULT_RECVLEN] = { 0 };
		while (socket_running) {
			int bytesReceived = recv(sock, buffer, sizeof(buffer), 0);
			if (bytesReceived <= 0) {
				logMessage("Disconnected from server");
				break;
			}
			//buffer[bytesReceived] = '\0';
			std::string received_msg(buffer);
			
			parseCommand(buffer);

			std::this_thread::sleep_for(std::chrono::milliseconds(100));

			char sendbuf[DEFAULT_SENDLEN] = { 0 };
			makeCommand(ref(sendbuf));

			int bytesSent = send(sock, sendbuf, DEFAULT_SENDLEN, 0);
			if (bytesSent <= 0)
			{
				logMessage("Sending data failed! " + std::to_string(bytesSent));
				break;
			}
		}

		//On connection lost/disconnect, clean up variables
		reset_sendVariables();
		reset_recvVariables();

		closesocket(sock);
		logMessage("Connection closed. Retrying in 2 seconds...");
		std::this_thread::sleep_for(std::chrono::seconds(2)); // Wait before retry
	}

	WSACleanup();
}


int main(int argc, char* argv[])
{
	wchar_t buf[MAX_PATH];
	DWORD sz = GetModuleFileNameW(nullptr, buf, MAX_PATH);
	if (sz > 0 && sz < MAX_PATH) {
		std::wstring exePath(buf);
		app_path = Utf16ToUtf8(exePath);
	}

	app_path = getParent(app_path);

	HANDLE mutex = CreateMutexA(nullptr, TRUE, app_path.c_str());

	if (GetLastError() == ERROR_ALREADY_EXISTS) {
		std::cerr << "Another instance is already running.\n";
		return 1;
	}

	SetUnhandledExceptionFilter(MyUnhandledExceptionFilter);

	printf("TUAS South Endfeed App\n");

	parseAppName(app_path);

	appName = appID + "_v" + program_version;
	std::cout << "App: " << appName << "\n";

	std::thread logThread(logWriterThread);

	logMessage(appName + " Starting!");
	logMessage(app_path);

	signal(SIGINT, my_handler);

	parseINI(app_path);
	parseProcessINI(app_path);
	parseIPINI(app_path);

	//TensorRT
	if (!ENABLE_IMAGE_LOGGING_ONLY && false)
	{
		try {
			if (!trt_detector.loadEngine("model/20260702_TUAS_SEF_3DSP.engine")) {
				printf("loadEngine failed\n");
			}

			std::string imagePath = app_path + "/initialize.jpg";
			cv::Mat frame = cv::imread(imagePath);

			LetterboxInfo lb;
			cv::Mat in = letterbox(frame, 640, 640, lb);

			std::vector<float> inputCHW;
			matToCHW(in, inputCHW);

			size_t inputBytes = inputCHW.size() * sizeof(float);

			if (!trt_detector.infer("images", inputCHW.data(), inputBytes)) {
				printf("infer failed\n");
				return 1;
			}

			auto outs = trt_detector.outputs();
			printf("Got %zu output tensors\n", outs.size());
			for (auto& o : outs) {
				printf("Output %s bytes=%zu dims=[", o.name.c_str(), o.bytes);
				for (int i = 0; i < o.dims.nbDims; i++) printf("%d%s", o.dims.d[i], (i + 1 < o.dims.nbDims ? "," : ""));
				printf("]\n");
			}

			model_loaded = true;
			model_initialized = true;

			/*
			const auto& out0 = outs.at(0);

			// out0.host is bytes; convert to float pointer
			const float* pred = reinterpret_cast<const float*>(out0.host.data());

			// dims: [1,8,8400]
			int C = out0.dims.d[1];   // 8
			int N = out0.dims.d[2];   // 8400
			int numClasses = C - 4;   // 4

			float confThresh = 0.25f;
			float iouThresh = 0.45f;

			std::vector<RectInfo> dets = decodeYoloV8_1xCxN(
				pred, C, N, numClasses, confThresh,
				lb, frame.cols, frame.rows
			);

			std::vector<RectInfo> finalDets = nms(dets, iouThresh);

			// Draw
			for (const auto& d : finalDets) {
				cv::rectangle(frame, d.box, cv::Scalar(0, 255, 0), 2);
				char txt[64];
				sprintf_s(txt, "cls=%d %.2f", d.class_id, d.score);
				cv::putText(frame, txt, cv::Point((int)d.box.x, (int)d.box.y - 5),
					cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
			}
			cv::imshow("test", frame);
			cv::waitKey(0);
			*/
		}
		catch (const std::exception& e) {
			printf("EXCEPTION: %s\n", e.what());
			return 1;
		}
		catch (...) {
			printf("UNKNOWN EXCEPTION\n");
			return 1;
		}
	}

	//YOLOv11
	
	bool useGPU = true;
	
	std::string modelPath = app_path + "/" + onnx_model_path;
	std::string labelsPath = app_path + "/INI/classes.txt";

	logMessage("Initializing YOLOv11 detector with model: " + modelPath);
	logMessage("Classes file: " + labelsPath);
	logMessage("Using confidence threshold: " + std::to_string(confThreshold) + ", IoU threshold: " + std::to_string(iouThreshold));

	// read model 
	//std::cout << "Loading model and labels..." << std::endl;

	//YOLO11Detector detector (modelPath, labelsPath, useGPU);
	yolo_detector.initialize(modelPath, labelsPath, useGPU);
	model_loaded = true;

	std::string imagePath = app_path + "/initialize.jpg";
	cv::Mat frame = cv::imread(imagePath);
	auto start_time = std::chrono::high_resolution_clock::now();

	// Perform detection with the updated thresholds
	std::vector<Detection> detections = yolo_detector.detect(frame, confThreshold, iouThreshold);
	
	//auto end_time = std::chrono::high_resolution_clock::now();
	//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

	//logMessage("Detection completed in " + std::to_string(duration) + "ms");

	//model_initialized = (detections.size() > 0) ? true : false;

	if (DEBUG_WITH_FILES || DEBUG_BATCH_JOB)
	{
		//test();

		if (DEBUG_BATCH_JOB) OfflineDebugBatchProcessingThread();

		//std::thread rtspT(rtsp_stream_receiver);
		//rtspT.join();
		//if (DEBUG_BATCH_JOB) ZedX_Processing_Test();
	}
	else if (MODE_DEBUG)
	{
		logMessage("Debug Mode running.. Testing for onnx.");

		//CURRENT_SENSOR_POSITION = "REAR_LEFT";

		DEBUG_IMG_PATH = DEBUG_SAMPLE_JOB + "/Image";
		DEBUG_PLY_PATH = DEBUG_SAMPLE_JOB + "/Depth";

		DEBUG_IMG_FILES = getAllFiles(DEBUG_IMG_PATH, ".jpg");
		DEBUG_PLY_FILES = getAllFiles(DEBUG_PLY_PATH, ".pcd");

		DEBUG_MAX_INDEX = (DEBUG_IMG_FILES.size() < DEBUG_PLY_FILES.size()) ? DEBUG_IMG_FILES.size() : DEBUG_PLY_FILES.size();
		DEBUG_CURRENT_INDEX = 0;

		/*
		demo_img = cv::imread("20250302_041440_568_TMini_Image.jpg", cv::IMREAD_COLOR);

		pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud = std::make_shared<pcl::PointCloud <pcl::PointXYZ>>();
		if (pcl::io::loadPLYFile<pcl::PointXYZ>("20250302_041440_568_TMini_Depth.pcd", *pointCloud) == -1)
		{
			std::cout << std::string("Couldn't read file by PLY ") << std::endl;
			//return false;
		}
		
		demo_ply->points.resize(pointCloud->points.size());

		std::transform(pointCloud->points.begin(), pointCloud->points.end(), demo_ply->points.begin(),
			[](pcl::PointXYZ val) { return pcl::PointXYZ(val.x * 1000, val.y * 1000, val.z * 1000); });
		*/
		//std::thread sckT(start_server);
		std::thread joblogThread(jobLogWriterThread);

		std::thread sckTMini_Setup(thread_tmini_control);
		std::thread sckTMini_Data_Stream(thread_tmini_data_stream);

		//need to specify and enable connection.
		//current_lane_ip = IP_ADDRESSES[0];
		
		//trigger tmini setup and stream.
		//tmini_ctrl_flag = true;
		//cond_tmini_ctrl.notify_one();

		//std::thread tProc(processingThread, std::ref(detector));
		std::thread tSave(data_save_thread);

		//tProc.join();
		joblogThread.join();
		sckTMini_Setup.join();
		sckTMini_Data_Stream.join();
		tSave.join();
		//sckT.join();
	}
	else
	{
		//std::thread sckT(start_server);
		std::thread sckClientThread(socketClient, SOCKET_IP, SOCKET_PORT);

		//Visionary T Mini Streaming setup.
		std::thread sckTMini_Setup(thread_tmini_control);
		std::thread sckTMini_Data_Stream(thread_tmini_data_stream);

		std::thread sckProcess(processingThread);
		std::thread sckSave(data_save_thread);


		if (sckProcess.joinable()) sckProcess.join();
		if (sckTMini_Setup.joinable()) sckTMini_Setup.join();
		if (sckTMini_Data_Stream.joinable()) sckTMini_Data_Stream.join();
		
		if (trigger_terminate)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			if (sckTMini_Setup.joinable())
			{
				logMessage("Program Just Shutting Down...");

				log_running.store(false);
				cvLog.notify_one();
				logThread.join();

				//std::cout << "Press any key to continue...";
				//system("pause"); // Windows only
				//std::exit(-1);
				return 0;
			}
		}

		if (sckSave.joinable()) sckSave.join();
		if (sckClientThread.joinable()) sckClientThread.join();
		
	}

	logMessage("Program Shutting Down...");

	log_running.store(false);
	cvLog.notify_one();
	logThread.join();

	//std::cout << "Press any key to continue...";
	//system("pause"); // Windows only

	return 1;
}