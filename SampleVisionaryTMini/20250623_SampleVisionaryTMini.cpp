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
#include "VA.h"

#pragma comment(lib, "Dbghelp.lib")
#include <DbgHelp.h>

#define DEFAULT_RECVLEN 50
#define DEFAULT_SENDLEN 50

const std::string program_version = "1.0";

struct bbx
{
	int label = 0;
	int x = 0;
	int y = 0;
	int center_x = 0;
	int center_y = 0;
	int w = 0;
	int h = 0;
	int prob = 0;
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

#pragma region INI Variables

std::string sensor_ip;
std::string sensor_port;

std::string SOCKET_IP = "127.0.0.1";
int SOCKET_PORT;

const int sendLen = 50;
const int recvLen = 50;

bool DEBUG_MODE = true;

std::string DEBUG_SAMPLE_JOB = "";

bool DEBUG_WITH_FILES = false;
std::string DEBUG_PATH;
std::string DEBUG_SENSOR_POSITION = "REAR_LEFT";

bool DEBUG_BATCH_JOB = false;
std::string DEBUG_BATCH_ROOT_DIR = "";
std::string DEBUG_BATCH_SAVE_DIR = "";

bool DEBUG_CONVERT_PCL_RANGE = false;

bool SEQUENTIAL_PROCESSING = false;

std::string CURRENT_SENSOR_POSITION = "REAR_LEFT";

int PROB_LIMIT = 50;

std::string onnx_model_path;

int LANE_NUM_AS_LEFT = 0;
int LANE_NUM_AS_RIGHT = 0;

IDEAL_POS L_Pos;
IDEAL_POS R_Pos;
IDEAL_POS JOB_IP_Pos;

int G_DATA_LIMIT = 3000;
int T_DATA_LIMIT = 2000;

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

std::vector<std::string> IP_ADDRESSES = {};
std::vector<std::string> CAM_IP_ADDRESSES = {};
#pragma endregion

std::string current_lane_ip = "";
std::string current_lane_cam_ip = "";

SOCKET connectSocket_ = NULL;
unsigned int heartbeat_, prev_heartbeat_;
int visionary_version_[4]{ 0 };
char block_n_[5]{};
char* block_;
int tzms_version_;

char* sendBuffer_;
char sendBuf_[DEFAULT_SENDLEN] = {};

char* recvBuffer_;
char recvBuf_[DEFAULT_RECVLEN] = {};

auto sck_connected = false;
//auto save_frame = false;

bool blnDataSave;
std::string SaveDir;
std::string SaveImageDir;
std::string SaveDepthDir;

//Job information
bool TZ_Offload_Cycle = false;
bool TZ_Mount_Cycle = false;

bool Job_Pos_B = false;
bool Job_Pos_S = false;
bool Job_Pos_C = false;

bool SPRD_TWL_Locked = false;
bool SPRD_TWL_Unlocked = false;
bool SPRD_Landed = false;

bool SPRD_20ft = false;
bool SPRD_40ft = false;
bool SPRD_45ft = false;

bool Target_20ft = false;
bool Target_40ft = false;
bool Target_45ft = false;

bool Hoist_Stopped = false;
bool Trolley_Stopped = false;
bool Gantry_Stopped = false;

bool Above_Safe_Height = false;

bool CHS_20ft, CHS_40ft, CHS_45ft;
bool CHS_XT, CHS_CST;

bool Logging_Enable = false;
bool Front_Left_Logging = false;
bool Front_Right_Logging = false;
bool Center_Left_Logging = false;
bool Center_Right_Logging = false;
bool Rear_Logging = false;

short Target_Height_Pos_mm = 0;
short Current_Hoist_Pos_mm = 0;
int Current_Trolley_Pos_mm = 0;
int Target_Trolley_Pos_mm = 0;

short intrim_Target_Height_Pos_mm = 0;
short intrim_Current_Hoist_Pos_mm = 0;
int intrim_Current_Trolley_Pos_mm = 0;
int intrim_Target_Trolley_Pos_mm = 0;

std::string saveName;
std::string saveDirName;

std::string job_name;
std::string job_result_folder_name;
//images, pointcloud, and .log files
std::string job_result_log_file_name;

//20250508 -- cam rgb
bool ENABLE_CAM_RGB = false;
bool cam_rgb_connected = false;
bool cam_rgb_fault = false;

bool sensor_connected = false;
bool sensor_fault = false;

std::chrono::system_clock::time_point sensor_last_attempted_time = std::chrono::system_clock::now();
std::chrono::system_clock::time_point sensor_last_connected_time = std::chrono::system_clock::now();

bool enable_stream = false;
bool enable_process = false;
bool enable_logging = false;

bool processThread_stopped = false;

bool ENABLE_SAVE_LOG_BIT = false;
bool enable_save_logs = false;

bool cam_connected = false;
bool enabled_stream = false;
bool enabled_process = false;
bool enabled_logging = false;

bool trigger_log_delete = false;

bool model_loaded = false;
bool model_initialized = false;

int tCntr_x, tCntr_y, tCntr_prob;
int tCone_x, tCone_y, tCone_prob;
int chassis_detected_type;

bool detected_xt = false;
bool detected_cst = false;
bool detected_chassis_type_unknown = false;

bool landout_detected = false;
bool clps_detected = false;
bool clps_ok_detected = false;

int target_lane_number = 1;

int devOut_x, devOut_y;
//2025.03.11
int devOut_x_mm, devOut_y_mm;
int devOut_pca_x, devOut_pca_y, devOut_pca_z;

bool landout_detected_pca = false;
int landout_current_count_pca = 0;
bool clps_detected_pca = false;
int clps_current_count_pca = 0;
bool clps_ok_detected_pca = false;
int clps_ok_current_count_pca = 0;

bbx Offload_Hole_Base;
bbx Offload_Hole_intrim;
int Offload_Hole_Base_Count = 0;
bool Offload_Hole_Base_Set = false;

pcl::PointXYZ PCA_intrim;
pcl::PointXYZ PCA_Base;
int PCA_Base_Count = 0;
bool PCA_Base_Set = false;

bbx LDO_intrim;
bbx LDO_Base;
int LDO_Base_Count = 0;
bool LDO_Base_Set = false;

int LDO_Current_Count = 0;
int CLPS_Current_Count = 0;

bbx CLPS_intrim;
bbx CLPS_Base;
int CLPS_Base_Count = 0;
bool CLPS_Base_Set = false;

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

void reset_jobVariables()
{
	tCntr_x = -10000; tCntr_y = -10000; tCntr_prob = -10000;
	tCone_x = -10000; tCone_y = -10000; tCone_prob = -10000;
	detected_xt = false; detected_cst = false; detected_chassis_type_unknown = false;

	devOut_x = -10000; devOut_y = -10000;
	devOut_pca_x = -10000; devOut_pca_y = -10000; devOut_pca_z = -10000;

	LDO_Current_Count = 0; CLPS_Current_Count = 0;

	landout_detected_pca = false;
	landout_current_count_pca = 0;
	clps_detected_pca = false;
	clps_current_count_pca = 0;
	clps_ok_detected_pca = false;
	clps_ok_current_count_pca = 0;

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
std::atomic<bool > tmini_ctrl_running(true);
bool tmini_ctrl_flag = false;

std::mutex mutex_tmini_data_stream;
std::condition_variable cond_tmini_data_stream;
//this one is for running/terminating thread.
std::atomic<bool> tmini_data_stream_running(true);
bool tmini_data_stream_flag = false;

std::mutex mutex_camRGB_ctrl;
std::condition_variable cond_camRGB_ctrl;
//this one is for running/terminating thread.
std::atomic<bool > camRGB_ctrl_running(true);
bool camRGB_ctrl_flag = false;
bool camRGB_stream_flag = false;

std::mutex mutex_processing;
std::condition_variable cond_processing;
//this one is for running/terminating thread.
std::atomic<bool> proc_running(true);

std::mutex mutex_seq_processing;
std::condition_variable cond_seq_processing;
//this one is for running/terminating thread.
std::atomic<bool> seq_proc_running(true);
bool seq_proc_flag = false;


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

cv::VideoCapture camRGB_cap;


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
class ThreadSafeQueue_rgb
{
private:
	std::queue <std::tuple<cv::Mat, cv::Mat, bool, std::string, std::chrono::system_clock::time_point>> queue;
	std::mutex mtx;
	std::condition_variable cv;

public:
	void push(std::tuple<cv::Mat, cv::Mat, bool, std::string, std::chrono::system_clock::time_point> value) {
		std::lock_guard<std::mutex> lock(mtx);
		queue.push(value);
		cv.notify_one(); // Notify the consumer
	}

	// Pop element from the queue (blocks if empty)
	std::tuple<cv::Mat, cv::Mat, bool, std::string, std::chrono::system_clock::time_point> pop() {
		std::unique_lock<std::mutex> lock(mtx);
		cv.wait(lock, [this] { return !queue.empty(); }); // Wait until queue is non-empty
		std::tuple<cv::Mat, cv::Mat, bool, std::string, std::chrono::system_clock::time_point> value = queue.front();
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
ThreadSafeQueue tsq;
ThreadSafeQueue_rgb tsq_rgb;

DataStack dataStack;
DataStack_RGB dataStackRGB;

bool flag = false;
bool saveFlag = false;

bool jobLogFlag = false;

void logMessage(const std::string& message);

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

// Function to write log entries to file
void logWriterThread() {
	std::ofstream logFile;

	createDirectory_ifexists("./Log");

	while (log_running || !logQueue.empty()) {
		try
		{
			std::unique_lock<std::mutex> lock(queueMutex);
			cvLog.wait(lock, [] { return !logQueue.empty() || !log_running; });

			// Get the current date and check if it has changed
			std::string newLogFileName = "./Log/" + appName + "_log_" + getCurrentDate() + ".log";

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

auto parseAppName() -> bool
{
	try
	{
		mINI::INIFile inireader("INI/seoho.ini");
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
auto parseINI() -> bool
{
	try
	{
		mINI::INIFile inireader("INI/seoho.ini");
		mINI::INIStructure inidata;

		inireader.read(inidata);

		appID = inidata.get("ID").get("APP");
		if (appID.find("LEFT") != std::string::npos)
		{
			CURRENT_SENSOR_POSITION = "REAR_LEFT";
			logMessage("Current application is for REAR LEFT sensor!");
		}
		else
		{
			CURRENT_SENSOR_POSITION = "REAR_RIGHT";
			logMessage("Current application is for REAR RIGHT sensor!");
		}
		//port
		//sensor_port = inidata.get("SENSOR").get("PORT");
		//logMessage("Port: " + sensor_port);

		//socket port
		SOCKET_PORT = std::stoi(inidata.get("SOCKET").get("SERVER_PORT"));
		logMessage("socket ip,port: " + SOCKET_IP + " , " + std::to_string(SOCKET_PORT));
		
		PROB_LIMIT = std::stoi(inidata.get("SYSTEM").get("PROB_LIMIT"));
		ENABLE_CAM_RGB = inidata.get("SYSTEM").get("ENABLE_CAM_RGB") == "1" ? true : false;
		ENABLE_SAVE_LOG_BIT = inidata.get("SYSTEM").get("ENABLE_SAVE_LOG_BIT") == "1" ? true : false;

		SEQUENTIAL_PROCESSING = inidata.get("SYSTEM").get("SEQUENTIAL_PROCESSING") == "1" ? true : false;

		onnx_model_path = inidata.get("MODEL").get("ONNX_PATH");
		logMessage("onnx model path: " + onnx_model_path);

		DEBUG_MODE = inidata.get("DEBUG").get("DEBUG_MODE") == "1" ? true : false;
		if (DEBUG_MODE) logMessage("Debug mode enabled!");

		DEBUG_SAMPLE_JOB = inidata.get("DEBUG").get("DEBUG_SAMPLE_JOB");

		DEBUG_CONVERT_PCL_RANGE = inidata.get("DEBUG").get("DEBUG_CONVERT_PCL_RANGE") == "1" ? true : false;

		DEBUG_SENSOR_POSITION = inidata.get("DEBUG").get("DEBUG_SENSOR_POSITION");
		
		DEBUG_WITH_FILES = inidata.get("DEBUG").get("DEBUG_WITH_FILES") == "1" ? true : false;

		DEBUG_PATH = inidata.get("DEBUG").get("DEBUG_PATH");

		DEBUG_BATCH_JOB = inidata.get("DEBUG").get("DEBUG_BATCH_JOB") == "1" ? true : false;
		if (DEBUG_BATCH_JOB) logMessage("Debug mode in batch mode!");
		DEBUG_BATCH_ROOT_DIR = inidata.get("DEBUG").get("DEBUG_BATCH_ROOT_DIR");
		DEBUG_BATCH_SAVE_DIR = inidata.get("DEBUG").get("DEBUG_BATCH_SAVE_DIR");

		if (DEBUG_MODE || DEBUG_WITH_FILES || DEBUG_BATCH_JOB) logMessage("Debug Sensor Pos: " + DEBUG_SENSOR_POSITION);

		logMessage("INI Configuration Complete!");
	}
	catch (std::exception& e)
	{
		logMessage("Error in parsing INI file : " + std::string(e.what()));
		return false;
	}
	return true;
}

void parseProcessINI()
{
	mINI::INIFile file("ini/process.ini");
	mINI::INIStructure ini;

	if (file.read(ini))
	{
		try
		{
			G_DATA_LIMIT = std::stoi(ini.get("MEASURE_DISTANCE").get("GANTRY"));
			T_DATA_LIMIT = std::stoi(ini.get("MEASURE_DISTANCE").get("TROLLEY"));

			{ //Left Pos
				L_Pos.CONE.x = std::stoi(ini.get("IDEAL_POS_VA").get("L_CONE_X")) - 5;
				L_Pos.CONE.y = std::stoi(ini.get("IDEAL_POS_VA").get("L_CONE_Y")) - 5;
				L_Pos.CONE.w = std::stoi(ini.get("IDEAL_POS_VA").get("L_CONE_W")) + 10;
				L_Pos.CONE.h = std::stoi(ini.get("IDEAL_POS_VA").get("L_CONE_H")) + 10;
				L_Pos.CONE.center_x = L_Pos.CONE.x + int(L_Pos.CONE.w / 2);
				L_Pos.CONE.center_y = L_Pos.CONE.y + int(L_Pos.CONE.h / 2);

				L_Pos.LANDED.x = std::stoi(ini.get("IDEAL_POS_VA").get("L_LANDED_X")) - 5;
				L_Pos.LANDED.y = std::stoi(ini.get("IDEAL_POS_VA").get("L_LANDED_Y")) - 5;
				L_Pos.LANDED.w = std::stoi(ini.get("IDEAL_POS_VA").get("L_LANDED_W")) + 10;
				L_Pos.LANDED.h = std::stoi(ini.get("IDEAL_POS_VA").get("L_LANDED_H")) + 10;
				L_Pos.LANDED.center_x = L_Pos.LANDED.x + int(L_Pos.LANDED.w / 2);
				L_Pos.LANDED.center_y = L_Pos.LANDED.y + int(L_Pos.LANDED.h / 2);

				L_Pos.GUIDE.x = std::stoi(ini.get("IDEAL_POS_VA").get("L_GUIDE_X")) - 5;
				L_Pos.GUIDE.y = std::stoi(ini.get("IDEAL_POS_VA").get("L_GUIDE_Y")) - 5;
				L_Pos.GUIDE.w = std::stoi(ini.get("IDEAL_POS_VA").get("L_GUIDE_W")) + 10;
				L_Pos.GUIDE.h = std::stoi(ini.get("IDEAL_POS_VA").get("L_GUIDE_H")) + 10;
				L_Pos.GUIDE.center_x = L_Pos.GUIDE.x + int(L_Pos.GUIDE.w / 2);
				L_Pos.GUIDE.center_y = L_Pos.GUIDE.y + int(L_Pos.GUIDE.h / 2);

				L_Pos.CONE_PCA.x = std::stof(ini.get("IDEAL_POS_PCA").get("L_CONE_X"));
				L_Pos.CONE_PCA.y = std::stof(ini.get("IDEAL_POS_PCA").get("L_CONE_Y"));
				L_Pos.CONE_PCA.z = std::stof(ini.get("IDEAL_POS_PCA").get("L_CONE_Z"));

				L_Pos.LANDED_PCA.x = std::stof(ini.get("IDEAL_POS_PCA").get("L_LANDED_X"));
				L_Pos.LANDED_PCA.y = std::stof(ini.get("IDEAL_POS_PCA").get("L_LANDED_Y"));
				L_Pos.LANDED_PCA.z = std::stof(ini.get("IDEAL_POS_PCA").get("L_LANDED_Z"));

				L_Pos.GUIDE_PCA.x = std::stof(ini.get("IDEAL_POS_PCA").get("L_GUIDE_X"));
				L_Pos.GUIDE_PCA.y = std::stof(ini.get("IDEAL_POS_PCA").get("L_GUIDE_Y"));
				L_Pos.GUIDE_PCA.z = std::stof(ini.get("IDEAL_POS_PCA").get("L_GUIDE_Z"));
			}
			{ //Right Pos
				R_Pos.CONE.x = std::stoi(ini.get("IDEAL_POS_VA").get("R_CONE_X")) - 5;
				R_Pos.CONE.y = std::stoi(ini.get("IDEAL_POS_VA").get("R_CONE_Y")) - 5;
				R_Pos.CONE.w = std::stoi(ini.get("IDEAL_POS_VA").get("R_CONE_W")) + 10;
				R_Pos.CONE.h = std::stoi(ini.get("IDEAL_POS_VA").get("R_CONE_H")) + 10;
				R_Pos.CONE.center_x = R_Pos.CONE.x + int(R_Pos.CONE.w / 2);
				R_Pos.CONE.center_y = R_Pos.CONE.y + int(R_Pos.CONE.h / 2);

				R_Pos.LANDED.x = std::stoi(ini.get("IDEAL_POS_VA").get("R_LANDED_X")) - 5;
				R_Pos.LANDED.y = std::stoi(ini.get("IDEAL_POS_VA").get("R_LANDED_Y")) - 5;
				R_Pos.LANDED.w = std::stoi(ini.get("IDEAL_POS_VA").get("R_LANDED_W")) + 10;
				R_Pos.LANDED.h = std::stoi(ini.get("IDEAL_POS_VA").get("R_LANDED_H")) + 10;
				R_Pos.LANDED.center_x = R_Pos.LANDED.x + int(R_Pos.LANDED.w / 2);
				R_Pos.LANDED.center_y = R_Pos.LANDED.y + int(R_Pos.LANDED.h / 2);

				R_Pos.GUIDE.x = std::stoi(ini.get("IDEAL_POS_VA").get("R_GUIDE_X")) - 5;
				R_Pos.GUIDE.y = std::stoi(ini.get("IDEAL_POS_VA").get("R_GUIDE_Y")) - 5;
				R_Pos.GUIDE.w = std::stoi(ini.get("IDEAL_POS_VA").get("R_GUIDE_W")) + 10;
				R_Pos.GUIDE.h = std::stoi(ini.get("IDEAL_POS_VA").get("R_GUIDE_H")) + 10;
				R_Pos.GUIDE.center_x = R_Pos.GUIDE.x + int(R_Pos.GUIDE.w / 2);
				R_Pos.GUIDE.center_y = R_Pos.GUIDE.y + int(R_Pos.GUIDE.h / 2);

				R_Pos.CONE_PCA.x = std::stof(ini.get("IDEAL_POS_PCA").get("R_CONE_X"));
				R_Pos.CONE_PCA.y = std::stof(ini.get("IDEAL_POS_PCA").get("R_CONE_Y"));
				R_Pos.CONE_PCA.z = std::stof(ini.get("IDEAL_POS_PCA").get("R_CONE_Z"));

				R_Pos.LANDED_PCA.x = std::stof(ini.get("IDEAL_POS_PCA").get("R_LANDED_X"));
				R_Pos.LANDED_PCA.y = std::stof(ini.get("IDEAL_POS_PCA").get("R_LANDED_Y"));
				R_Pos.LANDED_PCA.z = std::stof(ini.get("IDEAL_POS_PCA").get("R_LANDED_Z"));

				R_Pos.GUIDE_PCA.x = std::stof(ini.get("IDEAL_POS_PCA").get("R_GUIDE_X"));
				R_Pos.GUIDE_PCA.y = std::stof(ini.get("IDEAL_POS_PCA").get("R_GUIDE_Y"));
				R_Pos.GUIDE_PCA.z = std::stof(ini.get("IDEAL_POS_PCA").get("R_GUIDE_Z"));
			}

			//Landout-VA
			LDO_NEAR_X_THRESHOLD = std::stoi(ini.get("LANDOUT").get("NEAR_X_THRESHOLD"));
			LDO_NEAR_Y_THRESHOLD = std::stoi(ini.get("LANDOUT").get("NEAR_Y_THRESHOLD"));

			LDO_FAR_X_THRESHOLD = std::stoi(ini.get("LANDOUT").get("FAR_X_THRESHOLD"));
			LDO_FAR_Y_THRESHOLD = std::stoi(ini.get("LANDOUT").get("FAR_Y_THRESHOLD"));

			LDO_GANTRY_THRESHOLD = std::stoi(ini.get("LANDOUT").get("GANTRY_THRESHOLD_PCA"));
			LDO_TROLLEY_THRESHOLD = std::stoi(ini.get("LANDOUT").get("TROLLEY_THRESHOLD_PCA"));

			LDO_NCOUNT = std::stoi(ini.get("LANDOUT").get("NCOUNT"));

			//CLPS-VA
			CLPS_NEAR_X_THRESHOLD = std::stoi(ini.get("CLPS").get("NEAR_X_THRESHOLD"));
			CLPS_NEAR_Y_THRESHOLD = std::stoi(ini.get("CLPS").get("NEAR_Y_THRESHOLD"));

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

void parseIPINI()
{
	mINI::INIFile file("ini/IP_LIST.ini");
	mINI::INIStructure ini;
	if (file.read(ini))
	{
		try
		{
			if (appID == "TMINI_LEFT")
			{
				for (int i = 0; i < 6; i++)
				{
					auto ip = ini.get("LEFT_IP").get("LANE" + std::to_string(i + 1));
					if (ip != "") {
						IP_ADDRESSES.push_back(ip);
					}

					ip = ini.get("LEFT_CAM_IP").get("LANE" + std::to_string(i + 1));
					if (ip != "") {
						CAM_IP_ADDRESSES.push_back(ip);
					}
				}
			}
			else if (appID == "TMINI_RIGHT")
			{
				for (int i = 0; i < 6; i++)
				{
					auto ip = ini.get("RIGHT_IP").get("LANE" + std::to_string(i + 1));
					if (ip != "") {
						IP_ADDRESSES.push_back(ip);
					}

					ip = ini.get("RIGHT_CAM_IP").get("LANE" + std::to_string(i + 1));
					if (ip != "") {
						CAM_IP_ADDRESSES.push_back(ip);
					}
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

void reset_recvVariables()
{
	TZ_Offload_Cycle = false;
	TZ_Mount_Cycle = false;

	Job_Pos_B = false;
	Job_Pos_S = false;
	Job_Pos_C = false;

	Target_20ft = false;
	Target_40ft = false;
	Target_45ft = false;

	SPRD_TWL_Locked = false;
	SPRD_TWL_Unlocked = false;
	SPRD_Landed = false;

	SPRD_20ft = false;
	SPRD_40ft = false;
	SPRD_45ft = false;

	Hoist_Stopped = false;
	Trolley_Stopped = false;
	Gantry_Stopped = false;

	Above_Safe_Height = false;

	Logging_Enable = false;
	Front_Left_Logging = false;
	Front_Right_Logging = false;
	Center_Left_Logging = false;
	Center_Right_Logging = false;
	Rear_Logging = false;

	Target_Height_Pos_mm = 0;
	Current_Hoist_Pos_mm = 0;
	Current_Trolley_Pos_mm = 0;
	Target_Trolley_Pos_mm = 0;

	intrim_Target_Height_Pos_mm = 0;
	intrim_Current_Hoist_Pos_mm = 0;
	intrim_Current_Trolley_Pos_mm = 0;
	intrim_Target_Trolley_Pos_mm = 0;

	enable_stream = false;
	enable_process = false;
	enable_logging = false;

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

auto makeCommand(char* sendbuf) -> void {
	try
	{
		//send heartbeat
			//increment heartbeat
		heartbeat_ += 1;
		if (heartbeat_ > 65535) heartbeat_ = 0;
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
		//

		sendbuf[2] = static_cast<char>(op_byte_2.to_ulong());

		
		//detected_cone_x
		{
			std::bitset<32>int_bitArr(tCone_x);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[3] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[4] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[5] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[6] = static_cast<char>(fourth_quarter.to_ulong());
		}
		//detected_cone_y
		{
			std::bitset<32>int_bitArr(tCone_y);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[7] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[8] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[9] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[10] = static_cast<char>(fourth_quarter.to_ulong());
		}
		//detected_cone_prob
		{
			std::bitset<32>int_bitArr(tCone_prob);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[11] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[12] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[13] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[14] = static_cast<char>(fourth_quarter.to_ulong());
		}

		//detected_cntr_x
		{
			std::bitset<32>int_bitArr(tCntr_x);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[15] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[16] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[17] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[18] = static_cast<char>(fourth_quarter.to_ulong());
		}
		//detected_cntr_y
		{
			std::bitset<32>int_bitArr(tCntr_y);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[19] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[20] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[21] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[22] = static_cast<char>(fourth_quarter.to_ulong());
		}
		//detected_cntr_prob
		{
			std::bitset<32>int_bitArr(tCntr_prob);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[23] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[24] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[25] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[26] = static_cast<char>(fourth_quarter.to_ulong());
		}

		std::bitset<8>op_byte_27(0);
		if (detected_xt) op_byte_27.set(0);
		if (detected_cst) op_byte_27.set(1);
		if (detected_chassis_type_unknown) op_byte_27.set(2);
		if (landout_detected) op_byte_27.set(3);
		if (clps_detected) op_byte_27.set(4);
		if (clps_ok_detected) op_byte_27.set(5);
		//if (landout_detected_pca) op_byte_27.set(6);

		sendbuf[27] = static_cast<char>(op_byte_27.to_ulong());

		//dev out x
		{
			std::bitset<32>int_bitArr(devOut_x);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[28] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[29] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[30] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[31] = static_cast<char>(fourth_quarter.to_ulong());
		}
		//dev out y
		{
			std::bitset<32>int_bitArr(devOut_y);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[32] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[33] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[34] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[35] = static_cast<char>(fourth_quarter.to_ulong());
		}

		//dev out pca x
		{
			std::bitset<32>int_bitArr(devOut_pca_x);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[36] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[37] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[38] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[39] = static_cast<char>(fourth_quarter.to_ulong());
		}
		//dev out pca y
		{
			std::bitset<32>int_bitArr(devOut_pca_y);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[40] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[41] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[42] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[43] = static_cast<char>(fourth_quarter.to_ulong());
		}
		//dev out pca z
		{
			std::bitset<32>int_bitArr(devOut_pca_z);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[44] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[45] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[46] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[47] = static_cast<char>(fourth_quarter.to_ulong());
		}
		//dev out results
		{
			std::bitset<8>op_byte_48(0);
			if (landout_detected_pca) op_byte_48.set(0);
			if (clps_detected_pca) op_byte_48.set(1);
			if (clps_ok_detected_pca) op_byte_48.set(2);
			sendbuf[48] = static_cast<char>(op_byte_48.to_ulong());
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
	if (TZ_Offload_Cycle) job_info.append("Offload_");
	else if (TZ_Mount_Cycle) job_info.append("Mount_");
	else job_info.append("None_");

	if (Job_Pos_B) job_info.append("Big_");
	else if (Job_Pos_S) job_info.append("Small_");
	else if (Job_Pos_C) job_info.append("Center_");
	else job_info.append("None_");

	if (CHS_XT) job_info.append("XT_");
	else if (CHS_CST) job_info.append("CST_");
	else job_info.append("None_");

	if (CHS_20ft) job_info.append("CHS20_");
	else if (CHS_40ft) job_info.append("CHS40_");
	else if (CHS_45ft) job_info.append("CHS45_");
	else job_info.append("None_");

	if (SPRD_20ft) job_info.append("SPRD20_");
	else if (SPRD_40ft) job_info.append("SPRD40_");
	else if (SPRD_45ft) job_info.append("SPRD45_");
	else job_info.append("None_");

	job_info.append(appName);

	//if (target_lane_number == LANE_NUM_AS_LEFT) job_info.append(SENSOR_POSITION + "_LEFT_");	
	//else if (target_lane_number == LANE_NUM_AS_RIGHT) job_info.append(SENSOR_POSITION + "_RIGHT_");
	//else job_info.append(SENSOR_POSITION + "_");
	
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
			TZ_Offload_Cycle = (bool)byte2[0];
			TZ_Mount_Cycle = (bool)byte2[1];

			Job_Pos_B = (bool)byte2[2];
			Job_Pos_S = (bool)byte2[3];
			Job_Pos_C = (bool)byte2[4];

			//Target_20ft = (bool)byte13[4];
			//Target_40ft = (bool)byte13[5];
			//Target_45ft = (bool)byte13[6];

			//std::bitset<8> byte14(recvBuf[14]);
			bool temp_twl_l = (bool)byte2[5];
			if (temp_twl_l != SPRD_TWL_Locked)
			{
				if (enabled_process)
				{
					save_trigger_by_TWL_Locked = true;
					jobLogMessage("Save Triggered by SPRD TWL Locked");
				}
			}
			SPRD_TWL_Locked = temp_twl_l;

			SPRD_TWL_Unlocked = (bool)byte2[6];

			bool temp = (bool)byte2[7];
			if (temp != SPRD_Landed)
			{
				if (enabled_process)
				{
					save_trigger_by_landed = true;
					jobLogMessage("Save Triggered by SPRD Landed");
				}
			}
			SPRD_Landed = temp;

			std::bitset<8> byte3(recvBuf[3]);
			SPRD_20ft = (bool)byte3[0];
			SPRD_40ft = (bool)byte3[1];
			SPRD_45ft = (bool)byte3[2];

			Hoist_Stopped = (bool)byte3[3];
			Trolley_Stopped = (bool)byte3[4];
			Gantry_Stopped = (bool)byte3[5];

			Above_Safe_Height = (bool)byte3[6];

			std::bitset<8> byte4(recvBuf[4]);
			CHS_20ft = (bool)byte4[0];
			CHS_40ft = (bool)byte4[1];
			CHS_45ft = (bool)byte4[2];

			CHS_XT = (bool)byte4[3];
			CHS_CST = (bool)byte4[4];
		}
		{
			
			std::bitset<8> byte6(recvBuf[6]);
			std::bitset<8> byte7(recvBuf[7]);
			std::bitset<16> temp = concat(byte6, byte7);
			Target_Height_Pos_mm = bitset_to_long(temp);

			std::bitset<8> byte8(recvBuf[8]);
			std::bitset<8> byte9(recvBuf[9]);
			std::bitset<16> temp2 = concat(byte8, byte9);
			Current_Hoist_Pos_mm = bitset_to_long(temp2);

			std::bitset<8> byte10(recvBuf[10]);
			std::bitset<8> byte11(recvBuf[11]);
			std::bitset<8> byte12(recvBuf[12]);
			std::bitset<8> byte13(recvBuf[13]);

			std::bitset<16> byte1011 = concat(byte10, byte11);
			std::bitset<16> byte1213 = concat(byte12, byte13);

			std::bitset<32> byte1234 = concat(byte1011, byte1213);
			Current_Trolley_Pos_mm = static_cast<signed long>(byte1234.to_ulong());

			std::bitset<8> byte14(recvBuf[14]);
			std::bitset<8> byte15(recvBuf[15]);
			std::bitset<8> byte16(recvBuf[16]);
			std::bitset<8> byte17(recvBuf[17]);

			std::bitset<16> byte1415 = concat(byte14, byte15);
			std::bitset<16> byte1617 = concat(byte16, byte17);

			std::bitset<32> byte4567 = concat(byte1415, byte1617);
			Target_Trolley_Pos_mm = static_cast<signed long>(byte4567.to_ulong());
			
		}

		{
			std::bitset<8> byte20(recvBuf[20]);
			std::bitset<8> byte21(recvBuf[21]);
			std::bitset<16> temp = concat(byte20, byte21);
			auto tempVal = bitset_to_long(temp);
			if (tempVal != target_lane_number)
			{
				logMessage("Target Lane Number Changed from " + std::to_string(target_lane_number) + " to " + std::to_string(tempVal));
			}
			target_lane_number = tempVal;
		}

		{
			std::bitset<8> byte18(recvBuf[18]);
			enable_stream = (bool)byte18[0];
			
			bool tempVal = (bool)byte18[1];
			if (tempVal && tempVal != enable_process)
			{
				//rising edge.
				logMessage("Enable Process Detected!");

				if (target_lane_number > 0 && target_lane_number < 7)
				{
					current_lane_ip = IP_ADDRESSES[target_lane_number - 1];
					logMessage("Current Lane IP: " + current_lane_ip);
					current_lane_cam_ip = CAM_IP_ADDRESSES[target_lane_number - 1];
					logMessage("Current Lane CAM IP: " + current_lane_cam_ip);
				}
				else
				{
					logMessage("Invalid Lane Number Detected! Setting it to default=1 and reporting app fault");
					current_lane_ip = IP_ADDRESSES[0];
					logMessage("Current Lane IP: " + current_lane_ip);

					current_lane_cam_ip = CAM_IP_ADDRESSES[0];
					logMessage("Current Lane CAM IP: " + current_lane_cam_ip);
				}

				if (!SEQUENTIAL_PROCESSING)
				{
					dataStack.Clear_Stack();
					if (ENABLE_CAM_RGB) dataStackRGB.Clear_Stack();

					//trigger tmini setup and stream.
					tmini_ctrl_flag = true;
					cond_tmini_ctrl.notify_one();				
				}
				
				//streaming thread will be triggered from tmini_ctrl thread.

				reset_jobVariables();

				//Create necessary job result folders and specify names.
				job_result_folder_name = std::string("JOB_LOG/");
				createDirectory_ifexists(job_result_folder_name);
				job_result_folder_name += std::string("Lane") + std::to_string(target_lane_number) + "/";
				createDirectory_ifexists(job_result_folder_name);


				job_name = makeJobFolderName();
				job_result_folder_name.append(job_name);
				job_result_folder_name.append("_JOBLOG");
				createDirectory_ifexists(job_result_folder_name);

				//images, pointcloud
				createDirectory_ifexists(job_result_folder_name + "/Image");
				createDirectory_ifexists(job_result_folder_name + "/Depth");
				createDirectory_ifexists(job_result_folder_name + "/Image_RGB");

				//job info log.
				job_result_log_file_name = job_result_folder_name + "/" + job_name + "_job_info.log";
				//allow job result log thread to resume?

				logMessage(job_result_log_file_name);
				jobLogMessage("Testing if Job Log is created on each job start.");
				
				if (CURRENT_SENSOR_POSITION.find("LEFT") != std::string::npos) JOB_IP_Pos = L_Pos;
				else JOB_IP_Pos = R_Pos;
				
				enabled_process = true;

				if (SEQUENTIAL_PROCESSING)
				{
					seq_proc_flag = true;
					//wake seq processing thread.
					cond_seq_processing.notify_one();
				}
				else
				{
					proc_running.store(true);
					//Wake the processing thread.
					flag = true;
					cond_processing.notify_one();
				}
				trigger_log_delete = false;

				logMessage("Process Enabled for Lane: " + std::to_string(target_lane_number) + " with job name: " + job_name);
			}
			else if (!tempVal && tempVal != enable_process)
			{
				logMessage("Disable Process Detected!");
				
				if (SEQUENTIAL_PROCESSING)
				{
					seq_proc_flag = false;
					//wake seq processing thread.
					cond_seq_processing.notify_one();
				}
				else
				{
					flag = false;

					tmini_ctrl_flag = false;
					//streaming off.
					tmini_data_stream_flag = false;

					enabled_process = false;

				}
				
				if (!enable_save_logs && ENABLE_SAVE_LOG_BIT)
				{
					logMessage("Delete Log Triggered.");
					trigger_log_delete = true;
					//need to delete "saved" logs.
				}

				//wake data save thread
				//saveFlag = true;
				//cond_logging.notify_one();
			}
			enable_process = tempVal;

			//sensor re-connection check.
			if (enable_process &&!SEQUENTIAL_PROCESSING)
			{
				if (sensor_fault && !sensor_connected) //comm fault.
				{
					auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - sensor_last_attempted_time).count();
					if (ts > 10000) //10 seconds
					{
						logMessage("Sensor Connection Fault Detected! Attempting to reconnect...");
						sensor_connected = false;
						sensor_fault = false;
						tmini_ctrl_flag = true; //wake tmini ctrl thread.
						cond_tmini_ctrl.notify_one();
					}
				}
			}

			if (ENABLE_SAVE_LOG_BIT && trigger_log_delete && !enable_logging)
			{
				logMessage("Deleting save logs as per command.");

				//stop any files on queue
				if (tsq.GetQueueLen() > 0)
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
				saveDirName += std::string("Lane") + std::to_string(target_lane_number) + "/";
				createDirectory_ifexists(saveDirName);

				auto job_info = makeJobFolderName();
				saveDirName.append(job_info);

				createDirectory_ifexists("SAVE");
				createDirectory_ifexists(saveDirName);
				createDirectory_ifexists(saveDirName + "/Image");
				createDirectory_ifexists(saveDirName + "/Depth");
				createDirectory_ifexists(saveDirName + "/Image_RGB");

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
		//if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".ply", *slice_);
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
		//if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".ply", *slice_);
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
		//if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".ply", *slice_);
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
		//if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".ply", *slice_);
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
	if (save_path != "") pcl::io::savePLYFileBinary(save_path + "_no_filter" + std::to_string(res_cloud->points.size()) + ".ply", *res_cloud);

	//logMessage("map size: " + std::to_string(res_cloud->points.size()));
	pcl::PointCloud<pcl::PointXYZ>::Ptr nf_out(new pcl::PointCloud <pcl::PointXYZ>);
	int max_points = 0;
	int nf_index = -1;
	//nf1: 30, 0.1
	pcl::PointCloud<pcl::PointXYZ>::Ptr nf_cloud1(new pcl::PointCloud <pcl::PointXYZ>);
	//nf_cloud1 = open3d_NoiseFilter(res_cloud, 30, 1.0);
	//if (save_path != "") pcl::io::savePLYFileBinary(save_path + "_nf30_1.0_pc_" + std::to_string(nf_cloud1->points.size()) + ".ply", *nf_cloud1);
	if (nf_cloud1->points.size() > 0)
	{
		max_points = nf_cloud1->points.size();
		pcl::copyPointCloud(*nf_cloud1, *nf_out);
		nf_index = 1;
	}
	else
	{
		pcl::copyPointCloud(*res_cloud, *nf_out);
		nf_index = 0;
	}

	//if (save_path != "") pcl::io::savePLYFileBinary(save_path + "_nf_out_i_" + std::to_string(nf_index) + ".ply", *nf_out);

	return nf_out;
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

	//auto dt_now = std::chrono::system_clock::now();
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud ( new pcl::PointCloud <pcl::PointXYZ>);

	pointCloud->points.resize(input.size());

	std::transform(input.begin(), input.end(), pointCloud->points.begin(),
		[](visionary::PointXYZ val) { return pcl::PointXYZ(val.x * 1000, val.y * 1000, val.z * 1000); });

	//auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - dt_now).count();
	//std::cout << "make dur ms: " << std::to_string(dur) << std::endl;

	//pcl::io::savePLYFileBinary("temp.ply", *pointCloud);
	return pointCloud;
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

	//pcl::io::savePLYFileBinary("temp.ply", *pointCloud);
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
			if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_mostPointsZ_" + std::to_string(mostPoints_z_slice->points.size()) + ".ply", *mostPoints_z_slice);

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
					if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_mpz_slice_y_" + std::to_string(y_index) + ".ply", *slice_y);

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
				if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_hole_center_points_count_" + std::to_string(hole_center_by_y.size()) + ".ply", *hole_center_points);

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

pcl::PointXYZ pc_hole_detection_naive(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool isLeftSide, std::string save_path)
{
	pcl::PointXYZ hole_position(10000, 10000, 10000);
	try
	{
		//X-Z directional limit for bigger ROI.
		int local_min_x, local_max_x;
		int local_min_z, local_max_Z;

		pcl::PointCloud<pcl::PointXYZ>::Ptr init_hole_filter_x(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_hole_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		local_min_z = get_valid_min_z(pcInput);
		if (isLeftSide)
		{
			//max x
			local_max_x = get_valid_max_x(pcInput);
			auto val = pc_passThrough(pcInput, local_max_x - 200, local_max_x + 100, "x", init_hole_filter_x);
			val = pc_passThrough(init_hole_filter_x, local_min_z - 100, local_min_z + 200, "z", init_hole_filtered);

		}
		else
		{
			//min x
			local_min_x = get_valid_min_x(pcInput);
			auto val = pc_passThrough(pcInput, local_min_x - 100, local_min_x + 200, "x", init_hole_filter_x);
			val = pc_passThrough(init_hole_filter_x, local_min_z - 100, local_min_z + 200, "z", init_hole_filtered);
		}
		if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_init_hole_filtered.ply", *init_hole_filtered);

		//Get Y slices :: 20 height slices.
		//Cornercastings height = usually 120mm.

		int y_slice_height = 20;

		int minY, maxY;
		get_max_min_y(init_hole_filtered, ref(maxY), ref(minY));

		std::vector<pcl::PointXYZ> target_points = {};

		int i = 0;
		for (int cy = minY; cy < maxY; cy += y_slice_height)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(init_hole_filtered, cy, cy + y_slice_height, "y", slice_);
			//if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_hole_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".ply", *slice_);
			if (val)
			{
				//Get most points MinX and most points MinZ Slices

				pcl::PointCloud<pcl::PointXYZ>::Ptr mpz_slice(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr mpx_slice(new pcl::PointCloud<pcl::PointXYZ>);

				int slice_width = 20;

				int minZ, maxZ, minX, maxX;
				get_max_min_x(slice_, ref(maxX), ref(minX));
				get_max_min_z(slice_, ref(maxZ), ref(minZ));

				for (int cx = minX; cx < maxX; cx += slice_width)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr sliceX_(new pcl::PointCloud<pcl::PointXYZ>);
					auto valX = pc_passThrough(slice_, cx, cx + slice_width, "x", sliceX_);
					if (sliceX_->points.size() > mpx_slice->points.size())
					{
						pcl::copyPointCloud(*sliceX_, *mpx_slice);
					}
				}

				for (int cz = minZ; cz < maxZ; cz += slice_width)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr sliceZ_(new pcl::PointCloud<pcl::PointXYZ>);
					auto valZ = pc_passThrough(slice_, cz, cz + slice_width, "z", sliceZ_);
					if (sliceZ_->points.size() > mpz_slice->points.size())
					{
						pcl::copyPointCloud(*sliceZ_, *mpz_slice);
					}
				}
				
				//if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_hole_i_" + std::to_string(i) + "_mpx_" + std::to_string(mpx_slice->points.size()) + ".ply", *mpx_slice);
				//if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_hole_i_" + std::to_string(i) + "_mpz_" + std::to_string(mpz_slice->points.size()) + ".ply", *mpz_slice);

				//from minX slice, get minZ.
				int mpx_minZ = -10000, mpx_maxZ = -10000;
				if (mpx_slice->points.size() > 0) get_max_min_z(mpx_slice, ref(mpx_maxZ), ref(mpx_minZ));

				//from minZ slice, get minX.
				int mpz_minX = -10000, mpz_maxX = -10000;
				if (mpz_slice->points.size() > 0) get_max_min_x(mpz_slice, ref(mpz_maxX), ref(mpz_minX));
				 
				//then get closest "actual" point to X,Z values.
				if (isLeftSide)
				{
					if (mpx_minZ != -10000 && mpz_maxX != -10000)
					{
						pcl::PointXYZ refPoint(mpz_maxX, 0, mpx_minZ);

						//Get Closest Point.
						auto targetPt = get_closest_xz(init_hole_filtered, refPoint);
						targetPt.y = cy;

						pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
						tPt->points.push_back(targetPt);
						//if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_hole_targetPt_i_" + std::to_string(i) + ".ply", *tPt);

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
						targetPt.y = cy;

						pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
						tPt->points.push_back(targetPt);
						//if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_hole_targetPt_i_" + std::to_string(i) + ".ply", *tPt);

						target_points.push_back(targetPt);
					}
				}

				
			}
			i++;
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
pcl::PointXYZ pc_cone_detection_naive(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool isLeftSide, std::string save_path)
{
	pcl::PointXYZ cone_position(10000, 10000, 10000);
	try
	{
		//X-Z directional limit for bigger ROI.
		int local_min_x, local_max_x;
		int local_min_z, local_max_Z;
		
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filter_x(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		local_min_z = get_valid_min_z(pcInput);
		if (isLeftSide)
		{
			//max x
			local_max_x = get_valid_max_x(pcInput);
			auto val = pc_passThrough(pcInput, local_max_x - 300, local_max_x + 100, "x", init_cone_filter_x);
			val = pc_passThrough(init_cone_filter_x, local_min_z - 100, local_min_z + 300, "z", init_cone_filtered);

		}
		else
		{
			//min x
			local_min_x = get_valid_min_x(pcInput);
			auto val = pc_passThrough(pcInput, local_min_x - 100, local_min_x + 300, "x", init_cone_filter_x);
			val = pc_passThrough(init_cone_filter_x, local_min_z - 100, local_min_z + 300, "z", init_cone_filtered);
		}
		if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_init_cone_filtered.ply", *init_cone_filtered);


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
			//if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".ply", *slice_);
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
			if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_extended_slice_y_pc_" + std::to_string(slice_->points.size()) + ".ply", *slice_);
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

			if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_mpx_" + std::to_string(mpx_slice->points.size()) + ".ply", *mpx_slice);
			if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_mpz_" + std::to_string(mpz_slice->points.size()) + ".ply", *mpz_slice);

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
			if (isLeftSide)
			{
				if (mpx_minZ != -10000 && mpz_maxX != -10000)
				{
					pcl::PointXYZ refPoint(mpz_maxX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);
					
					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_targetPt.ply", *tPt);

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
					if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_targetPt_mpx.ply", *tPt);

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
					if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_targetPt_mpz.ply", *tPt);

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
					if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_targetPt.ply", *tPt);

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
					if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_targetPt_mpx.ply", *tPt);

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
					if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_targetPt_mpz.ply", *tPt);

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

//PointCloud :: Landed Detection
pcl::PointXYZ pc_landed_detection_naive(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool isLeftSide, std::string save_path)
{
	pcl::PointXYZ cone_position(10000, 10000, 10000);
	try
	{
		int local_min_x, local_max_x;
		int local_min_z, local_max_Z;

		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filter_x(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		local_min_z = get_valid_min_z(pcInput);
		if (isLeftSide)
		{
			//max x
			local_max_x = get_valid_max_x(pcInput);
			auto val = pc_passThrough(pcInput, local_max_x - 300, local_max_x, "x", init_cone_filter_x);
			val = pc_passThrough(init_cone_filter_x, local_min_z, local_min_z + 300, "z", init_cone_filtered);

		}
		else
		{
			//min x
			local_min_x = get_valid_min_x(pcInput);
			auto val = pc_passThrough(pcInput, local_min_x, local_min_x + 300, "x", init_cone_filter_x);
			val = pc_passThrough(init_cone_filter_x, local_min_z, local_min_z + 300, "z", init_cone_filtered);
		}
		if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_init_landed_cone_filtered.ply", *init_cone_filtered);

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

		if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_landed_mpx_" + std::to_string(mpx_slice->points.size()) + ".ply", *mpx_slice);
		if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_landed_mpz_" + std::to_string(mpz_slice->points.size()) + ".ply", *mpz_slice);

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
		if (isLeftSide)
		{
			if (mpx_minZ != -10000 && mpz_maxX != -10000)
			{
				pcl::PointXYZ refPoint(mpz_maxX, 0, mpx_minZ);

				//Get Closest Point.
				auto targetPt = get_closest_xz(init_cone_filtered, refPoint);
				targetPt.y = maxY;

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(targetPt);
				if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_landed_targetPt.ply", *tPt);

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
				if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_landed_targetPt_mpx.ply", *tPt);

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
				if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_landed_targetPt_mpz.ply", *tPt);

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
				if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_landed_targetPt.ply", *tPt);

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
				if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_landed_targetPt_mpx.ply", *tPt);

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
				if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_landed_targetPt_mpz.ply", *tPt);

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

//PointCloud :: Guide Detection

cv::Mat drawOnImage(cv::Mat input, std::vector<rectangle_info> det_results)
{
	cv::Mat result_image = input.clone();

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

		cv::Scalar clr(0, 0, 255);
		if (box.label == 1) clr = cv::Scalar(0, 255, 0);
		else if (box.label == 2) clr = cv::Scalar(255, 0, 0);
		else if (box.label == 3) clr = cv::Scalar(128, 128, 0);

		cv::rectangle(result_image, cv::Rect(box.x, box.y, box.w, box.h), clr);

		//put-text
		std::string label = std::to_string(box.label) + "," + std::to_string(box.prob) + "%";
		cv::putText(result_image, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 2);
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

			std::string pos = "RL";
			if (CURRENT_SENSOR_POSITION == "REAR_RIGHT") pos = "RR";

			if (!oImage.empty()) cv::imwrite(SaveOImageDir + "/" + timeNow + "_" + msg + "_TMini_" + pos + "_oImage.jpg", oImage);
			if (!resImage.empty()) cv::imwrite(SaveResImageDir + "/" + timeNow + "_" + msg + "_TMini_" + pos + "_resImage.jpg", resImage);

			std::string plyFilePath = SaveDepthDir + "/" + timeNow + "_" + msg + "_TMini_" + pos + "_Depth.ply";
			if (pointCloud.points.size() > 0) pcl::io::savePLYFileBinary(plyFilePath.c_str(), pointCloud);
		}
		else
		{
			auto SaveImageDir = saveDirName + "/Image";
			auto SaveDepthDir = saveDirName + "/Depth";

			auto timeNow = time_as_name(frameTime);

			std::string pos = "RL";
			if (CURRENT_SENSOR_POSITION == "REAR_RIGHT") pos = "RR";

			if (!oImage.empty()) cv::imwrite(SaveImageDir + "/" + timeNow + "_TMini_" + pos + "_Image.jpg", oImage);
			//-----------------------------------------------
			// Write point cloud to PLY
			std::string plyFilePath = SaveDepthDir + "/" + timeNow + "_TMini_" + pos + "_Depth.ply";
			if (pointCloud.points.size() > 0) pcl::io::savePLYFileBinary(plyFilePath.c_str(), pointCloud);
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

void data_save_thread()
{
	logMessage("Data Logging Thread Activated");
	while (logging_running)
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
			while (enable_logging || tsq.GetQueueLen() > 0)// || tsq_rgb.GetQueueLen() > 0)
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
					/*
					if (tsq_rgb.GetQueueLen() > 0)
					{
						//in blocked state until item is placed in queue.
						auto dataTup = tsq_rgb.pop();

						cv::Mat oImage; cv::Mat resImage;
						bool isJobLog = false; std::string msg;

						oImage = std::get<0>(dataTup);
						resImage = std::get<1>(dataTup);
						isJobLog = std::get<2>(dataTup);
						msg = std::get<3>(dataTup);
						auto frameTime = std::get<4>(dataTup);

						if (isJobLog)
						{
							auto SaveOImageDir = job_result_folder_name + "/Image_RGB/original";
							createDirectory_ifexists(SaveOImageDir);
							auto SaveResImageDir = job_result_folder_name + "/Image_RGB/result";

							auto timeNow = time_as_name(frameTime);

							//msg: {TWL/TWUL}_{LANDED/LANDOFF}

							std::string pos = "RL";
							if (CURRENT_SENSOR_POSITION == "REAR_RIGHT") pos = "RR";

							if (!oImage.empty()) cv::imwrite(SaveOImageDir + "/" + timeNow + "_" + msg + "_TMini_" + pos + "_oImage.jpg", oImage);
							if (!resImage.empty()) cv::imwrite(SaveResImageDir + "/" + timeNow + "_" + msg + "_TMini_" + pos + "_resImage.jpg", resImage);
						}
						else
						{
							auto SaveImageDir = saveDirName + "/Image_RGB";
							//auto SaveDepthDir = saveDirName + "/Depth";

							auto timeNow = time_as_name(frameTime);

							std::string pos = "RL";
							if (CURRENT_SENSOR_POSITION == "REAR_RIGHT") pos = "RR";

							if (!oImage.empty()) cv::imwrite(SaveImageDir + "/" + timeNow + "_TMini_" + pos + "_RGB_Image.jpg", oImage);
						}
						//do not hug the process.
						std::this_thread::sleep_for(std::chrono::milliseconds(25));
					}
					*/
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
}

auto tmini_setup() -> bool
{
	bool status = false;
	try
	{
		using namespace visionary;

		//Get IP from INI
		std::string deviceIpAddr = current_lane_ip;
		unsigned short deviceBlobCtrlPort = 2114;
		unsigned cnt = 100u;

		controlConnected = false;
		dataConnected = false;

		logMessage("Connecting to sensor : " + current_lane_ip);

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

			if (res)
			{
				sensor_last_attempted_time = std::chrono::system_clock::now();
				auto status = tmini_setup();
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
		}
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
		if (ENABLE_CAM_RGB)
		{
			if (!camRGB_cap.isOpened())
			{
				logMessage("Camera RGB Capture Failed!");
				cam_rgb_connected = false;
			}
			else
			{
				logMessage("Camera RGB Capture Opened!");
				cam_rgb_connected = true;
			}
		}
		else cam_rgb_connected = false;
		cam_rgb_fault = false;

		if (tmini_data_stream_flag) logMessage("Data Streaming Ready & Starting!");
		while (tmini_data_stream_flag)
		{
			//no wait.	
			bool isFrameGrabbed = false;
			if (ENABLE_CAM_RGB)
			{
				isFrameGrabbed = camRGB_cap.grab();
				if (!isFrameGrabbed) {
					//std::cout << "Error: Cannot grab frame from RTSP stream." << std::endl;
					cam_rgb_fault = true;
					//break;
				}
			}
			else cam_rgb_fault = true;

			auto waitDur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - last_frame_get_time).count();
			if (waitDur_ms > 200)
			{
				blnGetNewFrame = false;
				bool stepComplete = visionaryControl.stepAcquisition();
				if (stepComplete)
				{
					if (dataStream.getNextFrame())
					{
						blnGetNewFrame = true;
						last_frame_get_time = std::chrono::system_clock::now();
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

							auto pclPointCloud = makePCL_PointCloud(pointCloud);
							if (pclPointCloud == nullptr)
							{
								logMessage("[TMini-Data-Stream] Failed to create PCL Point Cloud from data handler. -- nullptr!");
							}

							cv::Mat camRGB_img;
							if (cam_rgb_connected && !cam_rgb_fault) camRGB_cap.retrieve(camRGB_img);

							if (DEBUG_MODE)
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

							if (pclPointCloud != nullptr)
							{
								logMessage("Data stack updating stack with new data...");
								auto res = dataStack.Update_Stack(im3, *pclPointCloud, last_frame_get_time);
								if (res) logMessage("Data Stack Updated Successfully!");
								else logMessage("Data Stack Update Failed!");

								//if (cam_rgb_connected && !cam_rgb_fault) res = dataStackRGB.Update_Stack(camRGB_img, last_frame_get_time);
								//printf("Data Stack Update Status : %d\n", res);

								if (false)//if (enable_logging)
								{
									auto dataTup = std::tuple<cv::Mat, cv::Mat, pcl::PointCloud<pcl::PointXYZ>, bool, std::string, std::chrono::system_clock::time_point>(im3, cv::Mat(), *pclPointCloud, false, "", last_frame_get_time);
									tsq.push(dataTup);

									if (cam_rgb_connected && !cam_rgb_fault)
									{
										auto dataTup_rgb = std::tuple<cv::Mat, cv::Mat, bool, std::string, std::chrono::system_clock::time_point>(camRGB_img, cv::Mat(), false, "", last_frame_get_time);
										tsq_rgb.push(dataTup_rgb);
									}
								}

								if (enable_stream)
								{
									cv::imshow(CURRENT_SENSOR_POSITION + " T Mini", im3);
									if (ENABLE_CAM_RGB) cv::imshow(CURRENT_SENSOR_POSITION + " Sec110", camRGB_img);

									cv::waitKey(1);
								}
								else
								{
									cv::destroyAllWindows();
								}
							}
							else
							{
								logMessage("[TMini-Data-Stream] PCL Point Cloud is nullptr!");
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
		if (ENABLE_CAM_RGB) camRGB_cap.release();

		logMessage("[TMini-Data-Stream] Data Stream Closed!");
	}
	catch (std::exception& ex)
	{
		logMessage("[TMini-Data-Stream] " + std::string(ex.what()));
		cv::destroyAllWindows();
		visionaryControl.close();
		dataStream.close();
		if (ENABLE_CAM_RGB) camRGB_cap.release();
	}
	catch (...)
	{
		logMessage("[TMini-Data-Stream] Unknown Exception!");
		cv::destroyAllWindows();
		visionaryControl.close();
		dataStream.close();
		if (ENABLE_CAM_RGB) camRGB_cap.release();
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

				if (ENABLE_CAM_RGB)
				{
					//setup rtsp stream.
					//open rtsp stream and start capturing. 
					//rtsp://192.168.15.230:554/live/0
					//for in-house testing.
					//"rtsp://admin:seoho098@192.168.0.163:554";
					//std::string rtsp_url = "rtsp://admin:seoho098@" + current_lane_cam_ip + ":554";
					std::string rtsp_url = "rtsp://" + current_lane_cam_ip + ":554/live/0";
					logMessage("RTSP URL: " + rtsp_url);
					bool opened = false;

					// Launch async task to open the stream
					auto future = std::async(std::launch::async, tryOpenStream, rtsp_url, std::ref(opened));

					// Wait for up to 5 seconds
					if (future.wait_for(std::chrono::seconds(10)) == std::future_status::ready) {
						camRGB_cap = future.get(); // Get the actual object

						if (opened) {
							std::cout << "Stream opened successfully.\n";
						}
						else {
							std::cerr << "Stream open failed.\n";
						}
					}
					else {
						std::cerr << "Timeout while trying to open the RTSP stream.\n";
					}
				}
				else
				{
					logMessage("SEC110 Stream is disabled.");
				}
				tmini_data_stream();
		
				//end sensor connection.
				//visionaryControl.close();
				//dataStream.close();

				//deflag sensor related bits.
				sensor_connected = false;
				sensor_fault = false;

				//dataStack.Clear_Stack();
				//if (ENABLE_CAM_RGB) dataStackRGB.Clear_Stack();

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

//modular functions
//Inference.
bool onnx_inference(VA& detector, std::string SENSOR_POSITION, cv::Mat image, std::vector<rectangle_info>& det_results, std::vector<std::vector<bbx>>& det_sorted_objects, int& det_count, std::string save_path = "", bool jobLog = false)
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
			cv::Mat res_img = drawOnImage(image, det_results);
			cv::imwrite(save_path + "_result.jpg", res_img);
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

bool chassis_type_selection_VA(std::vector<std::vector<bbx>>& det_sorted_objects, bool jobLog = false)
{
	try
	{
		if ((det_sorted_objects[1].size() > 0 || det_sorted_objects[2].size() > 0) && det_sorted_objects[3].size() == 0)
		{
			logMessage("Detected XT");
			if (jobLog) jobLogMessage("Detected XT");
			detected_xt = true;
			detected_cst = false;
			detected_chassis_type_unknown = false;
		}
		else if ((det_sorted_objects[1].size() == 0 && det_sorted_objects[2].size() == 0) && det_sorted_objects[3].size() > 0)
		{
			logMessage("Detected CST");
			if (jobLog) jobLogMessage("Detected CST");
			detected_cst = true;
			detected_xt = false;
			detected_chassis_type_unknown = false;
		}
		else if ((det_sorted_objects[1].size() > 0 || det_sorted_objects[2].size() > 0) && det_sorted_objects[3].size() > 0)
		{
			if (det_sorted_objects[1].size() > 0)
			{
				if (det_sorted_objects[1][0].prob >= det_sorted_objects[3][0].prob)
				{
					logMessage("Detected XT by prob");
					if (jobLog) jobLogMessage("Detected XT by prob");
					detected_xt = true;
					detected_cst = false;
					detected_chassis_type_unknown = false;
				}
				else
				{
					logMessage("Detected CST by prob");
					if (jobLog) jobLogMessage("Detected CST by prob");
					detected_xt = true;
					detected_cst = false;
					detected_chassis_type_unknown = false;
				}
			}
			else
			{
				if (det_sorted_objects[2][0].prob >= det_sorted_objects[3][0].prob)
				{
					logMessage("Detected XT by prob");
					if (jobLog) jobLogMessage("Detected XT by prob");
					detected_xt = true;
					detected_cst = false;
					detected_chassis_type_unknown = false;
				}
				else
				{
					logMessage("Detected CST by prob");
					if (jobLog) jobLogMessage("Detected CST by prob");
					detected_xt = true;
					detected_cst = false;
					detected_chassis_type_unknown = false;
				}
			}
		}
		else
		{
			logMessage("No valid detection to check chassis type");
			if (jobLog) jobLogMessage("No valid detection to check chassis type");
			detected_xt = false;
			detected_cst = false;
			detected_chassis_type_unknown = true;
		}

		return true;
	}
	catch (std::exception& ex)
	{
		logMessage("[Chassis-Type-Selection-VA] " + std::string(ex.what()));
		if (jobLog) jobLogMessage("[Chassis-Type-Selection-VA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Chassis-Type-Selection-VA] Unknown Exception!");
		if (jobLog) jobLogMessage("[Chassis-Type-Selection-VA] Unknown Exception!");
		return false;
	}
}
bool Target_Selections_VA(std::string SENSOR_POSITION, int sprd_size, std::vector<std::vector<bbx>> det_sorted_objects, bbx& target_hole, bbx& target_cone, bbx& target_guide, bool& detected_hole, bool jobLog = false)
{
	try
	{
		//Target cornercastings selection -- Get Highest Prob.
		if (det_sorted_objects[0].size() > 0)
		{
			if (sprd_size == 45 && det_sorted_objects[0].size() > 1)
			{
				std::vector<bbx> detected_holes = det_sorted_objects[0];
				//ascending order.
				std::sort(detected_holes.begin(), detected_holes.end(), [](const bbx& a, const bbx& b) {
					return a.center_x < b.center_x;  // Comparison based on the 'center_x' field
					});

				if (SENSOR_POSITION == "REAR_LEFT")
				{
					//take most left.
					target_hole = detected_holes[0];
				}
				else
				{
					//take most right.
					target_hole = detected_holes[detected_holes.size() - 1];
				}
			}
			else
			{
				//take highest prob. -- this happens to be the case if 45ft job but single hole is detected.
				target_hole = det_sorted_objects[0][0];
			}
			detected_hole = true;
		}

		// Combine vectors using std::copy
		std::vector<bbx> detected_cones;
		detected_cones.reserve(det_sorted_objects[1].size() + det_sorted_objects[2].size());  // Reserve space to avoid multiple allocations
		detected_cones.insert(detected_cones.end(), det_sorted_objects[1].begin(), det_sorted_objects[1].end());  // Insert first vector
		std::copy(det_sorted_objects[2].begin(), det_sorted_objects[2].end(), std::back_inserter(detected_cones));  // Copy second vector

		if (detected_cones.size() > 0)
		{
			if (detected_cones.size() > 1)
			{
				// Sort the vector by center_x (ascending order)
				std::sort(detected_cones.begin(), detected_cones.end(), [](const bbx& a, const bbx& b) {
					return a.center_x < b.center_x;  // Comparison based on the 'center_x' field
					});
			}

			if (SENSOR_POSITION == "REAR_LEFT")
			{
				//take most right
				target_cone = detected_cones[detected_cones.size() - 1];

			}
			else if (SENSOR_POSITION == "REAR_RIGHT")
			{
				//take most left
				target_cone = detected_cones[0];
			}
		}

		//Target Guide -- use highest prob for now.
		if (det_sorted_objects[3].size() > 0) target_guide = det_sorted_objects[3][0];

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

bool Pre_Land_chassis_position_VA(bbx target_cone, bbx target_hole, bool jobLog = false)
{
	try
	{
		if (target_cone.prob > 0)
		{
			if (TZ_Mount_Cycle)
			{
				if (!LDO_Base_Set)
				{
					if (target_cone.label == 1 && target_cone.prob >= 85)
					{
						LDO_intrim.x += target_cone.x;
						LDO_intrim.y += target_cone.y;
						LDO_intrim.w += target_cone.w;
						LDO_intrim.h += target_cone.h;
						LDO_intrim.center_x += target_cone.center_x;
						LDO_intrim.center_y += target_cone.center_y;
						LDO_intrim.prob += target_cone.prob;

						LDO_Base_Count++;
						if (LDO_Base_Count >= LDO_NCOUNT)
						{
							LDO_Base.x = (int)((double)LDO_intrim.x / (double)LDO_Base_Count);
							LDO_Base.y = (int)((double)LDO_intrim.y / (double)LDO_Base_Count);
							LDO_Base.w = (int)((double)LDO_intrim.w / (double)LDO_Base_Count);
							LDO_Base.h = (int)((double)LDO_intrim.h / (double)LDO_Base_Count);
							LDO_Base.center_x = (int)((double)LDO_intrim.center_x / (double)LDO_Base_Count);
							LDO_Base.center_y = (int)((double)LDO_intrim.center_y / (double)LDO_Base_Count);
							LDO_Base.prob = (int)((double)LDO_intrim.prob / (double)LDO_Base_Count);

							LDO_Base_Set = true;
							if (jobLog) jobLogMessage("LDO Base Set (XT): (CX:" + std::to_string(LDO_Base.center_x) + " , CY:" + std::to_string(LDO_Base.center_y) + ")");
						}
					}
				}
			}
			else if (TZ_Offload_Cycle)
			{
				if (!CLPS_Base_Set)
				{
					if (target_cone.prob >= 75)
					{
						CLPS_intrim.x += target_cone.x;
						CLPS_intrim.y += target_cone.y;
						CLPS_intrim.w += target_cone.w;
						CLPS_intrim.h += target_cone.h;
						CLPS_intrim.center_x += target_cone.center_x;
						CLPS_intrim.center_y += target_cone.center_y;
						CLPS_intrim.prob += target_cone.prob;

						CLPS_Base_Count++;
						if (CLPS_Base_Count >= CLPS_NCOUNT)
						{
							CLPS_Base.x = (int)((double)CLPS_intrim.x / (double)CLPS_Base_Count);
							CLPS_Base.y = (int)((double)CLPS_intrim.y / (double)CLPS_Base_Count);
							CLPS_Base.w = (int)((double)CLPS_intrim.w / (double)CLPS_Base_Count);
							CLPS_Base.h = (int)((double)CLPS_intrim.h / (double)CLPS_Base_Count);
							CLPS_Base.center_x = (int)((double)CLPS_intrim.center_x / (double)CLPS_Base_Count);
							CLPS_Base.center_y = (int)((double)CLPS_intrim.center_y / (double)CLPS_Base_Count);
							CLPS_Base.prob = (int)((double)CLPS_intrim.prob / (double)CLPS_Base_Count);

							CLPS_Base_Set = true;
							if (jobLog) jobLogMessage("CLPS Base Set (XT): (CX:" + std::to_string(CLPS_Base.center_x) + " , CY:" + std::to_string(CLPS_Base.center_y) + ")");
						}
					}
				}		
			}
		}

		if (target_hole.prob > 0)
		{
			if (TZ_Offload_Cycle)
			{
				if (!Offload_Hole_Base_Set)
				{
					if (target_hole.prob > 80)
					{
						Offload_Hole_intrim.x += target_hole.x;
						Offload_Hole_intrim.y += target_hole.y;
						Offload_Hole_intrim.w += target_hole.w;
						Offload_Hole_intrim.h += target_hole.h;
						Offload_Hole_intrim.center_x += target_hole.center_x;
						Offload_Hole_intrim.center_y += target_hole.center_y;
						Offload_Hole_intrim.prob += target_hole.prob;

						Offload_Hole_Base_Count++;
						if (Offload_Hole_Base_Count >= CLPS_NCOUNT)
						{
							Offload_Hole_Base.x = (int)((double)Offload_Hole_intrim.x / (double)Offload_Hole_Base_Count);
							Offload_Hole_Base.y = (int)((double)Offload_Hole_intrim.y / (double)Offload_Hole_Base_Count);
							Offload_Hole_Base.w = (int)((double)Offload_Hole_intrim.w / (double)Offload_Hole_Base_Count);
							Offload_Hole_Base.h = (int)((double)Offload_Hole_intrim.h / (double)Offload_Hole_Base_Count);
							Offload_Hole_Base.center_x = (int)((double)Offload_Hole_intrim.center_x / (double)Offload_Hole_Base_Count);
							Offload_Hole_Base.center_y = (int)((double)Offload_Hole_intrim.center_y / (double)Offload_Hole_Base_Count);
							Offload_Hole_Base.prob = (int)((double)Offload_Hole_intrim.prob / (double)Offload_Hole_Base_Count);

							Offload_Hole_Base_Set = true;
							if (jobLog) jobLogMessage("Offload Hole Base Set (Hole): (CX:" + std::to_string(Offload_Hole_Base.center_x) + " , CY:" + std::to_string(Offload_Hole_Base.center_y) + ")");
						}
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

bool Deviation_Output_VA(bbx target_hole, bbx target_cone, bbx target_guide, IDEAL_POS IP, int& tCntr_x, int& tCntr_y, int& tCntr_prob, int& tCone_x, int& tCone_y, int& tCone_prob, int& devOut_x, int& devOut_y, int& devOut_x_mm, int& devOut_y_mm, bool& usingLDO_Base, bool jobLog = false)
{
	try
	{
		//If target hole is present, deviation between target cone and hole.
		if (target_hole.prob > 0)
		{
			tCntr_x = target_hole.center_x;
			tCntr_y = target_hole.y + target_hole.h;
			tCntr_prob = target_hole.prob;
		}

		if (detected_xt)
		{
			if (target_cone.prob > 0)
			{
				tCone_x = target_cone.center_x;
				tCone_y = target_cone.center_y;
				tCone_prob = target_cone.prob;

				//if hole present,
				if (target_hole.prob > 0)
				{
					devOut_x = tCone_x - tCntr_x;
					devOut_y = tCntr_y - tCone_y;
				}
				//if hole absent,
				else
				{
					//2025.03.11 Modified.
					devOut_x = tCone_x - IP.CONE.center_x;
					devOut_y = tCone_y - IP.CONE.center_y;
				}
			}
		}
		else if (detected_cst)
		{
			if (target_guide.prob > 0)
			{
				tCone_x = target_guide.center_x;
				tCone_y = target_guide.center_y;
				tCone_prob = target_guide.prob;

				//if hole present,
				if (target_hole.prob > 0)
				{
					devOut_x = tCone_x - tCntr_x;
					devOut_y = tCntr_y - tCone_y;
				}
				//if hole absent,
				else
				{
					//2025.03.11 Modified.
					devOut_x = tCone_x - IP.GUIDE.center_x;
					devOut_y = tCone_y - IP.GUIDE.center_y;
				}
			}
		}
		else if (detected_chassis_type_unknown)
		{
			if (true)//(CHS_XT)
			{
				//use pre land average
				if (TZ_Mount_Cycle && LDO_Base_Set)
				{
					target_cone = LDO_Base;
					usingLDO_Base = true;
					if (target_cone.prob > 0)
					{
						tCone_x = target_cone.center_x;
						tCone_y = target_cone.center_y;
						tCone_prob = target_cone.prob;

						//if hole present,
						if (target_hole.prob > 0)
						{
							devOut_x = tCone_x - tCntr_x;
							devOut_y = tCntr_y - tCone_y;
						}
						//if hole absent,
						else
						{
							//2025.03.11 Modified.
							devOut_x = tCone_x - IP.CONE.center_x;
							devOut_y = tCone_y - IP.CONE.center_y;
						}
					}
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
		logMessage("VA Dev Out x: " + std::to_string(devOut_x) + " y: " + std::to_string(devOut_y));
		if (jobLog) jobLogMessage("VA Dev Out x: " + std::to_string(devOut_x) + " y: " + std::to_string(devOut_y));
		logMessage("VA Dev Out x mm: " + std::to_string(devOut_x_mm) + " y mm: " + std::to_string(devOut_y_mm));
		if (jobLog) jobLogMessage("VA Dev Out x mm: " + std::to_string(devOut_x_mm) + " y mm: " + std::to_string(devOut_y_mm));

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

bool Debug_Landed_Trigger(bbx target_hole, bbx target_cone)
{
	try
	{
		bool debug_landed_trigger = false;
		if (target_cone.prob > 0)
		{
			if (target_hole.prob > 0)
			{
				if (target_hole.y + target_hole.h > target_cone.y - 5)
				{
					debug_landed_trigger = true;
				}
				else if (target_hole.y + target_hole.h < target_cone.y)
				{
					debug_landed_trigger = false;
				}
			}
			else if (target_cone.label == 2) //landed detected
			{
				debug_landed_trigger = true;
			}
			else if (target_cone.label == 1)
			{
				debug_landed_trigger = false;
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

bool LandOut_Detected_VA(bbx target_hole, bbx target_cone, bool sprd_landed, bool detected_hole, bool usingLDO_Base, bool jobLog = false)
{
	try
	{
		if (TZ_Mount_Cycle)
		{
			//based on landed bit
			if (detected_xt)
			{
				if (sprd_landed)
				{
					//Case 1: if target cone is not landed.
					if (target_cone.label == 1)
					{
						if (detected_hole)
						{
							int devX = target_cone.center_x - target_hole.center_x;
							//(-): hole is below the cone's y.
							//(+): hole is above the cone's y.
							int devY = target_cone.y - (target_hole.center_y);

							//based on dev out and near
							if (std::abs(devX) > LDO_NEAR_X_THRESHOLD || std::abs(devY) > LDO_NEAR_Y_THRESHOLD)
							{
								if (LDO_Current_Count < LDO_NCOUNT) LDO_Current_Count++;

								if (LDO_Current_Count >= LDO_NCOUNT)
								{
									if (!landout_detected) 
									{
										logMessage("Landout detected! with CONE object! devX: " + std::to_string(devOut_x) + " devY:" + std::to_string(devOut_y));
										if (jobLog) jobLogMessage("Landout detected! with CONE object! devX: " + std::to_string(devOut_x) + " devY:" + std::to_string(devOut_y));
									}
									landout_detected = true;
								}
							}
							else if (devY > 0) //Landout.
							{
								if (LDO_Current_Count < LDO_NCOUNT) LDO_Current_Count++;

								if (LDO_Current_Count >= LDO_NCOUNT)
								{
									if (!landout_detected)
									{
										logMessage("Landout detected! with CONE object! devY:" + std::to_string(devY));
										if (jobLog) jobLogMessage("Landout detected! with CONE object! devY:" + std::to_string(devY));
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
						//Failed to detect hole, cone detected while SPRD_LANDED bit is on.
						else
						{
							//if detected probability is above 85%, then trust the result and increment LDO Count
							if (target_cone.prob > 85 && !usingLDO_Base)
							{
								if (LDO_Current_Count < LDO_NCOUNT) LDO_Current_Count++;

								if (LDO_Current_Count >= LDO_NCOUNT)
								{
									if (!landout_detected) 
									{
										logMessage("Landout detected! with CONE 85%");
										if (jobLog) jobLogMessage("Landout detected! with CONE 85%");
									}
									landout_detected = true;
								}
							}
						}
					}
					//Case 2: if target cone is landed
					else if (target_cone.label == 2 && !usingLDO_Base)
					{
						if (detected_hole)
						{
							//based on dev out and near
							if (std::abs(devOut_x) > LDO_NEAR_X_THRESHOLD || target_cone.h > 3 * target_hole.h)
							{
								if (LDO_Current_Count < LDO_NCOUNT) LDO_Current_Count++;

								if (LDO_Current_Count >= LDO_NCOUNT)
								{
									if (!landout_detected) 
									{
										logMessage("Landout detected with LANDED object!");
										if (jobLog) jobLogMessage("Landout detected with LANDED object!");
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
			else if (detected_cst)
			{
				;
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

bool CLPS_Detection_VA(bbx target_hole, bbx target_cone, bool sprd_landed, bool jobLog = false)
{
	try
	{
		if (TZ_Offload_Cycle)
		{
			if (CLPS_Base_Set)
			{
				if (target_cone.prob > 60)
				{
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
										logMessage("CLPS Detected by cone! diff_y: " + std::to_string(diff_y));
										if (jobLog) jobLogMessage("CLPS Detected by cone! diff_y: " + std::to_string(diff_y));
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
				}
				else
				{
					//if inference can be trusted, then non detection can be an indicator of CLPS.
				}
			}

			//2025.03.11
			//CLPS-OK Logic.
			//clps_ok_detected = false;
			if (true)//!sprd_landed && !clps_detected)
			{
				//if valid cornercastings and cone (label==1) detected, then clps-ok.
				if (target_hole.prob > 70 && (target_cone.prob > 70 && target_cone.label == 1))
				{
					auto diff_y = target_cone.center_y - target_hole.center_y;
					if (diff_y > 20)
					{
						clps_ok_detected = true;
						if (jobLog) jobLogMessage("VA CLPS-OK Detected by Cone and Hole, diff_y: " + std::to_string(diff_y));
					}
				}
				else if (clps_ok_detected)
				{
					if (target_hole.prob > 70 && (target_cone.prob > 70 && target_cone.label == 1))
					{
						auto diff_y = target_cone.center_y - target_hole.center_y;
						if (diff_y < 20)
						{
							clps_ok_detected = false;
							//if (jobLog) jobLogMessage("VA CLPS-OK Detected by Cone and Hole, diff_y: " + std::to_string(diff_y));
						}
					}
					else
					{
						if (target_cone.prob > 70 && target_cone.label == 2)
						{
							clps_ok_detected = false;
						}
					}
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

bool Target_PointCloud_Extraction(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, bbx target_hole, bbx target_cone, bbx target_guide, bool sprd_landed, IDEAL_POS IP, int x_limit, int z_limit, pcl::PointCloud<pcl::PointXYZ>::Ptr& pcHole, pcl::PointCloud<pcl::PointXYZ>::Ptr& pcCone, pcl::PointCloud<pcl::PointXYZ>::Ptr& pcGuide, std::string save_path = "", bool jobLog = false)
{
	try
	{
		if (target_hole.prob > 0)
		{
			std::string bPath = (save_path != std::string("")) ? save_path + "_Hole" : "";
			//logMessage(std::to_string(target_hole.x) + " " + std::to_string(target_hole.y) + " , " + std::to_string(x_limit) + " , " + std::to_string(z_limit));
			pcHole = map_box_to_pc_noisefiltertest(pointCloud, target_hole, x_limit, z_limit, bPath);
			if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_pcHole.ply", *pcHole);
		}
		if (target_cone.prob > 0)
		{
			if (target_cone.label == 1)
			{
				std::string bPath = (save_path != std::string("")) ? save_path + "_Cone" : "";
				pcCone = map_box_to_pc_noisefiltertest(pointCloud, target_cone, x_limit, z_limit, bPath);
				if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_pcCone.ply", *pcCone);
			}
			else
			{
				//modify detected landed area.
				if (target_hole.prob > 0)
				{
					std::string bPath = (save_path != std::string("")) ? save_path + "_PreLanded" : "";
					auto pcPreCone = map_box_to_pc_noisefiltertest(pointCloud, target_cone, x_limit, z_limit, bPath);
					if (save_path != std::string(""))pcl::io::savePLYFileBinary(save_path + "_pcPreLanded.ply", *pcCone);

					bbx modified_landed = target_cone;
					auto delta_y = target_hole.y + target_hole.h - target_cone.y;
					modified_landed.y = target_hole.y + target_hole.h;
					modified_landed.h -= delta_y;

					std::string bPath2 = (save_path != std::string("")) ? save_path + "_Landed" : "";
					pcCone = map_box_to_pc_noisefiltertest(pointCloud, modified_landed, x_limit, z_limit, bPath2);
					if (save_path != std::string(""))pcl::io::savePLYFileBinary(save_path + "_pcLanded.ply", *pcCone);
				}
				else
				{
					std::string bPath = (save_path != std::string("")) ? save_path + "_PreLanded" : "";
					auto pcPreCone = map_box_to_pc_noisefiltertest(pointCloud, target_cone, x_limit, z_limit, bPath);
					if (save_path != std::string(""))pcl::io::savePLYFileBinary(save_path + "_pcPreLanded.ply", *pcCone);

					//preset y down?
					int preset_y_hole = 20;
					bbx modified_landed = target_cone;
					modified_landed.y = target_cone.y + preset_y_hole;
					modified_landed.h = target_cone.h - preset_y_hole;
					std::string bPath2 = (save_path != std::string("")) ? save_path + "_Landed" : "";
					pcCone = map_box_to_pc_noisefiltertest(pointCloud, modified_landed, x_limit, z_limit, bPath2);
					if (save_path != std::string(""))pcl::io::savePLYFileBinary(save_path + "_pcLanded_preset.ply", *pcCone);
				}
			}
		}
		else
		{
			//extract based on preset.
			//preset
			if (!sprd_landed)
			{
				bbx preset_cone = IP.CONE;
				std::string bPath = (save_path != std::string("")) ? save_path + "_PreSet_Cone" : "";
				pcCone = map_box_to_pc_noisefiltertest(pointCloud, preset_cone, x_limit, z_limit, bPath);
				if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_pcPresetCone.ply", *pcCone);
			}
			else //assume landed.
			{
				bbx preset_cone = IP.LANDED;
				//modify detected landed area.
				if (target_hole.prob > 0)
				{
					std::string bPath = (save_path != std::string("")) ? save_path + "_PreSetLanded" : "";
					auto pcPreCone = map_box_to_pc_noisefiltertest(pointCloud, preset_cone, x_limit, z_limit, bPath);
					if (save_path != std::string(""))pcl::io::savePLYFileBinary(save_path + "_pcPreSetLanded.ply", *pcCone);

					bbx modified_landed = preset_cone;
					auto delta_y = target_hole.y + target_hole.h - preset_cone.y;
					modified_landed.y = target_hole.y + target_hole.h;
					modified_landed.h -= delta_y;

					std::string bPath2 = (save_path != std::string("")) ? save_path + "_Landed" : "";
					pcCone = map_box_to_pc_noisefiltertest(pointCloud, modified_landed, x_limit, z_limit, bPath2);
					if (save_path != std::string(""))pcl::io::savePLYFileBinary(save_path + "_pcLanded.ply", *pcCone);
				}
				else
				{
					std::string bPath = (save_path != std::string("")) ? save_path + "_PreLanded" : "";
					auto pcPreCone = map_box_to_pc_noisefiltertest(pointCloud, preset_cone, x_limit, z_limit, bPath);
					if (save_path != std::string(""))pcl::io::savePLYFileBinary(save_path + "_pcPreLanded.ply", *pcCone);

					//preset y down?
					int preset_y_hole = 20;
					bbx modified_landed = preset_cone;
					modified_landed.y = preset_cone.y + preset_y_hole;
					modified_landed.h = preset_cone.h - preset_y_hole;
					std::string bPath2 = (save_path != std::string("")) ? save_path + "_Landed" : "";
					pcCone = map_box_to_pc_noisefiltertest(pointCloud, modified_landed, x_limit, z_limit, bPath2);
					if (save_path != std::string(""))pcl::io::savePLYFileBinary(save_path + "_pcLanded_preset.ply", *pcCone);
				}
			}
		}

		if (target_guide.prob > 0)
		{
			pcGuide = map_box_to_pc(pointCloud, target_guide);
			if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_pcGuide.ply", *pcGuide);
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

bool CLPS_Detection_PCA(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, bbx target_hole, int g_data_limit, int t_data_limit, bool detected_hole, bool sprd_landed, int& pcUnderHole, std::string save_path = "", bool jobLog = false)
{
	try
	{
		if (TZ_Offload_Cycle)
		{
			//Hole detected, but not chassis.
			if (detected_hole)
			{
				//if hole lifted above certain level.
				int y_level = 414 - (414 / 3);

				if (Offload_Hole_Base_Set)
				{
					//new reference y_level.
					y_level = Offload_Hole_Base.y - Offload_Hole_Base.h * 1;
				}
				if (jobLog) jobLogMessage("PCA-CLPS Y Level Threshold: " + std::to_string(y_level));
				if (target_hole.y < y_level)
				{
					//extract region beneath the detected hole.
					bbx targetRegion;
					targetRegion.x = target_hole.x;
					targetRegion.y = target_hole.y + target_hole.h;
					targetRegion.w = target_hole.w;
					targetRegion.h = target_hole.h * 2;
					targetRegion.center_x = targetRegion.x + targetRegion.w / 2;
					targetRegion.center_y = targetRegion.y + targetRegion.h / 2;

					//then extract.
					pcl::PointCloud<pcl::PointXYZ>::Ptr pcHoleBeneath(new pcl::PointCloud<pcl::PointXYZ>);
					pcHoleBeneath = map_box_to_pc(pointCloud, targetRegion);
					//if (save_path != std::string(""))pcl::io::savePLYFileBinary(save_path + "_pcHoleBeneath.ply", *pcHoleBeneath);

					//Do a pre filter 
					pcl::PointCloud<pcl::PointXYZ>::Ptr pcFilterZ(new pcl::PointCloud <pcl::PointXYZ>);
					pc_passThrough(pcHoleBeneath, 0, g_data_limit, "z", pcFilterZ);
					pcl::PointCloud<pcl::PointXYZ>::Ptr pcTargetBase(new pcl::PointCloud <pcl::PointXYZ>);
					pc_passThrough(pcFilterZ, 0, t_data_limit, "x", pcTargetBase);
					if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_pcHoleBeneath_Target_" + std::to_string(pcTargetBase->points.size()) + ".ply", *pcTargetBase);

					pcUnderHole = pcTargetBase->points.size();

					if (pcTargetBase->points.size() > 300)
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

bool Position_Detection_PCA(std::string SENSOR_POSITION, bbx target_cone, pcl::PointCloud<pcl::PointXYZ>::Ptr pcHole, pcl::PointCloud<pcl::PointXYZ>::Ptr pcCone, pcl::PointXYZ& hole_pos, pcl::PointXYZ& cone_pos, std::string save_path = "", bool jobLog = false)
{
	try
	{
		if (pcHole->points.size() > 0)
		{
			bool isLeftSensor = (SENSOR_POSITION == "REAR_LEFT") ? true : false;

			hole_pos = pc_hole_detection_naive(pcHole, isLeftSensor, save_path);
			logMessage("Detected hole position: (T:" + std::to_string(hole_pos.x) + " , H:" + std::to_string(hole_pos.y) + " , G:" + std::to_string(hole_pos.z) + ")");
			if (jobLog) jobLogMessage("Detected hole position: (T:" + std::to_string(hole_pos.x) + " , H:" + std::to_string(hole_pos.y) + " , G:" + std::to_string(hole_pos.z) + ")");

			if (hole_pos.x != -10000)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr holeCloud(new pcl::PointCloud <pcl::PointXYZ>);
				holeCloud->points.push_back(hole_pos);
				if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_hole_pos.ply", *holeCloud);
			}
		}
		else
		{
			//;no preset.
		}

		if (pcCone->points.size() > 0)
		{
			bool isLeftSensor = (SENSOR_POSITION == "REAR_LEFT") ? true : false;

			if (target_cone.label == 1)
			{
				cone_pos = pc_cone_detection_naive(pcCone, isLeftSensor, save_path);
				logMessage("Detected cone position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");
				if (jobLog) jobLogMessage("Detected cone position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");

				if (cone_pos.x != -10000)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr coneCloud(new pcl::PointCloud <pcl::PointXYZ>);
					coneCloud->points.push_back(cone_pos);
					if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_pos.ply", *coneCloud);
				}
			}
			else
			{
				cone_pos = pc_landed_detection_naive(pcCone, isLeftSensor, save_path);
				logMessage("Detected landed position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");
				if (jobLog) jobLogMessage("Detected landed position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");

				if (cone_pos.x != -10000)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr coneCloud(new pcl::PointCloud <pcl::PointXYZ>);
					coneCloud->points.push_back(cone_pos);
					if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_landed_pos.ply", *coneCloud);
				}
			}
		}
		else
		{
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

bool Pre_Land_Chassis_Position_PCA(bbx target_cone, pcl::PointXYZ cone_pos, pcl::PointCloud<pcl::PointXYZ>::Ptr pcCone, bool jobLog = false)
{
	try
	{
		if (target_cone.prob > 0)
		{
			//initial chassis position.
			if (!PCA_Base_Set)
			{
				if (target_cone.prob >= 85 && pcCone->points.size() > 0)
				{
					if (cone_pos.x != -10000)
					{
						PCA_Base.x += cone_pos.x;
						PCA_Base.y += cone_pos.y;
						PCA_Base.z += cone_pos.z;
						PCA_Base_Count++;
						if (PCA_Base_Count >= LDO_NCOUNT)
						{
							PCA_Base.x = (int)((double)PCA_Base.x / (double)PCA_Base_Count);
							PCA_Base.y = (int)((double)PCA_Base.y / (double)PCA_Base_Count);
							PCA_Base.z = (int)((double)PCA_Base.z / (double)PCA_Base_Count);
							PCA_Base_Set = true;
							if (jobLog) jobLogMessage("PCA Base Set to: (T: " + std::to_string(PCA_Base.x) + " , H:" + std::to_string(PCA_Base.y) + " , G:" + std::to_string(PCA_Base.z) + ")");
						}
					}
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

bool Deviation_Output_PCA(std::string SENSOR_POSITION, pcl::PointXYZ hole_pos, pcl::PointXYZ cone_pos, bbx target_cone, IDEAL_POS IP, bool sprd_landed, int& devOut_pca_x, int& devOut_pca_y, int& devOut_pca_z, bool jobLog = false)
{
	try
	{
		bool isLeftSensor = (SENSOR_POSITION == "REAR_LEFT") ? true : false;

		if (hole_pos.x != -10000 && cone_pos.x != -10000)
		{
			//based on TUAS-3DSP.
			int x_offset = 14;
			int z_offset = 26;
			
			if (TZ_Mount_Cycle)
			{
				//Hole - Cone
				if (isLeftSensor) devOut_pca_x = (hole_pos.x - x_offset) - cone_pos.x;
				else devOut_pca_x = (hole_pos.x + x_offset) - cone_pos.x;
				devOut_pca_y = cone_pos.y - hole_pos.y;
				devOut_pca_z = cone_pos.z - (hole_pos.z + z_offset);

				if (sprd_landed || devOut_pca_y > -50)
				{
					if (std::abs(devOut_pca_x) > 50 || std::abs(devOut_pca_z) > 50)
					{
						if (landout_current_count_pca < LDO_NCOUNT)
						{
							landout_current_count_pca++;
						}
						if (landout_current_count_pca >= LDO_NCOUNT)
						{
							if (!landout_detected_pca)
							{
								logMessage("Landout detected by PCA!");
								if (jobLog) jobLogMessage("Landout detected by PCA!");
							}
							landout_detected_pca = true;
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
							if (!landout_detected_pca)
							{
								logMessage("Landout detected by PCA - DevY: " + std::to_string(devOut_pca_y) + "!");
								if (jobLog) jobLogMessage("Landout detected by PCA - DevY: " + std::to_string(devOut_pca_y) + "!");
							}
							landout_detected_pca = true;
						}
					}
					else
					{
						if (landout_current_count_pca > 0) landout_current_count_pca--;
						if (landout_current_count_pca == 0)
						{
							landout_detected_pca = false;
						}
					}
				}
				
				else
				{
					if (landout_current_count_pca > 0) landout_current_count_pca--;
					if (landout_current_count_pca == 0)
					{
						landout_detected_pca = false;
					}
				}
			}
			else if (TZ_Offload_Cycle)
			{
				//Compare landed position with ideal position.
				if (target_cone.label == 2)
				{
					devOut_pca_x = cone_pos.x - IP.LANDED_PCA.x;
					devOut_pca_y = cone_pos.y - IP.LANDED_PCA.y;
					devOut_pca_z = cone_pos.z - IP.LANDED_PCA.z;

					//here we may be able to deduce clps? if something is detected.
				}
				else
				{
					//unloaded cone is selected as target?
					//Cone - Hole
					devOut_pca_x = hole_pos.x - cone_pos.x;
					devOut_pca_y = cone_pos.y - hole_pos.y;
					devOut_pca_z = cone_pos.z - hole_pos.z;
				}
			}
		}
		else if (cone_pos.x != -10000)
		{
			//No hole, only cone.
			if (TZ_Mount_Cycle)
			{
				devOut_pca_x = cone_pos.x - IP.CONE_PCA.x;
				devOut_pca_y = cone_pos.y - IP.CONE_PCA.y;
				devOut_pca_z = cone_pos.z - IP.CONE_PCA.z;
			}
		}

		logMessage("Dev Out PCA T: " + std::to_string(devOut_pca_x) + " H: " + std::to_string(devOut_pca_y) + " G: " + std::to_string(devOut_pca_z));
		if (jobLog) jobLogMessage("Dev Out PCA T: " + std::to_string(devOut_pca_x) + " H: " + std::to_string(devOut_pca_y) + " G: " + std::to_string(devOut_pca_z));
		logMessage("LandOut Detected PCA: " + std::to_string(landout_detected_pca));
		if (jobLog) jobLogMessage("LandOut Detected PCA: " + std::to_string(landout_detected_pca));

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


void Sequential_Processing_Thread(VA& detector)
{
	try
	{
		logMessage("Sequential Processing Thread Activated!");

		while (seq_proc_running.load())
		{
			std::unique_lock<std::mutex> lock(mutex_seq_processing);
			bool res = cond_seq_processing.wait_for(lock,
				std::chrono::seconds(3600),
				[]() { return seq_proc_flag; });

			if (res)
			{
				logMessage("Sequential Processing Enabled! Current Position : " + CURRENT_SENSOR_POSITION);
				jobLogMessage("Sequential Processing Enabled! Current Position : " + CURRENT_SENSOR_POSITION);

				cv::Mat img;
				pcl::PointCloud<pcl::PointXYZ>::Ptr pcPointCloud(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ> pointCloud;
				std::chrono::system_clock::time_point last_ts, ts;

				IDEAL_POS IP;
				int g_data_limit = G_DATA_LIMIT;
				int t_data_limit = 2000;
				if (CURRENT_SENSOR_POSITION.find("LEFT") != std::string::npos)
				{
					IP = L_Pos;
					t_data_limit = -1 * T_DATA_LIMIT;
				}
				else
				{
					IP = R_Pos;
					t_data_limit = T_DATA_LIMIT;
				}

				int sprd_size = 40;
				if (SPRD_45ft) sprd_size = 45;
				else if (SPRD_20ft) sprd_size = 20;
				logMessage("SPRD Size: " + std::to_string(sprd_size));
				jobLogMessage("SPRD Size: " + std::to_string(sprd_size));

				//data log file.
				auto logHeader = std::string("Filename;Hole_x;Hole_y;Hole_w;Hole_h;Hole_prob;XT_index;XT_x;XT_y;XT_w;XT_h;XT_prob;Guide_x;Guide_y;Guide_w;Guide_y;Guide_prob;devout_x;devout_y;devout_x_mm;devout_y_mm;Landed_Trigger;Landout_Detected;CLPS_Detected;CLPS_OK;Hole_pc;Cone_pc;Guide_pc;hole_x;hole_y;hole_z;cone_x;cone_y;cone_z;devout_pca_x;devout_pca_y;devout_pca_z;landout_pca;clps_pca;clpsOk_pca;pcUnderHole;LDO_VA_Count;LDO_PCA_Count;CLPS_VA_Count;CLPS_PCA_Count;CLPSOK_PCA_Count");

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
				auto check_interval = 100;
				auto last_dt_checked = std::chrono::system_clock::now();

				//enable t mini.
				bool tmini_connected = false;
				int retry_count = 5;
				while (!tmini_connected)
				{
					tmini_connected = tmini_setup();
					if (tmini_connected)
					{
						sensor_last_connected_time = std::chrono::system_clock::now();
						sensor_connected = true;
						sensor_fault = false;

						logMessage("TMini Setup Success!");
						break;
					}
					else
					{
						if (retry_count <= 0)
						{
							logMessage("TMini Setup Failed! Exiting...");
							jobLogMessage("TMini Setup Failed! Exiting...");
							break;
						}

						sensor_connected = false;
						sensor_fault = true;

						logMessage("TMini Setup Failed!");

						retry_count--;
						std::this_thread::sleep_for(std::chrono::seconds(5)); //wait for 5 seconds before retrying.
					}
				}
				
				while (enable_process && tmini_connected)
				{
					auto get_dt = std::chrono::system_clock::now();
					auto wait_dur = std::chrono::duration_cast<std::chrono::milliseconds>(get_dt - last_dt_checked).count();
					if (wait_dur >= check_interval)
					{
						logMessage("Processing Cycle Start!");
						//Get Data
						blnGetNewFrame = false;
						bool stepComplete = visionaryControl.stepAcquisition();
						if (stepComplete)
						{
							if (dataStream.getNextFrame())
							{
								blnGetNewFrame = true;
								last_frame_get_time = std::chrono::system_clock::now();
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
									//cv::Mat im3; // I want im3 to be the CV_16UC1 of im2
									gImg.convertTo(img, CV_8UC1);

									pcPointCloud = makePCL_PointCloud(pointCloud);
									if (pcPointCloud == nullptr)
									{
										logMessage("[TMini-Data-Stream] Failed to create PCL Point Cloud from data handler. -- nullptr!");
									}

									continue;

									//processing here.
									frame_index++;
									std::string log_lines = std::to_string(frame_index) + ";";

									auto res_img = img.clone();
									auto t_now = std::chrono::system_clock::now();

									int det_count = 0;
									std::vector<rectangle_info> det_results;
									std::vector<std::vector<bbx>> det_sorted_objects(4, std::vector<bbx>(0));
									auto inference_status = onnx_inference(detector, CURRENT_SENSOR_POSITION, img, ref(det_results), ref(det_sorted_objects), ref(det_count), std::string(""), true);

									if (enable_logging)
									{
										if (save_trigger_by_landed || save_trigger_by_TWL_Locked)
										{
											//Draw and push to queue.
											res_img = drawOnImage(res_img, det_results);

											std::string msg = save_trigger_by_TWL_Locked ? "TWL" : "TWUL";
											std::string msg2 = save_trigger_by_landed ? "LANDED" : "LANDOFF";
											msg = msg + std::string("_") + msg2;

											cv::Mat saveMat = img.clone();
											cv::Mat resMat = res_img.clone();
											pcl::PointCloud<pcl::PointXYZ> savePointCloud;
											pcl::copyPointCloud(*pcPointCloud, savePointCloud);

											auto dataTup = std::tuple<cv::Mat, cv::Mat, pcl::PointCloud<pcl::PointXYZ>, bool, std::string, std::chrono::system_clock::time_point>(saveMat, resMat, savePointCloud, true, msg, ts);
											tsq.push(dataTup);

											if (save_trigger_by_landed) save_trigger_by_landed = false;
											if (save_trigger_by_TWL_Locked) save_trigger_by_TWL_Locked = false;
										}

										else
										{
											cv::Mat saveMat = img.clone();
											pcl::PointCloud<pcl::PointXYZ> savePointCloud;
											pcl::copyPointCloud(*pcPointCloud, savePointCloud);

											auto dataTup = std::tuple<cv::Mat, cv::Mat, pcl::PointCloud<pcl::PointXYZ>, bool, std::string, std::chrono::system_clock::time_point>(saveMat, cv::Mat(), savePointCloud, false, "", last_frame_get_time);
											tsq.push(dataTup);
										}
									}
									
									

									//Initialize variables
									bbx target_hole, target_cone, target_landed, target_guide;
									tCntr_x = -10000;
									tCntr_y = -10000;
									tCntr_prob = 0;
									tCone_x = -10000;
									tCone_y = -10000;
									tCone_prob = 0;

									devOut_x = -10000; devOut_y = -10000;
									devOut_x_mm = -10000; devOut_y_mm = -10000;

									devOut_pca_x = -10000; devOut_pca_y = -10000; devOut_pca_z = -10000;

									bool detected_hole = false;

									if (det_count == 0)
									{
										logMessage("No acceptable inference results are available. Proceeding to PCA");
										jobLogMessage("No acceptable inference results are available. Proceeding to PCA");
										detected_xt = false;
										detected_cst = false;
										detected_chassis_type_unknown = true;
									}
									else
									{
#pragma region Chassis Type detection
										chassis_type_selection_VA(det_sorted_objects, true);
#pragma endregion
#pragma region Target selections
										Target_Selections_VA(CURRENT_SENSOR_POSITION, sprd_size, det_sorted_objects, ref(target_hole), ref(target_cone), ref(target_guide), ref(detected_hole), true);
#pragma endregion
#pragma region VA Pre-Land Chassis Position
										Pre_Land_chassis_position_VA(target_cone, target_hole, true);
#pragma endregion
#pragma region VA deviation output
										bool usingLDO_Base = false;
										Deviation_Output_VA(target_hole, target_cone, target_guide, JOB_IP_Pos, ref(tCntr_x), ref(tCntr_y), ref(tCntr_prob), ref(tCone_x), ref(tCone_y), ref(tCone_prob), ref(devOut_x), ref(devOut_y), ref(devOut_x_mm), ref(devOut_y_mm), ref(usingLDO_Base), true);
#pragma endregion
#pragma region landout detection
										LandOut_Detected_VA(target_hole, target_cone, SPRD_Landed, detected_hole, usingLDO_Base, true);
#pragma endregion
#pragma region clps detection
										CLPS_Detection_VA(target_hole, target_cone, SPRD_Landed, true);
#pragma endregion
										auto proc_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t_now).count();
										logMessage("VA Proc Time in ms : " + std::to_string(proc_time_ms) + "ms");
									}

									log_lines += std::to_string(target_hole.x) + ";" + std::to_string(target_hole.y) + ";" + std::to_string(target_hole.w) + ";" + std::to_string(target_hole.h) + ";" + std::to_string(target_hole.prob) + ";";
									log_lines += std::to_string(target_cone.label) + ";" + std::to_string(target_cone.x) + ";" + std::to_string(target_cone.y) + ";" + std::to_string(target_cone.w) + ";" + std::to_string(target_cone.h) + ";" + std::to_string(target_cone.prob) + ";";
									log_lines += std::to_string(target_guide.x) + ";" + std::to_string(target_guide.y) + ";" + std::to_string(target_guide.w) + ";" + std::to_string(target_guide.h) + ";" + std::to_string(target_guide.prob) + ";";

									log_lines += std::to_string(devOut_x) + ";" + std::to_string(devOut_y) + ";";
									log_lines += std::to_string(devOut_x_mm) + ";" + std::to_string(devOut_y_mm) + ";";
									//2025.03.11
									log_lines += std::to_string(SPRD_Landed) + ";" + std::to_string(landout_detected) + ";" + std::to_string(clps_detected) + ";" + std::to_string(clps_ok_detected) + ";";

									/*
									//POINTCLOUD
									//Get target hole, target chassis, target guide
									pcl::PointCloud<pcl::PointXYZ>::Ptr pcHole(new pcl::PointCloud <pcl::PointXYZ>);
									pcl::PointCloud<pcl::PointXYZ>::Ptr pcCone(new pcl::PointCloud <pcl::PointXYZ>);
									pcl::PointCloud<pcl::PointXYZ>::Ptr pcGuide(new pcl::PointCloud <pcl::PointXYZ>);

#pragma region Target pointcloud extraction via VA-Based
									Target_PointCloud_Extraction(pcPointCloud, target_hole, target_cone, target_guide, SPRD_Landed, JOB_IP_Pos, t_data_limit, g_data_limit, ref(pcHole), ref(pcCone), ref(pcGuide), std::string(""), true);
#pragma endregion
#pragma region PCA CLPS Logic
									int refPCUnderHole = 0;
									CLPS_Detection_PCA(pcPointCloud, target_hole, g_data_limit, t_data_limit, detected_hole, SPRD_Landed, ref(refPCUnderHole), std::string(""), true);
#pragma endregion

									pcl::PointXYZ hole_pos(-10000, -10000, -10000);
									pcl::PointXYZ cone_pos(-10000, -10000, -10000);

#pragma region PCA - Hole, Cone detection
									Position_Detection_PCA(CURRENT_SENSOR_POSITION, target_cone, pcHole, pcCone, ref(hole_pos), ref(cone_pos), std::string(""), true);
#pragma endregion
#pragma region PCA Pre-Land Chassis Position
									Pre_Land_Chassis_Position_PCA(target_cone, cone_pos, pcCone, true);
#pragma endregion
									bool usingPCA_Base = false;
									if (PCA_Base_Set && cone_pos.x == -10000)
									{
										//if cone is not detected, then use PCA_Base as cone position.
										cone_pos.x = PCA_Base.x;
										cone_pos.y = PCA_Base.y;
										cone_pos.z = PCA_Base.z;
										usingPCA_Base = true;
									}

#pragma region dev_out_in pca
									Deviation_Output_PCA(CURRENT_SENSOR_POSITION, hole_pos, cone_pos, target_cone, IP, SPRD_Landed, ref(devOut_pca_x), ref(devOut_pca_y), ref(devOut_pca_z), true);
#pragma endregion
									
									log_lines += std::to_string(pcHole->points.size()) + ";" + std::to_string(pcCone->points.size()) + ";" + std::to_string(pcGuide->points.size()) + ";";
									log_lines += std::to_string(hole_pos.x) + ";" + std::to_string(hole_pos.y) + ";" + std::to_string(hole_pos.z) + ";";

									log_lines += std::to_string(cone_pos.x) + ";" + std::to_string(cone_pos.y) + ";" + std::to_string(cone_pos.z) + ";";

									log_lines += std::to_string(devOut_pca_x) + ";" + std::to_string(devOut_pca_y) + ";" + std::to_string(devOut_pca_z) + ";";
									log_lines += std::to_string(landout_detected_pca) + ";" + std::to_string(clps_detected_pca) + ";" + std::to_string(clps_ok_detected_pca);
									log_lines += ";" + std::to_string(refPCUnderHole) + ";";
									log_lines += std::to_string(LDO_Current_Count) + ";" + std::to_string(landout_current_count_pca) + ";";
									log_lines += std::to_string(CLPS_Current_Count) + ";" + std::to_string(clps_current_count_pca) + ";" + std::to_string(clps_ok_current_count_pca);
									*/

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
						int ms = 200 - wait_dur;
						logMessage("Waiting for cycle... " + std::to_string(ms) + " ms");
						std::this_thread::sleep_for(std::chrono::milliseconds(ms));
					}
				}

				if (!tmini_connected)
				{
					logMessage("TMini Setup Failed! Blocked state for Sequential Processing Thread...");
					jobLogMessage("TMini Setup Failed! Block state for Sequential Processing Thread...");
				}
				if (!enable_process)
				{
					logMessage("Sequential Processing Thread Disabled! To Blocked state...");
					jobLogMessage("Sequential Processing Thread Disabled! To blocked state...");
				}
			}
			else
			{
				logMessage("Sequential Processing Thread Timeout! No enable proc received in the last hour.");
				jobLogMessage("Sequential Processing Thread Timeout! No enable proc received in the last hour.");
				continue;
			}
		}
		//connect to sensor,
		//stream data
		//process.
	}
	catch (std::exception& ex)
	{
		logMessage("[Sequential-Processing-Thread] " + std::string(ex.what()));
	}
	catch (...)
	{
		logMessage("[Sequential-Processing-Thread] Unknown Exception!");
	}
}


void processingThread(VA& detector)
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
					[]() { return flag; });

				if (res)
				{
					logMessage("Enabled process! Current Position : " + CURRENT_SENSOR_POSITION);
					jobLogMessage("Enabled process! Current Position : " + CURRENT_SENSOR_POSITION);

					cv::Mat img;
					pcl::PointCloud<pcl::PointXYZ>::Ptr pcPointCloud(new pcl::PointCloud<pcl::PointXYZ>);
					pcl::PointCloud<pcl::PointXYZ> pointCloud;
					std::chrono::system_clock::time_point last_ts, ts;

					IDEAL_POS IP;
					int g_data_limit = G_DATA_LIMIT;
					int t_data_limit = 2000;
					if (CURRENT_SENSOR_POSITION.find("LEFT") != std::string::npos)
					{
						IP = L_Pos;
						t_data_limit = -1 * T_DATA_LIMIT;
					}
					else
					{
						IP = R_Pos;
						t_data_limit = T_DATA_LIMIT;
					}

					int sprd_size = 40;
					if (SPRD_45ft) sprd_size = 45;
					else if (SPRD_20ft) sprd_size = 20;
					logMessage("SPRD Size: " + std::to_string(sprd_size));
					jobLogMessage("SPRD Size: " + std::to_string(sprd_size));

					//data log file.
					auto logHeader = std::string("Filename;Hole_x;Hole_y;Hole_w;Hole_h;Hole_prob;XT_index;XT_x;XT_y;XT_w;XT_h;XT_prob;Guide_x;Guide_y;Guide_w;Guide_y;Guide_prob;devout_x;devout_y;devout_x_mm;devout_y_mm;Landed_Trigger;Landout_Detected;CLPS_Detected;CLPS_OK;Hole_pc;Cone_pc;Guide_pc;hole_x;hole_y;hole_z;cone_x;cone_y;cone_z;devout_pca_x;devout_pca_y;devout_pca_z;landout_pca;clps_pca;clpsOk_pca;pcUnderHole;LDO_VA_Count;LDO_PCA_Count;CLPS_VA_Count;CLPS_PCA_Count;CLPSOK_PCA_Count");

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
					auto check_interval = 100;
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
								//std::cout << print_time() << " Trying to retrieve stack!\n";

								logMessage("Retrieving stack data...");
								auto r_status = dataStack.Retrieve_Stack(ref(img), ref(pointCloud), ref(ts));
								if (r_status) logMessage("Stack data retrieved successfully!");
								else
								{
									logMessage("Failed to retrieve stack data!");
									std::this_thread::sleep_for(std::chrono::milliseconds(30)); //single frame capture time.
									continue;
								}

								if (enable_logging)
								{
									//save data to queue with copied data.
									cv::Mat saveMat = img.clone();
									pcl::PointCloud<pcl::PointXYZ> savePointCloud;
									pcl::copyPointCloud(pointCloud, savePointCloud);

									auto dataTup = std::tuple<cv::Mat, cv::Mat, pcl::PointCloud<pcl::PointXYZ>, bool, std::string, std::chrono::system_clock::time_point>(saveMat, cv::Mat(), savePointCloud, false, "", last_frame_get_time);
									tsq.push(dataTup);
								}
							
								{
									//Convert visionary::pointcloud to pcl::pointcloud
									if (pcPointCloud == nullptr)
									{
										pcPointCloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
										logMessage("Created new pcl::PointCloud<pcl::PointXYZ> instance.");
									}
									pcPointCloud->points.resize(pointCloud.size());
									std::copy(pointCloud.begin(), pointCloud.end(), pcPointCloud->points.begin());

									auto res_img = img.clone();

									if (r_status && last_ts < ts)
									{
										last_ts = ts;

										auto t_now = std::chrono::system_clock::now();

										int det_count = 0;
										std::vector<rectangle_info> det_results;
										std::vector<std::vector<bbx>> det_sorted_objects(4, std::vector<bbx>(0));
										auto inference_status = onnx_inference(detector, CURRENT_SENSOR_POSITION, img, ref(det_results), ref(det_sorted_objects), ref(det_count), std::string(""), true);

										if (save_trigger_by_landed || save_trigger_by_TWL_Locked)
										{
											//Draw and push to queue.
											res_img = drawOnImage(res_img, det_results);

											std::string msg = save_trigger_by_TWL_Locked ? "TWL" : "TWUL";
											std::string msg2 = save_trigger_by_landed ? "LANDED" : "LANDOFF";
											msg = msg + std::string("_") + msg2;

											auto dataTup = std::tuple<cv::Mat, cv::Mat, pcl::PointCloud<pcl::PointXYZ>, bool, std::string, std::chrono::system_clock::time_point>(img, res_img, *pcPointCloud, true, msg, ts);
											tsq.push(dataTup);

											if (save_trigger_by_landed) save_trigger_by_landed = false;
											if (save_trigger_by_TWL_Locked) save_trigger_by_TWL_Locked = false;
										}

										//Initialize variables
										bbx target_hole, target_cone, target_landed, target_guide;
										tCntr_x = -10000;
										tCntr_y = -10000;
										tCntr_prob = 0;
										tCone_x = -10000;
										tCone_y = -10000;
										tCone_prob = 0;

										devOut_x = -10000; devOut_y = -10000;
										devOut_x_mm = -10000; devOut_y_mm = -10000;

										devOut_pca_x = -10000; devOut_pca_y = -10000; devOut_pca_z = -10000;

										bool detected_hole = false;

										if (det_count == 0)
										{
											logMessage("No acceptable inference results are available. Proceeding to PCA");
											jobLogMessage("No acceptable inference results are available. Proceeding to PCA");
											detected_xt = false;
											detected_cst = false;
											detected_chassis_type_unknown = true;
										}
										else
										{
#pragma region Chassis Type detection
											chassis_type_selection_VA(det_sorted_objects, true);
#pragma endregion
#pragma region Target selections
											Target_Selections_VA(CURRENT_SENSOR_POSITION, sprd_size, det_sorted_objects, ref(target_hole), ref(target_cone), ref(target_guide), ref(detected_hole), true);
#pragma endregion
#pragma region VA Pre-Land Chassis Position
											Pre_Land_chassis_position_VA(target_cone, target_hole, true);
#pragma endregion
#pragma region VA deviation output
											bool usingLDO_Base = false;
											Deviation_Output_VA(target_hole, target_cone, target_guide, JOB_IP_Pos, ref(tCntr_x), ref(tCntr_y), ref(tCntr_prob), ref(tCone_x), ref(tCone_y), ref(tCone_prob), ref(devOut_x), ref(devOut_y), ref(devOut_x_mm), ref(devOut_y_mm), ref(usingLDO_Base), true);
#pragma endregion
#pragma region landout detection
											LandOut_Detected_VA(target_hole, target_cone, SPRD_Landed, detected_hole, usingLDO_Base, true);
#pragma endregion
#pragma region clps detection
											CLPS_Detection_VA(target_hole, target_cone, SPRD_Landed, true);
#pragma endregion
											auto proc_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t_now).count();
											logMessage("VA Proc Time in ms : " + std::to_string(proc_time_ms) + "ms");
										}

										log_lines += std::to_string(target_hole.x) + ";" + std::to_string(target_hole.y) + ";" + std::to_string(target_hole.w) + ";" + std::to_string(target_hole.h) + ";" + std::to_string(target_hole.prob) + ";";
										log_lines += std::to_string(target_cone.label) + ";" + std::to_string(target_cone.x) + ";" + std::to_string(target_cone.y) + ";" + std::to_string(target_cone.w) + ";" + std::to_string(target_cone.h) + ";" + std::to_string(target_cone.prob) + ";";
										log_lines += std::to_string(target_guide.x) + ";" + std::to_string(target_guide.y) + ";" + std::to_string(target_guide.w) + ";" + std::to_string(target_guide.h) + ";" + std::to_string(target_guide.prob) + ";";

										log_lines += std::to_string(devOut_x) + ";" + std::to_string(devOut_y) + ";";
										log_lines += std::to_string(devOut_x_mm) + ";" + std::to_string(devOut_y_mm) + ";";
										//2025.03.11
										log_lines += std::to_string(SPRD_Landed) + ";" + std::to_string(landout_detected) + ";" + std::to_string(clps_detected) + ";" + std::to_string(clps_ok_detected) + ";";


										//POINTCLOUD
										//Get target hole, target chassis, target guide
										pcl::PointCloud<pcl::PointXYZ>::Ptr basePointCloud(new pcl::PointCloud<pcl::PointXYZ>);
										pcl::PointCloud<pcl::PointXYZ>::Ptr pcHole(new pcl::PointCloud <pcl::PointXYZ>);
										pcl::PointCloud<pcl::PointXYZ>::Ptr pcCone(new pcl::PointCloud <pcl::PointXYZ>);
										pcl::PointCloud<pcl::PointXYZ>::Ptr pcGuide(new pcl::PointCloud <pcl::PointXYZ>);

										//Just copy the point cloud to basePointCloud.
										if (pcPointCloud == nullptr)
										{
											logMessage("[Processing-Thread] pcPointCloud is nullptr!");
										}
										else
										{
											pcl::copyPointCloud(*pcPointCloud, *basePointCloud);
										}

#pragma region Target pointcloud extraction via VA-Based
										Target_PointCloud_Extraction(basePointCloud, target_hole, target_cone, target_guide, SPRD_Landed, JOB_IP_Pos, t_data_limit, g_data_limit, ref(pcHole), ref(pcCone), ref(pcGuide), std::string(""), true);
#pragma endregion
#pragma region PCA CLPS Logic
										int refPCUnderHole = 0;
										CLPS_Detection_PCA(basePointCloud, target_hole, g_data_limit, t_data_limit, detected_hole, SPRD_Landed, ref(refPCUnderHole), std::string(""), true);
#pragma endregion

										pcl::PointXYZ hole_pos(-10000, -10000, -10000);
										pcl::PointXYZ cone_pos(-10000, -10000, -10000);

#pragma region PCA - Hole, Cone detection
										Position_Detection_PCA(CURRENT_SENSOR_POSITION, target_cone, pcHole, pcCone, ref(hole_pos), ref(cone_pos), std::string(""), true);
#pragma endregion
#pragma region PCA Pre-Land Chassis Position
										Pre_Land_Chassis_Position_PCA(target_cone, cone_pos, pcCone, true);
#pragma endregion
										bool usingPCA_Base = false;
										if (PCA_Base_Set && cone_pos.x == -10000)
										{
											//if cone is not detected, then use PCA_Base as cone position.
											cone_pos.x = PCA_Base.x;
											cone_pos.y = PCA_Base.y;
											cone_pos.z = PCA_Base.z;
											usingPCA_Base = true;
										}

#pragma region dev_out_in pca
										Deviation_Output_PCA(CURRENT_SENSOR_POSITION, hole_pos, cone_pos, target_cone, IP, SPRD_Landed, ref(devOut_pca_x), ref(devOut_pca_y), ref(devOut_pca_z), true);
#pragma endregion

										log_lines += std::to_string(pcHole->points.size()) + ";" + std::to_string(pcCone->points.size()) + ";" + std::to_string(pcGuide->points.size()) + ";";
										log_lines += std::to_string(hole_pos.x) + ";" + std::to_string(hole_pos.y) + ";" + std::to_string(hole_pos.z) + ";";

										log_lines += std::to_string(cone_pos.x) + ";" + std::to_string(cone_pos.y) + ";" + std::to_string(cone_pos.z) + ";";

										log_lines += std::to_string(devOut_pca_x) + ";" + std::to_string(devOut_pca_y) + ";" + std::to_string(devOut_pca_z) + ";";
										log_lines += std::to_string(landout_detected_pca) + ";" + std::to_string(clps_detected_pca) + ";" + std::to_string(clps_ok_detected_pca);
										log_lines += ";" + std::to_string(refPCUnderHole) + ";";
										log_lines += std::to_string(LDO_Current_Count) + ";" + std::to_string(landout_current_count_pca) + ";";
										log_lines += std::to_string(CLPS_Current_Count) + ";" + std::to_string(clps_current_count_pca) + ";" + std::to_string(clps_ok_current_count_pca);


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

void OfflineDebugBatchProcessingThread(VA& detector)
{
	logMessage("Debug Batch Job on " + DEBUG_BATCH_ROOT_DIR + " to be processed and saved to " + DEBUG_BATCH_SAVE_DIR);

	//Get list of sub directories given ROOT DIR.
	auto jobDirectories = ListSubDirectories(DEBUG_BATCH_ROOT_DIR);

	bool savePLY = true;

	for (const auto& jobDir : jobDirectories)
	{
		TZ_Mount_Cycle = false;
		TZ_Offload_Cycle = false;

		reset_jobVariables();

		logMessage("Processing: " + jobDir);

		//load .jpg, .ply files for processing
		auto image_filePath = jobDir + "/" + std::string("Image");
		auto depth_filePath = jobDir + "/" + std::string("Depth");

		std::filesystem::path pathObj = std::filesystem::path(jobDir).lexically_normal();
		std::string lastDirectory = pathObj.filename().string();

		auto image_files = getAllFiles(image_filePath, ".jpg");
		auto depth_files = getAllFiles(depth_filePath, ".ply");

		createDirectory_ifexists(DEBUG_BATCH_SAVE_DIR);
		auto save_file_path = DEBUG_BATCH_SAVE_DIR + "/" + lastDirectory;
		auto folder_created = createDirectory_ifexists(save_file_path);

		if (!folder_created)
		{
			logMessage("Skipping already existing folder: " + save_file_path);
			continue;
		}

		logMessage("Saving to " + save_file_path);

		auto logHeader = std::string("Filename;Hole_x;Hole_y;Hole_w;Hole_h;Hole_prob;XT_index;XT_x;XT_y;XT_w;XT_h;XT_prob;Guide_x;Guide_y;Guide_w;Guide_y;Guide_prob;devout_x;devout_y;devout_x_mm;devout_y_mm;Landed_Trigger;Landout_Detected;CLPS_Detected;CLPS_OK;Hole_pc;Cone_pc;Guide_pc;hole_x;hole_y;hole_z;cone_x;cone_y;cone_z;devout_pca_x;devout_pca_y;devout_pca_z;landout_pca;clps_pca;clpsOk_pca;pcUnderHole;LDO_VA_Count;LDO_PCA_Count;CLPS_VA_Count;CLPS_PCA_Count;CLPSOK_PCA_Count");

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

		//Get job information from the folder name.
		auto info_temp = split(lastDirectory, '_');
		if (info_temp.size() > 3)
		{
			if (info_temp[3] == "Mount")
			{
				TZ_Mount_Cycle = true;
				TZ_Offload_Cycle = false;
			}
			else if (info_temp[3] == "Offload")
			{
				TZ_Offload_Cycle = true;
				TZ_Mount_Cycle = false;
			}
			else
			{
				TZ_Mount_Cycle = false;
				TZ_Offload_Cycle = false;
			}
		}
		int sprd_size = 40;
		if (info_temp.size() > 7)
		{
			if (info_temp[7] == "SPRD45")
			{
				sprd_size = 45;
			}
			else if (info_temp[7] == "SRPD20")
			{
				sprd_size = 20;
			}
		}

		if (TZ_Mount_Cycle) logMessage("Mount Cycle Detected");
		else if (TZ_Offload_Cycle) logMessage("Offload Cycle Detected");
		
		logMessage("SPRDSIZE: " + std::to_string(sprd_size));

		IDEAL_POS IP;
		int g_data_limit = G_DATA_LIMIT;
		int t_data_limit = 2000;
		if (DEBUG_SENSOR_POSITION.find("LEFT") != std::string::npos)
		{
			IP = L_Pos;
			t_data_limit = -1 * T_DATA_LIMIT;
		}
		else
		{
			IP = R_Pos;
			t_data_limit = T_DATA_LIMIT;
		}

		//2025.03.11
		bool debug_landed_trigger = false;

		for (int i = 0; i < image_files.size(); i++)
		{
			logMessage("Processing Cycle Start!");

			std::filesystem::path pathObj(image_files.at(i));
			// Get the filename with extension
			std::string filename = pathObj.stem().string();

			std::string log_lines = filename + ";";

			//folder for each file
			auto save_current_file_path = save_file_path + "/" + filename;
			createDirectory_ifexists(save_current_file_path);
			
			//load image
			cv::Mat image = cv::imread(image_files.at(i));

			if (image.empty())
			{
				logMessage("Couldn't read file by JPG " + image_files.at(i));
				continue;
			}

			//load ply
			pcl::PointCloud<pcl::PointXYZ>::Ptr pcData = std::make_shared<pcl::PointCloud <pcl::PointXYZ>>();
			if (pcl::io::loadPLYFile<pcl::PointXYZ>(depth_files.at(i), *pcData) == -1)
			{
				std::cout << std::string("Couldn't read file by PLY ") << std::endl;
				//return false;
			}
			auto pointCloud = makePCL_PointCloud(*pcData, DEBUG_CONVERT_PCL_RANGE);
			pcl::io::savePLYFileBinary(save_current_file_path + "/" + filename + "_base_converted.ply", *pointCloud);

			auto t_now = std::chrono::system_clock::now();
			//logMessage("Here: " + save_current_file_path);
			int det_count = 0;
			std::vector<rectangle_info> det_results;
			std::vector<std::vector<bbx>> det_sorted_objects(4, std::vector<bbx>(0));
			auto inference_status = onnx_inference(detector, DEBUG_SENSOR_POSITION, image, ref(det_results), ref(det_sorted_objects), ref(det_count), save_current_file_path + "/" + filename);

			//Initialize variables
			bbx target_hole, target_cone, target_landed, target_guide;
			tCntr_x = -10000;
			tCntr_y = -10000;
			tCntr_prob = 0;
			tCone_x = -10000;
			tCone_y = -10000;
			tCone_prob = 0;

			devOut_x = -10000; devOut_y = -10000;
			devOut_x_mm = -10000; devOut_y_mm = -10000;

			devOut_pca_x = 0; devOut_pca_y = 0; devOut_pca_z = 0;

			bool detected_hole = false;

			if (det_count == 0)
			{
				logMessage("No acceptable inference results are available. Proceeding to PCA");

				detected_xt = false;
				detected_cst = false;
				detected_chassis_type_unknown = true;
			}
			else
			{
				#pragma region Chassis Type detection
				chassis_type_selection_VA(det_sorted_objects);	
				#pragma endregion
				#pragma region Target selections
				Target_Selections_VA(DEBUG_SENSOR_POSITION, sprd_size, det_sorted_objects, ref(target_hole), ref(target_cone), ref(target_guide), ref(detected_hole));
				#pragma endregion
				#pragma endregion
				#pragma region VA Pre-Land Chassis Position				
				Pre_Land_chassis_position_VA(target_cone, target_hole);	
				#pragma endregion
				#pragma region VA deviation output
				bool usingLDO_Base = false;
				Deviation_Output_VA(target_hole, target_cone, target_guide, IP, ref(tCntr_x), ref(tCntr_y), ref(tCntr_prob), ref(tCone_x), ref(tCone_y), ref(tCone_prob), ref(devOut_x), ref(devOut_y), ref(devOut_x_mm), ref(devOut_y_mm), ref(usingLDO_Base));
				#pragma endregion
				debug_landed_trigger = Debug_Landed_Trigger(target_hole, target_cone);
				#pragma region landout detection
				LandOut_Detected_VA(target_hole, target_cone, debug_landed_trigger, detected_hole, usingLDO_Base);
				#pragma endregion
				#pragma region clps detection
				CLPS_Detection_VA(target_hole, target_cone, debug_landed_trigger);
				#pragma endregion
				auto proc_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t_now).count();
				std::cout << print_time() << " VA Proc Time in ms : " << std::to_string(proc_time_ms) << std::endl;
			}

			log_lines += std::to_string(target_hole.x) + ";" + std::to_string(target_hole.y) + ";" + std::to_string(target_hole.w) + ";" + std::to_string(target_hole.h) + ";" + std::to_string(target_hole.prob) + ";";
			log_lines += std::to_string(target_cone.label) + ";" + std::to_string(target_cone.x) + ";" + std::to_string(target_cone.y) + ";" + std::to_string(target_cone.w) + ";" + std::to_string(target_cone.h) + ";" + std::to_string(target_cone.prob) + ";";
			log_lines += std::to_string(target_guide.x) + ";" + std::to_string(target_guide.y) + ";" + std::to_string(target_guide.w) + ";" + std::to_string(target_guide.h) + ";" + std::to_string(target_guide.prob) + ";";

			log_lines += std::to_string(devOut_x) + ";" + std::to_string(devOut_y) + ";";
			log_lines += std::to_string(devOut_x_mm) + ";" + std::to_string(devOut_y_mm) + ";";
			//2025.03.11
			log_lines += std::to_string(debug_landed_trigger) + ";" + std::to_string(landout_detected) + ";" + std::to_string(clps_detected) + ";" + std::to_string(clps_ok_detected) + ";"; 

			//POINTCLOUD
				
			//Get target hole, target chassis, target guide
			pcl::PointCloud<pcl::PointXYZ>::Ptr pcHole(new pcl::PointCloud <pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr pcCone(new pcl::PointCloud <pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr pcGuide(new pcl::PointCloud <pcl::PointXYZ>);

			logMessage("t limit: " + std::to_string(t_data_limit) + " , g limit: " + std::to_string(g_data_limit));

			#pragma region Target pointcloud extraction via VA-Based
			Target_PointCloud_Extraction(pointCloud, target_hole, target_cone, target_guide, debug_landed_trigger, IP, t_data_limit, g_data_limit, ref(pcHole), ref(pcCone), ref(pcGuide), save_current_file_path + "/" + filename);
			#pragma endregion
		
			#pragma region PCA CLPS Logic
			int refPCUnderHole = 0;
			CLPS_Detection_PCA(pointCloud, target_hole, g_data_limit, t_data_limit, detected_hole, false, ref(refPCUnderHole), save_current_file_path + "/" + filename);
			#pragma endregion

			pcl::PointXYZ hole_pos(-10000, -10000, -10000);
			pcl::PointXYZ cone_pos(-10000, -10000, -10000);

			#pragma region PCA - Hole, Cone detection
			Position_Detection_PCA(DEBUG_SENSOR_POSITION, target_cone, pcHole, pcCone, ref(hole_pos), ref(cone_pos), save_current_file_path + "/" + filename);
			#pragma endregion
			
			logMessage("Done PLY Position Logics");
			
			#pragma region PCA Pre-Land Chassis Position
			Pre_Land_Chassis_Position_PCA(target_cone, cone_pos, pcCone);
			#pragma endregion

			bool usingPCA_Base = false;
			if (PCA_Base_Set && cone_pos.x == -10000)
			{
				//if cone is not detected, then use PCA_Base as cone position.
				cone_pos.x = PCA_Base.x;
				cone_pos.y = PCA_Base.y;
				cone_pos.z = PCA_Base.z;
				usingPCA_Base = true;
			}
			
			#pragma region dev_out_in pca
			Deviation_Output_PCA(DEBUG_SENSOR_POSITION,hole_pos, cone_pos, target_cone, IP, debug_landed_trigger, ref(devOut_pca_x), ref(devOut_pca_y), ref(devOut_pca_z));
			#pragma endregion

			log_lines += std::to_string(pcHole->points.size()) + ";" + std::to_string(pcCone->points.size()) + ";" + std::to_string(pcGuide->points.size()) + ";";
			log_lines += std::to_string(hole_pos.x) + ";" + std::to_string(hole_pos.y) + ";" + std::to_string(hole_pos.z) + ";";

			log_lines += std::to_string(cone_pos.x) + ";" + std::to_string(cone_pos.y) + ";" + std::to_string(cone_pos.z) + ";";

			log_lines += std::to_string(devOut_pca_x) + ";" + std::to_string(devOut_pca_y) + ";" + std::to_string(devOut_pca_z) + ";";
			log_lines += std::to_string(landout_detected_pca) + ";" + std::to_string(clps_detected_pca) + ";" + std::to_string(clps_ok_detected_pca) + ";";

			log_lines += std::to_string(refPCUnderHole) + ";" + std::to_string(LDO_Current_Count) + ";" + std::to_string(landout_current_count_pca) + ";";
			log_lines += std::to_string(CLPS_Current_Count) + ";" + std::to_string(clps_current_count_pca) + ";";
			log_lines += std::to_string(clps_ok_current_count_pca);

			logMessage("Dev Out PCA x: " + std::to_string(devOut_pca_x) + " y: " + std::to_string(devOut_pca_y) + " z: " + std::to_string(devOut_pca_z));
			logMessage("LandOut PCA: " + std::to_string(landout_detected_pca));

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
}

pcl::PointXYZ pc_hole_detection_zedx(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool isLeftSide, std::string save_path)
{
	pcl::PointXYZ hole_position(10000, -10000, 10000);
	try
	{
		//Get Y slices :: 20 height slices.
		//Cornercastings height = usually 120mm.
		int y_slice_height = 20;

		int minY, maxY;
		get_max_min_y(pcInput, ref(maxY), ref(minY));

		std::vector<pcl::PointXYZ> target_points = {};

		int i = 0;
		for (int cy = minY; cy < maxY; cy += y_slice_height)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(pcInput, cy, cy + y_slice_height, "y", slice_);
			if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_hole_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".ply", *slice_);
			if (val)
			{
				if (slice_->points.size() > 75)
				{
					int minZ = -10000, maxZ = -10000;
					get_max_min_z(slice_, ref(maxZ), ref(minZ));

					//from minZ slice, get minX.
					int minX = -10000, maxX = -10000;
					get_max_min_x(slice_, ref(maxX), ref(minX));

					//Get maxZ and minX.
					auto targetPt = pcl::PointXYZ(minX, cy, maxZ);
					if (!isLeftSide) targetPt.x = maxX;

					target_points.push_back(targetPt);
					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_hole_targetPt_i_" + std::to_string(i) + ".ply", *tPt);
				}
			}
			i++;
		}

		//based on target points.
		//average x and z while get min y.
		float avg_x = 0.0f, avg_z = 0.0f;
		for (auto& it : target_points)
		{
			if (it.y > hole_position.y) hole_position.y = it.y;
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
pcl::PointXYZ pc_cone_detection_zedx(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool isLeftSide, std::string save_path)
{
	pcl::PointXYZ cone_position(10000, 10000, 10000);
	try
	{
		//slice through y direction with slice height = 20.

		//slice from top to bottom.
		//look for most points then append one slice above as well.

		int y_slice_height = 20;
		int max_slice_index = 6;

		int minY, maxY;
		get_max_min_y(pcInput, ref(maxY), ref(minY));

		pcl::PointCloud<pcl::PointXYZ>::Ptr prev_y_slice(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr mpy_above_slice(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr mpy_slice(new pcl::PointCloud<pcl::PointXYZ>);

		int i = 0;
		for (int cy = minY; cy < maxY; cy += y_slice_height)
		{
			if (i > max_slice_index) break;

			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(pcInput, cy, cy + y_slice_height, "y", slice_);
			
			if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".ply", *slice_);
			if (val)
			{
				if (slice_->points.size() > mpy_slice->points.size())
				{
					pcl::copyPointCloud(*slice_, *mpy_slice);
					pcl::copyPointCloud(*prev_y_slice, *mpy_above_slice);
				}
			}

			if (i > 0) pcl::copyPointCloud(*slice_, *prev_y_slice);
			i++;
		}

		if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_maxYslice_pc_" + std::to_string(mpy_slice->points.size()) + ".ply", *mpy_slice);
		if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_above_maxYslice_pc_" + std::to_string(mpy_above_slice->points.size()) + ".ply", *mpy_above_slice);

		//Combine two slices.
		pcl::PointCloud<pcl::PointXYZ>::Ptr combined_slice(new pcl::PointCloud<pcl::PointXYZ>);
		*combined_slice = *mpy_slice + *mpy_above_slice;

		if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_combined_pc_" + std::to_string(combined_slice->points.size()) + ".ply", *combined_slice);

		//slice through X, Y to get most points slice (30).
		pcl::PointCloud<pcl::PointXYZ>::Ptr mpx_slice(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr mpz_slice(new pcl::PointCloud<pcl::PointXYZ>);

		int mp_slice_width = 30;
		int minX, maxX, minZ, maxZ;
		get_max_min_x(combined_slice, ref(maxX), ref(minX));
		get_max_min_z(combined_slice, ref(maxZ), ref(minZ));

		int ix = 0;
		for (int cx = minX; cx < maxX; cx += 10)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(combined_slice, cx, cx + mp_slice_width, "x", slice_);
			if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_slice_x_i_" + std::to_string(ix) + "pc_" + std::to_string(slice_->points.size()) + ".ply", *slice_);
			if (val)
			{
				if (slice_->points.size() > mpx_slice->points.size())
				{
					pcl::copyPointCloud(*slice_, *mpx_slice);
				}
			}
			ix++;
		}

		int iz = 0;
		for (int cz = minZ; cz < maxZ; cz += 10)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(combined_slice, cz, cz + mp_slice_width, "z", slice_);
			if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_slice_z_i_" + std::to_string(iz) + "pc_" + std::to_string(slice_->points.size()) + ".ply", *slice_);
			if (val)
			{
				if (slice_->points.size() > mpz_slice->points.size())
				{
					pcl::copyPointCloud(*slice_, *mpz_slice);
				}
			}
			iz++;
		}

		if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_mpx_pc_" + std::to_string(mpx_slice->points.size()) + ".ply", *mpx_slice);
		if (save_path != std::string("")) pcl::io::savePLYFileBinary(save_path + "_cone_mpz_pc_" + std::to_string(mpz_slice->points.size()) + ".ply", *mpz_slice);

		int combined_minY, combined_maxY;
		get_max_min_y(combined_slice, ref(combined_maxY), ref(combined_minY));

		//get minZ from mpx_slice.
		int mpx_minZ, mpx_maxZ;
		get_max_min_z(mpx_slice, ref(mpx_maxZ), ref(mpx_minZ));
		
		cone_position.z = mpx_maxZ;

		//get minX from mpz_slice for LEFT SENSOR.
		//get maxX from mpz_slice for RIGHT SENSOR.
		int mpz_minX, mpz_maxX;
		get_max_min_x(mpz_slice, ref(mpz_maxX), ref(mpz_minX));

		if (isLeftSide) cone_position.x = mpz_minX;
		else cone_position.x = mpz_maxX;

		cone_position.y = combined_minY + (combined_maxY - combined_minY) / 2;
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

void ZedX_Processing_Test()
{
	logMessage("ZedX Processing Test!");
	//Get Hole and Cone extracted data,
	std:string data_path = "I:/LTSP/20250317_LTSP_ZEDX_40FT_normal_batch_Results/FLZ";

	//load .jpg, .ply files for processing
	auto hole_filePath = data_path + "/" + std::string("Hole_Original");
	logMessage(hole_filePath);
	auto cone_filePath = data_path + "/" + std::string("Cone_Original");

	std::filesystem::path pathObj = std::filesystem::path(data_path).lexically_normal();
	std::string lastDirectory = pathObj.filename().string();

	auto hole_files = getAllFiles(hole_filePath, ".ply");
	logMessage(std::to_string(hole_files.size()));
	auto cone_files = getAllFiles(cone_filePath, ".ply");

	std::string save_path = data_path;

	auto logHeader = std::string("Filename;Hole_pc;Cone_pc;Guide_pc;hole_x;hole_y;hole_z;cone_x;cone_y;cone_z;devout_pca_x;devout_pca_y;devout_pca_z;landout_pca");

	ofstream Simfile;
	std::string sim_result_txt = save_path + "/Hole/BatchResult.txt";
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

	int fileLen = hole_files.size();
	for (int i = 0; i < fileLen; i++)
	{
		std::string hole_file = hole_files[i];
		std::filesystem::path pathObj(hole_file);
		// Get the filename with extension
		std::string filename = pathObj.stem().string();
		std::string current_hole_save_path = save_path + "/Hole/" + filename;
		std::string current_cone_save_path = save_path + "/Cone/" + filename;
		std::string log_lines = filename + ";";
		//load data.
		pcl::PointCloud<pcl::PointXYZ>::Ptr pcHole(new pcl::PointCloud <pcl::PointXYZ>);
		if (pcl::io::loadPLYFile<pcl::PointXYZ>(hole_file, *pcHole) == -1)
		{
			std::cout << std::string("Couldn't read file by PLY ") << std::endl;
			//return false;
		}
		std::string cone_file = cone_files[i];
		pcl::PointCloud<pcl::PointXYZ>::Ptr pcCone(new pcl::PointCloud <pcl::PointXYZ>);
		if (pcl::io::loadPLYFile<pcl::PointXYZ>(cone_file, *pcCone) == -1)
		{
			std::cout << std::string("Couldn't read file by PLY ") << std::endl;
			//return false;
		}

		//base data
		pcl::io::savePLYFileBinary(current_hole_save_path + "_pcHole.ply", *pcHole);
		//base data
		pcl::io::savePLYFileBinary(current_cone_save_path + "_pcCone.ply", *pcCone);

		//noisefilter
		pcl::PointCloud<pcl::PointXYZ>::Ptr nf_hole(new pcl::PointCloud<pcl::PointXYZ>);
		nf_hole = open3d_NoiseFilter(pcHole, 30, 1.0);
		pcl::io::savePLYFileBinary(current_hole_save_path + "_pcNFHole.ply", *nf_hole);

		pcl::PointCloud<pcl::PointXYZ>::Ptr nf_cone(new pcl::PointCloud<pcl::PointXYZ>);
		nf_cone = open3d_NoiseFilter(pcCone, 30, 1.0);
		pcl::io::savePLYFileBinary(current_cone_save_path + "_pcNFCole.ply", *nf_cone);

		log_lines += std::to_string(pcHole->points.size()) + ";" + std::to_string(pcCone->points.size()) + "; 0; ";
		
		//hole processing.
		auto hole_pos = pc_hole_detection_zedx(nf_hole, true, current_hole_save_path);
		pcl::PointCloud<pcl::PointXYZ>::Ptr ptHole(new pcl::PointCloud <pcl::PointXYZ>);
		ptHole->points.push_back(hole_pos);
		pcl::io::savePLYFileBinary(current_hole_save_path + "_hole_pos.ply", *ptHole);
		log_lines += std::to_string(hole_pos.x) + ";" + std::to_string(hole_pos.y) + ";" + std::to_string(hole_pos.z) + ";";
		
		//cone processing.
		auto cone_pos = pc_cone_detection_zedx(nf_cone, true, current_cone_save_path);
		pcl::PointCloud<pcl::PointXYZ>::Ptr ptCone(new pcl::PointCloud <pcl::PointXYZ>);
		ptCone->points.push_back(cone_pos);
		pcl::io::savePLYFileBinary(current_cone_save_path + "_cone_pos.ply", *ptCone);
		log_lines += std::to_string(cone_pos.x) + ";" + std::to_string(cone_pos.y) + ";" + std::to_string(cone_pos.z) + ";";

		log_lines += "0;0;0;0;";
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

void handleClient(SOCKET clientSocket) {
	logMessage("Client connected. Sending initial message...");

	char buffer[DEFAULT_RECVLEN] = { 0 };
	while (socket_running) 
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

	while (socket_running) 
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
			std::cerr << "Socket creation failed\n";
			std::this_thread::sleep_for(std::chrono::seconds(2));
			continue;
		}

		sockaddr_in server_addr{};
		server_addr.sin_family = AF_INET;
		server_addr.sin_port = htons(server_port);
		inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr);

		std::cout << "Attempting to connect to " << server_ip << ":" << server_port << "...\n";
		if (connect(sock, (sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
			std::cerr << "Connection failed: " << WSAGetLastError() << "\n";
			closesocket(sock);
			std::this_thread::sleep_for(std::chrono::seconds(2));
			continue;
		}

		std::cout << "Connected to server!\n";

		//initial send?
		char sendbuf_init[DEFAULT_SENDLEN] = { 0 };
		//makeCommand(ref(sendbuf));
		int bytesSent_init = send(sock, sendbuf_init, DEFAULT_SENDLEN, 0);

		// Sample loop: Read from server
		char buffer[DEFAULT_RECVLEN] = { 0 };
		while (socket_running) {
			int bytesReceived = recv(sock, buffer, sizeof(buffer), 0);
			if (bytesReceived <= 0) {
				std::cerr << "Disconnected from server\n";
				break;
			}
			//buffer[bytesReceived] = '\0';
			std::string received_msg(buffer);
			//std::cout << "Received: " << buffer << "\n";

			parseCommand(buffer);

			std::this_thread::sleep_for(std::chrono::milliseconds(100));

			char sendbuf[DEFAULT_SENDLEN] = { 0 };
			makeCommand(ref(sendbuf));

			int bytesSent = send(sock, sendbuf, DEFAULT_SENDLEN, 0);
			if (bytesSent <= 0)
			{
				logMessage("Sending data failed! " + std::to_string(bytesSent));
				//std::cerr << "Send failed: " << strerror(errno) << "\n";
				break;
			}

			//std::cout << "Sent: " << sendbuf << "\n";
		}

		closesocket(sock);
		logMessage("Connection closed. Retrying in 2 seconds...");
		std::this_thread::sleep_for(std::chrono::seconds(2)); // Wait before retry
	}

	WSACleanup();
}

bool runStreamingDemo(const char ipAddress[], unsigned short dataPort, uint32_t numberOfFrames, bool executeExtTrigger)
{
	using namespace visionary;

	// Generate Visionary instance
	auto pDataHandler = std::make_shared<VisionaryTMiniData>();
	VisionaryDataStream dataStream(pDataHandler);
	VisionaryControl visionaryControl;

	std::printf("Made samples.\n");
	std::cout << ipAddress << " ;;; " << dataPort << std::endl;

	//-----------------------------------------------
	// Connect to devices data stream 
	if (!dataStream.open(ipAddress, htons(dataPort)))
	{
		std::printf("Failed to open data stream connection to device.\n");
		return false;   // connection failed
	}

	std::printf("Data stream opened.\n");

	//-----------------------------------------------
	// Connect to devices control channel
	if (!visionaryControl.open(VisionaryControl::ProtocolType::COLA_2, ipAddress, 5000/*ms*/))
	{
		std::printf("Failed to open control connection to device.\n");
		return false;   // connection failed
	}

	std::printf("Control stream opened.\n");


	//-----------------------------------------------
	// read Device Ident
	std::printf("DeviceIdent: '%s'\n", visionaryControl.getDeviceIdent().c_str());

	//-----------------------------------------------
	// Login as authorized client
	if (visionaryControl.login(IAuthentication::UserLevel::AUTHORIZED_CLIENT, "CLIENT"))
	{
		//-----------------------------------------------
		// An example of reading an writing device parameters is shown here.
		// Use the "SOPAS Communication Interface Description" PDF to determine data types for other variables
		//-----------------------------------------------
		// Set enDepthMask parameter to false

		std::printf("Setting enDepthMask to false\n");
		CoLaCommand setEnDepthMaskCommand = CoLaParameterWriter(CoLaCommandType::WRITE_VARIABLE, "enDepthMask").parameterBool(false).build();
		CoLaCommand setEnDepthMaskResponse = visionaryControl.sendCommand(setEnDepthMaskCommand);
		if (setEnDepthMaskResponse.getError() == CoLaError::OK)
		{
			std::printf("Successfully set enDepthMask to false\n");
		}


		//-----------------------------------------------
		// Read humidity parameter
		CoLaCommand getHumidity = CoLaParameterWriter(CoLaCommandType::READ_VARIABLE, "humidity").build();
		CoLaCommand humidityResponse = visionaryControl.sendCommand(getHumidity);
		const double humidity = CoLaParameterReader(humidityResponse).readLReal();
		std::printf("Read humidity = %f\n", humidity);

		//-----------------------------------------------
		// Read info messages variable
		CoLaCommand getMessagesCommand = CoLaParameterWriter(CoLaCommandType::READ_VARIABLE, "MSinfo").build();
		CoLaCommand messagesResponse = visionaryControl.sendCommand(getMessagesCommand);

		//-----------------------------------------------
	}

	{
		CoLaCommand setEnDepthMaskCommand = CoLaParameterWriter(CoLaCommandType::WRITE_VARIABLE, "enDepthMask").parameterBool(true).build();
		CoLaCommand setEnDepthMaskResponse = visionaryControl.sendCommand(setEnDepthMaskCommand);
		if (setEnDepthMaskResponse.getError() != CoLaError::OK)
		{
			std::printf("Failed to set enDepthMask to true\n");
		}
	}

	//-----------------------------------------------
	// Logout from device after reading variables.
	if (!visionaryControl.logout())
	{
		std::printf("Failed to logout\n");
	}

	//-----------------------------------------------
	// Stop image acquisition (works always, also when already stopped)
	visionaryControl.stopAcquisition();

	//-----------------------------------------------
	// Capture a single frame
	visionaryControl.stepAcquisition();
	if (dataStream.getNextFrame())
	{
		std::printf("Frame received through step called, frame #%d, timestamp: %u \n", pDataHandler->getFrameNum(), pDataHandler->getTimestampMS());

		//-----------------------------------------------
		// Convert data to a point cloud
		std::vector<PointXYZ> pointCloud;
		pDataHandler->generatePointCloud(pointCloud);
		pDataHandler->transformPointCloud(pointCloud);



		//-----------------------------------------------
		// Write point cloud to PLY
		const char plyFilePath[] = "VisionaryT.ply";
		std::printf("Writing frame to %s\n", plyFilePath);
		PointCloudPlyWriter::WriteFormatPLY(plyFilePath, pointCloud, pDataHandler->getIntensityMap(), true);
		std::printf("Finished writing frame to %s\n", plyFilePath);
	}

	//-----------------------------------------------
	// Start image acquisiton and continously receive frames
	visionaryControl.startAcquisition();
	for (uint32_t i = 0; i < numberOfFrames; i++)
	{
		if (!dataStream.getNextFrame())
		{
			continue;     // No valid frame received
		}
		std::printf("Frame received in continuous mode, frame #%d \n", pDataHandler->getFrameNum());
		std::vector<uint16_t> intensityMap = pDataHandler->getIntensityMap();
		std::vector<uint16_t> distanceMap = pDataHandler->getDistanceMap();
		std::vector<uint16_t> stateMap = pDataHandler->getStateMap();
	}

	//-----------------------------------------------
	// This part of the sample code is skipped by default because not every user has a working IO trigger hardware available. 
	// If you want to execute it set variable "executeExtTrigger" in main function to "true".
	if (executeExtTrigger)
	{
		// Capture single frames with external trigger
		// NOTE: This part of the sample only works if you have a working rising egde signal on IO1 which triggers an image!
		std::printf("\n=== Starting external trigger example: \n");
		// Login as authorized client
		if (visionaryControl.login(IAuthentication::UserLevel::AUTHORIZED_CLIENT, "CLIENT"))
		{
			// Set frontendMode to STOP (= 1)
			std::printf("Setting frontendMode to STOP (= 1)\n");
			CoLaCommand setFrontendModeCommand = CoLaParameterWriter(CoLaCommandType::WRITE_VARIABLE, "frontendMode").parameterUSInt(1).build();
			CoLaCommand setFrontendModeResponse = visionaryControl.sendCommand(setFrontendModeCommand);
			if (setFrontendModeResponse.getError() != CoLaError::OK)
			{
				std::printf("Failed to set frontendMode to STOP (= 1)\n");
			}

			// Set INOUT1_Function to Trigger (= 7)
			std::printf("Setting DIO1Fnc to Trigger (= 7)\n");
			CoLaCommand setDIO1FncCommand = CoLaParameterWriter(CoLaCommandType::WRITE_VARIABLE, "DIO1Fnc").parameterUSInt(7).build();
			CoLaCommand setDIO1FncResponse = visionaryControl.sendCommand(setDIO1FncCommand);
			if (setDIO1FncResponse.getError() != CoLaError::OK)
			{
				std::printf("Failed to set DIO1Fnc to Trigger (= 7)\n");
			}

			// Set INOUT2_Function to TriggerBusy (= 23)
			std::printf("Setting DIO2Fnc to TriggerBusy (= 23)\n");
			CoLaCommand setDIO2FncCommand = CoLaParameterWriter(CoLaCommandType::WRITE_VARIABLE, "DIO2Fnc").parameterUSInt(23).build();
			CoLaCommand setDIO2FncResponse = visionaryControl.sendCommand(setDIO2FncCommand);
			if (setDIO2FncResponse.getError() != CoLaError::OK)
			{
				std::printf("Failed to set DIO2Fnc to TriggerBusy (= 23)\n");
			}
		}

		// Re-Connect to device data stream (make sure there are no old images in the pipeline)
		dataStream.close();
		std::this_thread::sleep_for(std::chrono::seconds(1)); // This short deelay is necessary to not have any old frames in the pipeline.
		if (!dataStream.open(ipAddress, htons(dataPort)))
		{
			std::printf("Failed to open data stream connection to device.\n");
			return false;   // connection failed
		}

		std::printf("Please enable trigger on IO1 to receive an image: \n");
		long long startTime = std::chrono::system_clock::now().time_since_epoch().count();
		long long timeNow = startTime;

		// Limited time loop for receiving hardware trigger signals on IO1 pin
		bool frameReceived = false;
		long long triggerTimeOut = 100000000; // 10 sec = 100 000 000
		while ((timeNow - startTime) <= triggerTimeOut) {
			// Read variable IOValue
			CoLaCommand getIOValue = CoLaParameterWriter(CoLaCommandType::READ_VARIABLE, "IOValue").build();
			CoLaCommand IOValueResponse = visionaryControl.sendCommand(getIOValue);
			CoLaParameterReader IOValues(IOValueResponse);
			const int8_t IOValue1 = IOValues.readSInt();
			const int8_t IOValue2 = IOValues.readSInt(); // We need the IOValue of IO2 from the V3SIOsState struct
			std::printf("Read TriggerBusy = %d\n", IOValue2);

			// Receive the next frame
			if (IOValue2 == 0)
			{
				if (dataStream.getNextFrame())
				{
					std::printf("Frame received in external trigger mode, frame #%d \n", pDataHandler->getFrameNum());
					frameReceived = true;
				}
				timeNow = std::chrono::system_clock::now().time_since_epoch().count();
			}
		}

		if (frameReceived == false)
		{
			std::printf("TIMEOUT: No trigger signal received on IO1 within %.2f seconds!\n", (float)triggerTimeOut / 10000000);
		}
	}
	//-----------------------------------------------

	visionaryControl.close();
	dataStream.close();
	return true;
}

void rtsp_rgb_receiver()
{
	try
	{
		while (camRGB_ctrl_running.load())
		{
			std::unique_lock<std::mutex> lock(mutex_camRGB_ctrl);
			bool res = cond_camRGB_ctrl.wait_for(lock,
				std::chrono::seconds(3600),
				[]() { return camRGB_ctrl_flag; });

			if (res)
			{
				//open rtsp stream and start capturing. 
				//rtsp://192.168.15.230:554/live/0
				std::string rtsp_url = "rtsp://" + current_lane_cam_ip + ":554/live/0";
				camRGB_cap.open(rtsp_url, cv::CAP_FFMPEG);

				cv::Mat frame;
				while (camRGB_stream_flag) {
					// Grab a frame
					bool isFrameGrabbed = camRGB_cap.grab();
					if (!isFrameGrabbed) {
						std::cout << "Error: Cannot grab frame from RTSP stream." << std::endl;
						break;
					}

					// Retrieve and process the frame
					camRGB_cap.retrieve(frame);

					//auto res = dataStackRGB.Update_Stack(frame,last_frame_get_time);

					// Display the frame
					cv::imshow("RTSP Stream", frame);
					cv::waitKey(1);
					

				}
			}
		}
	}
	catch (std::exception& ex)
	{
		logMessage("[camRGB-Control] " + std::string(ex.what()));
	}
	catch (...)
	{
		logMessage("[camRGB-Control] Unknown Exception!");
	}
}

int rtsp_stream_receiver() {
	// Replace with your RTSP stream URL
	//"rtsp://admin:seoho098@192.168.0.163:554/Streaming/channels/101"
	//"rtsp://192.168.15.230:554/live/0"
	std::string rtsp_url = "rtsp://admin:seoho098@192.168.0.163:554";
	cv::VideoCapture capture(rtsp_url, cv::CAP_FFMPEG);

	if (!capture.isOpened()) {
		std::cout << "Error: Cannot open the RTSP stream." << std::endl;
		return -1;
	}

	cv::Mat frame;
	while (true) {
		// Grab a frame
		bool isFrameGrabbed = capture.grab();
		if (!isFrameGrabbed) {
			std::cout << "Error: Cannot grab frame from RTSP stream." << std::endl;
			break;
		}

		// Retrieve and process the frame
		capture.retrieve(frame);

		// Display the frame
		cv::imshow("RTSP Stream", frame);

		// Exit the loop on pressing 'q'
		if (cv::waitKey(30) == 'q') {
			break;
		}
	}

	capture.release();
	cv::destroyAllWindows();
	return 0;
}

int main(int argc, char* argv[])
{
	SetUnhandledExceptionFilter(MyUnhandledExceptionFilter);

	printf("Visionary T-Mini Logging App\n");

	parseAppName();

	appName = appID + "_v" + program_version;

	std::cout << "App: " << appName << "\n";

	std::thread logThread(logWriterThread);

	logMessage(appName + " Starting!");

	signal(SIGINT, my_handler);
	//std::set_terminate(unHandledExceptionHandler);

	parseINI();
	parseProcessINI();
	parseIPINI();
	
	//printf("Visionary T Mini App\n");
	VA detector;
	model_loaded = detector.load_onnx_model(s2ws(onnx_model_path));
	model_loaded ? logMessage("Model Loaded") : logMessage("Model Failed to Load");

	std::string initialize_image_path = "sample.jpg";
	cv::Mat initialize_image = cv::imread(initialize_image_path);
	std::vector<rectangle_info> initialize_results;
	model_initialized = detector.onnx_inference(initialize_image, initialize_results);
	model_initialized ? logMessage("Model Initialized") : logMessage("Model Failed to Initialized");
	
	//std::cout << "Build information:\n" << cv::getBuildInformation() << std::endl;

	if (DEBUG_WITH_FILES || DEBUG_BATCH_JOB)
	{
		//test();

		if (DEBUG_BATCH_JOB) OfflineDebugBatchProcessingThread(ref(detector));

		//std::thread rtspT(rtsp_stream_receiver);
		//rtspT.join();
		//if (DEBUG_BATCH_JOB) ZedX_Processing_Test();
	}
	else if (DEBUG_MODE)
	{
		logMessage("Debug Mode running.. Testing for onnx.");

		//CURRENT_SENSOR_POSITION = "REAR_LEFT";

		DEBUG_IMG_PATH = DEBUG_SAMPLE_JOB + "/Image";
		DEBUG_PLY_PATH = DEBUG_SAMPLE_JOB + "/Depth";

		DEBUG_IMG_FILES = getAllFiles(DEBUG_IMG_PATH, ".jpg");
		DEBUG_PLY_FILES = getAllFiles(DEBUG_PLY_PATH, ".ply");

		DEBUG_MAX_INDEX = (DEBUG_IMG_FILES.size() < DEBUG_PLY_FILES.size()) ? DEBUG_IMG_FILES.size() : DEBUG_PLY_FILES.size();
		DEBUG_CURRENT_INDEX = 0;

		/*
		demo_img = cv::imread("20250302_041440_568_TMini_Image.jpg", cv::IMREAD_COLOR);

		pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud = std::make_shared<pcl::PointCloud <pcl::PointXYZ>>();
		if (pcl::io::loadPLYFile<pcl::PointXYZ>("20250302_041440_568_TMini_Depth.ply", *pointCloud) == -1)
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

		std::thread tProc(processingThread, std::ref(detector));
		std::thread tSave(data_save_thread);

		tProc.join();
		joblogThread.join();
		sckTMini_Setup.join();
		sckTMini_Data_Stream.join();
		tSave.join();
		//sckT.join();
	}
	else if (SEQUENTIAL_PROCESSING)
	{
		logMessage("Sequential Processing Mode running..");
		std::thread sckT(start_server);
		std::thread joblogThread(jobLogWriterThread);
		std::thread tProc(Sequential_Processing_Thread, std::ref(detector));
		std::thread tSave(data_save_thread);

		tProc.join();
		//joblog_running.store(false);
		cvJobLog.notify_one();
		joblogThread.join();
		//sckTMini_Setup.join();
		//sckTMini_Data_Stream.join();
		tSave.join();
		sckT.join();
	}
	else
	{
		std::thread sckT(start_server);
		//std::thread sckClientThread(socketClient, SOCKET_IP, SOCKET_PORT);

		//Visionary T Mini Streaming setup.
		std::thread sckTMini_Setup(thread_tmini_control);
		std::thread sckTMini_Data_Stream(thread_tmini_data_stream);

		std::thread joblogThread(jobLogWriterThread);

		std::thread tProc(processingThread, std::ref(detector));

		std::thread tSave(data_save_thread);

		tProc.join();

		//job_log_running.store(false);
		cvJobLog.notify_one();
		joblogThread.join();

		sckTMini_Setup.join();
		sckTMini_Data_Stream.join();
		tSave.join();
		sckT.join();
		//sckClientThread.join();
	}

	logMessage("Program Shutting Down...");

	log_running.store(false);
	cvLog.notify_one();
	logThread.join();

	//std::cout << "Press any key to continue...";
	//system("pause"); // Windows only

	return true;
}