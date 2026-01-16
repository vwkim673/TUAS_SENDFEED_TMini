/*
Class: DataStack

Function: Data Frame storage to keep most up-to-date data.
*/

#include <iostream>
#include <tuple>
#include <mutex>
#include <vector>
//#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>

//#include "VisionaryControl.h"
//#include "CoLaParameterReader.h"
//#include "CoLaParameterWriter.h"
//#include "VisionaryTMiniData.h"    // Header specific for the Time of Flight data
//#include "VisionaryDataStream.h"
#include "PointXYZ.h"

#include <pcl/pcl_base.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>


using namespace std;
using namespace visionary;

auto print_time2() -> std::string
{
    auto now = std::chrono::system_clock::now();

    // Convert to time_t to use with std::localtime
    std::time_t current_time = std::chrono::system_clock::to_time_t(now);

    // Convert to tm struct (local time)
    std::tm local_time = *std::localtime(&current_time);

    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S");

    // Get the formatted time as a string
    std::string formatted_time = oss.str();

    // Output formatted current time
    return formatted_time;
}

class DataStack
{
    public:
        DataStack() = default;
        ~DataStack() = default;

        //Update
        //returns true if new data is updated to stack.
        auto Update_Stack(cv::Mat image, pcl::PointCloud<pcl::PointXYZ> pointcloud, std::chrono::system_clock::time_point ts) -> bool
        {
            bool update_status = false;

            //last updated is 1.
            if (newStack_i == 1)
            {
                //If currently data is retrieving data from stack2,
                if (isReading2)
                {
                    //Update1 again.make_tuple
                    try
                    {
                        std::lock_guard<std::mutex> lock(mtx1);
                        try
                        {
                            isWriting1 = true;

                            //Copy data to objects, then to tuple.
                            cv::Mat stack_image = image.clone(); // Use clone to avoid dangling reference
                            pcl::PointCloud<pcl::PointXYZ> stack_pointcloud;
                            pcl::copyPointCloud(pointcloud, stack_pointcloud); // Use copy to avoid dangling reference

                            //Create tuple with data.
                            d_tuple1 = std::make_tuple(image, pointcloud, ts);

                            isUsed1 = false;
                            isUpdated1 = true;

                            newStack_i = 1;

                            update_status = true;
                            isAvailable1 = true;
                            isEmpty1 = false;
                            //std::cout << "Updated Stack1 because stack2 is being read : " << std::to_string(newStack_i) << std::endl;

                            isWriting1 = false;
                        }
                        catch(const std::exception& e)
                        {
                            std::cerr << e.what() << '\n';
                            update_status = false;
                        }
                    }
                    catch (system_error& e)
                    {
                        //deadlock?
                        std::cerr << e.what() << '\n';
                        update_status = false;
                    }
                    catch (...)
                    {
                        //deadlock unknown error
                        std::cout << print_time2() << " unknown error from write mutex lock" << std::endl;
                        update_status = false;
                    }   
                }
                else
                {
                    //Update2.
                    try
                    {
                        std::lock_guard<std::mutex> lock(mtx2);
                        try
                        {
                            isWriting2 = true;

                            //Copy data to objects, then to tuple.
                            cv::Mat stack_image = image.clone(); // Use clone to avoid dangling reference
                            pcl::PointCloud<pcl::PointXYZ> stack_pointcloud;
                            pcl::copyPointCloud(pointcloud, stack_pointcloud); // Use copy to avoid dangling reference

                            //Create tuple with data.
                            d_tuple2 = std::make_tuple(image, pointcloud, ts);

                            isUsed2 = false;
                            isUpdated2 = true;

                            newStack_i = 2;

                            update_status = true;
                            isAvailable2 = true;
                            isEmpty2 = false;
                            //std::cout << "Updated Stack2 : " << std::to_string(newStack_i) << std::endl;

                            isWriting2 = false;
                        }
                        catch(const std::exception& e)
                        {
                            std::cerr << e.what() << '\n';
							update_status = false;
                        }
                    }
                    catch (system_error& e)
                    {
                        //deadlock?
                        std::cerr << e.what() << '\n';
						update_status = false;
                    }
                    catch (...)
                    {
                        //deadlock unknown error
                        std::cout << print_time2() << " unknown error from write mutex lock" << std::endl;
                        update_status = false;
                    }
                }
            }
            else
            {
                if (isReading1)
                {
                    //Update2 again.
                    try
                    {
                        std::lock_guard<std::mutex> lock(mtx2);
                        try
                        {
                            isWriting2 = true;

                            //Copy data to objects, then to tuple.
                            cv::Mat stack_image = image.clone(); // Use clone to avoid dangling reference
                            pcl::PointCloud<pcl::PointXYZ> stack_pointcloud;
                            pcl::copyPointCloud(pointcloud, stack_pointcloud); // Use copy to avoid dangling reference

                            //Create tuple with data.
                            d_tuple2 = std::make_tuple(image, pointcloud, ts);

                            isUsed2 = false;
                            isUpdated2 = true;

                            newStack_i = 2;

                            update_status = true;
                            isAvailable2 = true;
                            isEmpty2 = false;
                            //std::cout << "Updated Stack2 because stack1 is being read : " << std::to_string(newStack_i) << std::endl;

                            isWriting2 = false;
                        }
                        catch(const std::exception& e)
                        {
                            std::cerr << e.what() << '\n';
							update_status = false;
                        }
                    }
                    catch (system_error& e)
                    {
                        //deadlock?
                        std::cerr << e.what() << '\n';
						update_status = false;
                    }
                    catch (...)
                    {
                        //deadlock unknown error
                        std::cout << "unknown error from write mutex lock" << std::endl;
						update_status = false;
                    }   
                }
                else
                {
                    //Update1.
                    try
                    {
                        std::lock_guard<std::mutex> lock(mtx1);
                        try
                        {
                            isWriting1 = true;

                            //Copy data to objects, then to tuple.
                            cv::Mat stack_image = image.clone(); // Use clone to avoid dangling reference
                            pcl::PointCloud<pcl::PointXYZ> stack_pointcloud;
                            pcl::copyPointCloud(pointcloud, stack_pointcloud); // Use copy to avoid dangling reference

                            //Create tuple with data.
                            d_tuple1 = std::make_tuple(image, pointcloud, ts);

                            isUsed1 = false;
                            isUpdated1 = true;

                            newStack_i = 1;

                            update_status = true;
                            isAvailable1 = true;
                            isEmpty1 = false;
                            //std::cout << "Updated Stack1 : " << std::to_string(newStack_i) << std::endl;

                            isWriting1 = false;
                        }
                        catch(const std::exception& e)
                        {
                            std::cerr << e.what() << '\n';
							update_status = false;
                        }
                    }
                    catch (system_error& e)
                    {
                        //deadlock?
                        std::cerr << e.what() << '\n';
						update_status = false;
                    }
                    catch (...)
                    {
                        //deadlock unknown error
                        std::cout << "unknown error from write mutex lock" << std::endl;
						update_status = false;
                    }
                }
            }

            return update_status;
        }

        //Retrieve
        //returns true if unused recent data is returned.
        //returns false if no valid recent data is returned.
        auto Retrieve_Stack(cv::Mat& image, pcl::PointCloud<pcl::PointXYZ>& pointcloud, std::chrono::system_clock::time_point& ts) -> bool
        {
            bool retrieve_status = false;

            if (newStack_i == 1)
            {
                if (isWriting1)
                {
                    //read from stack 2.
                    if (!isUsed2 && isUpdated2 && isAvailable2)
                    {
                        try
                        {
                            std::lock_guard<std::mutex> lg (mtx2);
                            try
                            {
                                isReading2 = true;

                                //Copy data to reference objects.
								cv::Mat stack_image = std::get<0>(d_tuple2);
								pcl::PointCloud<pcl::PointXYZ> stack_pointcloud = std::get<1>(d_tuple2);
								std::chrono::system_clock::time_point stack_ts = std::get<2>(d_tuple2);
								
                                //Assign to output parameters.
								image = stack_image.clone(); // Use clone to avoid dangling reference
								pcl::copyPointCloud(stack_pointcloud, pointcloud); // Use copy to avoid dangling reference
								ts = stack_ts;

                                //image = std::get<0>(d_tuple2);
                                //pointcloud = std::get<1>(d_tuple2);
                                //ts = std::get<2>(d_tuple2);

                                isUsed2 = true;
                                isUpdated2 = false;
                                retrieve_status = true;

                                //std::cout <<  print_time2() << " Reading from stack2" << std::endl;

                                isReading2 = false;
                            }
                            catch(const std::exception& e)
                            {
                                std::cerr << e.what() << '\n';
								retrieve_status = false;
                            } 
                        }
                        catch(const std::exception& e)
                        {
                            std::cerr << e.what() << '\n';
                            retrieve_status = false;
                        }
                    }
                    else
                    {
                        //std::cout << "Can't read from both stack 1 and 2" << std::endl;
                        //put empty data.
                        image = cv::Mat();
						pointcloud = pcl::PointCloud<pcl::PointXYZ>();
						ts = std::chrono::system_clock::now();
						retrieve_status = false;
                    }
                }
                else
                {
                    //Read from stack 1.
                    if (!isUsed1 && isUpdated1 && isAvailable1)
                    {
                        try
                        {
                            std::lock_guard<std::mutex> lg (mtx1);
                            try
                            {
                                isReading1 = true;

                                //Copy data to reference objects.
                                cv::Mat stack_image = std::get<0>(d_tuple1);
                                pcl::PointCloud<pcl::PointXYZ> stack_pointcloud = std::get<1>(d_tuple1);
                                std::chrono::system_clock::time_point stack_ts = std::get<2>(d_tuple1);

                                //Assign to output parameters.
                                image = stack_image.clone(); // Use clone to avoid dangling reference
                                pcl::copyPointCloud(stack_pointcloud, pointcloud); // Use copy to avoid dangling reference
                                ts = stack_ts;

                                isUsed1 = true;
                                isUpdated1 = false;
                                retrieve_status = true;

                                //std::cout <<  print_time2() << " Reading from stack1" << std::endl;

                                isReading1 = false;
                            }
                            catch(const std::exception& e)
                            {
                                std::cerr << e.what() << '\n';
                                retrieve_status = false;
                            } 
                        }
                        catch(const std::exception& e)
                        {
                            std::cerr << e.what() << '\n';
                            retrieve_status = false;
                        }
                    }
                    else
                    {
                        //std::cout << "Can't read from both stack 1 and 2" << std::endl;
                        //put empty data.
                        image = cv::Mat();
                        pointcloud = pcl::PointCloud<pcl::PointXYZ>();
                        ts = std::chrono::system_clock::now();
                        retrieve_status = false;
                    }
                }
            }
            else
            {
                 if (isWriting2)
                {
                    //read from stack 1.
                    if (!isUsed1 && isAvailable1)
                    {
                        try
                        {
                            std::lock_guard<std::mutex> lg (mtx1);
                            try
                            {
                                isReading1 = true;

                                //Copy data to reference objects.
                                cv::Mat stack_image = std::get<0>(d_tuple1);
                                pcl::PointCloud<pcl::PointXYZ> stack_pointcloud = std::get<1>(d_tuple1);
                                std::chrono::system_clock::time_point stack_ts = std::get<2>(d_tuple1);

                                //Assign to output parameters.
                                image = stack_image.clone(); // Use clone to avoid dangling reference
                                pcl::copyPointCloud(stack_pointcloud, pointcloud); // Use copy to avoid dangling reference
                                ts = stack_ts;

                                isUsed1 = true;
                                isUpdated1 = false;
                                retrieve_status = true;

                                //std::cout <<  print_time2() << " Reading from stack1" << std::endl;

                                isReading1 = false;
                            }
                            catch(const std::exception& e)
                            {
                                std::cerr << e.what() << '\n';
                                retrieve_status = false;
                            } 
                        }
                        catch(const std::exception& e)
                        {
                            std::cerr << e.what() << '\n';
                            retrieve_status = false;
                        }
                    }
                    else
                    {
                        //std::cout << "Can't read from both stack 1 and 2" << std::endl;
                        //put empty data.
                        image = cv::Mat();
                        pointcloud = pcl::PointCloud<pcl::PointXYZ>();
                        ts = std::chrono::system_clock::now();
                        retrieve_status = false;
                    }
                }
                else
                {
                    //Read from stack 2.
					 if (!isUsed2 && isAvailable2)
                    {
                        try
                        {
                            std::lock_guard<std::mutex> lg (mtx2);
                            try
                            {
                                isReading2 = true;

                                //Copy data to reference objects.
                                cv::Mat stack_image = std::get<0>(d_tuple2);
                                pcl::PointCloud<pcl::PointXYZ> stack_pointcloud = std::get<1>(d_tuple2);
                                std::chrono::system_clock::time_point stack_ts = std::get<2>(d_tuple2);

                                //Assign to output parameters.
                                image = stack_image.clone(); // Use clone to avoid dangling reference
                                pcl::copyPointCloud(stack_pointcloud, pointcloud); // Use copy to avoid dangling reference
                                ts = stack_ts;

                                isUsed2 = true;
                                isUpdated2 = false;
                                retrieve_status = true;

                                //std::cout <<  print_time2() << " Reading from stack2" << std::endl;

                                isReading2 = false;
                            }
                            catch(const std::exception& e)
                            {
                                std::cerr << e.what() << '\n';
                                retrieve_status = false;
                            } 
                        }
                        catch(const std::exception& e)
                        {
                            std::cerr << e.what() << '\n';
                            retrieve_status = false;
                        }
                    }
                    else
                    {
                         //std::cout << "Can't read from both stack 1 and 2" << std::endl;
                         //put empty data.
                         image = cv::Mat();
                         pointcloud = pcl::PointCloud<pcl::PointXYZ>();
                         ts = std::chrono::system_clock::now();
                         retrieve_status = false;
                    }
                }          
            }

            return retrieve_status;
        }

        auto Clear_Stack() -> void
        {
            d_tuple1 = std::tuple<cv::Mat, pcl::PointCloud<pcl::PointXYZ>, std::chrono::system_clock::time_point>();
            d_tuple2 = std::tuple<cv::Mat, pcl::PointCloud<pcl::PointXYZ>, std::chrono::system_clock::time_point>();

            isEmpty1 = true;
            isEmpty2 = true;
            isAvailable1 = false;
            isAvailable2 = false;
        }

        auto Stack_Ready() -> bool
        {
			if (isAvailable1 && isAvailable2) return true;
			else return false;	
        }

    private:
        bool isReading1 = false, isReading2 = false;
        bool isWriting1 = false, isWriting2 = false;
        bool isUsed1 = false, isUsed2 = false;
        bool isUpdated1 = false, isUpdated2 = false;
        bool isEmpty1 = false, isEmpty2 = false;
        int newStack_i = 1;
        bool isAvailable1 = false, isAvailable2 = false;
 
        std::tuple<cv::Mat, pcl::PointCloud<pcl::PointXYZ>, std::chrono::system_clock::time_point> d_tuple1 = std::tuple<cv::Mat, pcl::PointCloud<pcl::PointXYZ>, std::chrono::system_clock::time_point>();
        std::tuple<cv::Mat, pcl::PointCloud<pcl::PointXYZ>, std::chrono::system_clock::time_point> d_tuple2 = std::tuple<cv::Mat, pcl::PointCloud<pcl::PointXYZ>, std::chrono::system_clock::time_point>();

        std::mutex mtx1, mtx2;
};
class DataStack_RGB
{
public:
    DataStack_RGB() = default;
    ~DataStack_RGB() = default;

    //Update
    //returns true if new data is updated to stack.
    auto Update_Stack(cv::Mat image, std::chrono::system_clock::time_point ts) -> bool
    {
        bool update_status = false;
        std::tuple<cv::Mat, std::chrono::system_clock::time_point> dTuple = std::make_tuple(image,ts);

        //last updated is 1.
        if (newStack_i == 1)
        {
            //If currently data is retrieving data from stack2,
            if (isReading2)
            {
                //Update1 again.make_tuple
                try
                {
                    std::lock_guard<std::mutex> lock(mtx1);
                    try
                    {
                        isWriting1 = true;

                        d_tuple1 = dTuple;

                        isUsed1 = false;
                        isUpdated1 = true;

                        newStack_i = 1;

                        update_status = true;
                        //std::cout << "Updated Stack1 because stack2 is being read : " << std::to_string(newStack_i) << std::endl;

                        isWriting1 = false;
                    }
                    catch (const std::exception& e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                }
                catch (system_error& e)
                {
                    //deadlock?
                    std::cerr << e.what() << '\n';
                }
                catch (...)
                {
                    //deadlock unknown error
                    std::cout << print_time2() << " unknown error from write mutex lock" << std::endl;
                }
            }
            else
            {
                //Update2.
                try
                {
                    std::lock_guard<std::mutex> lock(mtx2);
                    try
                    {
                        isWriting2 = true;

                        d_tuple2 = dTuple;

                        isUsed2 = false;
                        isUpdated2 = true;

                        newStack_i = 2;

                        update_status = true;
                        //std::cout << "Updated Stack2 : " << std::to_string(newStack_i) << std::endl;

                        isWriting2 = false;
                    }
                    catch (const std::exception& e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                }
                catch (system_error& e)
                {
                    //deadlock?
                    std::cerr << e.what() << '\n';
                }
                catch (...)
                {
                    //deadlock unknown error
                    std::cout << print_time2() << " unknown error from write mutex lock" << std::endl;
                }
            }
        }
        else
        {
            if (isReading1)
            {
                //Update2 again.
                try
                {
                    std::lock_guard<std::mutex> lock(mtx2);
                    try
                    {
                        isWriting2 = true;

                        d_tuple2 = dTuple;

                        isUsed2 = false;
                        isUpdated2 = true;

                        newStack_i = 2;

                        update_status = true;
                        //std::cout << "Updated Stack2 because stack1 is being read : " << std::to_string(newStack_i) << std::endl;

                        isWriting2 = false;
                    }
                    catch (const std::exception& e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                }
                catch (system_error& e)
                {
                    //deadlock?
                    std::cerr << e.what() << '\n';
                }
                catch (...)
                {
                    //deadlock unknown error
                    std::cout << "unknown error from write mutex lock" << std::endl;
                }
            }
            else
            {
                //Update1.
                try
                {
                    std::lock_guard<std::mutex> lock(mtx1);
                    try
                    {
                        isWriting1 = true;

                        d_tuple1 = dTuple;

                        isUsed1 = false;
                        isUpdated1 = true;

                        newStack_i = 1;

                        update_status = true;
                        //std::cout << "Updated Stack1 : " << std::to_string(newStack_i) << std::endl;

                        isWriting1 = false;
                    }
                    catch (const std::exception& e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                }
                catch (system_error& e)
                {
                    //deadlock?
                    std::cerr << e.what() << '\n';
                }
                catch (...)
                {
                    //deadlock unknown error
                    std::cout << "unknown error from write mutex lock" << std::endl;
                }
            }
        }

        return update_status;
    }

    //Retrieve
    //returns true if unused recent data is returned.
    //returns false if no valid recent data is returned.
    auto Retrieve_Stack(cv::Mat& image, std::chrono::system_clock::time_point& ts) -> bool
    {
        bool retrieve_status = false;

        if (newStack_i == 1)
        {
            if (isWriting1)
            {
                //read from stack 2.
                if (!isUsed2)
                {
                    try
                    {
                        std::lock_guard<std::mutex> lg(mtx2);
                        try
                        {
                            isReading2 = true;

                            image = std::get<0>(d_tuple2);
                            ts = std::get<1>(d_tuple2);

                            isUsed2 = true;
                            retrieve_status = true;

                            //std::cout <<  print_time2() << " Reading from stack2" << std::endl;

                            isReading2 = false;
                        }
                        catch (const std::exception& e)
                        {
                            std::cerr << e.what() << '\n';
                        }
                    }
                    catch (const std::exception& e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                }
                else
                {
                    std::cout << "Can't read from both stack 1 and 2" << std::endl;
                }
            }
            else
            {
                //Read from stack 1.
                if (!isUsed1)
                {
                    try
                    {
                        std::lock_guard<std::mutex> lg(mtx1);
                        try
                        {
                            isReading1 = true;

                            image = std::get<0>(d_tuple1);
                            ts = std::get<1>(d_tuple1);

                            isUsed1 = true;
                            retrieve_status = true;

                            //std::cout <<  print_time2() << " Reading from stack1" << std::endl;

                            isReading1 = false;
                        }
                        catch (const std::exception& e)
                        {
                            std::cerr << e.what() << '\n';
                        }
                    }
                    catch (const std::exception& e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                }
                else
                {
                    ;
                }
            }
        }
        else
        {
            if (isWriting2)
            {
                //read from stack 1.
                if (!isUsed1)
                {
                    try
                    {
                        std::lock_guard<std::mutex> lg(mtx1);
                        try
                        {
                            isReading1 = true;

                            image = std::get<0>(d_tuple1);
                            ts = std::get<1>(d_tuple1);

                            isUsed1 = true;
                            retrieve_status = true;

                            //std::cout <<  print_time2() << " Reading from stack1" << std::endl;

                            isReading1 = false;
                        }
                        catch (const std::exception& e)
                        {
                            std::cerr << e.what() << '\n';
                        }
                    }
                    catch (const std::exception& e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                }
                else
                {
                    std::cout << "Can't read from both stack 1 and 2" << std::endl;
                }
            }
            else
            {
                //Read from stack 2.
                if (!isUsed2)
                {
                    try
                    {
                        std::lock_guard<std::mutex> lg(mtx2);
                        try
                        {
                            isReading2 = true;

                            image = std::get<0>(d_tuple2);
                            ts = std::get<1>(d_tuple2);

                            isUsed2 = true;
                            retrieve_status = true;

                            //std::cout <<  print_time2() << " Reading from stack2" << std::endl;

                            isReading2 = false;
                        }
                        catch (const std::exception& e)
                        {
                            std::cerr << e.what() << '\n';
                        }
                    }
                    catch (const std::exception& e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                }
                else
                {
                    ;
                }
            }
        }

        return retrieve_status;
    }

    auto Clear_Stack() -> void
    {
        d_tuple1 = std::tuple<cv::Mat, std::chrono::system_clock::time_point>();
        d_tuple2 = std::tuple<cv::Mat, std::chrono::system_clock::time_point>();

        isEmpty1 = true;
        isEmpty2 = true;
        isAvailable1 = false;
        isAvailable2 = false;
    }


private:
    bool isReading1 = false, isReading2 = false;
    bool isWriting1 = false, isWriting2 = false;
    bool isUsed1 = false, isUsed2 = false;
    bool isUpdated1 = false, isUpdated2 = false;
    bool isEmpty1 = true, isEmpty2 = true;
    int newStack_i = 1;
    bool isAvailable1 = false, isAvailable2 = false;

    std::tuple<cv::Mat, std::chrono::system_clock::time_point> d_tuple1 = std::tuple<cv::Mat, std::chrono::system_clock::time_point>();
    std::tuple<cv::Mat, std::chrono::system_clock::time_point> d_tuple2 = std::tuple<cv::Mat, std::chrono::system_clock::time_point>();

    std::mutex mtx1, mtx2;
};