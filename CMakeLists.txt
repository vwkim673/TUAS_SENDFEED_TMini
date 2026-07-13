#
# Copyright note: Redistribution and use in source, with or without modification, are permitted.
# 
# Created: November 2018
# 
# SICK AG, Waldkirch
# email: TechSupport0905@sick.de

cmake_minimum_required(VERSION 3.8)

project(SampleVisionary
        VERSION 0.1.0
        LANGUAGES CXX)

### COMPILER FLAGS ###
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
  # We support only LONGHORN and above (WIN7...)
  add_compile_definitions("_WIN32_WINNT=_WIN32_WINNT_LONGHORN")
endif()

### BUILD ###
set(SICK_VISIONARY_CPP_SHARED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sick_visionary_cpp_shared")
if(EXISTS "${SICK_VISIONARY_CPP_SHARED_DIR}/CMakeLists.txt")
  add_subdirectory(sick_visionary_cpp_shared)
else()
  message(FATAL_ERROR
    "Missing 'sick_visionary_cpp_shared' directory.\n"
    "Extract 'sick_visionary_cpp_shared.7z' into:\n"
    "  ${SICK_VISIONARY_CPP_SHARED_DIR}\n"
    "so it contains a CMakeLists.txt."
  )
endif()

## Visionary-S sample ##
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/SampleVisionaryS/SampleVisionaryS.cpp")
  add_executable(SampleVisionaryS SampleVisionaryS/SampleVisionaryS.cpp)
  target_link_libraries(SampleVisionaryS sick_visionary_cpp_shared)
else()
  message(STATUS "Skipping SampleVisionaryS (sources not present).")
endif()

## Auto-IP scan sample ##
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/SampleAutoIP/SampleAutoIPScan.cpp")
  add_executable(SampleAutoIPScan SampleAutoIP/SampleAutoIPScan.cpp)
  target_link_libraries(SampleAutoIPScan sick_visionary_cpp_shared)
else()
  message(STATUS "Skipping SampleAutoIPScan (sources not present).")
endif()

## Visionary-T sample ##
add_executable(SampleVisionaryT SampleVisionaryT/SampleVisionaryT.cpp)
target_link_libraries(SampleVisionaryT sick_visionary_cpp_shared)

## Visionary-T VGA samples ##
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/SampleVisionaryTVGA/SampleVisionaryTVGA.cpp")
  add_executable(SampleVisionaryTVGA SampleVisionaryTVGA/SampleVisionaryTVGA.cpp)
  target_link_libraries(SampleVisionaryTVGA sick_visionary_cpp_shared)
else()
  message(STATUS "Skipping SampleVisionaryTVGA (sources not present).")
endif()

## Visionary-T Mini samples ##
add_executable(SampleVisionaryTMini
  SampleVisionaryTMini/SampleVisionaryTMini.cpp
  SampleVisionaryTMini/TrtRunner.cpp
  SampleVisionaryTMini/TrtRunner.h
  SampleVisionaryTMini/VA.cpp
  SampleVisionaryTMini/VA.h
  SampleVisionaryTMini/ChassisRoi.h
)
target_link_libraries(SampleVisionaryTMini sick_visionary_cpp_shared)

