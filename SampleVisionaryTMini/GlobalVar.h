#pragma once
//#include <iostream>

struct rectangle_info
{
	int batchNum = -1;

	int ClassInfo = -1;
	float X = 0.0f;
	float Y = 0.0f;
	float W = 0.0f;
	float H = 0.0f;

	float Prob = 0.0f;
	float Time = 0.0f;

	float Center_X = 0.0f;
	float Center_Y = 0.0f;

	std::string ClassName = std::string("");
};