#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <algorithm>
#include <limits>
#include <tuple>
#include <immintrin.h> // For AVX instructions if available

/**
 * @brief Simple optimized function to find min/max X,Y values in a point cloud
 * 
 * This version uses a single pass through the point cloud with direct comparisons.
 * 
 * @param cloud Input point cloud
 * @param minX Output minimum X value
 * @param maxX Output maximum X value
 * @param minY Output minimum Y value
 * @param maxY Output maximum Y value
 */
inline void getCloudXYRange(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    float& minX, float& maxX, float& minY, float& maxY
) {
    // Initialize with extreme values
    minX = std::numeric_limits<float>::max();
    maxX = std::numeric_limits<float>::lowest();
    minY = std::numeric_limits<float>::max();
    maxY = std::numeric_limits<float>::lowest();
    
    // Early check for empty cloud
    if (cloud->empty()) {
        return;
    }
    
    // Single pass through the point cloud
    for (const auto& point : cloud->points) {
        // Update X range
        if (point.x < minX) minX = point.x;
        if (point.x > maxX) maxX = point.x;
        
        // Update Y range
        if (point.y < minY) minY = point.y;
        if (point.y > maxY) maxY = point.y;
    }
}

/**
 * @brief Optimized function to find min/max X,Y values with filtering
 * 
 * This version filters out points outside specified Z range.
 * 
 * @param cloud Input point cloud
 * @param minX Output minimum X value
 * @param maxX Output maximum X value
 * @param minY Output minimum Y value
 * @param maxY Output maximum Y value
 * @param minZ Minimum Z value for filtering (optional)
 * @param maxZ Maximum Z value for filtering (optional)
 */
inline void getCloudXYRangeFiltered(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    float& minX, float& maxX, float& minY, float& maxY,
    float minZ = -std::numeric_limits<float>::max(),
    float maxZ = std::numeric_limits<float>::max()
) {
    // Initialize with extreme values
    minX = std::numeric_limits<float>::max();
    maxX = std::numeric_limits<float>::lowest();
    minY = std::numeric_limits<float>::max();
    maxY = std::numeric_limits<float>::lowest();
    
    // Early check for empty cloud
    if (cloud->empty()) {
        return;
    }
    
    // Single pass through the point cloud with Z filtering
    for (const auto& point : cloud->points) {
        // Skip points outside Z range
        if (point.z < minZ || point.z > maxZ) {
            continue;
        }
        
        // Update X range
        if (point.x < minX) minX = point.x;
        if (point.x > maxX) maxX = point.x;
        
        // Update Y range
        if (point.y < minY) minY = point.y;
        if (point.y > maxY) maxY = point.y;
    }
}

/**
 * @brief Optimized function to find min/max X,Y values using STL algorithms
 * 
 * This version uses std::minmax_element for potentially better optimization.
 * 
 * @param cloud Input point cloud
 * @param minX Output minimum X value
 * @param maxX Output maximum X value
 * @param minY Output minimum Y value
 * @param maxY Output maximum Y value
 */
inline void getCloudXYRangeSTL(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    float& minX, float& maxX, float& minY, float& maxY
) {
    // Initialize with extreme values
    minX = std::numeric_limits<float>::max();
    maxX = std::numeric_limits<float>::lowest();
    minY = std::numeric_limits<float>::max();
    maxY = std::numeric_limits<float>::lowest();
    
    // Early check for empty cloud
    if (cloud->empty()) {
        return;
    }
    
    // Find min/max X using STL algorithms
    auto x_minmax = std::minmax_element(
        cloud->points.begin(), cloud->points.end(),
        [](const pcl::PointXYZ& a, const pcl::PointXYZ& b) { return a.x < b.x; }
    );
    minX = x_minmax.first->x;
    maxX = x_minmax.second->x;
    
    // Find min/max Y using STL algorithms
    auto y_minmax = std::minmax_element(
        cloud->points.begin(), cloud->points.end(),
        [](const pcl::PointXYZ& a, const pcl::PointXYZ& b) { return a.y < b.y; }
    );
    minY = y_minmax.first->y;
    maxY = y_minmax.second->y;
}

/**
 * @brief Ultra-optimized function to find min/max X,Y values in a single pass
 * 
 * This version uses a single pass through the point cloud with branch prediction hints.
 * 
 * @param cloud Input point cloud
 * @return Tuple containing (minX, maxX, minY, maxY)
 */
inline std::tuple<float, float, float, float> getCloudXYRangeFast(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud
) {
    // Initialize with extreme values
    float minX = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float minY = std::numeric_limits<float>::max();
    float maxY = std::numeric_limits<float>::lowest();
    
    // Early check for empty cloud
    if (cloud->empty()) {
        return std::make_tuple(minX, maxX, minY, maxY);
    }
    
    // Pre-fetch the first point to initialize values
    // This reduces branches in the main loop
    const auto& firstPoint = cloud->points[0];
    minX = maxX = firstPoint.x;
    minY = maxY = firstPoint.y;
    
    // Single pass through the remaining points
    // Starting from index 1 since we already processed the first point
    const size_t size = cloud->points.size();
    for (size_t i = 1; i < size; ++i) {
        const auto& point = cloud->points[i];
        
        // Update X range with branch prediction hints
        if (__builtin_expect(point.x < minX, 0)) minX = point.x;
        if (__builtin_expect(point.x > maxX, 0)) maxX = point.x;
        
        // Update Y range with branch prediction hints
        if (__builtin_expect(point.y < minY, 0)) minY = point.y;
        if (__builtin_expect(point.y > maxY, 0)) maxY = point.y;
    }
    
    return std::make_tuple(minX, maxX, minY, maxY);
}

/**
 * @brief Optimized function to find min/max X,Y values with outlier rejection
 * 
 * This version filters out outliers by only considering points within
 * a certain percentile range.
 * 
 * @param cloud Input point cloud
 * @param minX Output minimum X value
 * @param maxX Output maximum X value
 * @param minY Output minimum Y value
 * @param maxY Output maximum Y value
 * @param lowerPercentile Lower percentile for outlier rejection (default 0.05 = 5%)
 * @param upperPercentile Upper percentile for outlier rejection (default 0.95 = 95%)
 */
inline void getCloudXYRangeRobust(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    float& minX, float& maxX, float& minY, float& maxY,
    float lowerPercentile = 0.05f,
    float upperPercentile = 0.95f
) {
    // Initialize with extreme values
    minX = std::numeric_limits<float>::max();
    maxX = std::numeric_limits<float>::lowest();
    minY = std::numeric_limits<float>::max();
    maxY = std::numeric_limits<float>::lowest();
    
    // Early check for empty cloud
    if (cloud->empty()) {
        return;
    }
    
    // Extract X and Y values into separate vectors
    std::vector<float> xValues, yValues;
    xValues.reserve(cloud->size());
    yValues.reserve(cloud->size());
    
    for (const auto& point : cloud->points) {
        xValues.push_back(point.x);
        yValues.push_back(point.y);
    }
    
    // Sort the values
    std::sort(xValues.begin(), xValues.end());
    std::sort(yValues.begin(), yValues.end());
    
    // Calculate indices for percentiles
    size_t lowerIdx = static_cast<size_t>(lowerPercentile * xValues.size());
    size_t upperIdx = static_cast<size_t>(upperPercentile * xValues.size());
    
    // Ensure indices are valid
    lowerIdx = std::min(std::max(lowerIdx, static_cast<size_t>(0)), xValues.size() - 1);
    upperIdx = std::min(std::max(upperIdx, static_cast<size_t>(0)), xValues.size() - 1);
    
    // Get percentile values
    minX = xValues[lowerIdx];
    maxX = xValues[upperIdx];
    minY = yValues[lowerIdx];
    maxY = yValues[upperIdx];
}

/**
 * @brief Optimized function to find min/max X,Y values using chunking for better cache utilization
 * 
 * This version processes the point cloud in chunks for better cache performance.
 * 
 * @param cloud Input point cloud
 * @param minX Output minimum X value
 * @param maxX Output maximum X value
 * @param minY Output minimum Y value
 * @param maxY Output maximum Y value
 */
inline void getCloudXYRangeCacheOptimized(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    float& minX, float& maxX, float& minY, float& maxY
) {
    // Initialize with extreme values
    minX = std::numeric_limits<float>::max();
    maxX = std::numeric_limits<float>::lowest();
    minY = std::numeric_limits<float>::max();
    maxY = std::numeric_limits<float>::lowest();
    
    // Early check for empty cloud
    if (cloud->empty()) {
        return;
    }
    
    // Process in chunks for better cache utilization
    constexpr size_t CHUNK_SIZE = 1024; // Adjust based on cache size
    const size_t size = cloud->points.size();
    const size_t numChunks = (size + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    // Process each chunk
    for (size_t chunk = 0; chunk < numChunks; ++chunk) {
        // Calculate chunk bounds
        size_t start = chunk * CHUNK_SIZE;
        size_t end = std::min(start + CHUNK_SIZE, size);
        
        // Local min/max for this chunk
        float chunkMinX = std::numeric_limits<float>::max();
        float chunkMaxX = std::numeric_limits<float>::lowest();
        float chunkMinY = std::numeric_limits<float>::max();
        float chunkMaxY = std::numeric_limits<float>::lowest();
        
        // Process points in this chunk
        for (size_t i = start; i < end; ++i) {
            const auto& point = cloud->points[i];
            
            // Update X range
            if (point.x < chunkMinX) chunkMinX = point.x;
            if (point.x > chunkMaxX) chunkMaxX = point.x;
            
            // Update Y range
            if (point.y < chunkMinY) chunkMinY = point.y;
            if (point.y > chunkMaxY) chunkMaxY = point.y;
        }
        
        // Update global min/max
        if (chunkMinX < minX) minX = chunkMinX;
        if (chunkMaxX > maxX) maxX = chunkMaxX;
        if (chunkMinY < minY) minY = chunkMinY;
        if (chunkMaxY > maxY) maxY = chunkMaxY;
    }
}

/**
 * @brief Optimized function to find min/max X,Y values with Z slicing
 * 
 * This version finds min/max X,Y values for points within a specific Z slice.
 * 
 * @param cloud Input point cloud
 * @param minX Output minimum X value
 * @param maxX Output maximum X value
 * @param minY Output minimum Y value
 * @param maxY Output maximum Y value
 * @param sliceMinZ Minimum Z value for the slice
 * @param sliceMaxZ Maximum Z value for the slice
 * @return Number of points in the slice
 */
inline int getCloudXYRangeInZSlice(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    float& minX, float& maxX, float& minY, float& maxY,
    float sliceMinZ, float sliceMaxZ
) {
    // Initialize with extreme values
    minX = std::numeric_limits<float>::max();
    maxX = std::numeric_limits<float>::lowest();
    minY = std::numeric_limits<float>::max();
    maxY = std::numeric_limits<float>::lowest();
    
    // Early check for empty cloud
    if (cloud->empty()) {
        return 0;
    }
    
    // Count points in slice
    int pointCount = 0;
    
    // Single pass through the point cloud
    for (const auto& point : cloud->points) {
        // Check if point is in the Z slice
        if (point.z >= sliceMinZ && point.z <= sliceMaxZ) {
            // Update X range
            if (point.x < minX) minX = point.x;
            if (point.x > maxX) maxX = point.x;
            
            // Update Y range
            if (point.y < minY) minY = point.y;
            if (point.y > maxY) maxY = point.y;
            
            // Increment point count
            pointCount++;
        }
    }
    
    return pointCount;
}