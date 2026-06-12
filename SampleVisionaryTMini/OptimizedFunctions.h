#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <algorithm>
#include <array>
#include <numeric>
#include <limits>

/**
 * @brief Improved comprehensive function that prioritizes the closest valid bucket
 * 
 * This version fixes the issue where the detection was one bucket further out
 * than the actual chassis end by prioritizing the closest valid bucket.
 * 
 * @param input Input point cloud
 * @param minZ Starting minimum Z value (default 300mm)
 * @param slice_step Slice width (default 50mm)
 * @param min_points_per_slice Minimum points needed for a valid slice (default 30)
 * @param min_x_spread Minimum required spread in X-axis (default 200mm)
 * @param min_y_spread Minimum required spread in Y-axis (default 100mm)
 * @param min_score_threshold Minimum score for a bucket to be considered valid (default 50.0f)
 * @return Valid minimum Z value (int)
 */
inline int get_valid_min_z_comprehensive(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
    int minZ = 300,
    int slice_step = 50,
    int min_points_per_slice = 30,
    float min_x_spread = 200.0f,
    float min_y_spread = 100.0f,
    float min_score_threshold = 50.0f
) {
    // Early check for empty cloud
    if (input->empty()) {
        return minZ;
    }
    
    // Fixed number of buckets for maximum performance
    constexpr int NUM_BUCKETS = 10;
    
    // Calculate the maximum Z value to consider
    int maxZ = minZ + (NUM_BUCKETS * slice_step / 2);
    
    // Structure to store comprehensive bucket information
    struct BucketInfo {
        int count = 0;                          // Point count
        float sum_z = 0.0f;                     // Sum of Z values
        float min_x = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float min_y = std::numeric_limits<float>::max();
        float max_y = std::numeric_limits<float>::lowest();
        // For calculating variance
        float sum_z_squared = 0.0f;             // Sum of Z^2 values
    };
    
    // Create array of bucket information
    std::array<BucketInfo, NUM_BUCKETS> buckets = {};
    
    // Single pass through the point cloud
    for (const auto& point : input->points) {
        // Apply all filters in a single check
        if (point.y >= 0 && point.y <= 2000 && point.z >= minZ && point.z <= maxZ) {
            // Find which buckets this point belongs to
            for (int i = 0; i < NUM_BUCKETS; i++) {
                int sliceMinZ = minZ + i * (slice_step / 2);
                int sliceMaxZ = sliceMinZ + slice_step;
                
                if (point.z >= sliceMinZ && point.z <= sliceMaxZ) {
                    // Update bucket information
                    BucketInfo& bucket = buckets[i];
                    bucket.count++;
                    bucket.sum_z += point.z;
                    bucket.sum_z_squared += point.z * point.z;
                    
                    // Update X and Y ranges
                    bucket.min_x = std::min(bucket.min_x, point.x);
                    bucket.max_x = std::max(bucket.max_x, point.x);
                    bucket.min_y = std::min(bucket.min_y, point.y);
                    bucket.max_y = std::max(bucket.max_y, point.y);
                }
            }
        }
    }
    
    // Calculate scores for each bucket based on multiple criteria
    std::array<float, NUM_BUCKETS> scores = {};
    std::array<bool, NUM_BUCKETS> is_valid_bucket = {};
    
    for (int i = 0; i < NUM_BUCKETS; i++) {
        const BucketInfo& bucket = buckets[i];
        
        // Skip buckets with too few points
        if (bucket.count < min_points_per_slice) {
            continue;
        }
        
        // Calculate spatial properties
        float x_spread = bucket.max_x - bucket.min_x;
        float y_spread = bucket.max_y - bucket.min_y;
        float area = x_spread * y_spread;
        float density = (area > 0) ? bucket.count / area : 0;
        
        // Calculate Z variance (lower is better - indicates flat surface)
        float mean_z = bucket.sum_z / bucket.count;
        float variance_z = (bucket.sum_z_squared / bucket.count) - (mean_z * mean_z);
        
        // Check if bucket has sufficient spread
        if (x_spread >= min_x_spread && y_spread >= min_y_spread) {
            // Calculate score based on multiple factors
            scores[i] = bucket.count * 0.5f +         // Point count
                        x_spread * 0.5f +             // X spread
                        y_spread * 0.5f +             // Y spread
                        density * 1000.0f -           // Point density
                        variance_z * 10.0f;           // Z variance (negative factor)
            
            // Mark bucket as valid if score is above threshold
            is_valid_bucket[i] = (scores[i] >= min_score_threshold);
        }
    }
    
    // Find the closest valid bucket
    for (int i = 0; i < NUM_BUCKETS; i++) {
        if (is_valid_bucket[i]) {
            // Found the closest valid bucket
            float avgZ = buckets[i].sum_z / buckets[i].count;
            return static_cast<int>(avgZ);
        }
    }
    
    // If no valid bucket found, return the original minZ
    return minZ;
}

/**
 * @brief Ultra-optimized version focusing on the closest valid slice
 * 
 * This version specifically addresses the issue of detecting one bucket further
 * than the actual chassis end by using a more direct approach focused on
 * finding the closest valid slice.
 * 
 * @param input Input point cloud
 * @param minZ Starting minimum Z value (default 300mm)
 * @param slice_step Slice width (default 50mm)
 * @param min_points_per_slice Minimum points needed for a valid slice (default 30)
 * @param min_x_spread Minimum required spread in X-axis (default 200mm)
 * @return Valid minimum Z value (int)
 */
inline int get_valid_min_z_closest_first(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
    int minZ = 300,
    int slice_step = 50,
    int min_points_per_slice = 30,
    float min_x_spread = 200.0f
) {
    // Early check for empty cloud
    if (input->empty()) {
        return minZ;
    }
    
    // Use a smaller slice step for more precision near the chassis end
    const float fine_slice_step = slice_step / 2.0f;  // 25mm increments
    const int NUM_BUCKETS = 20;  // More buckets for finer resolution
    
    // Calculate the maximum Z value to consider
    int maxZ = minZ + (NUM_BUCKETS * fine_slice_step);
    
    // Create arrays for bucket information
    std::array<int, NUM_BUCKETS> counts = {};
    std::array<float, NUM_BUCKETS> sums = {};
    std::array<float, NUM_BUCKETS> min_x_values = {};
    std::array<float, NUM_BUCKETS> max_x_values = {};
    
    // Initialize min_x_values to maximum float and max_x_values to minimum float
    for (int i = 0; i < NUM_BUCKETS; i++) {
        min_x_values[i] = std::numeric_limits<float>::max();
        max_x_values[i] = std::numeric_limits<float>::lowest();
    }
    
    // Single pass through the point cloud
    for (const auto& point : input->points) {
        // Apply all filters in a single check
        if (point.y >= 0 && point.y <= 2000 && point.z >= minZ && point.z <= maxZ) {
            // Find which bucket this point belongs to (non-overlapping for precision)
            int bucketIndex = static_cast<int>((point.z - minZ) / fine_slice_step);
            
            // Ensure index is within bounds
            if (bucketIndex >= 0 && bucketIndex < NUM_BUCKETS) {
                // Add point to this bucket
                counts[bucketIndex]++;
                sums[bucketIndex] += point.z;
                
                // Update X range
                min_x_values[bucketIndex] = std::min(min_x_values[bucketIndex], point.x);
                max_x_values[bucketIndex] = std::max(max_x_values[bucketIndex], point.x);
            }
        }
    }
    
    // Find the first (closest) bucket with enough points AND sufficient X spread
    for (int i = 0; i < NUM_BUCKETS; i++) {
        if (counts[i] >= min_points_per_slice) {
            // Calculate X spread
            float x_spread = max_x_values[i] - min_x_values[i];
            
            // Only consider this bucket valid if it has sufficient X spread
            if (x_spread >= min_x_spread) {
                // Calculate average Z for this slice
                float avgZ = sums[i] / counts[i];
                return static_cast<int>(avgZ);
            }
        }
    }
    
    // If no valid slice found, try a second pass with adjacent bucket merging
    // This helps handle cases where points are spread across bucket boundaries
    for (int i = 0; i < NUM_BUCKETS - 1; i++) {
        // Merge adjacent buckets
        int merged_count = counts[i] + counts[i + 1];
        
        if (merged_count >= min_points_per_slice) {
            // Calculate merged X spread
            float min_x = std::min(min_x_values[i], min_x_values[i + 1]);
            float max_x = std::max(max_x_values[i], max_x_values[i + 1]);
            float x_spread = max_x - min_x;
            
            // Only consider this merged bucket valid if it has sufficient X spread
            if (x_spread >= min_x_spread) {
                // Calculate weighted average Z for the merged slice
                float merged_sum = sums[i] + sums[i + 1];
                float avgZ = merged_sum / merged_count;
                return static_cast<int>(avgZ);
            }
        }
    }
    
    // If still no valid slice found, return the original minZ
    return minZ;
}