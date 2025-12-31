#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace BarcodeInspector {

    struct InspectionParams {
        bool USE_OTSU = false;
        double MANUAL_THRESH = 150.0;
        int MORPH_KERNEL_SIZE = 7;
        double AREA_THRESHOLD = 3000.0;
    };

    struct InspectionResult {
        bool is_defect = false;
        double max_defect_area = 0.0;
        cv::Mat processed_binary;
        cv::Mat defect_mask;
        cv::Mat debug_image;
    };

    std::vector<cv::Point2f> order_points(const std::vector<cv::Point2f>& pts);
    std::pair<cv::Mat, double> preprocess_image(const cv::Mat& src, const InspectionParams& params);
    InspectionResult run_inspection(
        const cv::Mat& target_img,
        const cv::Mat& golden_mask,
        const std::vector<cv::Point2f>& golden_points,
        const InspectionParams& params
    );
}

void Run_Barcode_Rulebase_Algorithm(const std::wstring& input_path);