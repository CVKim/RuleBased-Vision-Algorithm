#include "BarcodeRulebase.h"
#include "CommonUtils.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <numeric>
#include <algorithm>

namespace BarcodeInspector {
    bool load_golden_points_json(const std::wstring& path, std::vector<cv::Point2f>& out_points) {
        std::ifstream file(path);
        if (!file.is_open()) return false;
        std::stringstream buffer; buffer << file.rdbuf(); std::string content = buffer.str();
        for (char& c : content) { if (c == '[' || c == ']' || c == ',') c = ' '; }
        std::stringstream ss(content); float coord; std::vector<float> coords;
        while (ss >> coord) coords.push_back(coord);
        if (coords.size() != 8) return false;
        out_points.clear();
        for (size_t i = 0; i < 8; i += 2) out_points.push_back(cv::Point2f(coords[i], coords[i + 1]));
        return true;
    }

    std::vector<cv::Point2f> order_points(const std::vector<cv::Point2f>& pts) {
        if (pts.size() != 4) return pts;
        std::vector<cv::Point2f> rect(4);
        std::vector<float> sum_vals, diff_vals;
        for (const auto& p : pts) { sum_vals.push_back(p.x + p.y); diff_vals.push_back(p.y - p.x); }
        rect[0] = pts[std::distance(sum_vals.begin(), std::min_element(sum_vals.begin(), sum_vals.end()))]; // TL
        rect[2] = pts[std::distance(sum_vals.begin(), std::max_element(sum_vals.begin(), sum_vals.end()))]; // BR
        rect[1] = pts[std::distance(diff_vals.begin(), std::min_element(diff_vals.begin(), diff_vals.end()))]; // TR
        rect[3] = pts[std::distance(diff_vals.begin(), std::max_element(diff_vals.begin(), diff_vals.end()))]; // BL
        return rect;
    }

    // Preprocessing
    std::pair<cv::Mat, double> preprocess_image(const cv::Mat& src, const InspectionParams& params) {
        cv::Mat gray, binary;
        if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY); else gray = src.clone();
        if (params.USE_OTSU) cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        else cv::threshold(gray, binary, params.MANUAL_THRESH, 255, cv::THRESH_BINARY);
        if (params.MORPH_KERNEL_SIZE > 0) {
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(params.MORPH_KERNEL_SIZE, params.MORPH_KERNEL_SIZE));
            cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
        }
        std::vector<std::vector<cv::Point>> contours; cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Mat final_binary = cv::Mat::zeros(binary.size(), CV_8UC1); double area = 0.0;
        if (!contours.empty()) {
            auto max_it = std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });
            cv::drawContours(final_binary, std::vector<std::vector<cv::Point>>{*max_it}, -1, cv::Scalar(255), -1);
            area = cv::countNonZero(final_binary);
        }
        return { final_binary, area };
    }

    // Inspection
    InspectionResult run_inspection(const cv::Mat& target_img, const cv::Mat& golden_mask, const std::vector<cv::Point2f>& golden_points, const InspectionParams& params) {
        InspectionResult result; result.debug_image = target_img.clone();
        auto [target_binary, target_area] = preprocess_image(target_img, params);
        result.processed_binary = target_binary;
        if (target_area == 0) return result;

        std::vector<std::vector<cv::Point>> contours; cv::findContours(target_binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) return result;
        auto max_it = std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });
        if (max_it->size() < 4) return result;

        cv::RotatedRect rect = cv::minAreaRect(*max_it); cv::Point2f v_pts[4]; rect.points(v_pts);
        std::vector<cv::Point2f> target_points(v_pts, v_pts + 4);
        cv::Mat M = cv::getPerspectiveTransform(order_points(golden_points), order_points(target_points));
        cv::Mat warped_golden_mask; cv::warpPerspective(golden_mask, warped_golden_mask, M, target_binary.size());

        cv::Mat defect_mask_raw; cv::subtract(warped_golden_mask, target_binary, defect_mask_raw);
        std::vector<std::vector<cv::Point>> defect_contours; cv::findContours(defect_mask_raw, defect_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        result.defect_mask = cv::Mat::zeros(target_binary.size(), CV_8UC1);
        for (const auto& cnt : defect_contours) {
            double area = cv::contourArea(cnt);
            if (area >= params.AREA_THRESHOLD) {
                cv::drawContours(result.defect_mask, std::vector<std::vector<cv::Point>>{cnt}, -1, cv::Scalar(255), -1);
                cv::drawContours(result.debug_image, std::vector<std::vector<cv::Point>>{cnt}, -1, cv::Scalar(0, 0, 255), 2);
                if (area > result.max_defect_area) result.max_defect_area = area;
            }
        }
        result.is_defect = (result.max_defect_area > 0);
        cv::putText(result.debug_image, result.is_defect ? "NG" : "OK", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, result.is_defect ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 2);
        return result;
    }
}

void Run_Barcode_Rulebase_Algorithm(const std::wstring& input_path) {
    fs::path base_path = input_path;
    fs::path result_dir = base_path / "result_barcode_cpp";
    fs::path ok_dir = result_dir / "OK", ng_dir = result_dir / "NG";
    fs::create_directories(ok_dir); fs::create_directories(ng_dir);

    Logger logger(result_dir / "processing_barcode.log");
    logger.log(L"START Barcode Inspection: " + base_path.wstring());

    fs::path mask_path = base_path / "./golden/golden_template.bmp";
    fs::path points_path = base_path / "./golden/golden_points.json";

    cv::Mat golden_mask = imread_unicode(mask_path.wstring());
    std::vector<cv::Point2f> golden_points;
    bool points_loaded = BarcodeInspector::load_golden_points_json(points_path.wstring(), golden_points);

    if (golden_mask.empty() || !points_loaded) {
        logger.log(L"[ERROR] Template Image load failed (golden_template.bmp or golden_points.json 확인 필요)");
        return;
    }
    if (golden_mask.channels() == 3) cv::cvtColor(golden_mask, golden_mask, cv::COLOR_BGR2GRAY);
    cv::threshold(golden_mask, golden_mask, 127, 255, cv::THRESH_BINARY);

    BarcodeInspector::InspectionParams params;
    params.MANUAL_THRESH = 150.0;
    params.AREA_THRESHOLD = 3000.0;

    std::vector<fs::path> files;
    const std::vector<std::string> exts = { ".png", ".jpg", ".bmp" };
    for (const auto& entry : fs::directory_iterator(base_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(exts.begin(), exts.end(), ext) != exts.end()) {
                if (entry.path().filename() != "golden_template.bmp") files.push_back(entry.path());
            }
        }
    }

    for (const auto& file_path : files) {
        logger.log(L"Processing: " + file_path.filename().wstring());
        cv::Mat img = imread_unicode(file_path.wstring());
        if (img.empty()) continue;

        auto result = BarcodeInspector::run_inspection(img, golden_mask, golden_points, params);

        fs::path out_path = result.is_defect ? ng_dir : ok_dir;
        out_path /= ("result_" + file_path.filename().string());

        imwrite_unicode(out_path.wstring(), result.debug_image);

        std::wstringstream wss;
        wss << L"[RESULT] " << (result.is_defect ? L"NG" : L"OK")
            << L" | Defect: " << (int)result.max_defect_area;
        logger.log(wss.str());
    }
    logger.log(L"Inspection Finished.");
}