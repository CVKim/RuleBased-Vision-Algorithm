#include "CommonUtils.h"
#include "KBRulebase.h"

namespace ContourInspector {

    struct InspectionParams {
        int COST_ROW_SMOOTH_K = 3;
        float STEP_PENALTY_TARGET = 0.45f;
        float RAMP_START_FRAC = 0.10f;
        float RAMP_LEN_FRAC = 0.40f;
        float SPLINE_SMOOTHING_FACTOR = 100.0f;
        float LINEARITY_TH = 7.5f;
        float ANGLE_TH_DEG = 4.0f;
        std::string JUDGEMENT_MODE = "OR";
    };

    struct InspectionMetrics {
        bool kb_final = false, by_curve = false, by_tilt = false;
        float linearity_dev = 0.0f, tilt_angle_deg = 0.0f;
        cv::Point2f pca_mu;
        cv::Vec2f pca_v;
    };

    std::pair<cv::Mat, cv::Mat> calculate_cost_map(const cv::Mat& gray, const InspectionParams& params) {
        cv::Mat gray_32f, gray_blur, bg, enh, gray_norm, cost;
        gray.convertTo(gray_32f, CV_32F);
        cv::GaussianBlur(gray_32f, gray_blur, cv::Size(3, 3), 0);
        cv::GaussianBlur(gray_32f, bg, cv::Size(0, 0), 1.0);
        cv::addWeighted(gray_32f, 1.1, bg, -0.1, 0, enh);
        cv::normalize(enh, gray_norm, 0.0, 1.0, cv::NORM_MINMAX);
        cv::pow(1.0 - gray_norm, 1.1, cost);

        cv::blur(cost, cost, cv::Size(1, 3));
        if (params.COST_ROW_SMOOTH_K > 1) {
            cv::blur(cost, cost, cv::Size(1, params.COST_ROW_SMOOTH_K));
        }
        return { cost, enh };
    }

    std::vector<cv::Point> find_optimal_path(const cv::Mat& cost, const cv::Mat& enh, const InspectionParams& params) {
        int H = cost.rows, W = cost.cols;
        cv::Mat row_pen = cv::Mat::zeros(H, 1, CV_32F);

        // Lambda for DP execution
        auto run_dp = [&](const cv::Mat& penalty) -> std::pair<std::vector<cv::Point>, float> {
            cv::Mat acc(H, W, CV_32F, cv::Scalar::all(std::numeric_limits<float>::infinity()));
            cv::Mat back(H, W, CV_16S);
            cost.row(0).copyTo(acc.row(0));

            for (int r = 1; r < H; ++r) {
                float pen = penalty.empty() ? 0.0f : penalty.at<float>(r);
                for (int c = 0; c < W; ++c) {
                    float left = acc.at<float>(r - 1, std::max(0, c - 1)) + pen;
                    float mid = acc.at<float>(r - 1, c);
                    float right = acc.at<float>(r - 1, std::min(W - 1, c + 1)) + pen;

                    if (mid <= left && mid <= right) {
                        acc.at<float>(r, c) = mid + cost.at<float>(r, c);
                        back.at<short>(r, c) = 0;
                    }
                    else if (left < right) {
                        acc.at<float>(r, c) = left + cost.at<float>(r, c);
                        back.at<short>(r, c) = -1;
                    }
                    else {
                        acc.at<float>(r, c) = right + cost.at<float>(r, c);
                        back.at<short>(r, c) = 1;
                    }
                }
            }
            double min_val;
            cv::Point min_loc;
            cv::minMaxLoc(acc.row(H - 1), &min_val, nullptr, &min_loc, nullptr);

            std::vector<cv::Point> path;
            int c_min = min_loc.x;
            float total_cost = 0;

            for (int r = H - 1; r >= 0; --r) {
                path.push_back({ c_min, r });
                total_cost += cost.at<float>(r, c_min);
                if (r > 0) c_min += back.at<short>(r, c_min);
            }
            std::reverse(path.begin(), path.end());
            return { path, total_cost / H };
            };

        auto result_pen = run_dp(row_pen);
        auto result_nopen = run_dp(cv::Mat());

        return (result_pen.second <= 1.02 * result_nopen.second) ? result_pen.first : result_nopen.first;
    }

    InspectionMetrics calculate_path_metrics(const std::vector<double>& path_coords, const InspectionParams& params) {
        int H = static_cast<int>(path_coords.size());
        cv::Mat points(H, 2, CV_32F);
        for (int i = 0; i < H; ++i) {
            points.at<float>(i, 0) = static_cast<float>(path_coords[i]);
            points.at<float>(i, 1) = static_cast<float>(i);
        }

        cv::PCA pca(points, cv::Mat(), cv::PCA::DATA_AS_ROW);
        cv::Point2f mu(pca.mean.at<float>(0, 0), pca.mean.at<float>(0, 1));
        cv::Vec2f v(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
        cv::Vec2f n(-v[1], v[0]);

        float max_dev = 0;
        for (int i = 0; i < H; ++i) {
            float dev = std::abs((points.at<float>(i, 0) - mu.x) * n[0] + (points.at<float>(i, 1) - mu.y) * n[1]);
            if (dev > max_dev) max_dev = dev;
        }

        float angle_rad = std::atan2(v[1], v[0]);
        float angle_deg = angle_rad * 180.0f / CV_PI;
        float tilt_angle = std::abs(90.0f - std::abs(angle_deg));

        InspectionMetrics metrics;
        metrics.linearity_dev = max_dev;
        metrics.tilt_angle_deg = tilt_angle;
        metrics.pca_mu = mu;
        metrics.pca_v = v;
        return metrics;
    }

    cv::Mat draw_overlay(const cv::Mat& image, const std::vector<int>& path_coords_int, const InspectionMetrics& metrics, const InspectionParams& params) {
        cv::Mat img_disp = image.clone();
        for (size_t r = 1; r < path_coords_int.size(); ++r) {
            cv::line(img_disp, { path_coords_int[r - 1], (int)r - 1 }, { path_coords_int[r], (int)r }, { 255, 0, 0 }, 2, cv::LINE_AA);
        }

        cv::Point p0(static_cast<int>(metrics.pca_mu.x - 1000 * metrics.pca_v[0]), static_cast<int>(metrics.pca_mu.y - 1000 * metrics.pca_v[1]));
        cv::Point p1(static_cast<int>(metrics.pca_mu.x + 1000 * metrics.pca_v[0]), static_cast<int>(metrics.pca_mu.y + 1000 * metrics.pca_v[1]));
        cv::line(img_disp, p0, p1, { 0, 255, 255 }, 1, cv::LINE_AA);

        bool is_ng = metrics.kb_final;
        std::string status = is_ng ? "KB: NG" : "KB: OK";
        cv::Scalar color = is_ng ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 180, 0);
        std::string reason_str = "OK";

        if (is_ng) {
            std::vector<std::string> reasons;
            if (metrics.by_curve) reasons.push_back("Curve");
            if (metrics.by_tilt) reasons.push_back("Tilt");
            if (!reasons.empty()) reason_str = reasons[0];
            for (size_t i = 1; i < reasons.size(); ++i) reason_str += ", " + reasons[i];
        }

        std::string info1 = status + " (Reason: " + reason_str + ")";
        char info2_buf[256];
        sprintf_s(info2_buf, "C_Dev:%.1f(>%.0f) | T_Ang:%.1f(>%.0f)", metrics.linearity_dev, params.LINEARITY_TH, metrics.tilt_angle_deg, params.ANGLE_TH_DEG);

        cv::putText(img_disp, info1, { 10, 25 }, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
        cv::putText(img_disp, info2_buf, { 10, 50 }, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
        return img_disp;
    }

    std::pair<cv::Mat, InspectionMetrics> run_inspection(const cv::Mat& ori, const InspectionParams& params) {
        cv::Mat gray;
        cv::cvtColor(ori, gray, cv::COLOR_BGR2GRAY);
        int H = ori.rows, W = ori.cols;

        auto cost_result = calculate_cost_map(gray, params);
        cv::Mat cost = cost_result.first;
        cv::Mat enh = cost_result.second;

        auto path = find_optimal_path(cost, enh, params);
        std::vector<double> rows, cols;
        for (const auto& p : path) {
            rows.push_back(p.y);
            cols.push_back(p.x);
        }

        tk::spline s(rows, cols);
        std::vector<double> cols_f;
        for (int r = 0; r < H; ++r) cols_f.push_back(s(r));

        InspectionMetrics metrics = calculate_path_metrics(cols_f, params);
        metrics.by_curve = metrics.linearity_dev > params.LINEARITY_TH;
        metrics.by_tilt = metrics.tilt_angle_deg > params.ANGLE_TH_DEG;

        if (params.JUDGEMENT_MODE == "OR") {
            metrics.kb_final = metrics.by_curve || metrics.by_tilt;
        }
        else {
            metrics.kb_final = metrics.by_curve && metrics.by_tilt;
        }

        std::vector<int> cols_i;
        for (double val : cols_f) cols_i.push_back(std::min(W - 1, std::max(0, static_cast<int>(round(val)))));
        cv::Mat img_disp = draw_overlay(ori, cols_i, metrics, params);
        return { img_disp, metrics };
    }
}

// ============================================================================
// Implementation of the external interface function
// ============================================================================
void Run_KB_Rulebase_Algorithm(const std::wstring& input_path) {
    fs::path base_path = input_path;

    // Parameters setup
    ContourInspector::InspectionParams params;
    params.LINEARITY_TH = 7.5f;
    params.ANGLE_TH_DEG = 4.0f;
    params.JUDGEMENT_MODE = "OR";

    // Setup output directories
    fs::path result_dir = base_path / "result_2_step_OR_CPP";
    fs::path ok_dir = result_dir / "OK", ng_dir = result_dir / "NG";
    fs::create_directories(ok_dir);
    fs::create_directories(ng_dir);

    // Setup Logger
    fs::path log_path = result_dir / "processing.log";
    Logger logger(log_path);

    std::vector<fs::path> files;
    const std::vector<std::string> exts = { ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff" };

    // Check path
    if (!fs::exists(base_path)) {
        logger.log(L"[ERROR] Path does not exist: " + base_path.wstring());
        return;
    }

    // Collect files
    for (const auto& entry : fs::directory_iterator(base_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
            if (std::find(exts.begin(), exts.end(), ext) != exts.end()) {
                files.push_back(entry.path());
            }
        }
    }
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        logger.log(L"No image to process in: " + base_path.wstring());
        return;
    }

    logger.log(L"Total " + std::to_wstring(files.size()) + L" images found. Automatically process...");

    // Main Processing Loop
    for (const auto& src_path : files) {
        logger.log(L"--- Processing: " + src_path.filename().wstring() + L" ---");

        cv::Mat ori = imread_unicode(src_path.wstring());
        if (ori.empty()) {
            logger.log(L"[ERROR] Failed to load images: " + src_path.wstring());
            continue;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Execute Core Algorithm
        auto result = ContourInspector::run_inspection(ori, params);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;

        cv::Mat img_disp = result.first;
        ContourInspector::InspectionMetrics stats = result.second;

        std::string category = stats.kb_final ? "NG" : "OK";
        fs::path out_dir = (category == "NG") ? ng_dir : ok_dir;
        fs::path out_img_path = out_dir / ("result_" + src_path.stem().string() + ".bmp");

        imwrite_unicode(out_img_path.wstring(), img_disp);
        logger.log(L"[SAVE] " + out_img_path.wstring());

        std::wstring reason_str = L"OK";
        if (stats.kb_final) {
            std::vector<std::wstring> reasons;
            if (stats.by_curve) reasons.push_back(L"Curve");
            if (stats.by_tilt) reasons.push_back(L"Tilt");
            if (!reasons.empty()) reason_str = reasons[0];
            for (size_t i = 1; i < reasons.size(); ++i) reason_str += L", " + reasons[i];
        }

        std::wstringstream wss;
        wss << std::fixed << std::setprecision(2)
            << L"[STATS] " << src_path.filename().wstring()
            << L" | Final KB=" << (stats.kb_final ? L"NG" : L"OK") << L" (Reason: " << reason_str << L")"
            << L" | CurveDev=" << stats.linearity_dev << L" (>TH:" << params.LINEARITY_TH << L")"
            << L" | TiltAngle=" << stats.tilt_angle_deg << L" (>TH:" << params.ANGLE_TH_DEG << L")"
            << L" | ProcTime=" << std::setprecision(3) << elapsed_seconds.count() << L"s";
        logger.log(wss.str());
    }

    logger.log(L"All image processing has been completed.");
}