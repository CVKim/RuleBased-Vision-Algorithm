#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <windows.h>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// Helper: Convert wstring to utf8 string
inline std::string wstring_to_utf8(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
}

// Helper: Get current timestamp
inline std::wstring get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm buf;
    localtime_s(&buf, &in_time_t);
    std::wstringstream ss;
    ss << std::put_time(&buf, L"%Y-%m-%d %X");
    return ss.str();
}

// Class: Simple file and console logger
class Logger {
public:
    Logger(const fs::path& log_path) {
        log_file.open(log_path, std::ios::out | std::ios::trunc);
        if (log_file.is_open()) {
            log_file << "\xEF\xBB\xBF"; // UTF-8 BOM
        }
    }

    void log(const std::wstring& message) {
        std::wstringstream wss;
        wss << get_timestamp() << L" | " << message;

        std::wcout << wss.str() << std::endl;
        if (log_file.is_open()) {
            log_file << wstring_to_utf8(wss.str()) << std::endl;
        }
    }

private:
    std::ofstream log_file;
};

// Helper: Read image with unicode path
inline cv::Mat imread_unicode(const std::wstring& wpath) {
    std::ifstream file(wpath, std::ios::binary);
    if (!file) return cv::Mat();
    file.seekg(0, std::ios::end);
    std::vector<char> buffer(static_cast<size_t>(file.tellg()));
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), buffer.size());
    return cv::imdecode(buffer, cv::IMREAD_COLOR);
}

// Helper: Write image with unicode path
inline bool imwrite_unicode(const std::wstring& wpath, const cv::Mat& image) {
    std::vector<uchar> buffer;
    cv::imencode(".bmp", image, buffer);
    std::ofstream file(wpath, std::ios::binary);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    return true;
}

// Namespace: Spline interpolation logic
namespace tk {
    class spline {
    public:
        spline() {};
        spline(const std::vector<double>& x, const std::vector<double>& y);
        double operator()(double x) const;
    private:
        std::vector<double> m_x, m_y, m_b, m_c, m_d;
    };

    inline spline::spline(const std::vector<double>& x, const std::vector<double>& y) : m_x(x), m_y(y) {
        int n = static_cast<int>(x.size());
        if (n < 2) throw std::invalid_argument("spline: not enough points");
        if (n != y.size()) throw std::invalid_argument("spline: x and y vector must be of same size");

        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int i, int j) { return x[i] < x[j]; });

        for (int i = 0; i < n; i++) {
            m_x[i] = x[idx[i]];
            m_y[i] = y[idx[i]];
        }

        if (n < 3) {
            m_b.resize(1);
            m_b[0] = (m_y[1] - m_y[0]) / (m_x[1] - m_x[0]);
            return;
        }

        std::vector<double> h(n - 1);
        for (int i = 0; i < n - 1; i++) h[i] = m_x[i + 1] - m_x[i];

        std::vector<double> alpha(n - 1);
        for (int i = 1; i < n - 1; i++) alpha[i] = 3.0 / h[i] * (m_y[i + 1] - m_y[i]) - 3.0 / h[i - 1] * (m_y[i] - m_y[i - 1]);

        std::vector<double> l(n), mu(n), z(n);
        m_b.resize(n); m_c.resize(n); m_d.resize(n);

        l[0] = 1.0; mu[0] = 0.0; z[0] = 0.0;
        for (int i = 1; i < n - 1; i++) {
            l[i] = 2.0 * (m_x[i + 1] - m_x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }
        l[n - 1] = 1.0; z[n - 1] = 0.0; m_c[n - 1] = 0.0;

        for (int j = n - 2; j >= 0; j--) {
            m_c[j] = z[j] - mu[j] * m_c[j + 1];
            m_b[j] = (m_y[j + 1] - m_y[j]) / h[j] - h[j] * (m_c[j + 1] + 2.0 * m_c[j]) / 3.0;
            m_d[j] = (m_c[j + 1] - m_c[j]) / (3.0 * h[j]);
        }
    }

    inline double spline::operator()(double x) const {
        int n = static_cast<int>(m_x.size());
        if (n < 2) return 0.0;
        if (n == 2) {
            return m_y[0] + m_b[0] * (x - m_x[0]);
        }
        auto it = std::upper_bound(m_x.begin(), m_x.end(), x);
        int i = static_cast<int>(std::distance(m_x.begin(), it)) - 1;
        i = std::max(0, std::min(i, n - 2));
        double dx = x - m_x[i];
        return m_y[i] + m_b[i] * dx + m_c[i] * dx * dx + m_d[i] * dx * dx * dx;
    }
}