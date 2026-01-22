#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>

namespace Replica {

enum class LogLevel {
    INFO,
    SUCCESS,
    WARNING,
    ERROR,
    FOUND
};

class Logger {
public:
    static void log(LogLevel level, const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        struct tm time_info;
        localtime_s(&time_info, &time_t_now);

        std::ostringstream oss;
        oss << std::setfill('0') << std::setw(2) << time_info.tm_hour << ":"
            << std::setw(2) << time_info.tm_min << ":"
            << std::setw(2) << time_info.tm_sec;
        std::string time_str = oss.str();
        
        std::string color;
        std::string label;

        switch (level) {
            case LogLevel::INFO:
                color = "\033[36m"; // Cyan
                label = "INFO";
                break;
            case LogLevel::SUCCESS:
                color = "\033[32m"; // Green
                label = "PASS";
                break;
            case LogLevel::WARNING:
                color = "\033[33m"; // Yellow
                label = "WARN";
                break;
            case LogLevel::ERROR:
                color = "\033[31m"; // Red
                label = "FAIL";
                break;
            case LogLevel::FOUND:
                color = "\033[1;32m"; // Bold Green
                label = "FOUND";
                break;
        }

        std::cout << "\033[90m[" << time_str << "] " << color << "[" << label << "]\033[0m " << message << std::endl;
    }

    static void info(const std::string& msg) { log(LogLevel::INFO, msg); }
    static void success(const std::string& msg) { log(LogLevel::SUCCESS, msg); }
    static void warn(const std::string& msg) { log(LogLevel::WARNING, msg); }
    static void error(const std::string& msg) { log(LogLevel::ERROR, msg); }
    static void found(const std::string& msg) { log(LogLevel::FOUND, msg); }
};

}
