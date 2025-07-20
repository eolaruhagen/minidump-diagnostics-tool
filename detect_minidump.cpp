#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;

class MinidumpMonitor {
    private: 
        const std::string MINIDUMP_PATH = "C:/Windows/Minidump";
    public:
        void scanOnce() {
            std::cout << "Scanning for minidumps in " << MINIDUMP_PATH << std::endl;

            if (!fs::exists(MINIDUMP_PATH)) {
                std::cout << "Minidump directory does not exist: " << MINIDUMP_PATH << std::endl;
                return;
            }

            std::vector<std::string> dumpsFound;

            for (const auto& entry : fs::directory_iterator(MINIDUMP_PATH)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().filename().string();
                    std::string extension = entry.path().extension().string();

                    // Check if it's a minidump file (typically .dmp extension)
                    if (extension == ".dmp" || extension == ".mdmp") {
                        dumpsFound.push_back(filename);
                        std::cout << "Found minidump: " << filename << std::endl;
                    }
                }
            }

            if (dumpsFound.empty()) {
                std::cout << "No minidumps found." << std::endl;
            } else {
                std::cout << "Found: " << dumpsFound.size() << " minidumps." << std::endl;
                for (const auto& dump : dumpsFound) {
                    std::cout << " - " << dump << std::endl;
                }
            }
        }

        void continuousMonitor(int intervalSeconds = 20) {
            for (int i = 0; i < 5; ++i) {
                scanOnce();
                std::this_thread::sleep_for(std::chrono::seconds(intervalSeconds));
            }

            std::cout << "Monitoring completed. Should not reach here -> nothing found" << std::endl;
        }

        void showHelp() {
            std::cout << "Minidump CLI Tool - Commands:" << std::endl;
            std::cout << "  scan                    - Scan for existing minidump files" << std::endl;
            std::cout << "  monitor [interval]      - Monitor for new files (default: 30s)" << std::endl;
            std::cout << "  help                    - Show this help" << std::endl;
        }

};

int main(int argc, char* argv[]) {
    MinidumpMonitor monitor;
    std::cout << "This is a placeholder for detect_minidump.c++" << std::endl;

    if (argc < 2) {
        std::cout << "Error: No command provided!" << std::endl;
        monitor.showHelp();
        return 1;  // Return error code
    }
    
    std::string command = argv[1];

    if (command == "scan") {
        monitor.scanOnce();
    } else if (command == "monitor") { // Default interval
        monitor.continuousMonitor();
    } else if (command == "help") {
        monitor.showHelp();
    } else {
        std::cout << "Unknown command: " << command << std::endl;
        monitor.showHelp();
        return 1;  // Return error code
    }
    std::cout << "Command received: " << command << std::endl;

    return 0;
}

