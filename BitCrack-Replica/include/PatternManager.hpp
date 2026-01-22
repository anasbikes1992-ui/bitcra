#ifndef _PATTERN_MANAGER_HPP
#define _PATTERN_MANAGER_HPP

#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>

namespace Replica {

class PatternManager {
public:
    struct MaskPart {
        bool isWildcard;
        unsigned char value; // Only valid if not wildcard
    };

    static std::vector<MaskPart> parseMask(const std::string& mask) {
        if (mask.length() > 64) throw std::runtime_error("Mask too long (max 64 hex chars)");
        
        std::vector<MaskPart> parts;
        for (size_t i = 0; i < mask.length(); i += 2) {
            std::string byteStr = mask.substr(i, 2);
            if (byteStr == "??") {
                parts.push_back({true, 0});
            } else {
                parts.push_back({false, (unsigned char)std::stoul(byteStr, nullptr, 16)});
            }
        }
        return parts;
    }

    static void fillRandom(const std::vector<MaskPart>& mask, unsigned int privKey[8]) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<unsigned int> dist(0, 255);

        unsigned char bytes[32] = {0};
        for (size_t i = 0; i < 32; ++i) {
            if (i < mask.size()) {
                bytes[i] = mask[i].isWildcard ? (unsigned char)dist(gen) : mask[i].value;
            } else {
                bytes[i] = (unsigned char)dist(gen);
            }
        }

        // Convert 32 bytes to 8 uint32 (Big Endian as per BitCrack convention)
        for (int i = 0; i < 8; ++i) {
            privKey[i] = (bytes[i*4] << 24) | (bytes[i*4+1] << 16) | (bytes[i*4+2] << 8) | bytes[i*4+3];
        }
    }
};

} // namespace Replica

#endif
