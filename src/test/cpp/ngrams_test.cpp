#include <string>
#include <vector>
#include <stdlib.h>
#include <iostream>

int main()
{
    int nwords_ = 803537;
    int n = 2;
    int bucket = 10000000;
    std::vector<int32_t> line = {2, 5891, 3193, 2, 5891, 3193, 6, 0, 251, 176, 28, 12, 42, 87, 203471, 1, 0, 28, 11, 47,
    16, 128, 211, 243, 153, 1795, 2951, 1, 125, 1, 8};
    std::vector<uint32_t> _hashes = {688690635, 3583658857, 3320932157, 688690635, 3583658857, 3320932157, 1312329493,
    3020861980, 1801784193, 3210726659, 1694181484, 1412156564, 3499186239, 815078726, 1619100151, 722245873, 3020861980,
    1694181484, 383959538, 2685670126, 1630810064, 3158916199, 1024243015, 3408680699, 4254058618, 1084723594, 1769466727,
    722245873, 639349933, 722245873, 3617362777};
    std::vector<int32_t> hashes;
    for (int32_t i = 0; i < _hashes.size(); i++) {
        hashes.push_back(_hashes[i]);
    }

    for (int32_t i = 0; i < hashes.size(); i++) {
        uint64_t h = hashes[i];
        for (int32_t j = i + 1; j < hashes.size() && j < i + n; j++) {
            h = h * 116049371 + hashes[j];
            int32_t id = h % bucket;
            line.push_back(nwords_ + id);
        }
    }
    for (int32_t i = 0; i < line.size(); i++) {
        std::cout << line[i] << "\n";
    }

}