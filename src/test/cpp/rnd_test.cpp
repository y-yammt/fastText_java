#include <string>
#include <random>

#include <vector>
#include <iostream>
#include <stdint.h>

int main (int argc, char** argv)
{
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 4) {
        std::cerr << "Wrong input [seed, num, end(, start)]" << std::endl;
        exit(-2);
    }

    int32_t seed = std::stoi(args[1]);
    int32_t num = std::stoi(args[2]);
    int32_t end = std::stoi(args[3]);
    int32_t start = args.size() > 4 ? std::stoi(args[4]) : 0;
    std::cout << "seed=" << seed << ", num=" << num << ", start=" << start << ", end=" << end << "\n";

    std::minstd_rand rng(seed);

    std::uniform_int_distribution<> uniform(start, end);
    for (int i = 0; i < num; ++i) {
        int32_t r = uniform(rng);
        std::cout << "" << r << "\n";
    }

    const int32_t _test = uniform(rng);
    std::cout << "Test1:" << _test << "\n";
    std::cout << "Test2:" << _test << "\n";
    return 0;
}