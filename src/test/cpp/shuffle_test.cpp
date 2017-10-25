#include <string>
#include <random>
#include <algorithm>

#include <vector>
#include <iostream>
#include <stdint.h>

int main (int argc, char** argv)
{
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 3) {
        std::cerr << "Wrong input" << std::endl;
        exit(-2);
    }
    args.erase(args.begin());

    for (auto i = args.begin(); i != args.end(); ++i) {
        std::cout << *i << ", ";
    }
    std::cout << "\n";

    std::minstd_rand rng(12);
    std::shuffle(args.begin(), args.end(), rng);

    for (auto i = args.begin(); i != args.end(); ++i) {
        std::cout << *i << ", ";
    }
    std::cout << "\n";
    return 0;
}