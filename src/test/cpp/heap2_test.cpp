#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

int main (int argc, char** argv)
{
    std::vector<std::string> args(argv, argv + argc);
    std::vector<int> ints;
    for (int i = 1; i < args.size(); i++) {
        int32_t num = std::stoi(args[i]);
        ints.push_back(num);
    }

    for (int i = 0; i < ints.size(); i++) {
        std::cout << ints[i] << " ";
    }
    std::cout << "\n";
    std::push_heap(ints.begin(), ints.end());
    std::cout << ints.front() << "\n";
    return 0;
}