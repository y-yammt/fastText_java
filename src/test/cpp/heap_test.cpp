#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

int main (int argc, char** argv)
{
    std::vector<std::string> args(argv, argv + argc);
    std::vector<float> input;
    int32_t k = std::stoi(args[1]);
    for (int i = 2; i < args.size(); i++) {
        float num = std::stof(args[i]);
        input.push_back(num);
    }

    std::cout << "k=" << k << "\ninput=";
    for (int i = 0; i < input.size(); i++) {
        std::cout << input[i] << " ";
    }
    std::cout << "\n";

    std::vector<float> heap;
    for (int i = 0; i < input.size(); i++) {
        heap.push_back(input[i]);
        std::push_heap(heap.begin(), heap.end());
        if (heap.size() > k) {
            std::pop_heap(heap.begin(), heap.end());
            heap.pop_back();
        }
    }
    std::sort_heap(heap.begin(), heap.end());

    std::cout << "result=";
    for (int i = 0; i < heap.size(); i++) {
        std::cout << heap[i] << " ";
    }
    std::cout << "\n";
    return 0;
}