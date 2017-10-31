#include <iostream>     // std::cout
#include <algorithm>    // std::make_heap, std::pop_heap, std::push_heap, std::sort_heap
#include <vector>       // std::vector


class TestClass {
public:
  TestClass();
  static bool comparePairs(const std::pair<float, int32_t>&, const std::pair<float, int32_t>&);
};

TestClass::TestClass() {
}

bool TestClass::comparePairs(const std::pair<float, int32_t> &l, const std::pair<float, int32_t> &r) {
    return l.first > r.first;
}

int main() {
    TestClass _t;

    std::vector<std::pair<float, int>> heap;

    heap.push_back(std::make_pair(float(1.1), 1));
    heap.push_back(std::make_pair(float(1.2), 2));
    heap.push_back(std::make_pair(float(1.2), 3));
    heap.push_back(std::make_pair(float(-1.2), 4));
    heap.push_back(std::make_pair(float(5), 5));
    heap.push_back(std::make_pair(float(6), 6));
    heap.push_back(std::make_pair(float(6), 6));
    heap.push_back(std::make_pair(float(-10), 7));
    heap.push_back(std::make_pair(float(5), 8));
    heap.push_back(std::make_pair(float(9), 9));
    heap.push_back(std::make_pair(float(-8.8), 10));
    heap.push_back(std::make_pair(float(-8.8), 11));

    for (auto it = heap.cbegin(); it != heap.cend(); it++) {
        std::cout << it->first << "\t" << it->second << "\n";
    }

    std::cout << "\npush_heap:\n";
    std::push_heap(heap.begin(), heap.end(), _t.comparePairs);
    for (auto it = heap.cbegin(); it != heap.cend(); it++) {
        std::cout << it->first << "\t" << it->second << "\n";
    }

    std::cout << "\nagain.push_heap:\n";
    std::push_heap(heap.begin(), heap.end(), _t.comparePairs);
    for (auto it = heap.cbegin(); it != heap.cend(); it++) {
        std::cout << it->first << "\t" << it->second << "\n";
    }

    std::cout << "\npop_heap:\n";
    std::pop_heap(heap.begin(), heap.end(), _t.comparePairs);
    for (auto it = heap.cbegin(); it != heap.cend(); it++) {
        std::cout << it->first << "\t" << it->second << "\n";
    }

    std::cout << "\nsort_heap:\n";
    std::sort_heap(heap.begin(), heap.end(), _t.comparePairs);
    for (auto it = heap.cbegin(); it != heap.cend(); it++) {
        std::cout << it->first << "\t" << it->second << "\n";
    }
    return 0;
}