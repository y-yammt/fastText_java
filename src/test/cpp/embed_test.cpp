#include <string>
#include <algorithm>

#include <vector>
#include <iostream>
#include <stdint.h>

void _print(std::string name, std::vector<int32_t> vec) {
    std::cout << name << ":\n";
    for (auto i = vec.begin(); i != vec.end(); ++i) {
        std::cout << *i << " ";
    }
    std::cout << "\n";
    std::cout << name << ".size=" << vec.size() << "\n";
}

std::vector<int32_t> selectEmbeddings(std::vector<int32_t> norms, int32_t eosid, int32_t cutoff) {
    std::vector<int32_t> idx(norms.size(), 0);
    std::iota(idx.begin(), idx.end(), 0);
    //auto eosid = dict_->getId(Dictionary::EOS);
    _print("idx", idx);

    std::sort(idx.begin(), idx.end(),
        [&norms, eosid] (size_t i1, size_t i2) {
            return eosid ==i1 || (eosid != i2 && norms[i1] > norms[i2]);
        });

    _print("sort", idx);

    idx.erase(idx.begin() + cutoff, idx.end());
    return idx;
}

int main (int argc, char** argv)
{
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 4) {
        std::cerr << "Wrong input" << std::endl;
        exit(-2);
    }
    int32_t eosid = std::stoi(args[1]);
    int32_t cutoff = std::stoi(args[2]);
    std::cout << "eosid=" << eosid << ", cutoff=" << cutoff << "\n";

    std::vector<int32_t> nums(args.size() - 3, 0);
    for (int i = 3; i < args.size(); ++i) {
        nums[i - 3] = std::stoi(args[i]);
    }

    _print("input", nums);
    std::cout << "\n";

    std::vector<int32_t> res = selectEmbeddings(nums, eosid, cutoff);
    _print("selectEmbeddings", res);

    return 0;
}