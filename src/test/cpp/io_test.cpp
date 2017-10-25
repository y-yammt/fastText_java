#include <iostream>
#include <fstream>

#include <string>
#include <vector>
#include <stdlib.h>

class Utils {
public:
  Utils();
  int64_t size(std::ifstream&);
  void seek(std::ifstream&, int64_t);
};

Utils::Utils() {
}

int64_t Utils::size(std::ifstream& ifs) {
    ifs.seekg(std::streamoff(0), std::ios::end);
    return ifs.tellg();
}

void Utils::seek(std::ifstream& ifs, int64_t pos) {
    ifs.clear();
    ifs.seekg(std::streampos(pos));
}


// ./io_test ./test2.txt 50000
int main(int argc, char** argv)
{

    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 3) {
        std::cerr << "Wrong input" << std::endl;
        exit(-2);
    }
    std::string infile = args[1];
    int32_t _skip = std::stoi(args[2]);
    std::cout << "infile:\t" << infile << "\n";
    std::cout << "skip:\t" << _skip << "\n";

    Utils utils;

    std::ifstream ifs(infile);

    int64_t _size = utils.size(ifs);
    std::cout << "size:\t" << _size << "\n";

    utils.seek(ifs, _skip);

    if (ifs.is_open()) {
        std::cout << ifs.rdbuf();
        int64_t __size = utils.size(ifs);
        std::cout << "__size:\t" << _size << "\n";
        ifs.close();
    } else {
        std::cerr << "Test file cannot be opened!" << std::endl;
    }

    return 0;
}