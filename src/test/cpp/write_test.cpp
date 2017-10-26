#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>

enum class entry_type : int8_t {word=0, label=1};

int main(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 2) {
        std::cerr << "Wrong input" << std::endl;
        exit(-2);
    }
    std::string file = args[1];

    std::cout << "Open:\t" << file << "\n";
    std::ofstream out(file, std::ofstream::binary);
    if (!out.is_open()) {
        std::cerr << "Can't open '" << file << "'" << std::endl;
        exit(-12);
    }

    int32_t _int = 111111;
    int64_t _long = 222222;
    float _float = 333333.333;
    double _double = 444444444.444444;
    std::string _string = "555_555_555_555_555";
    float* _array = new float[2];
    _array[0] = 1.1;
    _array[1] = 2.3;

    out.write((char*) &_int, sizeof(int32_t));
    out.write((char*) &_long, sizeof(int64_t));
    out.write((char*) &_float, sizeof(float));
    out.write((char*) &_double, sizeof(double));
    out.write(_string.data(), _string.size() * sizeof(char));
    out.put(0);
    out.write((char*) _array, 2 * sizeof(float));
    out.write((char*) &(e.type), sizeof(entry_type));

    out.close();
}