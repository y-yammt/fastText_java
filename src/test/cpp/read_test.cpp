#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdlib.h>

#include <iomanip>
#include <vector>

int main(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 2) {
        std::cerr << "Wrong input" << std::endl;
        exit(-2);
    }
    std::string file = args[1];

    std::cout << "Open:\t" << file << "\n";

    std::ifstream in(file, std::ifstream::binary);

    if (!in.is_open()) {
        std::cerr << "Can't open '" << file << "'" << std::endl;
        exit(-12);
    }

    int32_t _int;
    int64_t _long;
    float _float;
    double _double;
    std::string _string;
    float* _array;


    in.read((char*) &_int, sizeof(int32_t));
    in.read((char*) &_long, sizeof(int64_t));
    in.read((char*) &_float, sizeof(float));
    in.read((char*) &_double, sizeof(double));
    std::cout << "read string\n";
    char c;
    while ((c = in.get()) != 0) {
        _string.push_back(c);
    }
    _array = new float[2];
    in.read((char*) _array, 2 * sizeof(float));

    in.close();

    std::cout << _int << "\n" << _long << "\n";
    std::cout << std::fixed << std::setprecision(5) << _float << "\n" << _double << "\n" << _string << "\n";
    std::cout << _array << "\t" << _array[0] << ", " << _array[1] << "\n";


}