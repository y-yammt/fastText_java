#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <cstring>

/*
 WARNING: Just note - this is a temporary test-class - all these cpp files will be removed.
*/

float distL2(const float* x, const float* y, int32_t d) {
    float dist = 0;
    for (auto i = 0; i < d; i++) {
        auto tmp = x[i] - y[i];
        dist += tmp * tmp;
    }
    return dist;
}

class ProductQuantizer {
    protected:
        const int32_t nbits_ = 8;
        const int32_t ksub_ = 1 << nbits_;
    public:
        ProductQuantizer() {}
        float assign_centroid(const float*, const float*, uint8_t*, int32_t) const;
};

float ProductQuantizer::assign_centroid(const float * x, const float* c0, uint8_t* code, int32_t d) const {
  const float* c = c0;
  float dis = distL2(x, c, d);
  code[0] = 0;
  for (auto j = 1; j < ksub_; j++) {
    c += d;
    float disij = distL2(x, c, d);
    if (disij < dis) {
      code[0] = (uint8_t) j;
      dis = disij;
    }
  }
  return dis;
}

void _print(std::string name, float* vec, int s) {
    std::cout << name << " (float)size=" << s << ":\n";
    for (int i = 0; i < s; i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << "\n";
}

void _print(std::string name, uint8_t* vec, int s) {
    std::cout << name << " (uint8_t)size=" << s << ":\n";
    for (int i = 0; i < s; i++) {
        std::cout << (int)vec[i] << "|";
    }
    std::cout << "\n";
}

int main(int argc, char** argv)
{
    std::cout << "------------" << "\n";
    float   _x[17]      = {-5.0, -4.0, -23.0, -24.0, -545.0, 546.0, 547.1, 553.2, 566.3, 577.4, 588.0, 599.0, 600.0, 601.0, 614.23, 620.0, 655.22253};
    float   _c0[17]     = {2.0, 3.23, 4.0, 33.3, 446.0, 543.1, -566.0, 590.0, 610.0, 611.0, 614.23, -620.0, 710.0, 722.0, 723.0, 731.0, 752.3333};
    uint8_t _codes[16]  = {8, 9, 0, 144, 0, 222, 1, 0, 0, 0, 0, 0, 0, 0, 0, 15};

    uint8_t * codes = _codes;
    float * c0 = _c0;
    float * x = _x;

    int s = 20;
    _print("original x", x, s);
    _print("original c0", c0, s);
    _print("original codes", codes, s);

    std::cout << "===============" << "\n";
    int d = 4;
/*
    const float* c = c0;
    for (int i = 0; i < 10; i++) {
        c += d;
        float disij = distL2(x, c, d);
        std::cout << "i=" << i << "\tdisij=" << disij << "\n";
    }
*/
    ProductQuantizer q;
    q.assign_centroid(x, c0, codes, 4);

    std::cout << "===============" << "\n";
    _print("after x", x, s);
    _print("after c0", c0, s);
    _print("after codes", codes, s);

    return 0;
}