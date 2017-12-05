#include <string>
#include <random>
#include <vector>
#include <iostream>
#include <stdint.h>

#include "real.h"
#include "matrix.h"
#include "vector.h"
#include "qmatrix.h"

using namespace fasttext;

/**
To compile: `g++ -Wall -std=c++0x real.h utils.h utils.cc vector.h vector.cc matrix.h matrix.cc productquantizer.h productquantizer.cc qmatrix.h qmatrix.cc test_matrix.cpp -o test_matrix`
**/
int main (int argc, char** argv)
{
    int m_size = 300;
    int n_size = 10;
    Matrix m(m_size, n_size);
    m.zero();
    for (int i = 0; i < m_size; i++) {
        for (int j = 0; j < n_size; j++) {
            m.data_[i * n_size + j] = real(i) * m_size + j;
        }
    }
    std::cout <<  m.at(1,1) << "\n";
    std::cout <<  m.at(11,9) << "\n";
    std::cout <<  m.at(9, 8) << "\n\n";

    Vector v(10);
    v.zero();
    std::cout << v << "\n";

    QMatrix q(m, 2, true);
    std::cout <<  q.pq_->centroids_.size() << "::\n";
    std::cout <<  q.npq_->centroids_.size() << "::\n";
    std::cout << "codesize:" << q.codesize_ << "\n";
    std::cout << "codes:" << (int)q.codes_[0] << "|" << (int)q.codes_[1] << "\n";
    std::cout << "norm_codes:" << (int)q.norm_codes_[0] << "|" << (int)q.norm_codes_[1] << "\n";

    v.addRow(q, 12);
    std::cout << v << "\n";

    return 0;
}