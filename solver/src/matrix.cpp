#include "matrix.h"

Matrix Matrix::operator*(const Matrix& rhs) const {
    assert(m_cols == rhs.m_rows);
    Matrix out(m_rows, rhs.m_cols);
    for (uint32_t i = 0; i < m_rows; ++i) {
        for (uint32_t j = 0; j < rhs.m_cols; ++j) {
            Element acc = 0;
            for (uint32_t k = 0; k < m_cols; ++k) {
                acc = field::add(acc, field::mul(at(i, k), rhs.at(k, j)));
            }
            out.at(i, j) = acc;
        }
    }
    return out;
}
