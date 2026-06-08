#ifndef MATRIX_H
#define MATRIX_H

#include "field.h"
#include <cstdint>
#include <vector>
#include <cassert>

class Matrix {
public:
    Matrix() : m_rows(0), m_cols(0) {}
    Matrix(uint32_t rows, uint32_t cols) : m_rows(rows), m_cols(cols), m_data(static_cast<size_t>(rows) * cols, 0) {}

    Element& at(uint32_t row, uint32_t col) {
        assert(row < m_rows && col < m_cols);
        return m_data[static_cast<size_t>(row) * m_cols + col];
    }
    const Element& at(uint32_t row, uint32_t col) const {
        assert(row < m_rows && col < m_cols);
        return m_data[static_cast<size_t>(row) * m_cols + col];
    }

    uint32_t rows() const { return m_rows; }
    uint32_t cols() const { return m_cols; }
    Element* data() { return m_data.data(); }
    const Element* data() const { return m_data.data(); }
    size_t size() const { return m_data.size(); }

    Matrix block(uint32_t bi, uint32_t bj, uint32_t b) const {
        Matrix out(b, b);
        for (uint32_t r = 0; r < b; ++r)
            for (uint32_t c = 0; c < b; ++c)
                out.at(r, c) = at(bi * b + r, bj * b + c);
        return out;
    }

    void set_block(uint32_t bi, uint32_t bj, uint32_t b, const Matrix& blk) {
        for (uint32_t r = 0; r < b; ++r)
            for (uint32_t c = 0; c < b; ++c)
                at(bi * b + r, bj * b + c) = blk.at(r, c);
    }

    Matrix operator+(const Matrix& rhs) const {
        Matrix out(m_rows, m_cols);
        for (size_t i = 0; i < m_data.size(); ++i)
            out.m_data[i] = field::add(m_data[i], rhs.m_data[i]);
        return out;
    }

    Matrix operator-(const Matrix& rhs) const {
        Matrix out(m_rows, m_cols);
        for (size_t i = 0; i < m_data.size(); ++i)
            out.m_data[i] = field::sub(m_data[i], rhs.m_data[i]);
        return out;
    }

    Matrix operator*(const Matrix& rhs) const;

private:
    uint32_t m_rows, m_cols;
    std::vector<Element> m_data;
};

#endif
