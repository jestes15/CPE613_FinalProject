#ifndef D2BC1CD8_176D_4ECC_A8BE_A31B45F1FD77
#define D2BC1CD8_176D_4ECC_A8BE_A31B45F1FD77

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <immintrin.h>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<double> arange(double start, double stop, double step = 1)
{
    std::vector<double> values;
    for (double value = start; value < stop; value += step)
        values.push_back(value);
    return values;
}

template <typename Type_> class vector_ext : public std::vector<Type_>
{
  protected:
    static auto size_check(vector_ext<Type_> &vec1, vector_ext<Type_> &vec2)
    {
        if (vec1.size() != vec2.size())
        {
            const std::string err = "vector_ext::size_check: vec1.size() != vec2.size()";
            throw std::range_error(err);
        }
    }

  public:
    using std::vector<Type_>::vector;
    using std::vector<Type_>::operator=;
    using std::vector<Type_>::assign;
    using std::vector<Type_>::get_allocator;
    using std::vector<Type_>::operator[];
    using std::vector<Type_>::front;
    using std::vector<Type_>::back;
    using std::vector<Type_>::data;
    using std::vector<Type_>::begin;
    using std::vector<Type_>::cbegin;
    using std::vector<Type_>::end;
    using std::vector<Type_>::cend;
    using std::vector<Type_>::rbegin;
    using std::vector<Type_>::crbegin;
    using std::vector<Type_>::rend;
    using std::vector<Type_>::crend;
    using std::vector<Type_>::empty;
    using std::vector<Type_>::size;
    using std::vector<Type_>::max_size;
    using std::vector<Type_>::reserve;
    using std::vector<Type_>::capacity;
    using std::vector<Type_>::shrink_to_fit;
    using std::vector<Type_>::clear;
    using std::vector<Type_>::insert;
    using std::vector<Type_>::emplace;
    using std::vector<Type_>::erase;
    using std::vector<Type_>::push_back;
    using std::vector<Type_>::emplace_back;
    using std::vector<Type_>::pop_back;
    using std::vector<Type_>::resize;
    using std::vector<Type_>::swap;

    auto operator+(vector_ext<Type_> &vector_obj) -> vector_ext<Type_>
    {
        size_check(*this, vector_obj);
        vector_ext<Type_> ret_vec(this->size());

        for (uint64_t idx = 0; idx < this->size(); idx += 8)
        {
            ret_vec[idx] = this->at(idx) + vector_obj[idx];
            ret_vec[idx + 1] = this->at(idx + 1) + vector_obj[idx + 1];
            ret_vec[idx + 2] = this->at(idx + 2) + vector_obj[idx + 2];
            ret_vec[idx + 3] = this->at(idx + 3) + vector_obj[idx + 3];
            ret_vec[idx + 4] = this->at(idx + 4) + vector_obj[idx + 4];
            ret_vec[idx + 5] = this->at(idx + 5) + vector_obj[idx + 5];
            ret_vec[idx + 6] = this->at(idx + 6) + vector_obj[idx + 6];
            ret_vec[idx + 7] = this->at(idx + 7) + vector_obj[idx + 7];
        }

        return ret_vec;
    }

    auto operator-(vector_ext<Type_> &vector_obj) -> vector_ext<Type_>
    {
        size_check(*this, vector_obj);
        vector_ext<Type_> ret_vec(this->size());

        for (uint64_t idx = 0; idx < this->size(); idx += 8)
        {
            ret_vec[idx] = this->at(idx) - vector_obj[idx];
            ret_vec[idx + 1] = this->at(idx + 1) - vector_obj[idx + 1];
            ret_vec[idx + 2] = this->at(idx + 2) - vector_obj[idx + 2];
            ret_vec[idx + 3] = this->at(idx + 3) - vector_obj[idx + 3];
            ret_vec[idx + 4] = this->at(idx + 4) - vector_obj[idx + 4];
            ret_vec[idx + 5] = this->at(idx + 5) - vector_obj[idx + 5];
            ret_vec[idx + 6] = this->at(idx + 6) - vector_obj[idx + 6];
            ret_vec[idx + 7] = this->at(idx + 7) - vector_obj[idx + 7];
        }

        return ret_vec;
    }
};

#endif /* D2BC1CD8_176D_4ECC_A8BE_A31B45F1FD77 */
