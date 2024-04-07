#include <complex>
#include <mkl.h>
#include <vector>

// helper function for fft and ifft:
DFTI_DESCRIPTOR *create_descriptor(MKL_LONG length)
{
    DFTI_DESCRIPTOR *handle = nullptr;
    // using DFTI_DOUBLE for double precision
    // using DFTI_REAL for using the real version
    bool valid = (DFTI_NO_ERROR == DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_REAL, 1, length)) &&
                 // the result should not be inplace:
                 (DFTI_NO_ERROR == DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)) &&
                 // make clear that the result should be a vector of complex:
                 (DFTI_NO_ERROR == DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
    // chosen normalization is fft(constant)[0] = constant:
    (DFTI_NO_ERROR == DftiSetValue(handle, DFTI_FORWARD_SCALE, 1. / length)) &&
        (DFTI_NO_ERROR == DftiCommitDescriptor(handle));
    if (!valid)
    {
        DftiFreeDescriptor(&handle);
        return nullptr; // nullptr means error
    }
    return handle;
}

std::vector<std::complex<double>> real_fft(std::vector<double> &in)
{
    size_t out_size = in.size() / 2 + 1; // so many complex numbers needed
    std::vector<std::complex<double>> result(out_size);
    DFTI_DESCRIPTOR *handle = create_descriptor(static_cast<MKL_LONG>(in.size()));
    bool valid = handle && (DFTI_NO_ERROR == DftiComputeForward(handle, in.data(), result.data()));
    if (handle)
    {
        valid &= (DFTI_NO_ERROR == DftiFreeDescriptor(&handle));
    }
    if (!valid)
    {
        result.clear(); // empty vector -> error
    }
    return result;
}

std::vector<double> real_fft(std::vector<std::complex<double>> &in, size_t original_size)
{
    size_t expected_size = original_size / 2 + 1;
    if (expected_size != in.size())
    {
        return {}; // empty vector -> error
    }
    std::vector<double> result(original_size);
    DFTI_DESCRIPTOR *handle = create_descriptor(static_cast<MKL_LONG>(original_size));
    bool valid = handle && (DFTI_NO_ERROR == DftiComputeBackward(handle, in.data(), result.data()));
    if (handle)
    {
        valid &= (DFTI_NO_ERROR == DftiFreeDescriptor(&handle));
    }
    if (!valid)
    {
        result.clear(); // empty vector -> error
    }
    return result;
}