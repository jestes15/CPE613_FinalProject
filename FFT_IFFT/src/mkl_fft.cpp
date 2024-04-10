#include "fft.h"

DFTI_DESCRIPTOR *create_descriptor(MKL_LONG length)
{
    DFTI_DESCRIPTOR *handle = nullptr;
    bool valid = (DFTI_NO_ERROR == DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_REAL, 1, length)) &&
                 (DFTI_NO_ERROR == DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)) &&
                 (DFTI_NO_ERROR == DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
    (DFTI_NO_ERROR == DftiSetValue(handle, DFTI_FORWARD_SCALE, 1. / length)) &&
        (DFTI_NO_ERROR == DftiCommitDescriptor(handle));
    if (!valid)
    {
        DftiFreeDescriptor(&handle);
        return nullptr;
    }
    return handle;
}

std::vector<std::complex<float>> _forward_fft_R2C(std::vector<float> in)
{
    size_t out_size = in.size() / 2 + 1; // so many complex numbers needed
    std::vector<std::complex<float>> result(out_size);
    DFTI_DESCRIPTOR *handle = create_descriptor(static_cast<MKL_LONG>(in.size()));
    bool valid = handle && (DFTI_NO_ERROR == DftiComputeForward(handle, in.data(), result.data()));
    if (handle)
    {
        valid &= (DFTI_NO_ERROR == DftiFreeDescriptor(&handle));
    }
    if (!valid)
    {
        result.clear();
    }
    return result;
}
std::vector<std::complex<float>> _forward_fft_C2C(std::vector<std::complex<float>> in)
{
    std::vector<std::complex<float>> out(in.size());

    DFTI_DESCRIPTOR_HANDLE descriptor;

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1, in.size());
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiCommitDescriptor(descriptor);
    DftiComputeForward(descriptor, in.data(), out.data());
    DftiFreeDescriptor(&descriptor);

    return out;
}

std::vector<float> _backward_fft_C2R(std::vector<std::complex<float>> in, int original_size)
{
    size_t expected_size = original_size / 2 + 1;
    if (expected_size != in.size())
    {
        return {};
    }
    std::vector<float> result(original_size);
    DFTI_DESCRIPTOR *handle = create_descriptor(static_cast<MKL_LONG>(original_size));
    bool valid = handle && (DFTI_NO_ERROR == DftiComputeBackward(handle, in.data(), result.data()));
    if (handle)
    {
        valid &= (DFTI_NO_ERROR == DftiFreeDescriptor(&handle));
    }
    if (!valid)
    {
        result.clear();
    }
    return result;
}
std::vector<float> _backward_fft_C2R_Complex(std::vector<std::complex<float>> in)
{
    std::vector<std::complex<float>> out(in.size());

    DFTI_DESCRIPTOR_HANDLE descriptor;

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1, in.size());
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / in.size());
    DftiCommitDescriptor(descriptor);
    DftiComputeBackward(descriptor, in.data(), out.data());
    DftiFreeDescriptor(&descriptor);

    std::vector<float> output(out.size());

    for (std::size_t i = 0; i < out.size(); ++i)
    {
        output[i] = out[i].real();
    }

    return output;
}