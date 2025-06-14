#include <cuda_runtime.h>

#include <armadillo>
#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "constants.hpp"
#include "cuda_batch_fft2d.h"

using namespace arma;

namespace {
constexpr auto T = constants::tile_size;

constexpr auto device_complex64_deleter = [](arma::cx_float* ptr) { cudaFree(ptr); };
using DeviceComplex64 = std::unique_ptr<arma::cx_float[], decltype(device_complex64_deleter)>;

}  // namespace

SCENARIO("Forward FFT is valid", "[high_res_init]") {
    GIVEN("Blank image") {
        DeviceComplex64 src_buffer{[]() {
                                       cx_float* ptr = nullptr;
                                       const auto error_code =
                                           cudaMallocManaged(&ptr, sizeof(cx_float) * T * T);
                                       REQUIRE(error_code == 0);
                                       return ptr;
                                   }(),
                                   device_complex64_deleter};

        DeviceComplex64 dst_buffer{[]() {
                                       cx_float* ptr = nullptr;
                                       const auto error_code =
                                           cudaMallocManaged(&ptr, sizeof(cx_float) * T * T);
                                       REQUIRE(error_code == 0);
                                       return ptr;
                                   }(),
                                   device_complex64_deleter};

        {
            // Fill values of ones.
            cx_fmat src{src_buffer.get(), T, T, false, true};
            src.fill(1.0f);
        }

        {
            // Fill impossible numbers
            cx_fmat dst{dst_buffer.get(), T, T, false, true};
            dst.fill(datum::nan);
        }

        WHEN("Compute forward FFT") {
            CudaBatchFft2d fftpack{1, T, T};

            fftpack.dft2(reinterpret_cast<float2_t*>(src_buffer.get()),
                         reinterpret_cast<float2_t*>(dst_buffer.get()));
            cudaDeviceSynchronize();

            THEN("All zeros except at the center") {
                cx_fmat dst{dst_buffer.get(), T, T, false, true};
                REQUIRE(imag(dst).is_zero());

                REQUIRE(max(vectorise(real(dst))) > 0.0f);
                REQUIRE(real(dst(0, 0)) > 0.0f);
                REQUIRE(abs(real(dst(0, 0)) - T * T) < 1e-3f);

                dst(0, 0) = 0.0f;
                REQUIRE(real(dst).is_zero());
            }
        }
    }
}