#include <Halide.h>

#include "complex.h"
#include "constants.hpp"
#include "linear_ops.h"
#include "types.h"
#include "vars.hpp"

namespace {
using namespace Halide;

using constants::tile_size;
using linear_ops::fft2C2C;
using std::ignore;
using vars::i;
using vars::x;
using vars::y;

class HighResRestore : public Generator<HighResRestore> {
   public:
    Input<Buffer<const float, 3>> f_high_res{"f_high_res"};
    Output<Buffer<float, 3>> high_res{"high_res"};

    GeneratorParam<float> gain{"gain", 1.0f / tile_size / tile_size, 1e-12f, 1.0f};

    void generate();
    void schedule();

   private:
    void setBounds();

    ComplexFunc cropped{"cropped"};
    Func ifft_internal;
    ComplexFunc ifft_transformed{"ifft_transformed"};
    Func f_high_res_internal;
};

void
HighResRestore::generate() {
    // Since we don't have darkfield images, the superresolution Nyquist
    // bandwidth is far less than 4x of the raw images. Crop to the center of
    // the Fourier domain. This is equivalent to downsampling in spatial domain.
    {
        using namespace types;
        static_assert(tile_size % 2 == 0,
                      "Tile size must be an even number for 2D sub-sampling to work.");
        cropped(x, y) = {f_high_res(x + tile_size / 2, y + tile_size / 2, RE) * gain,
                         f_high_res(x + tile_size / 2, y + tile_size / 2, IM) * gain};
    }

    // Compute the Spatial domain of the high-resolution image
    constexpr bool INVERSE = false;
    std::tie(ifft_transformed, ifft_internal, f_high_res_internal) =
        fft2C2C(cropped, tile_size, INVERSE, "f_high_res_internal");

    // iFFTShift in Fourier space is equivalent to phase ramp in spatial domain.
    const auto [phase_shifted, sign] = linear_ops::applyCheckerboard(ifft_transformed);

    // Mux the real and imaginary component to conform to the Numpy data type: complex64.
    high_res(i, x, y) = mux(i, {phase_shifted(x, y).re(), phase_shifted(x, y).im()});
}

void
HighResRestore::setBounds() {
    constexpr auto T = tile_size;
    high_res.dim(0).set_bounds(0, 2).set_stride(1);
    high_res.dim(1).set_bounds(0, T).set_stride(2);
    high_res.dim(2).set_bounds(0, T).set_stride(T * 2);

    constexpr auto T2 = tile_size * 2;
    f_high_res.dim(0).set_bounds(0, T2).set_stride(1);
    f_high_res.dim(1).set_bounds(0, T2).set_stride(T2);
    f_high_res.dim(2).set_bounds(0, 2).set_stride(T2 * T2);
}

void
HighResRestore::schedule() {
    assert(!using_autoscheduler() && "Autoschedule not implemented");
    assert(get_target().has_gpu_feature() && "Only GPU implementation is supported.");

    setBounds();

    const Var xi, yi;

    high_res
        .gpu_tile(x, y, xi, yi, 128, 1)  //
        .unroll(i);

    ifft_internal.compute_root();

    f_high_res_internal.compute_root()
        .gpu_tile(x, y, xi, yi, 128, 1)  //
        .unroll(i);
}

}  // namespace

HALIDE_REGISTER_GENERATOR(HighResRestore, high_res_restore)