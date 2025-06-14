#include <Halide.h>

#include "complex.h"
#include "constants.hpp"
#include "linear_ops.h"
#include "vars.hpp"

namespace {
using namespace Halide;

using constants::tile_size;
using linear_ops::fft2C2C;
using std::ignore;
using vars::i;
using vars::k;
using vars::x;
using vars::y;
const Var kx{"kx"};
const Var ky{"ky"};

constexpr bool FORWARD = true;

class HighResInit : public Generator<HighResInit> {
   public:
    Input<Buffer<float, 3>> low_res{"low_res"};
    Output<Buffer<float, 3>> f_high_res{"f_high_res"};

    void generate();
    void schedule();

   private:
    void setBounds();

    ComplexFunc cx_low_res{"cx_low_res"};
    Func f_low_res_internal;
    Func low_res_internal;
};

void
HighResInit::generate() {
    // Select the 1st image and convert to complex value. Also multiply it with
    // a phase ramp; it is equivalent to the FFT2Shift in Fourier space.
    constexpr auto first_frame_id = 0;
    cx_low_res(x, y) = {low_res(x, y, first_frame_id), 0.0f};

    // Compute the Fourier domain of the brightfield image
    //
    // There is a faster FFT subroutine: cufftr2c which assumes hermitian
    // symmetry in Fourier domain. Since this operation is one-off outside the
    // FPM-EPRY loop, we choose not to implement it.
    std::tie(ignore, f_low_res_internal, low_res_internal) =
        fft2C2C(cx_low_res, tile_size, FORWARD, "f_low_res");

    // Demultiplex the real/imaginary components
    Func demux{"demux"};
    demux(kx, ky, i) = f_low_res_internal(i, kx, ky);

    // Unfold the FFT result to the infinite Fourier plane.
    const Func tiled = BoundaryConditions::repeat_image(demux, {{0, tile_size}, {0, tile_size}});

    // Interpolation in spatial domain is zero-padding in Fourier domain.
    Func zeropadded{"zeropadded"};
    const Expr is_in_xrange = (-tile_size / 2 <= kx) && (kx < tile_size / 2);
    const Expr is_in_yrange = (-tile_size / 2 <= ky) && (ky < tile_size / 2);
    zeropadded(kx, ky, i) = select(is_in_xrange && is_in_yrange,  //
                                   tiled(kx, ky, i), 0.0f);

    // Center the Fourier space to the coordinate (T, T). The full view is (2T, 2T).
    f_high_res(kx, ky, i) = zeropadded(kx - tile_size, ky - tile_size, i);
}

void
HighResInit::setBounds() {
    constexpr auto T2 = tile_size * 2;
    f_high_res.dim(0).set_bounds(0, T2).set_stride(1);
    f_high_res.dim(1).set_bounds(0, T2).set_stride(T2);
    f_high_res.dim(2).set_bounds(0, 2).set_stride(T2 * T2);

    constexpr auto T = tile_size;
    low_res.dim(0).set_bounds(0, T).set_stride(1);
    low_res.dim(1).set_bounds(0, T).set_stride(T);
    low_res.dim(2).set_min(0).set_stride(T * T);
}

void
HighResInit::schedule() {
    assert(!using_autoscheduler() && "Autoschedule not implemented");
    assert(get_target().has_gpu_feature() && "Only GPU implementation is supported.");

    setBounds();

    const Var xi{"xi"};
    const Var yi{"yi"};

    f_high_res.reorder(i, kx, ky)
        .gpu_tile(kx, ky, xi, yi, 32, 32)  //
        .unroll(i);

    f_low_res_internal.compute_root();

    low_res_internal
        .compute_root()  //
        .gpu_tile(x, y, xi, yi, 128, 1)
        .unroll(i);
}

}  // namespace

HALIDE_REGISTER_GENERATOR(HighResInit, high_res_init)