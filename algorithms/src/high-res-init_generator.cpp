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

constexpr bool FORWARD = true;

class HighResInit : public Generator<HighResInit> {
   public:
    Input<Buffer<float, 3>> low_res{"low_res"};
    Output<Buffer<float, 3>> f_high_res{"f_high_res"};

    void generate();
    void schedule();

   private:
    void setBounds();

    Func cx_low_res{"cx_low_res"};
    Func f_low_res_internal;
};

void
HighResInit::generate() {
    // Select the 1st image and convert to complex value.
    cx_low_res(i, x, y) = mux(i, {low_res(x, y, 0), 0.0f});

    // Compute the Fourier domain of the brightfield image
    Func low_res_internal;

    std::tie(ignore, f_low_res_internal, low_res_internal) =
        fft2C2C(cx_low_res, tile_size, FORWARD, "f_low_res");

    // Take only the first brightfield image.
    Func first_frame;
    first_frame(x, y, i) = f_low_res_internal(i, x, y);

    // Interpolation in spatial domain is zero-padding in Fourier domain.
    f_high_res =
        BoundaryConditions::constant_exterior(first_frame, 0.0f, {{0, tile_size}, {0, tile_size}});
}

void
HighResInit::setBounds() {
    constexpr auto T = tile_size;
    f_high_res.dim(0).set_bounds(0, T).set_stride(1);
    f_high_res.dim(1).set_bounds(0, T).set_stride(T);
    f_high_res.dim(2).set_bounds(0, 2).set_stride(T * T);

    low_res.dim(0).set_bounds(0, T).set_stride(1);
    low_res.dim(1).set_bounds(0, T).set_stride(T);
    low_res.dim(2).set_min(0).set_stride(T * T);
}

void
HighResInit::schedule() {
    assert(!using_autoscheduler() && "Autoschedule not implemented");
    assert(get_target().has_gpu_feature() && "Only GPU implementation is supported.");

    setBounds();

    const Var xi, yi;

    f_high_res
        .gpu_tile(x, y, xi, yi, 128, 1)  //
        .unroll(i);

    f_low_res_internal.compute_root();

    cx_low_res.compute_root().gpu_tile(x, y, xi, yi, 128, 1).unroll(i);
}

}  // namespace

HALIDE_REGISTER_GENERATOR(HighResInit, high_res_init)