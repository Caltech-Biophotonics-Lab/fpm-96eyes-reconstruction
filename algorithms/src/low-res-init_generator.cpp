#include <Halide.h>

#include "complex.h"
#include "constants.hpp"
#include "linear_ops.h"
#include "vars.hpp"

namespace {
using namespace Halide;

using vars::k;
using vars::x;
using vars::y;

constexpr auto W = constants::width;
constexpr auto H = constants::height;

/** Given a stack of 16-bit raw images, apply gamma correction. */
class LowResInit : public Generator<LowResInit> {
   public:
    Input<Buffer<const uint16_t, 3>> raw{"raw"};
    Input<float> gamma{"gamma", 0.5f, 0.2f, 1.0f};
    Output<Buffer<float, 3>> amplitude{"amplitude"};

    void generate();
    void schedule();

   private:
    void setBounds();

    Func repeated_edge;
    Func interpolated;
};

void
LowResInit::generate() {
    std::tie(interpolated, repeated_edge) =
        linear_ops::deinterleaveGreen(raw, raw.width(), raw.height());
    amplitude = linear_ops::adjustBrightness(interpolated, gamma, 0.0f, 255.0f);
}

void
LowResInit::setBounds() {
    raw.dim(0).set_bounds(0, W).set_stride(1);
    raw.dim(1).set_bounds(0, H).set_stride(W);
    raw.dim(2).set_bounds(0, 2).set_stride(W * H);

    amplitude.dim(0).set_bounds(0, W).set_stride(1);
    amplitude.dim(1).set_bounds(0, H).set_stride(W);
    amplitude.dim(2).set_bounds(0, 2).set_stride(W * H);
}

void
LowResInit::schedule() {
    assert(using_autoscheduler() && "Manual schedule not implemented");

    setBounds();

    constexpr auto n_illuminations = 49;
    raw.set_estimates({{0, W}, {0, H}, {0, n_illuminations}});
    amplitude.set_estimates({{0, W}, {0, H}, {0, n_illuminations}});
}

}  // namespace

HALIDE_REGISTER_GENERATOR(LowResInit, low_res_init)