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

constexpr auto T = constants::tile_size;

/** Given a stack of 16-bit raw images, apply gamma correction. */
class LowResInit : public Generator<LowResInit> {
   public:
    Input<Buffer<const uint8_t, 3>> raw{"raw"};
    Input<float> gamma{"gamma", 0.5f, 0.2f, 1.0f};
    Output<Buffer<float, 3>> amplitude{"amplitude"};

    void generate();
    void schedule();

   private:
    void setBounds();

    Func average_brightness{"average_brightness"};
    Func normalized{"normalized"};
};

void
LowResInit::generate() {
    const RDom r{0, T, 0, T, "all_pixels"};
    average_brightness(k) = 0.0f;
    average_brightness(k) += raw(r.x, r.y, k);

    normalized(x, y, k) = average_brightness(0) / average_brightness(k) * raw(x, y, k);

    amplitude = linear_ops::adjustBrightness(normalized, gamma, 0.0f, 255.0f);
}

void
LowResInit::setBounds() {
    raw.dim(0).set_bounds(0, T).set_stride(1);
    raw.dim(1).set_bounds(0, T).set_stride(T);
    raw.dim(2).set_min(0).set_stride(T * T);

    amplitude.dim(0).set_bounds(0, T).set_stride(1);
    amplitude.dim(1).set_bounds(0, T).set_stride(T);
    amplitude.dim(2).set_min(0).set_stride(T * T);
}

void
LowResInit::schedule() {
    assert(using_autoscheduler() && "Manual schedule not implemented");

    setBounds();

    constexpr auto n_illuminations = 49;
    raw.set_estimates({{0, T}, {0, T}, {0, n_illuminations}});
    gamma.set_estimate(0.6f);
    amplitude.set_estimates({{0, T}, {0, T}, {0, n_illuminations}});
}

}  // namespace

HALIDE_REGISTER_GENERATOR(LowResInit, low_res_init)