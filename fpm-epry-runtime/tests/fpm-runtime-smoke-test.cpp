#include <armadillo>
#include <catch2/catch_test_macros.hpp>

#include "constants.hpp"
#include "fpm-epry-runtime.h"

using namespace arma;
using constants::tile_size;
using Halide::Runtime::Buffer;
using reconstruction::ComplexBuffer;

SCENARIO("Can run EPRY algorithm smoothly", "[runner]") {
    constexpr auto n_illuminations = 25;
    GIVEN("Raw data") {
        Mat<int32_t> k_offset(2, n_illuminations, fill::zeros);
        ComplexBuffer pupil{2, tile_size, tile_size};
        Buffer<uint8_t, 3> raw{tile_size, tile_size, n_illuminations};

        pupil.fill(0.0f);
        raw.fill(128);

        WHEN("Initialize FPMEpryRunner") {
            reconstruction::FPMEpryRunner runner{std::move(k_offset), std::move(pupil),
                                                 std::move(raw)};
            REQUIRE(runner.n_illuminations == n_illuminations);

            THEN("Can reconstruct images") {
                runner.reconstruct(5);

                AND_THEN("Can retrieve new pupil and high-res image") {
                    const auto new_pupil = runner.downloadPupil();
                }

                AND_THEN("Can download high res image") {
                    const auto high_res = runner.computeHighRes();
                }
            }
        }
    }
}
