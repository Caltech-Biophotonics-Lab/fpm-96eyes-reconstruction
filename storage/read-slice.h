#pragma once

#include <highfive/H5DataSet.hpp>

#include "HalideBuffer.h"

namespace storage {

using slice_t = Halide::Runtime::Buffer<const uint16_t, 2>;
using u8_cube_t = Halide::Runtime::Buffer<uint8_t, 3>;
using cx_fcube_t = Halide::Runtime::Buffer<const float, 4>;

/** A region of interest (ROI). */
struct roi_t {
    size_t left{};
    size_t top{};
    size_t width{};
};

/** Helper function to read a plane from Z-stack fluorescence image from HDF5
 * dataset. */
slice_t readSlice(const HighFive::DataSet& dataset, size_t well_id, size_t z, size_t width,
                  size_t height);

/** Helper function to read a small region of interest (ROI) the FPM raw images
 * from HDF5 dataset. */
u8_cube_t readFPMRaw(const HighFive::DataSet& dataset, size_t well_id, roi_t,
                     const std::vector<size_t>& frame_id);

/** Helper function to read the FPM quantitative phase image (FPM-QPI) from HDF5
 * dataset. */
cx_fcube_t readQPILayers(const HighFive::DataSet& dataset, size_t well_id, size_t width,
                         size_t height, size_t n_layers = 4);

}  // namespace storage