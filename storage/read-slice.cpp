#include "read-slice.h"

#include <highfive/H5File.hpp>

// Patch to encode std::complex<float> in HDF5 file.
#include "complex_float_support.hpp"

using Halide::Runtime::Buffer;

namespace storage {

slice_t
readSlice(const HighFive::DataSet& dataset, size_t well_id, size_t z, size_t width, size_t height) {
    Buffer<uint16_t, 2> image(width, height);
    dataset.select({z, well_id, 0, 0}, {1, 1, height, width}).read(image.data());
    return image;
}

u8_cube_t
readFPMRaw(const HighFive::DataSet& dataset, size_t well_id, roi_t roi,
           const std::vector<size_t>& frame_id) {
    const auto W = roi.width;

    assert(!frame_id.empty());
    Buffer<uint8_t, 3> low_res_images(W, W, frame_id.size());

    assert(std::all_of(frame_id.begin(), frame_id.end(), [](const auto& id) { return id < 49; }));

    auto* ptr = low_res_images.data();
    for (const auto& id : frame_id) {
        dataset.select({id, well_id, roi.top, roi.left}, {1, 1, W, W}).read(ptr);
        ptr += W * W;
    }
    return low_res_images;
}

cx_fcube_t
readQPILayers(const HighFive::DataSet& dataset, size_t well_id, size_t width, size_t height,
              size_t n_layers) {
    Halide::Runtime::Buffer<float, 4> raw(2, width, height, n_layers);

    dataset.select({0, well_id, 0, 0}, {n_layers, 1, height, width})
        .read(reinterpret_cast<std::complex<float>*>(raw.data()));

    return raw;
}

}  // namespace storage