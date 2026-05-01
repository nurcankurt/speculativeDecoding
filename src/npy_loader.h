// =============================================================================
// npy_loader.h - Minimal NumPy .npy file reader (no external dependencies)
// =============================================================================
// The .npy format stores arrays in a simple binary format:
//   - 6-byte magic number: \x93NUMPY
//   - 2-byte version (major.minor)
//   - Header length (2 bytes for v1, 4 bytes for v2)
//   - ASCII header: Python dict with 'descr', 'fortran_order', 'shape'
//   - Raw binary data in row-major (C) order
//
// This loader supports:
//   - float32 ('<f4') arrays for probability distributions
//   - int32 ('<i4') and int64 ('<i8') arrays for token indices
//   - 1D and 2D shapes
// =============================================================================

#ifndef NPY_LOADER_H
#define NPY_LOADER_H

#include <string>
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// NpyArray: Container for data loaded from a .npy file
// ---------------------------------------------------------------------------
struct NpyArray {
    std::vector<float> data_float;   // Populated when dtype is float32
    std::vector<int>   data_int;     // Populated when dtype is int32/int64
    std::vector<size_t> shape;       // Shape of the array (e.g., {k, 50257})
    std::string dtype;               // NumPy dtype string (e.g., "<f4")
    bool is_float;                   // true if float data, false if int data

    // Returns total number of elements (product of all dimensions)
    size_t total_elements() const {
        size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }

    // Returns number of dimensions
    size_t ndim() const { return shape.size(); }
};

// ---------------------------------------------------------------------------
// load_npy: Load a .npy file into an NpyArray struct
// ---------------------------------------------------------------------------
// Parameters:
//   filepath - Path to the .npy file
// Returns:
//   NpyArray with data and metadata populated
// Throws:
//   std::runtime_error if file cannot be opened or format is invalid
NpyArray load_npy(const std::string& filepath);

#endif // NPY_LOADER_H
