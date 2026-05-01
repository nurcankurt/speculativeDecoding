// =============================================================================
// npy_loader.cpp - Implementation of minimal .npy file reader
// =============================================================================

#include "npy_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <iostream>

// ---------------------------------------------------------------------------
// Helper: Parse the Python dict header string to extract dtype and shape
// ---------------------------------------------------------------------------
// The header looks like:
//   {'descr': '<f4', 'fortran_order': False, 'shape': (10, 50257), }
// We parse it manually without a Python interpreter.
static void parse_header(const std::string& header, std::string& dtype,
                         bool& fortran_order, std::vector<size_t>& shape) {
    // --- Extract 'descr' value ---
    // Find "descr" key, then extract the string value between quotes
    size_t pos = header.find("'descr'");
    if (pos == std::string::npos) pos = header.find("\"descr\"");
    if (pos == std::string::npos) {
        throw std::runtime_error("npy_loader: 'descr' not found in header");
    }
    // Skip to the colon, then find the opening quote of the value
    pos = header.find(':', pos);
    size_t q1 = header.find_first_of("'\"", pos + 1);
    size_t q2 = header.find_first_of("'\"", q1 + 1);
    dtype = header.substr(q1 + 1, q2 - q1 - 1);

    // --- Extract 'fortran_order' value ---
    fortran_order = (header.find("True") != std::string::npos &&
                     header.find("fortran_order") != std::string::npos);
    // We only support C-order (row-major)
    if (fortran_order) {
        throw std::runtime_error("npy_loader: Fortran order not supported");
    }

    // --- Extract 'shape' tuple ---
    // Find the opening '(' after 'shape'
    pos = header.find("'shape'");
    if (pos == std::string::npos) pos = header.find("\"shape\"");
    if (pos == std::string::npos) {
        throw std::runtime_error("npy_loader: 'shape' not found in header");
    }
    size_t paren_open = header.find('(', pos);
    size_t paren_close = header.find(')', paren_open);
    std::string shape_str = header.substr(paren_open + 1,
                                          paren_close - paren_open - 1);

    // Parse comma-separated integers from the shape string
    shape.clear();
    std::istringstream ss(shape_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace),
                    token.end());
        if (!token.empty()) {
            shape.push_back(static_cast<size_t>(std::stoull(token)));
        }
    }
}

// ---------------------------------------------------------------------------
// load_npy: Main function to load a .npy file
// ---------------------------------------------------------------------------
NpyArray load_npy(const std::string& filepath) {
    // Open file in binary mode
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("npy_loader: Cannot open file: " + filepath);
    }

    // --- Read and verify magic number ---
    // Expected: \x93NUMPY (6 bytes)
    char magic[6];
    file.read(magic, 6);
    if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY") {
        throw std::runtime_error("npy_loader: Invalid .npy magic number in: "
                                 + filepath);
    }

    // --- Read version ---
    uint8_t major_ver, minor_ver;
    file.read(reinterpret_cast<char*>(&major_ver), 1);
    file.read(reinterpret_cast<char*>(&minor_ver), 1);

    // --- Read header length ---
    // Version 1.x: 2-byte unsigned short (little-endian)
    // Version 2.x+: 4-byte unsigned int (little-endian)
    uint32_t header_len = 0;
    if (major_ver == 1) {
        uint16_t hl16 = 0;
        file.read(reinterpret_cast<char*>(&hl16), 2);
        header_len = hl16;
    } else {
        file.read(reinterpret_cast<char*>(&header_len), 4);
    }

    // --- Read header string ---
    std::string header(header_len, '\0');
    file.read(&header[0], header_len);

    // --- Parse header to extract dtype and shape ---
    std::string dtype;
    bool fortran_order;
    std::vector<size_t> shape;
    parse_header(header, dtype, fortran_order, shape);

    // --- Calculate total elements ---
    size_t total = 1;
    for (auto s : shape) total *= s;

    // --- Build result ---
    NpyArray result;
    result.shape = shape;
    result.dtype = dtype;

    // --- Read raw data based on dtype ---
    if (dtype == "<f4" || dtype == "float32") {
        // 32-bit float, little-endian
        result.is_float = true;
        result.data_float.resize(total);
        file.read(reinterpret_cast<char*>(result.data_float.data()),
                  total * sizeof(float));
    } else if (dtype == "<f8" || dtype == "float64") {
        // 64-bit float - convert to 32-bit
        result.is_float = true;
        result.data_float.resize(total);
        std::vector<double> temp(total);
        file.read(reinterpret_cast<char*>(temp.data()),
                  total * sizeof(double));
        for (size_t i = 0; i < total; i++) {
            result.data_float[i] = static_cast<float>(temp[i]);
        }
    } else if (dtype == "<i4" || dtype == "int32") {
        // 32-bit integer, little-endian
        result.is_float = false;
        result.data_int.resize(total);
        file.read(reinterpret_cast<char*>(result.data_int.data()),
                  total * sizeof(int));
    } else if (dtype == "<i8" || dtype == "int64") {
        // 64-bit integer - convert to 32-bit
        result.is_float = false;
        result.data_int.resize(total);
        std::vector<int64_t> temp(total);
        file.read(reinterpret_cast<char*>(temp.data()),
                  total * sizeof(int64_t));
        for (size_t i = 0; i < total; i++) {
            result.data_int[i] = static_cast<int>(temp[i]);
        }
    } else {
        throw std::runtime_error("npy_loader: Unsupported dtype: " + dtype);
    }

    if (!file) {
        throw std::runtime_error("npy_loader: Error reading data from: "
                                 + filepath);
    }

    std::cout << "[npy_loader] Loaded " << filepath
              << " | dtype=" << dtype << " | shape=(";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i + 1 < shape.size()) std::cout << ", ";
    }
    std::cout << ") | " << total << " elements" << std::endl;

    return result;
}
