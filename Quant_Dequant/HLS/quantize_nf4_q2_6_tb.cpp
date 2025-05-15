#include <iostream>
#include <cmath>
#include "quantize_nf4_q2_6.hpp"

#define Q_SIZE 65536
#define OUTPUT_WEIGHTS_SIZE (Q_SIZE / 2)
#define OUTPUT_Q1_SIZE (Q_SIZE / LAYER1_BLOCK_SIZE)
#define OUTPUT_Q2_SIZE (Q_SIZE / (LAYER1_BLOCK_SIZE * LAYER2_BLOCK_SIZE))

int main() {
    // Sample input data
    fixed8_t Q_bram[Q_SIZE];
    for (int i = 0; i < Q_SIZE; ++i) {
        // Fill with some values between -1.0 and 1.0
        Q_bram[i] = fixed8_t(std::sin(i * 0.00001)); 
    }

    // Output arrays
    ap_uint<8> output_weights[OUTPUT_WEIGHTS_SIZE];
    fixed8_t output_q1[OUTPUT_Q1_SIZE];
    fixed8_t output_q2[OUTPUT_Q2_SIZE];

    // Initialize output memory to something known (optional)
    for (int i = 0; i < OUTPUT_WEIGHTS_SIZE; ++i) output_weights[i] = 0;
    for (int i = 0; i < OUTPUT_Q1_SIZE; ++i) output_q1[i] = 0;
    for (int i = 0; i < OUTPUT_Q2_SIZE; ++i) output_q2[i] = 0;

    // Run quantization
    quantize_nf4_q2_6(
        Q_bram,
        output_weights,
        output_q1,
        output_q2,
        0,                  // start_addr
        Q_SIZE,             // end_addr
        0,                  // output_w_addr
        0,                  // output_q1_addr
        0                   // output_q2_addr
    );

    // Print a few output weights
    std::cout << "Output Weights (first 8 bytes):\n";
    for (int i = 0; i < 8; ++i) {
        std::cout << "Byte " << i << ": 0x" << std::hex << (int)output_weights[i] << std::dec << "\n";
    }

    // Print q1 constants
    std::cout << "\nOutput Q1 constants:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "Q1[" << i << "] = " <<  output_q1[i].to_float() << "\n";
    }

    // Print q2 constants
    std::cout << "\nOutput Q2 constants:\n";
    for (int i = 0; i < OUTPUT_Q2_SIZE; ++i) {
        std::cout << "Q2[" << i << "] = " << output_q2[i].to_float() << "\n";
    }

    return 0;
}
