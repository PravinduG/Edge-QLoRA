#include <iostream>
#include <cmath>
#include <chrono>  
#include "quantize_nf4_q2_6.hpp"
#include "dequantize_nf4_q2_6.hpp"
#include <random>

#define Q_SIZE 65536
#define OUTPUT_WEIGHTS_SIZE (Q_SIZE / 2)
#define OUTPUT_Q1_SIZE (Q_SIZE / LAYER1_BLOCK_SIZE)
#define OUTPUT_Q2_SIZE (Q_SIZE / (LAYER1_BLOCK_SIZE * LAYER2_BLOCK_SIZE))

int main() {

    // Set up random number generator for standard normal distribution
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 0.3);  // mean = 0, std = 1

    // Sample input data
    fixed8_t Q_bram[Q_SIZE];
    for (int i = 0; i < Q_SIZE; ++i) {
        float val = distribution(generator);
        val = std::max(-1.0f, std::min(1.0f, val));  // Clamp to [-1.0, 1.0]
        Q_bram[i] = fixed8_t(val);
    }
    // Output arrays
    ap_uint<8> output_weights[OUTPUT_WEIGHTS_SIZE];
    fixed8_t output_q1[OUTPUT_Q1_SIZE];
    fixed8_t output_q2[OUTPUT_Q2_SIZE];

    // For dequantized weights
    fixed8_t DQ_bram[Q_SIZE];

    // Initialize output memory to something known 
    for (int i = 0; i < OUTPUT_WEIGHTS_SIZE; ++i) output_weights[i] = 0;
    for (int i = 0; i < OUTPUT_Q1_SIZE; ++i) output_q1[i] = 0;
    for (int i = 0; i < OUTPUT_Q2_SIZE; ++i) output_q2[i] = 0;
    for (int i = 0; i < Q_SIZE; ++i) DQ_bram[i] = 0;


    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

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

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Print timing
    std::cout << "\nQuantization took " << elapsed.count() << " ms\n";

    // Print a few output weights
    std::cout << "Output Weights (first 8 bytes):\n";
    for (int i = 0; i < 8; ++i) {
        std::cout << "Byte " << i << ": 0x" << std::hex << (int)output_weights[i] << std::dec << "\n";
    }

    // Print q1 constants
    std::cout << "\nOutput Q1 constants:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "Q1[" << i << "] = " << output_q1[i].to_float() << "\n";
    }

    // Print q2 constants
    std::cout << "\nOutput Q2 constants:\n";
    for (int i = 0; i < OUTPUT_Q2_SIZE; ++i) {
        std::cout << "Q2[" << i << "] = " << output_q2[i].to_float() << "\n";
    }

    // Run dequantization 
    dequantize_nf4_q2_6(
        output_weights,
        output_q1,
        output_q2,
        0,          // start_addr
        Q_SIZE,     // end_addr
        0,          // input_w_addr
        0,          // input_q1_addr
        0,          // input_q2_addr
        DQ_bram
    );

    // Calculate RMSE between Q_bram and DQ_bram
    double mse = 0.0;
    for (int i = 0; i < Q_SIZE; ++i) {
        double diff = Q_bram[i].to_float() - DQ_bram[i].to_float();
        mse += diff * diff;
    }
    mse /= Q_SIZE;
    double rmse = std::sqrt(mse);

    std::cout << "Root Mean Square Error (RMSE) between original and dequantized: " << rmse << std::endl;

    std::cout << "Comparison:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << Q_bram[i] << " " << DQ_bram[i] << "\n";
    }

    return 0;
}
