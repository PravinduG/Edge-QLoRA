#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

#define LAYER1_BLOCK_SIZE 64
#define LAYER2_BLOCK_SIZE 256
#define NUM_NF4_CODES 16

// Maximum amount of weights to be dequantized at once
// const int MAX_SIZE = 65536; 

typedef ap_fixed<8, 2> fixed8_t;
typedef ap_uint<4> nf4_t; // 4-bit NF4 code

// Lookup table values (Q2.6)
const fixed8_t lookup_values[NUM_NF4_CODES] = {
    fixed8_t(-1.0), fixed8_t(-0.6961928), fixed8_t(-0.52507305), fixed8_t(-0.39491748),
    fixed8_t(-0.28444138), fixed8_t(-0.18477343), fixed8_t(-0.09105003), fixed8_t(0.0),
    fixed8_t(0.0795803), fixed8_t(0.1609302), fixed8_t(0.2461123), fixed8_t(0.33791524),
    fixed8_t(0.44070983), fixed8_t(0.562617), fixed8_t(0.72295684), fixed8_t(1.0)
};



// Top-level function

void dequantize_nf4_q2_6(
    ap_uint<8> *input_weights,           // Storage address for weights
    fixed8_t *input_q1,                  // Storage for first layer quant constants
    fixed8_t *input_q2,                  // Storage for second layer quant constants
    int start_addr, int end_addr,        // Range in output_weights
    int input_w_addr,                    // Base output address for weights
    int input_q1_addr,                   // Base output for first layer quant constants
    int input_q2_addr,                   // Base output for second layer quant constants
    fixed8_t *DQ_bram                    // Dequantized weights --> Allocate bram in top level function and pass pointer
    
) {
// #pragma HLS INTERFACE m_axi port=DQ_bram offset=slave bundle=gmem // Use BRAM. not DDR
#pragma HLS INTERFACE m_axi port=input_weights offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=input_q1 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=input_q2 offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=start_addr bundle=control
#pragma HLS INTERFACE s_axilite port=end_addr bundle=control
#pragma HLS INTERFACE s_axilite port=input_w_addr bundle=control
#pragma HLS INTERFACE s_axilite port=input_q1_addr bundle=control
#pragma HLS INTERFACE s_axilite port=input_q2_addr bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control


    // BRAM buffers
    ap_uint<8> weights_buf[LAYER1_BLOCK_SIZE * LAYER2_BLOCK_SIZE / 2]; // 8192 bytes
    fixed8_t q1_buf[LAYER2_BLOCK_SIZE];  // 256 values

    // Partition to enable full unrolling of loops
    #pragma HLS ARRAY_PARTITION variable=weights_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=q1_buf complete dim=1

    for (int addr = start_addr; addr < end_addr; addr += LAYER1_BLOCK_SIZE * LAYER2_BLOCK_SIZE) {
        int remaining = end_addr - addr;
        int blocks_64 = (remaining >= LAYER1_BLOCK_SIZE * LAYER2_BLOCK_SIZE)
                            ? LAYER2_BLOCK_SIZE
                            : (remaining + LAYER1_BLOCK_SIZE - 1) / LAYER1_BLOCK_SIZE;

        // Load Q2 scale (one per layer2 block)
        fixed8_t q2_scale = input_q2[input_q2_addr++];

        // Load Q1 block scales (256 values)
        for (int i = 0; i < blocks_64; i++) {
            // #pragma HLS PIPELINE II=1
            q1_buf[i] = input_q1[input_q1_addr++];
        }

        // Load 256 * 64 / 2 = 8192 bytes of quantized weights
        for (int i = 0; i < (blocks_64 * LAYER1_BLOCK_SIZE / 2); i++) {
            #pragma HLS PIPELINE II=1
            weights_buf[i] = input_weights[input_w_addr + i];
        }

        // Now process
        int out_index = addr;
        for (int b = 0; b < blocks_64; b++) {
            fixed8_t total_scale = q1_buf[b] * q2_scale;

            for (int i = 0; i < LAYER1_BLOCK_SIZE; i += 2) {
                #pragma HLS PIPELINE II=1
                int local_idx = b * (LAYER1_BLOCK_SIZE / 2) + (i / 2);
                ap_uint<8> packed = weights_buf[local_idx];

                nf4_t q0 = packed.range(3, 0);
                nf4_t q1 = packed.range(7, 4);

                fixed8_t v0 = lookup_values[q0] * total_scale;
                fixed8_t v1 = lookup_values[q1] * total_scale;

                if ((out_index + i) < end_addr)
                    DQ_bram[out_index + i] = v0;
                if ((out_index + i + 1) < end_addr)
                    DQ_bram[out_index + i + 1] = v1;
            }

            out_index += LAYER1_BLOCK_SIZE;
        }

        input_w_addr += blocks_64 * (LAYER1_BLOCK_SIZE / 2);
    }
}
