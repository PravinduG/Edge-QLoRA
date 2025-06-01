#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

#define MAX_SIZE 589824 // 768 * 768
#define LAYER1_BLOCK_SIZE 64
#define LAYER2_BLOCK_SIZE 256
#define NUM_NF4_CODES 16

typedef ap_fixed<8, 2> fixed8_t;
typedef ap_uint<4> nf4_t; // 4-bit NF4 code

// Lookup table values (Q2.6)
const fixed8_t lookup_values[NUM_NF4_CODES] = {
    fixed8_t(-1.0), fixed8_t(-0.6961928), fixed8_t(-0.52507305), fixed8_t(-0.39491748),
    fixed8_t(-0.28444138), fixed8_t(-0.18477343), fixed8_t(-0.09105003), fixed8_t(0.0),
    fixed8_t(0.0795803), fixed8_t(0.1609302), fixed8_t(0.2461123), fixed8_t(0.33791524),
    fixed8_t(0.44070983), fixed8_t(0.562617), fixed8_t(0.72295684), fixed8_t(1.0)
};

// Fixed-point absolute
inline fixed8_t fixed_abs(fixed8_t val) {
    return (val < 0) ? fixed8_t(-val) : val;
}

// Quantize to nearest NF4 index
inline nf4_t quantize_to_nf4_index(fixed8_t value) {

    // Create copy of lookup table and partition
    fixed8_t lookup_values[NUM_NF4_CODES];
    #pragma HLS BIND_STORAGE variable=lookup_values type=ram_1p impl=bram
    #pragma HLS ARRAY_PARTITION variable=lookup_values complete dim=1

    nf4_t best_idx = 0;
    fixed8_t min_dist = fixed_abs(value - lookup_values[0]);

    for (int i = 1; i < NUM_NF4_CODES; i++) {
        #pragma HLS UNROLL
        fixed8_t dist = fixed_abs(value - lookup_values[i]);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

// Top-level function

void quantize_nf4_q2_6(
    fixed8_t *Q_bram,               // Unquantized weights. Implement bram in top level func and pass in as pointer
    ap_uint<8> *output_weights,     // Packed 2x4-bit NF4 codes per byte
    fixed8_t *output_q1,            // 1 per 64 output_weight
    fixed8_t *output_q2,            // 1 per 256 output_q1
    int start_addr, int end_addr,   // Range in Q_bram
    int output_w_addr,              // Base output address for weights
    int output_q1_addr,             // Base output for first layer quant constants
    int output_q2_addr              // Base output for second layer quant constants
) {
// #pragma HLS INTERFACE m_axi port=Q_bram offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi     port=Q_bram         offset=slave   bundle=gmem depth = MAX_SIZE max_read_burst_length=16 max_write_burst_length=16 // data_width=32
#pragma HLS INTERFACE m_axi     port=output_weights offset=slave   bundle=gmem depth = MAX_SIZE max_read_burst_length=16 max_write_burst_length=16 // data_width=32
#pragma HLS INTERFACE m_axi     port=output_q1      offset=slave   bundle=gmem depth = MAX_SIZE max_read_burst_length=16 max_write_burst_length=16 // data_width=32
#pragma HLS INTERFACE m_axi     port=output_q2      offset=slave   bundle=gmem depth = MAX_SIZE max_read_burst_length=16 max_write_burst_length=16 // data_width=32
#pragma HLS INTERFACE s_axilite port=start_addr     bundle=control
#pragma HLS INTERFACE s_axilite port=end_addr       bundle=control
#pragma HLS INTERFACE s_axilite port=output_w_addr  bundle=control
#pragma HLS INTERFACE s_axilite port=output_q1_addr bundle=control
#pragma HLS INTERFACE s_axilite port=output_q2_addr bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control

    for (int addr = start_addr; addr < end_addr; addr += LAYER1_BLOCK_SIZE * LAYER2_BLOCK_SIZE) {
        // Actual number of elements left
        int remaining = end_addr - addr;
        int blocks_64 = (remaining >= LAYER1_BLOCK_SIZE * LAYER2_BLOCK_SIZE) ? LAYER2_BLOCK_SIZE : (remaining + LAYER1_BLOCK_SIZE-1) / LAYER1_BLOCK_SIZE;

        fixed8_t q1_constants[LAYER2_BLOCK_SIZE];
        #pragma HLS ARRAY_PARTITION variable=q1_constants complete

        // Level 1 Quantization (64 elements per block)
        for (int b = 0; b < blocks_64; b++) {
            int base = addr + b * LAYER1_BLOCK_SIZE;
            int blk_size = ((base + LAYER1_BLOCK_SIZE) <= end_addr) ? LAYER1_BLOCK_SIZE : (end_addr - base);

            // Find max abs value
            fixed8_t max_val = 0;
            for (int i = 0; i < blk_size; i++) {
                #pragma HLS PIPELINE II=1
                fixed8_t v = Q_bram[base + i];
                fixed8_t abs_v = fixed_abs(v);
                if (abs_v > max_val) max_val = abs_v;
            }

            output_q1[output_q1_addr + b] = max_val;
            q1_constants[b] = max_val;

            // Quantize and pack weights
            for (int i = 0; i < blk_size; i += 2) {
                #pragma HLS PIPELINE II=1
                fixed8_t v0 = Q_bram[base + i];
                fixed8_t v1 = (i + 1 < blk_size) ? Q_bram[base + i + 1] : fixed8_t(0);
                fixed8_t n0 = (max_val != 0) ? fixed8_t(v0 / max_val) : fixed8_t(0);
                fixed8_t n1 = (max_val != 0) ? fixed8_t(v1 / max_val) : fixed8_t(0);
                nf4_t q0 = quantize_to_nf4_index(n0);
                nf4_t q1 = quantize_to_nf4_index(n1);
                ap_uint<8> packed = (q1, q0);
                output_weights[output_w_addr + b * (LAYER1_BLOCK_SIZE/2) + (i / 2)] = packed;
            }
        }

        // Level 2 Quantization (over 256 Q1 constants)
        fixed8_t max_q1 = 0;
        for (int i = 0; i < blocks_64; i++) {
            #pragma HLS PIPELINE II=1
            fixed8_t abs_q = fixed_abs(q1_constants[i]);
            if (abs_q > max_q1) max_q1 = abs_q;
        }

        output_q2[output_q2_addr++] = max_q1;
        output_q1_addr += blocks_64;
        output_w_addr += blocks_64 * LAYER1_BLOCK_SIZE/2;
    }

}
