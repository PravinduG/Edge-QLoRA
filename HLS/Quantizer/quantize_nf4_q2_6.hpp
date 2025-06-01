#ifndef QUANTIZE_NF4_Q2_6_HPP
#define QUANTIZE_NF4_Q2_6_HPP

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

#define LAYER1_BLOCK_SIZE 64
#define LAYER2_BLOCK_SIZE 256
#define NUM_NF4_CODES 16

typedef ap_fixed<8, 2> fixed8_t;
typedef ap_uint<4> nf4_t;


// Top-level function declaration
void quantize_nf4_q2_6(
    fixed8_t *Q_bram,
    ap_uint<8> *output_weights,     // Packed 2x4-bit NF4 codes per byte
    fixed8_t *output_q1,            // 1 per 64 output_weight
    fixed8_t *output_q2,            // 1 per 256 output_q1
    int start_addr, int end_addr,   // Range in Q_bram
    int output_w_addr,              // Base output address for weights
    int output_q1_addr,             // Base output for first layer quant constants
    int output_q2_addr              // Base output for second layer quant constants
);


#endif // QUANTIZE_NF4_Q2_6_H
