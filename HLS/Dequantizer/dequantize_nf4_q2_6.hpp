#ifndef DEQUANTIZE_NF4_Q2_6_HPP
#define DEQUANTIZE_NF4_Q2_6_HPP

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
void dequantize_nf4_q2_6(
    ap_uint<8> *input_weights,           // Storage address for weights
    fixed8_t *input_q1,                  // Storage for first layer quant constants
    fixed8_t *input_q2,                  // Storage for second layer quant constants
    int start_addr, int end_addr,        // Range in output_weights
    int input_w_addr,                    // Base output address for weights
    int input_q1_addr,                   // Base output for first layer quant constants
    int input_q2_addr,                   // Base output for second layer quant constants
    fixed8_t *DQ_bram                    // Dequantized weights --> Allocate bram in top level function and pass pointer
       
    
);


#endif // DEQUANTIZE_NF4_Q2_6_H
