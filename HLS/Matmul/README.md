# Matrix Multiplication Accelerator (Optimized)

This repository contains an optimized implementation of a matrix multiplication accelerator for FPGA.

The design is based on the original [`mmult_accel`](https://github.com/Richielee630/MatMul_SA/tree/21b25c48df306e05120cc9cfb09d7fa21cad8208) and its testbench (`mmult_accel_tb`), with the following improvements:

## Changes from the Original Code

- Implemented **tiling of matrix A** to reduce BRAM utilization.
- Reduced **I/O port usage** by setting constraints on the AXI interfaces.

## Reference

Original source code available at:  
[https://github.com/Richielee630/MatMul_SA/tree/21b25c48df306e05120cc9cfb09d7fa21cad8208](https://github.com/Richielee630/MatMul_SA/tree/21b25c48df306e05120cc9cfb09d7fa21cad8208)
