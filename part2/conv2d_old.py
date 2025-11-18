import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal

KERNEL_DTYPE = nl.float32

"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512
    c_in_pmax = nl.tile_size.pmax
    c_out_pmax = TILE_M # TODO: can be TILE_N, but since out_channels is only guaranteed to be a multiple of 128, setting it to TILE_N would need additional edge case handling.
    
    out_tile_size = (1, out_width)  # (hardcoded here to fit nl.tile_size.gemm_stationary_fmax, which is 128)
    in_tile_size = (out_tile_size[0]+filter_height-1, out_tile_size[1]+filter_width-1)
    # last_out_tile_h = out_tile_size[0]
    # last_out_tile_w = out_tile_size[1]
    # if out_height%out_tile_size[0] != 0:
    #     last_out_tile_h = out_height%out_tile_size[0]
    # if out_width%out_tile_size[1] != 0:
    #     last_out_tile_w = out_width%out_tile_size[1]
    # last_in_tile_h = last_out_tile_h + filter_height - 1
    # last_in_tile_w = last_out_tile_w + filter_width - 1
    # print(f"\n\nUsing out_tile_size: {out_tile_size}, last_out_tile_h: {last_out_tile_h}, last_out_tile_w: {last_out_tile_w}, last_in_tile_h: {last_in_tile_h}, last_in_tile_w: {last_in_tile_w}\n\n")
    print(f"\n image has shape {input_height} x {input_width}, conv output shape {out_height} x {out_width}")
    
    
    # Reshape W
    n_c_in_tiles = in_channels // c_in_pmax
    n_c_out_tiles = out_channels // c_out_pmax
    
    W = W.reshape((c_out_pmax, n_c_out_tiles, c_in_pmax, n_c_in_tiles, filter_height, filter_width))
    W_sbuf_old = nl.ndarray((c_out_pmax, n_c_out_tiles, c_in_pmax, n_c_in_tiles, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=W_sbuf_old, src=W)
    W_sbuf = nl.ndarray((c_out_pmax, n_c_out_tiles, c_in_pmax, n_c_in_tiles, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
    
    for c_out_i in nl.affine_range(out_channels // c_out_pmax):
        for c_in_i in nl.affine_range(in_channels // c_in_pmax):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    W_tile = W_sbuf_old[:, c_out_i, :, c_in_i, i, j]
                    W_tile = nisa.nc_transpose(W_tile)
                    W_sbuf[:, c_out_i, :, c_in_i, i, j] = nisa.tensor_copy(src=W_tile)

    bias = bias.reshape((c_out_pmax, n_c_out_tiles))
    bias_sbuf = nl.ndarray((c_out_pmax, n_c_out_tiles, 1), dtype=bias.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=bias_sbuf, src=bias)
    
    out_tile_h = out_tile_size[0]
    out_tile_w = out_tile_size[1]
    in_tile_h = in_tile_size[0]
    in_tile_w = in_tile_size[1]
    for b in nl.affine_range(batch_size):
        for out_i in nl.sequential_range(0, out_height, out_tile_size[0]):
            for out_j in nl.affine_range(0, out_width, out_tile_size[1]):
                for c_out_i in nl.affine_range(out_channels // c_out_pmax):
                    c_out_start = c_out_i*c_out_pmax
                    c_out_end = (c_out_i+1)*c_out_pmax
                    
                    # bias_sbuf = nl.ndarray((c_out_pmax, 1), dtype=bias.dtype, buffer=nl.sbuf)
                    # nisa.dma_copy(dst=bias_sbuf, src=bias[c_out_start:c_out_end])
    
                    res_psum = nisa.memset(
                        shape=(out_tile_h*out_tile_w, c_out_pmax),
                        value=0.0,
                        dtype=KERNEL_DTYPE,
                    )
                    res_sbuf = nl.ndarray((c_out_pmax, out_tile_h, out_tile_w), dtype=X.dtype, buffer=nl.sbuf)
                    for c_in_i in nl.affine_range(in_channels // c_in_pmax):
                        c_in_start = c_in_i*c_in_pmax
                        c_in_end = (c_in_i+1)*c_in_pmax
                        
                        X_buf = nl.ndarray((c_in_pmax, in_tile_h, in_tile_w), dtype=X.dtype, buffer=nl.sbuf)
                        nisa.dma_copy(
                            dst=X_buf, 
                            src=X[
                                b, 
                                c_in_start:c_in_end,
                                out_i:out_i+in_tile_h, 
                                out_j:out_j+in_tile_w
                            ]
                        )
                        for i in nl.affine_range(filter_height):
                            for j in nl.affine_range(filter_width):
                                # W_tile = W_sbuf[:, c_in_start:c_in_end, i, j]
                                # W_tile = nisa.nc_transpose(W_tile)
                                # W_tile = nisa.tensor_copy(src=W_tile)
                                W_tile = W_sbuf[:, c_out_i, :, c_in_i, i, j]
                                X_tile_flat = nl.ndarray((c_in_pmax, out_tile_h*out_tile_w), dtype=X.dtype, buffer=nl.sbuf)
                                for dim1 in nl.affine_range(out_tile_h):
                                    start = dim1*out_tile_w
                                    end = (dim1+1)*out_tile_w
                                    X_tile_flat[:, start:end] = X_buf[
                                        :,
                                        i+dim1, 
                                        j:j+out_tile_w
                                    ]
                                # X_tile = X_buf[
                                #     :,
                                #     i:i+out_tile_h, 
                                #     j:j+out_tile_w
                                # ].reshape((c_in_pmax,-1)) # loosy line
                                res_psum += nisa.nc_matmul(X_tile_flat[...], W_tile[...])
                    # write back and clear res_psum
                    res_sbuf_flat = nl.copy(res_psum, dtype=X.dtype)
                    res_psum = nisa.memset(
                        shape=(out_tile_h*out_tile_w, c_out_pmax),
                        value=0.0,
                        dtype=KERNEL_DTYPE,
                    ) # TODO: can imporve
                    res_sbuf_flat = nisa.nc_transpose(res_sbuf_flat)
                    res_sbuf_flat = nisa.tensor_copy(src=res_sbuf_flat)
                    # Add bias
                    # bias_tile = bias_sbuf
                    
                    # bias_tile = nl.ndarray((c_out_pmax, 1), dtype=bias.dtype, buffer=nl.sbuf)
                    # bias_tile = nisa.tensor_copy(src=bias_sbuf[:, c_out_i, :])
                    # res_sbuf_flat = nisa.tensor_scalar(res_sbuf_flat, nl.add, bias_tile)
                    
                    for dim1 in nl.affine_range(out_tile_h):
                        start = dim1*out_tile_w
                        end = (dim1+1)*out_tile_w
                        res_sbuf[
                            :,
                            dim1, 
                            :
                        ] = res_sbuf_flat[:, start:end]
                    
                    # res_sbuf = res_sbuf.reshape((c_out_pmax, out_tile_h, out_tile_w))
                    nisa.dma_copy(
                        dst=X_out[
                            b, 
                            c_out_start:c_out_end, 
                            out_i:out_i+out_tile_h , 
                            out_j:out_j+out_tile_w
                        ],
                        src=res_sbuf
                    )
    return X_out

@nki.compiler.skip_middle_end_transformations
@nki.jit
def nki_matmul_tiled_(lhsT, rhs, result):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner"""

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Maximum free dimension of the stationary operand of general matrix multiplication on tensor engine
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128

  # Maximum partition dimension of a tile
  TILE_K = nl.tile_size.pmax  # 128

  # Maximum free dimension of the moving operand of general matrix multiplication on tensor engine
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # Allocate a tensor in PSUM
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        # Declare the tiles on SBUF
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        # Load tiles from lhsT and rhs
        nisa.dma_copy(dst=lhsT_tile, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rhs_tile, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])

        # Accumulate partial-sums into PSUM
        res_psum += nisa.nc_matmul(lhsT_tile[...], rhs_tile[...])

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)