import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

PSUM_DTYPE = nl.float32

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
    c_out_pmax = TILE_M
    
    
    # Reshape W
    n_c_in_tiles = in_channels // c_in_pmax
    n_c_out_tiles = out_channels // c_out_pmax
    
    W_sbuf = nl.ndarray((c_in_pmax, n_c_in_tiles, c_out_pmax, n_c_out_tiles, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
    
    for c_out_i in nl.affine_range(n_c_out_tiles):
        W_sbuf_slice = nl.ndarray((c_out_pmax, in_channels, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=W_sbuf_slice, src=W[c_out_i*c_out_pmax:(c_out_i+1)*c_out_pmax, :, :, :])
        for c_in_i in nl.affine_range(n_c_in_tiles):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    W_tile = nisa.tensor_copy(
                        src = W_sbuf_slice[
                            :,
                            c_in_i*c_in_pmax:(c_in_i+1)*c_in_pmax,
                            i,
                            j
                        ]
                    )
                    W_tile = nisa.nc_transpose(W_tile)
                    W_sbuf[:, c_in_i, :, c_out_i, i, j] = W_tile
                    

    bias_sbuf = nl.ndarray((c_out_pmax, n_c_out_tiles, 1), dtype=bias.dtype, buffer=nl.sbuf)
    for c_out_i in nl.affine_range(n_c_out_tiles):
        nisa.dma_copy(dst=bias_sbuf[:, c_out_i, 0], src=bias[c_out_i*c_out_pmax:(c_out_i+1)*c_out_pmax])
    
    out_tile_h = 2
    out_tile_w = out_width
    in_tile_h = out_tile_h + filter_height - 1
    in_tile_w = out_tile_w + filter_width - 1
    for b in nl.affine_range(batch_size):
        for out_i in nl.sequential_range(0, out_height//out_tile_h):
            X_buf = nl.ndarray((c_in_pmax, n_c_in_tiles, in_tile_h, input_width), dtype=X.dtype, buffer=nl.sbuf)
            # Load X
            for c_in_i in nl.affine_range(n_c_in_tiles): # TODO: can improve the order of this loop to use fewer dma_copy
                X_tile_data = X[
                    b,
                    c_in_i*c_in_pmax:(c_in_i+1)*c_in_pmax,
                    out_i*out_tile_h: out_i*out_tile_h + in_tile_h,
                    :
                ]
                nisa.dma_copy(dst=X_buf[:,c_in_i,:,:], src=X_tile_data)
            
            for c_out_i in nl.affine_range(out_channels // c_out_pmax):
                    res_psum = nl.zeros(
                        shape=(c_out_pmax, out_tile_h*out_tile_w),
                        dtype=PSUM_DTYPE,
                        buffer=nl.psum
                    )
                    for c_in_i in nl.affine_range(in_channels // c_in_pmax):
                        for i in nl.affine_range(filter_height):
                            for j in nl.affine_range(filter_width):
                                W_tile = W_sbuf[:, c_in_i, :, c_out_i, i, j]  #(c_in_pmax, c_out_pmax)
                                X_tile = nisa.tensor_copy(X_buf[:, c_in_i, i : i + out_tile_h, j : j + out_tile_w])
                                X_tile = X_tile.reshape((c_in_pmax, out_tile_h*out_tile_w))
                                res_psum += nisa.nc_matmul(W_tile[...], X_tile[...]) # (c_out_pmax, out_tile_h*out_tile_w)
                                

                    res_sbuf = nl.copy(res_psum, dtype=X.dtype) # (c_out_pmax, out_tile_h*out_tile_w)
                    # Add bias
                    bias_tile = nisa.tensor_copy(src=bias_sbuf[:, c_out_i, :])
                    # res_sbuf = nisa.tensor_scalar(res_sbuf, nl.add, bias_tile)
                    bias_tile = nl.broadcast_to(bias_tile, shape=(c_out_pmax, out_tile_h*out_tile_w))
                    res_sbuf = nisa.tensor_tensor(res_sbuf, bias_tile, op=nl.add)
                    # Write back to hbm
                    res_sbuf = res_sbuf.reshape((c_in_pmax, out_tile_h, out_tile_w))
                    
                    if pool_size == 2:
                        # Maxpooling
                        res_sbuf = res_sbuf.reshape(
                            (c_out_pmax, out_tile_h, out_tile_w // 2, 2)
                        )
                        res_sbuf = nisa.tensor_reduce(
                            op=nl.max,
                            data=res_sbuf,
                            axis=[1,3],
                            keepdims=False,
                        )
                        res_sbuf = res_sbuf.reshape(
                            (c_out_pmax, out_tile_h//2, out_tile_w//2)
                        )
                        
                        nisa.dma_copy(
                            dst=X_out[
                                b, 
                                c_out_i*c_out_pmax:(c_out_i+1)*c_out_pmax,
                                out_i:(out_i+1),
                                :
                            ],
                            src=res_sbuf
                        )
                        
                    else :
                        nisa.dma_copy(
                            dst=X_out[
                                b, 
                                c_out_i*c_out_pmax:(c_out_i+1)*c_out_pmax,
                                out_i*out_tile_h:(out_i+1)*out_tile_h ,
                                :
                            ],
                            src=res_sbuf
                        )
    return X_out

