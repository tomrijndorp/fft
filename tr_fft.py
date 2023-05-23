"""
Reference implementation for the FFT algorithm. The code structure is not very
Pythonic; rather, the code is written such that implementing the algorithm in
any language should be a fairly straightforward exercise.
"""

"""
MIT License

Copyright (c) 2023, Tom Rijndorp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import cmath

class FftInOut:
    def __init__(self, num_samples=None, sample_rate=None):
        """Pass in `num_samples` to initialize arrays (recommended).
        
        If you pass in a sample rate, upon calling `tr_fft`, the `frequencies`
        vector will be filled.
        """
        if num_samples is None:
            self.raw = None
            self.magnitude = None
            self.phase = None
            self.frequencies = None
            self.sample_rate = sample_rate
            print('__init__', self.sample_rate)
        else:
            assert(isinstance(num_samples, int))
            self.initialize(num_samples, sample_rate)

    def initialize(self, n, sample_rate):
        self.raw = [complex(0,0)] * n
        self.magnitude = [0.] * n
        self.phase = [0.] * n
        self.sample_rate = sample_rate
        if sample_rate is not None:
            self.frequencies = [0.] * n


def tr_fft(inout):
    """FFT implementation.
    
    Uses zero memory allocations and no recursion.
    
    Only uses the standard library.
    
    This implementation uses a standard decimation-in-time form of the FFT
    algorithm as invented by Cooley and Tukey.
    
    I tested this implementation for some amount and it seems to be doing the
    right thing.
    
    Example use:
    ```
    # Create the input / output data structure
    dft = DftInOut(8)  # create output for 8 samples
    # Populate the input (will be overwritten by the output)
    for i, sample in enumerate(dft.raw):
        dft.raw[i] = complex(1, 0)
    tr_fft(dft)
    print(dft.raw)
    ```
    """

    # Some helper functions below
    
    def is_power_of_two(n):
        """Pretty self descriptive."""
        if n % 2 != 0:
            return False
        while n >= 2:
            if n == 2:
                return True
            n >>= 1
        return False

    def bit_reverse(x, bit_len):
        """Bit reversal"""
        ret = 0;
        for idx in range(bit_len):
            bit_val = 1 if (x & (1 << idx) > 0) else 0
            rev_idx = (bit_len - idx - 1)
            ret |= bit_val << rev_idx
        return ret

    def butterfly(x0, x1, w0, w1, N):
        """FFT butterfly calculation."""
        W0 = cmath.exp(-1 * J * 2. * PI * w0 / N)
        W1 = cmath.exp(-1 * J * 2. * PI * w1 / N)
        X0 = x0 + W0 * x1
        X1 = x0 + W1 * x1
        return X0, X1
    
    def cplx_mod(z):
        """Complex modulus."""
        return math.sqrt(z.real ** 2 + z.imag ** 2)
    
    def cplx_arg(z):
        """Complex argument."""
        return math.atan2(z.imag, z.real)

    N = len(inout.raw)
    J = complex(0, 1)
    PI = math.pi
    
    assert is_power_of_two(N), "The number of FFT input samples needs to be a power of two."

    # Complex output vector
    X = inout.raw
    
    # Number of consecutive (log n) DFT stages 
    passes = math.floor(math.log(N, 2))

    # Because we need bit reversal for the weights, the easiest thing to do is to calculate
    # them ahead of time. We store them in the magnitude field so we don't have to allocate
    # extra memory.
    weights_mul_vec = inout.magnitude;
    for n in range(N):
        weights_mul_vec[n] = bit_reverse(n, passes)

    stride = N
    num_dfts_per_pass = N//2  # Same for every pass / stage
    for pass_idx in range(passes):
        weights_idx = 0
        stride //= 2
        idx0 = 0
        idx1 = 1
        for dft_i in range(num_dfts_per_pass):
            idx1 = idx0 + stride
            w0 = weights_mul_vec[weights_idx]
            w1 = weights_mul_vec[weights_idx+1]
            
            # Perform a 2-number DFT calculation
            R0, R1 = butterfly(X[idx0], X[idx1], w0, w1, N)
            X[idx0] = R0
            X[idx1] = R1

            # Increment indices
            idx0 += 1
            if (idx0 + stride) % (2 * stride) == 0:
                idx0 += stride

            # Increment weights
            if idx0 % stride == 0:
                weights_idx += 2

    # "Publish" the raw complex numbers. Note that we need to
    # perform a bit reversal for the indexing here. For that
    # reason, we temporarily store the complex output in the
    # magnitude and phase arrays.
    for n in range(N):
        inout.magnitude[n] = X[bit_reverse(n, passes)].real
        inout.phase[n] = X[bit_reverse(n, passes)].imag
    for n in range(N):
        X[n] = complex(inout.magnitude[n], inout.phase[n])

    # "Publish" magnitude and phase. Note: bit reversal.
    for k in range(N):
        inout.magnitude[k] = cplx_mod(inout.raw[k])
        inout.phase[k] = cplx_arg(inout.raw[k])
    
    # Optionally "publish" the frequency vector.
    if inout.frequencies is not None:
        T = N / inout.sample_rate
        df = 1. / T
        for k in range(N):
            inout.frequencies[k] = k * df;

    # Done.

        
def test_fft():
    """A basic parameterized unit test."""
    def make_input(freq, num_samples):
        ret = [0] * num_samples
        for i in range(num_samples):
            ret[i] = math.cos(2*math.pi*freq*i/num_samples)
        return ret
    
    def do_one_test(freq, num_samples):
        in_real = make_input(freq, num_samples)
        io = FftInOut(num_samples, sample_rate=1)
        for i in range(num_samples):
            io.raw[i] = complex(in_real[i], 0)
        tr_fft(io)
        assert io.magnitude[freq] > 1
        for i in range(num_samples // 2):
            if i == freq:
                assert io.magnitude[i] > 1
            else:
                assert io.magnitude[i] <0.01
    
    test_counter = 0
    num_samples = [2, 4, 8, 16, 32, 64]
    for ns in num_samples:
        freq = 0
        while True:
            do_one_test(freq, ns)
            freq = 1 if freq == 0 else freq << 1
            test_counter += 1
            if freq > ns // 2:
                break
    print(f"--- {test_counter} / {test_counter} tests passed ---")

                
if __name__ == "__main__":    
    test_fft()
