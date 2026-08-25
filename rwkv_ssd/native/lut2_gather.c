/**
 * Trinity LUT2: fused 2-bit unpack + 4-entry codebook gather.
 * Build with OpenMP for parallel multi-tensor layers.
 *
 * Not the same layout as GGML Q2_K / AQLM — per-tensor 4-float codebook + TR2 pack.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#define LUT2_HAS_AVX2 1
#else
#define LUT2_HAS_AVX2 0
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_MSC_VER)
#define LUT2_RESTRICT __restrict
#elif defined(__GNUC__)
#define LUT2_RESTRICT __restrict__
#else
#define LUT2_RESTRICT
#endif

static inline uint16_t lut2_float_to_bf16_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

static inline uint16_t lut2_read_u16(const uint8_t* p) {
    uint16_t value;
    memcpy(&value, p, sizeof(value));
    return value;
}

static inline uint32_t lut2_read_u32(const uint8_t* p) {
    uint32_t value;
    memcpy(&value, p, sizeof(value));
    return value;
}

/* Small IEEE-754 binary16 decoder.  Grouped LUT codebooks and residuals use
   fp16 records to keep their metadata overhead bounded. */
static inline float lut2_half_to_float(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t mantissa = (uint32_t)h & 0x03FFu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03FFu;
            bits = sign | ((uint32_t)(127 - 15 - shift) << 23)
                | (mantissa << 13);
        }
    } else if (exponent == 0x1Fu) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

void lut2_gather_packed(
    float* LUT2_RESTRICT out,
    const float* LUT2_RESTRICT codebook,
    const uint8_t* LUT2_RESTRICT packed,
    size_t n
) {
    size_t i = 0;
    size_t p = 0;
    const size_t n_full = n & ~(size_t)3;

    while (i < n_full) {
        uint8_t b = packed[p++];
        out[i++] = codebook[b & 3u];
        out[i++] = codebook[(b >> 2) & 3u];
        out[i++] = codebook[(b >> 4) & 3u];
        out[i++] = codebook[(b >> 6) & 3u];
    }
    if (i < n) {
        uint8_t b = packed[p];
        for (size_t shift = 0; i < n; ++i, shift += 2) {
            out[i] = codebook[(b >> shift) & 3u];
        }
    }
}

void lut2_gather_bf16_packed(
    uint16_t* LUT2_RESTRICT out,
    const float* LUT2_RESTRICT codebook,
    const uint8_t* LUT2_RESTRICT packed,
    size_t n
) {
    uint16_t cb_bf16[4];
    for (int k = 0; k < 4; ++k) {
        cb_bf16[k] = lut2_float_to_bf16_bits(codebook[k]);
    }
    size_t i = 0;
    size_t p = 0;
    const size_t n_full = n & ~(size_t)3;

    while (i < n_full) {
        uint8_t b = packed[p++];
        out[i++] = cb_bf16[b & 3u];
        out[i++] = cb_bf16[(b >> 2) & 3u];
        out[i++] = cb_bf16[(b >> 4) & 3u];
        out[i++] = cb_bf16[(b >> 6) & 3u];
    }
    if (i < n) {
        uint8_t b = packed[p];
        for (size_t shift = 0; i < n; ++i, shift += 2) {
            out[i] = cb_bf16[(b >> shift) & 3u];
        }
    }
}

void lut2_layer_gather(
    float* LUT2_RESTRICT flat_out,
    const float* LUT2_RESTRICT codebooks,
    const uint8_t* const* LUT2_RESTRICT packed_ptrs,
    const size_t* LUT2_RESTRICT offsets,
    const size_t* LUT2_RESTRICT numels,
    int n_tensors
) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n_tensors > 1)
#endif
    for (int t = 0; t < n_tensors; ++t) {
        const size_t off = offsets[t];
        const size_t n = numels[t];
        const float* cb = codebooks + (size_t)t * 4u;
        lut2_gather_packed(flat_out + off, cb, packed_ptrs[t], n);
    }
}

void lut2_layer_gather_bf16(
    uint16_t* LUT2_RESTRICT flat_out,
    const float* LUT2_RESTRICT codebooks,
    const uint8_t* const* LUT2_RESTRICT packed_ptrs,
    const size_t* LUT2_RESTRICT offsets,
    const size_t* LUT2_RESTRICT numels,
    int n_tensors
) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n_tensors > 1)
#endif
    for (int t = 0; t < n_tensors; ++t) {
        const size_t off = offsets[t];
        const size_t n = numels[t];
        const float* cb = codebooks + (size_t)t * 4u;
        lut2_gather_bf16_packed(flat_out + off, cb, packed_ptrs[t], n);
    }
}

/* y[row] = sum_c W[row,c] * x[c] with LUT2-packed W (no full W materialization).

   Hot loop walks packed bytes (4 weights each) instead of per-element
   bit-index math. That alone is ~4-8x on large GEMVs; OpenMP parallelizes
   across output rows when built with -fopenmp.
*/
void lut2_gemv_f32(
    float* LUT2_RESTRICT y,
    const float* LUT2_RESTRICT codebook,
    const uint8_t* LUT2_RESTRICT packed,
    const float* LUT2_RESTRICT x,
    int out_features,
    int in_features
) {
    const int cols_full = in_features & ~3;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out_features > 64)
#endif
    for (int row = 0; row < out_features; ++row) {
        float acc = 0.0f;
        const size_t row_base = (size_t)row * (size_t)in_features;
        const uint8_t* LUT2_RESTRICT prow = packed + (row_base >> 2);
        int col = 0;
        for (; col < cols_full; col += 4) {
            const uint8_t b = *prow++;
            acc += codebook[b & 3u] * x[col]
                + codebook[(b >> 2) & 3u] * x[col + 1]
                + codebook[(b >> 4) & 3u] * x[col + 2]
                + codebook[(b >> 6) & 3u] * x[col + 3];
        }
        if (col < in_features) {
            const uint8_t b = *prow;
            for (int shift = 0; col < in_features; ++col, shift += 2) {
                acc += codebook[(b >> shift) & 3u] * x[col];
            }
        }
        y[row] = acc;
    }
}

#ifdef _WIN32
#define LUT2_EXPORT __declspec(dllexport)
#else
#define LUT2_EXPORT __attribute__((visibility("default")))
#endif

LUT2_EXPORT void lut2_gather_packed_export(
    float* out, const float* codebook, const uint8_t* packed, size_t n
) {
    lut2_gather_packed(out, codebook, packed, n);
}

LUT2_EXPORT void lut2_layer_gather_export(
    float* flat_out,
    const float* codebooks,
    const uint8_t* const* packed_ptrs,
    const size_t* offsets,
    const size_t* numels,
    int n_tensors
) {
    lut2_layer_gather(flat_out, codebooks, packed_ptrs, offsets, numels, n_tensors);
}

LUT2_EXPORT void lut2_gather_bf16_packed_export(
    uint16_t* out, const float* codebook, const uint8_t* packed, size_t n
) {
    lut2_gather_bf16_packed(out, codebook, packed, n);
}

LUT2_EXPORT void lut2_layer_gather_bf16_export(
    uint16_t* flat_out,
    const float* codebooks,
    const uint8_t* const* packed_ptrs,
    const size_t* offsets,
    const size_t* numels,
    int n_tensors
) {
    lut2_layer_gather_bf16(flat_out, codebooks, packed_ptrs, offsets, numels, n_tensors);
}

LUT2_EXPORT void lut2_gemv_f32_export(
    float* y,
    const float* codebook,
    const uint8_t* packed,
    const float* x,
    int out_features,
    int in_features
) {
    lut2_gemv_f32(y, codebook, packed, x, out_features, in_features);
}

/* Direct GEMV for grouped Trinity LUT2 records.

   TR2\x03 stores fp32 codebooks, TR2\x04/TR2\x06 store fp16 codebooks, and
   TR2\x05/TR2\x07 add two fp16 residuals per group.  TR2\x06/TR2\x07 use
   the encoder's input-major (transposed) logical layout. */
void trinity_grouped_lut2_gemv_f32(
    float* LUT2_RESTRICT y,
    const uint8_t* LUT2_RESTRICT blob,
    const float* LUT2_RESTRICT x,
    int out_features,
    int in_features
) {
    const uint8_t magic0 = blob[0];
    const uint8_t magic1 = blob[1];
    const uint8_t magic2 = blob[2];
    const uint8_t magic3 = blob[3];
    const uint32_t group_size = lut2_read_u32(blob + 4);
    if (group_size == 0) {
        return;
    }
    const size_t n = (size_t)out_features * (size_t)in_features;
    const size_t groups = (n + group_size - 1u) / group_size;
    const int input_layout = (magic2 == '2' && (magic3 == 6 || magic3 == 7));
    const int residual = (magic3 == 5 || magic3 == 7);
    const int fp32_codebook = (magic3 == 3);
    const size_t record_bytes = residual ? 16u : (fp32_codebook ? 16u : 8u);
    const uint8_t* records = blob + 8u;
    const uint8_t* packed = records + groups * record_bytes;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out_features > 64)
#endif
    for (int row = 0; row < out_features; ++row) {
        float acc = 0.0f;
        for (int col = 0; col < in_features; ++col) {
            const size_t logical = input_layout
                ? (size_t)col * (size_t)out_features + (size_t)row
                : (size_t)row * (size_t)in_features + (size_t)col;
            const size_t group = logical / group_size;
            const size_t local = logical % group_size;
            const uint8_t b = packed[logical >> 2];
            const unsigned index = (unsigned)((b >> ((logical & 3u) * 2u)) & 3u);
            const uint8_t* record = records + group * record_bytes;
            float weight;
            if (fp32_codebook) {
                memcpy(&weight, record + index * 4u, sizeof(weight));
            } else {
                weight = lut2_half_to_float(lut2_read_u16(record + index * 2u));
            }
            if (residual) {
                const uint8_t p0 = record[8];
                const uint8_t p1 = record[9];
                if (local == p0) {
                    weight += lut2_half_to_float(lut2_read_u16(record + 10));
                } else if (local == p1) {
                    weight += lut2_half_to_float(lut2_read_u16(record + 12));
                }
            }
            acc += weight * x[col];
        }
        y[row] = acc;
    }
}

LUT2_EXPORT void trinity_grouped_lut2_gemv_f32_export(
    float* y, const uint8_t* blob, const float* x,
    int out_features, int in_features
) {
    trinity_grouped_lut2_gemv_f32(y, blob, x, out_features, in_features);
}

/* Direct GEMV for SG8\x01 grouped affine UINT8 records. */
static inline float scale_u8_grouped_row(
    const uint8_t* scales,
    const uint8_t* quant,
    const float* x,
    int in_features,
    uint32_t group_size,
    size_t row_base
) {
    /*
       SG8 stores an affine range per group.  The old hot loop reconstructed
       the weight for every element:

           (min + (max - min) * q / 255) * x

       That costs two floating-point multiplies and an add per element.  Sum
       the input and quantized-input products once per group instead:

           min * sum(x) + (max - min) / 255 * sum(q * x)

       The group size is normally 64 and the model dimensions are aligned to
       it, so this also gives the compiler a simple, vectorizable inner loop.
       Keep the generic boundary handling for small/synthetic tensors.
    */
    const float inv_255 = 1.0f / 255.0f;
    size_t col = 0;
    size_t logical = row_base;
    float acc = 0.0f;
    while (col < (size_t)in_features) {
        const size_t group = logical / (size_t)group_size;
        const size_t group_end = (group + 1u) * (size_t)group_size;
        size_t take = group_end - logical;
        if (take > (size_t)in_features - col) {
            take = (size_t)in_features - col;
        }
        float sum_x = 0.0f;
        float sum_qx = 0.0f;
        for (size_t j = 0; j < take; ++j) {
            const float xv = x[col + j];
            sum_x += xv;
            sum_qx += (float)quant[logical + j] * xv;
        }
        float mn;
        float mx;
        memcpy(&mn, scales + group * 8u, sizeof(mn));
        memcpy(&mx, scales + group * 8u + 4u, sizeof(mx));
        acc += mn * sum_x + (mx - mn) * inv_255 * sum_qx;
        col += take;
        logical += take;
    }
    return acc;
}

#if LUT2_HAS_AVX2
static inline float lut2_hsum256_ps(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sums = _mm_add_ps(low, high);
    sums = _mm_hadd_ps(sums, sums);
    sums = _mm_hadd_ps(sums, sums);
    return _mm_cvtss_f32(sums);
}

/* AVX2 version of the grouped-U8 algebra.  GCC does not consistently
   vectorize the scalar UINT8->float reduction under strict IEEE semantics;
   make the conversion and two reductions explicit. */
static inline float scale_u8_grouped_row_avx2(
    const uint8_t* scales,
    const uint8_t* quant,
    const float* x,
    int in_features,
    uint32_t group_size,
    size_t row_base
) {
    const float inv_255 = 1.0f / 255.0f;
    size_t col = 0;
    size_t logical = row_base;
    float acc = 0.0f;
    while (col < (size_t)in_features) {
        const size_t group = logical / (size_t)group_size;
        const size_t group_end = (group + 1u) * (size_t)group_size;
        size_t take = group_end - logical;
        if (take > (size_t)in_features - col) {
            take = (size_t)in_features - col;
        }
        float mn;
        float mx;
        memcpy(&mn, scales + group * 8u, sizeof(mn));
        memcpy(&mx, scales + group * 8u + 4u, sizeof(mx));
        const __m256 vzero = _mm256_setzero_ps();
        __m256 sum_x = vzero;
        __m256 sum_qx = vzero;
        size_t j = 0;
        for (; j + 8u <= take; j += 8u) {
            const __m256 xv = _mm256_loadu_ps(x + col + j);
            const __m128i q8 = _mm_loadl_epi64(
                (const __m128i*)(quant + logical + j)
            );
            const __m256i qi = _mm256_cvtepu8_epi32(q8);
            const __m256 qf = _mm256_cvtepi32_ps(qi);
            sum_x = _mm256_add_ps(sum_x, xv);
#if defined(__FMA__)
            sum_qx = _mm256_fmadd_ps(qf, xv, sum_qx);
#else
            sum_qx = _mm256_add_ps(sum_qx, _mm256_mul_ps(qf, xv));
#endif
        }
        float sum_x_scalar = lut2_hsum256_ps(sum_x);
        float sum_qx_scalar = lut2_hsum256_ps(sum_qx);
        for (; j < take; ++j) {
            const float xv = x[col + j];
            sum_x_scalar += xv;
            sum_qx_scalar += (float)quant[logical + j] * xv;
        }
        acc += mn * sum_x_scalar + (mx - mn) * inv_255 * sum_qx_scalar;
        col += take;
        logical += take;
    }
    return acc;
}
#endif

void scale_u8_grouped_gemv_f32(
    float* LUT2_RESTRICT y,
    const uint8_t* LUT2_RESTRICT blob,
    const float* LUT2_RESTRICT x,
    int out_features,
    int in_features
) {
    const uint32_t group_size = lut2_read_u32(blob + 4);
    if (group_size == 0) {
        return;
    }
    const size_t n = (size_t)out_features * (size_t)in_features;
    const size_t groups = (n + group_size - 1u) / group_size;
    const uint8_t* scales = blob + 8u;
    const uint8_t* quant = scales + groups * 8u;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out_features > 64)
#endif
    for (int row = 0; row < out_features; ++row) {
        const size_t row_base = (size_t)row * (size_t)in_features;
#if LUT2_HAS_AVX2
        if ((group_size & 7u) == 0u) {
            y[row] = scale_u8_grouped_row_avx2(
                scales, quant, x, in_features, group_size, row_base
            );
            continue;
        }
#endif
        y[row] = scale_u8_grouped_row(
            scales, quant, x, in_features, group_size, row_base
        );
    }
}

LUT2_EXPORT void scale_u8_grouped_gemv_f32_export(
    float* y, const uint8_t* blob, const float* x,
    int out_features, int in_features
) {
    scale_u8_grouped_gemv_f32(y, blob, x, out_features, in_features);
}

/* Fused CMix key -> ReLU^2 -> value GEMV.

   RWKV-7 CMix always feeds the value projection with the squared positive
   part of the key projection.  Keeping that intermediate in a native
   scratch buffer avoids a Python/NumPy round trip and a separate Torch
   ReLU/pow kernel.  The two projections have different shapes (normally
   10240x2560 followed by 2560x10240), so both dimensions are explicit. */
void scale_u8_grouped_cmix_gemv_f32(
    float* LUT2_RESTRICT y,
    const uint8_t* LUT2_RESTRICT key_blob,
    const uint8_t* LUT2_RESTRICT value_blob,
    const float* LUT2_RESTRICT x,
    int key_out_features,
    int key_in_features,
    int value_out_features,
    int value_in_features
) {
    if (key_blob == NULL || value_blob == NULL || x == NULL
        || y == NULL || key_out_features <= 0 || key_in_features <= 0
        || value_out_features <= 0 || value_in_features <= 0
        || value_in_features != key_out_features) {
        if (y != NULL && value_out_features > 0) {
            memset(y, 0, (size_t)value_out_features * sizeof(float));
        }
        return;
    }
    const uint32_t key_group_size = lut2_read_u32(key_blob + 4u);
    const uint32_t value_group_size = lut2_read_u32(value_blob + 4u);
    if (key_group_size == 0u || value_group_size == 0u) {
        memset(y, 0, (size_t)value_out_features * sizeof(float));
        return;
    }
    const size_t key_n = (size_t)key_out_features
        * (size_t)key_in_features;
    const size_t value_n = (size_t)value_out_features
        * (size_t)value_in_features;
    const size_t key_groups = (key_n + (size_t)key_group_size - 1u)
        / (size_t)key_group_size;
    const size_t value_groups = (value_n + (size_t)value_group_size - 1u)
        / (size_t)value_group_size;
    const uint8_t* key_scales = key_blob + 8u;
    const uint8_t* key_quant = key_scales + key_groups * 8u;
    const uint8_t* value_scales = value_blob + 8u;
    const uint8_t* value_quant = value_scales + value_groups * 8u;
    float* key_activation = (float*)malloc(
        (size_t)key_out_features * sizeof(float)
    );
    if (key_activation == NULL) {
        memset(y, 0, (size_t)value_out_features * sizeof(float));
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (key_out_features > 64)
#endif
    for (int row = 0; row < key_out_features; ++row) {
        const size_t row_base = (size_t)row * (size_t)key_in_features;
        float value;
#if LUT2_HAS_AVX2
        if ((key_group_size & 7u) == 0u) {
            value = scale_u8_grouped_row_avx2(
                key_scales, key_quant, x, key_in_features,
                key_group_size, row_base
            );
        } else
#endif
        {
            value = scale_u8_grouped_row(
                key_scales, key_quant, x, key_in_features,
                key_group_size, row_base
            );
        }
        key_activation[row] = value > 0.0f ? value * value : 0.0f;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (value_out_features > 64)
#endif
    for (int row = 0; row < value_out_features; ++row) {
        const size_t row_base = (size_t)row * (size_t)value_in_features;
#if LUT2_HAS_AVX2
        if ((value_group_size & 7u) == 0u) {
            y[row] = scale_u8_grouped_row_avx2(
                value_scales, value_quant, key_activation,
                value_in_features, value_group_size, row_base
            );
            continue;
        }
#endif
        y[row] = scale_u8_grouped_row(
            value_scales, value_quant, key_activation,
            value_in_features, value_group_size, row_base
        );
    }
    free(key_activation);
}

LUT2_EXPORT void scale_u8_grouped_cmix_gemv_f32_export(
    float* y,
    const uint8_t* key_blob,
    const uint8_t* value_blob,
    const float* x,
    int key_out_features,
    int key_in_features,
    int value_out_features,
    int value_in_features
) {
    scale_u8_grouped_cmix_gemv_f32(
        y, key_blob, value_blob, x,
        key_out_features, key_in_features,
        value_out_features, value_in_features
    );
}

/* Optional activation-quantized SG8 GEMV.

   The packed weights stay exactly the same.  For each input group, quantize
   the activation to signed INT8 with a per-group scale, retain the exact
   floating-point sum(x) for the affine minimum term, and evaluate the
   quantized q*x term with an AVX2 integer dot product:

       W*x ~= min*sum(x) + (max-min)/255 * act_scale * sum(q*x_i8)

   This is intentionally a separate ABI and is opt-in from Python because it
   changes activation rounding.  The normal float SG8 entry point remains the
   correctness/default path.  The fast path requires the common packed layout
   where each row starts on a group boundary; otherwise it falls back to the
   float implementation. */
typedef struct {
    int8_t* q;
    float* scale;
    float* sum_x;
    uint32_t group_size;
    int groups;
} lut2_i8_activation;

static int lut2_prepare_i8_activation(
    lut2_i8_activation* out,
    const float* x,
    int in_features,
    uint32_t group_size
) {
    if (out == NULL || x == NULL || in_features <= 0 || group_size == 0u
        || (in_features % (int)group_size) != 0) {
        return 0;
    }
    const int groups = (in_features + (int)group_size - 1)
        / (int)group_size;
    out->q = (int8_t*)malloc((size_t)in_features * sizeof(int8_t));
    out->scale = (float*)malloc((size_t)groups * sizeof(float));
    out->sum_x = (float*)malloc((size_t)groups * sizeof(float));
    out->group_size = group_size;
    out->groups = groups;
    if (out->q == NULL || out->scale == NULL || out->sum_x == NULL) {
        free(out->q);
        free(out->scale);
        free(out->sum_x);
        out->q = NULL;
        out->scale = NULL;
        out->sum_x = NULL;
        return 0;
    }
    for (int group = 0; group < groups; ++group) {
        const int begin = group * (int)group_size;
        const int end = begin + (int)group_size;
        float max_abs = 0.0f;
        float sum_x = 0.0f;
        for (int col = begin; col < end; ++col) {
            const float value = x[col];
            const float abs_value = value < 0.0f ? -value : value;
            if (abs_value > max_abs) {
                max_abs = abs_value;
            }
            sum_x += value;
        }
        const float scale = max_abs > 1.0e-12f ? max_abs / 127.0f : 1.0f;
        const float inv_scale = 1.0f / scale;
        out->scale[group] = scale;
        out->sum_x[group] = sum_x;
        for (int col = begin; col < end; ++col) {
            const float scaled = x[col] * inv_scale;
            int value = scaled >= 0.0f
                ? (int)(scaled + 0.5f)
                : (int)(scaled - 0.5f);
            if (value > 127) value = 127;
            if (value < -127) value = -127;
            out->q[col] = (int8_t)value;
        }
    }
    return 1;
}

static void lut2_release_i8_activation(lut2_i8_activation* value) {
    if (value == NULL) return;
    free(value->q);
    free(value->scale);
    free(value->sum_x);
    value->q = NULL;
    value->scale = NULL;
    value->sum_x = NULL;
}

#if LUT2_HAS_AVX2
static inline int32_t lut2_dot_u8_i8_avx2(
    const uint8_t* q,
    const int8_t* x,
    int count
) {
    __m256i acc = _mm256_setzero_si256();
    int col = 0;
    for (; col + 16 <= count; col += 16) {
        const __m128i q8 = _mm_loadu_si128((const __m128i*)(q + col));
        const __m128i x8 = _mm_loadu_si128((const __m128i*)(x + col));
        const __m256i q16 = _mm256_cvtepu8_epi16(q8);
        const __m256i x16 = _mm256_cvtepi8_epi16(x8);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(q16, x16));
    }
    const __m128i low = _mm256_castsi256_si128(acc);
    const __m128i high = _mm256_extracti128_si256(acc, 1);
    __m128i sum = _mm_add_epi32(low, high);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    int32_t result = _mm_cvtsi128_si32(sum);
    for (; col < count; ++col) {
        result += (int32_t)q[col] * (int32_t)x[col];
    }
    return result;
}
#endif

static inline float scale_u8_grouped_row_i8(
    const uint8_t* scales,
    const uint8_t* quant,
    const lut2_i8_activation* activation,
    int in_features,
    uint32_t group_size,
    size_t row_base
) {
    const float inv_255 = 1.0f / 255.0f;
    float acc = 0.0f;
    const size_t row_group = row_base / (size_t)group_size;
    const int groups = in_features / (int)group_size;
    for (int group = 0; group < groups; ++group) {
        const size_t weight_group = row_group + (size_t)group;
        const size_t weight_offset = weight_group * (size_t)group_size;
        int32_t dot = 0;
#if LUT2_HAS_AVX2
        dot = lut2_dot_u8_i8_avx2(
            quant + weight_offset,
            activation->q + (size_t)group * (size_t)group_size,
            (int)group_size
        );
#else
        for (uint32_t col = 0; col < group_size; ++col) {
            dot += (int32_t)quant[weight_offset + col]
                * (int32_t)activation->q[(size_t)group * group_size + col];
        }
#endif
        float mn;
        float mx;
        memcpy(&mn, scales + weight_group * 8u, sizeof(mn));
        memcpy(&mx, scales + weight_group * 8u + 4u, sizeof(mx));
        acc += mn * activation->sum_x[group]
            + (mx - mn) * inv_255 * activation->scale[group] * (float)dot;
    }
    return acc;
}

LUT2_EXPORT void scale_u8_grouped_i8_gemv_f32_export(
    float* y, const uint8_t* blob, const float* x,
    int out_features, int in_features
) {
    const uint32_t group_size = lut2_read_u32(blob + 4u);
    if (group_size == 0u || in_features <= 0
        || (in_features % (int)group_size) != 0) {
        scale_u8_grouped_gemv_f32(y, blob, x, out_features, in_features);
        return;
    }
    lut2_i8_activation activation = {0};
    if (!lut2_prepare_i8_activation(
            &activation, x, in_features, group_size)) {
        scale_u8_grouped_gemv_f32(y, blob, x, out_features, in_features);
        return;
    }
    const size_t n = (size_t)out_features * (size_t)in_features;
    const size_t groups = (n + group_size - 1u) / group_size;
    const uint8_t* scales = blob + 8u;
    const uint8_t* quant = scales + groups * 8u;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out_features > 16)
#endif
    for (int row = 0; row < out_features; ++row) {
        y[row] = scale_u8_grouped_row_i8(
            scales, quant, &activation, in_features, group_size,
            (size_t)row * (size_t)in_features
        );
    }
    lut2_release_i8_activation(&activation);
}

/* SG8 row-major transpose GEMV for the small RWKV-7 TMix projections.

   ChatRWKV evaluates these matrices as ``x @ W``.  The pack stores W in its
   natural [input, output] row-major shape, while the large attention/CMix
   maps use the opposite W @ x convention after the RWKV z-layout transpose.
   Keep the packed bytes unchanged and walk the row-major payload by output
   column here.  These projections are at most 2560x320, so the scalar
   fallback avoids an expensive transpose/materialization without adding a
   second pack format.  The optional AVX2 block path computes eight adjacent
   output columns together; this makes the normal g64 / 320-column layout a
   contiguous quantized read instead of one scalar strided read at a time. */
static inline void scale_u8_grouped_transposed_block(
    float* LUT2_RESTRICT y,
    const uint8_t* LUT2_RESTRICT scales,
    const uint8_t* LUT2_RESTRICT quant,
    const float* LUT2_RESTRICT x,
    int out_features,
    int in_features,
    uint32_t group_size,
    int col
) {
#if LUT2_HAS_AVX2
    if (group_size != 0u && col + 8 <= out_features) {
        __m256 acc = _mm256_setzero_ps();
        const __m256 inv_255 = _mm256_set1_ps(1.0f / 255.0f);
        int vector_ok = 1;
        for (int row = 0; row < in_features; ++row) {
            const size_t logical =
                (size_t)row * (size_t)out_features + (size_t)col;
            const size_t group = logical / (size_t)group_size;
            const size_t group_end = (group + 1u) * (size_t)group_size;
            if (logical + 7u >= group_end) {
                vector_ok = 0;
                break;
            }
            float mn;
            float mx;
            memcpy(&mn, scales + group * 8u, sizeof(mn));
            memcpy(&mx, scales + group * 8u + 4u, sizeof(mx));
            const __m128i q8 = _mm_loadl_epi64(
                (const __m128i*)(quant + logical)
            );
            const __m256 qf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q8));
            const __m256 delta = _mm256_set1_ps(mx - mn);
            const __m256 weights = _mm256_add_ps(
                _mm256_mul_ps(qf, _mm256_mul_ps(delta, inv_255)),
                _mm256_set1_ps(mn)
            );
            const __m256 xv = _mm256_set1_ps(x[row]);
#if defined(__FMA__)
            acc = _mm256_fmadd_ps(weights, xv, acc);
#else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(weights, xv));
#endif
        }
        if (vector_ok) {
            _mm256_storeu_ps(y + col, acc);
            return;
        }
    }
#endif
    const int end = col + 8 < out_features ? col + 8 : out_features;
    for (int out = col; out < end; ++out) {
        float acc = 0.0f;
        for (int row = 0; row < in_features; ++row) {
            const size_t logical =
                (size_t)row * (size_t)out_features + (size_t)out;
            const size_t group = logical / (size_t)group_size;
            float mn;
            float mx;
            memcpy(&mn, scales + group * 8u, sizeof(mn));
            memcpy(&mx, scales + group * 8u + 4u, sizeof(mx));
            const float weight = mn + (mx - mn)
                * ((float)quant[logical] * (1.0f / 255.0f));
            acc += weight * x[row];
        }
        y[out] = acc;
    }
}

void scale_u8_grouped_transposed_gemv_f32(
    float* LUT2_RESTRICT y,
    const uint8_t* LUT2_RESTRICT blob,
    const float* LUT2_RESTRICT x,
    int out_features,
    int in_features
) {
    const uint32_t group_size = lut2_read_u32(blob + 4u);
    if (group_size == 0u) {
        memset(y, 0, (size_t)out_features * sizeof(float));
        return;
    }
    const size_t n = (size_t)out_features * (size_t)in_features;
    const size_t groups = (n + (size_t)group_size - 1u) / (size_t)group_size;
    const uint8_t* scales = blob + 8u;
    const uint8_t* quant = scales + groups * 8u;
#if LUT2_HAS_AVX2
    if ((group_size & 7u) == 0u && (size_t)out_features % group_size == 0u) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out_features > 64)
#endif
        for (int col = 0; col < out_features; col += 8) {
            scale_u8_grouped_transposed_block(
                y, scales, quant, x, out_features, in_features,
                group_size, col
            );
        }
        return;
    }
#endif
    const float inv_255 = 1.0f / 255.0f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out_features > 64)
#endif
    for (int col = 0; col < out_features; ++col) {
        float acc = 0.0f;
        for (int row = 0; row < in_features; ++row) {
            const size_t logical =
                (size_t)row * (size_t)out_features + (size_t)col;
            const size_t group = logical / (size_t)group_size;
            float mn;
            float mx;
            memcpy(&mn, scales + group * 8u, sizeof(mn));
            memcpy(&mx, scales + group * 8u + 4u, sizeof(mx));
            const float weight = mn + (mx - mn)
                * ((float)quant[logical] * inv_255);
            acc += weight * x[row];
        }
        y[col] = acc;
    }
}

LUT2_EXPORT void scale_u8_grouped_transposed_gemv_f32_export(
    float* y, const uint8_t* blob, const float* x,
    int out_features, int in_features
) {
    scale_u8_grouped_transposed_gemv_f32(
        y, blob, x, out_features, in_features
    );
}

/* Batched form for the independent first/second TMix adapter projections.
   One OpenMP region amortizes thread-pool wakeup across w/a/v/g maps. */
static void scale_u8_grouped_transposed_tmix_gemv_f32_n(
    float* const* LUT2_RESTRICT y_ptrs,
    const uint8_t* const* LUT2_RESTRICT blob_ptrs,
    const float* const* LUT2_RESTRICT x_ptrs,
    const int* LUT2_RESTRICT out_features,
    const int* LUT2_RESTRICT in_features,
    int n_mats
) {
    if (n_mats <= 0 || n_mats > 4) {
        return;
    }
    const uint8_t* scales[4];
    const uint8_t* quant[4];
    uint32_t group_sizes[4];
    int block_prefix[5] = {0, 0, 0, 0, 0};
    for (int mat = 0; mat < n_mats; ++mat) {
        const size_t n = (size_t)out_features[mat]
            * (size_t)in_features[mat];
        const uint32_t group_size = lut2_read_u32(blob_ptrs[mat] + 4u);
        group_sizes[mat] = group_size;
        const size_t groups = group_size
            ? (n + (size_t)group_size - 1u) / (size_t)group_size
            : 0u;
        scales[mat] = blob_ptrs[mat] + 8u;
        quant[mat] = scales[mat] + groups * 8u;
        block_prefix[mat + 1] = block_prefix[mat]
            + (out_features[mat] + 7) / 8;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (block_prefix[n_mats] > 8)
#endif
    for (int work = 0; work < block_prefix[n_mats]; ++work) {
        int mat = 0;
        while (mat + 1 < n_mats && work >= block_prefix[mat + 1]) {
            ++mat;
        }
        const int col = (work - block_prefix[mat]) * 8;
        const int out_f = out_features[mat];
        const int in_f = in_features[mat];
        const uint32_t group_size = group_sizes[mat];
        if (group_size == 0u) {
            const int end = col + 8 < out_f ? col + 8 : out_f;
            memset(y_ptrs[mat] + col, 0, (size_t)(end - col) * sizeof(float));
            continue;
        }
        scale_u8_grouped_transposed_block(
            y_ptrs[mat], scales[mat], quant[mat], x_ptrs[mat],
            out_f, in_f, group_size, col
        );
    }
}

LUT2_EXPORT void scale_u8_grouped_transposed_tmix_gemv_f32_export(
    float* const* y_ptrs,
    const uint8_t* const* blob_ptrs,
    const float* const* x_ptrs,
    const int* out_features,
    const int* in_features,
    int n_mats
) {
    scale_u8_grouped_transposed_tmix_gemv_f32_n(
        y_ptrs, blob_ptrs, x_ptrs, out_features, in_features, n_mats
    );
}

/* Fused RWKV-7 TMix adapter pipeline.

   The compact w/a/g/v adapters are evaluated as two small transposed GEMV
   sweeps with an elementwise operation between them:

       w = W2 * tanh(W1 * xw)
       a = sigmoid(a0 + A2 * (A1 * xa))
       g = G2 * sigmoid(G1 * xg)
       v = V2 * (V1 * xv)

   Python used to launch two native batch calls and then several Torch
   elementwise kernels for every streamed layer/token.  Keep the packed SG8
   records and the reduction order in the existing helpers, but keep the
   intermediate rank-sized buffers and activations on the native side.  The
   v path is optional for layer 0, which has no v-gate in the RWKV-7 formula.

   blob_ptrs order: w1, w2, a1, a2, g1, g2, v1, v2
   x_ptrs order:    xw, xa, xg, xv
   rank_dims order: w_rank, a_rank, g_rank, v_rank
*/
LUT2_EXPORT void scale_u8_grouped_transposed_tmix_fused_f32_export(
    float* w_out,
    float* a_out,
    float* g_out,
    float* v_out,
    const uint8_t* const* blob_ptrs,
    const float* const* x_ptrs,
    const float* a0,
    int n_embd,
    const int* rank_dims,
    int has_v
) {
    if (w_out == NULL || a_out == NULL || g_out == NULL
        || blob_ptrs == NULL || x_ptrs == NULL || rank_dims == NULL
        || n_embd <= 0 || rank_dims[0] <= 0 || rank_dims[1] <= 0
        || rank_dims[2] <= 0 || (has_v && (v_out == NULL || rank_dims[3] <= 0))) {
        return;
    }

    const int w_rank = rank_dims[0];
    const int a_rank = rank_dims[1];
    const int g_rank = rank_dims[2];
    const int v_rank = has_v ? rank_dims[3] : 0;
    float* scratch = (float*)malloc(
        (size_t)(w_rank + a_rank + g_rank + v_rank) * sizeof(float)
    );
    if (scratch == NULL) {
        return;
    }
    float* w_mid = scratch;
    float* a_mid = w_mid + w_rank;
    float* g_mid = a_mid + a_rank;
    float* v_mid = g_mid + g_rank;

    const uint8_t* first_blobs[4] = {
        blob_ptrs[0], blob_ptrs[2], blob_ptrs[4], has_v ? blob_ptrs[6] : NULL
    };
    const float* first_x[4] = {
        x_ptrs[0], x_ptrs[1], x_ptrs[2], has_v ? x_ptrs[3] : NULL
    };
    float* first_y[4] = {
        w_mid, a_mid, g_mid, has_v ? v_mid : NULL
    };
    const int first_out[4] = {w_rank, a_rank, g_rank, v_rank};
    const int first_in[4] = {n_embd, n_embd, n_embd, n_embd};
    scale_u8_grouped_transposed_tmix_gemv_f32_n(
        first_y, first_blobs, first_x, first_out, first_in, has_v ? 4 : 3
    );

    for (int i = 0; i < w_rank; ++i) {
        w_mid[i] = tanhf(w_mid[i]);
    }
    for (int i = 0; i < g_rank; ++i) {
        g_mid[i] = 1.0f / (1.0f + expf(-g_mid[i]));
    }

    const uint8_t* second_blobs[4] = {
        blob_ptrs[1], blob_ptrs[3], blob_ptrs[5], has_v ? blob_ptrs[7] : NULL
    };
    const float* second_x[4] = {
        w_mid, a_mid, g_mid, has_v ? v_mid : NULL
    };
    float* second_y[4] = {
        w_out, a_out, g_out, has_v ? v_out : NULL
    };
    const int second_out[4] = {n_embd, n_embd, n_embd, n_embd};
    const int second_in[4] = {w_rank, a_rank, g_rank, v_rank};
    scale_u8_grouped_transposed_tmix_gemv_f32_n(
        second_y, second_blobs, second_x,
        second_out, second_in, has_v ? 4 : 3
    );

    for (int i = 0; i < n_embd; ++i) {
        const float bias = a0 != NULL ? a0[i] : 0.0f;
        a_out[i] = 1.0f / (1.0f + expf(-(bias + a_out[i])));
    }
    free(scratch);
}

/* Exact sparse grouped-U8 GEMV for post-ReLU CMix activations.

   The active column list is sorted and contains only nonzero input entries.
   Accumulate the same sum(x) and sum(q*x) terms as the dense kernel, but do
   not visit zero activation columns.  This is intentionally a separate ABI:
   arbitrary sparse inputs should use the dense AVX2 kernel when their active
   fraction is too high. */
static inline float scale_u8_grouped_row_sparse(
    const uint8_t* scales,
    const uint8_t* quant,
    const float* x,
    const int32_t* active_indices,
    int active_count,
    int in_features,
    uint32_t group_size,
    size_t row_base
) {
    const float inv_255 = 1.0f / 255.0f;
    float acc = 0.0f;
    int active = 0;
    while (active < active_count) {
        const int first_col = active_indices[active];
        if (first_col < 0 || first_col >= in_features) {
            ++active;
            continue;
        }
        const size_t first_logical = row_base + (size_t)first_col;
        const size_t group = first_logical / (size_t)group_size;
        const size_t group_end = (group + 1u) * (size_t)group_size;
        float sum_x = 0.0f;
        float sum_qx = 0.0f;
        while (active < active_count) {
            const int col = active_indices[active];
            if (col < 0 || col >= in_features) {
                ++active;
                continue;
            }
            const size_t logical = row_base + (size_t)col;
            if (logical >= group_end) {
                break;
            }
            const float xv = x[col];
            sum_x += xv;
            sum_qx += (float)quant[logical] * xv;
            ++active;
        }
        float mn;
        float mx;
        memcpy(&mn, scales + group * 8u, sizeof(mn));
        memcpy(&mx, scales + group * 8u + 4u, sizeof(mx));
        acc += mn * sum_x + (mx - mn) * inv_255 * sum_qx;
    }
    return acc;
}

void scale_u8_grouped_sparse_gemv_f32(
    float* LUT2_RESTRICT y,
    const uint8_t* LUT2_RESTRICT blob,
    const float* LUT2_RESTRICT x,
    const int32_t* LUT2_RESTRICT active_indices,
    int active_count,
    int out_features,
    int in_features
) {
    const uint32_t group_size = lut2_read_u32(blob + 4u);
    if (group_size == 0u || active_count <= 0) {
        memset(y, 0, (size_t)out_features * sizeof(float));
        return;
    }
    const size_t n = (size_t)out_features * (size_t)in_features;
    const size_t groups = (n + (size_t)group_size - 1u) / (size_t)group_size;
    const uint8_t* scales = blob + 8u;
    const uint8_t* quant = scales + groups * 8u;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out_features > 16)
#endif
    for (int row = 0; row < out_features; ++row) {
        y[row] = scale_u8_grouped_row_sparse(
            scales, quant, x, active_indices, active_count, in_features,
            group_size, (size_t)row * (size_t)in_features
        );
    }
}

LUT2_EXPORT void scale_u8_grouped_sparse_gemv_f32_export(
    float* y, const uint8_t* blob, const float* x,
    const int32_t* active_indices, int active_count,
    int out_features, int in_features
) {
    scale_u8_grouped_sparse_gemv_f32(
        y, blob, x, active_indices, active_count, out_features, in_features
    );
}

/* Batched SG8 TMix: attention maps with distinct input vectors.

   The ordinary grouped-U8 entry point launches one OpenMP region per matrix.
   TMix consumes three maps before the recurrent state update and the output
   map afterwards.  The ``n_mats`` helper supports both the public four-map
   compatibility ABI and the three-map QKV fast path, avoiding a throwaway
   output GEMV on every token.  The blobs may have different group sizes;
   each one is parsed once before entering the hot loop. */
static void scale_u8_grouped_tmix_gemv_f32_n(
    float* LUT2_RESTRICT y_out,
    const uint8_t* const* LUT2_RESTRICT blob_ptrs,
    const float* const* LUT2_RESTRICT x_ptrs,
    int out_features,
    int in_features,
    int n_mats
) {
    const size_t n = (size_t)out_features * (size_t)in_features;
    const uint8_t* scales[4];
    const uint8_t* quant[4];
    uint32_t group_sizes[4];
    for (int mat = 0; mat < n_mats; ++mat) {
        const uint8_t* blob = blob_ptrs[mat];
        const uint32_t group_size = lut2_read_u32(blob + 4u);
        group_sizes[mat] = group_size;
        const size_t groups = group_size
            ? (n + (size_t)group_size - 1u) / (size_t)group_size
            : 0u;
        scales[mat] = blob + 8u;
        quant[mat] = scales[mat] + groups * 8u;
    }
    for (int mat = 0; mat < n_mats; ++mat) {
        if (group_sizes[mat] == 0) {
            return;
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out_features > 16)
#endif
    for (int row = 0; row < out_features; ++row) {
        const size_t row_base = (size_t)row * (size_t)in_features;
        for (int mat = 0; mat < n_mats; ++mat) {
#if LUT2_HAS_AVX2
            if ((group_sizes[mat] & 7u) == 0u) {
                y_out[(size_t)mat * (size_t)out_features + (size_t)row] =
                    scale_u8_grouped_row_avx2(
                        scales[mat], quant[mat], x_ptrs[mat], in_features,
                        group_sizes[mat], row_base
                    );
                continue;
            }
#endif
            y_out[(size_t)mat * (size_t)out_features + (size_t)row] =
                scale_u8_grouped_row(
                    scales[mat], quant[mat], x_ptrs[mat], in_features,
                    group_sizes[mat], row_base
                );
        }
    }
}

void scale_u8_grouped_tmix_gemv_f32(
    float* LUT2_RESTRICT y_out,
    const uint8_t* const* LUT2_RESTRICT blob_ptrs,
    const float* const* LUT2_RESTRICT x_ptrs,
    int out_features,
    int in_features
) {
    scale_u8_grouped_tmix_gemv_f32_n(
        y_out, blob_ptrs, x_ptrs, out_features, in_features, 4
    );
}

void scale_u8_grouped_tmix_qkv_gemv_f32(
    float* LUT2_RESTRICT y_out,
    const uint8_t* const* LUT2_RESTRICT blob_ptrs,
    const float* const* LUT2_RESTRICT x_ptrs,
    int out_features,
    int in_features
) {
    scale_u8_grouped_tmix_gemv_f32_n(
        y_out, blob_ptrs, x_ptrs, out_features, in_features, 3
    );
}

LUT2_EXPORT void scale_u8_grouped_tmix_gemv_f32_export(
    float* y_out,
    const uint8_t* const* blob_ptrs,
    const float* const* x_ptrs,
    int out_features,
    int in_features
) {
    scale_u8_grouped_tmix_gemv_f32(
        y_out, blob_ptrs, x_ptrs, out_features, in_features
    );
}

LUT2_EXPORT void scale_u8_grouped_tmix_qkv_gemv_f32_export(
    float* y_out,
    const uint8_t* const* blob_ptrs,
    const float* const* x_ptrs,
    int out_features,
    int in_features
) {
    scale_u8_grouped_tmix_qkv_gemv_f32(
        y_out, blob_ptrs, x_ptrs, out_features, in_features
    );
}

/* QKV activation-quantized variant.  Each input vector gets its own
   per-group activation scale, while one OpenMP region still covers all three
   output maps.  The ABI mirrors the float QKV helper so Python can switch it
   without changing the layer representation. */
LUT2_EXPORT void scale_u8_grouped_tmix_qkv_i8_gemv_f32_export(
    float* y_out,
    const uint8_t* const* blob_ptrs,
    const float* const* x_ptrs,
    int out_features,
    int in_features
) {
    const uint32_t group_size0 = lut2_read_u32(blob_ptrs[0] + 4u);
    const uint32_t group_size1 = lut2_read_u32(blob_ptrs[1] + 4u);
    const uint32_t group_size2 = lut2_read_u32(blob_ptrs[2] + 4u);
    if (group_size0 == 0u || group_size0 != group_size1
        || group_size0 != group_size2 || in_features <= 0
        || (in_features % (int)group_size0) != 0) {
        scale_u8_grouped_tmix_qkv_gemv_f32(
            y_out, blob_ptrs, x_ptrs, out_features, in_features
        );
        return;
    }
    lut2_i8_activation activations[3] = {{0}, {0}, {0}};
    int prepared = 1;
    for (int mat = 0; mat < 3; ++mat) {
        if (!lut2_prepare_i8_activation(
                &activations[mat], x_ptrs[mat], in_features, group_size0)) {
            prepared = 0;
            break;
        }
    }
    if (!prepared) {
        for (int mat = 0; mat < 3; ++mat) {
            lut2_release_i8_activation(&activations[mat]);
        }
        scale_u8_grouped_tmix_qkv_gemv_f32(
            y_out, blob_ptrs, x_ptrs, out_features, in_features
        );
        return;
    }
    const size_t n = (size_t)out_features * (size_t)in_features;
    const size_t groups = (n + group_size0 - 1u) / group_size0;
    const uint8_t* scales[3];
    const uint8_t* quant[3];
    for (int mat = 0; mat < 3; ++mat) {
        scales[mat] = blob_ptrs[mat] + 8u;
        quant[mat] = scales[mat] + groups * 8u;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out_features > 16)
#endif
    for (int row = 0; row < out_features; ++row) {
        const size_t row_base = (size_t)row * (size_t)in_features;
        for (int mat = 0; mat < 3; ++mat) {
            y_out[(size_t)mat * (size_t)out_features + (size_t)row] =
                scale_u8_grouped_row_i8(
                    scales[mat], quant[mat], &activations[mat],
                    in_features, group_size0, row_base
                );
        }
    }
    for (int mat = 0; mat < 3; ++mat) {
        lut2_release_i8_activation(&activations[mat]);
    }
}

/* Batched TMix: 4 att linear maps with distinct input vectors in one parallel sweep. */
static void lut2_tmix_gemv_f32_n(
    float* LUT2_RESTRICT y_out,
    const float* LUT2_RESTRICT codebooks,
    const uint8_t* const* LUT2_RESTRICT packed_ptrs,
    const float* const* LUT2_RESTRICT x_ptrs,
    int out_features,
    int in_features,
    int n_mats
) {
    const int total_rows = n_mats * out_features;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (total_rows > 64)
#endif
    for (int tr = 0; tr < total_rows; ++tr) {
        const int mat = tr / out_features;
        const int row = tr % out_features;
        const float* cb = codebooks + (size_t)mat * 4u;
        const uint8_t* packed = packed_ptrs[mat];
        const float* x = x_ptrs[mat];
        float acc = 0.0f;
        const int row_base = row * in_features;
        for (int col = 0; col < in_features; ++col) {
            const size_t i = (size_t)row_base + (size_t)col;
            const uint8_t b = packed[i / 4];
            const int shift = (int)((i % 4) * 2);
            acc += cb[(b >> shift) & 3u] * x[col];
        }
        y_out[(size_t)mat * (size_t)out_features + (size_t)row] = acc;
    }
}

void lut2_tmix_gemv_f32(
    float* LUT2_RESTRICT y_out,
    const float* LUT2_RESTRICT codebooks,
    const uint8_t* const* LUT2_RESTRICT packed_ptrs,
    const float* const* LUT2_RESTRICT x_ptrs,
    int out_features,
    int in_features
) {
    lut2_tmix_gemv_f32_n(
        y_out, codebooks, packed_ptrs, x_ptrs, out_features, in_features, 4
    );
}

void lut2_tmix_qkv_gemv_f32(
    float* LUT2_RESTRICT y_out,
    const float* LUT2_RESTRICT codebooks,
    const uint8_t* const* LUT2_RESTRICT packed_ptrs,
    const float* const* LUT2_RESTRICT x_ptrs,
    int out_features,
    int in_features
) {
    lut2_tmix_gemv_f32_n(
        y_out, codebooks, packed_ptrs, x_ptrs, out_features, in_features, 3
    );
}

LUT2_EXPORT void lut2_tmix_gemv_f32_export(
    float* y_out,
    const float* codebooks,
    const uint8_t* const* packed_ptrs,
    const float* const* x_ptrs,
    int out_features,
    int in_features
) {
    lut2_tmix_gemv_f32(y_out, codebooks, packed_ptrs, x_ptrs, out_features, in_features);
}

LUT2_EXPORT void lut2_tmix_qkv_gemv_f32_export(
    float* y_out,
    const float* codebooks,
    const uint8_t* const* packed_ptrs,
    const float* const* x_ptrs,
    int out_features,
    int in_features
) {
    lut2_tmix_qkv_gemv_f32(y_out, codebooks, packed_ptrs, x_ptrs, out_features, in_features);
}
