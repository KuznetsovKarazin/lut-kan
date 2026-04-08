#include <Arduino.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "case_layer.h"

// STM32 Blue Pill: Serial = USB CDC (not available in Wokwi).
// Redirect to Serial1 = USART1 on PA9/PA10 (wired to $serialMonitor).
#if defined(ARDUINO_BLUEPILL_F103C8) || defined(STM32F1xx)
  #define Serial Serial1
#endif

// =============================================================================
// Pre-processing helpers
// =============================================================================

static inline float fast_tanh(float x) {
#if CASE_USE_TANH
  return tanhf(x);
#else
  return x;
#endif
}

static inline float preprocess_x(float x) {
  x = fast_tanh(x);
#if CASE_CLIP_X
  if (x < (float)CASE_X_MIN) x = (float)CASE_X_MIN;
  if (x > (float)CASE_X_MAX) x = (float)CASE_X_MAX;
#endif
  return x;
}

// =============================================================================
// Jacobi float baseline (per-edge)
// =============================================================================

#if CASE_BASIS_TYPE == 0

static inline void jacobi_basis(float x, float* P) {
  const float a = (float)CASE_ALPHA;
  const float b = (float)CASE_BETA;
  const int deg = (int)CASE_DEGREE;

  P[0] = 1.0f;
  if (deg == 0) return;

  P[1] = 0.5f * ((a - b) + (a + b + 2.0f) * x);

  for (int n = 1; n < deg; ++n) {
    const float nn = (float)n;
    const float two_n_ab = 2.0f * nn + a + b;
    const float A = 2.0f * (nn + 1.0f) * (nn + a + b + 1.0f) * two_n_ab;
    const float B = (two_n_ab + 1.0f) * ((two_n_ab + 2.0f) * two_n_ab * x + (a*a - b*b));
    const float C = 2.0f * (nn + a) * (nn + b) * (two_n_ab + 2.0f);
    P[n + 1] = (B * P[n] - C * P[n - 1]) / A;
  }
}

static void float_baseline_forward(const float* x_in, float* y_out) {
  const int in_dim = (int)CASE_IN_DIM;
  const int out_dim = (int)CASE_OUT_DIM;
  const int deg = (int)CASE_DEGREE;

  for (int j = 0; j < out_dim; ++j) y_out[j] = 0.0f;

  float P[CASE_DEGREE + 1];

  for (int i = 0; i < in_dim; ++i) {
    float x = preprocess_x(x_in[i]);
    jacobi_basis(x, P);

    for (int j = 0; j < out_dim; ++j) {
      const int edge = i * out_dim + j;
      const float* c = &CASE_FLOAT_COEFFS[edge * (deg + 1)];
      float acc = 0.0f;
      for (int k = 0; k <= deg; ++k) acc += c[k] * P[k];
      y_out[j] += acc;
    }
  }
}

#endif // CASE_BASIS_TYPE == 0

// =============================================================================
// B-spline float baseline (per-edge, Cox-de Boor recursion)
// =============================================================================

#if CASE_BASIS_TYPE == 1

// SiLU base function: x / (1 + exp(-x))
static inline float silu(float x) {
  return x / (1.0f + expf(-x));
}

// Evaluate all B-spline basis functions N_{i,k}(x) using Cox-de Boor.
// Returns the spline value = sum_i coef[i] * N_{i,k}(x).
static inline float bspline_eval_edge(float x, const float* coef, int num_coef) {
  const int k = (int)CASE_BSPLINE_DEGREE;
  const int M = (int)CASE_NUM_KNOTS_AUG;

  float B[CASE_NUM_KNOTS_AUG];

  // Degree 0 basis: indicator functions
  for (int i = 0; i < M - 1; ++i) {
    float left = CASE_KNOTS_AUG[i];
    float right = CASE_KNOTS_AUG[i + 1];
    if (i == M - 2) {
      B[i] = (x >= left && x <= right) ? 1.0f : 0.0f;
    } else {
      B[i] = (x >= left && x < right) ? 1.0f : 0.0f;
    }
  }

  // Elevate degree using recurrence (in-place)
  for (int d = 1; d <= k; ++d) {
    int n_basis = M - 1 - d;
    for (int i = 0; i < n_basis; ++i) {
      float denom1 = CASE_KNOTS_AUG[i + d] - CASE_KNOTS_AUG[i];
      float denom2 = CASE_KNOTS_AUG[i + d + 1] - CASE_KNOTS_AUG[i + 1];

      float term1 = 0.0f, term2 = 0.0f;
      if (denom1 > 1e-10f)
        term1 = (x - CASE_KNOTS_AUG[i]) / denom1 * B[i];
      if (denom2 > 1e-10f)
        term2 = (CASE_KNOTS_AUG[i + d + 1] - x) / denom2 * B[i + 1];

      B[i] = term1 + term2;
    }
  }

  // Dot product: sum_i coef[i] * B[i]
  float y = 0.0f;
  for (int i = 0; i < num_coef; ++i)
    y += coef[i] * B[i];
  return y;
}

static void float_baseline_forward(const float* x_in, float* y_out) {
  const int in_dim = (int)CASE_IN_DIM;
  const int out_dim = (int)CASE_OUT_DIM;
  const int num_coef = (int)CASE_NUM_COEF;

  for (int j = 0; j < out_dim; ++j) y_out[j] = 0.0f;

  for (int i = 0; i < in_dim; ++i) {
    float x_raw = x_in[i];
    float x = preprocess_x(x_raw);

    float base_val = silu(x);

    for (int j = 0; j < out_dim; ++j) {
      const int edge = i * out_dim + j;
      const float* c = &CASE_BSPLINE_COEFFS[edge * num_coef];

      const float sb = CASE_BSPLINE_SCALES[edge * 3 + 0];
      const float ss = CASE_BSPLINE_SCALES[edge * 3 + 1];
      const float m  = CASE_BSPLINE_SCALES[edge * 3 + 2];

      float spl = bspline_eval_edge(x, c, num_coef);
      y_out[j] += m * (sb * base_val + ss * spl);
    }
  }
}

#endif // CASE_BASIS_TYPE == 1

// =============================================================================
// LUT forward — VARIANT A: Original mixed float/int kernel
// (Same as previous version for reference & backward compatibility)
// =============================================================================

#define CASE_SEG_WIDTH   ((CASE_X_MAX - CASE_X_MIN) / (float)CASE_NUM_SEGMENTS)
#define CASE_INV_SEG_W   ((float)CASE_NUM_SEGMENTS / (CASE_X_MAX - CASE_X_MIN))

static inline float lut_eval_edge(int edge, float x) {
  float pos = (x - (float)CASE_X_MIN) * (float)CASE_INV_SEG_W;
  int seg = (int)pos;
  if (seg < 0) seg = 0;
  if (seg >= (int)CASE_NUM_SEGMENTS) seg = (int)CASE_NUM_SEGMENTS - 1;

  float t = pos - (float)seg;
  if (t < 0.0f) t = 0.0f;
  if (t > 1.0f) t = 1.0f;

  const int base = (edge * CASE_NUM_SEGMENTS + seg) * CASE_L;

#if CASE_INTERP_LINEAR
  const float u = t * (float)(CASE_L - 1);
  int idx = (int)u;
  float frac = u - (float)idx;
  if (idx < 0) idx = 0;
  if (idx >= (int)CASE_L - 1) { idx = (int)CASE_L - 2; frac = 1.0f; }

  const uint8_t q0 = LUT_RD_U8(&CASE_Q_TABLE[base + idx]);
  const uint8_t q1 = LUT_RD_U8(&CASE_Q_TABLE[base + (idx + 1)]);
  float q = (1.0f - frac) * (float)q0 + frac * (float)q1;
#else
  int idx = (int)(t * (float)(CASE_L - 1) + 0.5f);
  if (idx < 0) idx = 0;
  if (idx >= (int)CASE_L) idx = (int)CASE_L - 1;
  float q = (float)LUT_RD_U8(&CASE_Q_TABLE[base + idx]);
#endif

  const int meta_idx = edge * CASE_NUM_SEGMENTS + seg;
#if CASE_Q_SCHEME_ASYMM
  const float sc = LUT_RD_F32(&CASE_SCALE_F32[meta_idx]);
  const float ym = LUT_RD_F32(&CASE_YMIN_F32[meta_idx]);
  return ym + sc * q;
#else
  const int16_t qi = (int16_t)((int)q - 128);
  const float sc = LUT_RD_F32(&CASE_SCALE_F32[meta_idx]);
  return sc * (float)qi;
#endif
}

static void lut_forward(const float* x_in, float* y_out) {
  const int in_dim = (int)CASE_IN_DIM;
  const int out_dim = (int)CASE_OUT_DIM;

  for (int j = 0; j < out_dim; ++j) y_out[j] = 0.0f;

  for (int i = 0; i < in_dim; ++i) {
    float x = preprocess_x(x_in[i]);
    for (int j = 0; j < out_dim; ++j) {
      const int edge = i * out_dim + j;
      y_out[j] += lut_eval_edge(edge, x);
    }
  }
}

// =============================================================================
// LUT forward — VARIANT B: Fixed-point inner loop
//
// KEY DIFFERENCE from Variant A:
//   - Input scaling computed ONCE per input neuron (not per edge)
//   - Segment index, table index, interpolation fraction: ALL integer
//   - Only dequantization uses float (2 float ops per edge)
//
// Per-edge float ops: 2 (dequant only)  vs  Variant A: ~5 per edge
// Per-input float ops: 1 (input-to-fixedpoint conversion)
//
// This addresses Reviewer 1 Comment 2 and Reviewer 2 Comment 2
// regarding the "integer-only" claim precision.
// =============================================================================

// Fixed-point format: Q16.16 for input scaling
#define FP_SHIFT 16
#define FP_ONE   (1L << FP_SHIFT)

// Precomputed fixed-point inverse segment width:
//   INV_SEG_FP = round(NUM_SEGMENTS * 2^16 / (X_MAX - X_MIN))
// Computed as compile-time constant via float cast
#define CASE_INV_SEG_FP  ((int32_t)(((float)CASE_NUM_SEGMENTS / (CASE_X_MAX - CASE_X_MIN)) * (float)FP_ONE + 0.5f))

static void lut_fixedpoint_forward(const float* x_in, float* y_out) {
  const int in_dim = (int)CASE_IN_DIM;
  const int out_dim = (int)CASE_OUT_DIM;

  for (int j = 0; j < out_dim; ++j) y_out[j] = 0.0f;

  for (int i = 0; i < in_dim; ++i) {
    float x = preprocess_x(x_in[i]);

    // === ONE float multiply per input neuron (shared across all edges) ===
    // Convert (x - x_min) to Q16.16 fixed-point
    int32_t x_offset_fp = (int32_t)((x - (float)CASE_X_MIN) * (float)FP_ONE);

    // === Everything below is integer-only until dequantization ===

    // pos_fp = (x - x_min) * num_segments / (x_max - x_min) in Q16.16
    int32_t pos_fp = (int32_t)(((int64_t)x_offset_fp * CASE_INV_SEG_FP) >> FP_SHIFT);

    // Segment index (integer part of pos_fp)
    int seg = (int)(pos_fp >> FP_SHIFT);
    if (seg < 0) seg = 0;
    if (seg >= (int)CASE_NUM_SEGMENTS) seg = (int)CASE_NUM_SEGMENTS - 1;

    // t = fractional part within segment, as Q0.16
    int32_t t_fp = pos_fp - ((int32_t)seg << FP_SHIFT);
    if (t_fp < 0) t_fp = 0;
    if (t_fp > FP_ONE) t_fp = FP_ONE;

    // u = t * (L-1) in Q16.16
    int32_t u_fp = (int32_t)(((int64_t)t_fp * (CASE_L - 1)));  // Q16.16 * int = Q16.16

    // Table index (integer part of u)
    int idx = (int)(u_fp >> FP_SHIFT);
    if (idx < 0) idx = 0;
    if (idx >= (int)CASE_L - 1) idx = (int)CASE_L - 2;

    // Interpolation fraction as Q0.8 (0..255)
    // Take bits [15:8] of u_fp fractional part
    int32_t frac_fp = (u_fp >> (FP_SHIFT - 8)) & 0xFF;

    // === Per-edge: integer table lookup + interpolation + float dequant ===
    for (int j = 0; j < out_dim; ++j) {
      const int edge = i * out_dim + j;
      const int base = (edge * CASE_NUM_SEGMENTS + seg) * CASE_L;

      // Integer table lookup
      const uint8_t q0 = LUT_RD_U8(&CASE_Q_TABLE[base + idx]);
      const uint8_t q1 = LUT_RD_U8(&CASE_Q_TABLE[base + (idx + 1)]);

      // Integer interpolation: q = q0 + frac*(q1-q0)/256
      int16_t delta = (int16_t)q1 - (int16_t)q0;
      int32_t q_int = (int32_t)q0 + (int32_t)((frac_fp * delta) >> 8);

      // Dequantization (2 float ops per edge — the irreducible minimum)
      const int meta_idx = edge * CASE_NUM_SEGMENTS + seg;
#if CASE_Q_SCHEME_ASYMM
      const float sc = LUT_RD_F32(&CASE_SCALE_F32[meta_idx]);
      const float ym = LUT_RD_F32(&CASE_YMIN_F32[meta_idx]);
      y_out[j] += ym + sc * (float)q_int;
#else
      const float sc = LUT_RD_F32(&CASE_SCALE_F32[meta_idx]);
      y_out[j] += sc * (float)((int16_t)q_int - 128);
#endif
    }
  }
}

// =============================================================================
// VARIANT C: Quantization-only baseline (NO LUT)
//
// PURPOSE: Ablation study to isolate the source of speedup.
// This variant quantizes coefficients to int8 but still evaluates
// the polynomial/spline recurrence at runtime.
//
// Comparison:
//   Float baseline   = full float polynomial evaluation
//   Quant-only       = int8 coefficients + float recurrence
//   LUT (Variant A)  = table lookup, mixed float/int
//   LUT-FP (Var. B)  = table lookup, mostly integer
//
// Expected: Quant-only ≈ same speed as float (recurrence dominates),
//           proving that speedup comes from LUT structure, not quantization.
//
// This addresses Reviewer 3 Comment 2.
// =============================================================================

#if CASE_BASIS_TYPE == 0  // Jacobi only for this ablation

static void quant_only_forward(const float* x_in, float* y_out) {
  const int in_dim = (int)CASE_IN_DIM;
  const int out_dim = (int)CASE_OUT_DIM;
  const int deg = (int)CASE_DEGREE;

  for (int j = 0; j < out_dim; ++j) y_out[j] = 0.0f;

  float P[CASE_DEGREE + 1];

  for (int i = 0; i < in_dim; ++i) {
    float x = preprocess_x(x_in[i]);
    // Still compute full Jacobi recurrence in float (this is the expensive part)
    jacobi_basis(x, P);

    for (int j = 0; j < out_dim; ++j) {
      const int edge = i * out_dim + j;
      const float* c = &CASE_FLOAT_COEFFS[edge * (deg + 1)];

      // Quantize coefficients on-the-fly to int8, then dequantize and accumulate
      // This simulates "quantized weights but no LUT"
      float c_min = c[0], c_max = c[0];
      for (int k = 1; k <= deg; ++k) {
        if (c[k] < c_min) c_min = c[k];
        if (c[k] > c_max) c_max = c[k];
      }
      float c_scale = (c_max - c_min) / 255.0f;
      if (c_scale < 1e-10f) c_scale = 1e-10f;

      float acc = 0.0f;
      for (int k = 0; k <= deg; ++k) {
        uint8_t cq = (uint8_t)((c[k] - c_min) / c_scale + 0.5f);
        float c_deq = c_min + c_scale * (float)cq;
        acc += c_deq * P[k];
      }
      y_out[j] += acc;
    }
  }
}

#endif // CASE_BASIS_TYPE == 0

// =============================================================================
// Benchmark harness
// =============================================================================

static float x_in[CASE_IN_DIM];
static float y_float[CASE_OUT_DIM];
static float y_lut[CASE_OUT_DIM];
static float y_lut_fp[CASE_OUT_DIM];
#if CASE_BASIS_TYPE == 0
static float y_qonly[CASE_OUT_DIM];
#endif

static uint32_t lcg_state = 1;
static inline uint32_t lcg_u32() {
  lcg_state = 1664525u * lcg_state + 1013904223u;
  return lcg_state;
}
static inline float u01() {
  return (float)(lcg_u32() >> 8) * (1.0f / 16777216.0f);
}
static inline float rand_uniform(float a, float b) {
  return a + (b - a) * u01();
}
static inline float rand_gauss(float mean, float stddev) {
  float u1 = u01(), u2 = u01();
  if (u1 < 1e-7f) u1 = 1e-7f;
  return mean + stddev * sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

static void fill_input(int rep) {
  lcg_state = 0xC0FFEEu ^ (uint32_t)rep;

  if (strcmp(CASE_INPUT_MODE, "linspace") == 0) {
    for (int i = 0; i < (int)CASE_IN_DIM; ++i) {
      float t = (CASE_IN_DIM > 1) ? (float)i / (float)(CASE_IN_DIM - 1) : 0.5f;
      x_in[i] = CASE_X_MIN + (CASE_X_MAX - CASE_X_MIN) * t;
    }
    return;
  }
  if (strcmp(CASE_INPUT_MODE, "rng_uniform") == 0) {
    for (int i = 0; i < (int)CASE_IN_DIM; ++i)
      x_in[i] = rand_uniform(CASE_X_MIN, CASE_X_MAX);
    return;
  }
  if (strcmp(CASE_INPUT_MODE, "rng_gauss") == 0) {
    for (int i = 0; i < (int)CASE_IN_DIM; ++i)
      x_in[i] = rand_gauss(0.0f, 1.0f);
    return;
  }
  // mixed
  for (int i = 0; i < (int)CASE_IN_DIM; ++i) {
    if ((i & 1) == 0) x_in[i] = rand_uniform(CASE_X_MIN, CASE_X_MAX);
    else x_in[i] = rand_uniform(CASE_X_MAX, CASE_X_MAX + 2.0f);
  }
}

static float max_abs_diff(const float* a, const float* b, int n) {
  float m = 0.0f;
  for (int i = 0; i < n; ++i) {
    float d = fabsf(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

static uint32_t bench_us(void (*fn)(const float*, float*), const float* inp, float* out,
                         int iters, int warmup) {
  for (int k = 0; k < warmup; ++k) fn(inp, out);
  uint32_t t0 = (uint32_t)micros();
  for (int k = 0; k < iters; ++k) fn(inp, out);
  return (uint32_t)((uint32_t)micros() - t0);
}

static inline void sort_u32(uint32_t* a, int n) {
  for (int i = 1; i < n; ++i) {
    uint32_t key = a[i]; int j = i - 1;
    while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j--; }
    a[j + 1] = key;
  }
}
static inline uint32_t median_u32(uint32_t* a, int n) {
  sort_u32(a, n);
  return a[n / 2];
}

// Print full JSON result line with all variants
static void print_json_v2(
    uint32_t tf_med, uint32_t tl_med, uint32_t tl_fp_med,
    uint32_t tf_min, uint32_t tf_max,
    uint32_t tl_min, uint32_t tl_max,
    uint32_t tl_fp_min, uint32_t tl_fp_max,
#if CASE_BASIS_TYPE == 0
    uint32_t tq_med, uint32_t tq_min, uint32_t tq_max,
    float err_qonly,
#endif
    float max_err, float max_err_fp)
{
  Serial.print("LUTKAN:");
  Serial.print('{');
  Serial.print("\"target\":\"");     Serial.print(CASE_TARGET);    Serial.print("\",");
  Serial.print("\"case_id\":\"");    Serial.print(CASE_ID);        Serial.print("\",");

#if CASE_BASIS_TYPE == 0
  Serial.print("\"basis_type\":\"jacobi\",");
  Serial.print("\"poly_family\":\""); Serial.print(CASE_POLY_FAMILY); Serial.print("\",");
  Serial.print("\"degree\":");       Serial.print(CASE_DEGREE);     Serial.print(',');
#else
  Serial.print("\"basis_type\":\"bspline\",");
  Serial.print("\"poly_family\":\"bspline\",");
  Serial.print("\"degree\":");       Serial.print(CASE_BSPLINE_DEGREE); Serial.print(',');
  Serial.print("\"num_coef\":");     Serial.print(CASE_NUM_COEF);       Serial.print(',');
#endif

  Serial.print("\"in_dim\":");       Serial.print(CASE_IN_DIM);    Serial.print(',');
  Serial.print("\"out_dim\":");      Serial.print(CASE_OUT_DIM);   Serial.print(',');
  Serial.print("\"L\":");            Serial.print(CASE_L);         Serial.print(',');
  Serial.print("\"segments\":");     Serial.print(CASE_NUM_SEGMENTS); Serial.print(',');
  Serial.print("\"interp\":\"");     Serial.print(CASE_INTERP_NAME); Serial.print("\",");
  Serial.print("\"scheme\":\"");     Serial.print(CASE_Q_SCHEME_NAME); Serial.print("\",");

  // --- Original float baseline ---
  Serial.print("\"t_float_us\":");   Serial.print(tf_med);         Serial.print(',');
  Serial.print("\"t_float_min_us\":"); Serial.print(tf_min);       Serial.print(',');
  Serial.print("\"t_float_max_us\":"); Serial.print(tf_max);       Serial.print(',');

  // --- LUT Variant A (mixed float/int, original) ---
  Serial.print("\"t_lut_us\":");     Serial.print(tl_med);         Serial.print(',');
  Serial.print("\"t_lut_min_us\":"); Serial.print(tl_min);         Serial.print(',');
  Serial.print("\"t_lut_max_us\":"); Serial.print(tl_max);         Serial.print(',');

  // --- LUT Variant B (fixed-point inner loop) ---
  Serial.print("\"t_lut_fp_us\":");  Serial.print(tl_fp_med);      Serial.print(',');
  Serial.print("\"t_lut_fp_min_us\":"); Serial.print(tl_fp_min);   Serial.print(',');
  Serial.print("\"t_lut_fp_max_us\":"); Serial.print(tl_fp_max);   Serial.print(',');

#if CASE_BASIS_TYPE == 0
  // --- Quant-only baseline (Variant C, Jacobi only) ---
  Serial.print("\"t_qonly_us\":");   Serial.print(tq_med);         Serial.print(',');
  Serial.print("\"t_qonly_min_us\":"); Serial.print(tq_min);       Serial.print(',');
  Serial.print("\"t_qonly_max_us\":"); Serial.print(tq_max);       Serial.print(',');
  Serial.print("\"qonly_speedup\":");
  Serial.print((tq_med > 0) ? ((float)tf_med / (float)tq_med) : 0.0f, 4);
  Serial.print(',');
  Serial.print("\"qonly_max_err\":"); Serial.print(err_qonly, 8);  Serial.print(',');
#endif

  Serial.print("\"repeats\":");      Serial.print(CASE_REPEATS);   Serial.print(',');
  Serial.print("\"iters\":");        Serial.print(CASE_ITERS);     Serial.print(',');
  Serial.print("\"warmup\":");       Serial.print(CASE_WARMUP);    Serial.print(',');
  Serial.print("\"input_mode\":\""); Serial.print(CASE_INPUT_MODE); Serial.print("\",");

  // --- Speedups ---
  float speedup_lut = (tl_med > 0) ? ((float)tf_med / (float)tl_med) : 0.0f;
  float speedup_fp  = (tl_fp_med > 0) ? ((float)tf_med / (float)tl_fp_med) : 0.0f;
  Serial.print("\"speedup\":");      Serial.print(speedup_lut, 4); Serial.print(',');
  Serial.print("\"speedup_fp\":");   Serial.print(speedup_fp, 4);  Serial.print(',');
  Serial.print("\"max_abs_err\":");  Serial.print(max_err, 8);     Serial.print(',');
  Serial.print("\"max_abs_err_fp\":"); Serial.print(max_err_fp, 8);

  // --- Memory footprint (bytes) ---
  Serial.print(",\"lut_flash_bytes\":");
  uint32_t lut_bytes = (uint32_t)CASE_IN_DIM * CASE_OUT_DIM * CASE_NUM_SEGMENTS * CASE_L;
  uint32_t meta_bytes = (uint32_t)CASE_IN_DIM * CASE_OUT_DIM * CASE_NUM_SEGMENTS * 8u; // 2x float32
  Serial.print(lut_bytes + meta_bytes);

  Serial.print('}');
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(200);

  const int iters  = (int)CASE_ITERS;
  const int reps   = (int)CASE_REPEATS;
  const int warmup = (int)CASE_WARMUP;

  uint32_t tf_s[CASE_REPEATS];
  uint32_t tl_s[CASE_REPEATS];
  uint32_t tl_fp_s[CASE_REPEATS];
#if CASE_BASIS_TYPE == 0
  uint32_t tq_s[CASE_REPEATS];
#endif
  float err_max = 0.0f;
  float err_fp_max = 0.0f;
#if CASE_BASIS_TYPE == 0
  float err_qonly_max = 0.0f;
#endif

  for (int r = 0; r < reps; ++r) {
    fill_input(r);

    // Accuracy: float baseline vs all variants
    float_baseline_forward(x_in, y_float);
    lut_forward(x_in, y_lut);
    lut_fixedpoint_forward(x_in, y_lut_fp);
#if CASE_BASIS_TYPE == 0
    quant_only_forward(x_in, y_qonly);
#endif

    float err = max_abs_diff(y_float, y_lut, (int)CASE_OUT_DIM);
    float err_fp = max_abs_diff(y_float, y_lut_fp, (int)CASE_OUT_DIM);
    if (err > err_max) err_max = err;
    if (err_fp > err_fp_max) err_fp_max = err_fp;
#if CASE_BASIS_TYPE == 0
    float err_q = max_abs_diff(y_float, y_qonly, (int)CASE_OUT_DIM);
    if (err_q > err_qonly_max) err_qonly_max = err_q;
#endif

    // Timing: all variants
    tf_s[r] = bench_us(float_baseline_forward, x_in, y_float, iters, warmup);
    tl_s[r] = bench_us(lut_forward, x_in, y_lut, iters, warmup);
    tl_fp_s[r] = bench_us(lut_fixedpoint_forward, x_in, y_lut_fp, iters, warmup);
#if CASE_BASIS_TYPE == 0
    tq_s[r] = bench_us(quant_only_forward, x_in, y_qonly, iters, warmup);
#endif
  }

  // Compute min/max
  uint32_t tf_min = tf_s[0], tf_max = tf_s[0];
  uint32_t tl_min = tl_s[0], tl_max = tl_s[0];
  uint32_t tl_fp_min = tl_fp_s[0], tl_fp_max = tl_fp_s[0];
#if CASE_BASIS_TYPE == 0
  uint32_t tq_min = tq_s[0], tq_max = tq_s[0];
#endif
  for (int r = 1; r < reps; ++r) {
    if (tf_s[r] < tf_min) tf_min = tf_s[r];
    if (tf_s[r] > tf_max) tf_max = tf_s[r];
    if (tl_s[r] < tl_min) tl_min = tl_s[r];
    if (tl_s[r] > tl_max) tl_max = tl_s[r];
    if (tl_fp_s[r] < tl_fp_min) tl_fp_min = tl_fp_s[r];
    if (tl_fp_s[r] > tl_fp_max) tl_fp_max = tl_fp_s[r];
#if CASE_BASIS_TYPE == 0
    if (tq_s[r] < tq_min) tq_min = tq_s[r];
    if (tq_s[r] > tq_max) tq_max = tq_s[r];
#endif
  }

  print_json_v2(
    median_u32(tf_s, reps), median_u32(tl_s, reps), median_u32(tl_fp_s, reps),
    tf_min, tf_max,
    tl_min, tl_max,
    tl_fp_min, tl_fp_max,
#if CASE_BASIS_TYPE == 0
    median_u32(tq_s, reps), tq_min, tq_max,
    err_qonly_max,
#endif
    err_max, err_fp_max
  );
}

void loop() {
  delay(1000);
}
