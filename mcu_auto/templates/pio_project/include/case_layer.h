#pragma once
// =========================================================================
// This is a PLACEHOLDER header.
// gen_cases.py will overwrite it with actual LUT data per case.
// =========================================================================

#define CASE_TARGET "placeholder"
#define CASE_ID "placeholder"

// Basis type selector: 0 = Jacobi, 1 = B-spline
#define CASE_BASIS_TYPE 0

// ---- Jacobi-specific (used when CASE_BASIS_TYPE == 0) -------------------
#define CASE_POLY_FAMILY "chebyshev_t"
#define CASE_DEGREE 3
#define CASE_ALPHA (-0.5f)
#define CASE_BETA (-0.5f)

// ---- B-spline-specific (used when CASE_BASIS_TYPE == 1) -----------------
#define CASE_BSPLINE_DEGREE 3
#define CASE_NUM_COEF 7
// Augmented knot vector length = grid_points + 2*degree
#define CASE_NUM_KNOTS_AUG 11

// ---- Shared layer geometry ----------------------------------------------
#define CASE_IN_DIM 4
#define CASE_OUT_DIM 4
#define CASE_L 32
#define CASE_NUM_SEGMENTS 8
#define CASE_NUM_KNOTS (CASE_NUM_SEGMENTS + 1)

#define CASE_X_MIN (-3.0f)
#define CASE_X_MAX (3.0f)
#define CASE_USE_TANH 1
#define CASE_CLIP_X 1
#define CASE_INTERP_LINEAR 1
#define CASE_Q_SCHEME_ASYMM 1
#define CASE_ITERS 200
#define CASE_REPEATS 5
#define CASE_WARMUP 20
#define CASE_INPUT_MODE "linspace"

#define CASE_INTERP_NAME "linear"
#define CASE_Q_SCHEME_NAME "uint8_asymm"

static const float CASE_KNOTS[CASE_NUM_KNOTS] = {
  -3.0f,-2.25f,-1.5f,-0.75f,0.0f,0.75f,1.5f,2.25f,3.0f
};

// Shape: [EDGES][SEGMENTS][L] flattened as (edge*SEGMENTS+seg)*L + idx
static const uint8_t CASE_Q_TABLE[CASE_IN_DIM*CASE_OUT_DIM*CASE_NUM_SEGMENTS*CASE_L] = {0};

// Per (edge,segment) meta
static const uint16_t CASE_SCALE_F16[CASE_IN_DIM*CASE_OUT_DIM*CASE_NUM_SEGMENTS] = {0x3C00}; // 1.0
static const uint16_t CASE_YMIN_F16[CASE_IN_DIM*CASE_OUT_DIM*CASE_NUM_SEGMENTS] = {0x0000};

// Jacobi float coefficients per edge: (degree+1)
static const float CASE_FLOAT_COEFFS[CASE_IN_DIM*CASE_OUT_DIM*(CASE_DEGREE+1)] = {0};

// B-spline float coefficients per edge: (num_coef)
static const float CASE_BSPLINE_COEFFS[CASE_IN_DIM*CASE_OUT_DIM*CASE_NUM_COEF] = {0};
// Augmented knot vector for B-spline
static const float CASE_KNOTS_AUG[CASE_NUM_KNOTS_AUG] = {0};
// Per-edge scale factors: sb, ss, m  — packed [EDGES][3]
static const float CASE_BSPLINE_SCALES[CASE_IN_DIM*CASE_OUT_DIM*3] = {0};
