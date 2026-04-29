/*
 * ProbNet - tensor.h
 * Core tensor operations: AVX-512 matmul, LayerNorm, Softmax, RMSNorm
 * CPU-only (GPU path via CUDA backend if available)
 *
 * Usage:  #include "tensor.h"
 * Compile: gcc -O3 -march=native -mavx512f -mfma -fopenmp -ffast-math
 */
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <omp.h>
#ifdef __AVX512F__
#include <immintrin.h>
#define HAS_AVX512 1
#else
#define HAS_AVX512 0
#endif

/* ── Memory ────────────────────────────────────────────────────── */
static inline float* pn_alloc(size_t n) {
    if (n == 0) return NULL;
    void* p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(float)) != 0) {
        fprintf(stderr, "[ProbNet] OOM: %zu floats (%zu MB)\n", n, n*4/1048576);
        exit(1);
    }
    memset(p, 0, n * sizeof(float));
    return (float*)p;
}
#define pn_free(p)  do { if(p){free(p);(p)=NULL;} } while(0)

static inline void pn_copy(float* dst, const float* src, size_t n) {
    memcpy(dst, src, n * sizeof(float));
}
static inline void pn_zero(float* p, size_t n) {
    memset(p, 0, n * sizeof(float));
}
static inline void pn_scale(float* p, float s, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] *= s;
}
static inline void pn_add(float* dst, const float* src, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] += src[i];
}
static inline void pn_add_scaled(float* dst, const float* src, float s, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] += src[i] * s;
}

/* ── AVX-512 horizontal sum ─────────────────────────────────────── */
#if HAS_AVX512
static inline float hsum512(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 s  = _mm256_add_ps(lo, hi);
    __m128 a  = _mm256_castps256_ps128(s);
    __m128 b  = _mm256_extractf128_ps(s, 1);
    __m128 c  = _mm_add_ps(a, b);
    __m128 d  = _mm_movehl_ps(c, c);
    __m128 e  = _mm_add_ps(c, d);
    __m128 f  = _mm_shuffle_ps(e, e, 1);
    return _mm_cvtss_f32(_mm_add_ss(e, f));
}
static inline float dot_avx512(const float* a, const float* b, int K) {
    __m512 acc = _mm512_setzero_ps();
    int k = 0;
    for (; k <= K-16; k += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a+k), _mm512_loadu_ps(b+k), acc);
    float s = hsum512(acc);
    for (; k < K; k++) s += a[k]*b[k];
    return s;
}
#else
static inline float dot_avx512(const float* a, const float* b, int K) {
    float s = 0;
    for (int k = 0; k < K; k++) s += a[k]*b[k];
    return s;
}
#endif

/* ── Matrix multiply: C[M,N] = A[M,K] @ B^T[N,K]  ─────────────── */
static inline void matmul_nt(const float* A, const float* BT, float* C,
                              int M, int N, int K) {
    #pragma omp parallel for schedule(static) if((long)M*N*K > 32768)
    for (int m = 0; m < M; m++) {
        const float* ar = A + (size_t)m*K;
        float* cr = C + (size_t)m*N;
        for (int n = 0; n < N; n++)
            cr[n] = dot_avx512(ar, BT + (size_t)n*K, K);
    }
}

/* C[M,N] = A[M,K] @ B[K,N] */
static inline void matmul_nn(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    float* BT = pn_alloc((size_t)K*N);
    for (int k=0;k<K;k++) for (int n=0;n<N;n++) BT[(size_t)n*K+k]=B[(size_t)k*N+n];
    matmul_nt(A, BT, C, M, N, K);
    pn_free(BT);
}

static inline void add_bias(float* out, const float* bias, int M, int N) {
    for (int m=0;m<M;m++) {
        float* r = out + (size_t)m*N;
        for (int n=0;n<N;n++) r[n] += bias[n];
    }
}

/* ── Activations ────────────────────────────────────────────────── */
static inline float siluf(float x)    { return x / (1.0f + expf(-x)); }
static inline float reluf(float x)    { return x > 0.0f ? x : 0.0f; }
static inline float geluapproxf(float x) {
    return 0.5f*x*(1.0f+tanhf(0.7978845608f*(x+0.044715f*x*x*x)));
}

/* ── Normalization ──────────────────────────────────────────────── */
static inline void layernorm(const float* x, const float* gamma,
                              const float* beta, float* out, int M, int N, float eps) {
    #pragma omp parallel for schedule(static) if(M > 4)
    for (int m=0;m<M;m++) {
        const float* xr = x + (size_t)m*N;
        float* or_ = out + (size_t)m*N;
        double mean=0, var=0;
        for (int n=0;n<N;n++) mean+=xr[n];  mean/=N;
        for (int n=0;n<N;n++) { double d=xr[n]-mean; var+=d*d; } var/=N;
        float is = 1.0f/sqrtf((float)var+eps);
        for (int n=0;n<N;n++)
            or_[n] = (xr[n]-(float)mean)*is*(gamma?gamma[n]:1.0f)+(beta?beta[n]:0.0f);
    }
}

static inline void rmsnorm(const float* x, const float* gamma,
                            float* out, int M, int N, float eps) {
    #pragma omp parallel for schedule(static) if(M > 4)
    for (int m=0;m<M;m++) {
        const float* xr = x + (size_t)m*N;
        float* or_ = out + (size_t)m*N;
        float ss=0;
        for (int n=0;n<N;n++) ss+=xr[n]*xr[n];
        float is=1.0f/sqrtf(ss/N+eps);
        for (int n=0;n<N;n++)
            or_[n]=xr[n]*is*(gamma?gamma[n]:1.0f);
    }
}

/* ── Softmax ────────────────────────────────────────────────────── */
static inline void softmax_rows(float* x, int M, int N) {
    #pragma omp parallel for schedule(static) if(M > 8)
    for (int m=0;m<M;m++) {
        float* r=x+(size_t)m*N;
        float mx=-FLT_MAX;
        for (int n=0;n<N;n++) if(r[n]>mx) mx=r[n];
        float sum=0;
        for (int n=0;n<N;n++) { r[n]=expf(r[n]-mx); sum+=r[n]; }
        float inv=1.0f/sum;
        for (int n=0;n<N;n++) r[n]*=inv;
    }
}

/* ── Rotary Position Embedding (RoPE) ───────────────────────────── */
static inline void rope_apply(float* x, int M, int H, int D, int pos_offset) {
    /* x: [M, H, D]  D=head_dim  H=n_heads */
    int half = D/2;
    for (int m=0;m<M;m++) for (int h=0;h<H;h++) {
        float* xh = x + (size_t)m*H*D + (size_t)h*D;
        int pos = pos_offset + m;
        for (int i=0;i<half;i++) {
            float theta = (float)pos / powf(10000.0f, 2.0f*i/D);
            float c=cosf(theta), s=sinf(theta);
            float x0=xh[i], x1=xh[i+half];
            xh[i]      = x0*c - x1*s;
            xh[i+half] = x0*s + x1*c;
        }
    }
}

/* ── RNG: xoshiro256** ──────────────────────────────────────────── */
typedef struct { uint64_t s[4]; } RNG;
static inline uint64_t rng_next(RNG* r) {
    uint64_t t=r->s[1]<<17;
    r->s[2]^=r->s[0]; r->s[3]^=r->s[1];
    r->s[1]^=r->s[2]; r->s[0]^=r->s[3];
    r->s[2]^=t; r->s[3]=(r->s[3]<<45)|(r->s[3]>>19);
    return r->s[0]+r->s[3];
}
static inline float rng_uniform(RNG* r) {
    return (float)(rng_next(r)>>11)*(1.0f/(1ULL<<53));
}
static inline float rng_normal(RNG* r) {
    float u1=rng_uniform(r)+1e-10f, u2=rng_uniform(r);
    return sqrtf(-2.0f*logf(u1))*cosf(6.2831853f*u2);
}
static inline RNG rng_seed(uint64_t s) {
    RNG r={{s,s^0xdeadbeef,s^0xcafe,s^0x1234}};
    for(int i=0;i<20;i++) rng_next(&r);
    return r;
}
static inline void init_normal(float* w, size_t n, float std, RNG* r) {
    for (size_t i=0;i<n;i++) w[i]=rng_normal(r)*std;
}
static inline void init_ones(float* w, size_t n) {
    for (size_t i=0;i<n;i++) w[i]=1.0f;
}

/* ── Top-k / Top-p sampling utilities ──────────────────────────── */
/* Apply temperature scaling + top-k mask to logits */
static inline void apply_temperature(float* logits, int V, float temp) {
    if (temp > 0.0f && temp != 1.0f)
        for (int i=0;i<V;i++) logits[i]/=temp;
}

/* Sample from probability distribution */
static inline int sample_categorical(const float* probs, int V, RNG* rng) {
    float u = rng_uniform(rng);
    float cum = 0.0f;
    for (int i=0;i<V;i++) {
        cum += probs[i];
        if (u <= cum) return i;
    }
    return V-1;
}

/* Top-p (nucleus) sampling — thread-safe, idx_buf must be V ints */
static inline int sample_topp(float* logits, int V, float p, float temp, RNG* rng, int* idx_buf) {
    apply_temperature(logits, V, temp);
    /* compute softmax */
    float mx=-FLT_MAX;
    for(int i=0;i<V;i++) if(logits[i]>mx) mx=logits[i];
    float sum=0;
    for(int i=0;i<V;i++){logits[i]=expf(logits[i]-mx);sum+=logits[i];}
    for(int i=0;i<V;i++) logits[i]/=sum;
    /* sort indices by prob descending */
    for(int i=0;i<V;i++) idx_buf[i]=i;
    /* partial sort: find nucleus */
    float cum=0; int cutoff=V;
    for(int i=0;i<V&&cum<p;i++){
        int best=i;
        for(int j=i+1;j<V;j++) if(logits[j]>logits[best]) best=j;
        float tmp=logits[i];logits[i]=logits[best];logits[best]=tmp;
        int ti=idx_buf[i];idx_buf[i]=idx_buf[best];idx_buf[best]=ti;
        cum+=logits[i];
        if(cum>=p){cutoff=i+1;break;}
    }
    /* renormalize top-p */
    float s2=0; for(int i=0;i<cutoff;i++) s2+=logits[i];
    for(int i=0;i<cutoff;i++) logits[i]/=s2;
    int pick=sample_categorical(logits,cutoff,rng);
    return idx_buf[pick];
}

/* Top-k sampling — thread-safe, idx_buf must be V ints */
static inline int sample_topk(float* logits, int V, int k, float temp, RNG* rng, int* idx_buf) {
    apply_temperature(logits, V, temp);
    if (k >= V) {
        float mx=-FLT_MAX;
        for(int i=0;i<V;i++) if(logits[i]>mx) mx=logits[i];
        float sum=0;
        for(int i=0;i<V;i++){logits[i]=expf(logits[i]-mx);sum+=logits[i];}
        for(int i=0;i<V;i++) logits[i]/=sum;
        return sample_categorical(logits,V,rng);
    }
    /* find top-k */
    for(int i=0;i<V;i++) idx_buf[i]=i;
    for(int i=0;i<k;i++){
        int best=i;
        for(int j=i+1;j<V;j++) if(logits[j]>logits[best]) best=j;
        float tmp=logits[i];logits[i]=logits[best];logits[best]=tmp;
        int ti=idx_buf[i];idx_buf[i]=idx_buf[best];idx_buf[best]=ti;
    }
    float mx=logits[0],sum=0;
    for(int i=0;i<k;i++){logits[i]=expf(logits[i]-mx);sum+=logits[i];}
    for(int i=0;i<k;i++) logits[i]/=sum;
    int pick=sample_categorical(logits,k,rng);
    return idx_buf[pick];
}
