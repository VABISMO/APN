/*
 * ProbNet - attention.h
 * Multi-Head Attention with KV-cache for efficient autoregressive generation.
 * Supports standard MHA (GPT/LLaMA/Gemma style).
 * Causal masking built-in for decoder-only models.
 */
#pragma once
#include "tensor.h"
#include "optimizer.h"

typedef struct {
    int d_model;     /* model dimension */
    int n_heads;     /* number of attention heads */
    int n_kv_heads;  /* KV heads (= n_heads for MHA, < n_heads for GQA) */
    int head_dim;    /* d_model / n_heads */
    int max_seq_len; /* maximum sequence length (for KV cache) */
    int use_rope;    /* use rotary position embeddings */

    /* projection weights [d_model, d_model] stored as [out, in] for matmul_nt */
    float* W_q;      /* [n_heads*head_dim, d_model] */
    float* W_k;      /* [n_kv_heads*head_dim, d_model] */
    float* W_v;      /* [n_kv_heads*head_dim, d_model] */
    float* W_o;      /* [d_model, n_heads*head_dim] */

    float* b_q;      /* [n_heads*head_dim] or NULL */
    float* b_k;
    float* b_v;
    float* b_o;

    /* KV cache for autoregressive generation */
    float* kv_cache_k;  /* [max_seq_len, n_kv_heads, head_dim] */
    float* kv_cache_v;  /* [max_seq_len, n_kv_heads, head_dim] */
    int    kv_len;       /* filled positions in KV cache */

    /* forward cache (for training backward) */
    float* q_save, *k_save, *v_save;
    float* attn_save;  /* [M, n_heads, M_or_kv_len] */
    float* x_save;
    int M_cache;
} MHAttention;

static inline MHAttention* mha_new(int d_model, int n_heads, int n_kv_heads,
                                    int max_seq_len, int use_rope, uint64_t seed) {
    MHAttention* a = (MHAttention*)calloc(1, sizeof(MHAttention));
    a->d_model     = d_model;
    a->n_heads     = n_heads;
    a->n_kv_heads  = n_kv_heads > 0 ? n_kv_heads : n_heads;
    a->head_dim    = d_model / n_heads;
    a->max_seq_len = max_seq_len;
    a->use_rope    = use_rope;

    int D=d_model, H=n_heads, Hkv=a->n_kv_heads, hd=a->head_dim;
    RNG r = rng_seed(seed);
    float s = sqrtf(1.0f/D);

    a->W_q = pn_alloc((size_t)H*hd*D);  init_normal(a->W_q,(size_t)H*hd*D,  s,&r);
    a->W_k = pn_alloc((size_t)Hkv*hd*D);init_normal(a->W_k,(size_t)Hkv*hd*D,s,&r);
    a->W_v = pn_alloc((size_t)Hkv*hd*D);init_normal(a->W_v,(size_t)Hkv*hd*D,s,&r);
    a->W_o = pn_alloc((size_t)D*H*hd);  init_normal(a->W_o,(size_t)D*H*hd,  s,&r);

    a->b_q = pn_alloc(H*hd);
    a->b_k = pn_alloc(Hkv*hd);
    a->b_v = pn_alloc(Hkv*hd);
    a->b_o = pn_alloc(D);

    /* KV cache */
    a->kv_cache_k = pn_alloc((size_t)max_seq_len*Hkv*hd);
    a->kv_cache_v = pn_alloc((size_t)max_seq_len*Hkv*hd);
    a->kv_len = 0;

    return a;
}

static inline void mha_free(MHAttention* a) {
    if (!a) return;
    pn_free(a->W_q); pn_free(a->W_k); pn_free(a->W_v); pn_free(a->W_o);
    pn_free(a->b_q); pn_free(a->b_k); pn_free(a->b_v); pn_free(a->b_o);
    pn_free(a->kv_cache_k); pn_free(a->kv_cache_v);
    pn_free(a->q_save); pn_free(a->k_save); pn_free(a->v_save);
    pn_free(a->attn_save); pn_free(a->x_save);
    free(a);
}

static inline void mha_reset_cache(MHAttention* a) { a->kv_len = 0; }

/* Ensure scratch buffers */
static inline void mha_ensure(MHAttention* a, int M) {
    if (a->M_cache == M) return;
    pn_free(a->q_save); pn_free(a->k_save); pn_free(a->v_save);
    pn_free(a->attn_save); pn_free(a->x_save);
    int H=a->n_heads, Hkv=a->n_kv_heads, hd=a->head_dim, D=a->d_model;
    a->q_save   = pn_alloc((size_t)M*H*hd);
    a->k_save   = pn_alloc((size_t)M*Hkv*hd);
    a->v_save   = pn_alloc((size_t)M*Hkv*hd);
    a->attn_save= pn_alloc((size_t)M*H*M);  /* training: kv_len_total == M */
    a->x_save   = pn_alloc((size_t)M*D);
    a->M_cache  = M;
}

/*
 * Forward pass.
 * x:   [M, D]
 * out: [M, D]
 * use_cache: if 1, append K/V to kv_cache (for generation)
 * past_len:  how many tokens already in cache (for position encoding)
 */
static inline void mha_forward(MHAttention* a, const float* x, float* out,
                                 int M, int use_cache, int past_len) {
    int D=a->d_model, H=a->n_heads, Hkv=a->n_kv_heads, hd=a->head_dim;
    mha_ensure(a, M);
    pn_copy(a->x_save, x, (size_t)M*D);

    /* Project Q, K, V */
    matmul_nt(x, a->W_q, a->q_save, M, H*hd, D);  add_bias(a->q_save, a->b_q, M, H*hd);
    matmul_nt(x, a->W_k, a->k_save, M, Hkv*hd, D);add_bias(a->k_save, a->b_k, M, Hkv*hd);
    matmul_nt(x, a->W_v, a->v_save, M, Hkv*hd, D);add_bias(a->v_save, a->b_v, M, Hkv*hd);

    /* Apply RoPE */
    if (a->use_rope) {
        rope_apply(a->q_save, M, H,   hd, past_len);
        rope_apply(a->k_save, M, Hkv, hd, past_len);
    }

    /* KV cache management */
    const float* K_full;
    const float* V_full;
    int kv_len_total;

    if (use_cache) {
        /* Append current K, V to cache */
        for (int m=0;m<M;m++) {
            int pos = past_len + m;
            if (pos < a->max_seq_len) {
                pn_copy(a->kv_cache_k+(size_t)pos*Hkv*hd, a->k_save+(size_t)m*Hkv*hd, Hkv*hd);
                pn_copy(a->kv_cache_v+(size_t)pos*Hkv*hd, a->v_save+(size_t)m*Hkv*hd, Hkv*hd);
            }
        }
        a->kv_len = past_len + M;
        K_full = a->kv_cache_k;
        V_full = a->kv_cache_v;
        kv_len_total = a->kv_len;
    } else {
        K_full = a->k_save;
        V_full = a->v_save;
        kv_len_total = M;
    }

    float scale = 1.0f / sqrtf((float)hd);
    float* attn_out = pn_alloc((size_t)M*H*hd);
    float* attn_w   = pn_alloc((size_t)M*H*kv_len_total);

    for (int h=0;h<H;h++) {
        int hkv = h % Hkv;  /* GQA: multiple Q heads share one KV head */
        for (int m=0;m<M;m++) {
            float* q = a->q_save + (size_t)m*H*hd + (size_t)h*hd;
            float* aw = attn_w + (size_t)m*H*kv_len_total + (size_t)h*kv_len_total;
            /* Compute attention scores */
            for (int kp=0;kp<kv_len_total;kp++) {
                float* k = (float*)K_full + (size_t)kp*Hkv*hd + (size_t)hkv*hd;
                aw[kp] = dot_avx512(q, k, hd) * scale;
            }
            /* Causal mask: future positions → -inf */
            int cur_pos = past_len + m;
            for (int kp=cur_pos+1;kp<kv_len_total;kp++)
                aw[kp] = -1e9f;
        }
        /* Softmax over key dimension */
        for (int m=0;m<M;m++) {
            float* aw = attn_w + (size_t)m*H*kv_len_total + (size_t)h*kv_len_total;
            float mx=-FLT_MAX;
            for (int kp=0;kp<kv_len_total;kp++) if(aw[kp]>mx) mx=aw[kp];
            float s=0;
            for (int kp=0;kp<kv_len_total;kp++){aw[kp]=expf(aw[kp]-mx);s+=aw[kp];}
            for (int kp=0;kp<kv_len_total;kp++) aw[kp]/=s;
        }
        /* Weighted sum of values */
        for (int m=0;m<M;m++) {
            float* ao = attn_out + (size_t)m*H*hd + (size_t)h*hd;
            float* aw = attn_w + (size_t)m*H*kv_len_total + (size_t)h*kv_len_total;
            pn_zero(ao, hd);
            for (int kp=0;kp<kv_len_total;kp++) {
                float w = aw[kp];
                float* v = (float*)V_full + (size_t)kp*Hkv*hd + (size_t)hkv*hd;
                pn_add_scaled(ao, v, w, hd);
            }
        }
    }

    /* Save attention weights for backward (training only, no KV cache) */
    if (!use_cache) {
        /* Re-alloc attn_save with correct size and copy */
        pn_free(a->attn_save);
        a->attn_save = pn_alloc((size_t)M*H*kv_len_total);
        pn_copy(a->attn_save, attn_w, (size_t)M*H*kv_len_total);
    }

    /* Output projection */
    matmul_nt(attn_out, a->W_o, out, M, D, H*hd);
    add_bias(out, a->b_o, M, D);
    pn_free(attn_w); pn_free(attn_out);
}

/* Gradient state for MHA */
typedef struct {
    float *dW_q, *dW_k, *dW_v, *dW_o;
    float *db_q, *db_k, *db_v, *db_o;
} MHAGrad;

static inline MHAGrad mha_grad_new(MHAttention* a) {
    MHAGrad g;
    int D=a->d_model, H=a->n_heads, Hkv=a->n_kv_heads, hd=a->head_dim;
    g.dW_q=pn_alloc((size_t)H*hd*D);   g.db_q=pn_alloc(H*hd);
    g.dW_k=pn_alloc((size_t)Hkv*hd*D); g.db_k=pn_alloc(Hkv*hd);
    g.dW_v=pn_alloc((size_t)Hkv*hd*D); g.db_v=pn_alloc(Hkv*hd);
    g.dW_o=pn_alloc((size_t)D*H*hd);   g.db_o=pn_alloc(D);
    return g;
}

static inline void mha_grad_free(MHAGrad* g) {
    pn_free(g->dW_q); pn_free(g->dW_k); pn_free(g->dW_v); pn_free(g->dW_o);
    pn_free(g->db_q); pn_free(g->db_k); pn_free(g->db_v); pn_free(g->db_o);
}

static inline void mha_adamw_step(MHAttention* a, MHAGrad* g, AdamW* opt,
    AdamState* sq, AdamState* sk, AdamState* sv, AdamState* so) {
    int D=a->d_model, H=a->n_heads, Hkv=a->n_kv_heads, hd=a->head_dim;
    adamw_step(opt,sq,a->W_q,g->dW_q,(size_t)H*hd*D);
    adamw_step(opt,sq,a->b_q,g->db_q,H*hd);
    adamw_step(opt,sk,a->W_k,g->dW_k,(size_t)Hkv*hd*D);
    adamw_step(opt,sk,a->b_k,g->db_k,Hkv*hd);
    adamw_step(opt,sv,a->W_v,g->dW_v,(size_t)Hkv*hd*D);
    adamw_step(opt,sv,a->b_v,g->db_v,Hkv*hd);
    adamw_step(opt,so,a->W_o,g->dW_o,(size_t)D*H*hd);
    adamw_step(opt,so,a->b_o,g->db_o,D);
}

/*
 * Backward pass for multi-head attention.
 * dout: [M, D] gradient from next layer
 * dx:   [M, D] gradient flowing back to previous layer (output)
 *
 * Computes gradients for all weight matrices and propagates to input.
 * Only supports the training (non-KV-cache) path: M == kv_len_total.
 */
static inline void mha_backward(MHAttention* a, MHAGrad* g,
                                 const float* dout, float* dx, int M) {
    int D=a->d_model, H=a->n_heads, Hkv=a->n_kv_heads, hd=a->head_dim;
    pn_zero(g->dW_q,(size_t)H*hd*D); pn_zero(g->db_q,H*hd);
    pn_zero(g->dW_k,(size_t)Hkv*hd*D); pn_zero(g->db_k,Hkv*hd);
    pn_zero(g->dW_v,(size_t)Hkv*hd*D); pn_zero(g->db_v,Hkv*hd);
    pn_zero(g->dW_o,(size_t)D*H*hd); pn_zero(g->db_o,D);
    if (dx) pn_zero(dx,(size_t)M*D);

    float* d_attn_out = pn_alloc((size_t)M*H*hd);
    pn_zero(d_attn_out, (size_t)M*H*hd);

    /* Gradient through output projection W_o: d_attn_out = dout @ W_o^T */
    for (int m=0; m<M; m++) {
        const float* dr = dout + (size_t)m*D;
        for (int h=0; h<H; h++) {
            for (int d=0; d<hd; d++) {
                float s=0;
                for (int o=0; o<D; o++)
                    s += dr[o] * a->W_o[(size_t)h*hd*D + d*D + o];
                d_attn_out[m*H*hd + h*hd + d] = s;
            }
        }
    }

    /* Accumulate dW_o and db_o */
    for (int m=0; m<M; m++) {
        for (int o=0; o<D; o++) g->db_o[o] += dout[m*D+o];
        for (int h=0; h<H; h++) {
            for (int d=0; d<hd; d++) {
                float da = d_attn_out[(size_t)m*H*hd + (size_t)h*hd + d];
                for (int o=0; o<D; o++)
                    g->dW_o[(size_t)h*hd*D + (size_t)d*D + o] += da * dout[m*D+o];
            }
        }
    }

    /* Backprop through attention weights and values */
    float* d_attn_w = pn_alloc((size_t)M*H*M);
    float* d_q = pn_alloc((size_t)M*H*hd);
    float* d_k = pn_alloc((size_t)M*Hkv*hd);
    float* d_v = pn_alloc((size_t)M*Hkv*hd);
    pn_zero(d_q, (size_t)M*H*hd);
    pn_zero(d_k, (size_t)M*Hkv*hd);
    pn_zero(d_v, (size_t)M*Hkv*hd);

    float scale = 1.0f / sqrtf((float)hd);
    int S = M; /* sequence length == kv_len_total in training */

    for (int h=0; h<H; h++) {
        int hkv = h % Hkv;
        for (int m=0; m<M; m++) {
            float* aw = a->attn_save + (size_t)m*H*S + (size_t)h*S;
            float* daw = d_attn_w + (size_t)m*H*S + (size_t)h*S;

            /* dot product of d_attn_out with value for this head */
            float dot = 0;
            for (int d2=0; d2<hd; d2++)
                dot += d_attn_out[(size_t)m*H*hd + (size_t)h*hd + d2] *
                       a->v_save[(size_t)m*Hkv*hd + (size_t)hkv*hd + d2];

            /* softmax backward: d_w = w * (d_out_v - dot) for each key position */
            for (int kp=0; kp<S; kp++) {
                float val_dot = 0;
                for (int d2=0; d2<hd; d2++)
                    val_dot += d_attn_out[(size_t)m*H*hd + (size_t)h*hd + d2] *
                               a->v_save[(size_t)kp*Hkv*hd + (size_t)hkv*hd + d2];
                daw[kp] = aw[kp] * (val_dot - dot);
            }

            /* Gradient w.r.t. value: d_v += attn_weight * d_attn_out */
            for (int kp=0; kp<S; kp++) {
                for (int d2=0; d2<hd; d2++)
                    d_v[(size_t)kp*Hkv*hd + (size_t)hkv*hd + d2] +=
                        aw[kp] * d_attn_out[(size_t)m*H*hd + (size_t)h*hd + d2];
            }

            /* Gradient w.r.t. key: d_k += scale * daw * q */
            for (int kp=0; kp<S; kp++) {
                for (int d2=0; d2<hd; d2++)
                    d_k[(size_t)kp*Hkv*hd + (size_t)hkv*hd + d2] +=
                        daw[kp] * a->q_save[(size_t)m*H*hd + (size_t)h*hd + d2] * scale;
            }

            /* Gradient w.r.t. query: d_q += scale * sum_k(daw * k) */
            for (int d2=0; d2<hd; d2++) {
                float s = 0;
                for (int kp=0; kp<S; kp++)
                    s += daw[kp] * a->k_save[(size_t)kp*Hkv*hd + (size_t)hkv*hd + d2];
                d_q[(size_t)m*H*hd + (size_t)h*hd + d2] += s * scale;
            }
        }
    }

    /* Accumulate dW_q, dW_k, dW_v, db_q, db_k, db_v */
    for (int m=0; m<M; m++) {
        const float* xm = a->x_save + (size_t)m*D;
        for (int i=0; i<D; i++) {
            for (int j=0; j<H*hd; j++)
                g->dW_q[j*D+i] += xm[i] * d_q[(size_t)m*H*hd+j];
            for (int j=0; j<Hkv*hd; j++) {
                g->dW_k[j*D+i] += xm[i] * d_k[(size_t)m*Hkv*hd+j];
                g->dW_v[j*D+i] += xm[i] * d_v[(size_t)m*Hkv*hd+j];
            }
        }
        for (int j=0; j<H*hd; j++)   g->db_q[j] += d_q[(size_t)m*H*hd+j];
        for (int j=0; j<Hkv*hd; j++) { g->db_k[j] += d_k[(size_t)m*Hkv*hd+j]; g->db_v[j] += d_v[(size_t)m*Hkv*hd+j]; }
    }

    /* dx = d_q @ W_q^T + d_k @ W_k^T + d_v @ W_v^T */
    if (dx) {
        for (int m=0; m<M; m++) {
            float* dxm = dx + (size_t)m*D;
            const float* dqm = d_q + (size_t)m*H*hd;
            const float* dkm = d_k + (size_t)m*Hkv*hd;
            const float* dvm = d_v + (size_t)m*Hkv*hd;
            for (int i=0; i<D; i++) {
                float s = 0;
                for (int j=0; j<H*hd; j++)    s += dqm[j] * a->W_q[(size_t)j*D+i];
                for (int j=0; j<Hkv*hd; j++) { s += dkm[j] * a->W_k[(size_t)j*D+i];
                                                s += dvm[j] * a->W_v[(size_t)j*D+i]; }
                dxm[i] = s;
            }
        }
    }

    pn_free(d_attn_out); pn_free(d_attn_w);
    pn_free(d_q); pn_free(d_k); pn_free(d_v);
}
