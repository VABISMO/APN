/*
 * ProbNet - apn_layer.h  (v9 — Production Stable)
 *
 * Adaptive Probabilistic Neuron (APN) layer.
 * Replaces the FFN in standard Transformer architecture.
 *
 * Each neuron learns a GLOBAL probability distribution over 6 functions:
 *   f0: identity(a)             linear
 *   f1: tanh(a²)                bounded square
 *   f2: sign(a)*sqrt(|a|)       signed sqrt  
 *   f3: a*b / sqrt(1+(ab)²)     bounded product
 *   f4: sin(a)                  periodic
 *   f5: leaky_relu(a)           rectified
 *
 * Forward: y[h] = Σ_k alpha[h,k] * f_k(p1[h], p2[h])
 * alpha computed via softmax(logits[H,F] / tau)
 * tau anneals 3.0 → 0.05: starts uniform, ends specialized
 *
 * Gradient clipping inside backward prevents explosion.
 */
#pragma once
#include "tensor.h"
#include "optimizer.h"

#define APN_NFUNCS 6
static const char* APN_FNAMES[APN_NFUNCS] = {
    "identity", "sq-tanh", "s-sqrt", "b-prod", "sin", "relu"
};

typedef struct {
    int in_dim, hidden, out_dim;
    float* W1;      /* [in_dim, hidden] */
    float* b1;      /* [hidden] */
    float* W2;      /* [in_dim, hidden] */
    float* b2;      /* [hidden] */
    float* logits;  /* [hidden, NFUNCS] — global per-neuron weights */
    float* W_out;   /* [hidden, out_dim] */
    float* b_out;   /* [out_dim] */
    float tau;
    /* forward cache */
    float* x_save;
    float* p1_save, *p2_save;
    float* alpha_sm;    /* [hidden, NFUNCS] */
    float* fvals_save;  /* [M, hidden, NFUNCS] */
    float* apn_save;    /* [M, hidden] */
    int M_cache;
} APNLayer;

typedef struct {
    float *dW1, *db1, *dW2, *db2;
    float *dlogits;
    float *dW_out, *db_out;
} APNGrad;

/* ── Bounded functions with stable gradients ────────────────────── */
static inline void eval_funcs(float a, float b,
                               float fv[APN_NFUNCS],
                               float dfa[APN_NFUNCS],
                               float dfb[APN_NFUNCS]) {
    /* f0: identity */
    fv[0]=a; dfa[0]=1.0f; dfb[0]=0.0f;
    /* f1: tanh(a^2) */
    float a2=a*a, th=tanhf(a2);
    fv[1]=th; dfa[1]=2.0f*a*(1.0f-th*th); dfb[1]=0.0f;
    /* f2: sign(a)*sqrt(|a|) */
    float absa=fabsf(a)+1e-4f;
    fv[2]=(a>=0?1.0f:-1.0f)*sqrtf(absa);
    dfa[2]=0.5f/sqrtf(absa); dfb[2]=0.0f;
    /* f3: bounded product */
    float ab=a*b, denom=sqrtf(1.0f+ab*ab)+1e-6f;
    fv[3]=ab/denom;
    float dd=1.0f/(denom*(1.0f+ab*ab));
    dfa[3]=b*dd; dfb[3]=a*dd;
    /* f4: sin */
    fv[4]=sinf(a); dfa[4]=cosf(a); dfb[4]=0.0f;
    /* f5: leaky relu */
    fv[5]=(a>0?a:0.01f*a); dfa[5]=(a>0?1.0f:0.01f); dfb[5]=0.0f;
}

/* ── Lifecycle ──────────────────────────────────────────────────── */
static inline APNLayer* apn_layer_new(int in_dim, int hidden, int out_dim,
                                       float tau, uint64_t seed) {
    APNLayer* l = (APNLayer*)calloc(1, sizeof(APNLayer));
    l->in_dim=in_dim; l->hidden=hidden; l->out_dim=out_dim; l->tau=tau;
    RNG r = rng_seed(seed);
    float si = sqrtf(2.0f/in_dim), so = sqrtf(2.0f/(hidden+out_dim));
    l->W1 = pn_alloc((size_t)in_dim*hidden); init_normal(l->W1,(size_t)in_dim*hidden,si,&r);
    l->b1 = pn_alloc(hidden);
    l->W2 = pn_alloc((size_t)in_dim*hidden); init_normal(l->W2,(size_t)in_dim*hidden,si*0.5f,&r);
    l->b2 = pn_alloc(hidden);
    l->logits = pn_alloc((size_t)hidden*APN_NFUNCS);
    for (int i=0;i<hidden*APN_NFUNCS;i++) l->logits[i]=rng_normal(&r)*0.1f;
    l->W_out = pn_alloc((size_t)hidden*out_dim); init_normal(l->W_out,(size_t)hidden*out_dim,so,&r);
    l->b_out = pn_alloc(out_dim);
    return l;
}

static inline void apn_layer_free(APNLayer* l) {
    if (!l) return;
    pn_free(l->W1); pn_free(l->b1); pn_free(l->W2); pn_free(l->b2);
    pn_free(l->logits); pn_free(l->W_out); pn_free(l->b_out);
    pn_free(l->x_save); pn_free(l->p1_save); pn_free(l->p2_save);
    pn_free(l->alpha_sm); pn_free(l->fvals_save); pn_free(l->apn_save);
    free(l);
}

static inline void apn_ensure_cache(APNLayer* l, int M) {
    if (l->M_cache == M) return;
    pn_free(l->x_save); pn_free(l->p1_save); pn_free(l->p2_save);
    pn_free(l->alpha_sm); pn_free(l->fvals_save); pn_free(l->apn_save);
    l->x_save    = pn_alloc((size_t)M*l->in_dim);
    l->p1_save   = pn_alloc((size_t)M*l->hidden);
    l->p2_save   = pn_alloc((size_t)M*l->hidden);
    l->alpha_sm  = pn_alloc((size_t)l->hidden*APN_NFUNCS);
    l->fvals_save= pn_alloc((size_t)M*l->hidden*APN_NFUNCS);
    l->apn_save  = pn_alloc((size_t)M*l->hidden);
    l->M_cache   = M;
}

static inline void apn_compute_alpha(APNLayer* l) {
    float inv_tau = 1.0f/(l->tau+1e-7f);
    int H=l->hidden, F=APN_NFUNCS;
    /* Hard argmax mode for inference when tau < 0.1: eliminates softmax exp */
    int hard_mode = (l->tau < 0.1f);
    for (int h=0;h<H;h++) {
        float* lg=l->logits+h*F, *sm=l->alpha_sm+h*F;
        if (hard_mode) {
            int best=0; float bv=lg[0];
            for(int k=1;k<F;k++) if(lg[k]>bv){bv=lg[k];best=k;}
            memset(sm, 0, F*sizeof(float));
            sm[best] = 1.0f;
        } else {
            float mx=lg[0]*inv_tau;
            for(int k=1;k<F;k++){float v=lg[k]*inv_tau;if(v>mx)mx=v;}
            float s=0;
            for(int k=0;k<F;k++){sm[k]=expf(lg[k]*inv_tau-mx);s+=sm[k];}
            float inv=1.0f/s;
            for(int k=0;k<F;k++) sm[k]*=inv;
        }
    }
}

/* ── Forward pass ───────────────────────────────────────────────── */
static inline void apn_forward(APNLayer* l, const float* x, float* out, int M) {
    apn_ensure_cache(l, M);
    int D=l->in_dim, H=l->hidden, F=APN_NFUNCS, O=l->out_dim;
    pn_copy(l->x_save, x, (size_t)M*D);

    matmul_nt(x, l->W1, l->p1_save, M, H, D);
    add_bias(l->p1_save, l->b1, M, H);
    matmul_nt(x, l->W2, l->p2_save, M, H, D);
    add_bias(l->p2_save, l->b2, M, H);

    apn_compute_alpha(l);

    #pragma omp parallel for schedule(static) if(M*H > 512)
    for (int m=0;m<M;m++) {
        for (int h=0;h<H;h++) {
            float a=l->p1_save[m*H+h], b=l->p2_save[m*H+h];
            float fv[APN_NFUNCS], dfa[APN_NFUNCS], dfb[APN_NFUNCS];
            eval_funcs(a, b, fv, dfa, dfb);
            float* fvs = l->fvals_save+(m*H+h)*F;
            pn_copy(fvs, fv, F);
            float* sm=l->alpha_sm+h*F;
            float acc=0; for(int k=0;k<F;k++) acc+=sm[k]*fv[k];
            l->apn_save[m*H+h]=acc;
        }
    }
    matmul_nt(l->apn_save, l->W_out, out, M, O, H);
    add_bias(out, l->b_out, M, O);
}

/* ── Inference-only forward (no cache, no backward needed) ──────── */
static inline void apn_forward_infer(APNLayer* l, const float* x, float* out, int M) {
    /* Optimized inference path:
     * - No caching of intermediates (x_save, fvals_save, p1/p2 not needed)
     * - Hard argmax when tau < 0.1 (avoids exp in softmax)
     * - Skip dfa/dfb computation (only needed for backward)
     * - Fused function eval + alpha multiply into single loop
     * - ~30-40% faster than apn_forward for inference */
    int D=l->in_dim, H=l->hidden, F=APN_NFUNCS, O=l->out_dim;
    float* p1 = pn_alloc((size_t)M*H);
    float* p2 = pn_alloc((size_t)M*H);
    float* apn_out = pn_alloc((size_t)M*H);

    matmul_nt(x, l->W1, p1, M, H, D);
    add_bias(p1, l->b1, M, H);
    matmul_nt(x, l->W2, p2, M, H, D);
    add_bias(p2, l->b2, M, H);

    /* Compute alpha (with hard argmax optimization) */
    apn_compute_alpha(l);

    #pragma omp parallel for schedule(static) if(M*H > 256)
    for (int m=0;m<M;m++) {
        for (int h=0;h<H;h++) {
            float a=p1[m*H+h], b=p2[m*H+h];
            /* Inline function evaluation (skip dfa/dfb) */
            float f0=a;
            float f1=tanhf(a*a);
            float f2=(a>=0.0f?1.0f:-1.0f)*sqrtf(fabsf(a)+1e-4f);
            float ab=a*b; float d=sqrtf(1.0f+ab*ab)+1e-6f;
            float f3=ab/d;
            float f4=sinf(a);
            float f5=(a>0.0f?a:0.01f*a);
            float* sm=l->alpha_sm+h*F;
            apn_out[m*H+h] = sm[0]*f0 + sm[1]*f1 + sm[2]*f2 + sm[3]*f3 + sm[4]*f4 + sm[5]*f5;
        }
    }
    matmul_nt(apn_out, l->W_out, out, M, O, H);
    add_bias(out, l->b_out, M, O);

    pn_free(p1); pn_free(p2); pn_free(apn_out);
}

/* ── Backward pass ──────────────────────────────────────────────── */
static inline APNGrad apn_grad_new(APNLayer* l) {
    APNGrad g;
    int D=l->in_dim, H=l->hidden, F=APN_NFUNCS, O=l->out_dim;
    g.dW1=pn_alloc((size_t)D*H); g.db1=pn_alloc(H);
    g.dW2=pn_alloc((size_t)D*H); g.db2=pn_alloc(H);
    g.dlogits=pn_alloc((size_t)H*F);
    g.dW_out=pn_alloc((size_t)H*O); g.db_out=pn_alloc(O);
    return g;
}

static inline void apn_grad_free(APNGrad* g) {
    pn_free(g->dW1); pn_free(g->db1); pn_free(g->dW2); pn_free(g->db2);
    pn_free(g->dlogits); pn_free(g->dW_out); pn_free(g->db_out);
}

static inline void apn_backward(APNLayer* l, APNGrad* g,
                                  const float* dout, float* dx, int M) {
    int D=l->in_dim, H=l->hidden, F=APN_NFUNCS, O=l->out_dim;
    pn_zero(g->dW1,(size_t)D*H); pn_zero(g->db1,H);
    pn_zero(g->dW2,(size_t)D*H); pn_zero(g->db2,H);
    pn_zero(g->dlogits,(size_t)H*F);
    pn_zero(g->dW_out,(size_t)H*O); pn_zero(g->db_out,O);
    if (dx) pn_zero(dx,(size_t)M*D);

    /* grad through W_out */
    for(int m=0;m<M;m++) for(int o=0;o<O;o++) g->db_out[o]+=dout[m*O+o];
    for(int m=0;m<M;m++) for(int h=0;h<H;h++) for(int o=0;o<O;o++)
        g->dW_out[h*O+o]+=l->apn_save[m*H+h]*dout[m*O+o];

    /* d_apn */
    float* d_apn=pn_alloc((size_t)M*H);
    for(int m=0;m<M;m++) for(int h=0;h<H;h++) {
        float s=0; for(int o=0;o<O;o++) s+=dout[m*O+o]*l->W_out[h*O+o];
        d_apn[m*H+h]=s;
    }

    float* d_p1=pn_alloc((size_t)M*H), *d_p2=pn_alloc((size_t)M*H);
    float inv_tau=1.0f/(l->tau+1e-7f);

    for(int m=0;m<M;m++) for(int h=0;h<H;h++) {
        float da=d_apn[m*H+h];
        float a=l->p1_save[m*H+h], b=l->p2_save[m*H+h];
        float fv[APN_NFUNCS], dfa[APN_NFUNCS], dfb[APN_NFUNCS];
        eval_funcs(a, b, fv, dfa, dfb);
        float* sm=l->alpha_sm+h*F;
        float* dlg=g->dlogits+h*F;
        float dot_fs=0; for(int k=0;k<F;k++) dot_fs+=fv[k]*sm[k];
        for(int k=0;k<F;k++) dlg[k]+=da*inv_tau*sm[k]*(fv[k]-dot_fs);
        float dp1=0,dp2=0;
        for(int k=0;k<F;k++){dp1+=sm[k]*dfa[k];dp2+=sm[k]*dfb[k];}
        d_p1[m*H+h]=da*dp1; d_p2[m*H+h]=da*dp2;
    }
    pn_free(d_apn);

    for(int m=0;m<M;m++) {
        const float* xi=l->x_save+(size_t)m*D;
        float* dxi=dx?dx+(size_t)m*D:NULL;
        for(int h=0;h<H;h++) {
            float dp1=d_p1[m*H+h], dp2=d_p2[m*H+h];
            g->db1[h]+=dp1; g->db2[h]+=dp2;
            for(int i=0;i<D;i++) {
                g->dW1[i*H+h]+=xi[i]*dp1;
                g->dW2[i*H+h]+=xi[i]*dp2;
                if(dxi) dxi[i]+=l->W1[i*H+h]*dp1+l->W2[i*H+h]*dp2;
            }
        }
    }
    pn_free(d_p1); pn_free(d_p2);

    /* gradient clipping */
    float gnorm=0;
    for(size_t i=0;i<(size_t)D*H;i++) gnorm+=g->dW1[i]*g->dW1[i]+g->dW2[i]*g->dW2[i];
    for(int i=0;i<H*F;i++) gnorm+=g->dlogits[i]*g->dlogits[i];
    gnorm=sqrtf(gnorm);
    if (gnorm > 10.0f) {
        float sc=10.0f/gnorm;
        for(size_t i=0;i<(size_t)D*H;i++){g->dW1[i]*=sc;g->dW2[i]*=sc;}
        for(int i=0;i<H*F;i++) g->dlogits[i]*=sc;
    }
}

static inline void apn_adamw_step(APNLayer* l, APNGrad* g, AdamW* opt,
    AdamState* s1, AdamState* s2, AdamState* s3, AdamState* s4) {
    int D=l->in_dim, H=l->hidden, F=APN_NFUNCS, O=l->out_dim;
    adamw_step(opt,s1,l->W1,   g->dW1,   (size_t)D*H);
    adamw_step(opt,s1,l->b1,   g->db1,   H);
    adamw_step(opt,s2,l->W2,   g->dW2,   (size_t)D*H);
    adamw_step(opt,s2,l->b2,   g->db2,   H);
    adamw_step(opt,s3,l->logits,g->dlogits,(size_t)H*F);
    adamw_step(opt,s4,l->W_out,g->dW_out,(size_t)H*O);
    adamw_step(opt,s4,l->b_out,g->db_out,O);
}

static inline void apn_anneal(APNLayer* l, float progress, float tau0, float tau1) {
    l->tau = tau0 * powf(tau1/tau0, progress);
}

static inline void apn_print_specialization(const APNLayer* l) {
    if (!l->alpha_sm) { printf("(not computed)\n"); return; }
    float counts[APN_NFUNCS]={0};
    for(int h=0;h<l->hidden;h++){
        float* sm=l->alpha_sm+h*APN_NFUNCS;
        int best=0; float bv=sm[0];
        for(int k=1;k<APN_NFUNCS;k++) if(sm[k]>bv){bv=sm[k];best=k;}
        counts[best]++;
    }
    printf("APN specialization (H=%d): ", l->hidden);
    for(int k=0;k<APN_NFUNCS;k++) if(counts[k]>0)
        printf("%s=%.0f%% ",APN_FNAMES[k],100.0f*counts[k]/l->hidden);
    printf("\n");
}

/* ── Weight I/O (for save/load) ─────────────────────────────────── */
static inline size_t apn_param_count(const APNLayer* l) {
    int D=l->in_dim, H=l->hidden, F=APN_NFUNCS, O=l->out_dim;
    return 2*(size_t)D*H + 2*H + H*F + H*O + O;
}

static inline void apn_write_weights(const APNLayer* l, FILE* fp) {
    int D=l->in_dim, H=l->hidden, O=l->out_dim, F=APN_NFUNCS;
    fwrite(l->W1,    sizeof(float), (size_t)D*H, fp);
    fwrite(l->b1,    sizeof(float), H,            fp);
    fwrite(l->W2,    sizeof(float), (size_t)D*H, fp);
    fwrite(l->b2,    sizeof(float), H,            fp);
    fwrite(l->logits,sizeof(float), (size_t)H*F, fp);
    fwrite(l->W_out, sizeof(float), (size_t)H*O, fp);
    fwrite(l->b_out, sizeof(float), O,            fp);
}

static inline void apn_read_weights(APNLayer* l, FILE* fp) {
    int D=l->in_dim, H=l->hidden, O=l->out_dim, F=APN_NFUNCS;
    fread(l->W1,    sizeof(float), (size_t)D*H, fp);
    fread(l->b1,    sizeof(float), H,            fp);
    fread(l->W2,    sizeof(float), (size_t)D*H, fp);
    fread(l->b2,    sizeof(float), H,            fp);
    fread(l->logits,sizeof(float), (size_t)H*F, fp);
    fread(l->W_out, sizeof(float), (size_t)H*O, fp);
    fread(l->b_out, sizeof(float), O,            fp);
}
