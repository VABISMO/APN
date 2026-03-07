/*
 * ProbNet - optimizer.h
 * AdamW optimizer with gradient clipping and parameter groups
 */
#pragma once
#include "tensor.h"

typedef struct {
    float lr, beta1, beta2, eps, wd, clip_norm;
    int step;
} AdamW;

typedef struct {
    float* m;   /* first moment */
    float* v;   /* second moment */
    size_t n;
} AdamState;

static inline AdamW adamw_new(float lr, float wd, float clip_norm) {
    AdamW opt = {lr, 0.9f, 0.999f, 1e-8f, wd, clip_norm, 0};
    return opt;
}

static inline AdamState adam_state_new(size_t n) {
    AdamState s;
    s.m = pn_alloc(n); s.v = pn_alloc(n); s.n = n;
    return s;
}

static inline void adam_state_free(AdamState* s) {
    pn_free(s->m); pn_free(s->v); s->n = 0;
}

static inline void adamw_step(AdamW* opt, AdamState* s,
                               float* param, const float* grad, size_t n) {
    opt->step++;
    float b1=opt->beta1, b2=opt->beta2, eps=opt->eps;
    float bc1=1.0f-powf(b1,(float)opt->step);
    float bc2=1.0f-powf(b2,(float)opt->step);
    float lr_eff=opt->lr*sqrtf(bc2)/bc1;

    /* optional gradient clipping */
    float gnorm=0;
    if (opt->clip_norm > 0) {
        for (size_t i=0;i<n;i++) gnorm+=grad[i]*grad[i];
        gnorm=sqrtf(gnorm);
    }
    float gscale=1.0f;
    if (opt->clip_norm > 0 && gnorm > opt->clip_norm)
        gscale = opt->clip_norm / gnorm;

    for (size_t i=0;i<n;i++) {
        float g = grad[i] * gscale;
        s->m[i] = b1*s->m[i] + (1.0f-b1)*g;
        s->v[i] = b2*s->v[i] + (1.0f-b2)*g*g;
        float update = lr_eff * s->m[i] / (sqrtf(s->v[i])+eps);
        param[i] = param[i]*(1.0f-opt->lr*opt->wd) - update;
    }
}

/* ── Learning rate schedules ────────────────────────────────────── */
static inline float cosine_lr(float lr_max, float lr_min,
                               int step, int warmup, int total) {
    if (step < warmup)
        return lr_max * (float)step / (float)warmup;
    float progress = (float)(step-warmup) / (float)(total-warmup);
    return lr_min + 0.5f*(lr_max-lr_min)*(1.0f+cosf(3.14159265f*progress));
}

/* ── Gradient norm utility ──────────────────────────────────────── */
static inline float grad_norm(float** grads, size_t* sizes, int n_groups) {
    double s=0;
    for (int g=0;g<n_groups;g++)
        for (size_t i=0;i<sizes[g];i++) s+=grads[g][i]*grads[g][i];
    return (float)sqrt(s);
}
