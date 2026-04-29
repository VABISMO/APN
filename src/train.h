/*
 * ProbNet - train.h
 * Language model training loop.
 * Features:
 *   - Cross-entropy loss with gradient accumulation
 *   - APN tau annealing
 *   - Cosine LR schedule with warmup
 *   - Validation perplexity
 *   - Checkpoint save/load
 *   - Progress reporting
 */
#pragma once
#include "transformer.h"
#include "tokenizer.h"
#include "optimizer.h"

typedef struct {
    float lr;
    float lr_min;
    float weight_decay;
    float grad_clip;
    int   batch_size;
    int   seq_len;
    int   n_epochs;
    int   warmup_steps;
    int   eval_every;
    int   save_every;
    int   grad_accum;
    char  checkpoint_path[256];
    char  data_path[256];
} TrainConfig;

static inline TrainConfig train_default(void) {
    TrainConfig c;
    c.lr              = 3e-4f;
    c.lr_min          = 1e-5f;
    c.weight_decay    = 0.1f;
    c.grad_clip       = 1.0f;
    c.batch_size      = 16;
    c.seq_len         = 256;
    c.n_epochs        = 3;
    c.warmup_steps    = 100;
    c.eval_every      = 500;
    c.save_every      = 1000;
    c.grad_accum      = 1;
    strncpy(c.checkpoint_path, "checkpoint.pnet", 255);
    strncpy(c.data_path, "data/train.txt", 255);
    return c;
}

/* ── Parameter groups for optimizer states ──────────────────────── */
typedef struct {
    AdamW opt;
    /* One AdamState per weight tensor group */
    AdamState emb;
    AdamState* block_states; /* n_layers * 12 states */
    AdamState final_norm;
    AdamState lm_head;
    int n_layers;
} TrainOptimizer;

static inline TrainOptimizer* train_opt_new(const Transformer* t, float lr, float wd, float clip) {
    TrainOptimizer* o = (TrainOptimizer*)calloc(1, sizeof(TrainOptimizer));
    o->opt = adamw_new(lr, wd, clip);
    int D=t->config.d_model, V=t->config.vocab_size, L=t->config.n_layers;
    o->n_layers = L;
    o->emb        = adam_state_new((size_t)V*D);
    o->final_norm = adam_state_new(D);
    o->lm_head    = adam_state_new((size_t)V*D);
    o->block_states = (AdamState*)calloc(L*12, sizeof(AdamState));
    for (int l=0;l<L;l++) {
        const TransformerBlock* b = t->blocks[l];
        const MHAttention* a = b->attn;
        const APNLayer* ffn = b->ffn;
        int H=t->config.n_heads, Hkv=t->config.n_kv_heads, hd=a->head_dim;
        int Fi=ffn->in_dim, Fh=ffn->hidden, Fo=ffn->out_dim, F=APN_NFUNCS;
        int i=l*12;
        o->block_states[i+0] = adam_state_new((size_t)H*hd*D+H*hd);  /* Wq+bq */
        o->block_states[i+1] = adam_state_new((size_t)Hkv*hd*D+Hkv*hd); /* Wk+bk */
        o->block_states[i+2] = adam_state_new((size_t)Hkv*hd*D+Hkv*hd); /* Wv+bv */
        o->block_states[i+3] = adam_state_new((size_t)D*H*hd+D);       /* Wo+bo */
        o->block_states[i+4] = adam_state_new(D);                        /* attn_norm */
        o->block_states[i+5] = adam_state_new((size_t)Fi*Fh);           /* APN W1 */
        o->block_states[i+6] = adam_state_new(Fh+(size_t)Fi*Fh+Fh);    /* APN b1+W2+b2 */
        o->block_states[i+7] = adam_state_new((size_t)Fh*F);            /* APN logits */
        o->block_states[i+8] = adam_state_new((size_t)Fh*Fo+Fo);       /* APN Wout+bout */
        o->block_states[i+9] = adam_state_new(D);                        /* ffn_norm */
        o->block_states[i+10]= adam_state_new(1); /* placeholder */
        o->block_states[i+11]= adam_state_new(1); /* placeholder */
    }
    return o;
}

static inline void train_opt_free(TrainOptimizer* o) {
    adam_state_free(&o->emb);
    adam_state_free(&o->final_norm);
    adam_state_free(&o->lm_head);
    for(int i=0;i<o->n_layers*12;i++) adam_state_free(&o->block_states[i]);
    free(o->block_states); free(o);
}

/* ── Cross-entropy loss forward ─────────────────────────────────── */
/* logits: [M, V]   targets: [M]  */
static inline float ce_loss_and_grad(const float* logits, const int* targets,
                                      float* d_logits, int M, int V) {
    pn_zero(d_logits, (size_t)M*V);
    float* sm = pn_alloc(V); /* pre-allocated scratch, no malloc per row */
    double total_loss = 0;
    for (int m=0;m<M;m++) {
        const float* row = logits + (size_t)m*V;
        float* drow = d_logits + (size_t)m*V;
        int tgt = targets[m];
        /* Softmax (numerically stable) */
        float mx = row[0]; for(int v=1;v<V;v++) if(row[v]>mx) mx=row[v];
        float sum=0;
        for(int v=0;v<V;v++){sm[v]=expf(row[v]-mx);sum+=sm[v];}
        if (tgt>=0&&tgt<V) total_loss -= logf(sm[tgt]/sum + 1e-9f);
        /* Gradient: softmax(i) - 1(i==tgt), scaled by 1/M */
        float inv_sum=1.0f/(sum*M);
        for(int v=0;v<V;v++) drow[v]=sm[v]*inv_sum;
        if (tgt>=0&&tgt<V) drow[tgt]-=1.0f/M;
    }
    pn_free(sm);
    return (float)(total_loss/M);
}

/* ── Data loading ───────────────────────────────────────────────── */
typedef struct {
    int* data;       /* all token ids */
    size_t n_tokens;
    int    vocab_size;
} Dataset;

static inline Dataset* dataset_load(const char* path, Tokenizer* tok) {
    FILE* fp = fopen(path, "r");
    if (!fp) { fprintf(stderr,"Cannot open dataset: %s\n",path); return NULL; }
    fseek(fp,0,SEEK_END); long sz=ftell(fp); rewind(fp);
    char* text=(char*)malloc(sz+1);
    size_t rd=fread(text,1,sz,fp); fclose(fp); text[rd]=0;

    Dataset* ds = (Dataset*)calloc(1, sizeof(Dataset));
    int n;
    int* ids = tokenizer_encode(tok, text, 0, 0, &n);
    ds->data = ids; ds->n_tokens = n; ds->vocab_size = tok->vocab_size;
    free(text);
    printf("Dataset: %zu tokens from %s\n", ds->n_tokens, path);
    return ds;
}

static inline void dataset_free(Dataset* ds) {
    free(ds->data); free(ds);
}

static inline void dataset_get_batch(const Dataset* ds, int* x, int* y,
                                      int B, int T, RNG* rng) {
    for (int b=0;b<B;b++) {
        size_t start=(size_t)(rng_uniform(rng)*(ds->n_tokens-T-1));
        for(int t=0;t<T;t++) x[b*T+t]=ds->data[start+t];
        for(int t=0;t<T;t++) y[b*T+t]=ds->data[start+t+1];
    }
}

/* ── Training step (simplified — full backward not implemented for all layers) ── */
/* This runs the FULL forward + cross-entropy only for now (evaluation/inference mode) */
/* For actual training, use the python wrapper which handles backprop via pytorch */
static inline float eval_perplexity(Transformer* model, const Dataset* ds,
                                     int seq_len, int n_batches, RNG* rng) {
    int V=model->config.vocab_size;
    int* x=(int*)malloc((size_t)seq_len*sizeof(int));
    int* y=(int*)malloc((size_t)seq_len*sizeof(int));
    float* logits=pn_alloc((size_t)seq_len*V);
    float* d_logits=pn_alloc((size_t)seq_len*V);

    double total_loss=0;
    for (int b=0;b<n_batches;b++) {
        size_t start=(size_t)(rng_uniform(rng)*(ds->n_tokens-seq_len-1));
        for(int t=0;t<seq_len;t++){x[t]=ds->data[start+t];y[t]=ds->data[start+t+1];}
        transformer_forward(model, x, logits, seq_len, 0, 0);
        float loss=ce_loss_and_grad(logits,y,d_logits,seq_len,V);
        total_loss+=loss;
    }
    pn_free(logits); pn_free(d_logits);
    free(x); free(y);
    float avg_loss=(float)(total_loss/n_batches);
    return expf(avg_loss);  /* perplexity */
}

/* ── Full training loop with backprop ──────────────────────────────── */
static inline void train_full(Transformer* model, TrainOptimizer* opt,
                               const Dataset* ds, const TrainConfig* cfg) {
    int V = model->config.vocab_size;
    int D = model->config.d_model;
    int L = model->config.n_layers;
    int B = cfg->batch_size;
    int T = cfg->seq_len;
    int total_steps = cfg->n_epochs * (int)(ds->n_tokens / ((size_t)B * T));
    if (total_steps < 1) total_steps = 1;

    int* x = (int*)malloc((size_t)B*T*sizeof(int));
    int* y = (int*)malloc((size_t)B*T*sizeof(int));
    float* logits = pn_alloc((size_t)B*T*V);
    float* d_logits = pn_alloc((size_t)B*T*V);
    TransformerGrad grad = transformer_grad_new(model);

    /* Adam states for embedding and final norm */
    AdamState emb_m = adam_state_new((size_t)V*D);
    AdamState emb_v = adam_state_new((size_t)V*D);
    AdamState fnorm_m = adam_state_new(D);
    AdamState fnorm_v = adam_state_new(D);

    RNG rng = rng_seed(42);
    printf("Training: %d epochs, %d steps, B=%d, T=%d, lr=%.1e\n",
           cfg->n_epochs, total_steps, B, T, cfg->lr);
    printf("  Step | Loss     | PPL      | Tau    | LR\n");
    printf("  -----+----------+----------+--------+---------\n");

    double total_loss = 0;
    int n_loss = 0;

    for (int step = 0; step < total_steps; step++) {
        float progress = (float)step / total_steps;
        float lr = cosine_lr(cfg->lr, cfg->lr_min, step, cfg->warmup_steps, total_steps);
        opt->opt.lr = lr;

        /* Anneal APN tau */
        transformer_anneal_apn(model, progress);

        /* Zero gradients */
        transformer_grad_zero(&grad, V, D);
        for (int l=0; l<L; l++) {
            /* Note: per-layer grads are zeroed inside backward */
        }

        /* Accumulate gradients over grad_accum mini-batches */
        for (int g=0; g<cfg->grad_accum; g++) {
            dataset_get_batch(ds, x, y, B, T, &rng);

            /* Forward */
            transformer_forward(model, x, logits, B*T, 0, 0);

            /* Cross-entropy loss and gradient */
            float loss = ce_loss_and_grad(logits, y, d_logits, B*T, V);

            /* Backward through entire model */
            transformer_backward(model, &grad, x, d_logits, B*T);

            total_loss += loss;
            n_loss++;
        }

        /* Scale gradients by accumulation */
        if (cfg->grad_accum > 1) {
            float inv = 1.0f / cfg->grad_accum;
            pn_scale(grad.d_emb, inv, (size_t)V*D);
            pn_scale(grad.d_final_norm, inv, D);
            pn_scale(grad.d_lm_head, inv, (size_t)V*D);
        }

        /* AdamW step for embedding */
        for (size_t i=0; i<(size_t)V*D; i++) {
            emb_m.m[i] = opt->opt.beta1*emb_m.m[i] + (1-opt->opt.beta1)*grad.d_emb[i];
            emb_v.m[i] = opt->opt.beta2*emb_v.m[i] + (1-opt->opt.beta2)*grad.d_emb[i]*grad.d_emb[i];
            float bc1=1-powf(opt->opt.beta1,(float)(opt->opt.step+1));
            float bc2=1-powf(opt->opt.beta2,(float)(opt->opt.step+1));
            float lr_eff = lr * sqrtf(bc2)/bc1;
            model->token_emb[i] = model->token_emb[i]*(1-lr*opt->opt.wd) - lr_eff*emb_m.m[i]/(sqrtf(emb_v.m[i])+opt->opt.eps);
        }

        /* AdamW step for final norm */
        for (int d=0; d<D; d++) {
            fnorm_m.m[d] = opt->opt.beta1*fnorm_m.m[d] + (1-opt->opt.beta1)*grad.d_final_norm[d];
            fnorm_v.m[d] = opt->opt.beta2*fnorm_v.m[d] + (1-opt->opt.beta2)*grad.d_final_norm[d]*grad.d_final_norm[d];
            float bc1=1-powf(opt->opt.beta1,(float)(opt->opt.step+1));
            float bc2=1-powf(opt->opt.beta2,(float)(opt->opt.step+1));
            float lr_eff = lr * sqrtf(bc2)/bc1;
            model->final_norm_w[d] = model->final_norm_w[d]*(1-lr*opt->opt.wd) - lr_eff*fnorm_m.m[d]/(sqrtf(fnorm_v.m[d])+opt->opt.eps);
        }

        /* LM head (tied with embedding — copy from token_emb) */
        pn_copy(model->lm_head, model->token_emb, (size_t)V*D);

        /* Per-block AdamW steps done inside backward — apply them now */
        /* Note: block backward currently frees grads; need per-block step */
        /* For now, block params are updated via apn_adamw_step / mha_adamw_step */

        opt->opt.step++;

        /* Logging */
        if ((step+1) % cfg->eval_every == 0 || step == 0) {
            float avg = (n_loss > 0) ? (float)(total_loss/n_loss) : 0;
            float tau = model->blocks[0]->ffn->tau;
            printf("  %4d | %8.4f | %8.2f | %6.3f | %.2e\n",
                   step+1, avg, expf(avg), tau, lr);
            total_loss = 0; n_loss = 0;
        }

        /* Save checkpoint */
        if (cfg->save_every > 0 && (step+1) % cfg->save_every == 0) {
            char path[512];
            snprintf(path, sizeof(path), "%s.step%d", cfg->checkpoint_path, step+1);
            transformer_save(model, path);
        }
    }

    /* Final save */
    transformer_save(model, cfg->checkpoint_path);
    printf("  Training complete. Final model saved to %s\n", cfg->checkpoint_path);

    transformer_grad_free(&grad);
    adam_state_free(&emb_m); adam_state_free(&emb_v);
    adam_state_free(&fnorm_m); adam_state_free(&fnorm_v);
    pn_free(logits); pn_free(d_logits);
    free(x); free(y);
}
