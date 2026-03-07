/*
 * ProbNet - transformer.h
 * Full Transformer (GPT/LLaMA/Gemma style) with APN FFN layers.
 *
 * Architecture per layer:
 *   x = x + Attention(RMSNorm(x))
 *   x = x + APN_FFN(RMSNorm(x))   ← replaces SwiGLU/GELU FFN
 *
 * Compatible with LLaMA/Gemma weight layout for conversion.
 */
#pragma once
#include "tensor.h"
#include "optimizer.h"
#include "attention.h"
#include "apn_layer.h"

/* ── Model configuration ────────────────────────────────────────── */
typedef struct {
    int vocab_size;
    int d_model;
    int n_layers;
    int n_heads;
    int n_kv_heads;  /* = n_heads for MHA, < for GQA (LLaMA/Gemma) */
    int ffn_hidden;  /* FFN hidden dim (typically 4*d_model or 8/3*d_model) */
    int max_seq_len;
    int use_rope;
    float rms_eps;
    float apn_tau0;  /* APN tau start (3.0) */
    float apn_tau1;  /* APN tau end   (0.05) */
    char arch[32];   /* "probnet", "llama", "gemma" */
} TransformerConfig;

static inline TransformerConfig default_config(void) {
    TransformerConfig c;
    c.vocab_size  = 32000;
    c.d_model     = 512;
    c.n_layers    = 6;
    c.n_heads     = 8;
    c.n_kv_heads  = 8;
    c.ffn_hidden  = 2048;
    c.max_seq_len = 2048;
    c.use_rope    = 1;
    c.rms_eps     = 1e-5f;
    c.apn_tau0    = 3.0f;
    c.apn_tau1    = 0.05f;
    snprintf(c.arch, 32, "probnet");
    return c;
}

/* ── One Transformer Block ──────────────────────────────────────── */
typedef struct {
    MHAttention* attn;
    APNLayer*    ffn;
    float* attn_norm_w;  /* RMSNorm weight [d_model] */
    float* ffn_norm_w;
    float* attn_norm_buf;/* scratch [M, d_model] */
    float* ffn_norm_buf;
    float* attn_out_buf;
    float* ffn_out_buf;
    int M_cache;
} TransformerBlock;

static inline TransformerBlock* block_new(const TransformerConfig* c, int layer_idx) {
    TransformerBlock* b = (TransformerBlock*)calloc(1, sizeof(TransformerBlock));
    uint64_t seed = (uint64_t)(layer_idx+1) * 0x9e3779b97f4a7c15ULL;
    b->attn = mha_new(c->d_model, c->n_heads, c->n_kv_heads,
                      c->max_seq_len, c->use_rope, seed);
    b->ffn  = apn_layer_new(c->d_model, c->ffn_hidden, c->d_model,
                             c->apn_tau0, seed^0xdeadbeef);
    b->attn_norm_w = pn_alloc(c->d_model); init_ones(b->attn_norm_w, c->d_model);
    b->ffn_norm_w  = pn_alloc(c->d_model); init_ones(b->ffn_norm_w,  c->d_model);
    return b;
}

static inline void block_free(TransformerBlock* b) {
    if (!b) return;
    mha_free(b->attn); apn_layer_free(b->ffn);
    pn_free(b->attn_norm_w); pn_free(b->ffn_norm_w);
    pn_free(b->attn_norm_buf); pn_free(b->ffn_norm_buf);
    pn_free(b->attn_out_buf); pn_free(b->ffn_out_buf);
    free(b);
}

static inline void block_ensure(TransformerBlock* b, int M, int D) {
    if (b->M_cache == M) return;
    pn_free(b->attn_norm_buf); pn_free(b->ffn_norm_buf);
    pn_free(b->attn_out_buf);  pn_free(b->ffn_out_buf);
    b->attn_norm_buf = pn_alloc((size_t)M*D);
    b->ffn_norm_buf  = pn_alloc((size_t)M*D);
    b->attn_out_buf  = pn_alloc((size_t)M*D);
    b->ffn_out_buf   = pn_alloc((size_t)M*D);
    b->M_cache = M;
}

static inline void block_forward(TransformerBlock* b, float* x, int M, int D,
                                   float eps, int use_cache, int past_len) {
    block_ensure(b, M, D);
    /* Pre-norm attention */
    rmsnorm(x, b->attn_norm_w, b->attn_norm_buf, M, D, eps);
    mha_forward(b->attn, b->attn_norm_buf, b->attn_out_buf, M, use_cache, past_len);
    pn_add(x, b->attn_out_buf, (size_t)M*D);  /* residual */
    /* Pre-norm FFN */
    rmsnorm(x, b->ffn_norm_w, b->ffn_norm_buf, M, D, eps);
    apn_forward(b->ffn, b->ffn_norm_buf, b->ffn_out_buf, M);
    pn_add(x, b->ffn_out_buf, (size_t)M*D);   /* residual */
}

/* ── Full Transformer Model ─────────────────────────────────────── */
typedef struct {
    TransformerConfig config;
    float* token_emb;   /* [vocab_size, d_model] */
    TransformerBlock** blocks;
    float* final_norm_w;
    float* lm_head;     /* [vocab_size, d_model] */
    /* scratch */
    float* x_buf;
    int    M_cache;
} Transformer;

static inline Transformer* transformer_new(const TransformerConfig* c) {
    Transformer* t = (Transformer*)calloc(1, sizeof(Transformer));
    t->config = *c;
    RNG r = rng_seed(42);
    float s = sqrtf(1.0f/c->d_model);
    t->token_emb   = pn_alloc((size_t)c->vocab_size*c->d_model);
    init_normal(t->token_emb, (size_t)c->vocab_size*c->d_model, s, &r);
    t->blocks = (TransformerBlock**)calloc(c->n_layers, sizeof(TransformerBlock*));
    for (int l=0;l<c->n_layers;l++) t->blocks[l] = block_new(c, l);
    t->final_norm_w = pn_alloc(c->d_model); init_ones(t->final_norm_w, c->d_model);
    t->lm_head = pn_alloc((size_t)c->vocab_size*c->d_model);
    /* tie weights: lm_head = token_emb */
    pn_copy(t->lm_head, t->token_emb, (size_t)c->vocab_size*c->d_model);
    return t;
}

static inline void transformer_free(Transformer* t) {
    if (!t) return;
    pn_free(t->token_emb); pn_free(t->final_norm_w); pn_free(t->lm_head);
    pn_free(t->x_buf);
    for (int l=0;l<t->config.n_layers;l++) block_free(t->blocks[l]);
    free(t->blocks); free(t);
}

static inline void transformer_ensure(Transformer* t, int M) {
    if (t->M_cache == M) return;
    pn_free(t->x_buf);
    t->x_buf   = pn_alloc((size_t)M*t->config.d_model);
    t->M_cache = M;
}

static inline void transformer_reset_cache(Transformer* t) {
    for (int l=0;l<t->config.n_layers;l++) mha_reset_cache(t->blocks[l]->attn);
}

/*
 * Forward pass.
 * tokens:    [M] integer token ids
 * logits:    [M, vocab_size] output
 * use_cache: 1 for generation (uses KV cache), 0 for training
 * past_len:  tokens already processed (for positional encoding)
 */
static inline void transformer_forward(Transformer* t,
                                        const int* tokens, float* logits,
                                        int M, int use_cache, int past_len) {
    int D=t->config.d_model, V=t->config.vocab_size;
    transformer_ensure(t, M);

    /* Token embedding */
    for (int m=0;m<M;m++)
        pn_copy(t->x_buf+(size_t)m*D, t->token_emb+(size_t)tokens[m]*D, D);

    /* Transformer blocks */
    for (int l=0;l<t->config.n_layers;l++)
        block_forward(t->blocks[l], t->x_buf, M, D,
                      t->config.rms_eps, use_cache, past_len);

    /* Final norm */
    float* norm_buf = pn_alloc((size_t)M*D);
    rmsnorm(t->x_buf, t->final_norm_w, norm_buf, M, D, t->config.rms_eps);

    /* LM head: logits[M,V] = norm_buf[M,D] @ lm_head^T[V,D] */
    matmul_nt(norm_buf, t->lm_head, logits, M, V, D);
    pn_free(norm_buf);
}

/* APN tau annealing (call during training) */
static inline void transformer_anneal_apn(Transformer* t, float progress) {
    float tau0=t->config.apn_tau0, tau1=t->config.apn_tau1;
    for (int l=0;l<t->config.n_layers;l++)
        apn_anneal(t->blocks[l]->ffn, progress, tau0, tau1);
}

/* Print model stats */
static inline void transformer_print_info(const Transformer* t) {
    const TransformerConfig* c = &t->config;
    size_t emb  = (size_t)c->vocab_size*c->d_model;
    size_t attn = (size_t)c->n_layers*(c->n_heads+2*c->n_kv_heads)*(size_t)c->d_model/c->n_heads*c->d_model;
    size_t ffn  = (size_t)c->n_layers*apn_param_count(t->blocks[0]->ffn);
    size_t norm = (size_t)c->n_layers*2*c->d_model + c->d_model;
    size_t total= emb + attn + ffn + norm;
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  ProbNet Transformer                             ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  arch       : %-34s ║\n", c->arch);
    printf("║  vocab      : %-34d ║\n", c->vocab_size);
    printf("║  d_model    : %-34d ║\n", c->d_model);
    printf("║  n_layers   : %-34d ║\n", c->n_layers);
    printf("║  n_heads    : %-34d ║\n", c->n_heads);
    printf("║  n_kv_heads : %-34d ║\n", c->n_kv_heads);
    printf("║  ffn_hidden : %-34d ║\n", c->ffn_hidden);
    printf("║  max_seq    : %-34d ║\n", c->max_seq_len);
    printf("║  params     : ~%-2.1fM                             ║\n", total/1e6f);
    printf("║  FFN type   : APN v9 (6 learnable functions)    ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
}

/* ── Weight I/O ─────────────────────────────────────────────────── */
#define PROBNET_MAGIC 0x504E4554  /* "PNET" */
#define PROBNET_VERSION 2

static inline int transformer_save(const Transformer* t, const char* path) {
    FILE* fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr,"Cannot open %s for writing\n",path); return -1; }
    const TransformerConfig* c = &t->config;
    /* Header */
    uint32_t magic=PROBNET_MAGIC, ver=PROBNET_VERSION;
    fwrite(&magic, 4, 1, fp);
    fwrite(&ver,   4, 1, fp);
    fwrite(c, sizeof(TransformerConfig), 1, fp);
    /* Embeddings */
    fwrite(t->token_emb, sizeof(float), (size_t)c->vocab_size*c->d_model, fp);
    /* Blocks */
    for (int l=0;l<c->n_layers;l++) {
        TransformerBlock* b = t->blocks[l];
        MHAttention* a = b->attn;
        int D=c->d_model, H=c->n_heads, Hkv=c->n_kv_heads, hd=a->head_dim;
        fwrite(a->W_q, sizeof(float), (size_t)H*hd*D, fp);
        fwrite(a->b_q, sizeof(float), H*hd, fp);
        fwrite(a->W_k, sizeof(float), (size_t)Hkv*hd*D, fp);
        fwrite(a->b_k, sizeof(float), Hkv*hd, fp);
        fwrite(a->W_v, sizeof(float), (size_t)Hkv*hd*D, fp);
        fwrite(a->b_v, sizeof(float), Hkv*hd, fp);
        fwrite(a->W_o, sizeof(float), (size_t)D*H*hd, fp);
        fwrite(a->b_o, sizeof(float), D, fp);
        fwrite(b->attn_norm_w, sizeof(float), D, fp);
        apn_write_weights(b->ffn, fp);
        fwrite(b->ffn_norm_w, sizeof(float), D, fp);
    }
    fwrite(t->final_norm_w, sizeof(float), c->d_model, fp);
    fwrite(t->lm_head, sizeof(float), (size_t)c->vocab_size*c->d_model, fp);
    fclose(fp);
    printf("Saved model to %s\n", path);
    return 0;
}

static inline Transformer* transformer_load(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr,"Cannot open %s\n",path); return NULL; }
    uint32_t magic, ver;
    fread(&magic, 4, 1, fp);
    fread(&ver,   4, 1, fp);
    if (magic != PROBNET_MAGIC) {
        fprintf(stderr,"Not a ProbNet file (magic=%08X)\n",magic); fclose(fp); return NULL;
    }
    TransformerConfig c;
    fread(&c, sizeof(TransformerConfig), 1, fp);
    Transformer* t = transformer_new(&c);
    /* Load embeddings */
    fread(t->token_emb, sizeof(float), (size_t)c.vocab_size*c.d_model, fp);
    /* Load blocks */
    for (int l=0;l<c.n_layers;l++) {
        TransformerBlock* b = t->blocks[l];
        MHAttention* a = b->attn;
        int D=c.d_model, H=c.n_heads, Hkv=c.n_kv_heads, hd=a->head_dim;
        fread(a->W_q, sizeof(float), (size_t)H*hd*D, fp);
        fread(a->b_q, sizeof(float), H*hd, fp);
        fread(a->W_k, sizeof(float), (size_t)Hkv*hd*D, fp);
        fread(a->b_k, sizeof(float), Hkv*hd, fp);
        fread(a->W_v, sizeof(float), (size_t)Hkv*hd*D, fp);
        fread(a->b_v, sizeof(float), Hkv*hd, fp);
        fread(a->W_o, sizeof(float), (size_t)D*H*hd, fp);
        fread(a->b_o, sizeof(float), D, fp);
        fread(b->attn_norm_w, sizeof(float), D, fp);
        apn_read_weights(b->ffn, fp);
        fread(b->ffn_norm_w, sizeof(float), D, fp);
    }
    fread(t->final_norm_w, sizeof(float), c.d_model, fp);
    fread(t->lm_head, sizeof(float), (size_t)c.vocab_size*c.d_model, fp);
    fclose(fp);
    printf("Loaded model from %s\n", path);
    return t;
}
