/*
 * ProbNet - generate.h
 * Text generation: greedy, top-k, top-p (nucleus), beam search, temperature.
 * Uses KV cache for O(1) per-token generation after prefill.
 */
#pragma once
#include "transformer.h"
#include "tokenizer.h"

typedef struct {
    /* Sampling strategy */
    float temperature;  /* 1.0 = neutral, <1 = sharper, >1 = more random */
    int   top_k;        /* 0 = disabled */
    float top_p;        /* 0.0 = disabled, 0.9 = typical nucleus */
    /* Beam search */
    int   n_beams;      /* 1 = greedy/sampling, >1 = beam */
    /* Generation limits */
    int   max_new_tokens;
    int   min_new_tokens;
    /* Special tokens */
    int   eos_token_id;
    /* Repetition penalty */
    float repetition_penalty;  /* 1.0 = none, 1.1 = light penalty */
    /* Seed */
    uint64_t rng_seed;
    /* Print as generated */
    int   stream;
} GenerateConfig;

static inline GenerateConfig generate_default(void) {
    GenerateConfig c;
    c.temperature       = 0.8f;
    c.top_k             = 50;
    c.top_p             = 0.95f;
    c.n_beams           = 1;
    c.max_new_tokens    = 200;
    c.min_new_tokens    = 1;
    c.eos_token_id      = -1;
    c.repetition_penalty= 1.1f;
    c.rng_seed          = 12345;
    c.stream            = 1;
    return c;
}

/* Apply repetition penalty to logits */
static inline void apply_rep_penalty(float* logits, int V,
                                      const int* past_ids, int n_past, float penalty) {
    if (penalty == 1.0f || n_past == 0) return;
    for (int i=0;i<n_past;i++) {
        int id=past_ids[i];
        if (id>=0&&id<V) {
            if (logits[id]>0) logits[id]/=penalty;
            else              logits[id]*=penalty;
        }
    }
}

/*
 * Generate text autoregressively.
 * Returns: allocated array of new token ids (caller must free).
 * n_new: filled with number of generated tokens.
 */
static inline int* generate(Transformer* model, Tokenizer* tok,
                              const int* prompt_ids, int prompt_len,
                              const GenerateConfig* cfg, int* n_new) {
    int V = model->config.vocab_size;
    int max_new = cfg->max_new_tokens;
    int* new_ids = (int*)malloc((size_t)(max_new+1) * sizeof(int));
    int* all_ids = (int*)malloc((size_t)(prompt_len+max_new+1) * sizeof(int));
    memcpy(all_ids, prompt_ids, prompt_len*sizeof(int));

    float* logits = pn_alloc(V);
    RNG rng = rng_seed(cfg->rng_seed);

    /* ── Prefill: process entire prompt at once ── */
    transformer_reset_cache(model);

    /* Process prompt */
    transformer_forward(model, prompt_ids, logits, prompt_len, 1, 0);
    /* After prefill, logits has predictions for ALL prompt positions.
       We only care about the last one for the next token. */
    float* last_logits = pn_alloc(V);
    /* Re-run just the last token to get its logits cleanly */
    /* (For efficiency in real systems, use the last row of logits) */
    pn_copy(last_logits, logits + (size_t)(prompt_len-1)*V, V);

    int n_generated = 0;
    int past_len = prompt_len;

    /* ── Autoregressive decode ── */
    while (n_generated < max_new) {
        /* Apply repetition penalty */
        apply_rep_penalty(last_logits, V, all_ids, past_len, cfg->repetition_penalty);

        /* Sample next token */
        int next_id;
        if (cfg->temperature <= 0.0f) {
            /* Greedy */
            next_id = 0;
            float best = last_logits[0];
            for (int i=1;i<V;i++) if(last_logits[i]>best){best=last_logits[i];next_id=i;}
        } else if (cfg->top_p > 0.0f && cfg->top_p < 1.0f) {
            float* tmp = pn_alloc(V);
            pn_copy(tmp, last_logits, V);
            next_id = sample_topp(tmp, V, cfg->top_p, cfg->temperature, &rng);
            pn_free(tmp);
        } else if (cfg->top_k > 0) {
            float* tmp = pn_alloc(V);
            pn_copy(tmp, last_logits, V);
            next_id = sample_topk(tmp, V, cfg->top_k, cfg->temperature, &rng);
            pn_free(tmp);
        } else {
            float* tmp = pn_alloc(V);
            pn_copy(tmp, last_logits, V);
            apply_temperature(tmp, V, cfg->temperature);
            float mx=-FLT_MAX;
            for(int i=0;i<V;i++) if(tmp[i]>mx) mx=tmp[i];
            float sum=0;
            for(int i=0;i<V;i++){tmp[i]=expf(tmp[i]-mx);sum+=tmp[i];}
            for(int i=0;i<V;i++) tmp[i]/=sum;
            next_id=sample_categorical(tmp,V,&rng);
            pn_free(tmp);
        }

        new_ids[n_generated++] = next_id;
        all_ids[past_len++] = next_id;

        /* Stream output */
        if (cfg->stream) {
            const char* token_str = tokenizer_id_to_token(tok, next_id);
            /* Handle SentencePiece ▁ → space */
            if ((unsigned char)token_str[0]==0xe2 &&
                (unsigned char)token_str[1]==0x96 &&
                (unsigned char)token_str[2]==0x81) {
                printf(" %s", token_str+3);
            } else {
                printf("%s", token_str);
            }
            fflush(stdout);
        }

        /* Check EOS */
        if (cfg->eos_token_id >= 0 && next_id == cfg->eos_token_id) break;
        if (next_id == tok->eos_id) break;

        /* Get next logits: single token forward with KV cache */
        transformer_forward(model, &next_id, last_logits, 1, 1, past_len-1);
    }

    if (cfg->stream) printf("\n");

    pn_free(logits); pn_free(last_logits); free(all_ids);
    *n_new = n_generated;
    return new_ids;
}

/*
 * High-level: encode prompt, generate, decode, return string.
 * Caller must free the returned string.
 */
static inline char* generate_text(Transformer* model, Tokenizer* tok,
                                   const char* prompt, const GenerateConfig* cfg) {
    int prompt_len;
    int* prompt_ids = tokenizer_encode(tok, prompt, tok->add_bos, 0, &prompt_len);

    printf("Prompt: \"%s\"\n", prompt);
    printf("Generating (max %d tokens)...\n", cfg->max_new_tokens);
    if (cfg->stream) printf("Output: ");

    int n_new;
    int* new_ids = generate(model, tok, prompt_ids, prompt_len, cfg, &n_new);

    char* text = tokenizer_decode(tok, new_ids, n_new);
    free(prompt_ids); free(new_ids);
    return text;
}

/*
 * Interactive chat loop (REPL).
 */
static inline void chat_loop(Transformer* model, Tokenizer* tok,
                               const GenerateConfig* cfg) {
    printf("\n╔═══════════════════════════════════════════╗\n");
    printf("║  ProbNet Interactive Chat                 ║\n");
    printf("║  Type your message. 'quit' to exit.       ║\n");
    printf("╚═══════════════════════════════════════════╝\n\n");

    char line[4096];
    char context[65536]; context[0]=0;

    while (1) {
        printf("You: ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        char* nl=strchr(line,'\n'); if(nl)*nl=0;
        if (strcmp(line,"quit")==0||strcmp(line,"exit")==0) break;
        if (strlen(line)==0) continue;

        /* Build prompt with context */
        char prompt[65536];
        snprintf(prompt,sizeof(prompt),"%s\nUser: %s\nAssistant:",context,line);

        GenerateConfig gcfg = *cfg;
        gcfg.stream = 1;
        printf("Assistant: ");
        fflush(stdout);

        char* response = generate_text(model, tok, prompt, &gcfg);

        /* Update context (sliding window) */
        char new_ctx[65536];
        snprintf(new_ctx,sizeof(new_ctx),"%s\nUser: %s\nAssistant: %s",context,line,response);
        /* Trim context to avoid exceeding max_seq_len */
        int ctx_len = strlen(new_ctx);
        int keep = ctx_len > 20000 ? ctx_len-20000 : 0;
        strncpy(context, new_ctx+keep, sizeof(context)-1);

        free(response);
        printf("\n");
    }
    printf("Goodbye!\n");
}
