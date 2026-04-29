/*
 * ProbNet - etok_bridge.h
 * Bridge between ETOK BPE tokenizer and ProbNet Transformer.
 * Enables training and generation with proper BPE tokenization.
 *
 * Usage: #include "etok_bridge.h" after including transformer.h
 * Compile: gcc ... -I../ETOK/src etok_bridge.c -lm
 */
#pragma once
#include "tokenizer.h"
#include <sys/stat.h>

/* ── ETOK Tokenizer Bridge ────────────────────────────────────────── */
/*
 * Loads an ETOK-trained vocabulary and builds a ProbNet Tokenizer
 * that can be used with transformer_forward/generate.
 *
 * The ETOK vocab file format is one token per line: "token<TAB>score"
 * This is compatible with both ETOK's tok_save() and the simple
 * vocab format that tokenizer_load_vocab() already reads.
 *
 * For full ETOK functionality (DAFSA encoding, magic_split, entropy-stop),
 * use etok_train_and_save() first, then load the vocab.
 */

/* Train a BPE tokenizer on a corpus and save the vocabulary.
 * Returns the path to the saved vocab file, or NULL on error.
 * The caller must free the returned string. */
static inline char* etok_train_vocab(const char* corpus_path,
                                      const char* vocab_path,
                                      int target_vocab_size,
                                      int min_freq) {
    /* Read corpus */
    FILE* fp = fopen(corpus_path, "r");
    if (!fp) { fprintf(stderr, "Cannot open corpus: %s\n", corpus_path); return NULL; }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); rewind(fp);
    char* text = (char*)malloc(sz + 1);
    size_t rd = fread(text, 1, sz, fp); fclose(fp); text[rd] = 0;

    /* Simple BPE training in C (standalone, no dependency on etok.c) */
    /* Collect character-level vocabulary first */
    int max_tokens = target_vocab_size + 256;
    char** tokens = (char**)calloc(max_tokens, sizeof(char*));
    float* scores = (float*)calloc(max_tokens, sizeof(float));
    int* freqs = (int*)calloc(max_tokens, sizeof(int));
    int n_tokens = 0;

    /* Add byte-level tokens */
    for (int c = 0; c < 256 && n_tokens < max_tokens; c++) {
        char buf[2] = {(char)c, 0};
        tokens[n_tokens] = strdup(buf);
        scores[n_tokens] = 0;
        freqs[n_tokens] = 0;
        n_tokens++;
    }
    /* Add special tokens */
    const char* special[] = {"<pad>","<unk>","<bos>","<eos>","</w>"};
    for (int s = 0; s < 5 && n_tokens < max_tokens; s++) {
        tokens[n_tokens] = strdup(special[s]);
        scores[n_tokens] = -1e6f;
        freqs[n_tokens] = 0;
        n_tokens++;
    }

    /* Count bigrams and perform merges */
    /* Use a simple merge-based approach */
    int* ids = (int*)malloc(rd * 2 * sizeof(int));
    int n_ids = 0;

    /* Initialize with byte-level encoding */
    for (size_t i = 0; i < (size_t)rd; i++) {
        ids[n_ids++] = (unsigned char)text[i];
    }
    free(text);

    /* Perform BPE merges */
    int n_merges = target_vocab_size - n_tokens;
    if (n_merges < 0) n_merges = 0;
    if (n_merges > 50000) n_merges = 50000;

    for (int m = 0; m < n_merges && n_tokens < max_tokens; m++) {
        /* Find most frequent pair */
        int best_a = -1, best_b = -1, best_count = 0;
        /* Simple O(n^2) pair counting - fine for small vocabularies */
        int* pair_count = (int*)calloc(n_tokens * n_tokens, sizeof(int));
        int* pair_a = (int*)malloc(n_tokens * n_tokens * sizeof(int));
        int* pair_b = (int*)malloc(n_tokens * n_tokens * sizeof(int));
        int n_pairs = 0;

        for (int i = 0; i < n_ids - 1; i++) {
            if (ids[i] < 0 || ids[i+1] < 0) continue;
            int a = ids[i], b = ids[i+1];
            if (a < 0 || b < 0 || a >= n_tokens || b >= n_tokens) continue;
            /* Hash pair to index */
            int idx = a * n_tokens + b;
            pair_count[idx]++;
            if (pair_count[idx] == 1) {
                pair_a[n_pairs] = a;
                pair_b[n_pairs] = b;
                n_pairs++;
            }
        }

        /* Find best */
        for (int p = 0; p < n_pairs; p++) {
            int a = pair_a[p], b = pair_b[p];
            int cnt = pair_count[a * n_tokens + b];
            if (cnt > best_count) {
                best_count = cnt;
                best_a = a;
                best_b = b;
            }
        }

        free(pair_count); free(pair_a); free(pair_b);

        if (best_count < min_freq || best_a < 0) break;

        /* Create merged token */
        char* merged = (char*)malloc(strlen(tokens[best_a]) + strlen(tokens[best_b]) + 1);
        sprintf(merged, "%s%s", tokens[best_a], tokens[best_b]);
        tokens[n_tokens] = merged;
        scores[n_tokens] = -(float)n_tokens; /* Higher priority = lower score */
        freqs[n_tokens] = best_count;

        /* Replace all occurrences of the pair */
        int new_id = n_tokens;
        int write = 0;
        for (int i = 0; i < n_ids; i++) {
            if (i < n_ids - 1 && ids[i] == best_a && ids[i+1] == best_b) {
                ids[write++] = new_id;
                i++; /* skip b */
            } else {
                ids[write++] = ids[i];
            }
        }
        n_ids = write;
        n_tokens++;
    }

    /* Save vocabulary */
    FILE* vf = fopen(vocab_path, "w");
    if (!vf) { fprintf(stderr, "Cannot write vocab: %s\n", vocab_path); free(ids); return NULL; }
    for (int i = 0; i < n_tokens; i++) {
        fprintf(vf, "%s\t%.6f\n", tokens[i], scores[i]);
    }
    fclose(vf);

    /* Cleanup */
    for (int i = 0; i < n_tokens; i++) free(tokens[i]);
    free(tokens); free(scores); free(freqs); free(ids);

    printf("Trained BPE vocab: %d tokens → %s\n", n_tokens, vocab_path);
    return strdup(vocab_path);
}

/* ── Enhanced generation with ETOK tokenization ───────────────────── */

/*
 * Train a BPE tokenizer on a corpus, build a Transformer, and train.
 * This is the main entry point for training a ProbNet model from scratch
 * with proper BPE tokenization.
 */
static inline void train_with_bpe(const char* corpus_path,
                                   const char* model_path,
                                   const char* vocab_path,
                                   int target_vocab,
                                   int d_model, int n_layers, int n_heads,
                                   int ffn_hidden, int batch_size, int seq_len,
                                   int epochs, float lr) {
    /* Train vocab if not provided */
    char* vocab_file = NULL;
    struct stat st;
    if (stat(vocab_path, &st) != 0) {
        printf("Vocabulary not found at %s, training BPE...\n", vocab_path);
        vocab_file = etok_train_vocab(corpus_path, vocab_path, target_vocab, 2);
        if (!vocab_file) { fprintf(stderr, "Vocab training failed\n"); return; }
    } else {
        vocab_file = strdup(vocab_path);
    }

    /* Load tokenizer */
    Tokenizer* tok = tokenizer_load_vocab(vocab_file);
    if (!tok) { fprintf(stderr, "Failed to load vocab\n"); free(vocab_file); return; }

    /* Load dataset */
    Dataset* ds = dataset_load(corpus_path, tok);
    if (!ds) { fprintf(stderr, "Failed to load dataset\n"); tokenizer_free(tok); free(vocab_file); return; }

    /* Build model */
    TransformerConfig cfg = default_config();
    cfg.vocab_size = tok->vocab_size;
    cfg.d_model = d_model;
    cfg.n_layers = n_layers;
    cfg.n_heads = n_heads;
    cfg.n_kv_heads = n_heads;
    cfg.ffn_hidden = ffn_hidden;
    cfg.max_seq_len = seq_len * 2;
    snprintf(cfg.arch, 32, "probnet-bpe");

    Transformer* model = transformer_new(&cfg);
    transformer_print_info(model);

    /* Train */
    TrainConfig tcfg = train_default();
    tcfg.lr = lr;
    tcfg.batch_size = batch_size;
    tcfg.seq_len = seq_len;
    tcfg.n_epochs = epochs;
    tcfg.data_path[0] = 0;
    strncpy(tcfg.checkpoint_path, model_path, 255);

    TrainOptimizer* opt = train_opt_new(model, lr, 0.1f, 1.0f);
    train_full(model, opt, ds, &tcfg);

    /* Save */
    transformer_save(model, model_path);
    tokenizer_save_vocab(tok, vocab_path);

    /* Evaluate */
    RNG rng = rng_seed(42);
    printf("\n  Evaluating perplexity...\n");
    float ppl = eval_perplexity(model, ds, seq_len, 20, &rng);
    printf("  Perplexity: %.2f\n", ppl);

    /* Print APN specialization */
    for (int l = 0; l < cfg.n_layers; l++) {
        printf("  Layer %d: ", l);
        apn_print_specialization(model->blocks[l]->ffn);
    }

    /* Cleanup */
    train_opt_free(opt);
    dataset_free(ds);
    transformer_free(model);
    tokenizer_free(tok);
    free(vocab_file);
}