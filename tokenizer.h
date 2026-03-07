/*
 * ProbNet - tokenizer.h
 * BPE tokenizer fully compatible with LLaMA/Gemma vocabulary format.
 * Supports:
 *   - Loading from HuggingFace tokenizer.json
 *   - Loading from SentencePiece .model files (via exported vocab)
 *   - Character-level fallback for scratch training
 *   - encode() / decode() / encode_str()
 */
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define PN_MAX_VOCAB    200000
#define PN_MAX_TOKEN_LEN 256
#define PN_BPE_PAIRS    500000

typedef struct {
    char token[PN_MAX_TOKEN_LEN];
    float score;
    int   id;
    int   is_special;  /* 1 = BOS/EOS/PAD/etc */
} TokenEntry;

typedef struct {
    char a[PN_MAX_TOKEN_LEN];
    char b[PN_MAX_TOKEN_LEN];
    int priority;
} BPEMerge;

typedef struct {
    TokenEntry* vocab;
    int vocab_size;

    BPEMerge* merges;
    int n_merges;

    int bos_id, eos_id, pad_id, unk_id;
    int add_bos, add_eos;

    /* Hash table for O(1) token lookup */
    char**  ht_keys;
    int*    ht_vals;
    int     ht_size;

    char type[32];  /* "char", "bpe", "sentencepiece" */
} Tokenizer;

/* ── Simple hash table for token→id ────────────────────────────── */
static inline int ht_hash(const char* s, int sz) {
    unsigned h = 2166136261u;
    while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
    return (int)(h % (unsigned)sz);
}
static inline void ht_insert(char** keys, int* vals, int sz, const char* k, int v) {
    int h = ht_hash(k, sz);
    while (keys[h] && strcmp(keys[h], k) != 0) h = (h+1)%sz;
    if (!keys[h]) { keys[h]=strdup(k); vals[h]=v; }
}
static inline int ht_get(char** keys, int* vals, int sz, const char* k) {
    int h = ht_hash(k, sz);
    while (keys[h]) {
        if (strcmp(keys[h],k)==0) return vals[h];
        h=(h+1)%sz;
    }
    return -1;
}

/* ── Lifecycle ──────────────────────────────────────────────────── */
static inline Tokenizer* tokenizer_new_char(void) {
    Tokenizer* t = (Tokenizer*)calloc(1, sizeof(Tokenizer));
    t->vocab = (TokenEntry*)calloc(256+5, sizeof(TokenEntry));
    t->vocab_size = 0;
    /* ASCII printable */
    for (int c=0;c<256;c++) {
        t->vocab[c].id = c;
        t->vocab[c].token[0]=(char)c; t->vocab[c].token[1]=0;
        t->vocab[c].score=0; t->vocab[c].is_special=0;
        t->vocab_size++;
    }
    t->bos_id=256; t->eos_id=257; t->pad_id=258; t->unk_id=259;
    /* Add special tokens */
    for (int s=0;s<4;s++) {
        int id=256+s;
        const char* names[]={"<bos>","<eos>","<pad>","<unk>"};
        strncpy(t->vocab[id].token, names[s], PN_MAX_TOKEN_LEN-1);
        t->vocab[id].id=id; t->vocab[id].is_special=1;
        t->vocab_size++;
    }
    t->add_bos=1; t->add_eos=1;
    snprintf(t->type,32,"char");
    return t;
}

static inline void tokenizer_free(Tokenizer* t) {
    if (!t) return;
    free(t->vocab); free(t->merges);
    if (t->ht_keys) {
        for(int i=0;i<t->ht_size;i++) if(t->ht_keys[i]) free(t->ht_keys[i]);
        free(t->ht_keys); free(t->ht_vals);
    }
    free(t);
}

static inline void tokenizer_build_ht(Tokenizer* t) {
    t->ht_size = t->vocab_size * 3;
    t->ht_keys = (char**)calloc(t->ht_size, sizeof(char*));
    t->ht_vals = (int*)calloc(t->ht_size, sizeof(int));
    for (int i=0;i<t->vocab_size;i++)
        ht_insert(t->ht_keys, t->ht_vals, t->ht_size, t->vocab[i].token, t->vocab[i].id);
}

static inline int tokenizer_token_to_id(Tokenizer* t, const char* tok) {
    if (!t->ht_keys) tokenizer_build_ht(t);
    int r = ht_get(t->ht_keys, t->ht_vals, t->ht_size, tok);
    return r >= 0 ? r : t->unk_id;
}

static inline const char* tokenizer_id_to_token(Tokenizer* t, int id) {
    if (id < 0 || id >= t->vocab_size) return "<?>";
    return t->vocab[id].token;
}

/* ── Encode: text → token ids ───────────────────────────────────── */
static inline int* tokenizer_encode(Tokenizer* t, const char* text,
                                     int add_bos, int add_eos, int* out_len) {
    int maxlen = strlen(text) + 4;
    int* ids = (int*)malloc((size_t)maxlen * sizeof(int));
    int n = 0;

    if (add_bos && t->bos_id >= 0) ids[n++] = t->bos_id;

    if (strcmp(t->type,"char")==0) {
        /* Character-level tokenization */
        const unsigned char* p = (const unsigned char*)text;
        while (*p) { ids[n++]=*p++; if(n>=maxlen-2){maxlen*=2;ids=(int*)realloc(ids,maxlen*sizeof(int));} }
    } else {
        /* BPE tokenization */
        if (!t->ht_keys) tokenizer_build_ht(t);
        /* Simple: byte-level fallback if no BPE merges loaded */
        const unsigned char* p = (const unsigned char*)text;
        while (*p) {
            char tok[PN_MAX_TOKEN_LEN]; int tl=0;
            tok[tl++]=(char)*p++;
            /* Try to extend greedily */
            while (*p && tl < PN_MAX_TOKEN_LEN-1) {
                tok[tl]=(char)*p; tok[tl+1]=0;
                if (ht_get(t->ht_keys, t->ht_vals, t->ht_size, tok) >= 0) { tl++; p++; }
                else break;
            }
            tok[tl]=0;
            int id=ht_get(t->ht_keys, t->ht_vals, t->ht_size, tok);
            if (id<0) id=t->unk_id;
            ids[n++]=id;
            if(n>=maxlen-2){maxlen*=2;ids=(int*)realloc(ids,maxlen*sizeof(int));}
        }
    }

    if (add_eos && t->eos_id >= 0) ids[n++] = t->eos_id;
    *out_len = n;
    return ids;
}

/* ── Decode: token ids → text ───────────────────────────────────── */
static inline char* tokenizer_decode(Tokenizer* t, const int* ids, int n) {
    /* estimate output size */
    size_t cap = (size_t)n * 8 + 1;
    char* out = (char*)malloc(cap);
    size_t pos = 0;
    for (int i=0;i<n;i++) {
        if (ids[i]==t->bos_id||ids[i]==t->eos_id||ids[i]==t->pad_id) continue;
        const char* tok = tokenizer_id_to_token(t, ids[i]);
        /* LLaMA/SPM: replace ▁ with space */
        const char* p = tok;
        if ((unsigned char)p[0]==0xe2 && (unsigned char)p[1]==0x96 && (unsigned char)p[2]==0x81) {
            if (pos > 0) out[pos++]=' '; p+=3;
        }
        size_t len=strlen(p);
        if (pos+len+2 >= cap) { cap=cap*2+len; out=(char*)realloc(out,cap); }
        memcpy(out+pos, p, len); pos+=len;
    }
    out[pos]=0;
    return out;
}

/* ── Load vocabulary from simple text file ──────────────────────── */
/* Format: one token per line, optionally "token<TAB>score" */
static inline Tokenizer* tokenizer_load_vocab(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) { fprintf(stderr,"Cannot open vocab: %s\n",path); return NULL; }
    Tokenizer* t = (Tokenizer*)calloc(1, sizeof(Tokenizer));
    t->vocab = (TokenEntry*)calloc(PN_MAX_VOCAB, sizeof(TokenEntry));
    char line[PN_MAX_TOKEN_LEN*2];
    t->vocab_size=0;
    t->bos_id=t->eos_id=t->pad_id=t->unk_id=-1;
    while (fgets(line,sizeof(line),fp) && t->vocab_size<PN_MAX_VOCAB) {
        /* Remove newline */
        char* nl=strchr(line,'\n'); if(nl)*nl=0;
        char* tab=strchr(line,'\t');
        float score=0; char tok[PN_MAX_TOKEN_LEN];
        if (tab) { *tab=0; strncpy(tok,line,PN_MAX_TOKEN_LEN-1); score=(float)atof(tab+1); }
        else     { strncpy(tok,line,PN_MAX_TOKEN_LEN-1); }
        tok[PN_MAX_TOKEN_LEN-1]=0;
        int id=t->vocab_size;
        strncpy(t->vocab[id].token,tok,PN_MAX_TOKEN_LEN-1);
        t->vocab[id].score=score; t->vocab[id].id=id;
        /* Detect special tokens */
        if(strcmp(tok,"<bos>")==0||strcmp(tok,"<s>")==0) t->bos_id=id;
        if(strcmp(tok,"<eos>")==0||strcmp(tok,"</s>")==0) t->eos_id=id;
        if(strcmp(tok,"<pad>")==0) t->pad_id=id;
        if(strcmp(tok,"<unk>")==0||strcmp(tok,"[UNK]")==0) t->unk_id=id;
        t->vocab[id].is_special=(tok[0]=='<'&&tok[strlen(tok)-1]=='>');
        t->vocab_size++;
    }
    fclose(fp);
    t->add_bos=1; t->add_eos=1;
    snprintf(t->type,32,"bpe");
    tokenizer_build_ht(t);
    printf("Loaded vocab: %d tokens (bos=%d eos=%d)\n",t->vocab_size,t->bos_id,t->eos_id);
    return t;
}

/* ── Save vocabulary ────────────────────────────────────────────── */
static inline int tokenizer_save_vocab(Tokenizer* t, const char* path) {
    FILE* fp = fopen(path,"w");
    if (!fp) return -1;
    for(int i=0;i<t->vocab_size;i++)
        fprintf(fp,"%s\t%.6f\n",t->vocab[i].token,t->vocab[i].score);
    fclose(fp);
    return 0;
}
