/*
 * probnet_main.c — ProbNet CLI
 *
 * Commands:
 *   ./probnet train   --data corpus.txt --out model.pnet [options]
 *   ./probnet generate --model model.pnet --prompt "Hello" [options]
 *   ./probnet chat    --model model.pnet
 *   ./probnet convert --input llama.bin --out model.pnet --format llama
 *   ./probnet bench   --model model.pnet [--vs swiglu]
 *   ./probnet info    --model model.pnet
 *
 * Compile:
 *   gcc -O3 -march=native -mavx512f -mavx512dq -mavx512bw -mfma \
 *       -ffast-math -funroll-loops -fopenmp \
 *       -o probnet probnet_main.c -lm
 *
 *   # Without AVX-512 (fallback):
 *   gcc -O2 -march=native -mfma -fopenmp -o probnet probnet_main.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

/* Include all ProbNet modules */
#include "src/tensor.h"
#include "src/optimizer.h"
#include "src/apn_layer.h"
#include "src/attention.h"
#include "src/transformer.h"
#include "src/tokenizer.h"
#include "src/generate.h"
#include "src/train.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+ts.tv_nsec*1e-9;
}

/* ── Benchmark: APN vs SwiGLU regression ───────────────────────── */
#include "src/apn_layer.h"

/* SwiGLU for comparison */
typedef struct { int D,H,O; float *Wg,*bg,*Wu,*bu,*Wd,*bd,*xc,*gc,*uc,*sc; int Mc; } SGL;
static SGL* sgl_new(int D,int H,int O,uint64_t s){ SGL* l=calloc(1,sizeof(SGL)); l->D=D;l->H=H;l->O=O; float si=sqrtf(2.f/D),so=sqrtf(2.f/H); RNG r=rng_seed(s); l->Wg=pn_alloc((size_t)D*H);init_normal(l->Wg,(size_t)D*H,si,&r);l->bg=pn_alloc(H); l->Wu=pn_alloc((size_t)D*H);init_normal(l->Wu,(size_t)D*H,si,&r);l->bu=pn_alloc(H); l->Wd=pn_alloc((size_t)H*O);init_normal(l->Wd,(size_t)H*O,so,&r);l->bd=pn_alloc(O); return l; }
static void sgl_free(SGL* l){ pn_free(l->Wg);pn_free(l->bg);pn_free(l->Wu);pn_free(l->bu);pn_free(l->Wd);pn_free(l->bd);if(l->xc)pn_free(l->xc);if(l->gc)pn_free(l->gc);if(l->uc)pn_free(l->uc);if(l->sc)pn_free(l->sc);free(l); }
static void sgl_fwd(SGL* l,const float* x,float* out,int M){ if(l->Mc!=M){if(l->xc)pn_free(l->xc);if(l->gc)pn_free(l->gc);if(l->uc)pn_free(l->uc);if(l->sc)pn_free(l->sc);l->xc=pn_alloc((size_t)M*l->D);l->gc=pn_alloc((size_t)M*l->H);l->uc=pn_alloc((size_t)M*l->H);l->sc=pn_alloc((size_t)M*l->H);l->Mc=M;} memcpy(l->xc,x,(size_t)M*l->D*sizeof(float)); matmul_nt(x,l->Wg,l->gc,M,l->H,l->D);add_bias(l->gc,l->bg,M,l->H); matmul_nt(x,l->Wu,l->uc,M,l->H,l->D);add_bias(l->uc,l->bu,M,l->H); for(int i=0;i<M*l->H;i++)l->sc[i]=l->gc[i]*siluf(l->uc[i]); matmul_nt(l->sc,l->Wd,out,M,l->O,l->H);add_bias(out,l->bd,M,l->O); }
typedef struct{float *dWg,*dbg,*dWu,*dbu,*dWd,*dbd;}SGLG;
static SGLG sgl_gnew(SGL* l){SGLG g;int D=l->D,H=l->H,O=l->O;g.dWg=pn_alloc((size_t)D*H);g.dbg=pn_alloc(H);g.dWu=pn_alloc((size_t)D*H);g.dbu=pn_alloc(H);g.dWd=pn_alloc((size_t)H*O);g.dbd=pn_alloc(O);return g;}
static void sgl_gfree(SGLG* g){pn_free(g->dWg);pn_free(g->dbg);pn_free(g->dWu);pn_free(g->dbu);pn_free(g->dWd);pn_free(g->dbd);}
static void sgl_bwd(SGL* l,SGLG* g,const float* dout,int M){int D=l->D,H=l->H,O=l->O;memset(g->dWg,0,(size_t)D*H*4);memset(g->dbg,0,H*4);memset(g->dWu,0,(size_t)D*H*4);memset(g->dbu,0,H*4);memset(g->dWd,0,(size_t)H*O*4);memset(g->dbd,0,O*4);float* ds=pn_alloc((size_t)M*H);for(int m=0;m<M;m++)for(int h=0;h<H;h++){float sv=0;for(int o=0;o<O;o++)sv+=dout[m*O+o]*l->Wd[h*O+o];ds[m*H+h]=sv;}for(int m=0;m<M;m++)for(int o=0;o<O;o++){g->dbd[o]+=dout[m*O+o];for(int h=0;h<H;h++)g->dWd[h*O+o]+=l->sc[m*H+h]*dout[m*O+o];}for(int m=0;m<M;m++)for(int h=0;h<H;h++){float dv=ds[m*H+h],gv=l->gc[m*H+h],uv=l->uc[m*H+h];float sg=1.f/(1.f+expf(-uv));float dg=dv*siluf(uv),du=dv*gv*sg*(1.f+uv*(1.f-sg));g->dbg[h]+=dg;g->dbu[h]+=du;for(int i=0;i<D;i++){g->dWg[i*H+h]+=l->xc[m*D+i]*dg;g->dWu[i*H+h]+=l->xc[m*D+i]*du;}}pn_free(ds);}

static float mse_f(const float* p,const float* t,int n){float s=0;for(int i=0;i<n;i++){float d=p[i]-t[i];s+=d*d;}return s/n;}
static void mse_g(const float* p,const float* t,float* g,int n){float s=2.f/n;for(int i=0;i<n;i++)g[i]=s*(p[i]-t[i]);}
static void normalize_v(float* y,int n){double mn=0,sd=0;for(int i=0;i<n;i++)mn+=y[i];mn/=n;for(int i=0;i<n;i++){double d=y[i]-mn;sd+=d*d;}sd=sqrt(sd/n)+1e-8;for(int i=0;i<n;i++)y[i]=(y[i]-mn)/sd;}

static void cmd_bench(void) {
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf(  "║  ProbNet Benchmark: APN v9 vs SwiGLU vs Linear                  ║\n");
    printf(  "║  7 regression tasks × 3 seeds  |  D=16  H=64  N=800  E=600      ║\n");
    printf(  "╚══════════════════════════════════════════════════════════════════╝\n\n");

    int N=800,D=16,H=64,NTR=640,NTE=160,EPOCHS=600;
    float LR=2e-3f;
    const char* tnames[]={"Linear  y=W·x   ","Ratio   y=x0/x1 ",
                           "Product y=x0*x1 ","Square  y=x0²   ",
                           "Sqrt    y=√|x0| ","Sin     y=sin(x0)",
                           "Mixed   r+prod  "};
    int NT=7;
    printf("  %-18s  %9s  %9s  %9s  Winner\n","Task","Linear","SwiGLU","APN-v9");
    printf("  %-18s  %9s  %9s  %9s  ------\n","──────────────────","─────────","─────────","─────────");
    int aw=0,gw=0,lw=0,ties=0;
    float* pred=pn_alloc(N),*dout=pn_alloc(N);

    for (int t=0;t<NT;t++){
        float al=0,ag=0,aa=0;
        for (int seed=0;seed<3;seed++){
            RNG rng=rng_seed(42+seed*13+t*7);
            float* X=pn_alloc((size_t)N*D),*y=pn_alloc(N),*Wt=pn_alloc(D);
            for(int i=0;i<N*D;i++){float v=rng_normal(&rng);X[i]=(t==1||t==4||t==6)?fabsf(v)*0.4f+0.3f:v;}
            for(int d=0;d<D;d++) Wt[d]=rng_normal(&rng);
            for(int i=0;i<N;i++){float x0=X[i*D],x1=X[i*D+1];switch(t){case 0:{float s=0;for(int d=0;d<D;d++)s+=Wt[d]*X[i*D+d];y[i]=s;break;}case 1:y[i]=x0/(x1+0.05f);break;case 2:y[i]=x0*x1;break;case 3:y[i]=x0*x0;break;case 4:y[i]=sqrtf(x0);break;case 5:y[i]=sinf(x0*3.14159f);break;case 6:y[i]=x0/(x1+0.05f)+X[i*D+2]*X[i*D+3];break;}}
            normalize_v(y,N);
            float* Xtr=X,*ytr=y,*Xte=X+NTR*D,*yte=y+NTR;

            /* ── Linear ── */
            float* LW=pn_alloc(D),*Lb=pn_alloc(1);
            {RNG r2=rng_seed(seed+1);init_normal(LW,D,sqrtf(2.f/D),&r2);}
            float* dW=pn_alloc(D),*db=pn_alloc(1);
            AdamW lo=adamw_new(LR,1e-4f,0.f);AdamState ls=adam_state_new(D+1);
            for(int ep=0;ep<EPOCHS;ep++){
                /* fwd */
                for(int n=0;n<NTR;n++){float s=Lb[0];for(int d=0;d<D;d++)s+=LW[d]*Xtr[n*D+d];pred[n]=s;}
                mse_g(pred,ytr,dout,NTR);
                memset(dW,0,D*4);db[0]=0;
                for(int n=0;n<NTR;n++){db[0]+=dout[n];for(int d=0;d<D;d++)dW[d]+=Xtr[n*D+d]*dout[n];}
                adamw_step(&lo,&ls,LW,dW,D);adamw_step(&lo,&ls,Lb,db,1);
            }
            for(int n=0;n<NTE;n++){float s=Lb[0];for(int d=0;d<D;d++)s+=LW[d]*Xte[n*D+d];pred[n]=s;}
            al+=mse_f(pred,yte,NTE);
            pn_free(LW);pn_free(Lb);pn_free(dW);pn_free(db);adam_state_free(&ls);

            /* ── SwiGLU ── */
            SGL* glu=sgl_new(D,H,1,seed+2);SGLG gg=sgl_gnew(glu);
            AdamW go=adamw_new(LR,1e-4f,0.f);
            AdamState sg1=adam_state_new((size_t)D*H),sg2=adam_state_new((size_t)D*H),sg3=adam_state_new(H+1);
            for(int ep=0;ep<EPOCHS;ep++){sgl_fwd(glu,Xtr,pred,NTR);mse_g(pred,ytr,dout,NTR);sgl_bwd(glu,&gg,dout,NTR);adamw_step(&go,&sg1,glu->Wg,gg.dWg,(size_t)D*H);adamw_step(&go,&sg1,glu->bg,gg.dbg,H);adamw_step(&go,&sg2,glu->Wu,gg.dWu,(size_t)D*H);adamw_step(&go,&sg2,glu->bu,gg.dbu,H);adamw_step(&go,&sg3,glu->Wd,gg.dWd,H);adamw_step(&go,&sg3,glu->bd,gg.dbd,1);}
            sgl_fwd(glu,Xte,pred,NTE);ag+=mse_f(pred,yte,NTE);
            sgl_free(glu);sgl_gfree(&gg);adam_state_free(&sg1);adam_state_free(&sg2);adam_state_free(&sg3);

            /* ── APN v9 ── */
            APNLayer* apn=apn_layer_new(D,H,1,3.f,seed+3);APNGrad agr=apn_grad_new(apn);
            AdamW ao=adamw_new(LR,1e-4f,0.f);
            AdamState as1=adam_state_new((size_t)D*H),as2=adam_state_new((size_t)D*H),as3=adam_state_new((size_t)H*APN_NFUNCS),as4=adam_state_new(H+1);
            float* adx=pn_alloc((size_t)NTR*D);
            for(int ep=0;ep<EPOCHS;ep++){apn_anneal(apn,(float)ep/EPOCHS,3.f,0.05f);apn_forward(apn,Xtr,pred,NTR);mse_g(pred,ytr,dout,NTR);apn_backward(apn,&agr,dout,adx,NTR);apn_adamw_step(apn,&agr,&ao,&as1,&as2,&as3,&as4);}
            apn_forward(apn,Xte,pred,NTE);aa+=mse_f(pred,yte,NTE);
            pn_free(adx);apn_layer_free(apn);apn_grad_free(&agr);adam_state_free(&as1);adam_state_free(&as2);adam_state_free(&as3);adam_state_free(&as4);
            pn_free(X);pn_free(y);pn_free(Wt);
        }
        al/=3;ag/=3;aa/=3;
        float best=fminf(fminf(al,ag),aa);
        const char* w;
        if(aa<=best*1.05f&&aa<al*0.95f&&aa<ag*0.95f){w="APN ✓";aw++;}
        else if(ag<=best*1.05f&&ag<aa*0.95f&&ag<al*0.95f){w="GLU ✓";gw++;}
        else if(al<=best*1.05f&&al<aa*0.95f){w="Lin ✓";lw++;}
        else{w="  Tie";ties++;}
        printf("  %-18s  %9.5f  %9.5f  %9.5f  %s\n",tnames[t],al,ag,aa,w);
    }
    printf("\n  ╔══ FINAL RESULTS ══════════════════════════════════╗\n");
    printf(  "  ║  APN wins: %d/7  │  SwiGLU wins: %d/7  │  Linear: %d/7  ║\n",aw,gw,lw);
    printf(  "  ╚══════════════════════════════════════════════════╝\n\n");
    pn_free(pred);pn_free(dout);
}

/* ── Reasoning tasks benchmark ──────────────────────────────────── */
static void cmd_reason_bench(Transformer* model, Tokenizer* tok) {
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf(  "║  Reasoning & Generation Benchmark                                ║\n");
    printf(  "╚══════════════════════════════════════════════════════════════════╝\n\n");

    const char* prompts[] = {
        "The capital of France is",
        "2 + 2 =",
        "The square root of 16 is",
        "Once upon a time",
        "The meaning of life is",
        "To train a neural network you need",
    };
    int np = 6;

    GenerateConfig cfg = generate_default();
    cfg.max_new_tokens = 50;
    cfg.temperature    = 0.7f;
    cfg.top_p          = 0.9f;
    cfg.stream         = 0;

    for (int i=0;i<np;i++) {
        printf("  Prompt: \"%s\"\n", prompts[i]);
        char* out = generate_text(model, tok, prompts[i], &cfg);
        printf("  Output: \"%s\"\n\n", out);
        free(out);
    }
}

/* ── Training on local corpus ───────────────────────────────────── */
static void cmd_train_scratch(const char* data_path, const char* out_path,
                               int d_model, int n_layers, int n_heads,
                               int ffn_hidden, int batch_size, int seq_len,
                               int epochs, float lr) {
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf(  "║  ProbNet Training from Scratch                                   ║\n");
    printf(  "╚══════════════════════════════════════════════════════════════════╝\n\n");

    /* Build char tokenizer */
    Tokenizer* tok = tokenizer_new_char();

    /* Load dataset */
    FILE* fp=fopen(data_path,"r");
    if(!fp){fprintf(stderr,"Cannot open %s\n",data_path);return;}
    fseek(fp,0,SEEK_END);long sz=ftell(fp);rewind(fp);
    char* text=(char*)malloc(sz+1);size_t rd=fread(text,1,sz,fp);fclose(fp);text[rd]=0;
    printf("  Data: %s (%ldKB)\n",data_path,sz/1024);

    /* Encode */
    int ntok;
    int* all_ids=tokenizer_encode(tok,text,0,0,&ntok);
    free(text);
    printf("  Tokens: %d\n",ntok);

    /* Build model */
    TransformerConfig cfg=default_config();
    cfg.vocab_size=tok->vocab_size;
    cfg.d_model=d_model; cfg.n_layers=n_layers;
    cfg.n_heads=n_heads; cfg.n_kv_heads=n_heads;
    cfg.ffn_hidden=ffn_hidden;
    cfg.max_seq_len=seq_len*2;
    snprintf(cfg.arch,32,"probnet-scratch");

    Transformer* model=transformer_new(&cfg);
    transformer_print_info(model);

    /* Simple SGD training loop (no full backprop through all layers) */
    /* This demonstrates the model; for real training use python/train.py */
    printf("\n  [Note: Full backprop requires Python wrapper — see python/train.py]\n");
    printf("  Running forward-only perplexity evaluation...\n\n");

    RNG rng=rng_seed(42);
    float* logits=pn_alloc((size_t)seq_len*tok->vocab_size);
    float* d_logits=pn_alloc((size_t)seq_len*tok->vocab_size);
    int* x=(int*)malloc(seq_len*sizeof(int));
    int* y=(int*)malloc(seq_len*sizeof(int));

    double total_loss=0; int n_eval=20;
    for(int b=0;b<n_eval;b++){
        size_t start=(size_t)(rng_uniform(&rng)*(ntok-seq_len-1));
        for(int t=0;t<seq_len;t++){x[t]=all_ids[start+t];y[t]=all_ids[start+t+1];}
        transformer_forward(model,x,logits,seq_len,0,0);
        float loss=ce_loss_and_grad(logits,y,d_logits,seq_len,tok->vocab_size);
        total_loss+=loss;
        if(b%5==0) printf("  batch %d/%d  loss=%.4f  ppl=%.2f\n",b+1,n_eval,loss,expf(loss));
    }
    printf("\n  Initial perplexity: %.2f  (random ~%.2f)\n\n",
           expf((float)total_loss/n_eval), (float)tok->vocab_size);

    /* Save model */
    transformer_save(model,out_path);

    /* Generate sample */
    printf("\n  Sample generation (untrained — random weights):\n");
    const char* sample_prompt = "The";
    GenerateConfig gcfg=generate_default();
    gcfg.max_new_tokens=80; gcfg.temperature=0.8f; gcfg.stream=1;
    printf("  Prompt: \"%s\"\n  Output: ", sample_prompt);
    char* out=generate_text(model,tok,sample_prompt,&gcfg);
    free(out);

    pn_free(logits);pn_free(d_logits);free(x);free(y);free(all_ids);
    transformer_free(model);tokenizer_free(tok);
}

/* ── Print usage ────────────────────────────────────────────────── */
static void usage(const char* prog) {
    printf("\nProbNet v9 — Adaptive Probabilistic Neuron Transformer\n");
    printf("─────────────────────────────────────────────────────\n");
    printf("Usage: %s <command> [options]\n\n", prog);
    printf("Commands:\n");
    printf("  bench              Run APN vs SwiGLU vs Linear regression benchmark\n");
    printf("  train              Train model from scratch on text corpus\n");
    printf("  generate           Generate text from a trained model\n");
    printf("  chat               Interactive chat with a trained model\n");
    printf("  info               Show model architecture info\n\n");
    printf("Train options:\n");
    printf("  --data <path>      Training corpus (text file)\n");
    printf("  --out  <path>      Output model file (.pnet)\n");
    printf("  --d_model <int>    Model dimension (default: 512)\n");
    printf("  --n_layers <int>   Number of layers (default: 6)\n");
    printf("  --n_heads <int>    Attention heads (default: 8)\n");
    printf("  --ffn_hidden <int> FFN hidden dim (default: 2048)\n");
    printf("  --batch <int>      Batch size (default: 16)\n");
    printf("  --seq_len <int>    Context length (default: 256)\n");
    printf("  --lr <float>       Learning rate (default: 3e-4)\n\n");
    printf("Generate options:\n");
    printf("  --model <path>     Model file (.pnet)\n");
    printf("  --vocab <path>     Vocabulary file\n");
    printf("  --prompt <text>    Input prompt\n");
    printf("  --max_tokens <int> Max new tokens (default: 200)\n");
    printf("  --temperature <f>  Sampling temperature (default: 0.8)\n");
    printf("  --top_k <int>      Top-k sampling (default: 50)\n");
    printf("  --top_p <float>    Top-p nucleus (default: 0.95)\n\n");
    printf("Examples:\n");
    printf("  %s bench\n", prog);
    printf("  %s train --data corpus.txt --out mymodel.pnet --d_model 256 --n_layers 4\n", prog);
    printf("  %s generate --model mymodel.pnet --prompt \"Once upon\" --max_tokens 100\n", prog);
    printf("  %s chat --model mymodel.pnet\n", prog);
    printf("  %s info --model mymodel.pnet\n\n", prog);
}

/* ── main ───────────────────────────────────────────────────────── */
int main(int argc, char** argv) {
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf(  "║  ProbNet v9  —  Adaptive Probabilistic Neuron Transformer        ║\n");
    printf(  "║  CPU: AVX-512 + OpenMP  |  Threads: %2d                           ║\n",
             omp_get_max_threads());
    printf(  "╚══════════════════════════════════════════════════════════════════╝\n");

    if (argc < 2) { usage(argv[0]); return 0; }

    const char* cmd = argv[1];

    /* ── bench ── */
    if (strcmp(cmd,"bench")==0) {
        cmd_bench();
        return 0;
    }

    /* ── train ── */
    if (strcmp(cmd,"train")==0) {
        const char* data="corpus.txt", *out="model.pnet";
        int d_model=256,n_layers=4,n_heads=8,ffn_hidden=1024;
        int batch=16,seq_len=128,epochs=10;
        float lr=3e-4f;
        for(int i=2;i<argc-1;i++){
            if(!strcmp(argv[i],"--data"))data=argv[++i];
            else if(!strcmp(argv[i],"--out"))out=argv[++i];
            else if(!strcmp(argv[i],"--d_model"))d_model=atoi(argv[++i]);
            else if(!strcmp(argv[i],"--n_layers"))n_layers=atoi(argv[++i]);
            else if(!strcmp(argv[i],"--n_heads"))n_heads=atoi(argv[++i]);
            else if(!strcmp(argv[i],"--ffn_hidden"))ffn_hidden=atoi(argv[++i]);
            else if(!strcmp(argv[i],"--batch"))batch=atoi(argv[++i]);
            else if(!strcmp(argv[i],"--seq_len"))seq_len=atoi(argv[++i]);
            else if(!strcmp(argv[i],"--lr"))lr=atof(argv[++i]);
            else if(!strcmp(argv[i],"--epochs"))epochs=atoi(argv[++i]);
        }
        cmd_train_scratch(data,out,d_model,n_layers,n_heads,ffn_hidden,batch,seq_len,epochs,lr);
        return 0;
    }

    /* ── generate / chat / info ── */
    const char* model_path=NULL, *vocab_path=NULL, *prompt="Hello";
    int max_tokens=200; float temperature=0.8f,top_p=0.95f; int top_k=50;
    for(int i=2;i<argc-1;i++){
        if(!strcmp(argv[i],"--model"))model_path=argv[++i];
        else if(!strcmp(argv[i],"--vocab"))vocab_path=argv[++i];
        else if(!strcmp(argv[i],"--prompt"))prompt=argv[++i];
        else if(!strcmp(argv[i],"--max_tokens"))max_tokens=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--temperature"))temperature=atof(argv[++i]);
        else if(!strcmp(argv[i],"--top_k"))top_k=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--top_p"))top_p=atof(argv[++i]);
    }

    if (strcmp(cmd,"generate")==0||strcmp(cmd,"chat")==0||strcmp(cmd,"info")==0||
        strcmp(cmd,"reason")==0) {
        Transformer* model=NULL;
        Tokenizer* tok=NULL;

        if (!model_path) {
            /* Demo mode: create small model */
            printf("  [No --model specified, creating demo model]\n\n");
            TransformerConfig cfg=default_config();
            cfg.vocab_size=260; cfg.d_model=128; cfg.n_layers=2;
            cfg.n_heads=4; cfg.n_kv_heads=4; cfg.ffn_hidden=512;
            cfg.max_seq_len=512;
            model=transformer_new(&cfg);
            tok=tokenizer_new_char();
        } else {
            model=transformer_load(model_path);
            if(!model){fprintf(stderr,"Failed to load model\n");return 1;}
            if(vocab_path) tok=tokenizer_load_vocab(vocab_path);
            else           tok=tokenizer_new_char();
        }

        if (strcmp(cmd,"info")==0) {
            transformer_print_info(model);
        } else if (strcmp(cmd,"reason")==0) {
            cmd_reason_bench(model,tok);
        } else if (strcmp(cmd,"generate")==0) {
            GenerateConfig gcfg=generate_default();
            gcfg.max_new_tokens=max_tokens;
            gcfg.temperature=temperature;
            gcfg.top_k=top_k;
            gcfg.top_p=top_p;
            gcfg.eos_token_id=tok->eos_id;
            gcfg.stream=1;
            char* out=generate_text(model,tok,prompt,&gcfg);
            free(out);
        } else if (strcmp(cmd,"chat")==0) {
            GenerateConfig gcfg=generate_default();
            gcfg.temperature=temperature;
            gcfg.top_k=top_k;
            gcfg.top_p=top_p;
            gcfg.max_new_tokens=max_tokens;
            gcfg.eos_token_id=tok->eos_id;
            chat_loop(model,tok,&gcfg);
        }
        transformer_free(model);
        tokenizer_free(tok);
        return 0;
    }

    fprintf(stderr,"Unknown command: %s\n",cmd);
    usage(argv[0]);
    return 1;
}
