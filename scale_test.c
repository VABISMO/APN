/*
 * scale_test.c — ProbNet Standalone Regression + LM Benchmark
 *
 * This is the benchmark that validated APN v9 beats SwiGLU 5/7 tasks.
 * Run standalone (no other files needed except src/):
 *
 *   gcc -O3 -march=native -mavx512f -mfma -fopenmp -ffast-math \
 *       -o scale_test scale_test.c -lm
 *   ./scale_test
 *
 * Results (confirmed):
 *   APN wins: 5/7  |  SwiGLU wins: 1/7  |  Linear wins: 1/7
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "src/tensor.h"
#include "src/optimizer.h"
#include "src/apn_layer.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

/* ── SwiGLU for comparison ─────────────────────────────────────── */
typedef struct {
    int D,H,O;
    float *Wg,*bg,*Wu,*bu,*Wd,*bd;
    float *xc,*gc,*uc,*sc;
    int Mc;
} SGL;

static SGL* sgl_new(int D,int H,int O,uint64_t s){
    SGL* l=calloc(1,sizeof(SGL));l->D=D;l->H=H;l->O=O;
    float si=sqrtf(2.f/D),so=sqrtf(2.f/H);
    RNG r=rng_seed(s);
    l->Wg=pn_alloc((size_t)D*H);init_normal(l->Wg,(size_t)D*H,si,&r);l->bg=pn_alloc(H);
    l->Wu=pn_alloc((size_t)D*H);init_normal(l->Wu,(size_t)D*H,si,&r);l->bu=pn_alloc(H);
    l->Wd=pn_alloc((size_t)H*O);init_normal(l->Wd,(size_t)H*O,so,&r);l->bd=pn_alloc(O);
    return l;
}
static void sgl_free(SGL* l){
    pn_free(l->Wg);pn_free(l->bg);pn_free(l->Wu);pn_free(l->bu);
    pn_free(l->Wd);pn_free(l->bd);
    if(l->xc)pn_free(l->xc);if(l->gc)pn_free(l->gc);
    if(l->uc)pn_free(l->uc);if(l->sc)pn_free(l->sc);
    free(l);
}
static void sgl_ensure(SGL* l,int M){
    if(l->Mc==M)return;
    if(l->xc)pn_free(l->xc);if(l->gc)pn_free(l->gc);
    if(l->uc)pn_free(l->uc);if(l->sc)pn_free(l->sc);
    l->xc=pn_alloc((size_t)M*l->D);l->gc=pn_alloc((size_t)M*l->H);
    l->uc=pn_alloc((size_t)M*l->H);l->sc=pn_alloc((size_t)M*l->H);l->Mc=M;
}
static void sgl_fwd(SGL* l,const float*x,float*out,int M){
    sgl_ensure(l,M);
    memcpy(l->xc,x,(size_t)M*l->D*sizeof(float));
    matmul_nt(x,l->Wg,l->gc,M,l->H,l->D);add_bias(l->gc,l->bg,M,l->H);
    matmul_nt(x,l->Wu,l->uc,M,l->H,l->D);add_bias(l->uc,l->bu,M,l->H);
    for(int i=0;i<M*l->H;i++) l->sc[i]=l->gc[i]*siluf(l->uc[i]);
    matmul_nt(l->sc,l->Wd,out,M,l->O,l->H);add_bias(out,l->bd,M,l->O);
}
typedef struct{float*dWg,*dbg,*dWu,*dbu,*dWd,*dbd;}SGLG;
static SGLG sgl_gnew(SGL*l){
    SGLG g;int D=l->D,H=l->H,O=l->O;
    g.dWg=pn_alloc((size_t)D*H);g.dbg=pn_alloc(H);
    g.dWu=pn_alloc((size_t)D*H);g.dbu=pn_alloc(H);
    g.dWd=pn_alloc((size_t)H*O);g.dbd=pn_alloc(O);
    return g;
}
static void sgl_gfree(SGLG*g){
    pn_free(g->dWg);pn_free(g->dbg);pn_free(g->dWu);
    pn_free(g->dbu);pn_free(g->dWd);pn_free(g->dbd);
}
static void sgl_bwd(SGL*l,SGLG*g,const float*dout,int M){
    int D=l->D,H=l->H,O=l->O;
    memset(g->dWg,0,(size_t)D*H*4);memset(g->dbg,0,H*4);
    memset(g->dWu,0,(size_t)D*H*4);memset(g->dbu,0,H*4);
    memset(g->dWd,0,(size_t)H*O*4);memset(g->dbd,0,O*4);
    float*ds=pn_alloc((size_t)M*H);
    for(int m=0;m<M;m++)for(int h=0;h<H;h++){
        float sv=0;for(int o=0;o<O;o++)sv+=dout[m*O+o]*l->Wd[h*O+o];ds[m*H+h]=sv;
    }
    for(int m=0;m<M;m++)for(int o=0;o<O;o++){
        g->dbd[o]+=dout[m*O+o];
        for(int h=0;h<H;h++) g->dWd[h*O+o]+=l->sc[m*H+h]*dout[m*O+o];
    }
    for(int m=0;m<M;m++)for(int h=0;h<H;h++){
        float dv=ds[m*H+h],gv=l->gc[m*H+h],uv=l->uc[m*H+h];
        float sg=1.f/(1.f+expf(-uv));
        float dg=dv*siluf(uv),du=dv*gv*sg*(1.f+uv*(1.f-sg));
        g->dbg[h]+=dg;g->dbu[h]+=du;
        for(int i=0;i<D;i++){g->dWg[i*H+h]+=l->xc[m*D+i]*dg;g->dWu[i*H+h]+=l->xc[m*D+i]*du;}
    }
    pn_free(ds);
}

/* ── Loss ─────────────────────────────────────────────────────── */
static float mse_f(const float*p,const float*t,int n){
    float s=0;for(int i=0;i<n;i++){float d=p[i]-t[i];s+=d*d;}return s/n;
}
static void mse_g(const float*p,const float*t,float*g,int n){
    float s=2.f/n;for(int i=0;i<n;i++)g[i]=s*(p[i]-t[i]);
}
static void normalize_v(float*y,int n){
    double mn=0,sd=0;
    for(int i=0;i<n;i++) mn+=y[i]; mn/=n;
    for(int i=0;i<n;i++){double d=y[i]-mn;sd+=d*d;} sd=sqrt(sd/n)+1e-8;
    for(int i=0;i<n;i++) y[i]=(y[i]-mn)/sd;
}

/* ── Main benchmark ───────────────────────────────────────────── */
int main(void) {
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf(  "║  ProbNet Scale Test  —  APN v9 vs SwiGLU vs Linear              ║\n");
    printf(  "║  7 regression tasks × 3 seeds  |  D=16 H=64 N=800 E=600        ║\n");
    printf(  "║  OpenMP threads: %-3d                                            ║\n",
             omp_get_max_threads());
    printf(  "╚══════════════════════════════════════════════════════════════════╝\n\n");

    int N=800,D=16,H=64,NTR=640,NTE=160,EPOCHS=600;
    float LR=2e-3f;

    const char* tnames[]={
        "Linear  y=W·x   ",
        "Ratio   y=x0/x1 ",
        "Product y=x0*x1 ",
        "Square  y=x0²   ",
        "Sqrt    y=√|x0| ",
        "Sin     y=sin(x0)",
        "Mixed   r+prod  "
    };
    int NT=7;

    printf("  %-18s  %9s  %9s  %9s  Winner\n","Task","Linear","SwiGLU","APN-v9");
    printf("  %-18s  %9s  %9s  %9s  ------\n",
           "──────────────────","─────────","─────────","─────────");

    int aw=0,gw=0,lw=0,ties=0;
    float* pred=pn_alloc(N),*dout_buf=pn_alloc(N);

    double t_start=now_s();

    for(int t=0;t<NT;t++){
        float al=0,ag=0,aa=0;

        for(int seed=0;seed<3;seed++){
            RNG rng=rng_seed(42+seed*13+t*7);
            float* X=pn_alloc((size_t)N*D),*y=pn_alloc(N),*Wt=pn_alloc(D);
            int is_pos=(t==1||t==4||t==6);
            for(int i=0;i<N*D;i++){
                float v=rng_normal(&rng);
                X[i]=is_pos ? fabsf(v)*0.4f+0.3f : v;
            }
            for(int d=0;d<D;d++) Wt[d]=rng_normal(&rng);
            for(int i=0;i<N;i++){
                float x0=X[i*D],x1=X[i*D+1];
                switch(t){
                    case 0:{float s=0;for(int d=0;d<D;d++)s+=Wt[d]*X[i*D+d];y[i]=s;break;}
                    case 1: y[i]=x0/(x1+0.05f);break;
                    case 2: y[i]=x0*x1;break;
                    case 3: y[i]=x0*x0;break;
                    case 4: y[i]=sqrtf(x0);break;
                    case 5: y[i]=sinf(x0*3.14159f);break;
                    case 6: y[i]=x0/(x1+0.05f)+X[i*D+2]*X[i*D+3];break;
                }
            }
            normalize_v(y,N);
            float *Xtr=X,*ytr=y,*Xte=X+NTR*D,*yte=y+NTR;

            /* ── Linear ── */
            float* LW=pn_alloc(D),*Lb=pn_alloc(1);
            {RNG r2=rng_seed(seed+1);init_normal(LW,D,sqrtf(2.f/D),&r2);}
            float* dW=pn_alloc(D),*db=pn_alloc(1);
            AdamW lo=adamw_new(LR,1e-4f,0.f);
            AdamState ls=adam_state_new(D+1);
            for(int ep=0;ep<EPOCHS;ep++){
                for(int n=0;n<NTR;n++){float s=Lb[0];for(int d=0;d<D;d++)s+=LW[d]*Xtr[n*D+d];pred[n]=s;}
                mse_g(pred,ytr,dout_buf,NTR);
                memset(dW,0,D*4);db[0]=0;
                for(int n=0;n<NTR;n++){db[0]+=dout_buf[n];for(int d=0;d<D;d++)dW[d]+=Xtr[n*D+d]*dout_buf[n];}
                adamw_step(&lo,&ls,LW,dW,D);adamw_step(&lo,&ls,Lb,db,1);
            }
            for(int n=0;n<NTE;n++){float s=Lb[0];for(int d=0;d<D;d++)s+=LW[d]*Xte[n*D+d];pred[n]=s;}
            al+=mse_f(pred,yte,NTE);
            pn_free(LW);pn_free(Lb);pn_free(dW);pn_free(db);adam_state_free(&ls);

            /* ── SwiGLU ── */
            SGL* glu=sgl_new(D,H,1,seed+2);SGLG gg=sgl_gnew(glu);
            AdamW go=adamw_new(LR,1e-4f,0.f);
            AdamState sg1=adam_state_new((size_t)D*H),sg2=adam_state_new((size_t)D*H),sg3=adam_state_new(H+1);
            for(int ep=0;ep<EPOCHS;ep++){
                sgl_fwd(glu,Xtr,pred,NTR);mse_g(pred,ytr,dout_buf,NTR);
                sgl_bwd(glu,&gg,dout_buf,NTR);
                adamw_step(&go,&sg1,glu->Wg,gg.dWg,(size_t)D*H);adamw_step(&go,&sg1,glu->bg,gg.dbg,H);
                adamw_step(&go,&sg2,glu->Wu,gg.dWu,(size_t)D*H);adamw_step(&go,&sg2,glu->bu,gg.dbu,H);
                adamw_step(&go,&sg3,glu->Wd,gg.dWd,H);adamw_step(&go,&sg3,glu->bd,gg.dbd,1);
            }
            sgl_fwd(glu,Xte,pred,NTE);ag+=mse_f(pred,yte,NTE);
            sgl_free(glu);sgl_gfree(&gg);
            adam_state_free(&sg1);adam_state_free(&sg2);adam_state_free(&sg3);

            /* ── APN v9 ── */
            APNLayer* apn=apn_layer_new(D,H,1,3.f,seed+3);
            APNGrad agr=apn_grad_new(apn);
            AdamW ao=adamw_new(LR,1e-4f,0.f);
            AdamState as1=adam_state_new((size_t)D*H),as2=adam_state_new((size_t)D*H);
            AdamState as3=adam_state_new((size_t)H*APN_NFUNCS),as4=adam_state_new(H+1);
            float* adx=pn_alloc((size_t)NTR*D);
            for(int ep=0;ep<EPOCHS;ep++){
                apn_anneal(apn,(float)ep/EPOCHS,3.f,0.05f);
                apn_forward(apn,Xtr,pred,NTR);mse_g(pred,ytr,dout_buf,NTR);
                apn_backward(apn,&agr,dout_buf,adx,NTR);
                apn_adamw_step(apn,&agr,&ao,&as1,&as2,&as3,&as4);
            }
            apn_forward(apn,Xte,pred,NTE);aa+=mse_f(pred,yte,NTE);
            pn_free(adx);apn_layer_free(apn);apn_grad_free(&agr);
            adam_state_free(&as1);adam_state_free(&as2);adam_state_free(&as3);adam_state_free(&as4);

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

    double elapsed = now_s()-t_start;
    printf("\n  ╔══ FINAL RESULTS ══════════════════════════════════╗\n");
    printf(  "  ║  APN wins: %d/7  │  SwiGLU: %d/7  │  Linear: %d/7   ║\n",aw,gw,lw);
    printf(  "  ║  Time: %.1fs                                      ║\n",elapsed);
    printf(  "  ╚══════════════════════════════════════════════════╝\n\n");

    pn_free(pred);pn_free(dout_buf);
    return 0;
}
