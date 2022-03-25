#include <petscvec_kokkos.hpp>
#include <petsc/private/pcimpl.h>
#include <petsc/private/kspimpl.h>
#include <petscksp.h>            /*I "petscksp.h" I*/
#include "petscsection.h"
#include <petscdmcomposite.h>
#include <Kokkos_Core.hpp>

typedef Kokkos::TeamPolicy<>::member_type team_member;

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp>

#define PCBJKOKKOS_SHARED_LEVEL 1
#define PCBJKOKKOS_VEC_SIZE 16
#define PCBJKOKKOS_TEAM_SIZE 16
#define PCBJKOKKOS_VERBOSE_LEVEL 0

typedef enum {BATCH_KSP_BICG_IDX,BATCH_KSP_TFQMR_IDX,BATCH_KSP_GMRES_IDX,NUM_BATCH_TYPES} KSPIndex;
typedef struct {
  Vec                                              vec_diag;
  PetscInt                                         nBlocks; /* total number of blocks */
  PetscInt                                         n; // cache host version of d_bid_eqOffset_k[nBlocks]
  KSP                                              ksp; // Used just for options. Should have one for each block
  Kokkos::View<PetscInt*, Kokkos::LayoutRight>     *d_bid_eqOffset_k;
  Kokkos::View<PetscScalar*, Kokkos::LayoutRight>  *d_idiag_k;
  Kokkos::View<PetscInt*>                          *d_isrow_k;
  Kokkos::View<PetscInt*>                          *d_isicol_k;
  KSPIndex                                         ksp_type_idx;
  PetscInt                                         nwork;
  PetscInt                                         const_block_size; // used to decide to use shared memory for work vectors
  PetscInt                                         *dm_Nf;  // Number of fields in each DM
  PetscInt                                         num_dms;
  // diagnostics
  PetscBool                                        reason;
  PetscBool                                        monitor;
  PetscInt                                         batch_target;
} PC_PCBJKOKKOS;

static PetscErrorCode  PCBJKOKKOSCreateKSP_BJKOKKOS(PC pc)
{
  const char    *prefix;
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS*)pc->data;
  DM             dm;

  PetscFunctionBegin;
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc),&jac->ksp));
  PetscCall(KSPSetErrorIfNotConverged(jac->ksp,pc->erroriffailure));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)jac->ksp,(PetscObject)pc,1));
  PetscCall(PCGetOptionsPrefix(pc,&prefix));
  PetscCall(KSPSetOptionsPrefix(jac->ksp,prefix));
  PetscCall(KSPAppendOptionsPrefix(jac->ksp,"pc_bjkokkos_"));
  PetscCall(PCGetDM(pc,&dm));
  if (dm) {
    PetscCall(KSPSetDM(jac->ksp, dm));
    PetscCall(KSPSetDMActive(jac->ksp, PETSC_FALSE));
  }
  jac->reason       = PETSC_FALSE;
  jac->monitor      = PETSC_FALSE;
  jac->batch_target = 0;
  PetscFunctionReturn(0);
}

// y <-- Ax
KOKKOS_INLINE_FUNCTION PetscErrorCode MatMult(const team_member team,  const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, const PetscInt start, const PetscInt end, const PetscScalar *x_loc, PetscScalar *y_loc)
{
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team,start,end), [=] (const int rowb) {
      int rowa = ic[rowb];
      int n = glb_Aai[rowa+1] - glb_Aai[rowa];
      const PetscInt    *aj  = glb_Aaj + glb_Aai[rowa];
      const PetscScalar *aa  = glb_Aaa + glb_Aai[rowa];
      PetscScalar sum;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange (team, n), [=] (const int i, PetscScalar& lsum) {
          lsum += aa[i] * x_loc[r[aj[i]]-start];
        }, sum);
      Kokkos::single(Kokkos::PerThread (team),[=]() {y_loc[rowb-start] = sum;});
    });
  team.team_barrier();
  return 0;
}

// temp buffer per thread with reduction at end?
KOKKOS_INLINE_FUNCTION PetscErrorCode MatMultTranspose(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, const PetscInt start, const PetscInt end, const PetscScalar *x_loc, PetscScalar *y_loc)
{
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team,end-start), [=] (int i) { y_loc[i] = 0;});
  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team,start,end), [=] (const int rowb) {
      int rowa = ic[rowb];
      int n = glb_Aai[rowa+1] - glb_Aai[rowa];
      const PetscInt    *aj  = glb_Aaj + glb_Aai[rowa];
      const PetscScalar *aa  = glb_Aaa + glb_Aai[rowa];
      const PetscScalar xx = x_loc[rowb-start]; // rowb = ic[rowa] = ic[r[rowb]]
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,n), [=] (const int &i) {
          PetscScalar val = aa[i] * xx;
          Kokkos::atomic_fetch_add(&y_loc[r[aj[i]]-start], val);
        });
    });
  team.team_barrier();
  return 0;
}

typedef struct Batch_MetaData_TAG
{
  PetscInt           flops;
  PetscInt           its;
  KSPConvergedReason reason;
}Batch_MetaData;

// Solve A(BB^-1)x = y with TFQMR. Right preconditioned to get un-preconditioned residual
KOKKOS_INLINE_FUNCTION PetscErrorCode BJSolve_TFQMR(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, PetscScalar *work_space, const PetscInt stride, PetscReal rtol, PetscReal atol, PetscReal dtol,PetscInt maxit, Batch_MetaData *metad, const PetscInt start, const PetscInt end, const PetscScalar glb_idiag[], const PetscScalar *glb_b, PetscScalar *glb_x, bool monitor)
{
  using Kokkos::parallel_reduce;
  using Kokkos::parallel_for;
  int               Nblk = end-start, i,m;
  PetscReal         dp,dpold,w,dpest,tau,psi,cm,r0;
  PetscScalar       *ptr = work_space, rho,rhoold,a,s,b,eta,etaold,psiold,cf,dpi;
  const PetscScalar *Diag = &glb_idiag[start];
  PetscScalar       *XX = ptr; ptr += stride;
  PetscScalar       *R = ptr; ptr += stride;
  PetscScalar       *RP = ptr; ptr += stride;
  PetscScalar       *V = ptr; ptr += stride;
  PetscScalar       *T = ptr; ptr += stride;
  PetscScalar       *Q = ptr; ptr += stride;
  PetscScalar       *P = ptr; ptr += stride;
  PetscScalar       *U = ptr; ptr += stride;
  PetscScalar       *D = ptr; ptr += stride;
  PetscScalar       *AUQ = V;

  // init: get b, zero x
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=] (int rowb) {
      int rowa = ic[rowb];
      R[rowb-start] = glb_b[rowa];
      XX[rowb-start] = 0;
    });
  team.team_barrier();
  parallel_reduce(Kokkos::TeamVectorRange (team, Nblk), [=] (const int idx, PetscScalar& lsum) {lsum += R[idx]*PetscConj(R[idx]);}, dpi);
  team.team_barrier();
  r0 = dp = PetscSqrtReal(PetscRealPart(dpi));
  // diagnostics
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
  if (monitor) Kokkos::single (Kokkos::PerTeam (team), [=] () { printf("%3d KSP Residual norm %14.12e \n", 0, (double)dp);});
#endif
  if (dp < atol) {metad->reason = KSP_CONVERGED_ATOL_NORMAL; return 0;}
  if (0 == maxit) {metad->reason = KSP_DIVERGED_ITS; return 0;}

  /* Make the initial Rp = R */
  parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {RP[idx] = R[idx];});
  team.team_barrier();
  /* Set the initial conditions */
  etaold = 0.0;
  psiold = 0.0;
  tau    = dp;
  dpold  = dp;

  /* rhoold = (r,rp)     */
  parallel_reduce(Kokkos::TeamVectorRange (team, Nblk), [=] (const int idx, PetscScalar& dot) {dot += R[idx]*PetscConj(RP[idx]);}, rhoold);
  team.team_barrier();
  parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {U[idx] = R[idx]; P[idx] = R[idx]; T[idx] = Diag[idx]*P[idx]; D[idx] = 0;});
  team.team_barrier();
  MatMult         (team,glb_Aai,glb_Aaj,glb_Aaa,r,ic,start,end,T,V);

  i=0;
  do {
    /* s <- (v,rp)          */
    parallel_reduce(Kokkos::TeamVectorRange (team, Nblk), [=] (const int idx, PetscScalar& dot) {dot += V[idx]*PetscConj(RP[idx]);}, s);
    team.team_barrier();
    a    = rhoold / s;                              /* a <- rho / s         */
    /* q <- u - a v    VecWAXPY(w,alpha,x,y): w = alpha x + y.     */
    /* t <- u + q           */
    parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Q[idx] = U[idx] - a*V[idx]; T[idx] = U[idx] + Q[idx];});
    team.team_barrier();
    // KSP_PCApplyBAorAB
    parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {T[idx] = Diag[idx]*T[idx]; });
    team.team_barrier();
    MatMult         (team,glb_Aai,glb_Aaj,glb_Aaa,r,ic,start,end,T,AUQ);
    /* r <- r - a K (u + q) */
    parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {R[idx] = R[idx] - a*AUQ[idx]; });
    team.team_barrier();
    parallel_reduce(Kokkos::TeamVectorRange (team, Nblk), [=] (const int idx, PetscScalar& lsum) {lsum += R[idx]*PetscConj(R[idx]);}, dpi);
    team.team_barrier();
    dp = PetscSqrtReal(PetscRealPart(dpi));
    for (m=0; m<2; m++) {
      if (!m) w = PetscSqrtReal(dp*dpold);
      else w = dp;
      psi = w / tau;
      cm  = 1.0 / PetscSqrtReal(1.0 + psi * psi);
      tau = tau * psi * cm;
      eta = cm * cm * a;
      cf  = psiold * psiold * etaold / a;
      if (!m) {
        /* D = U + cf D */
        parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {D[idx] = U[idx] + cf*D[idx]; });
      } else {
        /* D = Q + cf D */
        parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {D[idx] = Q[idx] + cf*D[idx]; });
      }
      team.team_barrier();
      parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {XX[idx] = XX[idx] + eta*D[idx]; });
      team.team_barrier();
      dpest = PetscSqrtReal(2*i + m + 2.0) * tau;
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
      if (monitor && m==1) Kokkos::single (Kokkos::PerTeam (team), [=] () { printf("%3d KSP Residual norm %14.12e \n", i+1, (double)dpest);});
#endif
      if (dpest < atol) {metad->reason = KSP_CONVERGED_ATOL_NORMAL; goto done;}
      if (dpest/r0 < rtol) {metad->reason = KSP_CONVERGED_RTOL_NORMAL; goto done;}
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
      if (dpest/r0 > dtol) {metad->reason = KSP_DIVERGED_DTOL; Kokkos::single (Kokkos::PerTeam (team), [=] () {printf("ERROR block %d diverged: %d it, res=%e, r_0=%e\n",team.league_rank(),i,dpest,r0);}); goto done;}
#else
      if (dpest/r0 > dtol) {metad->reason = KSP_DIVERGED_DTOL; goto done;}
#endif
      if (i+1 == maxit) {metad->reason = KSP_DIVERGED_ITS; goto done;}

      etaold = eta;
      psiold = psi;
    }

    /* rho <- (r,rp)       */
    parallel_reduce(Kokkos::TeamVectorRange (team, Nblk), [=] (const int idx, PetscScalar& dot) {dot += R[idx]*PetscConj(RP[idx]);}, rho);
    team.team_barrier();
    b    = rho / rhoold;                            /* b <- rho / rhoold   */
    /* u <- r + b q        */
    /* p <- u + b(q + b p) */
    parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {U[idx] = R[idx] + b*Q[idx]; Q[idx] = Q[idx] + b*P[idx]; P[idx] = U[idx] + b*Q[idx];});
    /* v <- K p  */
    team.team_barrier();
    parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {T[idx] = Diag[idx]*P[idx]; });
    team.team_barrier();
    MatMult         (team,glb_Aai,glb_Aaj,glb_Aaa,r,ic,start,end,T,V);

    rhoold = rho;
    dpold  = dp;

    i++;
  } while (i<maxit);
  done:
  // KSPUnwindPreconditioner
  parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {XX[idx] = Diag[idx]*XX[idx]; });
  team.team_barrier();
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=] (int rowb) {
      int rowa = ic[rowb];
      glb_x[rowa] = XX[rowb-start];
    });
  metad->its = i+1;
  if (1) {
    int nnz;
    parallel_reduce(Kokkos::TeamVectorRange (team, start, end), [=] (const int idx, int& lsum) {lsum += (glb_Aai[idx+1] - glb_Aai[idx]);}, nnz);
    metad->flops = 2*(metad->its*(10*Nblk + 2*nnz) + 5*Nblk);
  } else {
    metad->flops = 2*(metad->its*(10*Nblk + 2*50*Nblk) + 5*Nblk); // guess
  }
  return 0;
}

// Solve Ax = y with biCG
KOKKOS_INLINE_FUNCTION PetscErrorCode BJSolve_BICG(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, PetscScalar *work_space, const PetscInt stride, PetscReal rtol, PetscReal atol, PetscReal dtol,PetscInt maxit, Batch_MetaData *metad, const PetscInt start, const PetscInt end, const PetscScalar glb_idiag[], const PetscScalar *glb_b, PetscScalar *glb_x, bool monitor)
{
  using Kokkos::parallel_reduce;
  using Kokkos::parallel_for;
  int               Nblk = end-start, i;
  PetscReal         dp, r0;
  PetscScalar       *ptr = work_space, dpi, a=1.0, beta, betaold=1.0, b, b2, ma, mac;
  const PetscScalar *Di = &glb_idiag[start];
  PetscScalar       *XX = ptr; ptr += stride;
  PetscScalar       *Rl = ptr; ptr += stride;
  PetscScalar       *Zl = ptr; ptr += stride;
  PetscScalar       *Pl = ptr; ptr += stride;
  PetscScalar       *Rr = ptr; ptr += stride;
  PetscScalar       *Zr = ptr; ptr += stride;
  PetscScalar       *Pr = ptr; ptr += stride;

  /*     r <- b (x is 0) */
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=] (int rowb) {
      int rowa = ic[rowb];
      //PetscCall(VecCopy(Rr,Rl));
      Rl[rowb-start] = Rr[rowb-start] = glb_b[rowa];
      XX[rowb-start] = 0;
    });
  team.team_barrier();
  /*     z <- Br         */
  parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Zr[idx] = Di[idx]*Rr[idx]; Zl[idx] = Di[idx]*Rl[idx]; });
  team.team_barrier();
  /*    dp <- r'*r       */
  parallel_reduce(Kokkos::TeamVectorRange (team, Nblk), [=] (const int idx, PetscScalar& lsum) {lsum += Rr[idx]*PetscConj(Rr[idx]);}, dpi);
  team.team_barrier();
  r0 = dp = PetscSqrtReal(PetscRealPart(dpi));
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
  if (monitor) Kokkos::single (Kokkos::PerTeam (team), [=] () { printf("%3d KSP Residual norm %14.12e \n", 0, (double)dp);});
#endif
  if (dp < atol) {metad->reason = KSP_CONVERGED_ATOL_NORMAL; return 0;}
  if (0 == maxit) {metad->reason = KSP_DIVERGED_ITS; return 0;}
  i = 0;
  do {
    /*     beta <- r'z     */
    parallel_reduce(Kokkos::TeamVectorRange (team, Nblk), [=] (const int idx, PetscScalar& dot) {dot += Zr[idx]*PetscConj(Rl[idx]);}, beta);
    team.team_barrier();
#if PCBJKOKKOS_VERBOSE_LEVEL >= 6
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
    Kokkos::single (Kokkos::PerTeam (team), [=] () {printf("%7d beta = Z.R = %22.14e \n",i,(double)beta);});
#endif
#endif
    if (!i) {
      if (beta == 0.0) {
        metad->reason = KSP_DIVERGED_BREAKDOWN_BICG;
        goto done;
      }
      /*     p <- z          */
      parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Pr[idx] = Zr[idx]; Pl[idx] = Zl[idx];});
    } else {
      b    = beta/betaold;
      /*     p <- z + b* p   */
      b2    = PetscConj(b);
      parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Pr[idx] = b*Pr[idx] + Zr[idx]; Pl[idx] = b2*Pl[idx] + Zl[idx];});
    }
    team.team_barrier();
    betaold = beta;
    /*     z <- Kp         */
    MatMult         (team,glb_Aai,glb_Aaj,glb_Aaa,r,ic,start,end,Pr,Zr);
    MatMultTranspose(team,glb_Aai,glb_Aaj,glb_Aaa,r,ic,start,end,Pl,Zl);
    /*     dpi <- z'p      */
    parallel_reduce(Kokkos::TeamVectorRange (team, Nblk), [=] (const int idx, PetscScalar& lsum) {lsum += Zr[idx]*PetscConj(Pl[idx]);}, dpi);
    team.team_barrier();
    //
    a       = beta/dpi;                           /*     a = beta/p'z    */
    ma      = -a;
    mac      = PetscConj(ma);
    /*     x <- x + ap     */
    parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {XX[idx] = XX[idx] + a*Pr[idx]; Rr[idx] = Rr[idx] + ma*Zr[idx]; Rl[idx] = Rl[idx] + mac*Zl[idx];});team.team_barrier();
    team.team_barrier();
    /*    dp <- r'*r       */
    parallel_reduce(Kokkos::TeamVectorRange (team, Nblk), [=] (const int idx, PetscScalar& lsum) {lsum +=  Rr[idx]*PetscConj(Rr[idx]);}, dpi);
    team.team_barrier();
    dp = PetscSqrtReal(PetscRealPart(dpi));
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
    if (monitor) Kokkos::single (Kokkos::PerTeam (team), [=] () { printf("%3d KSP Residual norm %14.12e \n", i+1, (double)dp);});
#endif
    if (dp < atol) {metad->reason = KSP_CONVERGED_ATOL_NORMAL; goto done;}
    if (dp/r0 < rtol) {metad->reason = KSP_CONVERGED_RTOL_NORMAL; goto done;}
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
    if (dp/r0 > dtol) {metad->reason = KSP_DIVERGED_DTOL; Kokkos::single (Kokkos::PerTeam (team), [=] () {printf("ERROR block %d diverged: %d it, res=%e, r_0=%e\n",team.league_rank(),i,dp,r0);}); goto done;}
#else
    if (dp/r0 > dtol) {metad->reason = KSP_DIVERGED_DTOL; goto done;}
#endif
    if (i+1 == maxit) {metad->reason = KSP_DIVERGED_ITS; goto done;}
    /* z <- Br  */
    parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Zr[idx] = Di[idx]*Rr[idx]; Zl[idx] = Di[idx]*Rl[idx];});
    i++;
    team.team_barrier();
  } while (i<maxit);
 done:
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=] (int rowb) {
      int rowa = ic[rowb];
      glb_x[rowa] = XX[rowb-start];
    });
  metad->its = i+1;
  if (1) {
    int nnz;
    parallel_reduce(Kokkos::TeamVectorRange (team, start, end), [=] (const int idx, int& lsum) {lsum += (glb_Aai[idx+1] - glb_Aai[idx]);}, nnz);
    metad->flops = 2*(metad->its*(10*Nblk + 2*nnz) + 5*Nblk);
  } else {
    metad->flops = 2*(metad->its*(10*Nblk + 2*50*Nblk) + 5*Nblk); // guess
  }
  return 0;
}

// KSP solver solve Ax = b; x is output, bin is input
static PetscErrorCode PCApply_BJKOKKOS(PC pc,Vec bin,Vec xout)
{
  PC_PCBJKOKKOS    *jac = (PC_PCBJKOKKOS*)pc->data;
  Mat               A   = pc->pmat;
  Mat_SeqAIJKokkos *aijkok;

  PetscFunctionBegin;
  PetscCheck(jac->vec_diag && A,PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"Not setup???? %p %p",jac->vec_diag,A);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  if (!aijkok) {
    SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"No aijkok");
  } else {
    using scr_mem_t  = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using vect2D_scr_t = Kokkos::View<PetscScalar**, Kokkos::LayoutLeft, scr_mem_t>;
    PetscInt          *d_bid_eqOffset, maxit = jac->ksp->max_it, scr_bytes_team, stride, global_buff_size;
    const PetscInt    conc = Kokkos::DefaultExecutionSpace().concurrency(), openmp = !!(conc < 1000), team_size = (openmp==0 && PCBJKOKKOS_VEC_SIZE != 1) ? PCBJKOKKOS_TEAM_SIZE : 1;
    const PetscInt    nwork = jac->nwork, nBlk = jac->nBlocks;
    PetscScalar       *glb_xdata=NULL;
    PetscReal         rtol = jac->ksp->rtol, atol = jac->ksp->abstol, dtol = jac->ksp->divtol;
    const PetscScalar *glb_idiag =jac->d_idiag_k->data(), *glb_bdata=NULL;
    const PetscInt    *glb_Aai = aijkok->i_device_data(), *glb_Aaj = aijkok->j_device_data();
    const PetscScalar *glb_Aaa = aijkok->a_device_data();
    Kokkos::View<Batch_MetaData*, Kokkos::DefaultExecutionSpace> d_metadata("solver meta data", nBlk);
    PCFailedReason    pcreason;
    KSPIndex          ksp_type_idx = jac->ksp_type_idx;
    PetscMemType      mtype;
    PetscContainer    container;
    PetscInt          batch_sz;
    VecScatter        plex_batch=NULL;
    Vec               bvec;
    PetscBool         monitor = jac->monitor; // captured
    PetscInt          view_bid = jac->batch_target;
    // get field major is to map plex IO to/from block/field major
    PetscCall(PetscObjectQuery((PetscObject) A, "plex_batch_is", (PetscObject *) &container));
    PetscCall(VecDuplicate(bin,&bvec));
    if (container) {
      PetscCall(PetscContainerGetPointer(container, (void **) &plex_batch));
      PetscCall(VecScatterBegin(plex_batch,bin,bvec,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(plex_batch,bin,bvec,INSERT_VALUES,SCATTER_FORWARD));
    } else {
      PetscCall(VecCopy(bin, bvec));
    }
    // get x
    PetscCall(VecGetArrayAndMemType(xout,&glb_xdata,&mtype));
#if defined(PETSC_HAVE_CUDA)
    PetscCheck(PetscMemTypeDevice(mtype),PetscObjectComm((PetscObject) pc),PETSC_ERR_ARG_WRONG,"No GPU data for x %" PetscInt_FMT " != %" PetscInt_FMT "",mtype,PETSC_MEMTYPE_DEVICE);
#endif
    PetscCall(VecGetArrayReadAndMemType(bvec,&glb_bdata,&mtype));
#if defined(PETSC_HAVE_CUDA)
    PetscCheck(PetscMemTypeDevice(mtype),PetscObjectComm((PetscObject) pc),PETSC_ERR_ARG_WRONG,"No GPU data for b");
#endif
    // get batch size
    PetscCall(PetscObjectQuery((PetscObject) A, "batch size", (PetscObject *) &container));
    if (container) {
      PetscInt *pNf=NULL;
      PetscCall(PetscContainerGetPointer(container, (void **) &pNf));
      batch_sz = *pNf;
    } else batch_sz = 1;
    PetscCheck(nBlk%batch_sz == 0,PetscObjectComm((PetscObject) pc),PETSC_ERR_ARG_WRONG,"batch_sz = %" PetscInt_FMT ", nBlk = %" PetscInt_FMT,batch_sz,nBlk);
    d_bid_eqOffset = jac->d_bid_eqOffset_k->data();
    // solve each block independently
    if (jac->const_block_size) { // use shared memory for work vectors only if constant block size - todo: test efficiency loss
      scr_bytes_team = jac->const_block_size*nwork*sizeof(PetscScalar);
      stride = jac->const_block_size; // captured
      global_buff_size = 0;
    } else {
      scr_bytes_team = 0;
      stride = jac->n; // captured
      global_buff_size = jac->n*nwork;
    }
    Kokkos::View<PetscScalar*, Kokkos::DefaultExecutionSpace> d_work_vecs_k("workvectors", global_buff_size); // global work vectors
    PetscInfo(pc,"\tn = %" PetscInt_FMT ". %d shared mem words/team. %" PetscInt_FMT " global mem words, rtol=%e, num blocks %" PetscInt_FMT ", team_size=%" PetscInt_FMT ", %" PetscInt_FMT " vector threads\n",jac->n,scr_bytes_team/sizeof(PetscScalar),global_buff_size,rtol,nBlk,
               team_size, PCBJKOKKOS_VEC_SIZE);
    PetscScalar  *d_work_vecs = scr_bytes_team ? NULL : d_work_vecs_k.data();
    const PetscInt *d_isicol = jac->d_isicol_k->data(), *d_isrow = jac->d_isrow_k->data();
    Kokkos::parallel_for("Solve", Kokkos::TeamPolicy<>(nBlk, team_size, PCBJKOKKOS_VEC_SIZE).set_scratch_size(PCBJKOKKOS_SHARED_LEVEL, Kokkos::PerTeam(scr_bytes_team)),
        KOKKOS_LAMBDA (const team_member team) {
        const int    blkID = team.league_rank(), start = d_bid_eqOffset[blkID], end = d_bid_eqOffset[blkID+1];
        vect2D_scr_t work_vecs(team.team_scratch(PCBJKOKKOS_SHARED_LEVEL), scr_bytes_team ? (end-start) : 0, nwork);
        PetscScalar *work_buff = (scr_bytes_team) ? work_vecs.data() : &d_work_vecs[start];
        bool        print = monitor && (blkID==view_bid);
        switch (ksp_type_idx) {
        case BATCH_KSP_BICG_IDX:
          BJSolve_BICG(team, glb_Aai, glb_Aaj, glb_Aaa, d_isrow, d_isicol, work_buff, stride, rtol, atol, dtol, maxit, &d_metadata[blkID], start, end, glb_idiag, glb_bdata, glb_xdata, print);
          break;
        case BATCH_KSP_TFQMR_IDX:
          BJSolve_TFQMR(team, glb_Aai, glb_Aaj, glb_Aaa, d_isrow, d_isicol, work_buff, stride, rtol, atol, dtol, maxit, &d_metadata[blkID], start, end, glb_idiag, glb_bdata, glb_xdata, print);
          break;
        case BATCH_KSP_GMRES_IDX:
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
          printf("GMRES not implemented %d\n",ksp_type_idx);
#else
          /* void */
#endif
          break;
        default:
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
          printf("Unknown KSP type %d\n",ksp_type_idx);
#else
          /* void */;
#endif
        }
    });
    auto h_metadata = Kokkos::create_mirror(Kokkos::HostSpace::memory_space(), d_metadata);
    Kokkos::fence();
    Kokkos::deep_copy (h_metadata, d_metadata);
#if PCBJKOKKOS_VERBOSE_LEVEL >= 3
#if PCBJKOKKOS_VERBOSE_LEVEL >= 4
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Iterations\n"));
#endif
    // assume species major
#if PCBJKOKKOS_VERBOSE_LEVEL < 4
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"max iterations per species (%s) :",ksp_type_idx==BATCH_KSP_BICG_IDX ? "bicg" : "tfqmr"));
#endif
    for (PetscInt dmIdx=0, s=0, head=0 ; dmIdx < jac->num_dms; dmIdx += batch_sz) {
      for (PetscInt f=0, idx=head ; f < jac->dm_Nf[dmIdx] ; f++,s++,idx++) {
#if PCBJKOKKOS_VERBOSE_LEVEL >= 4
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%2D:", s));
        for (int bid=0 ; bid<batch_sz ; bid++) {
         PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%3D ", h_metadata[idx + bid*jac->dm_Nf[dmIdx]].its));
        }
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n"));
#else
        PetscInt count=0;
        for (int bid=0 ; bid<batch_sz ; bid++) {
          if (h_metadata[idx + bid*jac->dm_Nf[dmIdx]].its > count) count = h_metadata[idx + bid*jac->dm_Nf[dmIdx]].its;
        }
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%3D ", count));
#endif
      }
      head += batch_sz*jac->dm_Nf[dmIdx];
    }
#if PCBJKOKKOS_VERBOSE_LEVEL < 4
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n"));
#endif
#endif
    PetscInt count=0, mbid=0;
    for (int blkID=0;blkID<nBlk;blkID++) {
      PetscCall(PetscLogGpuFlops((PetscLogDouble)h_metadata[blkID].flops));
      if (jac->reason) {
        if (jac->batch_target==blkID) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF,  "    Linear solve converged due to %s iterations %d, batch %" PetscInt_FMT ", species %" PetscInt_FMT "\n", KSPConvergedReasons[h_metadata[blkID].reason], h_metadata[blkID].its, blkID%batch_sz, blkID/batch_sz));
        } else if (jac->batch_target==-1 && h_metadata[blkID].its > count) {
          count = h_metadata[blkID].its;
          mbid = blkID;
        }
        if (h_metadata[blkID].reason < 0) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "ERROR reason=%s, its=%" PetscInt_FMT ". species %" PetscInt_FMT ", batch %" PetscInt_FMT "\n",
                              KSPConvergedReasons[h_metadata[blkID].reason],h_metadata[blkID].its,blkID/batch_sz,blkID%batch_sz));
        }
      }
    }
    if (jac->batch_target==-1 && jac->reason) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,  "    Linear solve converged due to %s iterations %d, batch %" PetscInt_FMT ", specie %" PetscInt_FMT "\n", KSPConvergedReasons[h_metadata[mbid].reason], h_metadata[mbid].its,mbid%batch_sz,mbid/batch_sz));
    }
    PetscCall(VecRestoreArrayAndMemType(xout,&glb_xdata));
    PetscCall(VecRestoreArrayReadAndMemType(bvec,&glb_bdata));
    {
      int errsum;
      Kokkos::parallel_reduce(nBlk, KOKKOS_LAMBDA (const int idx, int& lsum) {
          if (d_metadata[idx].reason < 0) ++lsum;
        }, errsum);
      pcreason = errsum ? PC_SUBPC_ERROR : PC_NOERROR;
    }
    PetscCall(PCSetFailedReason(pc,pcreason));
    // map back to Plex space
    if (plex_batch) {
      PetscCall(VecCopy(xout, bvec));
      PetscCall(VecScatterBegin(plex_batch,bvec,xout,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(plex_batch,bvec,xout,INSERT_VALUES,SCATTER_REVERSE));
    }
    PetscCall(VecDestroy(&bvec));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_BJKOKKOS(PC pc)
{
  PC_PCBJKOKKOS    *jac = (PC_PCBJKOKKOS*)pc->data;
  Mat               A   = pc->pmat;
  Mat_SeqAIJKokkos *aijkok;
  PetscBool         flg;

  PetscFunctionBegin;
  PetscCheck(!pc->useAmat,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"No support for using 'use_amat'");
  PetscCheck(A,PetscObjectComm((PetscObject)A),PETSC_ERR_USER,"No matrix - A is used above");
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A,&flg,MATSEQAIJKOKKOS,MATMPIAIJKOKKOS,MATAIJKOKKOS,""));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"must use '-dm_mat_type aijkokkos -dm_vec_type kokkos' for -pc_type bjkokkos");
  if (!(aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr))) {
    SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_USER,"No aijkok");
  } else {
    if (!jac->vec_diag) {
      Vec               *subX;
      DM                pack,*subDM;
      PetscInt          nDMs, n;
      PetscContainer    container;
      PetscCall(PetscObjectQuery((PetscObject) A, "plex_batch_is", (PetscObject *) &container));
      { // Permute the matrix to get a block diagonal system: d_isrow_k, d_isicol_k
        MatOrderingType   rtype;
        IS                isrow,isicol;
        const PetscInt    *rowindices,*icolindices;

        if (container) rtype = MATORDERINGNATURAL; // if we have a vecscatter then don't reorder here (all the reorder stuff goes away in future)
        else rtype = MATORDERINGRCM;
        // get permutation. Not what I expect so inverted here
        PetscCall(MatGetOrdering(A,rtype,&isrow,&isicol));
        PetscCall(ISDestroy(&isrow));
        PetscCall(ISInvertPermutation(isicol,PETSC_DECIDE,&isrow));
        PetscCall(ISGetIndices(isrow,&rowindices));
        PetscCall(ISGetIndices(isicol,&icolindices));
        const Kokkos::View<PetscInt*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_isrow_k((PetscInt*)rowindices,A->rmap->n);
        const Kokkos::View<PetscInt*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_isicol_k ((PetscInt*)icolindices,A->rmap->n);
        jac->d_isrow_k = new Kokkos::View<PetscInt*>(Kokkos::create_mirror(DefaultMemorySpace(),h_isrow_k));
        jac->d_isicol_k = new Kokkos::View<PetscInt*>(Kokkos::create_mirror(DefaultMemorySpace(),h_isicol_k));
        Kokkos::deep_copy (*jac->d_isrow_k, h_isrow_k);
        Kokkos::deep_copy (*jac->d_isicol_k, h_isicol_k);
        PetscCall(ISRestoreIndices(isrow,&rowindices));
        PetscCall(ISRestoreIndices(isicol,&icolindices));
        PetscCall(ISDestroy(&isrow));
        PetscCall(ISDestroy(&isicol));
      }
      // get block sizes
      PetscCall(PCGetDM(pc, &pack));
      PetscCheck(pack,PetscObjectComm((PetscObject)A),PETSC_ERR_USER,"no DM. Requires a composite DM");
      PetscCall(PetscObjectTypeCompare((PetscObject)pack,DMCOMPOSITE,&flg));
      PetscCheck(flg,PetscObjectComm((PetscObject)pack),PETSC_ERR_USER,"Not for type %s",((PetscObject)pack)->type_name);
      PetscCall(DMCompositeGetNumberDM(pack,&nDMs));
      jac->num_dms = nDMs;
      PetscCall(DMCreateGlobalVector(pack, &jac->vec_diag));
      PetscCall(VecGetLocalSize(jac->vec_diag,&n));
      jac->n = n;
      jac->d_idiag_k = new Kokkos::View<PetscScalar*, Kokkos::LayoutRight>("idiag", n);
      // options
      PetscCall(PCBJKOKKOSCreateKSP_BJKOKKOS(pc));
      PetscCall(KSPSetFromOptions(jac->ksp));
      PetscCall(PetscObjectTypeCompareAny((PetscObject)jac->ksp,&flg,KSPBICG,""));
      if (flg) {jac->ksp_type_idx = BATCH_KSP_BICG_IDX; jac->nwork = 7;}
      else {
        PetscCall(PetscObjectTypeCompareAny((PetscObject)jac->ksp,&flg,KSPTFQMR,""));
        if (flg) {jac->ksp_type_idx = BATCH_KSP_TFQMR_IDX; jac->nwork = 10;}
        else {
          PetscCall(PetscObjectTypeCompareAny((PetscObject)jac->ksp,&flg,KSPGMRES,""));
          if (flg) {jac->ksp_type_idx = BATCH_KSP_GMRES_IDX; jac->nwork = 0;}
          SETERRQ(PetscObjectComm((PetscObject)jac->ksp),PETSC_ERR_ARG_WRONG,"unsupported type %s", ((PetscObject)jac->ksp)->type_name);
        }
      }
      {
        PetscViewer       viewer;
        PetscBool         flg;
        PetscViewerFormat format;
        PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)jac->ksp),((PetscObject)jac->ksp)->options,((PetscObject)jac->ksp)->prefix,"-ksp_converged_reason",&viewer,&format,&flg));
        jac->reason = flg;
        PetscCall(PetscViewerDestroy(&viewer));
        PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)jac->ksp),((PetscObject)jac->ksp)->options,((PetscObject)jac->ksp)->prefix,"-ksp_monitor",&viewer,&format,&flg));
        jac->monitor = flg;
        PetscCall(PetscViewerDestroy(&viewer));
        PetscCall(PetscOptionsGetInt(((PetscObject)jac->ksp)->options,((PetscObject)jac->ksp)->prefix,"-ksp_batch_target",&jac->batch_target,&flg));
        PetscCheck(jac->batch_target < jac->num_dms,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"-ksp_batch_target (%" PetscInt_FMT ") >= number of DMs (%" PetscInt_FMT ")",jac->batch_target,jac->num_dms);
        if (!jac->monitor && !flg) jac->batch_target = -1; // turn it off
      }
      // get blocks - jac->d_bid_eqOffset_k
      PetscCall(PetscMalloc(sizeof(*subX)*nDMs, &subX));
      PetscCall(PetscMalloc(sizeof(*subDM)*nDMs, &subDM));
      PetscCall(PetscMalloc(sizeof(*jac->dm_Nf)*nDMs, &jac->dm_Nf));
      PetscCall(PetscInfo(pc, "Have %" PetscInt_FMT " DMs, n=%" PetscInt_FMT " rtol=%g type = %s\n", nDMs, n, jac->ksp->rtol, ((PetscObject)jac->ksp)->type_name));
      PetscCall(DMCompositeGetEntriesArray(pack,subDM));
      jac->nBlocks = 0;
      for (PetscInt ii=0;ii<nDMs;ii++) {
        PetscSection section;
        PetscInt Nf;
        DM dm = subDM[ii];
        PetscCall(DMGetLocalSection(dm, &section));
        PetscCall(PetscSectionGetNumFields(section, &Nf));
        jac->nBlocks += Nf;
#if PCBJKOKKOS_VERBOSE_LEVEL <= 2
        if (ii==0) PetscCall(PetscInfo(pc,"%" PetscInt_FMT ") %" PetscInt_FMT " blocks (%" PetscInt_FMT " total)\n",ii,Nf,jac->nBlocks));
#else
        PetscCall(PetscInfo(pc,"%" PetscInt_FMT ") %" PetscInt_FMT " blocks (%" PetscInt_FMT " total)\n",ii,Nf,jac->nBlocks));
#endif
        jac->dm_Nf[ii] = Nf;
      }
      { // d_bid_eqOffset_k
        Kokkos::View<PetscInt*, Kokkos::LayoutRight, Kokkos::HostSpace> h_block_offsets("block_offsets", jac->nBlocks+1);
        PetscCall(DMCompositeGetAccessArray(pack, jac->vec_diag, nDMs, NULL, subX));
        h_block_offsets[0] = 0;
        jac->const_block_size = -1;
        for (PetscInt ii=0, idx = 0;ii<nDMs;ii++) {
          PetscInt nloc,nblk;
          PetscCall(VecGetSize(subX[ii],&nloc));
          nblk = nloc/jac->dm_Nf[ii];
          PetscCheck(nloc%jac->dm_Nf[ii] == 0,PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"nloc%jac->dm_Nf[ii] DMs",nloc,jac->dm_Nf[ii]);
          for (PetscInt jj=0;jj<jac->dm_Nf[ii];jj++, idx++) {
            h_block_offsets[idx+1] = h_block_offsets[idx] + nblk;
#if PCBJKOKKOS_VERBOSE_LEVEL <= 2
            if (idx==0) PetscCall(PetscInfo(pc,"\t%" PetscInt_FMT ") Add block with %" PetscInt_FMT " equations of %" PetscInt_FMT "\n",idx+1,nblk,jac->nBlocks));
#else
            PetscCall(PetscInfo(pc,"\t%" PetscInt_FMT ") Add block with %" PetscInt_FMT " equations of %" PetscInt_FMT "\n",idx+1,nblk,jac->nBlocks));
#endif
            if (jac->const_block_size == -1) jac->const_block_size = nblk;
            else if (jac->const_block_size > 0 && jac->const_block_size != nblk) jac->const_block_size = 0;
          }
        }
        PetscCall(DMCompositeRestoreAccessArray(pack, jac->vec_diag, jac->nBlocks, NULL, subX));
        PetscCall(PetscFree(subX));
        PetscCall(PetscFree(subDM));
        jac->d_bid_eqOffset_k = new Kokkos::View<PetscInt*, Kokkos::LayoutRight>(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(),h_block_offsets));
        Kokkos::deep_copy (*jac->d_bid_eqOffset_k, h_block_offsets);
      }
    }
    { // get jac->d_idiag_k (PC setup),
      const PetscInt    *d_ai=aijkok->i_device_data(), *d_aj=aijkok->j_device_data();
      const PetscScalar *d_aa = aijkok->a_device_data();
      const PetscInt    conc = Kokkos::DefaultExecutionSpace().concurrency(), openmp = !!(conc < 1000), team_size = (openmp==0 && PCBJKOKKOS_VEC_SIZE != 1) ? PCBJKOKKOS_TEAM_SIZE : 1;
      PetscInt          *d_bid_eqOffset = jac->d_bid_eqOffset_k->data(), *r = jac->d_isrow_k->data(), *ic = jac->d_isicol_k->data();
      PetscScalar       *d_idiag = jac->d_idiag_k->data();
      Kokkos::parallel_for("Diag", Kokkos::TeamPolicy<>(jac->nBlocks, team_size, PCBJKOKKOS_VEC_SIZE), KOKKOS_LAMBDA (const team_member team) {
          const PetscInt blkID = team.league_rank();
          Kokkos::parallel_for
            (Kokkos::TeamThreadRange(team,d_bid_eqOffset[blkID],d_bid_eqOffset[blkID+1]),
             [=] (const int rowb) {
               const PetscInt    rowa = ic[rowb], ai = d_ai[rowa], *aj = d_aj + ai; // grab original data
               const PetscScalar *aa  = d_aa + ai;
               const PetscInt    nrow = d_ai[rowa + 1] - ai;
               int found;
               Kokkos::parallel_reduce
                 (Kokkos::ThreadVectorRange (team, nrow),
                  [=] (const int& j, int &count) {
                    const PetscInt colb = r[aj[j]];
                    if (colb==rowb) {
                      d_idiag[rowb] = 1./aa[j];
                      count++;
                    }}, found);
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
               if (found!=1) Kokkos::single (Kokkos::PerThread (team), [=] () {printf("ERRORrow %d) found = %d\n",rowb,found);});
#endif
             });
        });
    }
  }
  PetscFunctionReturn(0);
}

/* Default destroy, if it has never been setup */
static PetscErrorCode PCReset_BJKOKKOS(PC pc)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS*)pc->data;

  PetscFunctionBegin;
  PetscCall(KSPDestroy(&jac->ksp));
  PetscCall(VecDestroy(&jac->vec_diag));
  if (jac->d_bid_eqOffset_k) delete jac->d_bid_eqOffset_k;
  if (jac->d_idiag_k) delete jac->d_idiag_k;
  if (jac->d_isrow_k) delete jac->d_isrow_k;
  if (jac->d_isicol_k) delete jac->d_isicol_k;
  jac->d_bid_eqOffset_k = NULL;
  jac->d_idiag_k = NULL;
  jac->d_isrow_k = NULL;
  jac->d_isicol_k = NULL;
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBJKOKKOSGetKSP_C",NULL)); // not published now (causes configure errors)
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBJKOKKOSSetKSP_C",NULL));
  PetscCall(PetscFree(jac->dm_Nf));
  jac->dm_Nf = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_BJKOKKOS(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PCReset_BJKOKKOS(pc));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_BJKOKKOS(PC pc,PetscViewer viewer)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS*)pc->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  if (!jac->ksp) PetscCall(PCBJKOKKOSCreateKSP_BJKOKKOS(pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Batched device linear solver: Krylov (KSP) method with Jacobi preconditioning\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer,"\t\tnwork = %" PetscInt_FMT ", rel tol = %e, abs tol = %e, div tol = %e, max it =%" PetscInt_FMT ", type = %s\n",jac->nwork,jac->ksp->rtol,
                                   jac->ksp->abstol, jac->ksp->divtol, jac->ksp->max_it,
                                   ((PetscObject)jac->ksp)->type_name));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_BJKOKKOS(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"PC BJKOKKOS options"));
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCBJKOKKOSSetKSP_BJKOKKOS(PC pc,KSP ksp)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS*)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)ksp));
  PetscCall(KSPDestroy(&jac->ksp));
  jac->ksp = ksp;
  PetscFunctionReturn(0);
}

/*@C
   PCBJKOKKOSSetKSP - Sets the KSP context for a KSP PC.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  ksp - the KSP solver

   Notes:
   The PC and the KSP must have the same communicator

   Level: advanced

@*/
PetscErrorCode  PCBJKOKKOSSetKSP(PC pc,KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  PetscCheckSameComm(pc,1,ksp,2);
  PetscCall(PetscTryMethod(pc,"PCBJKOKKOSSetKSP_C",(PC,KSP),(pc,ksp)));
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCBJKOKKOSGetKSP_BJKOKKOS(PC pc,KSP *ksp)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS*)pc->data;

  PetscFunctionBegin;
  if (!jac->ksp) PetscCall(PCBJKOKKOSCreateKSP_BJKOKKOS(pc));
  *ksp = jac->ksp;
  PetscFunctionReturn(0);
}

/*@C
   PCBJKOKKOSGetKSP - Gets the KSP context for a KSP PC.

   Not Collective but KSP returned is parallel if PC was parallel

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
.  ksp - the KSP solver

   Notes:
   You must call KSPSetUp() before calling PCBJKOKKOSGetKSP().

   If the PC is not a PCBJKOKKOS object it raises an error

   Level: advanced

@*/
PetscErrorCode  PCBJKOKKOSGetKSP(PC pc,KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(ksp,2);
  PetscCall(PetscUseMethod(pc,"PCBJKOKKOSGetKSP_C",(PC,KSP*),(pc,ksp)));
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------*/

/*MC
     PCBJKOKKOS -  Defines a preconditioner that applies a Krylov solver and preconditioner to the blocks in a AIJASeq matrix on the GPU.

   Options Database Key:
.     -pc_bjkokkos_ - options prefix with ksp options

   Level: intermediate

   Notes:
    For use with -ksp_type preonly to bypass any CPU work

   Developer Notes:

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCSHELL, PCCOMPOSITE, PCSetUseAmat(), PCBJKOKKOSGetKSP()

M*/

PETSC_EXTERN PetscErrorCode PCCreate_BJKOKKOS(PC pc)
{
  PC_PCBJKOKKOS *jac;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(pc,&jac));
  pc->data = (void*)jac;

  jac->ksp              = NULL;
  jac->vec_diag         = NULL;
  jac->d_bid_eqOffset_k = NULL;
  jac->d_idiag_k        = NULL;
  jac->d_isrow_k        = NULL;
  jac->d_isicol_k       = NULL;
  jac->nBlocks          = 1;

  PetscCall(PetscMemzero(pc->ops,sizeof(struct _PCOps)));
  pc->ops->apply           = PCApply_BJKOKKOS;
  pc->ops->applytranspose  = NULL;
  pc->ops->setup           = PCSetUp_BJKOKKOS;
  pc->ops->reset           = PCReset_BJKOKKOS;
  pc->ops->destroy         = PCDestroy_BJKOKKOS;
  pc->ops->setfromoptions  = PCSetFromOptions_BJKOKKOS;
  pc->ops->view            = PCView_BJKOKKOS;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBJKOKKOSGetKSP_C",PCBJKOKKOSGetKSP_BJKOKKOS));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBJKOKKOSSetKSP_C",PCBJKOKKOSSetKSP_BJKOKKOS));
  PetscFunctionReturn(0);
}
