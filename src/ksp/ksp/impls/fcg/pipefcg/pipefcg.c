/*
    Contributed by Patrick Sanan and Sascha M. Schnepp
*/

#include <../src/ksp/ksp/impls/fcg/pipefcg/pipefcgimpl.h>       /*I  "petscksp.h"  I*/

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
  "@article{SSM2016,\n"
  "  author = {P. Sanan and S.M. Schnepp and D.A. May},\n"
  "  title = {Pipelined, Flexible Krylov Subspace Methods},\n"
  "  journal = {SIAM Journal on Scientific Computing},\n"
  "  volume = {38},\n"
  "  number = {5},\n"
  "  pages = {C441-C470},\n"
  "  year = {2016},\n"
  "  doi = {10.1137/15M1049130},\n"
  "  URL = {http://dx.doi.org/10.1137/15M1049130},\n"
  "  eprint = {http://dx.doi.org/10.1137/15M1049130}\n"
  "}\n";

#define KSPPIPEFCG_DEFAULT_MMAX 15
#define KSPPIPEFCG_DEFAULT_NPREALLOC 5
#define KSPPIPEFCG_DEFAULT_VECB 5
#define KSPPIPEFCG_DEFAULT_TRUNCSTRAT KSP_FCD_TRUNC_TYPE_NOTAY

static PetscErrorCode KSPAllocateVectors_PIPEFCG(KSP ksp, PetscInt nvecsneeded, PetscInt chunksize)
{
  PetscErrorCode  ierr;
  PetscInt        i;
  KSP_PIPEFCG     *pipefcg;
  PetscInt        nnewvecs, nvecsprev;

  PetscFunctionBegin;
  pipefcg = (KSP_PIPEFCG*)ksp->data;

  /* Allocate enough new vectors to add chunksize new vectors, reach nvecsneedtotal, or to reach mmax+1, whichever is smallest */
  if (pipefcg->nvecs < PetscMin(pipefcg->mmax+1,nvecsneeded)) {
    nvecsprev = pipefcg->nvecs;
    nnewvecs = PetscMin(PetscMax(nvecsneeded-pipefcg->nvecs,chunksize),pipefcg->mmax+1-pipefcg->nvecs);
    ierr = KSPCreateVecs(ksp,nnewvecs,&pipefcg->pQvecs[pipefcg->nchunks],0,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents((PetscObject)ksp,nnewvecs,pipefcg->pQvecs[pipefcg->nchunks]);CHKERRQ(ierr);
    ierr = KSPCreateVecs(ksp,nnewvecs,&pipefcg->pZETAvecs[pipefcg->nchunks],0,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents((PetscObject)ksp,nnewvecs,pipefcg->pZETAvecs[pipefcg->nchunks]);CHKERRQ(ierr);
    ierr = KSPCreateVecs(ksp,nnewvecs,&pipefcg->pPvecs[pipefcg->nchunks],0,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents((PetscObject)ksp,nnewvecs,pipefcg->pPvecs[pipefcg->nchunks]);CHKERRQ(ierr);
    ierr = KSPCreateVecs(ksp,nnewvecs,&pipefcg->pSvecs[pipefcg->nchunks],0,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents((PetscObject)ksp,nnewvecs,pipefcg->pSvecs[pipefcg->nchunks]);CHKERRQ(ierr);
    pipefcg->nvecs += nnewvecs;
    for (i=0;i<nnewvecs;++i) {
      pipefcg->Qvecs[nvecsprev + i]    = pipefcg->pQvecs[pipefcg->nchunks][i];
      pipefcg->ZETAvecs[nvecsprev + i] = pipefcg->pZETAvecs[pipefcg->nchunks][i];
      pipefcg->Pvecs[nvecsprev + i]    = pipefcg->pPvecs[pipefcg->nchunks][i];
      pipefcg->Svecs[nvecsprev + i]    = pipefcg->pSvecs[pipefcg->nchunks][i];
    }
    pipefcg->chunksizes[pipefcg->nchunks] = nnewvecs;
    ++pipefcg->nchunks;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetUp_PIPEFCG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_PIPEFCG    *pipefcg;
  const PetscInt nworkstd = 5;

  PetscFunctionBegin;
  pipefcg = (KSP_PIPEFCG*)ksp->data;

  /* Allocate "standard" work vectors (not including the basis and transformed basis vectors) */
  ierr = KSPSetWorkVecs(ksp,nworkstd);CHKERRQ(ierr);

  /* Allocated space for pointers to additional work vectors
   note that mmax is the number of previous directions, so we add 1 for the current direction,
   and an extra 1 for the prealloc (which might be empty) */
  ierr = PetscMalloc4(pipefcg->mmax+1,&(pipefcg->Pvecs),pipefcg->mmax+1,&(pipefcg->pPvecs),pipefcg->mmax+1,&(pipefcg->Svecs),pipefcg->mmax+1,&(pipefcg->pSvecs));CHKERRQ(ierr);
  ierr = PetscMalloc4(pipefcg->mmax+1,&(pipefcg->Qvecs),pipefcg->mmax+1,&(pipefcg->pQvecs),pipefcg->mmax+1,&(pipefcg->ZETAvecs),pipefcg->mmax+1,&(pipefcg->pZETAvecs));CHKERRQ(ierr);
  ierr = PetscMalloc4(pipefcg->mmax+1,&(pipefcg->Pold),pipefcg->mmax+1,&(pipefcg->Sold),pipefcg->mmax+1,&(pipefcg->Qold),pipefcg->mmax+1,&(pipefcg->ZETAold));CHKERRQ(ierr);
  ierr = PetscMalloc1(pipefcg->mmax+1,&(pipefcg->chunksizes));CHKERRQ(ierr);
  ierr = PetscMalloc3(pipefcg->mmax+2,&(pipefcg->dots),pipefcg->mmax+1,&(pipefcg->etas),pipefcg->mmax+2,&(pipefcg->redux));CHKERRQ(ierr);

  /* If the requested number of preallocated vectors is greater than mmax reduce nprealloc */
  if (pipefcg->nprealloc > pipefcg->mmax+1) {
    ierr = PetscInfo(NULL,"Requested nprealloc=%d is greater than m_max+1=%d. Resetting nprealloc = m_max+1.\n",pipefcg->nprealloc, pipefcg->mmax+1);CHKERRQ(ierr);
  }

  /* Preallocate additional work vectors */
  ierr = KSPAllocateVectors_PIPEFCG(ksp,pipefcg->nprealloc,pipefcg->nprealloc);CHKERRQ(ierr);

  ierr = PetscLogObjectMemory((PetscObject)ksp,(pipefcg->mmax+1)*4*sizeof(Vec*)+(pipefcg->mmax+1)*4*sizeof(Vec**)+(pipefcg->mmax+1)*4*sizeof(Vec*)+
    (pipefcg->mmax+1)*sizeof(PetscInt)+(pipefcg->mmax+2)*sizeof(Vec*)+(pipefcg->mmax+2)*sizeof(PetscScalar)+(pipefcg->mmax+1)*sizeof(PetscReal));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_PIPEFCG_cycle(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,idx,kdx,mi;
  KSP_PIPEFCG    *pipefcg;
  PetscScalar    alpha=0.0,gamma,*betas,*dots;
  PetscReal      dp=0.0, delta,*eta,*etas;
  Vec            B,R,Z,X,Qcurr,W,ZETAcurr,M,N,Pcurr,Scurr,*redux;
  Mat            Amat,Pmat;

  PetscFunctionBegin;
  /* We have not checked these routines for use with complex numbers. The inner products
     are likely not defined correctly for that case */
  if (PetscDefined(USE_COMPLEX) && !PetscDefined(SKIP_COMPLEX)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"PIPEFGMRES has not been implemented for use with complex scalars");

#define VecXDot(x,y,a)         (((pipefcg->type) == (KSP_CG_HERMITIAN)) ? VecDot       (x,y,a)   : VecTDot       (x,y,a))
#define VecXDotBegin(x,y,a)    (((pipefcg->type) == (KSP_CG_HERMITIAN)) ? VecDotBegin  (x,y,a)   : VecTDotBegin  (x,y,a))
#define VecXDotEnd(x,y,a)      (((pipefcg->type) == (KSP_CG_HERMITIAN)) ? VecDotEnd    (x,y,a)   : VecTDotEnd    (x,y,a))
#define VecMXDot(x,n,y,a)      (((pipefcg->type) == (KSP_CG_HERMITIAN)) ? VecMDot      (x,n,y,a) : VecMTDot      (x,n,y,a))
#define VecMXDotBegin(x,n,y,a) (((pipefcg->type) == (KSP_CG_HERMITIAN)) ? VecMDotBegin (x,n,y,a) : VecMTDotBegin (x,n,y,a))
#define VecMXDotEnd(x,n,y,a)   (((pipefcg->type) == (KSP_CG_HERMITIAN)) ? VecMDotEnd   (x,n,y,a) : VecMTDotEnd   (x,n,y,a))

  pipefcg       = (KSP_PIPEFCG*)ksp->data;
  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Z             = ksp->work[1];
  W             = ksp->work[2];
  M             = ksp->work[3];
  N             = ksp->work[4];

  redux = pipefcg->redux;
  dots  = pipefcg->dots;
  etas  = pipefcg->etas;
  betas = dots;        /* dots takes the result of all dot products of which the betas are a subset */

  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);

  /* Compute cycle initial residual */
  ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);
  ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);                   /* r <- b - Ax */
  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                /* z <- Br     */

  Pcurr = pipefcg->Pvecs[0];
  Scurr = pipefcg->Svecs[0];
  Qcurr = pipefcg->Qvecs[0];
  ZETAcurr = pipefcg->ZETAvecs[0];
  ierr  = VecCopy(Z,Pcurr);CHKERRQ(ierr);
  ierr  = KSP_MatMult(ksp,Amat,Pcurr,Scurr);CHKERRQ(ierr);  /* S = Ap     */
  ierr  = VecCopy(Scurr,W);CHKERRQ(ierr);                   /* w = s = Az */

  /* Initial state of pipelining intermediates */
  redux[0] = R;
  redux[1] = W;
  ierr     = VecMXDotBegin(Z,2,redux,dots);CHKERRQ(ierr);
  ierr     = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)Z));CHKERRQ(ierr); /* perform asynchronous reduction */
  ierr     = KSP_PCApply(ksp,W,M);CHKERRQ(ierr);            /* m = B(w) */
  ierr     = KSP_MatMult(ksp,Amat,M,N);CHKERRQ(ierr);       /* n = Am   */
  ierr     = VecCopy(M,Qcurr);CHKERRQ(ierr);                /* q = m    */
  ierr     = VecCopy(N,ZETAcurr);CHKERRQ(ierr);             /* zeta = n */
  ierr     = VecMXDotEnd(Z,2,redux,dots);CHKERRQ(ierr);
  gamma    = dots[0];
  delta    = PetscRealPart(dots[1]);
  etas[0]  = delta;
  alpha    = gamma/delta;

  i = 0;
  do {
    ksp->its++;

    /* Update X, R, Z, W */
    ierr = VecAXPY(X,+alpha,Pcurr);CHKERRQ(ierr);           /* x <- x + alpha * pi    */
    ierr = VecAXPY(R,-alpha,Scurr);CHKERRQ(ierr);           /* r <- r - alpha * si    */
    ierr = VecAXPY(Z,-alpha,Qcurr);CHKERRQ(ierr);           /* z <- z - alpha * qi    */
    ierr = VecAXPY(W,-alpha,ZETAcurr);CHKERRQ(ierr);        /* w <- w - alpha * zetai */

    /* Compute norm for convergence check */
    switch (ksp->normtype) {
      case KSP_NORM_PRECONDITIONED:
        ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);         /* dp <- sqrt(z'*z) = sqrt(e'*A'*B'*B*A*e) */
        break;
      case KSP_NORM_UNPRECONDITIONED:
        ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);         /* dp <- sqrt(r'*r) = sqrt(e'*A'*A*e)      */
        break;
      case KSP_NORM_NATURAL:
        dp = PetscSqrtReal(PetscAbsScalar(gamma));          /* dp <- sqrt(r'*z) = sqrt(e'*A'*B*A*e)    */
        break;
      case KSP_NORM_NONE:
        dp = 0.0;
        break;
      default: SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
    }

    /* Check for convergence */
    ksp->rnorm = dp;
    ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,ksp->its,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,ksp->its,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) PetscFunctionReturn(0);

    /* Computations of current iteration done */
    ++i;

    /* If needbe, allocate a new chunk of vectors in P and C */
    ierr = KSPAllocateVectors_PIPEFCG(ksp,i+1,pipefcg->vecb);CHKERRQ(ierr);

    /* Note that we wrap around and start clobbering old vectors */
    idx = i % (pipefcg->mmax+1);
    Pcurr    = pipefcg->Pvecs[idx];
    Scurr    = pipefcg->Svecs[idx];
    Qcurr    = pipefcg->Qvecs[idx];
    ZETAcurr = pipefcg->ZETAvecs[idx];
    eta      = pipefcg->etas+idx;

    /* number of old directions to orthogonalize against */
    switch(pipefcg->truncstrat) {
      case KSP_FCD_TRUNC_TYPE_STANDARD:
        mi = pipefcg->mmax;
        break;
      case KSP_FCD_TRUNC_TYPE_NOTAY:
        mi = ((i-1) % pipefcg->mmax)+1;
        break;
      default:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unrecognized Truncation Strategy");
    }

    /* Pick old p,s,q,zeta in a way suitable for VecMDot */
    ierr = VecCopy(Z,Pcurr);CHKERRQ(ierr);
    for (k=PetscMax(0,i-mi),j=0;k<i;++j,++k) {
      kdx = k % (pipefcg->mmax+1);
      pipefcg->Pold[j]    = pipefcg->Pvecs[kdx];
      pipefcg->Sold[j]    = pipefcg->Svecs[kdx];
      pipefcg->Qold[j]    = pipefcg->Qvecs[kdx];
      pipefcg->ZETAold[j] = pipefcg->ZETAvecs[kdx];
      redux[j]            = pipefcg->Svecs[kdx];
    }
    redux[j]   = R;   /* If the above loop is not executed redux contains only R => all beta_k = 0, only gamma, delta != 0 */
    redux[j+1] = W;

    ierr = VecMXDotBegin(Z,j+2,redux,betas);CHKERRQ(ierr);  /* Start split reductions for beta_k = (z,s_k), gamma = (z,r), delta = (z,w) */
    ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)Z));CHKERRQ(ierr); /* perform asynchronous reduction */
    ierr = VecWAXPY(N,-1.0,R,W);CHKERRQ(ierr);              /* m = u + B(w-r): (a) ntmp = w-r              */
    ierr = KSP_PCApply(ksp,N,M);CHKERRQ(ierr);              /* m = u + B(w-r): (b) mtmp = B(ntmp) = B(w-r) */
    ierr = VecAXPY(M,1.0,Z);CHKERRQ(ierr);                  /* m = u + B(w-r): (c) m = z + mtmp            */
    ierr = KSP_MatMult(ksp,Amat,M,N);CHKERRQ(ierr);         /* n = Am                                      */
    ierr = VecMXDotEnd(Z,j+2,redux,betas);CHKERRQ(ierr);    /* Finish split reductions */
    gamma = betas[j];
    delta = PetscRealPart(betas[j+1]);

    *eta = 0.;
    for (k=PetscMax(0,i-mi),j=0;k<i;++j,++k) {
      kdx = k % (pipefcg->mmax+1);
      betas[j] /= -etas[kdx];                               /* betak  /= etak */
      *eta -= ((PetscReal)(PetscAbsScalar(betas[j])*PetscAbsScalar(betas[j]))) * etas[kdx];
                                                            /* etaitmp = -betaik^2 * etak */
    }
    *eta += delta;                                          /* etai    = delta -betaik^2 * etak */
    if (*eta < 0.) {
      pipefcg->norm_breakdown = PETSC_TRUE;
      ierr = PetscInfo(ksp,"Restart due to square root breakdown at it = \n",ksp->its);CHKERRQ(ierr);
      break;
    } else {
      alpha= gamma/(*eta);                                  /* alpha = gamma/etai */
    }

    /* project out stored search directions using classical G-S */
    ierr = VecCopy(Z,Pcurr);CHKERRQ(ierr);
    ierr = VecCopy(W,Scurr);CHKERRQ(ierr);
    ierr = VecCopy(M,Qcurr);CHKERRQ(ierr);
    ierr = VecCopy(N,ZETAcurr);CHKERRQ(ierr);
    ierr = VecMAXPY(Pcurr   ,j,betas,pipefcg->Pold);CHKERRQ(ierr);    /* pi    <- ui - sum_k beta_k p_k    */
    ierr = VecMAXPY(Scurr   ,j,betas,pipefcg->Sold);CHKERRQ(ierr);    /* si    <- wi - sum_k beta_k s_k    */
    ierr = VecMAXPY(Qcurr   ,j,betas,pipefcg->Qold);CHKERRQ(ierr);    /* qi    <- m  - sum_k beta_k q_k    */
    ierr = VecMAXPY(ZETAcurr,j,betas,pipefcg->ZETAold);CHKERRQ(ierr); /* zetai <- n  - sum_k beta_k zeta_k */

  } while (ksp->its < ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_PIPEFCG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_PIPEFCG    *pipefcg;
  PetscScalar    gamma;
  PetscReal      dp=0.0;
  Vec            B,R,Z,X;
  Mat            Amat,Pmat;

#define VecXDot(x,y,a)         (((pipefcg->type) == (KSP_CG_HERMITIAN)) ? VecDot       (x,y,a)   : VecTDot       (x,y,a))

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);

  pipefcg       = (KSP_PIPEFCG*)ksp->data;
  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Z             = ksp->work[1];

  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);

  /* Compute initial residual needed for convergence check*/
  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);                 /* r <- b - Ax                             */
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);                      /* r <- b (x is 0)                         */
  }
  switch (ksp->normtype) {
    case KSP_NORM_PRECONDITIONED:
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);            /* z <- Br                                 */
      ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);           /* dp <- dqrt(z'*z) = sqrt(e'*A'*B'*B*A*e) */
      break;
    case KSP_NORM_UNPRECONDITIONED:
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);           /* dp <- sqrt(r'*r) = sqrt(e'*A'*A*e)      */
      break;
    case KSP_NORM_NATURAL:
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);            /* z <- Br                                 */
      ierr = VecXDot(Z,R,&gamma);CHKERRQ(ierr);
      dp = PetscSqrtReal(PetscAbsScalar(gamma));            /* dp <- sqrt(r'*z) = sqrt(e'*A'*B*A*e)    */
      break;
    case KSP_NORM_NONE:
      dp = 0.0;
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }

  /* Initial Convergence Check */
  ierr       = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr       = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ksp->rnorm = dp;
  ierr       = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  do {
    /* A cycle is broken only if a norm breakdown occurs. If not the entire solve happens in a single cycle.
       This is coded this way to allow both truncation and truncation-restart strategy
       (see KSPFCDGetNumOldDirections()) */
    ierr = KSPSolve_PIPEFCG_cycle(ksp);CHKERRQ(ierr);
    if (ksp->reason) PetscFunctionReturn(0);
    if (pipefcg->norm_breakdown) {
      pipefcg->n_restarts++;
      pipefcg->norm_breakdown = PETSC_FALSE;
    }
  } while (ksp->its < ksp->max_it);

  if (ksp->its >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_PIPEFCG(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  KSP_PIPEFCG    *pipefcg;

  PetscFunctionBegin;
  pipefcg = (KSP_PIPEFCG*)ksp->data;

  /* Destroy "standard" work vecs */
  VecDestroyVecs(ksp->nwork,&ksp->work);

  /* Destroy vectors of old directions and the arrays that manage pointers to them */
  if (pipefcg->nvecs) {
    for (i=0;i<pipefcg->nchunks;++i) {
      ierr = VecDestroyVecs(pipefcg->chunksizes[i],&pipefcg->pPvecs[i]);CHKERRQ(ierr);
      ierr = VecDestroyVecs(pipefcg->chunksizes[i],&pipefcg->pSvecs[i]);CHKERRQ(ierr);
      ierr = VecDestroyVecs(pipefcg->chunksizes[i],&pipefcg->pQvecs[i]);CHKERRQ(ierr);
      ierr = VecDestroyVecs(pipefcg->chunksizes[i],&pipefcg->pZETAvecs[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree4(pipefcg->Pvecs,pipefcg->Svecs,pipefcg->pPvecs,pipefcg->pSvecs);CHKERRQ(ierr);
  ierr = PetscFree4(pipefcg->Qvecs,pipefcg->ZETAvecs,pipefcg->pQvecs,pipefcg->pZETAvecs);CHKERRQ(ierr);
  ierr = PetscFree4(pipefcg->Pold,pipefcg->Sold,pipefcg->Qold,pipefcg->ZETAold);CHKERRQ(ierr);
  ierr = PetscFree(pipefcg->chunksizes);CHKERRQ(ierr);
  ierr = PetscFree3(pipefcg->dots,pipefcg->etas,pipefcg->redux);CHKERRQ(ierr);
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPView_PIPEFCG(KSP ksp,PetscViewer viewer)
{
  KSP_PIPEFCG    *pipefcg = (KSP_PIPEFCG*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii,isstring;
  const char     *truncstr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);

  if (pipefcg->truncstrat == KSP_FCD_TRUNC_TYPE_STANDARD) {
    truncstr = "Using standard truncation strategy";
  } else if (pipefcg->truncstrat == KSP_FCD_TRUNC_TYPE_NOTAY) {
    truncstr = "Using Notay's truncation strategy";
  } else {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Undefined FCD truncation strategy");
  }

  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  max previous directions = %D\n",pipefcg->mmax);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  preallocated %D directions\n",PetscMin(pipefcg->nprealloc,pipefcg->mmax+1));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  %s\n",truncstr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  restarts performed = %D \n", pipefcg->n_restarts);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer,
      "max previous directions = %D, preallocated %D directions, %s truncation strategy",
      pipefcg->mmax,pipefcg->nprealloc,truncstr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  KSPPIPEFCGSetMmax - set the maximum number of previous directions PIPEFCG will store for orthogonalization

  Note: mmax + 1 directions are stored (mmax previous ones along with the current one)
  and whether all are used in each iteration also depends on the truncation strategy
  (see KSPPIPEFCGSetTruncationType)

  Logically Collective on ksp

  Input Parameters:
+  ksp - the Krylov space context
-  mmax - the maximum number of previous directions to orthogonalize against

  Level: intermediate

  Options Database:
. -ksp_pipefcg_mmax <N>

.seealso: KSPPIPEFCG, KSPPIPEFCGSetTruncationType(), KSPPIPEFCGSetNprealloc()
@*/
PetscErrorCode KSPPIPEFCGSetMmax(KSP ksp,PetscInt mmax)
{
  KSP_PIPEFCG *pipefcg=(KSP_PIPEFCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveInt(ksp,mmax,2);
  pipefcg->mmax=mmax;
  PetscFunctionReturn(0);
}

/*@
  KSPPIPEFCGGetMmax - get the maximum number of previous directions PIPEFCG will store

  Note: PIPEFCG stores mmax+1 directions at most (mmax previous ones, and the current one)

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  mmax - the maximum number of previous directons allowed for orthogonalization

  Options Database:
. -ksp_pipefcg_mmax <N>

   Level: intermediate

.seealso: KSPPIPEFCG, KSPPIPEFCGGetTruncationType(), KSPPIPEFCGGetNprealloc(), KSPPIPEFCGSetMmax()
@*/
PetscErrorCode KSPPIPEFCGGetMmax(KSP ksp,PetscInt *mmax)
{
  KSP_PIPEFCG *pipefcg=(KSP_PIPEFCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *mmax=pipefcg->mmax;
  PetscFunctionReturn(0);
}

/*@
  KSPPIPEFCGSetNprealloc - set the number of directions to preallocate with PIPEFCG

  Logically Collective on ksp

  Input Parameters:
+  ksp - the Krylov space context
-  nprealloc - the number of vectors to preallocate

  Level: advanced

  Options Database:
. -ksp_pipefcg_nprealloc <N>

.seealso: KSPPIPEFCG, KSPPIPEFCGSetTruncationType(), KSPPIPEFCGGetNprealloc()
@*/
PetscErrorCode KSPPIPEFCGSetNprealloc(KSP ksp,PetscInt nprealloc)
{
  KSP_PIPEFCG *pipefcg=(KSP_PIPEFCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveInt(ksp,nprealloc,2);
  pipefcg->nprealloc = nprealloc;
  PetscFunctionReturn(0);
}

/*@
  KSPPIPEFCGGetNprealloc - get the number of directions to preallocate by PIPEFCG

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  nprealloc - the number of directions preallocated

  Options Database:
. -ksp_pipefcg_nprealloc <N>

   Level: advanced

.seealso: KSPPIPEFCG, KSPPIPEFCGGetTruncationType(), KSPPIPEFCGSetNprealloc()
@*/
PetscErrorCode KSPPIPEFCGGetNprealloc(KSP ksp,PetscInt *nprealloc)
{
  KSP_PIPEFCG *pipefcg=(KSP_PIPEFCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *nprealloc = pipefcg->nprealloc;
  PetscFunctionReturn(0);
}

/*@
  KSPPIPEFCGSetTruncationType - specify how many of its stored previous directions PIPEFCG uses during orthoganalization

  Logically Collective on ksp

  KSP_FCD_TRUNC_TYPE_STANDARD uses all (up to mmax) stored directions
  KSP_FCD_TRUNC_TYPE_NOTAY uses max(1,mod(i,mmax)) stored directions at iteration i=0,1,..

  Input Parameters:
+  ksp - the Krylov space context
-  truncstrat - the choice of strategy

  Level: intermediate

  Options Database:
.  -ksp_pipefcg_truncation_type <standard,notay> - which stored search directions to orthogonalize against

.seealso: KSPPIPEFCG, KSPPIPEFCGGetTruncationType, KSPFCDTruncationType
@*/
PetscErrorCode KSPPIPEFCGSetTruncationType(KSP ksp,KSPFCDTruncationType truncstrat)
{
  KSP_PIPEFCG *pipefcg=(KSP_PIPEFCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ksp,truncstrat,2);
  pipefcg->truncstrat=truncstrat;
  PetscFunctionReturn(0);
}

/*@
  KSPPIPEFCGGetTruncationType - get the truncation strategy employed by PIPEFCG

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  truncstrat - the strategy type

  Options Database:
. -ksp_pipefcg_truncation_type <standard,notay> - which stored basis vectors to orthogonalize against

   Level: intermediate

.seealso: KSPPIPEFCG, KSPPIPEFCGSetTruncationType, KSPFCDTruncationType
@*/
PetscErrorCode KSPPIPEFCGGetTruncationType(KSP ksp,KSPFCDTruncationType *truncstrat)
{
  KSP_PIPEFCG *pipefcg=(KSP_PIPEFCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *truncstrat=pipefcg->truncstrat;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetFromOptions_PIPEFCG(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;
  KSP_PIPEFCG    *pipefcg=(KSP_PIPEFCG*)ksp->data;
  PetscInt       mmax,nprealloc;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP PIPEFCG options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_pipefcg_mmax","Number of search directions to storue","KSPPIPEFCGSetMmax",pipefcg->mmax,&mmax,&flg);CHKERRQ(ierr);
  if (flg) {ierr = KSPPIPEFCGSetMmax(ksp,mmax);CHKERRQ(ierr);}
  ierr = PetscOptionsInt("-ksp_pipefcg_nprealloc","Number of directions to preallocate","KSPPIPEFCGSetNprealloc",pipefcg->nprealloc,&nprealloc,&flg);CHKERRQ(ierr);
  if (flg) {ierr = KSPPIPEFCGSetNprealloc(ksp,nprealloc);CHKERRQ(ierr);}
  ierr = PetscOptionsEnum("-ksp_pipefcg_truncation_type","Truncation approach for directions","KSPFCGSetTruncationType",KSPFCDTruncationTypes,(PetscEnum)pipefcg->truncstrat,(PetscEnum*)&pipefcg->truncstrat,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC

  KSPPIPEFCG - Implements a Pipelined, Flexible Conjugate Gradient method.

  Options Database Keys:
+   -ksp_pipefcg_mmax <N> - The number of previous search directions to store
.   -ksp_pipefcg_nprealloc <N> - The number of previous search directions to preallocate
-   -ksp_pipefcg_truncation_type <standard,notay> - which stored search directions to orthogonalize against

  Notes:
   Supports left preconditioning only.

   The natural "norm" for this method is (u,Au), where u is the preconditioned residual. As with standard CG, this norm is available at no additional computational cost. Choosing preconditioned or unpreconditioned norms involve an extra blocking global reduction, thus removing any benefit from pipelining.

   MPI configuration may be necessary for reductions to make asynchronous progress, which is important for performance of pipelined methods.
   See the FAQ on the PETSc website for details.

  Reference:
    P. Sanan, S.M. Schnepp, and D.A. May,
    "Pipelined, Flexible Krylov Subspace Methods,"
    SIAM Journal on Scientific Computing 2016 38:5, C441-C470,
    DOI: 10.1137/15M1049130

  Level: intermediate

.seealso: KSPFCG, KSPPIPECG, KSPPIPECR, KSPGCR, KSPPIPEGCR, KSPFGMRES, KSPCG, KSPPIPEFCGSetMmax(), KSPPIPEFCGGetMmax(), KSPPIPEFCGSetNprealloc(), KSPPIPEFCGGetNprealloc(), KSPPIPEFCGSetTruncationType(), KSPPIPEFCGGetTruncationType()

M*/
PETSC_EXTERN PetscErrorCode KSPCreate_PIPEFCG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_PIPEFCG    *pipefcg;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&pipefcg);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  pipefcg->type       = KSP_CG_SYMMETRIC;
#else
  pipefcg->type       = KSP_CG_HERMITIAN;
#endif
  pipefcg->mmax       = KSPPIPEFCG_DEFAULT_MMAX;
  pipefcg->nprealloc  = KSPPIPEFCG_DEFAULT_NPREALLOC;
  pipefcg->nvecs      = 0;
  pipefcg->vecb       = KSPPIPEFCG_DEFAULT_VECB;
  pipefcg->nchunks    = 0;
  pipefcg->truncstrat = KSPPIPEFCG_DEFAULT_TRUNCSTRAT;
  pipefcg->n_restarts = 0;

  ksp->data = (void*)pipefcg;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);

  ksp->ops->setup          = KSPSetUp_PIPEFCG;
  ksp->ops->solve          = KSPSolve_PIPEFCG;
  ksp->ops->destroy        = KSPDestroy_PIPEFCG;
  ksp->ops->view           = KSPView_PIPEFCG;
  ksp->ops->setfromoptions = KSPSetFromOptions_PIPEFCG;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  PetscFunctionReturn(0);
}
