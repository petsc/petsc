/*
    Contributed by Sascha M. Schnepp and Patrick Sanan
*/

#include "petscsys.h"
#include <../src/ksp/ksp/impls/gcr/pipegcr/pipegcrimpl.h>       /*I  "petscksp.h"  I*/

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

#define KSPPIPEGCR_DEFAULT_MMAX 15
#define KSPPIPEGCR_DEFAULT_NPREALLOC 5
#define KSPPIPEGCR_DEFAULT_VECB 5
#define KSPPIPEGCR_DEFAULT_TRUNCSTRAT KSP_FCD_TRUNC_TYPE_NOTAY
#define KSPPIPEGCR_DEFAULT_UNROLL_W PETSC_TRUE

#include <petscksp.h>

static PetscErrorCode KSPAllocateVectors_PIPEGCR(KSP ksp, PetscInt nvecsneeded, PetscInt chunksize)
{
  PetscErrorCode  ierr;
  PetscInt        i;
  KSP_PIPEGCR     *pipegcr;
  PetscInt        nnewvecs, nvecsprev;

  PetscFunctionBegin;
  pipegcr = (KSP_PIPEGCR*)ksp->data;

  /* Allocate enough new vectors to add chunksize new vectors, reach nvecsneedtotal, or to reach mmax+1, whichever is smallest */
  if (pipegcr->nvecs < PetscMin(pipegcr->mmax+1,nvecsneeded)) {
    nvecsprev = pipegcr->nvecs;
    nnewvecs = PetscMin(PetscMax(nvecsneeded-pipegcr->nvecs,chunksize),pipegcr->mmax+1-pipegcr->nvecs);
    ierr = KSPCreateVecs(ksp,nnewvecs,&pipegcr->ppvecs[pipegcr->nchunks],0,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents((PetscObject)ksp,nnewvecs,pipegcr->ppvecs[pipegcr->nchunks]);CHKERRQ(ierr);
    ierr = KSPCreateVecs(ksp,nnewvecs,&pipegcr->psvecs[pipegcr->nchunks],0,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents((PetscObject)ksp,nnewvecs,pipegcr->psvecs[pipegcr->nchunks]);CHKERRQ(ierr);
    ierr = KSPCreateVecs(ksp,nnewvecs,&pipegcr->pqvecs[pipegcr->nchunks],0,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents((PetscObject)ksp,nnewvecs,pipegcr->pqvecs[pipegcr->nchunks]);CHKERRQ(ierr);
    if (pipegcr->unroll_w) {
      ierr = KSPCreateVecs(ksp,nnewvecs,&pipegcr->ptvecs[pipegcr->nchunks],0,NULL);CHKERRQ(ierr);
      ierr = PetscLogObjectParents((PetscObject)ksp,nnewvecs,pipegcr->ptvecs[pipegcr->nchunks]);CHKERRQ(ierr);
    }
    pipegcr->nvecs += nnewvecs;
    for (i=0;i<nnewvecs;i++) {
      pipegcr->qvecs[nvecsprev+i] = pipegcr->pqvecs[pipegcr->nchunks][i];
      pipegcr->pvecs[nvecsprev+i] = pipegcr->ppvecs[pipegcr->nchunks][i];
      pipegcr->svecs[nvecsprev+i] = pipegcr->psvecs[pipegcr->nchunks][i];
      if (pipegcr->unroll_w) {
        pipegcr->tvecs[nvecsprev+i] = pipegcr->ptvecs[pipegcr->nchunks][i];
      }
    }
    pipegcr->chunksizes[pipegcr->nchunks] = nnewvecs;
    pipegcr->nchunks++;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_PIPEGCR_cycle(KSP ksp)
{
  KSP_PIPEGCR    *pipegcr = (KSP_PIPEGCR*)ksp->data;
  PetscErrorCode ierr;
  Mat            A, B;
  Vec            x,r,b,z,w,m,n,p,s,q,t,*redux;
  PetscInt       i,j,k,idx,kdx,mi;
  PetscScalar    alpha=0.0,gamma,*betas,*dots;
  PetscReal      rnorm=0.0, delta,*eta,*etas;

  PetscFunctionBegin;
  /* !!PS We have not checked these routines for use with complex numbers. The inner products
     are likely not defined correctly for that case */
  if (PetscDefined(USE_COMPLEX) && !PetscDefined(SKIP_COMPLEX)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"PIPEFGMRES has not been implemented for use with complex scalars");

  ierr = KSPGetOperators(ksp, &A, &B);CHKERRQ(ierr);
  x = ksp->vec_sol;
  b = ksp->vec_rhs;
  r = ksp->work[0];
  z = ksp->work[1];
  w = ksp->work[2]; /* w = Az = AB(r)                 (pipelining intermediate) */
  m = ksp->work[3]; /* m = B(w) = B(Az) = B(AB(r))    (pipelining intermediate) */
  n = ksp->work[4]; /* n = AB(w) = AB(Az) = AB(AB(r)) (pipelining intermediate) */
  p = pipegcr->pvecs[0];
  s = pipegcr->svecs[0];
  q = pipegcr->qvecs[0];
  t = pipegcr->unroll_w ? pipegcr->tvecs[0] : NULL;

  redux = pipegcr->redux;
  dots  = pipegcr->dots;
  etas  = pipegcr->etas;
  betas = dots;        /* dots takes the result of all dot products of which the betas are a subset */

  /* cycle initial residual */
  ierr = KSP_MatMult(ksp,A,x,r);CHKERRQ(ierr);
  ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);                   /* r <- b - Ax */
  ierr = KSP_PCApply(ksp,r,z);CHKERRQ(ierr);                /* z <- B(r)   */
  ierr = KSP_MatMult(ksp,A,z,w);CHKERRQ(ierr);              /* w <- Az     */

  /* initialization of other variables and pipelining intermediates */
  ierr    = VecCopy(z,p);CHKERRQ(ierr);
  ierr    = KSP_MatMult(ksp,A,p,s);CHKERRQ(ierr);

  /* overlap initial computation of delta, gamma */
  redux[0] = w;
  redux[1] = r;
  ierr     = VecMDotBegin(w,2,redux,dots);CHKERRQ(ierr);    /* Start split reductions for gamma = (w,r), delta = (w,w) */
  ierr     = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)s));CHKERRQ(ierr); /* perform asynchronous reduction */
  ierr     = KSP_PCApply(ksp,s,q);CHKERRQ(ierr);            /* q = B(s) */
  if (pipegcr->unroll_w) {
    ierr     = KSP_MatMult(ksp,A,q,t);CHKERRQ(ierr);        /* t = Aq   */
  }
  ierr     = VecMDotEnd(w,2,redux,dots);CHKERRQ(ierr);      /* Finish split reduction */
  delta    = PetscRealPart(dots[0]);
  etas[0]  = delta;
  gamma    = dots[1];
  alpha    = gamma/delta;

  i = 0;
  do {
    ierr     = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
    ksp->its++;
    ierr     = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);

    /* update solution, residuals, .. */
    ierr = VecAXPY(x,+alpha,p);CHKERRQ(ierr);
    ierr = VecAXPY(r,-alpha,s);CHKERRQ(ierr);
    ierr = VecAXPY(z,-alpha,q);CHKERRQ(ierr);
    if (pipegcr->unroll_w) {
      ierr = VecAXPY(w,-alpha,t);CHKERRQ(ierr);
    } else {
      ierr = KSP_MatMult(ksp,A,z,w);CHKERRQ(ierr);
    }

    /* Computations of current iteration done */
    i++;

    if (pipegcr->modifypc) {
      ierr = (*pipegcr->modifypc)(ksp,ksp->its,ksp->rnorm,pipegcr->modifypc_ctx);CHKERRQ(ierr);
    }

    /* If needbe, allocate a new chunk of vectors */
    ierr = KSPAllocateVectors_PIPEGCR(ksp,i+1,pipegcr->vecb);CHKERRQ(ierr);

    /* Note that we wrap around and start clobbering old vectors */
    idx = i % (pipegcr->mmax+1);
    p   = pipegcr->pvecs[idx];
    s   = pipegcr->svecs[idx];
    q   = pipegcr->qvecs[idx];
    if (pipegcr->unroll_w) {
      t   = pipegcr->tvecs[idx];
    }
    eta = pipegcr->etas+idx;

    /* number of old directions to orthogonalize against */
    switch(pipegcr->truncstrat) {
      case KSP_FCD_TRUNC_TYPE_STANDARD:
        mi = pipegcr->mmax;
        break;
      case KSP_FCD_TRUNC_TYPE_NOTAY:
        mi = ((i-1) % pipegcr->mmax)+1;
        break;
      default:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unrecognized Truncation Strategy");
    }

    /* Pick old p,s,q,zeta in a way suitable for VecMDot */
    for (k=PetscMax(0,i-mi),j=0;k<i;j++,k++) {
      kdx = k % (pipegcr->mmax+1);
      pipegcr->pold[j] = pipegcr->pvecs[kdx];
      pipegcr->sold[j] = pipegcr->svecs[kdx];
      pipegcr->qold[j] = pipegcr->qvecs[kdx];
      if (pipegcr->unroll_w) {
        pipegcr->told[j] = pipegcr->tvecs[kdx];
      }
      redux[j]         = pipegcr->svecs[kdx];
    }
    /* If the above loop is not run redux contains only r and w => all beta_k = 0, only gamma, delta != 0 */
    redux[j]   = r;
    redux[j+1] = w;

    /* Dot products */
    /* Start split reductions for beta_k = (w,s_k), gamma = (w,r), delta = (w,w) */
    ierr = VecMDotBegin(w,j+2,redux,dots);CHKERRQ(ierr);
    ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)w));CHKERRQ(ierr);

    /* B(w-r) + u stabilization */
    ierr = VecWAXPY(n,-1.0,r,w);CHKERRQ(ierr);              /* m = u + B(w-r): (a) ntmp = w-r              */
    ierr = KSP_PCApply(ksp,n,m);CHKERRQ(ierr);              /* m = u + B(w-r): (b) mtmp = B(ntmp) = B(w-r) */
    ierr = VecAXPY(m,1.0,z);CHKERRQ(ierr);                  /* m = u + B(w-r): (c) m = z + mtmp            */
    if (pipegcr->unroll_w) {
      ierr = KSP_MatMult(ksp,A,m,n);CHKERRQ(ierr);          /* n = Am                                      */
    }

    /* Finish split reductions for beta_k = (w,s_k), gamma = (w,r), delta = (w,w) */
    ierr = VecMDotEnd(w,j+2,redux,dots);CHKERRQ(ierr);
    gamma = dots[j];
    delta = PetscRealPart(dots[j+1]);

    /* compute new residual norm.
       this cannot be done before this point so that the natural norm
       is available for free and the communication involved is overlapped */
    switch (ksp->normtype) {
    case KSP_NORM_PRECONDITIONED:
      ierr = VecNorm(z,NORM_2,&rnorm);CHKERRQ(ierr);        /* ||r|| <- sqrt(z'*z) */
      break;
    case KSP_NORM_UNPRECONDITIONED:
      ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);        /* ||r|| <- sqrt(r'*r) */
      break;
    case KSP_NORM_NATURAL:
      rnorm = PetscSqrtReal(PetscAbsScalar(gamma));         /* ||r|| <- sqrt(r,w)  */
      break;
    case KSP_NORM_NONE:
      rnorm = 0.0;
      break;
    default: SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
    }

    /* Check for convergence */
    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
    ksp->rnorm = rnorm;
    ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
    KSPLogResidualHistory(ksp,rnorm);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,ksp->its,rnorm);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,ksp->its,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) PetscFunctionReturn(0);

    /* compute new eta and scale beta */
    *eta = 0.;
    for (k=PetscMax(0,i-mi),j=0;k<i;j++,k++) {
      kdx = k % (pipegcr->mmax+1);
      betas[j] /= -etas[kdx];                               /* betak  /= etak */
      *eta -= ((PetscReal)(PetscAbsScalar(betas[j])*PetscAbsScalar(betas[j]))) * etas[kdx];
                                                            /* etaitmp = -betaik^2 * etak */
    }
    *eta += delta;                                          /* etai    = delta -betaik^2 * etak */

    /* check breakdown of eta = (s,s) */
    if (*eta < 0.) {
      pipegcr->norm_breakdown = PETSC_TRUE;
      ierr = PetscInfo1(ksp,"Restart due to square root breakdown at it = \n",ksp->its);CHKERRQ(ierr);
      break;
    } else {
      alpha= gamma/(*eta);                                  /* alpha = gamma/etai */
    }

    /* project out stored search directions using classical G-S */
    ierr = VecCopy(z,p);CHKERRQ(ierr);
    ierr = VecCopy(w,s);CHKERRQ(ierr);
    ierr = VecCopy(m,q);CHKERRQ(ierr);
    if (pipegcr->unroll_w) {
      ierr = VecCopy(n,t);CHKERRQ(ierr);
      ierr = VecMAXPY(t,j,betas,pipegcr->told);CHKERRQ(ierr); /* ti <- n  - sum_k beta_k t_k */
    }
    ierr = VecMAXPY(p,j,betas,pipegcr->pold);CHKERRQ(ierr); /* pi <- ui - sum_k beta_k p_k */
    ierr = VecMAXPY(s,j,betas,pipegcr->sold);CHKERRQ(ierr); /* si <- wi - sum_k beta_k s_k */
    ierr = VecMAXPY(q,j,betas,pipegcr->qold);CHKERRQ(ierr); /* qi <- m  - sum_k beta_k q_k */

  } while (ksp->its < ksp->max_it);
  if (ksp->its >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_PIPEGCR(KSP ksp)
{
  KSP_PIPEGCR    *pipegcr = (KSP_PIPEGCR*)ksp->data;
  PetscErrorCode ierr;
  Mat            A, B;
  Vec            x,b,r,z,w;
  PetscScalar    gamma;
  PetscReal      rnorm=0.0;
  PetscBool      issym;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);

  ierr = KSPGetOperators(ksp, &A, &B);CHKERRQ(ierr);
  x = ksp->vec_sol;
  b = ksp->vec_rhs;
  r = ksp->work[0];
  z = ksp->work[1];
  w = ksp->work[2]; /* w = Az = AB(r)                 (pipelining intermediate) */

  /* compute initial residual */
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,A,x,r);CHKERRQ(ierr);
    ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);                 /* r <- b - Ax       */
  } else {
    ierr = VecCopy(b,r);CHKERRQ(ierr);                      /* r <- b            */
  }

  /* initial residual norm */
  ierr = KSP_PCApply(ksp,r,z);CHKERRQ(ierr);                /* z <- B(r)         */
  ierr = KSP_MatMult(ksp,A,z,w);CHKERRQ(ierr);              /* w <- Az           */
  ierr = VecDot(r,w,&gamma);CHKERRQ(ierr);                  /* gamma = (r,w)     */

  switch (ksp->normtype) {
    case KSP_NORM_PRECONDITIONED:
      ierr = VecNorm(z,NORM_2,&rnorm);CHKERRQ(ierr);        /* ||r|| <- sqrt(z'*z) */
      break;
    case KSP_NORM_UNPRECONDITIONED:
      ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);        /* ||r|| <- sqrt(r'*r) */
      break;
    case KSP_NORM_NATURAL:
      rnorm = PetscSqrtReal(PetscAbsScalar(gamma));         /* ||r|| <- sqrt(r,w)  */
      break;
    case KSP_NORM_NONE:
      rnorm = 0.0;
      break;
    default: SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }

  /* Is A symmetric? */
  ierr = PetscObjectTypeCompareAny((PetscObject)A,&issym,MATSBAIJ,MATSEQSBAIJ,MATMPISBAIJ,"");CHKERRQ(ierr);
  if (!issym) {
    ierr = PetscInfo(A,"Matrix type is not any of MATSBAIJ,MATSEQSBAIJ,MATMPISBAIJ. Is matrix A symmetric (as required by CR methods)?");CHKERRQ(ierr);
  }

  /* logging */
  ierr        = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->its    = 0;
  ksp->rnorm0 = rnorm;
  ierr        = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
  ierr        = KSPLogResidualHistory(ksp,ksp->rnorm0);CHKERRQ(ierr);
  ierr        = KSPMonitor(ksp,ksp->its,ksp->rnorm0);CHKERRQ(ierr);
  ierr        = (*ksp->converged)(ksp,ksp->its,ksp->rnorm0,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  do {
    ierr = KSPSolve_PIPEGCR_cycle(ksp);CHKERRQ(ierr);
    if (ksp->reason) PetscFunctionReturn(0);
    if (pipegcr->norm_breakdown) {
      pipegcr->n_restarts++;
      pipegcr->norm_breakdown = PETSC_FALSE;
    }
  } while (ksp->its < ksp->max_it);

  if (ksp->its >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPView_PIPEGCR(KSP ksp, PetscViewer viewer)
{
  KSP_PIPEGCR    *pipegcr = (KSP_PIPEGCR*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      isascii,isstring;
  const char     *truncstr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);

  if (pipegcr->truncstrat == KSP_FCD_TRUNC_TYPE_STANDARD) {
    truncstr = "Using standard truncation strategy";
  } else if (pipegcr->truncstrat == KSP_FCD_TRUNC_TYPE_NOTAY) {
    truncstr = "Using Notay's truncation strategy";
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Undefined FCD truncation strategy");

  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  max previous directions = %D\n",pipegcr->mmax);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  preallocated %D directions\n",PetscMin(pipegcr->nprealloc,pipegcr->mmax+1));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  %s\n",truncstr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  w unrolling = %D \n", pipegcr->unroll_w);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  restarts performed = %D \n", pipegcr->n_restarts);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer, "max previous directions = %D, preallocated %D directions, %s truncation strategy", pipegcr->mmax,pipegcr->nprealloc,truncstr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetUp_PIPEGCR(KSP ksp)
{
  KSP_PIPEGCR   *pipegcr = (KSP_PIPEGCR*)ksp->data;
  PetscErrorCode ierr;
  Mat            A;
  PetscBool      diagonalscale;
  const PetscInt nworkstd = 5;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  ierr = KSPGetOperators(ksp, &A, NULL);CHKERRQ(ierr);

  /* Allocate "standard" work vectors */
  ierr = KSPSetWorkVecs(ksp,nworkstd);CHKERRQ(ierr);

  /* Allocated space for pointers to additional work vectors
    note that mmax is the number of previous directions, so we add 1 for the current direction */
  ierr = PetscMalloc6(pipegcr->mmax+1,&(pipegcr->pvecs),pipegcr->mmax+1,&(pipegcr->ppvecs),pipegcr->mmax+1,&(pipegcr->svecs), pipegcr->mmax+1,&(pipegcr->psvecs),pipegcr->mmax+1,&(pipegcr->qvecs),pipegcr->mmax+1,&(pipegcr->pqvecs));CHKERRQ(ierr);
  if (pipegcr->unroll_w) {
    ierr = PetscMalloc3(pipegcr->mmax+1,&(pipegcr->tvecs),pipegcr->mmax+1,&(pipegcr->ptvecs),pipegcr->mmax+2,&(pipegcr->told));CHKERRQ(ierr);
  }
  ierr = PetscMalloc4(pipegcr->mmax+2,&(pipegcr->pold),pipegcr->mmax+2,&(pipegcr->sold),pipegcr->mmax+2,&(pipegcr->qold),pipegcr->mmax+2,&(pipegcr->chunksizes));CHKERRQ(ierr);
  ierr = PetscMalloc3(pipegcr->mmax+2,&(pipegcr->dots),pipegcr->mmax+1,&(pipegcr->etas),pipegcr->mmax+2,&(pipegcr->redux));CHKERRQ(ierr);
  /* If the requested number of preallocated vectors is greater than mmax reduce nprealloc */
  if (pipegcr->nprealloc > pipegcr->mmax+1) {
    ierr = PetscInfo2(NULL,"Requested nprealloc=%d is greater than m_max+1=%d. Resetting nprealloc = m_max+1.\n",pipegcr->nprealloc, pipegcr->mmax+1);CHKERRQ(ierr);
  }

  /* Preallocate additional work vectors */
  ierr = KSPAllocateVectors_PIPEGCR(ksp,pipegcr->nprealloc,pipegcr->nprealloc);CHKERRQ(ierr);

  ierr = PetscLogObjectMemory(
    (PetscObject)ksp,
    (pipegcr->mmax + 1) * 4 * sizeof(Vec*) +        /* old dirs  */
    (pipegcr->mmax + 1) * 4 * sizeof(Vec**) +       /* old pdirs */
    (pipegcr->mmax + 1) * 4 * sizeof(Vec*) +        /* p/s/qold/told */
    (pipegcr->mmax + 1) *     sizeof(PetscInt) +    /* chunksizes */
    (pipegcr->mmax + 2) *     sizeof(Vec*) +        /* redux */
    (pipegcr->mmax + 2) *     sizeof(PetscScalar) + /* dots */
    (pipegcr->mmax + 1) *     sizeof(PetscReal)     /* etas */);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPReset_PIPEGCR(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_PIPEGCR    *pipegcr = (KSP_PIPEGCR*)ksp->data;

  PetscFunctionBegin;
  if (pipegcr->modifypc_destroy) {
    ierr = (*pipegcr->modifypc_destroy)(pipegcr->modifypc_ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_PIPEGCR(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  KSP_PIPEGCR    *pipegcr = (KSP_PIPEGCR*)ksp->data;

  PetscFunctionBegin;
  VecDestroyVecs(ksp->nwork,&ksp->work); /* Destroy "standard" work vecs */

  /* Destroy vectors for old directions and the arrays that manage pointers to them */
  if (pipegcr->nvecs) {
    for (i=0;i<pipegcr->nchunks;i++) {
      ierr = VecDestroyVecs(pipegcr->chunksizes[i],&pipegcr->ppvecs[i]);CHKERRQ(ierr);
      ierr = VecDestroyVecs(pipegcr->chunksizes[i],&pipegcr->psvecs[i]);CHKERRQ(ierr);
      ierr = VecDestroyVecs(pipegcr->chunksizes[i],&pipegcr->pqvecs[i]);CHKERRQ(ierr);
      if (pipegcr->unroll_w) {
        ierr = VecDestroyVecs(pipegcr->chunksizes[i],&pipegcr->ptvecs[i]);CHKERRQ(ierr);
      }
    }
  }

  ierr = PetscFree6(pipegcr->pvecs,pipegcr->ppvecs,pipegcr->svecs,pipegcr->psvecs,pipegcr->qvecs,pipegcr->pqvecs);CHKERRQ(ierr);
  ierr = PetscFree4(pipegcr->pold,pipegcr->sold,pipegcr->qold,pipegcr->chunksizes);CHKERRQ(ierr);
  ierr = PetscFree3(pipegcr->dots,pipegcr->etas,pipegcr->redux);CHKERRQ(ierr);
  if (pipegcr->unroll_w) {
    ierr = PetscFree3(pipegcr->tvecs,pipegcr->ptvecs,pipegcr->told);CHKERRQ(ierr);
  }

  ierr = KSPReset_PIPEGCR(ksp);CHKERRQ(ierr);
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  KSPPIPEGCRSetUnrollW - Set to PETSC_TRUE to use PIPEGCR with unrolling of the w vector

  Logically Collective on ksp

  Input Parameters:
+  ksp - the Krylov space context
-  unroll_w - use unrolling

  Level: intermediate

  Options Database:
. -ksp_pipegcr_unroll_w

.seealso: KSPPIPEGCR, KSPPIPEGCRSetTruncationType(), KSPPIPEGCRSetNprealloc(),KSPPIPEGCRGetUnrollW()
@*/
PetscErrorCode KSPPIPEGCRSetUnrollW(KSP ksp,PetscBool unroll_w)
{
  KSP_PIPEGCR *pipegcr=(KSP_PIPEGCR*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,unroll_w,2);
  pipegcr->unroll_w=unroll_w;
  PetscFunctionReturn(0);
}

/*@
  KSPPIPEGCRGetUnrollW - Get information on PIPEGCR unrolling the w vector

  Logically Collective on ksp

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  unroll_w - PIPEGCR uses unrolling (bool)

  Level: intermediate

  Options Database:
. -ksp_pipegcr_unroll_w

.seealso: KSPPIPEGCR, KSPPIPEGCRGetTruncationType(), KSPPIPEGCRGetNprealloc(),KSPPIPEGCRSetUnrollW()
@*/
PetscErrorCode KSPPIPEGCRGetUnrollW(KSP ksp,PetscBool *unroll_w)
{
  KSP_PIPEGCR *pipegcr=(KSP_PIPEGCR*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *unroll_w=pipegcr->unroll_w;
  PetscFunctionReturn(0);
}

/*@
  KSPPIPEGCRSetMmax - set the maximum number of previous directions PIPEGCR will store for orthogonalization

  Note: mmax + 1 directions are stored (mmax previous ones along with a current one)
  and whether all are used in each iteration also depends on the truncation strategy
  (see KSPPIPEGCRSetTruncationType)

  Logically Collective on ksp

  Input Parameters:
+  ksp - the Krylov space context
-  mmax - the maximum number of previous directions to orthogonalize againt

  Level: intermediate

  Options Database:
. -ksp_pipegcr_mmax <N>

.seealso: KSPPIPEGCR, KSPPIPEGCRSetTruncationType(), KSPPIPEGCRSetNprealloc()
@*/
PetscErrorCode KSPPIPEGCRSetMmax(KSP ksp,PetscInt mmax)
{
  KSP_PIPEGCR *pipegcr=(KSP_PIPEGCR*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveInt(ksp,mmax,2);
  pipegcr->mmax=mmax;
  PetscFunctionReturn(0);
}

/*@
  KSPPIPEGCRGetMmax - get the maximum number of previous directions PIPEGCR will store

  Note: PIPEGCR stores mmax+1 directions at most (mmax previous ones, and one current one)

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  mmax - the maximum number of previous directons allowed for orthogonalization

  Options Database:
. -ksp_pipegcr_mmax <N>

   Level: intermediate

.seealso: KSPPIPEGCR, KSPPIPEGCRGetTruncationType(), KSPPIPEGCRGetNprealloc(), KSPPIPEGCRSetMmax()
@*/

PetscErrorCode KSPPIPEGCRGetMmax(KSP ksp,PetscInt *mmax)
{
  KSP_PIPEGCR *pipegcr=(KSP_PIPEGCR*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *mmax=pipegcr->mmax;
  PetscFunctionReturn(0);
}

/*@
  KSPPIPEGCRSetNprealloc - set the number of directions to preallocate with PIPEGCR

  Logically Collective on ksp

  Input Parameters:
+  ksp - the Krylov space context
-  nprealloc - the number of vectors to preallocate

  Level: advanced

  Options Database:
. -ksp_pipegcr_nprealloc <N>

.seealso: KSPPIPEGCR, KSPPIPEGCRGetTruncationType(), KSPPIPEGCRGetNprealloc()
@*/
PetscErrorCode KSPPIPEGCRSetNprealloc(KSP ksp,PetscInt nprealloc)
{
  KSP_PIPEGCR *pipegcr=(KSP_PIPEGCR*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveInt(ksp,nprealloc,2);
  pipegcr->nprealloc = nprealloc;
  PetscFunctionReturn(0);
}

/*@
  KSPPIPEGCRGetNprealloc - get the number of directions preallocate by PIPEGCR

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  nprealloc - the number of directions preallocated

  Options Database:
. -ksp_pipegcr_nprealloc <N>

   Level: advanced

.seealso: KSPPIPEGCR, KSPPIPEGCRGetTruncationType(), KSPPIPEGCRSetNprealloc()
@*/
PetscErrorCode KSPPIPEGCRGetNprealloc(KSP ksp,PetscInt *nprealloc)
{
  KSP_PIPEGCR *pipegcr=(KSP_PIPEGCR*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *nprealloc = pipegcr->nprealloc;
  PetscFunctionReturn(0);
}

/*@
  KSPPIPEGCRSetTruncationType - specify how many of its stored previous directions PIPEGCR uses during orthoganalization

  Logically Collective on ksp

  KSP_FCD_TRUNC_TYPE_STANDARD uses all (up to mmax) stored directions
  KSP_FCD_TRUNC_TYPE_NOTAY uses the last max(1,mod(i,mmax)) directions at iteration i=0,1,..

  Input Parameters:
+  ksp - the Krylov space context
-  truncstrat - the choice of strategy

  Level: intermediate

  Options Database:
. -ksp_pipegcr_truncation_type <standard,notay> - which stored basis vectors to orthogonalize against

.seealso: KSPPIPEGCR, KSPPIPEGCRSetTruncationType, KSPPIPEGCRTruncationType, KSPFCDTruncationType
@*/
PetscErrorCode KSPPIPEGCRSetTruncationType(KSP ksp,KSPFCDTruncationType truncstrat)
{
  KSP_PIPEGCR *pipegcr=(KSP_PIPEGCR*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ksp,truncstrat,2);
  pipegcr->truncstrat=truncstrat;
  PetscFunctionReturn(0);
}

/*@
  KSPPIPEGCRGetTruncationType - get the truncation strategy employed by PIPEGCR

  Not Collective

  KSP_FCD_TRUNC_TYPE_STANDARD uses all (up to mmax) stored directions
  KSP_FCD_TRUNC_TYPE_NOTAY uses the last max(1,mod(i,mmax)) directions at iteration i=0,1,..

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  truncstrat - the strategy type

  Options Database:
. -ksp_pipegcr_truncation_type <standard,notay> - which stored basis vectors to orthogonalize against

   Level: intermediate

.seealso: KSPPIPEGCR, KSPPIPEGCRSetTruncationType, KSPPIPEGCRTruncationType, KSPFCDTruncationType
@*/
PetscErrorCode KSPPIPEGCRGetTruncationType(KSP ksp,KSPFCDTruncationType *truncstrat)
{
  KSP_PIPEGCR *pipegcr=(KSP_PIPEGCR*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *truncstrat=pipegcr->truncstrat;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetFromOptions_PIPEGCR(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;
  KSP_PIPEGCR    *pipegcr = (KSP_PIPEGCR*)ksp->data;
  PetscInt       mmax,nprealloc;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP PIPEGCR options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_pipegcr_mmax","Number of search directions to storue","KSPPIPEGCRSetMmax",pipegcr->mmax,&mmax,&flg);CHKERRQ(ierr);
  if (flg) ierr = KSPPIPEGCRSetMmax(ksp,mmax);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_pipegcr_nprealloc","Number of directions to preallocate","KSPPIPEGCRSetNprealloc",pipegcr->nprealloc,&nprealloc,&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPPIPEGCRSetNprealloc(ksp,nprealloc);CHKERRQ(ierr); }
  ierr = PetscOptionsEnum("-ksp_pipegcr_truncation_type","Truncation approach for directions","KSPFCGSetTruncationType",KSPFCDTruncationTypes,(PetscEnum)pipegcr->truncstrat,(PetscEnum*)&pipegcr->truncstrat,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_pipegcr_unroll_w","Use unrolling of w","KSPPIPEGCRSetUnrollW",pipegcr->unroll_w,&pipegcr->unroll_w,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Force these parameters to not be EXTERN_C */
typedef PetscErrorCode (*KSPPIPEGCRModifyPCFunction)(KSP,PetscInt,PetscReal,void*);
typedef PetscErrorCode (*KSPPIPEGCRDestroyFunction)(void*);

static PetscErrorCode  KSPPIPEGCRSetModifyPC_PIPEGCR(KSP ksp,KSPPIPEGCRModifyPCFunction function,void *data,KSPPIPEGCRDestroyFunction destroy)
{
  KSP_PIPEGCR *pipegcr = (KSP_PIPEGCR*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  pipegcr->modifypc         = function;
  pipegcr->modifypc_destroy = destroy;
  pipegcr->modifypc_ctx     = data;
  PetscFunctionReturn(0);
}

/*@C
 KSPPIPEGCRSetModifyPC - Sets the routine used by PIPEGCR to modify the preconditioner.

 Logically Collective on ksp

 Input Parameters:
 +  ksp      - iterative context obtained from KSPCreate()
 .  function - user defined function to modify the preconditioner
 .  ctx      - user provided context for the modify preconditioner function
 -  destroy  - the function to use to destroy the user provided application context.

 Calling Sequence of function:
  PetscErrorCode function (KSP ksp, PetscInt n, PetscReal rnorm, void *ctx)

 ksp   - iterative context
 n     - the total number of PIPEGCR iterations that have occurred
 rnorm - 2-norm residual value
 ctx   - the user provided application context

 Level: intermediate

 Notes:
 The default modifypc routine is KSPPIPEGCRModifyPCNoChange()

 .seealso: KSPPIPEGCRModifyPCNoChange()

 @*/
PetscErrorCode  KSPPIPEGCRSetModifyPC(KSP ksp,PetscErrorCode (*function)(KSP,PetscInt,PetscReal,void*),void *data,PetscErrorCode (*destroy)(void*))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(ksp,"KSPPIPEGCRSetModifyPC_C",(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*),void *data,PetscErrorCode (*)(void*)),(ksp,function,data,destroy));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPPIPEGCR - Implements a Pipelined Generalized Conjugate Residual method.

  Options Database Keys:
+   -ksp_pipegcr_mmax <N>  - the max number of Krylov directions to orthogonalize against
.   -ksp_pipegcr_unroll_w - unroll w at the storage cost of a maximum of (mmax+1) extra vectors with the benefit of better pipelining (default: PETSC_TRUE)
.   -ksp_pipegcr_nprealloc <N> - the number of vectors to preallocated for storing Krylov directions. Once exhausted new directions are allocated blockwise (default: 5)
-   -ksp_pipegcr_truncation_type <standard,notay> - which previous search directions to orthogonalize against

  Notes:
    The PIPEGCR Krylov method supports non-symmetric matrices and permits the use of a preconditioner
    which may vary from one iteration to the next. Users can can define a method to vary the
    preconditioner between iterates via KSPPIPEGCRSetModifyPC().
    Restarts are solves with x0 not equal to zero. When a restart occurs, the initial starting
    solution is given by the current estimate for x which was obtained by the last restart
    iterations of the PIPEGCR algorithm.
    The method implemented requires at most the storage of 4 x mmax + 5 vectors, roughly twice as much as GCR.

    Only supports left preconditioning.

    The natural "norm" for this method is (u,Au), where u is the preconditioned residual. This norm is available at no additional computational cost, as with standard CG. Choosing preconditioned or unpreconditioned norm types involves a blocking reduction which prevents any benefit from pipelining.

  Reference:
    P. Sanan, S.M. Schnepp, and D.A. May,
    "Pipelined, Flexible Krylov Subspace Methods,"
    SIAM Journal on Scientific Computing 2016 38:5, C441-C470,
    DOI: 10.1137/15M1049130

   Level: intermediate

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPPIPEFGMRES, KSPPIPECG, KSPPIPECR, KSPPIPEFCG,KSPPIPEGCRSetTruncationType(),KSPPIPEGCRSetNprealloc(),KSPPIPEGCRSetUnrollW(),KSPPIPEGCRSetMmax()

M*/
PETSC_EXTERN PetscErrorCode KSPCreate_PIPEGCR(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_PIPEGCR    *pipegcr;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&pipegcr);CHKERRQ(ierr);
  pipegcr->mmax       = KSPPIPEGCR_DEFAULT_MMAX;
  pipegcr->nprealloc  = KSPPIPEGCR_DEFAULT_NPREALLOC;
  pipegcr->nvecs      = 0;
  pipegcr->vecb       = KSPPIPEGCR_DEFAULT_VECB;
  pipegcr->nchunks    = 0;
  pipegcr->truncstrat = KSPPIPEGCR_DEFAULT_TRUNCSTRAT;
  pipegcr->n_restarts = 0;
  pipegcr->unroll_w   = KSPPIPEGCR_DEFAULT_UNROLL_W;

  ksp->data       = (void*)pipegcr;

  /* natural norm is for free, precond+unprecond norm require non-overlapped reduction */
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);

  ksp->ops->setup          = KSPSetUp_PIPEGCR;
  ksp->ops->solve          = KSPSolve_PIPEGCR;
  ksp->ops->reset          = KSPReset_PIPEGCR;
  ksp->ops->destroy        = KSPDestroy_PIPEGCR;
  ksp->ops->view           = KSPView_PIPEGCR;
  ksp->ops->setfromoptions = KSPSetFromOptions_PIPEGCR;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;

  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPPIPEGCRSetModifyPC_C",KSPPIPEGCRSetModifyPC_PIPEGCR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
