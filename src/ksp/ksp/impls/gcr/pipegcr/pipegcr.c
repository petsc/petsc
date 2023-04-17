
#include "petscsys.h"
#include <../src/ksp/ksp/impls/gcr/pipegcr/pipegcrimpl.h> /*I  "petscksp.h"  I*/

static PetscBool  cited      = PETSC_FALSE;
static const char citation[] = "@article{SSM2016,\n"
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

#define KSPPIPEGCR_DEFAULT_MMAX       15
#define KSPPIPEGCR_DEFAULT_NPREALLOC  5
#define KSPPIPEGCR_DEFAULT_VECB       5
#define KSPPIPEGCR_DEFAULT_TRUNCSTRAT KSP_FCD_TRUNC_TYPE_NOTAY
#define KSPPIPEGCR_DEFAULT_UNROLL_W   PETSC_TRUE

#include <petscksp.h>

static PetscErrorCode KSPAllocateVectors_PIPEGCR(KSP ksp, PetscInt nvecsneeded, PetscInt chunksize)
{
  PetscInt     i;
  KSP_PIPEGCR *pipegcr;
  PetscInt     nnewvecs, nvecsprev;

  PetscFunctionBegin;
  pipegcr = (KSP_PIPEGCR *)ksp->data;

  /* Allocate enough new vectors to add chunksize new vectors, reach nvecsneedtotal, or to reach mmax+1, whichever is smallest */
  if (pipegcr->nvecs < PetscMin(pipegcr->mmax + 1, nvecsneeded)) {
    nvecsprev = pipegcr->nvecs;
    nnewvecs  = PetscMin(PetscMax(nvecsneeded - pipegcr->nvecs, chunksize), pipegcr->mmax + 1 - pipegcr->nvecs);
    PetscCall(KSPCreateVecs(ksp, nnewvecs, &pipegcr->ppvecs[pipegcr->nchunks], 0, NULL));
    PetscCall(KSPCreateVecs(ksp, nnewvecs, &pipegcr->psvecs[pipegcr->nchunks], 0, NULL));
    PetscCall(KSPCreateVecs(ksp, nnewvecs, &pipegcr->pqvecs[pipegcr->nchunks], 0, NULL));
    if (pipegcr->unroll_w) { PetscCall(KSPCreateVecs(ksp, nnewvecs, &pipegcr->ptvecs[pipegcr->nchunks], 0, NULL)); }
    pipegcr->nvecs += nnewvecs;
    for (i = 0; i < nnewvecs; i++) {
      pipegcr->qvecs[nvecsprev + i] = pipegcr->pqvecs[pipegcr->nchunks][i];
      pipegcr->pvecs[nvecsprev + i] = pipegcr->ppvecs[pipegcr->nchunks][i];
      pipegcr->svecs[nvecsprev + i] = pipegcr->psvecs[pipegcr->nchunks][i];
      if (pipegcr->unroll_w) pipegcr->tvecs[nvecsprev + i] = pipegcr->ptvecs[pipegcr->nchunks][i];
    }
    pipegcr->chunksizes[pipegcr->nchunks] = nnewvecs;
    pipegcr->nchunks++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSolve_PIPEGCR_cycle(KSP ksp)
{
  KSP_PIPEGCR *pipegcr = (KSP_PIPEGCR *)ksp->data;
  Mat          A, B;
  Vec          x, r, b, z, w, m, n, p, s, q, t, *redux;
  PetscInt     i, j, k, idx, kdx, mi;
  PetscScalar  alpha = 0.0, gamma, *betas, *dots;
  PetscReal    rnorm = 0.0, delta, *eta, *etas;

  PetscFunctionBegin;
  /* !!PS We have not checked these routines for use with complex numbers. The inner products
     are likely not defined correctly for that case */
  PetscCheck(!PetscDefined(USE_COMPLEX) || PetscDefined(SKIP_COMPLEX), PETSC_COMM_WORLD, PETSC_ERR_SUP, "PIPEFGMRES has not been implemented for use with complex scalars");

  PetscCall(KSPGetOperators(ksp, &A, &B));
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
  betas = dots; /* dots takes the result of all dot products of which the betas are a subset */

  /* cycle initial residual */
  PetscCall(KSP_MatMult(ksp, A, x, r));
  PetscCall(VecAYPX(r, -1.0, b));       /* r <- b - Ax */
  PetscCall(KSP_PCApply(ksp, r, z));    /* z <- B(r)   */
  PetscCall(KSP_MatMult(ksp, A, z, w)); /* w <- Az     */

  /* initialization of other variables and pipelining intermediates */
  PetscCall(VecCopy(z, p));
  PetscCall(KSP_MatMult(ksp, A, p, s));

  /* overlap initial computation of delta, gamma */
  redux[0] = w;
  redux[1] = r;
  PetscCall(VecMDotBegin(w, 2, redux, dots));                               /* Start split reductions for gamma = (w,r), delta = (w,w) */
  PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)s))); /* perform asynchronous reduction */
  PetscCall(KSP_PCApply(ksp, s, q));                                        /* q = B(s) */
  if (pipegcr->unroll_w) { PetscCall(KSP_MatMult(ksp, A, q, t)); /* t = Aq   */ }
  PetscCall(VecMDotEnd(w, 2, redux, dots)); /* Finish split reduction */
  delta   = PetscRealPart(dots[0]);
  etas[0] = delta;
  gamma   = dots[1];
  alpha   = gamma / delta;

  i = 0;
  do {
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->its++;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));

    /* update solution, residuals, .. */
    PetscCall(VecAXPY(x, +alpha, p));
    PetscCall(VecAXPY(r, -alpha, s));
    PetscCall(VecAXPY(z, -alpha, q));
    if (pipegcr->unroll_w) {
      PetscCall(VecAXPY(w, -alpha, t));
    } else {
      PetscCall(KSP_MatMult(ksp, A, z, w));
    }

    /* Computations of current iteration done */
    i++;

    if (pipegcr->modifypc) PetscCall((*pipegcr->modifypc)(ksp, ksp->its, ksp->rnorm, pipegcr->modifypc_ctx));

    /* If needbe, allocate a new chunk of vectors */
    PetscCall(KSPAllocateVectors_PIPEGCR(ksp, i + 1, pipegcr->vecb));

    /* Note that we wrap around and start clobbering old vectors */
    idx = i % (pipegcr->mmax + 1);
    p   = pipegcr->pvecs[idx];
    s   = pipegcr->svecs[idx];
    q   = pipegcr->qvecs[idx];
    if (pipegcr->unroll_w) t = pipegcr->tvecs[idx];
    eta = pipegcr->etas + idx;

    /* number of old directions to orthogonalize against */
    switch (pipegcr->truncstrat) {
    case KSP_FCD_TRUNC_TYPE_STANDARD:
      mi = pipegcr->mmax;
      break;
    case KSP_FCD_TRUNC_TYPE_NOTAY:
      mi = ((i - 1) % pipegcr->mmax) + 1;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unrecognized Truncation Strategy");
    }

    /* Pick old p,s,q,zeta in a way suitable for VecMDot */
    for (k = PetscMax(0, i - mi), j = 0; k < i; j++, k++) {
      kdx              = k % (pipegcr->mmax + 1);
      pipegcr->pold[j] = pipegcr->pvecs[kdx];
      pipegcr->sold[j] = pipegcr->svecs[kdx];
      pipegcr->qold[j] = pipegcr->qvecs[kdx];
      if (pipegcr->unroll_w) pipegcr->told[j] = pipegcr->tvecs[kdx];
      redux[j] = pipegcr->svecs[kdx];
    }
    /* If the above loop is not run redux contains only r and w => all beta_k = 0, only gamma, delta != 0 */
    redux[j]     = r;
    redux[j + 1] = w;

    /* Dot products */
    /* Start split reductions for beta_k = (w,s_k), gamma = (w,r), delta = (w,w) */
    PetscCall(VecMDotBegin(w, j + 2, redux, dots));
    PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)w)));

    /* B(w-r) + u stabilization */
    PetscCall(VecWAXPY(n, -1.0, r, w)); /* m = u + B(w-r): (a) ntmp = w-r              */
    PetscCall(KSP_PCApply(ksp, n, m));  /* m = u + B(w-r): (b) mtmp = B(ntmp) = B(w-r) */
    PetscCall(VecAXPY(m, 1.0, z));      /* m = u + B(w-r): (c) m = z + mtmp            */
    if (pipegcr->unroll_w) { PetscCall(KSP_MatMult(ksp, A, m, n)); /* n = Am                                      */ }

    /* Finish split reductions for beta_k = (w,s_k), gamma = (w,r), delta = (w,w) */
    PetscCall(VecMDotEnd(w, j + 2, redux, dots));
    gamma = dots[j];
    delta = PetscRealPart(dots[j + 1]);

    /* compute new residual norm.
       this cannot be done before this point so that the natural norm
       is available for free and the communication involved is overlapped */
    switch (ksp->normtype) {
    case KSP_NORM_PRECONDITIONED:
      PetscCall(VecNorm(z, NORM_2, &rnorm)); /* ||r|| <- sqrt(z'*z) */
      break;
    case KSP_NORM_UNPRECONDITIONED:
      PetscCall(VecNorm(r, NORM_2, &rnorm)); /* ||r|| <- sqrt(r'*r) */
      break;
    case KSP_NORM_NATURAL:
      rnorm = PetscSqrtReal(PetscAbsScalar(gamma)); /* ||r|| <- sqrt(r,w)  */
      break;
    case KSP_NORM_NONE:
      rnorm = 0.0;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "%s", KSPNormTypes[ksp->normtype]);
    }

    /* Check for convergence */
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->rnorm = rnorm;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
    PetscCall(KSPLogResidualHistory(ksp, rnorm));
    PetscCall(KSPMonitor(ksp, ksp->its, rnorm));
    PetscCall((*ksp->converged)(ksp, ksp->its, rnorm, &ksp->reason, ksp->cnvP));
    if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

    /* compute new eta and scale beta */
    *eta = 0.;
    for (k = PetscMax(0, i - mi), j = 0; k < i; j++, k++) {
      kdx = k % (pipegcr->mmax + 1);
      betas[j] /= -etas[kdx]; /* betak  /= etak */
      *eta -= ((PetscReal)(PetscAbsScalar(betas[j]) * PetscAbsScalar(betas[j]))) * etas[kdx];
      /* etaitmp = -betaik^2 * etak */
    }
    *eta += delta; /* etai    = delta -betaik^2 * etak */

    /* check breakdown of eta = (s,s) */
    if (*eta < 0.) {
      pipegcr->norm_breakdown = PETSC_TRUE;
      PetscCall(PetscInfo(ksp, "Restart due to square root breakdown at it = %" PetscInt_FMT "\n", ksp->its));
      break;
    } else {
      alpha = gamma / (*eta); /* alpha = gamma/etai */
    }

    /* project out stored search directions using classical G-S */
    PetscCall(VecCopy(z, p));
    PetscCall(VecCopy(w, s));
    PetscCall(VecCopy(m, q));
    if (pipegcr->unroll_w) {
      PetscCall(VecCopy(n, t));
      PetscCall(VecMAXPY(t, j, betas, pipegcr->told)); /* ti <- n  - sum_k beta_k t_k */
    }
    PetscCall(VecMAXPY(p, j, betas, pipegcr->pold)); /* pi <- ui - sum_k beta_k p_k */
    PetscCall(VecMAXPY(s, j, betas, pipegcr->sold)); /* si <- wi - sum_k beta_k s_k */
    PetscCall(VecMAXPY(q, j, betas, pipegcr->qold)); /* qi <- m  - sum_k beta_k q_k */

  } while (ksp->its < ksp->max_it);
  if (ksp->its >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSolve_PIPEGCR(KSP ksp)
{
  KSP_PIPEGCR *pipegcr = (KSP_PIPEGCR *)ksp->data;
  Mat          A, B;
  Vec          x, b, r, z, w;
  PetscScalar  gamma;
  PetscReal    rnorm = 0.0;
  PetscBool    issym;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(citation, &cited));

  PetscCall(KSPGetOperators(ksp, &A, &B));
  x = ksp->vec_sol;
  b = ksp->vec_rhs;
  r = ksp->work[0];
  z = ksp->work[1];
  w = ksp->work[2]; /* w = Az = AB(r)                 (pipelining intermediate) */

  /* compute initial residual */
  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp, A, x, r));
    PetscCall(VecAYPX(r, -1.0, b)); /* r <- b - Ax       */
  } else {
    PetscCall(VecCopy(b, r)); /* r <- b            */
  }

  /* initial residual norm */
  PetscCall(KSP_PCApply(ksp, r, z));    /* z <- B(r)         */
  PetscCall(KSP_MatMult(ksp, A, z, w)); /* w <- Az           */
  PetscCall(VecDot(r, w, &gamma));      /* gamma = (r,w)     */

  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    PetscCall(VecNorm(z, NORM_2, &rnorm)); /* ||r|| <- sqrt(z'*z) */
    break;
  case KSP_NORM_UNPRECONDITIONED:
    PetscCall(VecNorm(r, NORM_2, &rnorm)); /* ||r|| <- sqrt(r'*r) */
    break;
  case KSP_NORM_NATURAL:
    rnorm = PetscSqrtReal(PetscAbsScalar(gamma)); /* ||r|| <- sqrt(r,w)  */
    break;
  case KSP_NORM_NONE:
    rnorm = 0.0;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "%s", KSPNormTypes[ksp->normtype]);
  }

  /* Is A symmetric? */
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &issym, MATSBAIJ, MATSEQSBAIJ, MATMPISBAIJ, ""));
  if (!issym) PetscCall(PetscInfo(A, "Matrix type is not any of MATSBAIJ,MATSEQSBAIJ,MATMPISBAIJ. Is matrix A symmetric (as required by CR methods)?"));

  /* logging */
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its    = 0;
  ksp->rnorm0 = rnorm;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  PetscCall(KSPLogResidualHistory(ksp, ksp->rnorm0));
  PetscCall(KSPMonitor(ksp, ksp->its, ksp->rnorm0));
  PetscCall((*ksp->converged)(ksp, ksp->its, ksp->rnorm0, &ksp->reason, ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  do {
    PetscCall(KSPSolve_PIPEGCR_cycle(ksp));
    if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);
    if (pipegcr->norm_breakdown) {
      pipegcr->n_restarts++;
      pipegcr->norm_breakdown = PETSC_FALSE;
    }
  } while (ksp->its < ksp->max_it);

  if (ksp->its >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPView_PIPEGCR(KSP ksp, PetscViewer viewer)
{
  KSP_PIPEGCR *pipegcr = (KSP_PIPEGCR *)ksp->data;
  PetscBool    isascii, isstring;
  const char  *truncstr;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));

  if (pipegcr->truncstrat == KSP_FCD_TRUNC_TYPE_STANDARD) {
    truncstr = "Using standard truncation strategy";
  } else if (pipegcr->truncstrat == KSP_FCD_TRUNC_TYPE_NOTAY) {
    truncstr = "Using Notay's truncation strategy";
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Undefined FCD truncation strategy");

  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  max previous directions = %" PetscInt_FMT "\n", pipegcr->mmax));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  preallocated %" PetscInt_FMT " directions\n", PetscMin(pipegcr->nprealloc, pipegcr->mmax + 1)));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  %s\n", truncstr));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  w unrolling = %s \n", PetscBools[pipegcr->unroll_w]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  restarts performed = %" PetscInt_FMT " \n", pipegcr->n_restarts));
  } else if (isstring) {
    PetscCall(PetscViewerStringSPrintf(viewer, "max previous directions = %" PetscInt_FMT ", preallocated %" PetscInt_FMT " directions, %s truncation strategy", pipegcr->mmax, pipegcr->nprealloc, truncstr));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSetUp_PIPEGCR(KSP ksp)
{
  KSP_PIPEGCR   *pipegcr = (KSP_PIPEGCR *)ksp->data;
  Mat            A;
  PetscBool      diagonalscale;
  const PetscInt nworkstd = 5;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
  PetscCheck(!diagonalscale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);

  PetscCall(KSPGetOperators(ksp, &A, NULL));

  /* Allocate "standard" work vectors */
  PetscCall(KSPSetWorkVecs(ksp, nworkstd));

  /* Allocated space for pointers to additional work vectors
    note that mmax is the number of previous directions, so we add 1 for the current direction */
  PetscCall(PetscMalloc6(pipegcr->mmax + 1, &(pipegcr->pvecs), pipegcr->mmax + 1, &(pipegcr->ppvecs), pipegcr->mmax + 1, &(pipegcr->svecs), pipegcr->mmax + 1, &(pipegcr->psvecs), pipegcr->mmax + 1, &(pipegcr->qvecs), pipegcr->mmax + 1, &(pipegcr->pqvecs)));
  if (pipegcr->unroll_w) PetscCall(PetscMalloc3(pipegcr->mmax + 1, &(pipegcr->tvecs), pipegcr->mmax + 1, &(pipegcr->ptvecs), pipegcr->mmax + 2, &(pipegcr->told)));
  PetscCall(PetscMalloc4(pipegcr->mmax + 2, &(pipegcr->pold), pipegcr->mmax + 2, &(pipegcr->sold), pipegcr->mmax + 2, &(pipegcr->qold), pipegcr->mmax + 2, &(pipegcr->chunksizes)));
  PetscCall(PetscMalloc3(pipegcr->mmax + 2, &(pipegcr->dots), pipegcr->mmax + 1, &(pipegcr->etas), pipegcr->mmax + 2, &(pipegcr->redux)));
  /* If the requested number of preallocated vectors is greater than mmax reduce nprealloc */
  if (pipegcr->nprealloc > pipegcr->mmax + 1) PetscCall(PetscInfo(NULL, "Requested nprealloc=%" PetscInt_FMT " is greater than m_max+1=%" PetscInt_FMT ". Resetting nprealloc = m_max+1.\n", pipegcr->nprealloc, pipegcr->mmax + 1));

  /* Preallocate additional work vectors */
  PetscCall(KSPAllocateVectors_PIPEGCR(ksp, pipegcr->nprealloc, pipegcr->nprealloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPReset_PIPEGCR(KSP ksp)
{
  KSP_PIPEGCR *pipegcr = (KSP_PIPEGCR *)ksp->data;

  PetscFunctionBegin;
  if (pipegcr->modifypc_destroy) PetscCall((*pipegcr->modifypc_destroy)(pipegcr->modifypc_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPDestroy_PIPEGCR(KSP ksp)
{
  PetscInt     i;
  KSP_PIPEGCR *pipegcr = (KSP_PIPEGCR *)ksp->data;

  PetscFunctionBegin;
  PetscCall(VecDestroyVecs(ksp->nwork, &ksp->work)); /* Destroy "standard" work vecs */

  /* Destroy vectors for old directions and the arrays that manage pointers to them */
  if (pipegcr->nvecs) {
    for (i = 0; i < pipegcr->nchunks; i++) {
      PetscCall(VecDestroyVecs(pipegcr->chunksizes[i], &pipegcr->ppvecs[i]));
      PetscCall(VecDestroyVecs(pipegcr->chunksizes[i], &pipegcr->psvecs[i]));
      PetscCall(VecDestroyVecs(pipegcr->chunksizes[i], &pipegcr->pqvecs[i]));
      if (pipegcr->unroll_w) PetscCall(VecDestroyVecs(pipegcr->chunksizes[i], &pipegcr->ptvecs[i]));
    }
  }

  PetscCall(PetscFree6(pipegcr->pvecs, pipegcr->ppvecs, pipegcr->svecs, pipegcr->psvecs, pipegcr->qvecs, pipegcr->pqvecs));
  PetscCall(PetscFree4(pipegcr->pold, pipegcr->sold, pipegcr->qold, pipegcr->chunksizes));
  PetscCall(PetscFree3(pipegcr->dots, pipegcr->etas, pipegcr->redux));
  if (pipegcr->unroll_w) PetscCall(PetscFree3(pipegcr->tvecs, pipegcr->ptvecs, pipegcr->told));

  PetscCall(KSPReset_PIPEGCR(ksp));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPPIPEGCRSetModifyPC_C", NULL));
  PetscCall(KSPDestroyDefault(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPPIPEGCRSetUnrollW - Set to `PETSC_TRUE` to use `KSPPIPEGCR` with unrolling of the w vector

  Logically Collective

  Input Parameters:
+  ksp - the Krylov space context
-  unroll_w - use unrolling

  Level: intermediate

  Options Database Key:
. -ksp_pipegcr_unroll_w <bool> -  use unrolling

.seealso: [](chapter_ksp), `KSPPIPEGCR`, `KSPPIPEGCRSetTruncationType()`, `KSPPIPEGCRSetNprealloc()`, `KSPPIPEGCRGetUnrollW()`
@*/
PetscErrorCode KSPPIPEGCRSetUnrollW(KSP ksp, PetscBool unroll_w)
{
  KSP_PIPEGCR *pipegcr = (KSP_PIPEGCR *)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveBool(ksp, unroll_w, 2);
  pipegcr->unroll_w = unroll_w;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPPIPEGCRGetUnrollW - Get information on `KSPPIPEGCR` if it uses unrolling the w vector

  Logically Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  unroll_w - `KSPPIPEGCR` uses unrolling (bool)

  Level: intermediate

.seealso: [](chapter_ksp), `KSPPIPEGCR`, `KSPPIPEGCRGetTruncationType()`, `KSPPIPEGCRGetNprealloc()`, `KSPPIPEGCRSetUnrollW()`
@*/
PetscErrorCode KSPPIPEGCRGetUnrollW(KSP ksp, PetscBool *unroll_w)
{
  KSP_PIPEGCR *pipegcr = (KSP_PIPEGCR *)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  *unroll_w = pipegcr->unroll_w;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPPIPEGCRSetMmax - set the maximum number of previous directions `KSPPIPEGCR` will store for orthogonalization

  Logically Collective

  Input Parameters:
+  ksp - the Krylov space context
-  mmax - the maximum number of previous directions to orthogonalize against

  Options Database Key:
. -ksp_pipegcr_mmax <N> - maximum number of previous directions

  Level: intermediate

  Note:
   mmax + 1 directions are stored (mmax previous ones along with a current one)
  and whether all are used in each iteration also depends on the truncation strategy
  (see `KSPPIPEGCRSetTruncationType`)

.seealso: [](chapter_ksp), `KSPPIPEGCR`, `KSPPIPEGCRSetTruncationType()`, `KSPPIPEGCRSetNprealloc()`
@*/
PetscErrorCode KSPPIPEGCRSetMmax(KSP ksp, PetscInt mmax)
{
  KSP_PIPEGCR *pipegcr = (KSP_PIPEGCR *)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveInt(ksp, mmax, 2);
  pipegcr->mmax = mmax;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPPIPEGCRGetMmax - get the maximum number of previous directions `KSPPIPEGCR` will store

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  mmax - the maximum number of previous directions allowed for orthogonalization

   Level: intermediate

.seealso: [](chapter_ksp), `KSPPIPEGCR`, `KSPPIPEGCRGetTruncationType()`, `KSPPIPEGCRGetNprealloc()`, `KSPPIPEGCRSetMmax()`
@*/

PetscErrorCode KSPPIPEGCRGetMmax(KSP ksp, PetscInt *mmax)
{
  KSP_PIPEGCR *pipegcr = (KSP_PIPEGCR *)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  *mmax = pipegcr->mmax;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPPIPEGCRSetNprealloc - set the number of directions to preallocate with `KSPPIPEGCR`

  Logically Collective

  Input Parameters:
+  ksp - the Krylov space context
-  nprealloc - the number of vectors to preallocate

  Level: advanced

  Options Database Key:
. -ksp_pipegcr_nprealloc <N> - number of vectors to preallocate

.seealso: [](chapter_ksp), `KSPPIPEGCR`, `KSPPIPEGCRGetTruncationType()`, `KSPPIPEGCRGetNprealloc()`
@*/
PetscErrorCode KSPPIPEGCRSetNprealloc(KSP ksp, PetscInt nprealloc)
{
  KSP_PIPEGCR *pipegcr = (KSP_PIPEGCR *)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveInt(ksp, nprealloc, 2);
  pipegcr->nprealloc = nprealloc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPPIPEGCRGetNprealloc - get the number of directions preallocate by `KSPPIPEGCR`

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  nprealloc - the number of directions preallocated

   Level: advanced

.seealso: [](chapter_ksp), `KSPPIPEGCR`, `KSPPIPEGCRGetTruncationType()`, `KSPPIPEGCRSetNprealloc()`
@*/
PetscErrorCode KSPPIPEGCRGetNprealloc(KSP ksp, PetscInt *nprealloc)
{
  KSP_PIPEGCR *pipegcr = (KSP_PIPEGCR *)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  *nprealloc = pipegcr->nprealloc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPPIPEGCRSetTruncationType - specify how many of its stored previous directions `KSPPIPEGCR` uses during orthogonalization

  Logically Collective

  Input Parameters:
+  ksp - the Krylov space context
-  truncstrat - the choice of strategy
.vb
  KSP_FCD_TRUNC_TYPE_STANDARD uses all (up to mmax) stored directions
  KSP_FCD_TRUNC_TYPE_NOTAY uses the last max(1,mod(i,mmax)) directions at iteration i=0,1,..
.ve

  Options Database Key:
. -ksp_pipegcr_truncation_type <standard,notay> - which stored basis vectors to orthogonalize against

  Level: intermediate

.seealso: [](chapter_ksp), `KSPPIPEGCR`, `KSPPIPEGCRSetTruncationType`, `KSPPIPEGCRTruncationType`, `KSPFCDTruncationType`
@*/
PetscErrorCode KSPPIPEGCRSetTruncationType(KSP ksp, KSPFCDTruncationType truncstrat)
{
  KSP_PIPEGCR *pipegcr = (KSP_PIPEGCR *)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(ksp, truncstrat, 2);
  pipegcr->truncstrat = truncstrat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPPIPEGCRGetTruncationType - get the truncation strategy employed by `KSPPIPEGCR`

  Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  truncstrat - the strategy type
.vb
  KSP_FCD_TRUNC_TYPE_STANDARD uses all (up to mmax) stored directions
  KSP_FCD_TRUNC_TYPE_NOTAY uses the last max(1,mod(i,mmax)) directions at iteration i=0,1,..
.ve

  Options Database Key:
. -ksp_pipegcr_truncation_type <standard,notay> - which stored basis vectors to orthogonalize against

   Level: intermediate

.seealso: [](chapter_ksp), `KSPPIPEGCR`, `KSPPIPEGCRSetTruncationType`, `KSPPIPEGCRTruncationType`, `KSPFCDTruncationType`
@*/
PetscErrorCode KSPPIPEGCRGetTruncationType(KSP ksp, KSPFCDTruncationType *truncstrat)
{
  KSP_PIPEGCR *pipegcr = (KSP_PIPEGCR *)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  *truncstrat = pipegcr->truncstrat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSetFromOptions_PIPEGCR(KSP ksp, PetscOptionItems *PetscOptionsObject)
{
  KSP_PIPEGCR *pipegcr = (KSP_PIPEGCR *)ksp->data;
  PetscInt     mmax, nprealloc;
  PetscBool    flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "KSP PIPEGCR options");
  PetscCall(PetscOptionsInt("-ksp_pipegcr_mmax", "Number of search directions to storue", "KSPPIPEGCRSetMmax", pipegcr->mmax, &mmax, &flg));
  if (flg) PetscCall(KSPPIPEGCRSetMmax(ksp, mmax));
  PetscCall(PetscOptionsInt("-ksp_pipegcr_nprealloc", "Number of directions to preallocate", "KSPPIPEGCRSetNprealloc", pipegcr->nprealloc, &nprealloc, &flg));
  if (flg) PetscCall(KSPPIPEGCRSetNprealloc(ksp, nprealloc));
  PetscCall(PetscOptionsEnum("-ksp_pipegcr_truncation_type", "Truncation approach for directions", "KSPFCGSetTruncationType", KSPFCDTruncationTypes, (PetscEnum)pipegcr->truncstrat, (PetscEnum *)&pipegcr->truncstrat, NULL));
  PetscCall(PetscOptionsBool("-ksp_pipegcr_unroll_w", "Use unrolling of w", "KSPPIPEGCRSetUnrollW", pipegcr->unroll_w, &pipegcr->unroll_w, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Force these parameters to not be EXTERN_C */
typedef PetscErrorCode (*KSPPIPEGCRModifyPCFunction)(KSP, PetscInt, PetscReal, void *);
typedef PetscErrorCode (*KSPPIPEGCRDestroyFunction)(void *);

static PetscErrorCode KSPPIPEGCRSetModifyPC_PIPEGCR(KSP ksp, KSPPIPEGCRModifyPCFunction function, void *data, KSPPIPEGCRDestroyFunction destroy)
{
  KSP_PIPEGCR *pipegcr = (KSP_PIPEGCR *)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  pipegcr->modifypc         = function;
  pipegcr->modifypc_destroy = destroy;
  pipegcr->modifypc_ctx     = data;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
 KSPPIPEGCRSetModifyPC - Sets the routine used by `KSPPIPEGCR` to modify the preconditioner at each iteration

 Logically Collective

 Input Parameters:
 +  ksp      - iterative context obtained from KSPCreate()
 .  function - user defined function to modify the preconditioner
 .  ctx      - user provided context for the modify preconditioner function
 -  destroy  - the function to use to destroy the user provided application context.

 Calling Sequence of `function`:
$  PetscErrorCode function (KSP ksp, PetscInt n, PetscReal rnorm, void *ctx)
+ ksp   - iterative context
. n     - the total number of PIPEGCR iterations that have occurred
. rnorm - 2-norm residual value
- ctx   - the user provided application context

 Calling Sequence of `destroy`:
$ PetscErrorCode destroy(void *ctx)

 Level: intermediate

 Notes:
 The default modifypc routine is `KSPPIPEGCRModifyPCNoChange()`

.seealso: [](chapter_ksp), `KSPPIPEGCR`, `KSPPIPEGCRModifyPCNoChange()`
 @*/
PetscErrorCode KSPPIPEGCRSetModifyPC(KSP ksp, PetscErrorCode (*function)(KSP, PetscInt, PetscReal, void *), void *data, PetscErrorCode (*destroy)(void *))
{
  PetscFunctionBegin;
  PetscUseMethod(ksp, "KSPPIPEGCRSetModifyPC_C", (KSP, PetscErrorCode(*)(KSP, PetscInt, PetscReal, void *), void *data, PetscErrorCode (*)(void *)), (ksp, function, data, destroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     KSPPIPEGCR - Implements a Pipelined Generalized Conjugate Residual method. [](sec_flexibleksp). [](sec_pipelineksp)

  Options Database Keys:
+   -ksp_pipegcr_mmax <N>  - the max number of Krylov directions to orthogonalize against
.   -ksp_pipegcr_unroll_w - unroll w at the storage cost of a maximum of (mmax+1) extra vectors with the benefit of better pipelining (default: `PETSC_TRUE`)
.   -ksp_pipegcr_nprealloc <N> - the number of vectors to preallocated for storing Krylov directions. Once exhausted new directions are allocated blockwise (default: 5)
-   -ksp_pipegcr_truncation_type <standard,notay> - which previous search directions to orthogonalize against

  Level: intermediate

  Notes:
    The `KSPPIPEGCR` Krylov method supports non-symmetric matrices and permits the use of a preconditioner
    which may vary from one iteration to the next. Users can can define a method to vary the
    preconditioner between iterates via `KSPPIPEGCRSetModifyPC()`.
    Restarts are solves with x0 not equal to zero. When a restart occurs, the initial starting
    solution is given by the current estimate for x which was obtained by the last restart
    iterations of the PIPEGCR algorithm.
    The method implemented requires at most the storage of 4 x mmax + 5 vectors, roughly twice as much as GCR.

    Only supports left preconditioning.

    The natural "norm" for this method is (u,Au), where u is the preconditioned residual. This norm is available at no additional computational cost, as with standard CG.
    Choosing preconditioned or unpreconditioned norm types involves a blocking reduction which prevents any benefit from pipelining.

   MPI configuration may be necessary for reductions to make asynchronous progress, which is important for performance of pipelined methods.
   See [](doc_faq_pipelined)

  Contributed by:
  Sascha M. Schnepp and Patrick Sanan

  Reference:
    P. Sanan, S.M. Schnepp, and D.A. May,
    "Pipelined, Flexible Krylov Subspace Methods,"
    SIAM Journal on Scientific Computing 2016 38:5, C441-C470,
    DOI: 10.1137/15M1049130

.seealso: [](chapter_ksp), [](sec_flexibleksp), [](sec_pipelineksp), [](doc_faq_pipelined), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`,
          `KSPPIPEFGMRES`, `KSPPIPECG`, `KSPPIPECR`, `KSPPIPEFCG`, `KSPPIPEGCRSetTruncationType()`, `KSPPIPEGCRSetNprealloc()`, `KSPPIPEGCRSetUnrollW()`, `KSPPIPEGCRSetMmax()`
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_PIPEGCR(KSP ksp)
{
  KSP_PIPEGCR *pipegcr;

  PetscFunctionBegin;
  PetscCall(PetscNew(&pipegcr));
  pipegcr->mmax       = KSPPIPEGCR_DEFAULT_MMAX;
  pipegcr->nprealloc  = KSPPIPEGCR_DEFAULT_NPREALLOC;
  pipegcr->nvecs      = 0;
  pipegcr->vecb       = KSPPIPEGCR_DEFAULT_VECB;
  pipegcr->nchunks    = 0;
  pipegcr->truncstrat = KSPPIPEGCR_DEFAULT_TRUNCSTRAT;
  pipegcr->n_restarts = 0;
  pipegcr->unroll_w   = KSPPIPEGCR_DEFAULT_UNROLL_W;

  ksp->data = (void *)pipegcr;

  /* natural norm is for free, precond+unprecond norm require non-overlapped reduction */
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NATURAL, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 1));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT, 1));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_LEFT, 1));

  ksp->ops->setup          = KSPSetUp_PIPEGCR;
  ksp->ops->solve          = KSPSolve_PIPEGCR;
  ksp->ops->reset          = KSPReset_PIPEGCR;
  ksp->ops->destroy        = KSPDestroy_PIPEGCR;
  ksp->ops->view           = KSPView_PIPEGCR;
  ksp->ops->setfromoptions = KSPSetFromOptions_PIPEGCR;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;

  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPPIPEGCRSetModifyPC_C", KSPPIPEGCRSetModifyPC_PIPEGCR));
  PetscFunctionReturn(PETSC_SUCCESS);
}
