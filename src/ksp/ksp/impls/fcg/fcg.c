/*
    This file implements the FCG (Flexible Conjugate Gradient) method

*/

#include <../src/ksp/ksp/impls/fcg/fcgimpl.h>       /*I  "petscksp.h"  I*/
extern PetscErrorCode KSPComputeExtremeSingularValues_CG(KSP,PetscReal*,PetscReal*);
extern PetscErrorCode KSPComputeEigenvalues_CG(KSP,PetscInt,PetscReal*,PetscReal*,PetscInt*);

#define KSPFCG_DEFAULT_MMAX 30          /* maximum number of search directions to keep */
#define KSPFCG_DEFAULT_NPREALLOC 10     /* number of search directions to preallocate */
#define KSPFCG_DEFAULT_VECB 5           /* number of search directions to allocate each time new direction vectors are needed */
#define KSPFCG_DEFAULT_TRUNCSTRAT KSP_FCD_TRUNC_TYPE_NOTAY

static PetscErrorCode KSPAllocateVectors_FCG(KSP ksp, PetscInt nvecsneeded, PetscInt chunksize)
{
  PetscInt        i;
  KSP_FCG         *fcg = (KSP_FCG*)ksp->data;
  PetscInt        nnewvecs, nvecsprev;

  PetscFunctionBegin;
  /* Allocate enough new vectors to add chunksize new vectors, reach nvecsneedtotal, or to reach mmax+1, whichever is smallest */
  if (fcg->nvecs < PetscMin(fcg->mmax+1,nvecsneeded)) {
    nvecsprev = fcg->nvecs;
    nnewvecs = PetscMin(PetscMax(nvecsneeded-fcg->nvecs,chunksize),fcg->mmax+1-fcg->nvecs);
    PetscCall(KSPCreateVecs(ksp,nnewvecs,&fcg->pCvecs[fcg->nchunks],0,NULL));
    PetscCall(PetscLogObjectParents((PetscObject)ksp,nnewvecs,fcg->pCvecs[fcg->nchunks]));
    PetscCall(KSPCreateVecs(ksp,nnewvecs,&fcg->pPvecs[fcg->nchunks],0,NULL));
    PetscCall(PetscLogObjectParents((PetscObject)ksp,nnewvecs,fcg->pPvecs[fcg->nchunks]));
    fcg->nvecs += nnewvecs;
    for (i=0;i<nnewvecs;++i) {
      fcg->Cvecs[nvecsprev + i] = fcg->pCvecs[fcg->nchunks][i];
      fcg->Pvecs[nvecsprev + i] = fcg->pPvecs[fcg->nchunks][i];
    }
    fcg->chunksizes[fcg->nchunks] = nnewvecs;
    ++fcg->nchunks;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode    KSPSetUp_FCG(KSP ksp)
{
  KSP_FCG        *fcg = (KSP_FCG*)ksp->data;
  PetscInt       maxit = ksp->max_it;
  const PetscInt nworkstd = 2;

  PetscFunctionBegin;

  /* Allocate "standard" work vectors (not including the basis and transformed basis vectors) */
  PetscCall(KSPSetWorkVecs(ksp,nworkstd));

  /* Allocated space for pointers to additional work vectors
   note that mmax is the number of previous directions, so we add 1 for the current direction,
   and an extra 1 for the prealloc (which might be empty) */
  PetscCall(PetscMalloc5(fcg->mmax+1,&fcg->Pvecs,fcg->mmax+1,&fcg->Cvecs,fcg->mmax+1,&fcg->pPvecs,fcg->mmax+1,&fcg->pCvecs,fcg->mmax+2,&fcg->chunksizes));
  PetscCall(PetscLogObjectMemory((PetscObject)ksp,2*(fcg->mmax+1)*sizeof(Vec*) + 2*(fcg->mmax + 1)*sizeof(Vec**) + (fcg->mmax + 2)*sizeof(PetscInt)));

  /* If the requested number of preallocated vectors is greater than mmax reduce nprealloc */
  if (fcg->nprealloc > fcg->mmax+1) {
    PetscCall(PetscInfo(NULL,"Requested nprealloc=%d is greater than m_max+1=%d. Resetting nprealloc = m_max+1.\n",fcg->nprealloc, fcg->mmax+1));
  }

  /* Preallocate additional work vectors */
  PetscCall(KSPAllocateVectors_FCG(ksp,fcg->nprealloc,fcg->nprealloc));
  /*
  If user requested computations of eigenvalues then allocate work
  work space needed
  */
  if (ksp->calc_sings) {
    /* get space to store tridiagonal matrix for Lanczos */
    PetscCall(PetscMalloc4(maxit,&fcg->e,maxit,&fcg->d,maxit,&fcg->ee,maxit,&fcg->dd));
    PetscCall(PetscLogObjectMemory((PetscObject)ksp,2*(maxit+1)*(sizeof(PetscScalar)+sizeof(PetscReal))));

    ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_CG;
    ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_CG;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_FCG(KSP ksp)
{
  PetscInt       i,k,idx,mi;
  KSP_FCG        *fcg = (KSP_FCG*)ksp->data;
  PetscScalar    alpha=0.0,beta = 0.0,dpi,s;
  PetscReal      dp=0.0;
  Vec            B,R,Z,X,Pcurr,Ccurr;
  Mat            Amat,Pmat;
  PetscInt       eigs = ksp->calc_sings; /* Variables for eigen estimation - START*/
  PetscInt       stored_max_it = ksp->max_it;
  PetscScalar    alphaold = 0,betaold = 1.0,*e = NULL,*d = NULL;/* Variables for eigen estimation  - FINISH */

  PetscFunctionBegin;

#define VecXDot(x,y,a) (((fcg->type) == (KSP_CG_HERMITIAN)) ? VecDot(x,y,a) : VecTDot(x,y,a))
#define VecXMDot(a,b,c,d) (((fcg->type) == (KSP_CG_HERMITIAN)) ? VecMDot(a,b,c,d) : VecMTDot(a,b,c,d))

  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Z             = ksp->work[1];

  PetscCall(PCGetOperators(ksp->pc,&Amat,&Pmat));
  if (eigs) {e = fcg->e; d = fcg->d; e[0] = 0.0; }
  /* Compute initial residual needed for convergence check*/
  ksp->its = 0;
  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp,Amat,X,R));
    PetscCall(VecAYPX(R,-1.0,B));                    /*   r <- b - Ax     */
  } else {
    PetscCall(VecCopy(B,R));                         /*   r <- b (x is 0) */
  }
  switch (ksp->normtype) {
    case KSP_NORM_PRECONDITIONED:
      PetscCall(KSP_PCApply(ksp,R,Z));               /*   z <- Br         */
      PetscCall(VecNorm(Z,NORM_2,&dp));              /*   dp <- dqrt(z'*z) = sqrt(e'*A'*B'*B*A*e)     */
      KSPCheckNorm(ksp,dp);
      break;
    case KSP_NORM_UNPRECONDITIONED:
      PetscCall(VecNorm(R,NORM_2,&dp));              /*   dp <- sqrt(r'*r) = sqrt(e'*A'*A*e)     */
      KSPCheckNorm(ksp,dp);
      break;
    case KSP_NORM_NATURAL:
      PetscCall(KSP_PCApply(ksp,R,Z));               /*   z <- Br         */
      PetscCall(VecXDot(R,Z,&s));
      KSPCheckDot(ksp,s);
      dp = PetscSqrtReal(PetscAbsScalar(s));                   /*   dp <- sqrt(r'*z) = sqrt(e'*A'*B*A*e)  */
      break;
    case KSP_NORM_NONE:
      dp = 0.0;
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }

  /* Initial Convergence Check */
  PetscCall(KSPLogResidualHistory(ksp,dp));
  PetscCall(KSPMonitor(ksp,0,dp));
  ksp->rnorm = dp;
  if (ksp->normtype == KSP_NORM_NONE) {
    PetscCall(KSPConvergedSkip(ksp,0,dp,&ksp->reason,ksp->cnvP));
  } else {
    PetscCall((*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP));
  }
  if (ksp->reason) PetscFunctionReturn(0);

  /* Apply PC if not already done for convergence check */
  if (ksp->normtype == KSP_NORM_UNPRECONDITIONED || ksp->normtype == KSP_NORM_NONE) {
    PetscCall(KSP_PCApply(ksp,R,Z));               /*   z <- Br         */
  }

  i = 0;
  do {
    ksp->its = i+1;

    /*  If needbe, allocate a new chunk of vectors in P and C */
    PetscCall(KSPAllocateVectors_FCG(ksp,i+1,fcg->vecb));

    /* Note that we wrap around and start clobbering old vectors */
    idx = i % (fcg->mmax+1);
    Pcurr = fcg->Pvecs[idx];
    Ccurr = fcg->Cvecs[idx];

    /* number of old directions to orthogonalize against */
    switch(fcg->truncstrat) {
      case KSP_FCD_TRUNC_TYPE_STANDARD:
        mi = fcg->mmax;
        break;
      case KSP_FCD_TRUNC_TYPE_NOTAY:
        mi = ((i-1) % fcg->mmax)+1;
        break;
      default:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unrecognized Truncation Strategy");
    }

    /* Compute a new column of P (Currently does not support modified G-S or iterative refinement)*/
    PetscCall(VecCopy(Z,Pcurr));

    {
      PetscInt l,ndots;

      l = PetscMax(0,i-mi);
      ndots = i-l;
      if (ndots) {
        PetscInt    j;
        Vec         *Pold,  *Cold;
        PetscScalar *dots;

        PetscCall(PetscMalloc3(ndots,&dots,ndots,&Cold,ndots,&Pold));
        for (k=l,j=0;j<ndots;++k,++j) {
          idx = k % (fcg->mmax+1);
          Cold[j] = fcg->Cvecs[idx];
          Pold[j] = fcg->Pvecs[idx];
        }
        PetscCall(VecXMDot(Z,ndots,Cold,dots));
        for (k=0;k<ndots;++k) {
          dots[k] = -dots[k];
        }
        PetscCall(VecMAXPY(Pcurr,ndots,dots,Pold));
        PetscCall(PetscFree3(dots,Cold,Pold));
      }
    }

    /* Update X and R */
    betaold = beta;
    PetscCall(VecXDot(Pcurr,R,&beta));                 /*  beta <- pi'*r       */
    KSPCheckDot(ksp,beta);
    PetscCall(KSP_MatMult(ksp,Amat,Pcurr,Ccurr));      /*  w <- A*pi (stored in ci)   */
    PetscCall(VecXDot(Pcurr,Ccurr,&dpi));              /*  dpi <- pi'*w        */
    alphaold = alpha;
    alpha = beta / dpi;                                          /*  alpha <- beta/dpi    */
    PetscCall(VecAXPY(X,alpha,Pcurr));                 /*  x <- x + alpha * pi  */
    PetscCall(VecAXPY(R,-alpha,Ccurr));                /*  r <- r - alpha * wi  */

    /* Compute norm for convergence check */
    switch (ksp->normtype) {
      case KSP_NORM_PRECONDITIONED:
        PetscCall(KSP_PCApply(ksp,R,Z));               /*   z <- Br             */
        PetscCall(VecNorm(Z,NORM_2,&dp));              /*   dp <- sqrt(z'*z) = sqrt(e'*A'*B'*B*A*e)  */
        KSPCheckNorm(ksp,dp);
      break;
      case KSP_NORM_UNPRECONDITIONED:
        PetscCall(VecNorm(R,NORM_2,&dp));              /*   dp <- sqrt(r'*r) = sqrt(e'*A'*A*e)   */
        KSPCheckNorm(ksp,dp);
        break;
      case KSP_NORM_NATURAL:
        PetscCall(KSP_PCApply(ksp,R,Z));               /*   z <- Br             */
        PetscCall(VecXDot(R,Z,&s));
        KSPCheckDot(ksp,s);
        dp = PetscSqrtReal(PetscAbsScalar(s));                   /*   dp <- sqrt(r'*z) = sqrt(e'*A'*B*A*e)  */
        break;
      case KSP_NORM_NONE:
        dp = 0.0;
        break;
      default: SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
    }

    /* Check for convergence */
    ksp->rnorm = dp;
    PetscCall(KSPLogResidualHistory(ksp,dp));
    PetscCall(KSPMonitor(ksp,i+1,dp));
    PetscCall((*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP));
    if (ksp->reason) break;

    /* Apply PC if not already done for convergence check */
    if (ksp->normtype == KSP_NORM_UNPRECONDITIONED || ksp->normtype == KSP_NORM_NONE) {
      PetscCall(KSP_PCApply(ksp,R,Z));               /*   z <- Br         */
    }

    /* Compute current C (which is W/dpi) */
    PetscCall(VecScale(Ccurr,1.0/dpi));              /*   w <- ci/dpi   */

    if (eigs) {
      if (i > 0) {
        PetscCheck(ksp->max_it == stored_max_it,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Can not change maxit AND calculate eigenvalues");
        e[i] = PetscSqrtReal(PetscAbsScalar(beta/betaold))/alphaold;
        d[i] = PetscSqrtReal(PetscAbsScalar(beta/betaold))*e[i] + 1.0/alpha;
      } else {
        d[i] = PetscSqrtReal(PetscAbsScalar(beta))*e[i] + 1.0/alpha;
      }
      fcg->ned = ksp->its-1;
    }
    ++i;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_FCG(KSP ksp)
{
  PetscInt       i;
  KSP_FCG        *fcg = (KSP_FCG*)ksp->data;

  PetscFunctionBegin;

  /* Destroy "standard" work vecs */
  VecDestroyVecs(ksp->nwork,&ksp->work);

  /* Destroy P and C vectors and the arrays that manage pointers to them */
  if (fcg->nvecs) {
    for (i=0;i<fcg->nchunks;++i) {
      PetscCall(VecDestroyVecs(fcg->chunksizes[i],&fcg->pPvecs[i]));
      PetscCall(VecDestroyVecs(fcg->chunksizes[i],&fcg->pCvecs[i]));
    }
  }
  PetscCall(PetscFree5(fcg->Pvecs,fcg->Cvecs,fcg->pPvecs,fcg->pCvecs,fcg->chunksizes));
  /* free space used for singular value calculations */
  if (ksp->calc_sings) {
    PetscCall(PetscFree4(fcg->e,fcg->d,fcg->ee,fcg->dd));
  }
  PetscCall(KSPDestroyDefault(ksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPView_FCG(KSP ksp,PetscViewer viewer)
{
  KSP_FCG        *fcg = (KSP_FCG*)ksp->data;
  PetscBool      iascii,isstring;
  const char     *truncstr;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring));

  if (fcg->truncstrat == KSP_FCD_TRUNC_TYPE_STANDARD) truncstr = "Using standard truncation strategy";
  else if (fcg->truncstrat == KSP_FCD_TRUNC_TYPE_NOTAY) truncstr = "Using Notay's truncation strategy";
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Undefined FCG truncation strategy");

  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  m_max=%D\n",fcg->mmax));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  preallocated %D directions\n",PetscMin(fcg->nprealloc,fcg->mmax+1)));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  %s\n",truncstr));
  } else if (isstring) {
    PetscCall(PetscViewerStringSPrintf(viewer,"m_max %D nprealloc %D %s",fcg->mmax,fcg->nprealloc,truncstr));
  }
  PetscFunctionReturn(0);
}

/*@
  KSPFCGSetMmax - set the maximum number of previous directions FCG will store for orthogonalization

  Note: mmax + 1 directions are stored (mmax previous ones along with a current one)
  and whether all are used in each iteration also depends on the truncation strategy
  (see KSPFCGSetTruncationType())

  Logically Collective on ksp

  Input Parameters:
+  ksp - the Krylov space context
-  mmax - the maximum number of previous directions to orthogonalize againt

  Level: intermediate

.seealso: KSPFCG, KSPFCGGetTruncationType(), KSPFCGGetNprealloc()
@*/
PetscErrorCode KSPFCGSetMmax(KSP ksp,PetscInt mmax)
{
  KSP_FCG *fcg = (KSP_FCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveInt(ksp,mmax,2);
  fcg->mmax = mmax;
  PetscFunctionReturn(0);
}

/*@
  KSPFCGGetMmax - get the maximum number of previous directions FCG will store

  Note: FCG stores mmax+1 directions at most (mmax previous ones, and one current one)

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  mmax - the maximum number of previous directions allowed for orthogonalization

   Level: intermediate

.seealso: KSPFCG, KSPFCGGetTruncationType(), KSPFCGGetNprealloc(), KSPFCGSetMmax()
@*/

PetscErrorCode KSPFCGGetMmax(KSP ksp,PetscInt *mmax)
{
  KSP_FCG *fcg=(KSP_FCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *mmax = fcg->mmax;
  PetscFunctionReturn(0);
}

/*@
  KSPFCGSetNprealloc - set the number of directions to preallocate with FCG

  Logically Collective on ksp

  Input Parameters:
+  ksp - the Krylov space context
-  nprealloc - the number of vectors to preallocate

  Level: advanced

  Options Database:
. -ksp_fcg_nprealloc <N> - number of directions to preallocate

.seealso: KSPFCG, KSPFCGGetTruncationType(), KSPFCGGetNprealloc()
@*/
PetscErrorCode KSPFCGSetNprealloc(KSP ksp,PetscInt nprealloc)
{
  KSP_FCG *fcg=(KSP_FCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveInt(ksp,nprealloc,2);
  PetscCheck(nprealloc <= fcg->mmax+1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Cannot preallocate more than m_max+1 vectors");
  fcg->nprealloc = nprealloc;
  PetscFunctionReturn(0);
}

/*@
  KSPFCGGetNprealloc - get the number of directions preallocate by FCG

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  nprealloc - the number of directions preallocated

   Level: advanced

.seealso: KSPFCG, KSPFCGGetTruncationType(), KSPFCGSetNprealloc()
@*/
PetscErrorCode KSPFCGGetNprealloc(KSP ksp,PetscInt *nprealloc)
{
  KSP_FCG *fcg=(KSP_FCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *nprealloc = fcg->nprealloc;
  PetscFunctionReturn(0);
}

/*@
  KSPFCGSetTruncationType - specify how many of its stored previous directions FCG uses during orthoganalization

  Logically Collective on ksp

  KSP_FCD_TRUNC_TYPE_STANDARD uses all (up to mmax) stored directions
  KSP_FCD_TRUNC_TYPE_NOTAY uses the last max(1,mod(i,mmax)) stored directions at iteration i=0,1,..

  Input Parameters:
+  ksp - the Krylov space context
-  truncstrat - the choice of strategy

  Level: intermediate

  Options Database:
. -ksp_fcg_truncation_type <standard, notay> - specify how many of its stored previous directions FCG uses during orthoganalization

  .seealso: KSPFCDTruncationType, KSPFCGGetTruncationType
@*/
PetscErrorCode KSPFCGSetTruncationType(KSP ksp,KSPFCDTruncationType truncstrat)
{
  KSP_FCG *fcg=(KSP_FCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ksp,truncstrat,2);
  fcg->truncstrat=truncstrat;
  PetscFunctionReturn(0);
}

/*@
  KSPFCGGetTruncationType - get the truncation strategy employed by FCG

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  truncstrat - the strategy type

   Level: intermediate

.seealso: KSPFCG, KSPFCGSetTruncationType, KSPFCDTruncationType
@*/
PetscErrorCode KSPFCGGetTruncationType(KSP ksp,KSPFCDTruncationType *truncstrat)
{
  KSP_FCG *fcg=(KSP_FCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *truncstrat=fcg->truncstrat;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetFromOptions_FCG(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  KSP_FCG        *fcg=(KSP_FCG*)ksp->data;
  PetscInt       mmax,nprealloc;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"KSP FCG Options");
  PetscCall(PetscOptionsInt("-ksp_fcg_mmax","Maximum number of search directions to store","KSPFCGSetMmax",fcg->mmax,&mmax,&flg));
  if (flg) {
    PetscCall(KSPFCGSetMmax(ksp,mmax));
  }
  PetscCall(PetscOptionsInt("-ksp_fcg_nprealloc","Number of directions to preallocate","KSPFCGSetNprealloc",fcg->nprealloc,&nprealloc,&flg));
  if (flg) {
    PetscCall(KSPFCGSetNprealloc(ksp,nprealloc));
  }
  PetscCall(PetscOptionsEnum("-ksp_fcg_truncation_type","Truncation approach for directions","KSPFCGSetTruncationType",KSPFCDTruncationTypes,(PetscEnum)fcg->truncstrat,(PetscEnum*)&fcg->truncstrat,NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/*MC
      KSPFCG - Implements the Flexible Conjugate Gradient method (FCG)

  Options Database Keys:
+   -ksp_fcg_mmax <N>  - maximum number of search directions
.   -ksp_fcg_nprealloc <N> - number of directions to preallocate
-   -ksp_fcg_truncation_type <standard,notay> - truncation approach for directions

    Contributed by Patrick Sanan

   Notes:
   Supports left preconditioning only.

   Level: beginner

  References:
+ * - Notay, Y."Flexible Conjugate Gradients", SIAM J. Sci. Comput. 22:4, 2000
- * - Axelsson, O. and Vassilevski, P. S. "A Black Box Generalized Conjugate Gradient Solver with Inner Iterations and Variable step Preconditioning",
    SIAM J. Matrix Anal. Appl. 12:4, 1991

 .seealso : KSPGCR, KSPFGMRES, KSPCG, KSPFCGSetMmax(), KSPFCGGetMmax(), KSPFCGSetNprealloc(), KSPFCGGetNprealloc(), KSPFCGSetTruncationType(), KSPFCGGetTruncationType()

M*/
PETSC_EXTERN PetscErrorCode KSPCreate_FCG(KSP ksp)
{
  KSP_FCG        *fcg;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(ksp,&fcg));
#if !defined(PETSC_USE_COMPLEX)
  fcg->type       = KSP_CG_SYMMETRIC;
#else
  fcg->type       = KSP_CG_HERMITIAN;
#endif
  fcg->mmax       = KSPFCG_DEFAULT_MMAX;
  fcg->nprealloc  = KSPFCG_DEFAULT_NPREALLOC;
  fcg->nvecs      = 0;
  fcg->vecb       = KSPFCG_DEFAULT_VECB;
  fcg->nchunks    = 0;
  fcg->truncstrat = KSPFCG_DEFAULT_TRUNCSTRAT;

  ksp->data = (void*)fcg;

  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,1));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,1));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));

  ksp->ops->setup          = KSPSetUp_FCG;
  ksp->ops->solve          = KSPSolve_FCG;
  ksp->ops->destroy        = KSPDestroy_FCG;
  ksp->ops->view           = KSPView_FCG;
  ksp->ops->setfromoptions = KSPSetFromOptions_FCG;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  PetscFunctionReturn(0);
}
