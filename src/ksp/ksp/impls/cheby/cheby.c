
#include <../src/ksp/ksp/impls/cheby/chebyshevimpl.h>    /*I "petscksp.h" I*/

static PetscErrorCode KSPReset_Chebyshev(KSP ksp)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (cheb->kspest) {
    ierr = KSPReset(cheb->kspest);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscScalar chebyhash(PetscInt xx)
{
  unsigned int x = xx;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x);
  return (PetscScalar)((PetscInt64)x-2147483648)*5.e-10; /* center around zero, scaled about -1. to 1.*/
}

/*
 * Must be passed a KSP solver that has "converged", with KSPSetComputeEigenvalues() called before the solve
 */
static PetscErrorCode KSPChebyshevComputeExtremeEigenvalues_Private(KSP kspest,PetscReal *emin,PetscReal *emax)
{
  PetscErrorCode ierr;
  PetscInt       n,neig;
  PetscReal      *re,*im,min,max;

  PetscFunctionBegin;
  ierr = KSPGetIterationNumber(kspest,&n);CHKERRQ(ierr);
  ierr = PetscMalloc2(n,&re,n,&im);CHKERRQ(ierr);
  ierr = KSPComputeEigenvalues(kspest,n,re,im,&neig);CHKERRQ(ierr);
  min  = PETSC_MAX_REAL;
  max  = PETSC_MIN_REAL;
  for (n=0; n<neig; n++) {
    min = PetscMin(min,re[n]);
    max = PetscMax(max,re[n]);
  }
  ierr  = PetscFree2(re,im);CHKERRQ(ierr);
  *emax = max;
  *emin = min;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetUp_Chebyshev(KSP ksp)
{
  KSP_Chebyshev    *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode   ierr;
  PetscBool        flg;
  Mat              Pmat,Amat;
  PetscObjectId    amatid,    pmatid;
  PetscObjectState amatstate, pmatstate;
  PetscFunctionBegin;
  ierr = KSPSetWorkVecs(ksp,3);CHKERRQ(ierr);
  if ((cheb->emin == 0. || cheb->emax == 0.) && !cheb->kspest) { /* We need to estimate eigenvalues */
    ierr = KSPChebyshevEstEigSet(ksp,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  }
  if (cheb->kspest) {
    ierr = KSPGetOperators(ksp,&Amat,&Pmat);CHKERRQ(ierr);
    ierr = MatGetOption(Pmat, MAT_SPD, &flg);CHKERRQ(ierr);
    if (flg) {
      const char *prefix;
      ierr = KSPGetOptionsPrefix(cheb->kspest,&prefix);CHKERRQ(ierr);
      ierr = PetscOptionsHasName(NULL,prefix,"-ksp_type",&flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = KSPSetType(cheb->kspest, KSPCG);CHKERRQ(ierr);
      }
    }
    ierr = PetscObjectGetId((PetscObject)Amat,&amatid);CHKERRQ(ierr);
    ierr = PetscObjectGetId((PetscObject)Pmat,&pmatid);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)Amat,&amatstate);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)Pmat,&pmatstate);CHKERRQ(ierr);
    if (amatid != cheb->amatid || pmatid != cheb->pmatid || amatstate != cheb->amatstate || pmatstate != cheb->pmatstate) {
      PetscReal          max=0.0,min=0.0;
      Vec                B;
      KSPConvergedReason reason;
      ierr = KSPSetPC(cheb->kspest,ksp->pc);CHKERRQ(ierr);
      if (cheb->usenoisy) {
        PetscInt       n,i,istart;
        PetscScalar    *xx;

        B    = ksp->work[1];
        ierr = VecGetOwnershipRange(B,&istart,NULL);CHKERRQ(ierr);
        ierr = VecGetLocalSize(B,&n);CHKERRQ(ierr);
        ierr = VecGetArrayWrite(B,&xx);CHKERRQ(ierr);
        for (i=0; i<n; i++) xx[i] = chebyhash(i+istart);
        ierr = VecRestoreArrayWrite(B,&xx);CHKERRQ(ierr);
      } else {
        PetscBool change;

        if (!ksp->vec_rhs) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Chebyshev must use a noisy right hand side to estimate the eigenvalues when no right hand side is available");
        ierr = PCPreSolveChangeRHS(ksp->pc,&change);CHKERRQ(ierr);
        if (change) {
          B = ksp->work[1];
          ierr = VecCopy(ksp->vec_rhs,B);CHKERRQ(ierr);
        } else B = ksp->vec_rhs;
      }
      ierr = KSPSolve(cheb->kspest,B,ksp->work[0]);CHKERRQ(ierr);
      ierr = KSPGetConvergedReason(cheb->kspest,&reason);CHKERRQ(ierr);
      if (reason == KSP_DIVERGED_ITS) {
        ierr = PetscInfo(ksp,"Eigen estimator ran for prescribed number of iterations\n");CHKERRQ(ierr);
      } else if (reason == KSP_DIVERGED_PC_FAILED) {
        PetscInt       its;
        PCFailedReason pcreason;

        ierr = KSPGetIterationNumber(cheb->kspest,&its);CHKERRQ(ierr);
        if (ksp->normtype == KSP_NORM_NONE) {
          PetscInt  sendbuf,recvbuf;
          ierr = PCGetFailedReasonRank(ksp->pc,&pcreason);CHKERRQ(ierr);
          sendbuf = (PetscInt)pcreason;
          ierr = MPI_Allreduce(&sendbuf,&recvbuf,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)ksp));CHKERRQ(ierr);
          ierr = PCSetFailedReason(ksp->pc,(PCFailedReason) recvbuf);CHKERRQ(ierr);
        }
        ierr = PCGetFailedReason(ksp->pc,&pcreason);CHKERRQ(ierr);
        ksp->reason = KSP_DIVERGED_PC_FAILED;
        ierr = PetscInfo3(ksp,"Eigen estimator failed: %s %s at iteration %D",KSPConvergedReasons[reason],PCFailedReasons[pcreason],its);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      } else if (reason == KSP_CONVERGED_RTOL || reason == KSP_CONVERGED_ATOL) {
        ierr = PetscInfo(ksp,"Eigen estimator converged prematurely. Should not happen except for small or low rank problem\n");CHKERRQ(ierr);
      } else if (reason < 0) {
        ierr = PetscInfo1(ksp,"Eigen estimator failed %s, using estimates anyway\n",KSPConvergedReasons[reason]);CHKERRQ(ierr);
      }

      ierr = KSPChebyshevComputeExtremeEigenvalues_Private(cheb->kspest,&min,&max);CHKERRQ(ierr);
      ierr = KSPSetPC(cheb->kspest,NULL);CHKERRQ(ierr);

      cheb->emin_computed = min;
      cheb->emax_computed = max;
      cheb->emin = cheb->tform[0]*min + cheb->tform[1]*max;
      cheb->emax = cheb->tform[2]*min + cheb->tform[3]*max;

      cheb->amatid    = amatid;
      cheb->pmatid    = pmatid;
      cheb->amatstate = amatstate;
      cheb->pmatstate = pmatstate;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPChebyshevSetEigenvalues_Chebyshev(KSP ksp,PetscReal emax,PetscReal emin)
{
  KSP_Chebyshev  *chebyshevP = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (emax <= emin) SETERRQ2(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_INCOMP,"Maximum eigenvalue must be larger than minimum: max %g min %g",(double)emax,(double)emin);
  if (emax*emin <= 0.0) SETERRQ2(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_INCOMP,"Both eigenvalues must be of the same sign: max %g min %g",(double)emax,(double)emin);
  chebyshevP->emax = emax;
  chebyshevP->emin = emin;

  ierr = KSPChebyshevEstEigSet(ksp,0.,0.,0.,0.);CHKERRQ(ierr); /* Destroy any estimation setup */
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPChebyshevEstEigSet_Chebyshev(KSP ksp,PetscReal a,PetscReal b,PetscReal c,PetscReal d)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a != 0.0 || b != 0.0 || c != 0.0 || d != 0.0) {
    if (!cheb->kspest) { /* should this block of code be moved to KSPSetUp_Chebyshev()? */
      ierr = KSPCreate(PetscObjectComm((PetscObject)ksp),&cheb->kspest);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(cheb->kspest,ksp->errorifnotconverged);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)cheb->kspest,(PetscObject)ksp,1);CHKERRQ(ierr);
      /* use PetscObjectSet/AppendOptionsPrefix() instead of KSPSet/AppendOptionsPrefix() so that the PC prefix is not changed */
      ierr = PetscObjectSetOptionsPrefix((PetscObject)cheb->kspest,((PetscObject)ksp)->prefix);CHKERRQ(ierr);
      ierr = PetscObjectAppendOptionsPrefix((PetscObject)cheb->kspest,"esteig_");CHKERRQ(ierr);
      ierr = KSPSetSkipPCSetFromOptions(cheb->kspest,PETSC_TRUE);CHKERRQ(ierr);

      ierr = KSPSetComputeEigenvalues(cheb->kspest,PETSC_TRUE);CHKERRQ(ierr);

      /* We cannot turn off convergence testing because GMRES will break down if you attempt to keep iterating after a zero norm is obtained */
      ierr = KSPSetTolerances(cheb->kspest,1.e-12,PETSC_DEFAULT,PETSC_DEFAULT,cheb->eststeps);CHKERRQ(ierr);
    }
    if (a >= 0) cheb->tform[0] = a;
    if (b >= 0) cheb->tform[1] = b;
    if (c >= 0) cheb->tform[2] = c;
    if (d >= 0) cheb->tform[3] = d;
    cheb->amatid    = 0;
    cheb->pmatid    = 0;
    cheb->amatstate = -1;
    cheb->pmatstate = -1;
  } else {
    ierr = KSPDestroy(&cheb->kspest);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPChebyshevEstEigSetUseNoisy_Chebyshev(KSP ksp,PetscBool use)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;

  PetscFunctionBegin;
  cheb->usenoisy = use;
  PetscFunctionReturn(0);
}

/*@
   KSPChebyshevSetEigenvalues - Sets estimates for the extreme eigenvalues
   of the preconditioned problem.

   Logically Collective on ksp

   Input Parameters:
+  ksp - the Krylov space context
-  emax, emin - the eigenvalue estimates

  Options Database:
.  -ksp_chebyshev_eigenvalues emin,emax

   Note: Call KSPChebyshevEstEigSet() or use the option -ksp_chebyshev_esteig a,b,c,d to have the KSP
         estimate the eigenvalues and use these estimated values automatically

   Level: intermediate

@*/
PetscErrorCode  KSPChebyshevSetEigenvalues(KSP ksp,PetscReal emax,PetscReal emin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveReal(ksp,emax,2);
  PetscValidLogicalCollectiveReal(ksp,emin,3);
  ierr = PetscTryMethod(ksp,"KSPChebyshevSetEigenvalues_C",(KSP,PetscReal,PetscReal),(ksp,emax,emin));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   KSPChebyshevEstEigSet - Automatically estimate the eigenvalues to use for Chebyshev

   Logically Collective on ksp

   Input Parameters:
+  ksp - the Krylov space context
.  a - multiple of min eigenvalue estimate to use for min Chebyshev bound (or PETSC_DECIDE)
.  b - multiple of max eigenvalue estimate to use for min Chebyshev bound (or PETSC_DECIDE)
.  c - multiple of min eigenvalue estimate to use for max Chebyshev bound (or PETSC_DECIDE)
-  d - multiple of max eigenvalue estimate to use for max Chebyshev bound (or PETSC_DECIDE)

  Options Database:
.  -ksp_chebyshev_esteig a,b,c,d

   Notes:
   The Chebyshev bounds are set using
.vb
   minbound = a*minest + b*maxest
   maxbound = c*minest + d*maxest
.ve
   The default configuration targets the upper part of the spectrum for use as a multigrid smoother, so only the maximum eigenvalue estimate is used.
   The minimum eigenvalue estimate obtained by Krylov iteration is typically not accurate until the method has converged.

   If 0.0 is passed for all transform arguments (a,b,c,d), eigenvalue estimation is disabled.

   The default transform is (0,0.1; 0,1.1) which targets the "upper" part of the spectrum, as desirable for use with multigrid.

   The eigenvalues are estimated using the Lanczo (KSPCG) or Arnoldi (KSPGMRES) process using a noisy right hand side vector.

   Level: intermediate

@*/
PetscErrorCode KSPChebyshevEstEigSet(KSP ksp,PetscReal a,PetscReal b,PetscReal c,PetscReal d)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveReal(ksp,a,2);
  PetscValidLogicalCollectiveReal(ksp,b,3);
  PetscValidLogicalCollectiveReal(ksp,c,4);
  PetscValidLogicalCollectiveReal(ksp,d,5);
  ierr = PetscTryMethod(ksp,"KSPChebyshevEstEigSet_C",(KSP,PetscReal,PetscReal,PetscReal,PetscReal),(ksp,a,b,c,d));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   KSPChebyshevEstEigSetUseNoisy - use a noisy right hand side in order to do the estimate instead of the given right hand side

   Logically Collective

   Input Arguments:
+  ksp - linear solver context
-  use - PETSC_TRUE to use noisy

   Options Database:
.  -ksp_chebyshev_esteig_noisy <true,false>

  Notes:
    This alledgely works better for multigrid smoothers

  Level: intermediate

.seealso: KSPChebyshevEstEigSet()
@*/
PetscErrorCode KSPChebyshevEstEigSetUseNoisy(KSP ksp,PetscBool use)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscTryMethod(ksp,"KSPChebyshevEstEigSetUseNoisy_C",(KSP,PetscBool),(ksp,use));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  KSPChebyshevEstEigGetKSP - Get the Krylov method context used to estimate eigenvalues for the Chebyshev method.  If
  a Krylov method is not being used for this purpose, NULL is returned.  The reference count of the returned KSP is
  not incremented: it should not be destroyed by the user.

  Input Parameters:
. ksp - the Krylov space context

  Output Parameters:
. kspest the eigenvalue estimation Krylov space context

  Level: intermediate

.seealso: KSPChebyshevEstEigSet()
@*/
PetscErrorCode KSPChebyshevEstEigGetKSP(KSP ksp, KSP *kspest)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(kspest,2);
  *kspest = NULL;
  ierr = PetscTryMethod(ksp,"KSPChebyshevEstEigGetKSP_C",(KSP,KSP*),(ksp,kspest));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPChebyshevEstEigGetKSP_Chebyshev(KSP ksp, KSP *kspest)
{
  KSP_Chebyshev *cheb = (KSP_Chebyshev*)ksp->data;

  PetscFunctionBegin;
  *kspest = cheb->kspest;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetFromOptions_Chebyshev(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       neigarg = 2, nestarg = 4;
  PetscReal      eminmax[2] = {0., 0.};
  PetscReal      tform[4] = {PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE};
  PetscBool      flgeig, flgest;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP Chebyshev Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_chebyshev_esteig_steps","Number of est steps in Chebyshev","",cheb->eststeps,&cheb->eststeps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-ksp_chebyshev_eigenvalues","extreme eigenvalues","KSPChebyshevSetEigenvalues",eminmax,&neigarg,&flgeig);CHKERRQ(ierr);
  if (flgeig) {
    if (neigarg != 2) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_INCOMP,"-ksp_chebyshev_eigenvalues: must specify 2 parameters, min and max eigenvalues");
    ierr = KSPChebyshevSetEigenvalues(ksp, eminmax[1], eminmax[0]);CHKERRQ(ierr);
  }
  ierr = PetscOptionsRealArray("-ksp_chebyshev_esteig","estimate eigenvalues using a Krylov method, then use this transform for Chebyshev eigenvalue bounds","KSPChebyshevEstEigSet",tform,&nestarg,&flgest);CHKERRQ(ierr);
  if (flgest) {
    switch (nestarg) {
    case 0:
      ierr = KSPChebyshevEstEigSet(ksp,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
      break;
    case 2:                     /* Base everything on the max eigenvalues */
      ierr = KSPChebyshevEstEigSet(ksp,PETSC_DECIDE,tform[0],PETSC_DECIDE,tform[1]);CHKERRQ(ierr);
      break;
    case 4:                     /* Use the full 2x2 linear transformation */
      ierr = KSPChebyshevEstEigSet(ksp,tform[0],tform[1],tform[2],tform[3]);CHKERRQ(ierr);
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_INCOMP,"Must specify either 0, 2, or 4 parameters for eigenvalue estimation");
    }
  }

  /* We need to estimate eigenvalues; need to set this here so that KSPSetFromOptions() is called on the estimator */
  if ((cheb->emin == 0. || cheb->emax == 0.) && !cheb->kspest) {
   ierr = KSPChebyshevEstEigSet(ksp,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  }

  if (cheb->kspest) {
    ierr = PetscOptionsBool("-ksp_chebyshev_esteig_noisy","Use noisy right hand side for estimate","KSPChebyshevEstEigSetUseNoisy",cheb->usenoisy,&cheb->usenoisy,NULL);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(cheb->kspest);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_Chebyshev(KSP ksp)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       k,kp1,km1,ktmp,i;
  PetscScalar    alpha,omegaprod,mu,omega,Gamma,c[3],scale;
  PetscReal      rnorm = 0.0;
  Vec            sol_orig,b,p[3],r;
  Mat            Amat,Pmat;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);
  ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->its = 0;
  ierr   = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
  /* These three point to the three active solutions, we
     rotate these three at each solution update */
  km1      = 0; k = 1; kp1 = 2;
  sol_orig = ksp->vec_sol; /* ksp->vec_sol will be asigned to rotating vector p[k], thus save its address */
  b        = ksp->vec_rhs;
  p[km1]   = sol_orig;
  p[k]     = ksp->work[0];
  p[kp1]   = ksp->work[1];
  r        = ksp->work[2];

  /* use scale*B as our preconditioner */
  scale = 2.0/(cheb->emax + cheb->emin);

  /*   -alpha <=  scale*lambda(B^{-1}A) <= alpha   */
  alpha     = 1.0 - scale*(cheb->emin);
  Gamma     = 1.0;
  mu        = 1.0/alpha;
  omegaprod = 2.0/alpha;

  c[km1] = 1.0;
  c[k]   = mu;

  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,sol_orig,r);CHKERRQ(ierr);     /*  r = b - A*p[km1] */
    ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(b,r);CHKERRQ(ierr);
  }

  /* calculate residual norm if requested, we have done one iteration */
  if (ksp->normtype) {
    switch (ksp->normtype) {
    case KSP_NORM_PRECONDITIONED:
      ierr = KSP_PCApply(ksp,r,p[k]);CHKERRQ(ierr);  /* p[k] = B^{-1}r */
      ierr = VecNorm(p[k],NORM_2,&rnorm);CHKERRQ(ierr);
      break;
    case KSP_NORM_UNPRECONDITIONED:
    case KSP_NORM_NATURAL:
      ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
    }
    ierr         = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
    ksp->rnorm   = rnorm;
    ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
    ierr = KSPLogResidualHistory(ksp,rnorm);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,0,rnorm);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,0,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  } else ksp->reason = KSP_CONVERGED_ITERATING;
  if (ksp->reason || ksp->max_it==0) {
    if (ksp->max_it==0) ksp->reason = KSP_DIVERGED_ITS; /* This for a V(0,x) cycle */
    PetscFunctionReturn(0);
  }
  if (ksp->normtype != KSP_NORM_PRECONDITIONED) {
    ierr = KSP_PCApply(ksp,r,p[k]);CHKERRQ(ierr);  /* p[k] = B^{-1}r */
  }
  ierr = VecAYPX(p[k],scale,p[km1]);CHKERRQ(ierr);  /* p[k] = scale B^{-1}r + p[km1] */
  ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->its = 1;
  ierr   = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);

  for (i=1; i<ksp->max_it; i++) {
    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
    ksp->its++;
    ierr   = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);

    ierr = KSP_MatMult(ksp,Amat,p[k],r);CHKERRQ(ierr);          /*  r = b - Ap[k]    */
    ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
    /* calculate residual norm if requested */
    if (ksp->normtype) {
      switch (ksp->normtype) {
      case KSP_NORM_PRECONDITIONED:
        ierr = KSP_PCApply(ksp,r,p[kp1]);CHKERRQ(ierr);             /*  p[kp1] = B^{-1}r  */
        ierr = VecNorm(p[kp1],NORM_2,&rnorm);CHKERRQ(ierr);
        break;
      case KSP_NORM_UNPRECONDITIONED:
      case KSP_NORM_NATURAL:
        ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);
        break;
      default:
        rnorm = 0.0;
        break;
      }
      KSPCheckNorm(ksp,rnorm);
      ierr         = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
      ksp->rnorm   = rnorm;
      ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
      ierr = KSPLogResidualHistory(ksp,rnorm);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,i,rnorm);CHKERRQ(ierr);
      ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
      if (ksp->reason) break;
      if (ksp->normtype != KSP_NORM_PRECONDITIONED) {
        ierr = KSP_PCApply(ksp,r,p[kp1]);CHKERRQ(ierr);             /*  p[kp1] = B^{-1}r  */
      }
    } else {
      ierr = KSP_PCApply(ksp,r,p[kp1]);CHKERRQ(ierr);             /*  p[kp1] = B^{-1}r  */
    }
    ksp->vec_sol = p[k];

    c[kp1] = 2.0*mu*c[k] - c[km1];
    omega  = omegaprod*c[k]/c[kp1];

    /* y^{k+1} = omega(y^{k} - y^{k-1} + Gamma*r^{k}) + y^{k-1} */
    ierr = VecAXPBYPCZ(p[kp1],1.0-omega,omega,omega*Gamma*scale,p[km1],p[k]);CHKERRQ(ierr);

    ktmp = km1;
    km1  = k;
    k    = kp1;
    kp1  = ktmp;
  }
  if (!ksp->reason) {
    if (ksp->normtype) {
      ierr = KSP_MatMult(ksp,Amat,p[k],r);CHKERRQ(ierr);       /*  r = b - Ap[k]    */
      ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
      switch (ksp->normtype) {
      case KSP_NORM_PRECONDITIONED:
        ierr = KSP_PCApply(ksp,r,p[kp1]);CHKERRQ(ierr); /* p[kp1] = B^{-1}r */
        ierr = VecNorm(p[kp1],NORM_2,&rnorm);CHKERRQ(ierr);
        break;
      case KSP_NORM_UNPRECONDITIONED:
      case KSP_NORM_NATURAL:
        ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);
        break;
      default:
        rnorm = 0.0;
        break;
      }
      KSPCheckNorm(ksp,rnorm);
      ierr         = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
      ksp->rnorm   = rnorm;
      ierr         = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
      ierr = KSPLogResidualHistory(ksp,rnorm);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,i,rnorm);CHKERRQ(ierr);
    }
    if (ksp->its >= ksp->max_it) {
      if (ksp->normtype != KSP_NORM_NONE) {
        ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
        if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
      } else ksp->reason = KSP_CONVERGED_ITS;
    }
  }

  /* make sure solution is in vector x */
  ksp->vec_sol = sol_orig;
  if (k) {
    ierr = VecCopy(p[k],sol_orig);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static  PetscErrorCode KSPView_Chebyshev(KSP ksp,PetscViewer viewer)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  eigenvalue estimates used:  min = %g, max = %g\n",(double)cheb->emin,(double)cheb->emax);CHKERRQ(ierr);
    if (cheb->kspest) {
      ierr = PetscViewerASCIIPrintf(viewer,"  eigenvalues estimate via %s min %g, max %g\n",((PetscObject)(cheb->kspest))->type_name,(double)cheb->emin_computed,(double)cheb->emax_computed);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  eigenvalues estimated using %s with translations  [%g %g; %g %g]\n",((PetscObject) cheb->kspest)->type_name,(double)cheb->tform[0],(double)cheb->tform[1],(double)cheb->tform[2],(double)cheb->tform[3]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = KSPView(cheb->kspest,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      if (cheb->usenoisy) {
        ierr = PetscViewerASCIIPrintf(viewer,"  estimating eigenvalues using noisy right hand side\n");CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_Chebyshev(KSP ksp)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDestroy(&cheb->kspest);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPChebyshevSetEigenvalues_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPChebyshevEstEigSet_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPChebyshevEstEigSetUseNoisy_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPChebyshevEstEigGetKSP_C",NULL);CHKERRQ(ierr);
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPCHEBYSHEV - The preconditioned Chebyshev iterative method

   Options Database Keys:
+   -ksp_chebyshev_eigenvalues <emin,emax> - set approximations to the smallest and largest eigenvalues
                  of the preconditioned operator. If these are accurate you will get much faster convergence.
.   -ksp_chebyshev_esteig <a,b,c,d> - estimate eigenvalues using a Krylov method, then use this
                         transform for Chebyshev eigenvalue bounds (KSPChebyshevEstEigSet())
.   -ksp_chebyshev_esteig_steps - number of estimation steps
-   -ksp_chebyshev_esteig_noisy - use noisy number generator to create right hand side for eigenvalue estimator

   Level: beginner

   Notes:
    The Chebyshev method requires both the matrix and preconditioner to
          be symmetric positive (semi) definite.
          Only support for left preconditioning.

          Chebyshev is configured as a smoother by default, targetting the "upper" part of the spectrum.
          The user should call KSPChebyshevSetEigenvalues() if they have eigenvalue estimates.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPChebyshevSetEigenvalues(), KSPChebyshevEstEigSet(), KSPChebyshevEstEigSetUseNoisy()
           KSPRICHARDSON, KSPCG, PCMG

M*/

PETSC_EXTERN PetscErrorCode KSPCreate_Chebyshev(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_Chebyshev  *chebyshevP;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&chebyshevP);CHKERRQ(ierr);

  ksp->data = (void*)chebyshevP;
  ierr      = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr      = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr      = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);
  ierr      = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1);CHKERRQ(ierr);

  chebyshevP->emin = 0.;
  chebyshevP->emax = 0.;

  chebyshevP->tform[0] = 0.0;
  chebyshevP->tform[1] = 0.1;
  chebyshevP->tform[2] = 0;
  chebyshevP->tform[3] = 1.1;
  chebyshevP->eststeps = 10;
  chebyshevP->usenoisy = PETSC_TRUE;
  ksp->setupnewmatrix = PETSC_TRUE;

  ksp->ops->setup          = KSPSetUp_Chebyshev;
  ksp->ops->solve          = KSPSolve_Chebyshev;
  ksp->ops->destroy        = KSPDestroy_Chebyshev;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_Chebyshev;
  ksp->ops->view           = KSPView_Chebyshev;
  ksp->ops->reset          = KSPReset_Chebyshev;

  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPChebyshevSetEigenvalues_C",KSPChebyshevSetEigenvalues_Chebyshev);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPChebyshevEstEigSet_C",KSPChebyshevEstEigSet_Chebyshev);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPChebyshevEstEigSetUseNoisy_C",KSPChebyshevEstEigSetUseNoisy_Chebyshev);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPChebyshevEstEigGetKSP_C",KSPChebyshevEstEigGetKSP_Chebyshev);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
