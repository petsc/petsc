/*
 This file implements the deflated GMRES.

 */

#include <../src/ksp/ksp/impls/gmres/dgmres/dgmresimpl.h>       /*I  "petscksp.h"  I*/

PetscLogEvent KSP_DGMRESComputeDeflationData, KSP_DGMRESApplyDeflation;

#define GMRES_DELTA_DIRECTIONS 10
#define GMRES_DEFAULT_MAXK     30
static PetscErrorCode    KSPDGMRESGetNewVectors(KSP,PetscInt);
static PetscErrorCode    KSPDGMRESUpdateHessenberg(KSP,PetscInt,PetscBool,PetscReal*);
static PetscErrorCode    KSPDGMRESBuildSoln(PetscScalar*,Vec,Vec,KSP,PetscInt);

PetscErrorCode  KSPDGMRESSetEigen(KSP ksp,PetscInt nb_eig)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod((ksp),"KSPDGMRESSetEigen_C",(KSP,PetscInt),(ksp,nb_eig));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode  KSPDGMRESSetMaxEigen(KSP ksp,PetscInt max_neig)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod((ksp),"KSPDGMRESSetMaxEigen_C",(KSP,PetscInt),(ksp,max_neig));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode  KSPDGMRESForce(KSP ksp,PetscBool force)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod((ksp),"KSPDGMRESForce_C",(KSP,PetscBool),(ksp,force));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode  KSPDGMRESSetRatio(KSP ksp,PetscReal ratio)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod((ksp),"KSPDGMRESSetRatio_C",(KSP,PetscReal),(ksp,ratio));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode  KSPDGMRESComputeSchurForm(KSP ksp,PetscInt *neig)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod((ksp),"KSPDGMRESComputeSchurForm_C",(KSP, PetscInt*),(ksp, neig));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode  KSPDGMRESComputeDeflationData(KSP ksp,PetscInt *curneigh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod((ksp),"KSPDGMRESComputeDeflationData_C",(KSP,PetscInt*),(ksp,curneigh));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode  KSPDGMRESApplyDeflation(KSP ksp, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod((ksp),"KSPDGMRESApplyDeflation_C",(KSP, Vec, Vec),(ksp, x, y));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPDGMRESImproveEig(KSP ksp, PetscInt neig)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod((ksp), "KSPDGMRESImproveEig_C",(KSP, PetscInt),(ksp, neig));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPSetUp_DGMRES(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_DGMRES     *dgmres = (KSP_DGMRES*) ksp->data;
  PetscInt       neig    = dgmres->neig+EIG_OFFSET;
  PetscInt       max_k   = dgmres->max_k+1;

  PetscFunctionBegin;
  ierr = KSPSetUp_GMRES(ksp);CHKERRQ(ierr);
  if (!dgmres->neig) PetscFunctionReturn(0);

  /* Allocate workspace for the Schur vectors*/
  ierr          = PetscMalloc1(neig*max_k, &SR);CHKERRQ(ierr);
  dgmres->wr    = NULL;
  dgmres->wi    = NULL;
  dgmres->perm  = NULL;
  dgmres->modul = NULL;
  dgmres->Q     = NULL;
  dgmres->Z     = NULL;

  UU   = NULL;
  XX   = NULL;
  MX   = NULL;
  AUU  = NULL;
  XMX  = NULL;
  XMU  = NULL;
  UMX  = NULL;
  AUAU = NULL;
  TT   = NULL;
  TTF  = NULL;
  INVP = NULL;
  X1   = NULL;
  X2   = NULL;
  MU   = NULL;
  PetscFunctionReturn(0);
}

/*
 Run GMRES, possibly with restart.  Return residual history if requested.
 input parameters:

 .       gmres  - structure containing parameters and work areas

 output parameters:
 .        nres    - residuals (from preconditioned system) at each step.
 If restarting, consider passing nres+it.  If null,
 ignored
 .        itcount - number of iterations used.  nres[0] to nres[itcount]
 are defined.  If null, ignored.

 Notes:
 On entry, the value in vector VEC_VV(0) should be the initial residual
 (this allows shortcuts where the initial preconditioned residual is 0).
 */
PetscErrorCode KSPDGMRESCycle(PetscInt *itcount,KSP ksp)
{
  KSP_DGMRES     *dgmres = (KSP_DGMRES*)(ksp->data);
  PetscReal      res_norm,res,hapbnd,tt;
  PetscErrorCode ierr;
  PetscInt       it     = 0;
  PetscInt       max_k  = dgmres->max_k;
  PetscBool      hapend = PETSC_FALSE;
  PetscReal      res_old;
  PetscInt       test = 0;

  PetscFunctionBegin;
  ierr    = VecNormalize(VEC_VV(0),&res_norm);CHKERRQ(ierr);
  KSPCheckNorm(ksp,res_norm);
  res     = res_norm;
  *GRS(0) = res_norm;

  /* check for the convergence */
  ierr       = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = res;
  else ksp->rnorm = 0.0;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
  dgmres->it = (it - 1);
  ierr       = KSPLogResidualHistory(ksp,ksp->rnorm);CHKERRQ(ierr);
  ierr       = KSPMonitor(ksp,ksp->its,ksp->rnorm);CHKERRQ(ierr);
  if (!res) {
    if (itcount) *itcount = 0;
    ksp->reason = KSP_CONVERGED_ATOL;
    ierr        = PetscInfo(ksp,"Converged due to zero residual norm on entry\n");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* record the residual norm to test if deflation is needed */
  res_old = res;

  ierr = (*ksp->converged)(ksp,ksp->its,ksp->rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  while (!ksp->reason && it < max_k && ksp->its < ksp->max_it) {
    if (it) {
      ierr = KSPLogResidualHistory(ksp,ksp->rnorm);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,ksp->its,ksp->rnorm);CHKERRQ(ierr);
    }
    dgmres->it = (it - 1);
    if (dgmres->vv_allocated <= it + VEC_OFFSET + 1) {
      ierr = KSPDGMRESGetNewVectors(ksp,it+1);CHKERRQ(ierr);
    }
    if (dgmres->r > 0) {
      if (ksp->pc_side == PC_LEFT) {
        /* Apply the first preconditioner */
        ierr = KSP_PCApplyBAorAB(ksp,VEC_VV(it), VEC_TEMP,VEC_TEMP_MATOP);CHKERRQ(ierr);
        /* Then apply Deflation as a preconditioner */
        ierr = KSPDGMRESApplyDeflation(ksp, VEC_TEMP, VEC_VV(1+it));CHKERRQ(ierr);
      } else if (ksp->pc_side == PC_RIGHT) {
        ierr = KSPDGMRESApplyDeflation(ksp, VEC_VV(it), VEC_TEMP);CHKERRQ(ierr);
        ierr = KSP_PCApplyBAorAB(ksp, VEC_TEMP, VEC_VV(1+it), VEC_TEMP_MATOP);CHKERRQ(ierr);
      }
    } else {
      ierr = KSP_PCApplyBAorAB(ksp,VEC_VV(it),VEC_VV(1+it),VEC_TEMP_MATOP);CHKERRQ(ierr);
    }
    dgmres->matvecs += 1;
    /* update hessenberg matrix and do Gram-Schmidt */
    ierr = (*dgmres->orthog)(ksp,it);CHKERRQ(ierr);

    /* vv(i+1) . vv(i+1) */
    ierr = VecNormalize(VEC_VV(it+1),&tt);CHKERRQ(ierr);
    /* save the magnitude */
    *HH(it+1,it)  = tt;
    *HES(it+1,it) = tt;

    /* check for the happy breakdown */
    hapbnd = PetscAbsScalar(tt / *GRS(it));
    if (hapbnd > dgmres->haptol) hapbnd = dgmres->haptol;
    if (tt < hapbnd) {
      ierr   = PetscInfo2(ksp,"Detected happy breakdown, current hapbnd = %g tt = %g\n",(double)hapbnd,(double)tt);CHKERRQ(ierr);
      hapend = PETSC_TRUE;
    }
    ierr = KSPDGMRESUpdateHessenberg(ksp,it,hapend,&res);CHKERRQ(ierr);

    it++;
    dgmres->it = (it-1);     /* For converged */
    ksp->its++;
    if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = res;
    else ksp->rnorm = 0.0;
    if (ksp->reason) break;

    ierr = (*ksp->converged)(ksp,ksp->its,ksp->rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);

    /* Catch error in happy breakdown and signal convergence and break from loop */
    if (hapend) {
      if (!ksp->reason) {
        if (ksp->errorifnotconverged) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"You reached the happy break down, but convergence was not indicated. Residual norm = %g",(double)res);
        else {
          ksp->reason = KSP_DIVERGED_BREAKDOWN;
          break;
        }
      }
    }
  }

  /* Monitor if we know that we will not return for a restart */
  if (it && (ksp->reason || ksp->its >= ksp->max_it)) {
    ierr = KSPLogResidualHistory(ksp,ksp->rnorm);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,ksp->its,ksp->rnorm);CHKERRQ(ierr);
  }
  if (itcount) *itcount = it;

  /*
   Down here we have to solve for the "best" coefficients of the Krylov
   columns, add the solution values together, and possibly unwind the
   preconditioning from the solution
   */
  /* Form the solution (or the solution so far) */
  ierr = KSPDGMRESBuildSoln(GRS(0),ksp->vec_sol,ksp->vec_sol,ksp,it-1);CHKERRQ(ierr);

  /* Compute data for the deflation to be used during the next restart */
  if (!ksp->reason && ksp->its < ksp->max_it) {
    test = max_k *PetscLogReal(ksp->rtol/res) /PetscLogReal(res/res_old);
    /* Compute data for the deflation if the residual rtol will not be reached in the remaining number of steps allowed  */
    if ((test > dgmres->smv*(ksp->max_it-ksp->its)) || dgmres->force) {
      ierr =  KSPDGMRESComputeDeflationData(ksp,NULL);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSolve_DGMRES(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,its,itcount;
  KSP_DGMRES     *dgmres    = (KSP_DGMRES*) ksp->data;
  PetscBool      guess_zero = ksp->guess_zero;

  PetscFunctionBegin;
  if (ksp->calc_sings && !dgmres->Rsvd) SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_ORDER,"Must call KSPSetComputeSingularValues() before KSPSetUp() is called");

  ierr            = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->its        = 0;
  dgmres->matvecs = 0;
  ierr            = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);

  itcount     = 0;
  ksp->reason = KSP_CONVERGED_ITERATING;
  while (!ksp->reason) {
    ierr = KSPInitialResidual(ksp,ksp->vec_sol,VEC_TEMP,VEC_TEMP_MATOP,VEC_VV(0),ksp->vec_rhs);CHKERRQ(ierr);
    if (ksp->pc_side == PC_LEFT) {
      dgmres->matvecs += 1;
      if (dgmres->r > 0) {
        ierr = KSPDGMRESApplyDeflation(ksp, VEC_VV(0), VEC_TEMP);CHKERRQ(ierr);
        ierr = VecCopy(VEC_TEMP, VEC_VV(0));CHKERRQ(ierr);
      }
    }

    ierr     = KSPDGMRESCycle(&its,ksp);CHKERRQ(ierr);
    itcount += its;
    if (itcount >= ksp->max_it) {
      if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
      break;
    }
    ksp->guess_zero = PETSC_FALSE; /* every future call to KSPInitialResidual() will have nonzero guess */
  }
  ksp->guess_zero = guess_zero; /* restore if user provided nonzero initial guess */

  for (i = 0; i < dgmres->r; i++) {
    ierr = VecViewFromOptions(UU[i],(PetscObject)ksp,"-ksp_dgmres_view_deflation_vecs");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode KSPDestroy_DGMRES(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_DGMRES     *dgmres  = (KSP_DGMRES*) ksp->data;
  PetscInt       neig1    = dgmres->neig+EIG_OFFSET;
  PetscInt       max_neig = dgmres->max_neig;

  PetscFunctionBegin;
  if (dgmres->r) {
    ierr = VecDestroyVecs(max_neig, &UU);CHKERRQ(ierr);
    ierr = VecDestroyVecs(max_neig, &MU);CHKERRQ(ierr);
    if (XX) {
      ierr = VecDestroyVecs(neig1, &XX);CHKERRQ(ierr);
      ierr = VecDestroyVecs(neig1, &MX);CHKERRQ(ierr);
    }

    ierr = PetscFree(TT);CHKERRQ(ierr);
    ierr = PetscFree(TTF);CHKERRQ(ierr);
    ierr = PetscFree(INVP);CHKERRQ(ierr);

    ierr = PetscFree(XMX);CHKERRQ(ierr);
    ierr = PetscFree(UMX);CHKERRQ(ierr);
    ierr = PetscFree(XMU);CHKERRQ(ierr);
    ierr = PetscFree(X1);CHKERRQ(ierr);
    ierr = PetscFree(X2);CHKERRQ(ierr);
    ierr = PetscFree(dgmres->work);CHKERRQ(ierr);
    ierr = PetscFree(dgmres->iwork);CHKERRQ(ierr);
    ierr = PetscFree(dgmres->wr);CHKERRQ(ierr);
    ierr = PetscFree(dgmres->wi);CHKERRQ(ierr);
    ierr = PetscFree(dgmres->modul);CHKERRQ(ierr);
    ierr = PetscFree(dgmres->Q);CHKERRQ(ierr);
    ierr = PetscFree(ORTH);CHKERRQ(ierr);
    ierr = PetscFree(AUAU);CHKERRQ(ierr);
    ierr = PetscFree(AUU);CHKERRQ(ierr);
    ierr = PetscFree(SR2);CHKERRQ(ierr);
  }
  ierr = PetscFree(SR);CHKERRQ(ierr);
  ierr = KSPDestroy_GMRES(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
 KSPDGMRESBuildSoln - create the solution from the starting vector and the
 current iterates.

 Input parameters:
 nrs - work area of size it + 1.
 vs  - index of initial guess
 vdest - index of result.  Note that vs may == vdest (replace
 guess with the solution).

 This is an internal routine that knows about the GMRES internals.
 */
static PetscErrorCode KSPDGMRESBuildSoln(PetscScalar *nrs,Vec vs,Vec vdest,KSP ksp,PetscInt it)
{
  PetscScalar    tt;
  PetscErrorCode ierr;
  PetscInt       ii,k,j;
  KSP_DGMRES     *dgmres = (KSP_DGMRES*) (ksp->data);

  /* Solve for solution vector that minimizes the residual */

  PetscFunctionBegin;
  /* If it is < 0, no gmres steps have been performed */
  if (it < 0) {
    ierr = VecCopy(vs,vdest);CHKERRQ(ierr);     /* VecCopy() is smart, exists immediately if vguess == vdest */
    PetscFunctionReturn(0);
  }
  if (*HH(it,it) == 0.0) SETERRQ2(PetscObjectComm((PetscObject)ksp), PETSC_ERR_CONV_FAILED,"Likely your matrix is the zero operator. HH(it,it) is identically zero; it = %D GRS(it) = %g",it,(double)PetscAbsScalar(*GRS(it)));
  if (*HH(it,it) != 0.0) nrs[it] = *GRS(it) / *HH(it,it);
  else nrs[it] = 0.0;

  for (ii=1; ii<=it; ii++) {
    k  = it - ii;
    tt = *GRS(k);
    for (j=k+1; j<=it; j++) tt = tt - *HH(k,j) * nrs[j];
    if (*HH(k,k) == 0.0) SETERRQ2(PetscObjectComm((PetscObject)ksp), PETSC_ERR_CONV_FAILED,"Likely your matrix is singular. HH(k,k) is identically zero; it = %D k = %D",it,k);
    nrs[k] = tt / *HH(k,k);
  }

  /* Accumulate the correction to the solution of the preconditioned problem in TEMP */
  ierr = VecSet(VEC_TEMP,0.0);CHKERRQ(ierr);
  ierr = VecMAXPY(VEC_TEMP,it+1,nrs,&VEC_VV(0));CHKERRQ(ierr);

  /* Apply deflation */
  if (ksp->pc_side==PC_RIGHT && dgmres->r > 0) {
    ierr = KSPDGMRESApplyDeflation(ksp, VEC_TEMP, VEC_TEMP_MATOP);CHKERRQ(ierr);
    ierr = VecCopy(VEC_TEMP_MATOP, VEC_TEMP);CHKERRQ(ierr);
  }
  ierr = KSPUnwindPreconditioner(ksp,VEC_TEMP,VEC_TEMP_MATOP);CHKERRQ(ierr);

  /* add solution to previous solution */
  if (vdest != vs) {
    ierr = VecCopy(vs,vdest);CHKERRQ(ierr);
  }
  ierr = VecAXPY(vdest,1.0,VEC_TEMP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
 Do the scalar work for the orthogonalization.  Return new residual norm.
 */
static PetscErrorCode KSPDGMRESUpdateHessenberg(KSP ksp,PetscInt it,PetscBool hapend,PetscReal *res)
{
  PetscScalar *hh,*cc,*ss,tt;
  PetscInt    j;
  KSP_DGMRES  *dgmres = (KSP_DGMRES*) (ksp->data);

  PetscFunctionBegin;
  hh = HH(0,it);
  cc = CC(0);
  ss = SS(0);

  /* Apply all the previously computed plane rotations to the new column
   of the Hessenberg matrix */
  for (j=1; j<=it; j++) {
    tt  = *hh;
    *hh = PetscConj(*cc) * tt + *ss * *(hh+1);
    hh++;
    *hh = *cc++ * *hh -(*ss++ * tt);
  }

  /*
   compute the new plane rotation, and apply it to:
   1) the right-hand-side of the Hessenberg system
   2) the new column of the Hessenberg matrix
   thus obtaining the updated value of the residual
   */
  if (!hapend) {
    tt = PetscSqrtScalar(PetscConj(*hh) * *hh + PetscConj(*(hh+1)) * *(hh+1));
    if (tt == 0.0) {
      ksp->reason = KSP_DIVERGED_NULL;
      PetscFunctionReturn(0);
    }
    *cc        = *hh / tt;
    *ss        = *(hh+1) / tt;
    *GRS(it+1) = -(*ss * *GRS(it));
    *GRS(it)   = PetscConj(*cc) * *GRS(it);
    *hh        = PetscConj(*cc) * *hh + *ss * *(hh+1);
    *res       = PetscAbsScalar(*GRS(it+1));
  } else {
    /* happy breakdown: HH(it+1, it) = 0, therfore we don't need to apply
     another rotation matrix (so RH doesn't change).  The new residual is
     always the new sine term times the residual from last time (GRS(it)),
     but now the new sine rotation would be zero...so the residual should
     be zero...so we will multiply "zero" by the last residual.  This might
     not be exactly what we want to do here -could just return "zero". */

    *res = 0.0;
  }
  PetscFunctionReturn(0);
}
/*
 This routine allocates more work vectors, starting from VEC_VV(it).
 */
static PetscErrorCode KSPDGMRESGetNewVectors(KSP ksp,PetscInt it)
{
  KSP_DGMRES     *dgmres = (KSP_DGMRES*) ksp->data;
  PetscErrorCode ierr;
  PetscInt       nwork = dgmres->nwork_alloc,k,nalloc;

  PetscFunctionBegin;
  nalloc = PetscMin(ksp->max_it,dgmres->delta_allocate);
  /* Adjust the number to allocate to make sure that we don't exceed the
   number of available slots */
  if (it + VEC_OFFSET + nalloc >= dgmres->vecs_allocated) {
    nalloc = dgmres->vecs_allocated - it - VEC_OFFSET;
  }
  if (!nalloc) PetscFunctionReturn(0);

  dgmres->vv_allocated += nalloc;

  ierr = KSPCreateVecs(ksp,nalloc,&dgmres->user_work[nwork],0,NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(ksp,nalloc,dgmres->user_work[nwork]);CHKERRQ(ierr);

  dgmres->mwork_alloc[nwork] = nalloc;
  for (k=0; k<nalloc; k++) {
    dgmres->vecs[it+VEC_OFFSET+k] = dgmres->user_work[nwork][k];
  }
  dgmres->nwork_alloc++;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPBuildSolution_DGMRES(KSP ksp,Vec ptr,Vec *result)
{
  KSP_DGMRES     *dgmres = (KSP_DGMRES*) ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ptr) {
    if (!dgmres->sol_temp) {
      ierr = VecDuplicate(ksp->vec_sol,&dgmres->sol_temp);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)ksp,(PetscObject)dgmres->sol_temp);CHKERRQ(ierr);
    }
    ptr = dgmres->sol_temp;
  }
  if (!dgmres->nrs) {
    /* allocate the work area */
    ierr = PetscMalloc1(dgmres->max_k,&dgmres->nrs);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)ksp,dgmres->max_k*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  ierr = KSPDGMRESBuildSoln(dgmres->nrs,ksp->vec_sol,ptr,ksp,dgmres->it);CHKERRQ(ierr);
  if (result) *result = ptr;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPView_DGMRES(KSP ksp,PetscViewer viewer)
{
  KSP_DGMRES     *dgmres = (KSP_DGMRES*) ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii,isharmonic;

  PetscFunctionBegin;
  ierr = KSPView_GMRES(ksp,viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (dgmres->force) PetscViewerASCIIPrintf(viewer, "    Adaptive strategy is used: FALSE\n");
    else PetscViewerASCIIPrintf(viewer, "    Adaptive strategy is used: TRUE\n");
    ierr = PetscOptionsHasName(((PetscObject)ksp)->options,((PetscObject)ksp)->prefix, "-ksp_dgmres_harmonic_ritz", &isharmonic);CHKERRQ(ierr);
    if (isharmonic) {
      ierr = PetscViewerASCIIPrintf(viewer, "   Frequency of extracted eigenvalues = %D using Harmonic Ritz values \n", dgmres->neig);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "   Frequency of extracted eigenvalues = %D using Ritz values \n", dgmres->neig);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "   Total number of extracted eigenvalues = %D\n", dgmres->r);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "   Maximum number of eigenvalues set to be extracted = %D\n", dgmres->max_neig);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "   relaxation parameter for the adaptive strategy(smv)  = %g\n", dgmres->smv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "   Number of matvecs : %D\n", dgmres->matvecs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* New DGMRES functions */

PetscErrorCode  KSPDGMRESSetEigen_DGMRES(KSP ksp,PetscInt neig)
{
  KSP_DGMRES *dgmres = (KSP_DGMRES*) ksp->data;

  PetscFunctionBegin;
  if (neig< 0 && neig >dgmres->max_k) SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE,"The value of neig must be positive and less than the restart value ");
  dgmres->neig=neig;
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPDGMRESSetMaxEigen_DGMRES(KSP ksp,PetscInt max_neig)
{
  KSP_DGMRES *dgmres = (KSP_DGMRES*) ksp->data;

  PetscFunctionBegin;
  if (max_neig < 0 && max_neig >dgmres->max_k) SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE,"The value of max_neig must be positive and less than the restart value ");
  dgmres->max_neig=max_neig;
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPDGMRESSetRatio_DGMRES(KSP ksp,PetscReal ratio)
{
  KSP_DGMRES *dgmres = (KSP_DGMRES*) ksp->data;

  PetscFunctionBegin;
  if (ratio <= 0) SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE,"The relaxation parameter value must be positive");
  dgmres->smv=ratio;
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPDGMRESForce_DGMRES(KSP ksp,PetscBool force)
{
  KSP_DGMRES *dgmres = (KSP_DGMRES*) ksp->data;

  PetscFunctionBegin;
  dgmres->force = force;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetFromOptions_DGMRES(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       neig;
  PetscInt       max_neig;
  KSP_DGMRES     *dgmres = (KSP_DGMRES*) ksp->data;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = KSPSetFromOptions_GMRES(PetscOptionsObject,ksp);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP DGMRES Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_dgmres_eigen","Number of smallest eigenvalues to extract at each restart","KSPDGMRESSetEigen",dgmres->neig, &neig, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPDGMRESSetEigen(ksp, neig);CHKERRQ(ierr);
  }
  ierr = PetscOptionsInt("-ksp_dgmres_max_eigen","Maximum Number of smallest eigenvalues to extract ","KSPDGMRESSetMaxEigen",dgmres->max_neig, &max_neig, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPDGMRESSetMaxEigen(ksp, max_neig);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-ksp_dgmres_ratio","Relaxation parameter for the smaller number of matrix-vectors product allowed","KSPDGMRESSetRatio",dgmres->smv,&dgmres->smv,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_dgmres_improve","Improve the computation of eigenvalues by solving a new generalized eigenvalue problem (experimental - not stable at this time)",NULL,dgmres->improve,&dgmres->improve,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_dgmres_force","Sets DGMRES always at restart active, i.e do not use the adaptive strategy","KSPDGMRESForce",dgmres->force,&dgmres->force,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPDGMRESComputeDeflationData_DGMRES(KSP ksp, PetscInt *ExtrNeig)
{
  KSP_DGMRES     *dgmres = (KSP_DGMRES*) ksp->data;
  PetscErrorCode ierr;
  PetscInt       i,j, k;
  PetscBLASInt   nr, bmax;
  PetscInt       r = dgmres->r;
  PetscInt       neig;          /* number of eigenvalues to extract at each restart */
  PetscInt       neig1    = dgmres->neig + EIG_OFFSET;  /* max number of eig that can be extracted at each restart */
  PetscInt       max_neig = dgmres->max_neig;  /* Max number of eigenvalues to extract during the iterative process */
  PetscInt       N        = dgmres->max_k+1;
  PetscInt       n        = dgmres->it+1;
  PetscReal      alpha;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(KSP_DGMRESComputeDeflationData, ksp, 0,0,0);CHKERRQ(ierr);
  if (dgmres->neig == 0 || (max_neig < (r+neig1) && !dgmres->improve)) {
    ierr = PetscLogEventEnd(KSP_DGMRESComputeDeflationData, ksp, 0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr =  KSPDGMRESComputeSchurForm(ksp, &neig);CHKERRQ(ierr);
  /* Form the extended Schur vectors X=VV*Sr */
  if (!XX) {
    ierr = VecDuplicateVecs(VEC_VV(0), neig1, &XX);CHKERRQ(ierr);
  }
  for (j = 0; j<neig; j++) {
    ierr = VecZeroEntries(XX[j]);CHKERRQ(ierr);
    ierr = VecMAXPY(XX[j], n, &SR[j*N], &VEC_VV(0));CHKERRQ(ierr);
  }

  /* Orthogonalize X against U */
  if (!ORTH) {
    ierr = PetscMalloc1(max_neig, &ORTH);CHKERRQ(ierr);
  }
  if (r > 0) {
    /* modified Gram-Schmidt */
    for (j = 0; j<neig; j++) {
      for (i=0; i<r; i++) {
        /* First, compute U'*X[j] */
        ierr = VecDot(XX[j], UU[i], &alpha);CHKERRQ(ierr);
        /* Then, compute X(j)=X(j)-U*U'*X(j) */
        ierr = VecAXPY(XX[j], -alpha, UU[i]);CHKERRQ(ierr);
      }
    }
  }
  /* Compute MX = M^{-1}*A*X */
  if (!MX) {
    ierr = VecDuplicateVecs(VEC_VV(0), neig1, &MX);CHKERRQ(ierr);
  }
  for (j = 0; j<neig; j++) {
    ierr = KSP_PCApplyBAorAB(ksp, XX[j], MX[j], VEC_TEMP_MATOP);CHKERRQ(ierr);
  }
  dgmres->matvecs += neig;

  if ((r+neig1) > max_neig && dgmres->improve) {    /* Improve the approximate eigenvectors in X by solving a new generalized eigenvalue -- Quite expensive to do this actually */
    ierr = KSPDGMRESImproveEig(ksp, neig);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(KSP_DGMRESComputeDeflationData, ksp, 0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);   /* We return here since data for M have been improved in  KSPDGMRESImproveEig()*/
  }

  /* Compute XMX = X'*M^{-1}*A*X -- size (neig, neig) */
  if (!XMX) {
    ierr = PetscMalloc1(neig1*neig1, &XMX);CHKERRQ(ierr);
  }
  for (j = 0; j < neig; j++) {
    ierr = VecMDot(MX[j], neig, XX, &(XMX[j*neig1]));CHKERRQ(ierr);
  }

  if (r > 0) {
    /* Compute UMX = U'*M^{-1}*A*X -- size (r, neig) */
    if (!UMX) {
      ierr = PetscMalloc1(max_neig*neig1, &UMX);CHKERRQ(ierr);
    }
    for (j = 0; j < neig; j++) {
      ierr = VecMDot(MX[j], r, UU, &(UMX[j*max_neig]));CHKERRQ(ierr);
    }
    /* Compute XMU = X'*M^{-1}*A*U -- size(neig, r) */
    if (!XMU) {
      ierr = PetscMalloc1(max_neig*neig1, &XMU);CHKERRQ(ierr);
    }
    for (j = 0; j<r; j++) {
      ierr = VecMDot(MU[j], neig, XX, &(XMU[j*neig1]));CHKERRQ(ierr);
    }
  }

  /* Form the new matrix T = [T UMX; XMU XMX]; */
  if (!TT) {
    ierr = PetscMalloc1(max_neig*max_neig, &TT);CHKERRQ(ierr);
  }
  if (r > 0) {
    /* Add XMU to T */
    for (j = 0; j < r; j++) {
      ierr = PetscArraycpy(&(TT[max_neig*j+r]), &(XMU[neig1*j]), neig);CHKERRQ(ierr);
    }
    /* Add [UMX; XMX] to T */
    for (j = 0; j < neig; j++) {
      k = r+j;
      ierr = PetscArraycpy(&(TT[max_neig*k]), &(UMX[max_neig*j]), r);CHKERRQ(ierr);
      ierr = PetscArraycpy(&(TT[max_neig*k + r]), &(XMX[neig1*j]), neig);CHKERRQ(ierr);
    }
  } else { /* Add XMX to T */
    for (j = 0; j < neig; j++) {
      ierr = PetscArraycpy(&(TT[max_neig*j]), &(XMX[neig1*j]), neig);CHKERRQ(ierr);
    }
  }

  dgmres->r += neig;
  r          = dgmres->r;
  ierr       = PetscBLASIntCast(r,&nr);CHKERRQ(ierr);
  /*LU Factorize T with Lapack xgetrf routine */

  ierr = PetscBLASIntCast(max_neig,&bmax);CHKERRQ(ierr);
  if (!TTF) {
    ierr = PetscMalloc1(bmax*bmax, &TTF);CHKERRQ(ierr);
  }
  ierr = PetscArraycpy(TTF, TT, bmax*r);CHKERRQ(ierr);
  if (!INVP) {
    ierr = PetscMalloc1(bmax, &INVP);CHKERRQ(ierr);
  }
  {
    PetscBLASInt info;
    PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&nr, &nr, TTF, &bmax, INVP, &info));
    if (info) SETERRQ1(PetscObjectComm((PetscObject)ksp), PETSC_ERR_LIB,"Error in LAPACK routine XGETRF INFO=%d",(int) info);
  }

  /* Save X in U and MX in MU for the next cycles and increase the size of the invariant subspace */
  if (!UU) {
    ierr = VecDuplicateVecs(VEC_VV(0), max_neig, &UU);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(VEC_VV(0), max_neig, &MU);CHKERRQ(ierr);
  }
  for (j=0; j<neig; j++) {
    ierr = VecCopy(XX[j], UU[r-neig+j]);CHKERRQ(ierr);
    ierr = VecCopy(MX[j], MU[r-neig+j]);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(KSP_DGMRESComputeDeflationData, ksp, 0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPDGMRESComputeSchurForm_DGMRES(KSP ksp, PetscInt *neig)
{
  KSP_DGMRES     *dgmres = (KSP_DGMRES*) ksp->data;
  PetscErrorCode ierr;
  PetscInt       N = dgmres->max_k + 1, n=dgmres->it+1;
  PetscBLASInt   bn, bN;
  PetscReal      *A;
  PetscBLASInt   ihi;
  PetscBLASInt   ldA;          /* leading dimension of A */
  PetscBLASInt   ldQ;          /* leading dimension of Q */
  PetscReal      *Q;           /*  orthogonal matrix of  (left) schur vectors */
  PetscReal      *work;        /* working vector */
  PetscBLASInt   lwork;        /* size of the working vector */
  PetscInt       *perm;        /* Permutation vector to sort eigenvalues */
  PetscInt       i, j;
  PetscBLASInt   NbrEig;       /* Number of eigenvalues really extracted */
  PetscReal      *wr, *wi, *modul; /* Real and imaginary part and modul of the eigenvalues of A*/
  PetscBLASInt   *select;
  PetscBLASInt   *iwork;
  PetscBLASInt   liwork;
  PetscScalar    *Ht;           /* Transpose of the Hessenberg matrix */
  PetscScalar    *t;            /* Store the result of the solution of H^T*t=h_{m+1,m}e_m */
  PetscBLASInt   *ipiv;         /* Permutation vector to be used in LAPACK */
  PetscBool      flag;            /* determine whether to use Ritz vectors or harmonic Ritz vectors */

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(N,&bN);CHKERRQ(ierr);
  ihi  = ldQ = bn;
  ldA  = bN;
  ierr = PetscBLASIntCast(5*N,&lwork);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PetscObjectComm((PetscObject)ksp), -1, "NO SUPPORT FOR COMPLEX VALUES AT THIS TIME");
#endif

  ierr = PetscMalloc1(ldA*ldA, &A);CHKERRQ(ierr);
  ierr = PetscMalloc1(ldQ*n, &Q);CHKERRQ(ierr);
  ierr = PetscMalloc1(lwork, &work);CHKERRQ(ierr);
  if (!dgmres->wr) {
    ierr = PetscMalloc1(n, &dgmres->wr);CHKERRQ(ierr);
    ierr = PetscMalloc1(n, &dgmres->wi);CHKERRQ(ierr);
  }
  wr   = dgmres->wr;
  wi   = dgmres->wi;
  ierr = PetscMalloc1(n,&modul);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&perm);CHKERRQ(ierr);
  /* copy the Hessenberg matrix to work space */
  ierr = PetscArraycpy(A, dgmres->hes_origin, ldA*ldA);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(((PetscObject)ksp)->options,((PetscObject)ksp)->prefix, "-ksp_dgmres_harmonic_ritz", &flag);CHKERRQ(ierr);
  if (flag) {
    /* Compute the matrix H + H^{-T}*h^2_{m+1,m}e_m*e_m^T */
    /* Transpose the Hessenberg matrix */
    ierr = PetscMalloc1(bn*bn, &Ht);CHKERRQ(ierr);
    for (i = 0; i < bn; i++) {
      for (j = 0; j < bn; j++) {
        Ht[i * bn + j] = dgmres->hes_origin[j * ldA + i];
      }
    }

    /* Solve the system H^T*t = h_{m+1,m}e_m */
    ierr    = PetscCalloc1(bn, &t);CHKERRQ(ierr);
    t[bn-1] = dgmres->hes_origin[(bn -1) * ldA + bn]; /* Pick the last element H(m+1,m) */
    ierr    = PetscMalloc1(bn, &ipiv);CHKERRQ(ierr);
    /* Call the LAPACK routine dgesv to solve the system Ht^-1 * t */
    {
      PetscBLASInt info;
      PetscBLASInt nrhs = 1;
      PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&bn, &nrhs, Ht, &bn, ipiv, t, &bn, &info));
      if (info) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB, "Error while calling the Lapack routine DGESV");
    }
    /* Now form H + H^{-T}*h^2_{m+1,m}e_m*e_m^T */
    for (i = 0; i < bn; i++) A[(bn-1)*bn+i] += t[i];
    ierr = PetscFree(t);CHKERRQ(ierr);
    ierr = PetscFree(Ht);CHKERRQ(ierr);
  }
  /* Compute eigenvalues with the Schur form */
  {
    PetscBLASInt info=0;
    PetscBLASInt ilo = 1;
    PetscStackCallBLAS("LAPACKhseqr",LAPACKhseqr_("S", "I", &bn, &ilo, &ihi, A, &ldA, wr, wi, Q, &ldQ, work, &lwork, &info));
    if (info) SETERRQ1(PetscObjectComm((PetscObject)ksp), PETSC_ERR_LIB,"Error in LAPACK routine XHSEQR %d",(int) info);
  }
  ierr = PetscFree(work);CHKERRQ(ierr);

  /* sort the eigenvalues */
  for (i=0; i<n; i++) modul[i] = PetscSqrtReal(wr[i]*wr[i]+wi[i]*wi[i]);
  for (i=0; i<n; i++) perm[i] = i;

  ierr = PetscSortRealWithPermutation(n, modul, perm);CHKERRQ(ierr);
  /* save the complex modulus of the largest eigenvalue in magnitude */
  if (dgmres->lambdaN < modul[perm[n-1]]) dgmres->lambdaN=modul[perm[n-1]];
  /* count the number of extracted eigenvalues (with complex conjugates) */
  NbrEig = 0;
  while (NbrEig < dgmres->neig) {
    if (wi[perm[NbrEig]] != 0) NbrEig += 2;
    else NbrEig += 1;
  }
  /* Reorder the Schur decomposition so that the cluster of smallest eigenvalues appears in the leading diagonal blocks of A */

  ierr = PetscCalloc1(n, &select);CHKERRQ(ierr);

  if (!dgmres->GreatestEig) {
    for (j = 0; j < NbrEig; j++) select[perm[j]] = 1;
  } else {
    for (j = 0; j < NbrEig; j++) select[perm[n-j-1]] = 1;
  }
  /* call Lapack dtrsen */
  lwork  =  PetscMax(1, 4 * NbrEig *(bn-NbrEig));
  liwork = PetscMax(1, 2 * NbrEig *(bn-NbrEig));
  ierr   = PetscMalloc1(lwork, &work);CHKERRQ(ierr);
  ierr   = PetscMalloc1(liwork, &iwork);CHKERRQ(ierr);
  {
    PetscBLASInt info=0;
    PetscReal    CondEig;         /* lower bound on the reciprocal condition number for the selected cluster of eigenvalues */
    PetscReal    CondSub;         /* estimated reciprocal condition number of the specified invariant subspace. */
    PetscStackCallBLAS("LAPACKtrsen",LAPACKtrsen_("B", "V", select, &bn, A, &ldA, Q, &ldQ, wr, wi, &NbrEig, &CondEig, &CondSub, work, &lwork, iwork, &liwork, &info));
    if (info == 1) SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_LIB, "UNABLE TO REORDER THE EIGENVALUES WITH THE LAPACK ROUTINE : ILL-CONDITIONED PROBLEM");
  }
  ierr = PetscFree(select);CHKERRQ(ierr);

  /* Extract the Schur vectors */
  for (j = 0; j < NbrEig; j++) {
    ierr = PetscArraycpy(&SR[j*N], &(Q[j*ldQ]), n);CHKERRQ(ierr);
  }
  *neig = NbrEig;
  ierr  = PetscFree(A);CHKERRQ(ierr);
  ierr  = PetscFree(work);CHKERRQ(ierr);
  ierr  = PetscFree(perm);CHKERRQ(ierr);
  ierr  = PetscFree(work);CHKERRQ(ierr);
  ierr  = PetscFree(iwork);CHKERRQ(ierr);
  ierr  = PetscFree(modul);CHKERRQ(ierr);
  ierr  = PetscFree(Q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPDGMRESApplyDeflation_DGMRES(KSP ksp, Vec x, Vec y)
{
  KSP_DGMRES     *dgmres = (KSP_DGMRES*) ksp->data;
  PetscInt       i, r     = dgmres->r;
  PetscErrorCode ierr;
  PetscReal      alpha    = 1.0;
  PetscInt       max_neig = dgmres->max_neig;
  PetscBLASInt   br,bmax;
  PetscReal      lambda = dgmres->lambdaN;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(r,&br);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(max_neig,&bmax);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(KSP_DGMRESApplyDeflation, ksp, 0, 0, 0);CHKERRQ(ierr);
  if (!r) {
    ierr = VecCopy(x,y);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* Compute U'*x */
  if (!X1) {
    ierr = PetscMalloc1(bmax, &X1);CHKERRQ(ierr);
    ierr = PetscMalloc1(bmax, &X2);CHKERRQ(ierr);
  }
  ierr = VecMDot(x, r, UU, X1);CHKERRQ(ierr);

  /* Solve T*X1=X2 for X1*/
  ierr = PetscArraycpy(X2, X1, br);CHKERRQ(ierr);
  {
    PetscBLASInt info;
    PetscBLASInt nrhs = 1;
    PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_("N", &br, &nrhs, TTF, &bmax, INVP, X1, &bmax, &info));
    if (info) SETERRQ1(PetscObjectComm((PetscObject)ksp), PETSC_ERR_LIB,"Error in LAPACK routine XGETRS %d", (int) info);
  }
  /* Iterative refinement -- is it really necessary ?? */
  if (!WORK) {
    ierr = PetscMalloc1(3*bmax, &WORK);CHKERRQ(ierr);
    ierr = PetscMalloc1(bmax, &IWORK);CHKERRQ(ierr);
  }
  {
    PetscBLASInt info;
    PetscReal    berr, ferr;
    PetscBLASInt nrhs = 1;
    PetscStackCallBLAS("LAPACKgerfs",LAPACKgerfs_("N", &br, &nrhs, TT, &bmax, TTF, &bmax, INVP, X2, &bmax,X1, &bmax, &ferr, &berr, WORK, IWORK, &info));
    if (info) SETERRQ1(PetscObjectComm((PetscObject)ksp), PETSC_ERR_LIB,"Error in LAPACK routine XGERFS %d", (int) info);
  }

  for (i = 0; i < r; i++) X2[i] =  X1[i]/lambda - X2[i];

  /* Compute X2=U*X2 */
  ierr = VecZeroEntries(y);CHKERRQ(ierr);
  ierr = VecMAXPY(y, r, X2, UU);CHKERRQ(ierr);
  ierr = VecAXPY(y, alpha, x);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(KSP_DGMRESApplyDeflation, ksp, 0, 0, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPDGMRESImproveEig_DGMRES(KSP ksp, PetscInt neig)
{
  KSP_DGMRES   *dgmres = (KSP_DGMRES*) ksp->data;
  PetscInt     j,r_old, r = dgmres->r;
  PetscBLASInt i     = 0;
  PetscInt     neig1 = dgmres->neig + EIG_OFFSET;
  PetscInt     bmax  = dgmres->max_neig;
  PetscInt     aug   = r + neig;         /* actual size of the augmented invariant basis */
  PetscInt     aug1  = bmax+neig1;       /* maximum size of the augmented invariant basis */
  PetscBLASInt ldA;            /* leading dimension of AUAU and AUU*/
  PetscBLASInt N;              /* size of AUAU */
  PetscReal    *Q;             /*  orthogonal matrix of  (left) schur vectors */
  PetscReal    *Z;             /*  orthogonal matrix of  (right) schur vectors */
  PetscReal    *work;          /* working vector */
  PetscBLASInt lwork;          /* size of the working vector */
  PetscInt     *perm;          /* Permutation vector to sort eigenvalues */
  PetscReal    *wr, *wi, *beta, *modul; /* Real and imaginary part and modul of the eigenvalues of A*/
  PetscInt     ierr;
  PetscBLASInt NbrEig = 0,nr,bm;
  PetscBLASInt *select;
  PetscBLASInt liwork, *iwork;

  PetscFunctionBegin;
  /* Block construction of the matrices AUU=(AU)'*U and (AU)'*AU*/
  if (!AUU) {
    ierr = PetscMalloc1(aug1*aug1, &AUU);CHKERRQ(ierr);
    ierr = PetscMalloc1(aug1*aug1, &AUAU);CHKERRQ(ierr);
  }
  /* AUU = (AU)'*U = [(MU)'*U (MU)'*X; (MX)'*U (MX)'*X]
   * Note that MU and MX have been computed previously either in ComputeDataDeflation() or down here in a previous call to this function */
  /* (MU)'*U size (r x r) -- store in the <r> first columns of AUU*/
  for (j=0; j < r; j++) {
    ierr = VecMDot(UU[j], r, MU, &AUU[j*aug1]);CHKERRQ(ierr);
  }
  /* (MU)'*X size (r x neig) -- store in AUU from the column <r>*/
  for (j = 0; j < neig; j++) {
    ierr = VecMDot(XX[j], r, MU, &AUU[(r+j) *aug1]);CHKERRQ(ierr);
  }
  /* (MX)'*U size (neig x r) -- store in the <r> first columns of AUU from the row <r>*/
  for (j = 0; j < r; j++) {
    ierr = VecMDot(UU[j], neig, MX, &AUU[j*aug1+r]);CHKERRQ(ierr);
  }
  /* (MX)'*X size (neig neig) --  store in AUU from the column <r> and the row <r>*/
  for (j = 0; j < neig; j++) {
    ierr = VecMDot(XX[j], neig, MX, &AUU[(r+j) *aug1 + r]);CHKERRQ(ierr);
  }

  /* AUAU = (AU)'*AU = [(MU)'*MU (MU)'*MX; (MX)'*MU (MX)'*MX] */
  /* (MU)'*MU size (r x r) -- store in the <r> first columns of AUAU*/
  for (j=0; j < r; j++) {
    ierr = VecMDot(MU[j], r, MU, &AUAU[j*aug1]);CHKERRQ(ierr);
  }
  /* (MU)'*MX size (r x neig) -- store in AUAU from the column <r>*/
  for (j = 0; j < neig; j++) {
    ierr = VecMDot(MX[j], r, MU, &AUAU[(r+j) *aug1]);CHKERRQ(ierr);
  }
  /* (MX)'*MU size (neig x r) -- store in the <r> first columns of AUAU from the row <r>*/
  for (j = 0; j < r; j++) {
    ierr = VecMDot(MU[j], neig, MX, &AUAU[j*aug1+r]);CHKERRQ(ierr);
  }
  /* (MX)'*MX size (neig neig) --  store in AUAU from the column <r> and the row <r>*/
  for (j = 0; j < neig; j++) {
    ierr = VecMDot(MX[j], neig, MX, &AUAU[(r+j) *aug1 + r]);CHKERRQ(ierr);
  }

  /* Computation of the eigenvectors */
  ierr  = PetscBLASIntCast(aug1,&ldA);CHKERRQ(ierr);
  ierr  = PetscBLASIntCast(aug,&N);CHKERRQ(ierr);
  lwork = 8 * N + 20; /* sizeof the working space */
  ierr  = PetscMalloc1(N, &wr);CHKERRQ(ierr);
  ierr  = PetscMalloc1(N, &wi);CHKERRQ(ierr);
  ierr  = PetscMalloc1(N, &beta);CHKERRQ(ierr);
  ierr  = PetscMalloc1(N, &modul);CHKERRQ(ierr);
  ierr  = PetscMalloc1(N, &perm);CHKERRQ(ierr);
  ierr  = PetscMalloc1(N*N, &Q);CHKERRQ(ierr);
  ierr  = PetscMalloc1(N*N, &Z);CHKERRQ(ierr);
  ierr  = PetscMalloc1(lwork, &work);CHKERRQ(ierr);
  {
    PetscBLASInt info=0;
    PetscStackCallBLAS("LAPACKgges",LAPACKgges_("V", "V", "N", NULL, &N, AUAU, &ldA, AUU, &ldA, &i, wr, wi, beta, Q, &N, Z, &N, work, &lwork, NULL, &info));
    if (info) SETERRQ1 (PetscObjectComm((PetscObject)ksp), PETSC_ERR_LIB,"Error in LAPACK routine XGGES %d", (int) info);
  }
  for (i=0; i<N; i++) {
    if (beta[i] !=0.0) {
      wr[i] /=beta[i];
      wi[i] /=beta[i];
    }
  }
  /* sort the eigenvalues */
  for (i=0; i<N; i++) modul[i]=PetscSqrtReal(wr[i]*wr[i]+wi[i]*wi[i]);
  for (i=0; i<N; i++) perm[i] = i;
  ierr = PetscSortRealWithPermutation(N, modul, perm);CHKERRQ(ierr);
  /* Save the norm of the largest eigenvalue */
  if (dgmres->lambdaN < modul[perm[N-1]]) dgmres->lambdaN = modul[perm[N-1]];
  /* Allocate space to extract the first r schur vectors   */
  if (!SR2) {
    ierr = PetscMalloc1(aug1*bmax, &SR2);CHKERRQ(ierr);
  }
  /* count the number of extracted eigenvalues (complex conjugates count as 2) */
  while (NbrEig < bmax) {
    if (wi[perm[NbrEig]] == 0) NbrEig += 1;
    else NbrEig += 2;
  }
  if (NbrEig > bmax) NbrEig = bmax - 1;
  r_old     = r; /* previous size of r */
  dgmres->r = r = NbrEig;

  /* Select the eigenvalues to reorder */
  ierr = PetscCalloc1(N, &select);CHKERRQ(ierr);
  if (!dgmres->GreatestEig) {
    for (j = 0; j < NbrEig; j++) select[perm[j]] = 1;
  } else {
    for (j = 0; j < NbrEig; j++) select[perm[N-j-1]] = 1;
  }
  /* Reorder and extract the new <r> schur vectors */
  lwork  = PetscMax(4 * N + 16,  2 * NbrEig *(N - NbrEig));
  liwork = PetscMax(N + 6,  2 * NbrEig *(N - NbrEig));
  ierr   = PetscFree(work);CHKERRQ(ierr);
  ierr   = PetscMalloc1(lwork, &work);CHKERRQ(ierr);
  ierr   = PetscMalloc1(liwork, &iwork);CHKERRQ(ierr);
  {
    PetscBLASInt info=0;
    PetscReal    Dif[2];
    PetscBLASInt ijob  = 2;
    PetscBLASInt wantQ = 1, wantZ = 1;
    PetscStackCallBLAS("LAPACKtgsen",LAPACKtgsen_(&ijob, &wantQ, &wantZ, select, &N, AUAU, &ldA, AUU, &ldA, wr, wi, beta, Q, &N, Z, &N, &NbrEig, NULL, NULL, &(Dif[0]), work, &lwork, iwork, &liwork, &info));
    if (info == 1) SETERRQ(PetscObjectComm((PetscObject)ksp), -1, "UNABLE TO REORDER THE EIGENVALUES WITH THE LAPACK ROUTINE : ILL-CONDITIONED PROBLEM");
  }
  ierr = PetscFree(select);CHKERRQ(ierr);

  for (j=0; j<r; j++) {
    ierr = PetscArraycpy(&SR2[j*aug1], &(Z[j*N]), N);CHKERRQ(ierr);
  }

  /* Multiply the Schur vectors SR2 by U (and X)  to get a new U
   -- save it temporarily in MU */
  for (j = 0; j < r; j++) {
    ierr = VecZeroEntries(MU[j]);CHKERRQ(ierr);
    ierr = VecMAXPY(MU[j], r_old, &SR2[j*aug1], UU);CHKERRQ(ierr);
    ierr = VecMAXPY(MU[j], neig, &SR2[j*aug1+r_old], XX);CHKERRQ(ierr);
  }
  /* Form T = U'*MU*U */
  for (j = 0; j < r; j++) {
    ierr = VecCopy(MU[j], UU[j]);CHKERRQ(ierr);
    ierr = KSP_PCApplyBAorAB(ksp, UU[j], MU[j], VEC_TEMP_MATOP);CHKERRQ(ierr);
  }
  dgmres->matvecs += r;
  for (j = 0; j < r; j++) {
    ierr = VecMDot(MU[j], r, UU, &TT[j*bmax]);CHKERRQ(ierr);
  }
  /* Factorize T */
  ierr = PetscArraycpy(TTF, TT, bmax*r);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(r,&nr);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(bmax,&bm);CHKERRQ(ierr);
  {
    PetscBLASInt info;
    PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&nr, &nr, TTF, &bm, INVP, &info));
    if (info) SETERRQ1(PetscObjectComm((PetscObject)ksp), PETSC_ERR_LIB,"Error in LAPACK routine XGETRF INFO=%d",(int) info);
  }
  /* Free Memory */
  ierr = PetscFree(wr);CHKERRQ(ierr);
  ierr = PetscFree(wi);CHKERRQ(ierr);
  ierr = PetscFree(beta);CHKERRQ(ierr);
  ierr = PetscFree(modul);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);
  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(Z);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* end new DGMRES functions */

/*MC
     KSPDGMRES - Implements the deflated GMRES as defined in [1,2].
                 In this implementation, the adaptive strategy allows to switch to the deflated GMRES when the
                 stagnation occurs.

   Options Database Keys:
   GMRES Options (inherited):
+   -ksp_gmres_restart <restart> - the number of Krylov directions to orthogonalize against
.   -ksp_gmres_haptol <tol> - sets the tolerance for "happy ending" (exact convergence)
.   -ksp_gmres_preallocate - preallocate all the Krylov search directions initially (otherwise groups of
                             vectors are allocated as needed)
.   -ksp_gmres_classicalgramschmidt - use classical (unmodified) Gram-Schmidt to orthogonalize against the Krylov space (fast) (the default)
.   -ksp_gmres_modifiedgramschmidt - use modified Gram-Schmidt in the orthogonalization (more stable, but slower)
.   -ksp_gmres_cgs_refinement_type <refine_never,refine_ifneeded,refine_always> - determine if iterative refinement is used to increase the
                                   stability of the classical Gram-Schmidt  orthogonalization.
-   -ksp_gmres_krylov_monitor - plot the Krylov space generated

   DGMRES Options Database Keys:
+   -ksp_dgmres_eigen <neig> - number of smallest eigenvalues to extract at each restart
.   -ksp_dgmres_max_eigen <max_neig> - maximum number of eigenvalues that can be extracted during the iterative
                                       process
.   -ksp_dgmres_force - use the deflation at each restart; switch off the adaptive strategy.
-   -ksp_dgmres_view_deflation_vecs <viewerspec> - View the deflation vectors, where viewerspec is a key that can be
                                                   parsed by PetscOptionsGetViewer().  If neig > 1, viewerspec should
                                                   end with ":append".  No vectors will be viewed if the adaptive
                                                   strategy chooses not to deflate, so -ksp_dgmres_force should also
                                                   be given.
                                                   The deflation vectors span a subspace that may be a good
                                                   approximation of the subspace of smallest eigenvectors of the
                                                   preconditioned operator, so this option can aid in understanding
                                                   the performance of a preconditioner.

 Level: beginner

 Notes:
    Left and right preconditioning are supported, but not symmetric preconditioning. Complex arithmetic is not yet supported

 References:
+  1. - J. Erhel, K. Burrage and B. Pohl,  Restarted GMRES preconditioned by deflation,J. Computational and Applied Mathematics, 69(1996).
-  2. - D. NUENTSA WAKAM and F. PACULL, Memory Efficient Hybrid Algebraic Solvers for Linear Systems Arising from Compressible Flows, Computers and Fluids,
   In Press, http://dx.doi.org/10.1016/j.compfluid.2012.03.023

 Contributed by: Desire NUENTSA WAKAM,INRIA

 .seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPFGMRES, KSPLGMRES,
 KSPGMRESSetRestart(), KSPGMRESSetHapTol(), KSPGMRESSetPreAllocateVectors(), KSPGMRESSetOrthogonalization(), KSPGMRESGetOrthogonalization(),
 KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESModifiedGramSchmidtOrthogonalization(),
 KSPGMRESCGSRefinementType, KSPGMRESSetCGSRefinementType(), KSPGMRESGetCGSRefinementType(), KSPGMRESMonitorKrylov(), KSPSetPCSide()

 M*/

PETSC_EXTERN PetscErrorCode KSPCreate_DGMRES(KSP ksp)
{
  KSP_DGMRES     *dgmres;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr      = PetscNewLog(ksp,&dgmres);CHKERRQ(ierr);
  ksp->data = (void*) dgmres;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1);CHKERRQ(ierr);

  ksp->ops->buildsolution                = KSPBuildSolution_DGMRES;
  ksp->ops->setup                        = KSPSetUp_DGMRES;
  ksp->ops->solve                        = KSPSolve_DGMRES;
  ksp->ops->destroy                      = KSPDestroy_DGMRES;
  ksp->ops->view                         = KSPView_DGMRES;
  ksp->ops->setfromoptions               = KSPSetFromOptions_DGMRES;
  ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_GMRES;
  ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_GMRES;

  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetPreAllocateVectors_C",KSPGMRESSetPreAllocateVectors_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetOrthogonalization_C",KSPGMRESSetOrthogonalization_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetRestart_C",KSPGMRESSetRestart_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetHapTol_C",KSPGMRESSetHapTol_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetCGSRefinementType_C",KSPGMRESSetCGSRefinementType_GMRES);CHKERRQ(ierr);
  /* -- New functions defined in DGMRES -- */
  ierr = PetscObjectComposeFunction((PetscObject)ksp, "KSPDGMRESSetEigen_C",KSPDGMRESSetEigen_DGMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp, "KSPDGMRESSetMaxEigen_C",KSPDGMRESSetMaxEigen_DGMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp, "KSPDGMRESSetRatio_C",KSPDGMRESSetRatio_DGMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp, "KSPDGMRESForce_C",KSPDGMRESForce_DGMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp, "KSPDGMRESComputeSchurForm_C",KSPDGMRESComputeSchurForm_DGMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp, "KSPDGMRESComputeDeflationData_C",KSPDGMRESComputeDeflationData_DGMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp, "KSPDGMRESApplyDeflation_C",KSPDGMRESApplyDeflation_DGMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp, "KSPDGMRESImproveEig_C", KSPDGMRESImproveEig_DGMRES);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("DGMRESCompDefl",  KSP_CLASSID, &KSP_DGMRESComputeDeflationData);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DGMRESApplyDefl", KSP_CLASSID, &KSP_DGMRESApplyDeflation);CHKERRQ(ierr);

  dgmres->haptol         = 1.0e-30;
  dgmres->q_preallocate  = 0;
  dgmres->delta_allocate = GMRES_DELTA_DIRECTIONS;
  dgmres->orthog         = KSPGMRESClassicalGramSchmidtOrthogonalization;
  dgmres->nrs            = NULL;
  dgmres->sol_temp       = NULL;
  dgmres->max_k          = GMRES_DEFAULT_MAXK;
  dgmres->Rsvd           = NULL;
  dgmres->cgstype        = KSP_GMRES_CGS_REFINE_NEVER;
  dgmres->orthogwork     = NULL;

  /* Default values for the deflation */
  dgmres->r           = 0;
  dgmres->neig        = DGMRES_DEFAULT_EIG;
  dgmres->max_neig    = DGMRES_DEFAULT_MAXEIG-1;
  dgmres->lambdaN     = 0.0;
  dgmres->smv         = SMV;
  dgmres->matvecs     = 0;
  dgmres->GreatestEig = PETSC_FALSE; /* experimental */
  dgmres->HasSchur    = PETSC_FALSE;
  PetscFunctionReturn(0);
}
