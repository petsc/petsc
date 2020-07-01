
/*
    This file implements GMRES (a Generalized Minimal Residual) method.
    Reference:  Saad and Schultz, 1986.


    Some comments on left vs. right preconditioning, and restarts.
    Left and right preconditioning.
    If right preconditioning is chosen, then the problem being solved
    by gmres is actually
       My =  AB^-1 y = f
    so the initial residual is
          r = f - Mx
    Note that B^-1 y = x or y = B x, and if x is non-zero, the initial
    residual is
          r = f - A x
    The final solution is then
          x = B^-1 y

    If left preconditioning is chosen, then the problem being solved is
       My = B^-1 A x = B^-1 f,
    and the initial residual is
       r  = B^-1(f - Ax)

    Restarts:  Restarts are basically solves with x0 not equal to zero.
    Note that we can eliminate an extra application of B^-1 between
    restarts as long as we don't require that the solution at the end
    of an unsuccessful gmres iteration always be the solution x.
 */

#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>       /*I  "petscksp.h"  I*/
#define GMRES_DELTA_DIRECTIONS 10
#define GMRES_DEFAULT_MAXK     30
static PetscErrorCode KSPGMRESUpdateHessenberg(KSP,PetscInt,PetscBool,PetscReal*);
static PetscErrorCode KSPGMRESBuildSoln(PetscScalar*,Vec,Vec,KSP,PetscInt);

PetscErrorCode    KSPSetUp_GMRES(KSP ksp)
{
  PetscInt       hh,hes,rs,cc;
  PetscErrorCode ierr;
  PetscInt       max_k,k;
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;

  PetscFunctionBegin;
  max_k = gmres->max_k;          /* restart size */
  hh    = (max_k + 2) * (max_k + 1);
  hes   = (max_k + 1) * (max_k + 1);
  rs    = (max_k + 2);
  cc    = (max_k + 1);

  ierr = PetscCalloc5(hh,&gmres->hh_origin,hes,&gmres->hes_origin,rs,&gmres->rs_origin,cc,&gmres->cc_origin,cc,&gmres->ss_origin);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ksp,(hh + hes + rs + 2*cc)*sizeof(PetscScalar));CHKERRQ(ierr);

  if (ksp->calc_sings) {
    /* Allocate workspace to hold Hessenberg matrix needed by lapack */
    ierr = PetscMalloc1((max_k + 3)*(max_k + 9),&gmres->Rsvd);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)ksp,(max_k + 3)*(max_k + 9)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMalloc1(6*(max_k+2),&gmres->Dsvd);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)ksp,6*(max_k+2)*sizeof(PetscReal));CHKERRQ(ierr);
  }

  /* Allocate array to hold pointers to user vectors.  Note that we need
   4 + max_k + 1 (since we need it+1 vectors, and it <= max_k) */
  gmres->vecs_allocated = VEC_OFFSET + 2 + max_k + gmres->nextra_vecs;

  ierr = PetscMalloc1(gmres->vecs_allocated,&gmres->vecs);CHKERRQ(ierr);
  ierr = PetscMalloc1(VEC_OFFSET+2+max_k,&gmres->user_work);CHKERRQ(ierr);
  ierr = PetscMalloc1(VEC_OFFSET+2+max_k,&gmres->mwork_alloc);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ksp,(VEC_OFFSET+2+max_k)*(sizeof(Vec*)+sizeof(PetscInt)) + gmres->vecs_allocated*sizeof(Vec));CHKERRQ(ierr);

  if (gmres->q_preallocate) {
    gmres->vv_allocated = VEC_OFFSET + 2 + max_k;

    ierr = KSPCreateVecs(ksp,gmres->vv_allocated,&gmres->user_work[0],0,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(ksp,gmres->vv_allocated,gmres->user_work[0]);CHKERRQ(ierr);

    gmres->mwork_alloc[0] = gmres->vv_allocated;
    gmres->nwork_alloc    = 1;
    for (k=0; k<gmres->vv_allocated; k++) {
      gmres->vecs[k] = gmres->user_work[0][k];
    }
  } else {
    gmres->vv_allocated = 5;

    ierr = KSPCreateVecs(ksp,5,&gmres->user_work[0],0,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(ksp,5,gmres->user_work[0]);CHKERRQ(ierr);

    gmres->mwork_alloc[0] = 5;
    gmres->nwork_alloc    = 1;
    for (k=0; k<gmres->vv_allocated; k++) {
      gmres->vecs[k] = gmres->user_work[0][k];
    }
  }
  PetscFunctionReturn(0);
}

/*
    Run gmres, possibly with restart.  Return residual history if requested.
    input parameters:

.        gmres  - structure containing parameters and work areas

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
PetscErrorCode KSPGMRESCycle(PetscInt *itcount,KSP ksp)
{
  KSP_GMRES      *gmres = (KSP_GMRES*)(ksp->data);
  PetscReal      res_norm,res,hapbnd,tt;
  PetscErrorCode ierr;
  PetscInt       it     = 0, max_k = gmres->max_k;
  PetscBool      hapend = PETSC_FALSE;

  PetscFunctionBegin;
  if (itcount) *itcount = 0;
  ierr    = VecNormalize(VEC_VV(0),&res_norm);CHKERRQ(ierr);
  KSPCheckNorm(ksp,res_norm);
  res     = res_norm;
  *GRS(0) = res_norm;

  /* check for the convergence */
  ierr       = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->rnorm = res;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
  gmres->it  = (it - 1);
  ierr = KSPLogResidualHistory(ksp,res);CHKERRQ(ierr);
  ierr = KSPMonitor(ksp,ksp->its,res);CHKERRQ(ierr);
  if (!res) {
    ksp->reason = KSP_CONVERGED_ATOL;
    ierr        = PetscInfo(ksp,"Converged due to zero residual norm on entry\n");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = (*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  while (!ksp->reason && it < max_k && ksp->its < ksp->max_it) {
    if (it) {
      ierr = KSPLogResidualHistory(ksp,res);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,ksp->its,res);CHKERRQ(ierr);
    }
    gmres->it = (it - 1);
    if (gmres->vv_allocated <= it + VEC_OFFSET + 1) {
      ierr = KSPGMRESGetNewVectors(ksp,it+1);CHKERRQ(ierr);
    }
    ierr = KSP_PCApplyBAorAB(ksp,VEC_VV(it),VEC_VV(1+it),VEC_TEMP_MATOP);CHKERRQ(ierr);

    /* update hessenberg matrix and do Gram-Schmidt */
    ierr = (*gmres->orthog)(ksp,it);CHKERRQ(ierr);
    if (ksp->reason) break;

    /* vv(i+1) . vv(i+1) */
    ierr = VecNormalize(VEC_VV(it+1),&tt);CHKERRQ(ierr);
    KSPCheckNorm(ksp,tt);

    /* save the magnitude */
    *HH(it+1,it)  = tt;
    *HES(it+1,it) = tt;

    /* check for the happy breakdown */
    hapbnd = PetscAbsScalar(tt / *GRS(it));
    if (hapbnd > gmres->haptol) hapbnd = gmres->haptol;
    if (tt < hapbnd) {
      ierr   = PetscInfo2(ksp,"Detected happy breakdown, current hapbnd = %14.12e tt = %14.12e\n",(double)hapbnd,(double)tt);CHKERRQ(ierr);
      hapend = PETSC_TRUE;
    }
    ierr = KSPGMRESUpdateHessenberg(ksp,it,hapend,&res);CHKERRQ(ierr);

    it++;
    gmres->it = (it-1);   /* For converged */
    ksp->its++;
    ksp->rnorm = res;
    if (ksp->reason) break;

    ierr = (*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);

    /* Catch error in happy breakdown and signal convergence and break from loop */
    if (hapend) {
      if (ksp->normtype == KSP_NORM_NONE) { /* convergence test was skipped in this case */
        ksp->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      } else if (!ksp->reason) {
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
    ierr = KSPLogResidualHistory(ksp,res);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,ksp->its,res);CHKERRQ(ierr);
  }

  if (itcount) *itcount = it;


  /*
    Down here we have to solve for the "best" coefficients of the Krylov
    columns, add the solution values together, and possibly unwind the
    preconditioning from the solution
   */
  /* Form the solution (or the solution so far) */
  ierr = KSPGMRESBuildSoln(GRS(0),ksp->vec_sol,ksp->vec_sol,ksp,it-1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSolve_GMRES(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       its,itcount,i;
  KSP_GMRES      *gmres     = (KSP_GMRES*)ksp->data;
  PetscBool      guess_zero = ksp->guess_zero;
  PetscInt       N = gmres->max_k + 1;
  PetscBLASInt   bN;

  PetscFunctionBegin;
  if (ksp->calc_sings && !gmres->Rsvd) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ORDER,"Must call KSPSetComputeSingularValues() before KSPSetUp() is called");

  ierr     = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->its = 0;
  ierr     = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);

  itcount     = 0;
  gmres->fullcycle = 0;
  ksp->reason = KSP_CONVERGED_ITERATING;
  while (!ksp->reason) {
    ierr     = KSPInitialResidual(ksp,ksp->vec_sol,VEC_TEMP,VEC_TEMP_MATOP,VEC_VV(0),ksp->vec_rhs);CHKERRQ(ierr);
    ierr     = KSPGMRESCycle(&its,ksp);CHKERRQ(ierr);
    /* Store the Hessenberg matrix and the basis vectors of the Krylov subspace
    if the cycle is complete for the computation of the Ritz pairs */
    if (its == gmres->max_k) {
      gmres->fullcycle++;
      if (ksp->calc_ritz) {
        if (!gmres->hes_ritz) {
          ierr = PetscMalloc1(N*N,&gmres->hes_ritz);CHKERRQ(ierr);
          ierr = PetscLogObjectMemory((PetscObject)ksp,N*N*sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = VecDuplicateVecs(VEC_VV(0),N,&gmres->vecb);CHKERRQ(ierr);
        }
        ierr = PetscBLASIntCast(N,&bN);CHKERRQ(ierr);
        ierr = PetscArraycpy(gmres->hes_ritz,gmres->hes_origin,bN*bN);CHKERRQ(ierr);
        for (i=0; i<gmres->max_k+1; i++) {
          ierr = VecCopy(VEC_VV(i),gmres->vecb[i]);CHKERRQ(ierr);
        }
      }
    }
    itcount += its;
    if (itcount >= ksp->max_it) {
      if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
      break;
    }
    ksp->guess_zero = PETSC_FALSE; /* every future call to KSPInitialResidual() will have nonzero guess */
  }
  ksp->guess_zero = guess_zero; /* restore if user provided nonzero initial guess */
  PetscFunctionReturn(0);
}

PetscErrorCode KSPReset_GMRES(KSP ksp)
{
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  /* Free the Hessenberg matrices */
  ierr = PetscFree5(gmres->hh_origin,gmres->hes_origin,gmres->rs_origin,gmres->cc_origin,gmres->ss_origin);CHKERRQ(ierr);
  ierr = PetscFree(gmres->hes_ritz);CHKERRQ(ierr);

  /* free work vectors */
  ierr = PetscFree(gmres->vecs);CHKERRQ(ierr);
  for (i=0; i<gmres->nwork_alloc; i++) {
    ierr = VecDestroyVecs(gmres->mwork_alloc[i],&gmres->user_work[i]);CHKERRQ(ierr);
  }
  gmres->nwork_alloc = 0;
  if (gmres->vecb)  {
    ierr = VecDestroyVecs(gmres->max_k+1,&gmres->vecb);CHKERRQ(ierr);
  }

  ierr = PetscFree(gmres->user_work);CHKERRQ(ierr);
  ierr = PetscFree(gmres->mwork_alloc);CHKERRQ(ierr);
  ierr = PetscFree(gmres->nrs);CHKERRQ(ierr);
  ierr = VecDestroy(&gmres->sol_temp);CHKERRQ(ierr);
  ierr = PetscFree(gmres->Rsvd);CHKERRQ(ierr);
  ierr = PetscFree(gmres->Dsvd);CHKERRQ(ierr);
  ierr = PetscFree(gmres->orthogwork);CHKERRQ(ierr);

  gmres->sol_temp       = 0;
  gmres->vv_allocated   = 0;
  gmres->vecs_allocated = 0;
  gmres->sol_temp       = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPDestroy_GMRES(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPReset_GMRES(ksp);CHKERRQ(ierr);
  ierr = PetscFree(ksp->data);CHKERRQ(ierr);
  /* clear composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetPreAllocateVectors_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetOrthogonalization_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESGetOrthogonalization_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESGetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetHapTol_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetCGSRefinementType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESGetCGSRefinementType_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
    KSPGMRESBuildSoln - create the solution from the starting vector and the
    current iterates.

    Input parameters:
        nrs - work area of size it + 1.
        vs  - index of initial guess
        vdest - index of result.  Note that vs may == vdest (replace
                guess with the solution).

     This is an internal routine that knows about the GMRES internals.
 */
static PetscErrorCode KSPGMRESBuildSoln(PetscScalar *nrs,Vec vs,Vec vdest,KSP ksp,PetscInt it)
{
  PetscScalar    tt;
  PetscErrorCode ierr;
  PetscInt       ii,k,j;
  KSP_GMRES      *gmres = (KSP_GMRES*)(ksp->data);

  PetscFunctionBegin;
  /* Solve for solution vector that minimizes the residual */

  /* If it is < 0, no gmres steps have been performed */
  if (it < 0) {
    ierr = VecCopy(vs,vdest);CHKERRQ(ierr); /* VecCopy() is smart, exists immediately if vguess == vdest */
    PetscFunctionReturn(0);
  }
  if (*HH(it,it) != 0.0) {
    nrs[it] = *GRS(it) / *HH(it,it);
  } else {
    if (ksp->errorifnotconverged) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"You reached the break down in GMRES; HH(it,it) = 0");
    else ksp->reason = KSP_DIVERGED_BREAKDOWN;

    ierr = PetscInfo2(ksp,"Likely your matrix or preconditioner is singular. HH(it,it) is identically zero; it = %D GRS(it) = %g\n",it,(double)PetscAbsScalar(*GRS(it)));CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  for (ii=1; ii<=it; ii++) {
    k  = it - ii;
    tt = *GRS(k);
    for (j=k+1; j<=it; j++) tt = tt - *HH(k,j) * nrs[j];
    if (*HH(k,k) == 0.0) {
      if (ksp->errorifnotconverged) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"Likely your matrix or preconditioner is singular. HH(k,k) is identically zero; k = %D\n",k);
      else {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        ierr = PetscInfo1(ksp,"Likely your matrix or preconditioner is singular. HH(k,k) is identically zero; k = %D\n",k);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
    }
    nrs[k] = tt / *HH(k,k);
  }

  /* Accumulate the correction to the solution of the preconditioned problem in TEMP */
  ierr = VecSet(VEC_TEMP,0.0);CHKERRQ(ierr);
  ierr = VecMAXPY(VEC_TEMP,it+1,nrs,&VEC_VV(0));CHKERRQ(ierr);

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
static PetscErrorCode KSPGMRESUpdateHessenberg(KSP ksp,PetscInt it,PetscBool hapend,PetscReal *res)
{
  PetscScalar *hh,*cc,*ss,tt;
  PetscInt    j;
  KSP_GMRES   *gmres = (KSP_GMRES*)(ksp->data);

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
    *hh = *cc++ * *hh - (*ss++ * tt);
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
      if (ksp->errorifnotconverged) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"tt == 0.0");
      else {
        ksp->reason = KSP_DIVERGED_NULL;
        PetscFunctionReturn(0);
      }
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
PetscErrorCode KSPGMRESGetNewVectors(KSP ksp,PetscInt it)
{
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       nwork = gmres->nwork_alloc,k,nalloc;

  PetscFunctionBegin;
  nalloc = PetscMin(ksp->max_it,gmres->delta_allocate);
  /* Adjust the number to allocate to make sure that we don't exceed the
    number of available slots */
  if (it + VEC_OFFSET + nalloc >= gmres->vecs_allocated) {
    nalloc = gmres->vecs_allocated - it - VEC_OFFSET;
  }
  if (!nalloc) PetscFunctionReturn(0);

  gmres->vv_allocated += nalloc;

  ierr = KSPCreateVecs(ksp,nalloc,&gmres->user_work[nwork],0,NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(ksp,nalloc,gmres->user_work[nwork]);CHKERRQ(ierr);

  gmres->mwork_alloc[nwork] = nalloc;
  for (k=0; k<nalloc; k++) {
    gmres->vecs[it+VEC_OFFSET+k] = gmres->user_work[nwork][k];
  }
  gmres->nwork_alloc++;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPBuildSolution_GMRES(KSP ksp,Vec ptr,Vec *result)
{
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ptr) {
    if (!gmres->sol_temp) {
      ierr = VecDuplicate(ksp->vec_sol,&gmres->sol_temp);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)ksp,(PetscObject)gmres->sol_temp);CHKERRQ(ierr);
    }
    ptr = gmres->sol_temp;
  }
  if (!gmres->nrs) {
    /* allocate the work area */
    ierr = PetscMalloc1(gmres->max_k,&gmres->nrs);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)ksp,gmres->max_k);CHKERRQ(ierr);
  }

  ierr = KSPGMRESBuildSoln(gmres->nrs,ksp->vec_sol,ptr,ksp,gmres->it);CHKERRQ(ierr);
  if (result) *result = ptr;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPView_GMRES(KSP ksp,PetscViewer viewer)
{
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  const char     *cstr;
  PetscErrorCode ierr;
  PetscBool      iascii,isstring;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (gmres->orthog == KSPGMRESClassicalGramSchmidtOrthogonalization) {
    switch (gmres->cgstype) {
    case (KSP_GMRES_CGS_REFINE_NEVER):
      cstr = "Classical (unmodified) Gram-Schmidt Orthogonalization with no iterative refinement";
      break;
    case (KSP_GMRES_CGS_REFINE_ALWAYS):
      cstr = "Classical (unmodified) Gram-Schmidt Orthogonalization with one step of iterative refinement";
      break;
    case (KSP_GMRES_CGS_REFINE_IFNEEDED):
      cstr = "Classical (unmodified) Gram-Schmidt Orthogonalization with one step of iterative refinement when needed";
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Unknown orthogonalization");
    }
  } else if (gmres->orthog == KSPGMRESModifiedGramSchmidtOrthogonalization) {
    cstr = "Modified Gram-Schmidt Orthogonalization";
  } else {
    cstr = "unknown orthogonalization";
  }
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  restart=%D, using %s\n",gmres->max_k,cstr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  happy breakdown tolerance %g\n",(double)gmres->haptol);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer,"%s restart %D",cstr,gmres->max_k);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   KSPGMRESMonitorKrylov - Calls VecView() for each new direction in the GMRES accumulated Krylov space.

   Collective on ksp

   Input Parameters:
+  ksp - the KSP context
.  its - iteration number
.  fgnorm - 2-norm of residual (or gradient)
-  dummy - an collection of viewers created with KSPViewerCreate()

   Options Database Keys:
.   -ksp_gmres_kyrlov_monitor

   Notes:
    A new PETSCVIEWERDRAW is created for each Krylov vector so they can all be simultaneously viewed
   Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorDefault(), VecView(), KSPViewersCreate(), KSPViewersDestroy()
@*/
PetscErrorCode  KSPGMRESMonitorKrylov(KSP ksp,PetscInt its,PetscReal fgnorm,void *dummy)
{
  PetscViewers   viewers = (PetscViewers)dummy;
  KSP_GMRES      *gmres  = (KSP_GMRES*)ksp->data;
  PetscErrorCode ierr;
  Vec            x;
  PetscViewer    viewer;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscViewersGetViewer(viewers,gmres->it+1,&viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscViewerSetType(viewer,PETSCVIEWERDRAW);CHKERRQ(ierr);
    ierr = PetscViewerDrawSetInfo(viewer,NULL,"Krylov GMRES Monitor",PETSC_DECIDE,PETSC_DECIDE,300,300);CHKERRQ(ierr);
  }
  x    = VEC_VV(gmres->it+1);
  ierr = VecView(x,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetFromOptions_GMRES(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       restart;
  PetscReal      haptol;
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP GMRES Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_gmres_restart","Number of Krylov search directions","KSPGMRESSetRestart",gmres->max_k,&restart,&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPGMRESSetRestart(ksp,restart);CHKERRQ(ierr); }
  ierr = PetscOptionsReal("-ksp_gmres_haptol","Tolerance for exact convergence (happy ending)","KSPGMRESSetHapTol",gmres->haptol,&haptol,&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPGMRESSetHapTol(ksp,haptol);CHKERRQ(ierr); }
  flg  = PETSC_FALSE;
  ierr = PetscOptionsBool("-ksp_gmres_preallocate","Preallocate Krylov vectors","KSPGMRESSetPreAllocateVectors",flg,&flg,NULL);CHKERRQ(ierr);
  if (flg) {ierr = KSPGMRESSetPreAllocateVectors(ksp);CHKERRQ(ierr);}
  ierr = PetscOptionsBoolGroupBegin("-ksp_gmres_classicalgramschmidt","Classical (unmodified) Gram-Schmidt (fast)","KSPGMRESSetOrthogonalization",&flg);CHKERRQ(ierr);
  if (flg) {ierr = KSPGMRESSetOrthogonalization(ksp,KSPGMRESClassicalGramSchmidtOrthogonalization);CHKERRQ(ierr);}
  ierr = PetscOptionsBoolGroupEnd("-ksp_gmres_modifiedgramschmidt","Modified Gram-Schmidt (slow,more stable)","KSPGMRESSetOrthogonalization",&flg);CHKERRQ(ierr);
  if (flg) {ierr = KSPGMRESSetOrthogonalization(ksp,KSPGMRESModifiedGramSchmidtOrthogonalization);CHKERRQ(ierr);}
  ierr = PetscOptionsEnum("-ksp_gmres_cgs_refinement_type","Type of iterative refinement for classical (unmodified) Gram-Schmidt","KSPGMRESSetCGSRefinementType",
                          KSPGMRESCGSRefinementTypes,(PetscEnum)gmres->cgstype,(PetscEnum*)&gmres->cgstype,&flg);CHKERRQ(ierr);
  flg  = PETSC_FALSE;
  ierr = PetscOptionsBool("-ksp_gmres_krylov_monitor","Plot the Krylov directions","KSPMonitorSet",flg,&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    PetscViewers viewers;
    ierr = PetscViewersCreate(PetscObjectComm((PetscObject)ksp),&viewers);CHKERRQ(ierr);
    ierr = KSPMonitorSet(ksp,KSPGMRESMonitorKrylov,viewers,(PetscErrorCode (*)(void**))PetscViewersDestroy);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPGMRESSetHapTol_GMRES(KSP ksp,PetscReal tol)
{
  KSP_GMRES *gmres = (KSP_GMRES*)ksp->data;

  PetscFunctionBegin;
  if (tol < 0.0) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Tolerance must be non-negative");
  gmres->haptol = tol;
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPGMRESGetRestart_GMRES(KSP ksp,PetscInt *max_k)
{
  KSP_GMRES *gmres = (KSP_GMRES*)ksp->data;

  PetscFunctionBegin;
  *max_k = gmres->max_k;
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPGMRESSetRestart_GMRES(KSP ksp,PetscInt max_k)
{
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (max_k < 1) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Restart must be positive");
  if (!ksp->setupstage) {
    gmres->max_k = max_k;
  } else if (gmres->max_k != max_k) {
    gmres->max_k    = max_k;
    ksp->setupstage = KSP_SETUP_NEW;
    /* free the data structures, then create them again */
    ierr = KSPReset_GMRES(ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPGMRESSetOrthogonalization_GMRES(KSP ksp,FCN fcn)
{
  PetscFunctionBegin;
  ((KSP_GMRES*)ksp->data)->orthog = fcn;
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPGMRESGetOrthogonalization_GMRES(KSP ksp,FCN *fcn)
{
  PetscFunctionBegin;
  *fcn = ((KSP_GMRES*)ksp->data)->orthog;
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPGMRESSetPreAllocateVectors_GMRES(KSP ksp)
{
  KSP_GMRES *gmres;

  PetscFunctionBegin;
  gmres = (KSP_GMRES*)ksp->data;
  gmres->q_preallocate = 1;
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPGMRESSetCGSRefinementType_GMRES(KSP ksp,KSPGMRESCGSRefinementType type)
{
  KSP_GMRES *gmres = (KSP_GMRES*)ksp->data;

  PetscFunctionBegin;
  gmres->cgstype = type;
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPGMRESGetCGSRefinementType_GMRES(KSP ksp,KSPGMRESCGSRefinementType *type)
{
  KSP_GMRES *gmres = (KSP_GMRES*)ksp->data;

  PetscFunctionBegin;
  *type = gmres->cgstype;
  PetscFunctionReturn(0);
}

/*@
   KSPGMRESSetCGSRefinementType - Sets the type of iterative refinement to use
         in the classical Gram Schmidt orthogonalization.

   Logically Collective on ksp

   Input Parameters:
+  ksp - the Krylov space context
-  type - the type of refinement

  Options Database:
.  -ksp_gmres_cgs_refinement_type <refine_never,refine_ifneeded,refine_always>

   Level: intermediate

.seealso: KSPGMRESSetOrthogonalization(), KSPGMRESCGSRefinementType, KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESGetCGSRefinementType(),
          KSPGMRESGetOrthogonalization()
@*/
PetscErrorCode  KSPGMRESSetCGSRefinementType(KSP ksp,KSPGMRESCGSRefinementType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ksp,type,2);
  ierr = PetscTryMethod(ksp,"KSPGMRESSetCGSRefinementType_C",(KSP,KSPGMRESCGSRefinementType),(ksp,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   KSPGMRESGetCGSRefinementType - Gets the type of iterative refinement to use
         in the classical Gram Schmidt orthogonalization.

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  type - the type of refinement

  Options Database:
.  -ksp_gmres_cgs_refinement_type <refine_never,refine_ifneeded,refine_always>

   Level: intermediate

.seealso: KSPGMRESSetOrthogonalization(), KSPGMRESCGSRefinementType, KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESSetCGSRefinementType(),
          KSPGMRESGetOrthogonalization()
@*/
PetscErrorCode  KSPGMRESGetCGSRefinementType(KSP ksp,KSPGMRESCGSRefinementType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscUseMethod(ksp,"KSPGMRESGetCGSRefinementType_C",(KSP,KSPGMRESCGSRefinementType*),(ksp,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@
   KSPGMRESSetRestart - Sets number of iterations at which GMRES, FGMRES and LGMRES restarts.

   Logically Collective on ksp

   Input Parameters:
+  ksp - the Krylov space context
-  restart - integer restart value

  Options Database:
.  -ksp_gmres_restart <positive integer>

    Note: The default value is 30.

   Level: intermediate

.seealso: KSPSetTolerances(), KSPGMRESSetOrthogonalization(), KSPGMRESSetPreAllocateVectors(), KSPGMRESGetRestart()
@*/
PetscErrorCode  KSPGMRESSetRestart(KSP ksp, PetscInt restart)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(ksp,restart,2);

  ierr = PetscTryMethod(ksp,"KSPGMRESSetRestart_C",(KSP,PetscInt),(ksp,restart));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   KSPGMRESGetRestart - Gets number of iterations at which GMRES, FGMRES and LGMRES restarts.

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.   restart - integer restart value

    Note: The default value is 30.

   Level: intermediate

.seealso: KSPSetTolerances(), KSPGMRESSetOrthogonalization(), KSPGMRESSetPreAllocateVectors(), KSPGMRESSetRestart()
@*/
PetscErrorCode  KSPGMRESGetRestart(KSP ksp, PetscInt *restart)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(ksp,"KSPGMRESGetRestart_C",(KSP,PetscInt*),(ksp,restart));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   KSPGMRESSetHapTol - Sets tolerance for determining happy breakdown in GMRES, FGMRES and LGMRES.

   Logically Collective on ksp

   Input Parameters:
+  ksp - the Krylov space context
-  tol - the tolerance

  Options Database:
.  -ksp_gmres_haptol <positive real value>

   Note: Happy breakdown is the rare case in GMRES where an 'exact' solution is obtained after
         a certain number of iterations. If you attempt more iterations after this point unstable
         things can happen hence very occasionally you may need to set this value to detect this condition

   Level: intermediate

.seealso: KSPSetTolerances()
@*/
PetscErrorCode  KSPGMRESSetHapTol(KSP ksp,PetscReal tol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveReal(ksp,tol,2);
  ierr = PetscTryMethod((ksp),"KSPGMRESSetHapTol_C",(KSP,PetscReal),((ksp),(tol)));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPGMRES - Implements the Generalized Minimal Residual method.
                (Saad and Schultz, 1986) with restart


   Options Database Keys:
+   -ksp_gmres_restart <restart> - the number of Krylov directions to orthogonalize against
.   -ksp_gmres_haptol <tol> - sets the tolerance for "happy ending" (exact convergence)
.   -ksp_gmres_preallocate - preallocate all the Krylov search directions initially (otherwise groups of
                             vectors are allocated as needed)
.   -ksp_gmres_classicalgramschmidt - use classical (unmodified) Gram-Schmidt to orthogonalize against the Krylov space (fast) (the default)
.   -ksp_gmres_modifiedgramschmidt - use modified Gram-Schmidt in the orthogonalization (more stable, but slower)
.   -ksp_gmres_cgs_refinement_type <refine_never,refine_ifneeded,refine_always> - determine if iterative refinement is used to increase the
                                   stability of the classical Gram-Schmidt  orthogonalization.
-   -ksp_gmres_krylov_monitor - plot the Krylov space generated

   Level: beginner

   Notes:
    Left and right preconditioning are supported, but not symmetric preconditioning.

   References:
.     1. - YOUCEF SAAD AND MARTIN H. SCHULTZ, GMRES: A GENERALIZED MINIMAL RESIDUAL ALGORITHM FOR SOLVING NONSYMMETRIC LINEAR SYSTEMS.
          SIAM J. ScI. STAT. COMPUT. Vo|. 7, No. 3, July 1986.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPFGMRES, KSPLGMRES,
           KSPGMRESSetRestart(), KSPGMRESSetHapTol(), KSPGMRESSetPreAllocateVectors(), KSPGMRESSetOrthogonalization(), KSPGMRESGetOrthogonalization(),
           KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESModifiedGramSchmidtOrthogonalization(),
           KSPGMRESCGSRefinementType, KSPGMRESSetCGSRefinementType(), KSPGMRESGetCGSRefinementType(), KSPGMRESMonitorKrylov(), KSPSetPCSide()

M*/

PETSC_EXTERN PetscErrorCode KSPCreate_GMRES(KSP ksp)
{
  KSP_GMRES      *gmres;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr      = PetscNewLog(ksp,&gmres);CHKERRQ(ierr);
  ksp->data = (void*)gmres;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,4);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_SYMMETRIC,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);

  ksp->ops->buildsolution                = KSPBuildSolution_GMRES;
  ksp->ops->setup                        = KSPSetUp_GMRES;
  ksp->ops->solve                        = KSPSolve_GMRES;
  ksp->ops->reset                        = KSPReset_GMRES;
  ksp->ops->destroy                      = KSPDestroy_GMRES;
  ksp->ops->view                         = KSPView_GMRES;
  ksp->ops->setfromoptions               = KSPSetFromOptions_GMRES;
  ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_GMRES;
  ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_GMRES;
#if !defined(PETSC_USE_COMPLEX) && !defined(PETSC_HAVE_ESSL)
  ksp->ops->computeritz                  = KSPComputeRitz_GMRES;
#endif
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetPreAllocateVectors_C",KSPGMRESSetPreAllocateVectors_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetOrthogonalization_C",KSPGMRESSetOrthogonalization_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESGetOrthogonalization_C",KSPGMRESGetOrthogonalization_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetRestart_C",KSPGMRESSetRestart_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESGetRestart_C",KSPGMRESGetRestart_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetHapTol_C",KSPGMRESSetHapTol_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetCGSRefinementType_C",KSPGMRESSetCGSRefinementType_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESGetCGSRefinementType_C",KSPGMRESGetCGSRefinementType_GMRES);CHKERRQ(ierr);

  gmres->haptol         = 1.0e-30;
  gmres->q_preallocate  = 0;
  gmres->delta_allocate = GMRES_DELTA_DIRECTIONS;
  gmres->orthog         = KSPGMRESClassicalGramSchmidtOrthogonalization;
  gmres->nrs            = 0;
  gmres->sol_temp       = 0;
  gmres->max_k          = GMRES_DEFAULT_MAXK;
  gmres->Rsvd           = 0;
  gmres->cgstype        = KSP_GMRES_CGS_REFINE_NEVER;
  gmres->orthogwork     = 0;
  PetscFunctionReturn(0);
}

