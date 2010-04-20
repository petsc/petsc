#define PETSCKSP_DLL

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

#include "../src/ksp/ksp/impls/gmres/gmresimpl.h"       /*I  "petscksp.h"  I*/
#define GMRES_DELTA_DIRECTIONS 10
#define GMRES_DEFAULT_MAXK     30
static PetscErrorCode    GMRESGetNewVectors(KSP,PetscInt);
static PetscErrorCode    GMRESUpdateHessenberg(KSP,PetscInt,PetscTruth,PetscReal*);
static PetscErrorCode    BuildGmresSoln(PetscScalar*,Vec,Vec,KSP,PetscInt);

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_GMRES"
PetscErrorCode    KSPSetUp_GMRES(KSP ksp)
{
  PetscInt       hh,hes,rs,cc;
  PetscErrorCode ierr;
  PetscInt       max_k,k;
  KSP_GMRES      *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(PETSC_ERR_SUP,"no symmetric preconditioning for KSPGMRES");
  } 

  max_k         = gmres->max_k;  /* restart size */
  hh            = (max_k + 2) * (max_k + 1);
  hes           = (max_k + 1) * (max_k + 1);
  rs            = (max_k + 2);
  cc            = (max_k + 1);

  ierr = PetscMalloc5(hh,PetscScalar,&gmres->hh_origin,hes,PetscScalar,&gmres->hes_origin,rs,PetscScalar,&gmres->rs_origin,cc,PetscScalar,&gmres->cc_origin,cc,PetscScalar,& gmres->ss_origin);CHKERRQ(ierr);
  ierr = PetscMemzero(gmres->hh_origin,hh*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(gmres->hes_origin,hes*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(gmres->rs_origin,rs*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(gmres->cc_origin,cc*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(gmres->ss_origin,cc*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp,(hh + hes + rs + 2*cc)*sizeof(PetscScalar));CHKERRQ(ierr);

  if (ksp->calc_sings) {
    /* Allocate workspace to hold Hessenberg matrix needed by lapack */
    ierr = PetscMalloc((max_k + 3)*(max_k + 9)*sizeof(PetscScalar),&gmres->Rsvd);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(ksp,(max_k + 3)*(max_k + 9)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMalloc(5*(max_k+2)*sizeof(PetscReal),&gmres->Dsvd);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(ksp,5*(max_k+2)*sizeof(PetscReal));CHKERRQ(ierr);
  }

  /* Allocate array to hold pointers to user vectors.  Note that we need
   4 + max_k + 1 (since we need it+1 vectors, and it <= max_k) */
  ierr = PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void*),&gmres->vecs);CHKERRQ(ierr);
  gmres->vecs_allocated = VEC_OFFSET + 2 + max_k;
  ierr = PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void*),&gmres->user_work);CHKERRQ(ierr);
  ierr = PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(PetscInt),&gmres->mwork_alloc);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp,(VEC_OFFSET+2+max_k)*(2*sizeof(void*)+sizeof(PetscInt)));CHKERRQ(ierr);

  if (gmres->q_preallocate) {
    gmres->vv_allocated   = VEC_OFFSET + 2 + max_k;
    ierr = KSPGetVecs(ksp,gmres->vv_allocated,&gmres->user_work[0],0,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(ksp,gmres->vv_allocated,gmres->user_work[0]);CHKERRQ(ierr);
    gmres->mwork_alloc[0] = gmres->vv_allocated;
    gmres->nwork_alloc    = 1;
    for (k=0; k<gmres->vv_allocated; k++) {
      gmres->vecs[k] = gmres->user_work[0][k];
    }
  } else {
    gmres->vv_allocated    = 5;
    ierr = KSPGetVecs(ksp,5,&gmres->user_work[0],0,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(ksp,5,gmres->user_work[0]);CHKERRQ(ierr);
    gmres->mwork_alloc[0]  = 5;
    gmres->nwork_alloc     = 1;
    for (k=0; k<gmres->vv_allocated; k++) {
      gmres->vecs[k] = gmres->user_work[0][k];
    }
  }
  PetscFunctionReturn(0);
}

/*
    Run gmres, possibly with restart.  Return residual history if requested.
    input parameters:

.	gmres  - structure containing parameters and work areas

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
#undef __FUNCT__  
#define __FUNCT__ "GMREScycle"
PetscErrorCode GMREScycle(PetscInt *itcount,KSP ksp)
{
  KSP_GMRES      *gmres = (KSP_GMRES *)(ksp->data);
  PetscReal      res_norm,res,hapbnd,tt;
  PetscErrorCode ierr;
  PetscInt       it = 0, max_k = gmres->max_k;
  PetscTruth     hapend = PETSC_FALSE;

  PetscFunctionBegin;
  ierr    = VecNormalize(VEC_VV(0),&res_norm);CHKERRQ(ierr);
  res     = res_norm;
  *GRS(0) = res_norm;

  /* check for the convergence */
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->rnorm = res;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
  gmres->it = (it - 1);
  KSPLogResidualHistory(ksp,res);
  KSPMonitor(ksp,ksp->its,res); 
  if (!res) {
    if (itcount) *itcount = 0;
    ksp->reason = KSP_CONVERGED_ATOL;
    ierr = PetscInfo(ksp,"Converged due to zero residual norm on entry\n");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = (*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  while (!ksp->reason && it < max_k && ksp->its < ksp->max_it) {
    if (it) {
      KSPLogResidualHistory(ksp,res);
      KSPMonitor(ksp,ksp->its,res); 
    }
    gmres->it = (it - 1);
    if (gmres->vv_allocated <= it + VEC_OFFSET + 1) {
      ierr = GMRESGetNewVectors(ksp,it+1);CHKERRQ(ierr);
    }
    ierr = KSP_PCApplyBAorAB(ksp,VEC_VV(it),VEC_VV(1+it),VEC_TEMP_MATOP);CHKERRQ(ierr);

    /* update hessenberg matrix and do Gram-Schmidt */
    ierr = (*gmres->orthog)(ksp,it);CHKERRQ(ierr);

    /* vv(i+1) . vv(i+1) */
    ierr = VecNormalize(VEC_VV(it+1),&tt);CHKERRQ(ierr);
    /* save the magnitude */
    *HH(it+1,it)    = tt;
    *HES(it+1,it)   = tt;

    /* check for the happy breakdown */
    hapbnd  = PetscAbsScalar(tt / *GRS(it));
    if (hapbnd > gmres->haptol) hapbnd = gmres->haptol;
    if (tt < hapbnd) {
      ierr = PetscInfo2(ksp,"Detected happy breakdown, current hapbnd = %G tt = %G\n",hapbnd,tt);CHKERRQ(ierr);
      hapend = PETSC_TRUE;
    }
    ierr = GMRESUpdateHessenberg(ksp,it,hapend,&res);CHKERRQ(ierr);

    it++;
    gmres->it  = (it-1);  /* For converged */
    ksp->its++;
    ksp->rnorm = res;
    if (ksp->reason) break;

    ierr = (*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);

    /* Catch error in happy breakdown and signal convergence and break from loop */
    if (hapend) {
      if (!ksp->reason) {
        SETERRQ1(0,"You reached the happy break down, but convergence was not indicated. Residual norm = %G",res);
      }
      break;
    }
  }

  /* Monitor if we know that we will not return for a restart */
  if (it && (ksp->reason || ksp->its >= ksp->max_it)) {
    KSPLogResidualHistory(ksp,res);
    KSPMonitor(ksp,ksp->its,res);
  }

  if (itcount) *itcount    = it;


  /*
    Down here we have to solve for the "best" coefficients of the Krylov
    columns, add the solution values together, and possibly unwind the
    preconditioning from the solution
   */
  /* Form the solution (or the solution so far) */
  ierr = BuildGmresSoln(GRS(0),ksp->vec_sol,ksp->vec_sol,ksp,it-1);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_GMRES"
PetscErrorCode KSPSolve_GMRES(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       its,itcount;
  KSP_GMRES      *gmres = (KSP_GMRES *)ksp->data;
  PetscTruth     guess_zero = ksp->guess_zero;

  PetscFunctionBegin;
  if (ksp->calc_sings && !gmres->Rsvd) {
    SETERRQ(PETSC_ERR_ORDER,"Must call KSPSetComputeSingularValues() before KSPSetUp() is called");
  }
  if (ksp->normtype != KSP_NORM_PRECONDITIONED && ksp->pc_side != PC_RIGHT) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Use right preconditioning -ksp_right_pc if want unpreconditioned norm)");

  ierr     = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its = 0;
  ierr     = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);

  itcount     = 0;
  ksp->reason = KSP_CONVERGED_ITERATING;
  while (!ksp->reason) {
    ierr     = KSPInitialResidual(ksp,ksp->vec_sol,VEC_TEMP,VEC_TEMP_MATOP,VEC_VV(0),ksp->vec_rhs);CHKERRQ(ierr);
    ierr     = GMREScycle(&its,ksp);CHKERRQ(ierr);
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

#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_GMRES_Internal" 
PetscErrorCode KSPDestroy_GMRES_Internal(KSP ksp)
{
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  /* Free the Hessenberg matrices */
  ierr = PetscFree5(gmres->hh_origin,gmres->hes_origin,gmres->rs_origin,gmres->cc_origin,gmres->ss_origin);CHKERRQ(ierr);

  /* Free the pointer to user variables */
  ierr = PetscFree(gmres->vecs);CHKERRQ(ierr);

  /* free work vectors */
  for (i=0; i<gmres->nwork_alloc; i++) {
    ierr = VecDestroyVecs(gmres->user_work[i],gmres->mwork_alloc[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(gmres->user_work);CHKERRQ(ierr);
  ierr = PetscFree(gmres->mwork_alloc);CHKERRQ(ierr);
  ierr = PetscFree(gmres->nrs);CHKERRQ(ierr);
  if (gmres->sol_temp) {
    ierr = VecDestroy(gmres->sol_temp);CHKERRQ(ierr);
  }
  ierr = PetscFree(gmres->Rsvd);CHKERRQ(ierr);
  ierr = PetscFree(gmres->Dsvd);CHKERRQ(ierr);
  ierr = PetscFree(gmres->orthogwork);CHKERRQ(ierr);
  gmres->sol_temp       = 0;
  gmres->vv_allocated   = 0;
  gmres->vecs_allocated = 0;
  gmres->sol_temp       = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_GMRES" 
PetscErrorCode KSPDestroy_GMRES(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDestroy_GMRES_Internal(ksp);CHKERRQ(ierr);
  ierr = PetscFree(ksp->data);CHKERRQ(ierr);
  /* clear composed functions */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetPreAllocateVectors_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetOrthogonalization_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetRestart_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetHapTol_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetCGSRefinementType_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
    BuildGmresSoln - create the solution from the starting vector and the
    current iterates.

    Input parameters:
        nrs - work area of size it + 1.
	vs  - index of initial guess
	vdest - index of result.  Note that vs may == vdest (replace
	        guess with the solution).

     This is an internal routine that knows about the GMRES internals.
 */
#undef __FUNCT__  
#define __FUNCT__ "BuildGmresSoln"
static PetscErrorCode BuildGmresSoln(PetscScalar* nrs,Vec vs,Vec vdest,KSP ksp,PetscInt it)
{
  PetscScalar    tt;
  PetscErrorCode ierr;
  PetscInt       ii,k,j;
  KSP_GMRES      *gmres = (KSP_GMRES *)(ksp->data);

  PetscFunctionBegin;
  /* Solve for solution vector that minimizes the residual */

  /* If it is < 0, no gmres steps have been performed */
  if (it < 0) {
    ierr = VecCopy(vs,vdest);CHKERRQ(ierr); /* VecCopy() is smart, exists immediately if vguess == vdest */
    PetscFunctionReturn(0);
  }
  if (*HH(it,it) == 0.0) SETERRQ2(PETSC_ERR_CONV_FAILED,"HH(it,it) is identically zero; it = %D GRS(it) = %G",it,PetscAbsScalar(*GRS(it)));
  if (*HH(it,it) != 0.0) {
    nrs[it] = *GRS(it) / *HH(it,it);
  } else {
    nrs[it] = 0.0;
  }
  for (ii=1; ii<=it; ii++) {
    k   = it - ii;
    tt  = *GRS(k);
    for (j=k+1; j<=it; j++) tt  = tt - *HH(k,j) * nrs[j];
    if (*HH(k,k) == 0.0) SETERRQ2(PETSC_ERR_CONV_FAILED,"HH(k,k) is identically zero; it = %D k = %D",it,k);
    nrs[k]   = tt / *HH(k,k);
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
#undef __FUNCT__  
#define __FUNCT__ "GMRESUpdateHessenberg"
static PetscErrorCode GMRESUpdateHessenberg(KSP ksp,PetscInt it,PetscTruth hapend,PetscReal *res)
{
  PetscScalar *hh,*cc,*ss,tt;
  PetscInt    j;
  KSP_GMRES   *gmres = (KSP_GMRES *)(ksp->data);

  PetscFunctionBegin;
  hh  = HH(0,it);
  cc  = CC(0);
  ss  = SS(0);

  /* Apply all the previously computed plane rotations to the new column
     of the Hessenberg matrix */
  for (j=1; j<=it; j++) {
    tt  = *hh;
#if defined(PETSC_USE_COMPLEX)
    *hh = PetscConj(*cc) * tt + *ss * *(hh+1);
#else
    *hh = *cc * tt + *ss * *(hh+1);
#endif
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
#if defined(PETSC_USE_COMPLEX)
    tt        = PetscSqrtScalar(PetscConj(*hh) * *hh + PetscConj(*(hh+1)) * *(hh+1));
#else
    tt        = PetscSqrtScalar(*hh * *hh + *(hh+1) * *(hh+1));
#endif
    if (tt == 0.0) {
      ksp->reason = KSP_DIVERGED_NULL;
      PetscFunctionReturn(0);
    }
    *cc       = *hh / tt;
    *ss       = *(hh+1) / tt;
    *GRS(it+1) = - (*ss * *GRS(it));
#if defined(PETSC_USE_COMPLEX)
    *GRS(it)   = PetscConj(*cc) * *GRS(it);
    *hh       = PetscConj(*cc) * *hh + *ss * *(hh+1);
#else
    *GRS(it)   = *cc * *GRS(it);
    *hh       = *cc * *hh + *ss * *(hh+1);
#endif
    *res      = PetscAbsScalar(*GRS(it+1));
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
#undef __FUNCT__  
#define __FUNCT__ "GMRESGetNewVectors" 
static PetscErrorCode GMRESGetNewVectors(KSP ksp,PetscInt it)
{
  KSP_GMRES      *gmres = (KSP_GMRES *)ksp->data;
  PetscErrorCode ierr;
  PetscInt       nwork = gmres->nwork_alloc,k,nalloc;

  PetscFunctionBegin;
  nalloc = PetscMin(ksp->max_it,gmres->delta_allocate);
  /* Adjust the number to allocate to make sure that we don't exceed the
    number of available slots */
  if (it + VEC_OFFSET + nalloc >= gmres->vecs_allocated){
    nalloc = gmres->vecs_allocated - it - VEC_OFFSET;
  }
  if (!nalloc) PetscFunctionReturn(0);

  gmres->vv_allocated += nalloc;
  ierr = KSPGetVecs(ksp,nalloc,&gmres->user_work[nwork],0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(ksp,nalloc,gmres->user_work[nwork]);CHKERRQ(ierr);
  gmres->mwork_alloc[nwork] = nalloc;
  for (k=0; k<nalloc; k++) {
    gmres->vecs[it+VEC_OFFSET+k] = gmres->user_work[nwork][k];
  }
  gmres->nwork_alloc++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPBuildSolution_GMRES"
PetscErrorCode KSPBuildSolution_GMRES(KSP ksp,Vec  ptr,Vec *result)
{
  KSP_GMRES      *gmres = (KSP_GMRES *)ksp->data; 
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ptr) {
    if (!gmres->sol_temp) {
      ierr = VecDuplicate(ksp->vec_sol,&gmres->sol_temp);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(ksp,gmres->sol_temp);CHKERRQ(ierr);
    }
    ptr = gmres->sol_temp;
  }
  if (!gmres->nrs) {
    /* allocate the work area */
    ierr = PetscMalloc(gmres->max_k*sizeof(PetscScalar),&gmres->nrs);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(ksp,gmres->max_k*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  ierr = BuildGmresSoln(gmres->nrs,ksp->vec_sol,ptr,ksp,gmres->it);CHKERRQ(ierr);
  if (result) *result = ptr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPView_GMRES" 
PetscErrorCode KSPView_GMRES(KSP ksp,PetscViewer viewer)
{
  KSP_GMRES      *gmres = (KSP_GMRES *)ksp->data; 
  const char     *cstr;
  PetscErrorCode ierr;
  PetscTruth     iascii,isstring;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
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
	SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Unknown orthogonalization");
    }
  } else if (gmres->orthog == KSPGMRESModifiedGramSchmidtOrthogonalization) {
    cstr = "Modified Gram-Schmidt Orthogonalization";
  } else {
    cstr = "unknown orthogonalization";
  }
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  GMRES: restart=%D, using %s\n",gmres->max_k,cstr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  GMRES: happy breakdown tolerance %G\n",gmres->haptol);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer,"%s restart %D",cstr,gmres->max_k);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for KSP GMRES",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESMonitorKrylov"
/*@C
   KSPGMRESMonitorKrylov - Calls VecView() for each direction in the 
   GMRES accumulated Krylov space.

   Collective on KSP

   Input Parameters:
+  ksp - the KSP context
.  its - iteration number
.  fgnorm - 2-norm of residual (or gradient)
-  a viewers object created with PetscViewersCreate()

   Level: intermediate

.keywords: KSP, nonlinear, vector, monitor, view, Krylov space

.seealso: KSPMonitorSet(), KSPMonitorDefault(), VecView(), PetscViewersCreate(), PetscViewersDestroy()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESMonitorKrylov(KSP ksp,PetscInt its,PetscReal fgnorm,void *dummy)
{
  PetscViewers   viewers = (PetscViewers)dummy;
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscErrorCode ierr;
  Vec            x;
  PetscViewer    viewer;

  PetscFunctionBegin;
  ierr = PetscViewersGetViewer(viewers,gmres->it+1,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSC_VIEWER_DRAW);CHKERRQ(ierr);

  x      = VEC_VV(gmres->it+1);
  ierr   = VecView(x,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions_GMRES"
PetscErrorCode KSPSetFromOptions_GMRES(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       restart;
  PetscReal      haptol;
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscTruth     flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP GMRES Options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ksp_gmres_restart","Number of Krylov search directions","KSPGMRESSetRestart",gmres->max_k,&restart,&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPGMRESSetRestart(ksp,restart);CHKERRQ(ierr); }
    ierr = PetscOptionsReal("-ksp_gmres_haptol","Tolerance for exact convergence (happy ending)","KSPGMRESSetHapTol",gmres->haptol,&haptol,&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPGMRESSetHapTol(ksp,haptol);CHKERRQ(ierr); }
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-ksp_gmres_preallocate","Preallocate Krylov vectors","KSPGMRESSetPreAllocateVectors",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {ierr = KSPGMRESSetPreAllocateVectors(ksp);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroupBegin("-ksp_gmres_classicalgramschmidt","Classical (unmodified) Gram-Schmidt (fast)","KSPGMRESSetOrthogonalization",&flg);CHKERRQ(ierr);
    if (flg) {ierr = KSPGMRESSetOrthogonalization(ksp,KSPGMRESClassicalGramSchmidtOrthogonalization);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroupEnd("-ksp_gmres_modifiedgramschmidt","Modified Gram-Schmidt (slow,more stable)","KSPGMRESSetOrthogonalization",&flg);CHKERRQ(ierr);
    if (flg) {ierr = KSPGMRESSetOrthogonalization(ksp,KSPGMRESModifiedGramSchmidtOrthogonalization);CHKERRQ(ierr);}
    ierr = PetscOptionsEnum("-ksp_gmres_cgs_refinement_type","Type of iterative refinement for classical (unmodified) Gram-Schmidt","KSPGMRESSetCGSRefinementType",
                            KSPGMRESCGSRefinementTypes,(PetscEnum)gmres->cgstype,(PetscEnum*)&gmres->cgstype,&flg);CHKERRQ(ierr);   
    flg  = PETSC_FALSE; 
    ierr = PetscOptionsTruth("-ksp_gmres_krylov_monitor","Plot the Krylov directions","KSPMonitorSet",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      PetscViewers viewers;
      ierr = PetscViewersCreate(((PetscObject)ksp)->comm,&viewers);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPGMRESMonitorKrylov,viewers,(PetscErrorCode (*)(void*))PetscViewersDestroy);CHKERRQ(ierr);
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode KSPComputeExtremeSingularValues_GMRES(KSP,PetscReal *,PetscReal *);
EXTERN PetscErrorCode KSPComputeEigenvalues_GMRES(KSP,PetscInt,PetscReal *,PetscReal *,PetscInt *);


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESSetHapTol_GMRES" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetHapTol_GMRES(KSP ksp,PetscReal tol)
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;
  if (tol < 0.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Tolerance must be non-negative");
  gmres->haptol = tol;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESSetRestart_GMRES" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetRestart_GMRES(KSP ksp,PetscInt max_k)
{
  KSP_GMRES      *gmres = (KSP_GMRES *)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (max_k < 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Restart must be positive");
  if (!ksp->setupcalled) {
    gmres->max_k = max_k;
  } else if (gmres->max_k != max_k) {
     gmres->max_k = max_k;
     ksp->setupcalled = 0;
     /* free the data structures, then create them again */
     ierr = KSPDestroy_GMRES_Internal(ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

typedef PetscErrorCode (*FCN)(KSP,PetscInt); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESSetOrthogonalization_GMRES" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetOrthogonalization_GMRES(KSP ksp,FCN fcn)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ((KSP_GMRES *)ksp->data)->orthog = fcn;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESSetPreAllocateVectors_GMRES" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetPreAllocateVectors_GMRES(KSP ksp)
{
  KSP_GMRES *gmres;

  PetscFunctionBegin;
  gmres = (KSP_GMRES *)ksp->data;
  gmres->q_preallocate = 1;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESSetCGSRefinementType_GMRES"
PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetCGSRefinementType_GMRES(KSP ksp,KSPGMRESCGSRefinementType type)
{
  KSP_GMRES *gmres = (KSP_GMRES*)ksp->data;

  PetscFunctionBegin;
  gmres->cgstype = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESSetCGSRefinementType"
/*@
   KSPGMRESSetCGSRefinementType - Sets the type of iterative refinement to use
         in the classical Gram Schmidt orthogonalization.
   of the preconditioned problem.

   Collective on KSP

   Input Parameters:
+  ksp - the Krylov space context
-  type - the type of refinement

  Options Database:
.  -ksp_gmres_cgs_refinement_type <never,ifneeded,always>

   Level: intermediate

.keywords: KSP, GMRES, iterative refinement

.seealso: KSPGMRESSetOrthogonalization(), KSPGMRESCGSRefinementType, KSPGMRESClassicalGramSchmidtOrthogonalization()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetCGSRefinementType(KSP ksp,KSPGMRESCGSRefinementType type)
{
  PetscErrorCode ierr,(*f)(KSP,KSPGMRESCGSRefinementType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPGMRESSetCGSRefinementType_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGMRESSetRestart"
/*@
   KSPGMRESSetRestart - Sets number of iterations at which GMRES, FGMRES and LGMRES restarts.

   Collective on KSP

   Input Parameters:
+  ksp - the Krylov space context
-  restart - integer restart value

  Options Database:
.  -ksp_gmres_restart <positive integer>

    Note: The default value is 30.

   Level: intermediate

.keywords: KSP, GMRES, restart, iterations

.seealso: KSPSetTolerances(), KSPGMRESSetOrthogonalization(), KSPGMRESSetPreAllocateVectors()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetRestart(KSP ksp, PetscInt restart) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(ksp,"KSPGMRESSetRestart_C",(KSP,PetscInt),(ksp,restart));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGMRESSetHapTol"
/*@
   KSPGMRESSetHapTol - Sets tolerance for determining happy breakdown in GMRES, FGMRES and LGMRES.

   Collective on KSP

   Input Parameters:
+  ksp - the Krylov space context
-  tol - the tolerance

  Options Database:
.  -ksp_gmres_haptol <positive real value>

   Note: Happy breakdown is the rare case in GMRES where an 'exact' solution is obtained after
         a certain number of iterations. If you attempt more iterations after this point unstable 
         things can happen hence very occasionally you may need to set this value to detect this condition

   Level: intermediate

.keywords: KSP, GMRES, tolerance

.seealso: KSPSetTolerances()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetHapTol(KSP ksp,PetscReal tol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
.   -ksp_gmres_cgs_refinement_type <never,ifneeded,always> - determine if iterative refinement is used to increase the 
                                   stability of the classical Gram-Schmidt  orthogonalization.
-   -ksp_gmres_krylov_monitor - plot the Krylov space generated

   Level: beginner

   Notes: Left and right preconditioning are supported, but not symmetric preconditioning.

   References:
     GMRES: A GENERALIZED MINIMAL RESIDUAL ALGORITHM FOR SOLVING NONSYMMETRIC LINEAR SYSTEMS. YOUCEF SAAD AND MARTIN H. SCHULTZ,
          SIAM J. ScI. STAT. COMPUT. Vo|. 7, No. 3, July 1986, pp. 856--869.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPFGMRES, KSPLGMRES,
           KSPGMRESSetRestart(), KSPGMRESSetHapTol(), KSPGMRESSetPreAllocateVectors(), KSPGMRESSetOrthogonalization()
           KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESModifiedGramSchmidtOrthogonalization(),
           KSPGMRESCGSRefinementType, KSPGMRESSetCGSRefinementType(), KSPGMRESMonitorKrylov(), KSPSetPreconditionerSide()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_GMRES"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_GMRES(KSP ksp)
{
  KSP_GMRES      *gmres;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_GMRES,&gmres);CHKERRQ(ierr);
  ksp->data                              = (void*)gmres;


  ksp->normtype                          = KSP_NORM_PRECONDITIONED;
  ksp->ops->buildsolution                = KSPBuildSolution_GMRES;
  ksp->ops->setup                        = KSPSetUp_GMRES;
  ksp->ops->solve                        = KSPSolve_GMRES;
  ksp->ops->destroy                      = KSPDestroy_GMRES;
  ksp->ops->view                         = KSPView_GMRES;
  ksp->ops->setfromoptions               = KSPSetFromOptions_GMRES;
  ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_GMRES;
  ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_GMRES;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetPreAllocateVectors_C",
                                    "KSPGMRESSetPreAllocateVectors_GMRES",
                                     KSPGMRESSetPreAllocateVectors_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetOrthogonalization_C",
                                    "KSPGMRESSetOrthogonalization_GMRES",
                                     KSPGMRESSetOrthogonalization_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetRestart_C",
                                    "KSPGMRESSetRestart_GMRES",
                                     KSPGMRESSetRestart_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetHapTol_C",
                                    "KSPGMRESSetHapTol_GMRES",
                                     KSPGMRESSetHapTol_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetCGSRefinementType_C",
                                    "KSPGMRESSetCGSRefinementType_GMRES",
                                     KSPGMRESSetCGSRefinementType_GMRES);CHKERRQ(ierr);

  gmres->haptol              = 1.0e-30;
  gmres->q_preallocate       = 0;
  gmres->delta_allocate      = GMRES_DELTA_DIRECTIONS;
  gmres->orthog              = KSPGMRESClassicalGramSchmidtOrthogonalization;
  gmres->nrs                 = 0;
  gmres->sol_temp            = 0;
  gmres->max_k               = GMRES_DEFAULT_MAXK;
  gmres->Rsvd                = 0;
  gmres->cgstype             = KSP_GMRES_CGS_REFINE_NEVER;
  gmres->orthogwork          = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

