/*$Id: gmres.c,v 1.176 2001/08/07 03:03:51 balay Exp $*/

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

#include "src/ksp/ksp/impls/gmres/gmresp.h"       /*I  "petscksp.h"  I*/
#define GMRES_DELTA_DIRECTIONS 10
#define GMRES_DEFAULT_MAXK     30
static int    GMRESGetNewVectors(KSP,int);
static int    GMRESUpdateHessenberg(KSP,int,PetscTruth,PetscReal*);
static int    BuildGmresSoln(PetscScalar*,Vec,Vec,KSP,int);

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_GMRES"
int    KSPSetUp_GMRES(KSP ksp)
{
  unsigned  int size,hh,hes,rs,cc;
  int       ierr,max_k,k;
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(2,"no symmetric preconditioning for KSPGMRES");
  } 

  max_k         = gmres->max_k;
  hh            = (max_k + 2) * (max_k + 1);
  hes           = (max_k + 1) * (max_k + 1);
  rs            = (max_k + 2);
  cc            = (max_k + 1);
  size          = (hh + hes + rs + 2*cc) * sizeof(PetscScalar);

  ierr = PetscMalloc(size,&gmres->hh_origin);CHKERRQ(ierr);
  ierr = PetscMemzero(gmres->hh_origin,size);CHKERRQ(ierr);
  PetscLogObjectMemory(ksp,size);
  gmres->hes_origin = gmres->hh_origin + hh;
  gmres->rs_origin  = gmres->hes_origin + hes;
  gmres->cc_origin  = gmres->rs_origin + rs;
  gmres->ss_origin  = gmres->cc_origin + cc;

  if (ksp->calc_sings) {
    /* Allocate workspace to hold Hessenberg matrix needed by lapack */
    size = (max_k + 3)*(max_k + 9)*sizeof(PetscScalar);
    ierr = PetscMalloc(size,&gmres->Rsvd);CHKERRQ(ierr);
    ierr = PetscMalloc(5*(max_k+2)*sizeof(PetscReal),&gmres->Dsvd);CHKERRQ(ierr);
    PetscLogObjectMemory(ksp,size+5*(max_k+2)*sizeof(PetscReal));
  }

  /* Allocate array to hold pointers to user vectors.  Note that we need
   4 + max_k + 1 (since we need it+1 vectors, and it <= max_k) */
  ierr = PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void *),&gmres->vecs);CHKERRQ(ierr);
  gmres->vecs_allocated = VEC_OFFSET + 2 + max_k;
  ierr = PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void *),&gmres->user_work);CHKERRQ(ierr);
  ierr = PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(int),&gmres->mwork_alloc);CHKERRQ(ierr);
  PetscLogObjectMemory(ksp,(VEC_OFFSET+2+max_k)*(2*sizeof(void *)+sizeof(int)));

  if (gmres->q_preallocate) {
    gmres->vv_allocated   = VEC_OFFSET + 2 + max_k;
    ierr = VecDuplicateVecs(VEC_RHS,gmres->vv_allocated,&gmres->user_work[0]);CHKERRQ(ierr);
    PetscLogObjectParents(ksp,gmres->vv_allocated,gmres->user_work[0]);
    gmres->mwork_alloc[0] = gmres->vv_allocated;
    gmres->nwork_alloc    = 1;
    for (k=0; k<gmres->vv_allocated; k++) {
      gmres->vecs[k] = gmres->user_work[0][k];
    }
  } else {
    gmres->vv_allocated    = 5;
    ierr = VecDuplicateVecs(ksp->vec_rhs,5,&gmres->user_work[0]);CHKERRQ(ierr);
    PetscLogObjectParents(ksp,5,gmres->user_work[0]);
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
int GMREScycle(int *itcount,KSP ksp)
{
  KSP_GMRES    *gmres = (KSP_GMRES *)(ksp->data);
  PetscReal    res_norm,res,hapbnd,tt;
  int          ierr,it = 0, max_k = gmres->max_k;
  PetscTruth   hapend = PETSC_FALSE;

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
  if (!res) {
    if (itcount) *itcount = 0;
    ksp->reason = KSP_CONVERGED_ATOL;
    PetscLogInfo(ksp,"GMRESCycle: Converged due to zero residual norm on entry\n");
    PetscFunctionReturn(0);
  }

  ierr = (*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  while (!ksp->reason && it < max_k && ksp->its < ksp->max_it) {
    KSPLogResidualHistory(ksp,res);
    gmres->it = (it - 1);
    KSPMonitor(ksp,ksp->its,res); 
    if (gmres->vv_allocated <= it + VEC_OFFSET + 1) {
      ierr = GMRESGetNewVectors(ksp,it+1);CHKERRQ(ierr);
    }
    ierr = KSP_PCApplyBAorAB(ksp,ksp->B,ksp->pc_side,VEC_VV(it),VEC_VV(1+it),VEC_TEMP_MATOP);CHKERRQ(ierr);

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
      PetscLogInfo(ksp,"Detected happy breakdown, current hapbnd = %g tt = %g\n",hapbnd,tt);
      hapend = PETSC_TRUE;
    }
    ierr = GMRESUpdateHessenberg(ksp,it,hapend,&res);CHKERRQ(ierr);
    it++;
    gmres->it  = (it-1);  /* For converged */
    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = res;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);

    ierr = (*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);

    /* Catch error in happy breakdown and signal convergence and break from loop */
    if (hapend) {
      if (!ksp->reason) {
        SETERRQ1(0,"You reached the happy break down, but convergence was not indicated. Residual norm = %g",res);
      }
      break;
    }
  }

  /* Monitor if we know that we will not return for a restart */
  if (ksp->reason || ksp->its >= ksp->max_it) {
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
  ierr = BuildGmresSoln(GRS(0),VEC_SOLN,VEC_SOLN,ksp,it-1);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_GMRES"
int KSPSolve_GMRES(KSP ksp)
{
  int        ierr,its,itcount;
  KSP_GMRES  *gmres = (KSP_GMRES *)ksp->data;
  PetscTruth guess_zero = ksp->guess_zero;

  PetscFunctionBegin;
  if (ksp->calc_sings && !gmres->Rsvd) {
    SETERRQ(1,"Must call KSPSetComputeSingularValues() before KSPSetUp() is called");
  }

  ierr     = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its = 0;
  ierr     = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);

  itcount     = 0;
  ksp->reason = KSP_CONVERGED_ITERATING;
  while (!ksp->reason) {
    ierr     = KSPInitialResidual(ksp,VEC_SOLN,VEC_TEMP,VEC_TEMP_MATOP,VEC_VV(0),VEC_RHS);CHKERRQ(ierr);
    ierr     = GMREScycle(&its,ksp);CHKERRQ(ierr);
    itcount += its;  
    if (itcount >= ksp->max_it) {
      ksp->reason = KSP_DIVERGED_ITS;
      break;
    }
    ksp->guess_zero = PETSC_FALSE; /* every future call to KSPInitialResidual() will have nonzero guess */
  }
  ksp->guess_zero = guess_zero; /* restore if user provided nonzero initial guess */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_GMRES_Internal" 
int KSPDestroy_GMRES_Internal(KSP ksp)
{
  KSP_GMRES *gmres = (KSP_GMRES*)ksp->data;
  int       i,ierr;

  PetscFunctionBegin;
  /* Free the Hessenberg matrix */
  if (gmres->hh_origin) {ierr = PetscFree(gmres->hh_origin);CHKERRQ(ierr);}

  /* Free the pointer to user variables */
  if (gmres->vecs) {ierr = PetscFree(gmres->vecs);CHKERRQ(ierr);}

  /* free work vectors */
  for (i=0; i<gmres->nwork_alloc; i++) {
    ierr = VecDestroyVecs(gmres->user_work[i],gmres->mwork_alloc[i]);CHKERRQ(ierr);
  }
  if (gmres->user_work)  {ierr = PetscFree(gmres->user_work);CHKERRQ(ierr);}
  if (gmres->mwork_alloc) {ierr = PetscFree(gmres->mwork_alloc);CHKERRQ(ierr);}
  if (gmres->nrs) {ierr = PetscFree(gmres->nrs);CHKERRQ(ierr);}
  if (gmres->sol_temp) {ierr = VecDestroy(gmres->sol_temp);CHKERRQ(ierr);}
  if (gmres->Rsvd) {ierr = PetscFree(gmres->Rsvd);CHKERRQ(ierr);}
  if (gmres->Dsvd) {ierr = PetscFree(gmres->Dsvd);CHKERRQ(ierr);}

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_GMRES" 
int KSPDestroy_GMRES(KSP ksp)
{
  KSP_GMRES *gmres = (KSP_GMRES*)ksp->data;
  int       ierr;

  PetscFunctionBegin;
  ierr = KSPDestroy_GMRES_Internal(ksp);CHKERRQ(ierr);
  ierr = PetscFree(gmres);CHKERRQ(ierr);
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
static int BuildGmresSoln(PetscScalar* nrs,Vec vs,Vec vdest,KSP ksp,int it)
{
  PetscScalar tt,zero = 0.0,one = 1.0;
  int         ierr,ii,k,j;
  KSP_GMRES   *gmres = (KSP_GMRES *)(ksp->data);

  PetscFunctionBegin;
  /* Solve for solution vector that minimizes the residual */

  /* If it is < 0, no gmres steps have been performed */
  if (it < 0) {
    if (vdest != vs) {
      ierr = VecCopy(vs,vdest);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  if (*HH(it,it) == 0.0) SETERRQ2(1,"HH(it,it) is identically zero; it = %d GRS(it) = %g",it,PetscAbsScalar(*GRS(it)));
  if (*HH(it,it) != 0.0) {
    nrs[it] = *GRS(it) / *HH(it,it);
  } else {
    nrs[it] = 0.0;
  }
  for (ii=1; ii<=it; ii++) {
    k   = it - ii;
    tt  = *GRS(k);
    for (j=k+1; j<=it; j++) tt  = tt - *HH(k,j) * nrs[j];
    nrs[k]   = tt / *HH(k,k);
  }

  /* Accumulate the correction to the solution of the preconditioned problem in TEMP */
  ierr = VecSet(&zero,VEC_TEMP);CHKERRQ(ierr);
  ierr = VecMAXPY(it+1,nrs,VEC_TEMP,&VEC_VV(0));CHKERRQ(ierr);

  ierr = KSPUnwindPreconditioner(ksp,VEC_TEMP,VEC_TEMP_MATOP);CHKERRQ(ierr);
  /* add solution to previous solution */
  if (vdest != vs) {
    ierr = VecCopy(vs,vdest);CHKERRQ(ierr);
  }
  ierr = VecAXPY(&one,VEC_TEMP,vdest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
   Do the scalar work for the orthogonalization.  Return new residual.
 */
#undef __FUNCT__  
#define __FUNCT__ "GMRESUpdateHessenberg"
static int GMRESUpdateHessenberg(KSP ksp,int it,PetscTruth hapend,PetscReal *res)
{
  PetscScalar *hh,*cc,*ss,tt;
  int         j;
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
    if (tt == 0.0) {SETERRQ(PETSC_ERR_KSP_BRKDWN,"Your matrix or preconditioner is the null operator");}
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
static int GMRESGetNewVectors(KSP ksp,int it)
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;
  int       nwork = gmres->nwork_alloc,k,nalloc,ierr;

  PetscFunctionBegin;
  nalloc = gmres->delta_allocate;
  /* Adjust the number to allocate to make sure that we don't exceed the
    number of available slots */
  if (it + VEC_OFFSET + nalloc >= gmres->vecs_allocated){
    nalloc = gmres->vecs_allocated - it - VEC_OFFSET;
  }
  if (!nalloc) PetscFunctionReturn(0);

  gmres->vv_allocated += nalloc;
  ierr = VecDuplicateVecs(ksp->vec_rhs,nalloc,&gmres->user_work[nwork]);CHKERRQ(ierr);
  PetscLogObjectParents(ksp,nalloc,gmres->user_work[nwork]);CHKERRQ(ierr);
  gmres->mwork_alloc[nwork] = nalloc;
  for (k=0; k<nalloc; k++) {
    gmres->vecs[it+VEC_OFFSET+k] = gmres->user_work[nwork][k];
  }
  gmres->nwork_alloc++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPBuildSolution_GMRES"
int KSPBuildSolution_GMRES(KSP ksp,Vec  ptr,Vec *result)
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data; 
  int       ierr;

  PetscFunctionBegin;
  if (!ptr) {
    if (!gmres->sol_temp) {
      ierr = VecDuplicate(ksp->vec_sol,&gmres->sol_temp);CHKERRQ(ierr);
      PetscLogObjectParent(ksp,gmres->sol_temp);
    }
    ptr = gmres->sol_temp;
  }
  if (!gmres->nrs) {
    /* allocate the work area */
    ierr = PetscMalloc(gmres->max_k*sizeof(PetscScalar),&gmres->nrs);CHKERRQ(ierr);
    PetscLogObjectMemory(ksp,gmres->max_k*sizeof(PetscScalar));
  }

  ierr = BuildGmresSoln(gmres->nrs,VEC_SOLN,ptr,ksp,gmres->it);CHKERRQ(ierr);
  *result = ptr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPView_GMRES" 
int KSPView_GMRES(KSP ksp,PetscViewer viewer)
{
  KSP_GMRES  *gmres = (KSP_GMRES *)ksp->data; 
  const char *cstr;
  int        ierr;
  PetscTruth isascii,isstring;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (gmres->orthog == KSPGMRESClassicalGramSchmidtOrthogonalization) {
    if (gmres->cgstype == KSP_GMRES_CGS_REFINE_NEVER) {
      cstr = "Classical (unmodified) Gram-Schmidt Orthogonalization with no iterative refinement";
    } else if (gmres->cgstype == KSP_GMRES_CGS_REFINE_ALWAYS) {
      cstr = "Classical (unmodified) Gram-Schmidt Orthogonalization with one step of iterative refinement";
    } else {
      cstr = "Classical (unmodified) Gram-Schmidt Orthogonalization with one step of iterative refinement when needed";
    }
  } else if (gmres->orthog == KSPGMRESModifiedGramSchmidtOrthogonalization) {
    cstr = "Modified Gram-Schmidt Orthogonalization";
  } else {
    cstr = "unknown orthogonalization";
  }
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  GMRES: restart=%d, using %s\n",gmres->max_k,cstr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  GMRES: happy breakdown tolerance %g\n",gmres->haptol);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer,"%s restart %d",cstr,gmres->max_k);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported for KSP GMRES",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESKrylovMonitor"
/*@C
   KSPGMRESKrylovMonitor - Calls VecView() for each direction in the 
   GMRES accumulated Krylov space.

   Collective on KSP

   Input Parameters:
+  ksp - the KSP context
.  its - iteration number
.  fgnorm - 2-norm of residual (or gradient)
-  a viewers object created with PetscViewersCreate()

   Level: intermediate

.keywords: KSP, nonlinear, vector, monitor, view, Krylov space

.seealso: KSPSetMonitor(), KSPDefaultMonitor(), VecView(), PetscViewersCreate(), PetscViewersDestroy()
@*/
int KSPGMRESKrylovMonitor(KSP ksp,int its,PetscReal fgnorm,void *dummy)
{
  PetscViewers viewers = (PetscViewers)dummy;
  KSP_GMRES    *gmres = (KSP_GMRES*)ksp->data;
  int          ierr;
  Vec          x;
  PetscViewer  viewer;

  PetscFunctionBegin;
  ierr = PetscViewersGetViewer(viewers,gmres->it+1,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSC_VIEWER_DRAW);CHKERRQ(ierr);

  x      = VEC_VV(gmres->it+1);
  ierr   = VecView(x,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions_GMRES"
int KSPSetFromOptions_GMRES(KSP ksp)
{
  int             ierr,restart,indx;
  PetscReal       haptol;
  KSP_GMRES       *gmres = (KSP_GMRES*)ksp->data;
  PetscTruth      flg;
  const char      *types[] = {"never","ifneeded","always"};

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP GMRES Options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ksp_gmres_restart","Number of Krylov search directions","KSPGMRESSetRestart",gmres->max_k,&restart,&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPGMRESSetRestart(ksp,restart);CHKERRQ(ierr); }
    ierr = PetscOptionsReal("-ksp_gmres_haptol","Tolerance for exact convergence (happy ending)","KSPGMRESSetHapTol",gmres->haptol,&haptol,&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPGMRESSetHapTol(ksp,haptol);CHKERRQ(ierr); }
    ierr = PetscOptionsName("-ksp_gmres_preallocate","Preallocate Krylov vectors","KSPGMRESSetPreAllocateVectors",&flg);CHKERRQ(ierr);
    if (flg) {ierr = KSPGMRESSetPreAllocateVectors(ksp);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroupBegin("-ksp_gmres_classicalgramschmidt","Classical (unmodified) Gram-Schmidt (fast)","KSPGMRESSetOrthogonalization",&flg);CHKERRQ(ierr);
    if (flg) {ierr = KSPGMRESSetOrthogonalization(ksp,KSPGMRESClassicalGramSchmidtOrthogonalization);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroupEnd("-ksp_gmres_modifiedgramschmidt","Modified Gram-Schmidt (slow,more stable)","KSPGMRESSetOrthogonalization",&flg);CHKERRQ(ierr);
    if (flg) {ierr = KSPGMRESSetOrthogonalization(ksp,KSPGMRESModifiedGramSchmidtOrthogonalization);CHKERRQ(ierr);}
    ierr = PetscOptionsEList("-ksp_gmres_cgs_refinement_type","Type of iterative refinement for classical (unmodified) Gram-Schmidt","KSPGMRESSetCGSRefinementType()",types,3,types[(int)gmres->cgstype],&indx,&flg);CHKERRQ(ierr);    
    if (flg) {
      ierr = KSPGMRESSetCGSRefinementType(ksp,(KSPGMRESCGSRefinementType)indx);CHKERRQ(ierr);
    }

    ierr = PetscOptionsName("-ksp_gmres_krylov_monitor","Plot the Krylov directions","KSPSetMonitor",&flg);CHKERRQ(ierr);
    if (flg) {
      PetscViewers viewers;
      ierr = PetscViewersCreate(ksp->comm,&viewers);CHKERRQ(ierr);
      ierr = KSPSetMonitor(ksp,KSPGMRESKrylovMonitor,viewers,(int (*)(void*))PetscViewersDestroy);CHKERRQ(ierr);
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN int KSPComputeExtremeSingularValues_GMRES(KSP,PetscReal *,PetscReal *);
EXTERN int KSPComputeEigenvalues_GMRES(KSP,int,PetscReal *,PetscReal *,int *);


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESSetHapTol_GMRES" 
int KSPGMRESSetHapTol_GMRES(KSP ksp,double tol)
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;
  if (tol < 0.0) SETERRQ(1,"Tolerance must be non-negative");
  gmres->haptol = tol;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESSetRestart_GMRES" 
int KSPGMRESSetRestart_GMRES(KSP ksp,int max_k)
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;
  int       ierr;

  PetscFunctionBegin;
  if (max_k < 1) SETERRQ(1,"Restart must be positive");
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

typedef int (*FCN)(KSP,int); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESSetOrthogonalization_GMRES" 
int KSPGMRESSetOrthogonalization_GMRES(KSP ksp,FCN fcn)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ((KSP_GMRES *)ksp->data)->orthog = fcn;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESSetPreAllocateVectors_GMRES" 
int KSPGMRESSetPreAllocateVectors_GMRES(KSP ksp)
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
int KSPGMRESSetCGSRefinementType_GMRES(KSP ksp,KSPGMRESCGSRefinementType type)
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
int KSPGMRESSetCGSRefinementType(KSP ksp,KSPGMRESCGSRefinementType type)
{
  int ierr,(*f)(KSP,KSPGMRESCGSRefinementType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPGMRESSetCGSRefinementType_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_GMRES"
int KSPCreate_GMRES(KSP ksp)
{
  KSP_GMRES *gmres;
  int       ierr;

  PetscFunctionBegin;
  ierr = PetscNew(KSP_GMRES,&gmres);CHKERRQ(ierr);
  ierr  = PetscMemzero(gmres,sizeof(KSP_GMRES));CHKERRQ(ierr);
  PetscLogObjectMemory(ksp,sizeof(KSP_GMRES));
  ksp->data                              = (void*)gmres;
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
  PetscFunctionReturn(0);
}
EXTERN_C_END

