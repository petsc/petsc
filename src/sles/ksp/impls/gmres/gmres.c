/*$Id: gmres.c,v 1.145 2000/05/05 22:17:37 balay Exp bsmith $*/

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
    Note that we can elliminate an extra application of B^-1 between
    restarts as long as we don't require that the solution at the end
    of a unsuccessful gmres iteration always be the solution x.
 */

#include "src/sles/ksp/impls/gmres/gmresp.h"       /*I  "petscksp.h"  I*/
#define GMRES_DELTA_DIRECTIONS 10
#define GMRES_DEFAULT_MAXK     30
static int    GMRESGetNewVectors(KSP,int);
static int    GMRESUpdateHessenberg(KSP,int,PetscTruth,PetscReal*);
static int    BuildGmresSoln(Scalar*,Vec,Vec,KSP,int);

#undef __FUNC__
#define __FUNC__ /*<a name="KSPSetUp_GMRES"></a>*/"KSPSetUp_GMRES"
int    KSPSetUp_GMRES(KSP ksp)
{
  unsigned  int size,hh,hes,rs,cc;
  int       ierr,max_k,k;
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(2,0,"no symmetric preconditioning for KSPGMRES");
  }
  max_k         = gmres->max_k;
  hh            = (max_k + 2) * (max_k + 1);
  hes           = (max_k + 1) * (max_k + 1);
  rs            = (max_k + 2);
  cc            = (max_k + 1);
  size          = (hh + hes + rs + 2*cc) * sizeof(Scalar);

  gmres->hh_origin  = (Scalar*)PetscMalloc(size);CHKPTRQ(gmres->hh_origin);
  PLogObjectMemory(ksp,size);
  gmres->hes_origin = gmres->hh_origin + hh;
  gmres->rs_origin  = gmres->hes_origin + hes;
  gmres->cc_origin  = gmres->rs_origin + rs;
  gmres->ss_origin  = gmres->cc_origin + cc;

  if (ksp->calc_sings) {
    /* Allocate workspace to hold Hessenberg matrix needed by Eispack */
    size = (max_k + 3)*(max_k + 9)*sizeof(Scalar);
    gmres->Rsvd = (Scalar*)PetscMalloc(size);CHKPTRQ(gmres->Rsvd);
    gmres->Dsvd = (PetscReal*)PetscMalloc(5*(max_k+2)*sizeof(PetscReal));CHKPTRQ(gmres->Dsvd);
    PLogObjectMemory(ksp,size+5*(max_k+2)*sizeof(PetscReal));
  }

  /* Allocate array to hold pointers to user vectors.  Note that we need
   4 + max_k + 1 (since we need it+1 vectors, and it <= max_k) */
  gmres->vecs = (Vec*)PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void *));CHKPTRQ(gmres->vecs);
  gmres->vecs_allocated = VEC_OFFSET + 2 + max_k;
  gmres->user_work   = (Vec **)PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void *));CHKPTRQ(gmres->user_work);
  gmres->mwork_alloc = (int*)PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(int));CHKPTRQ(gmres->mwork_alloc);
  PLogObjectMemory(ksp,(VEC_OFFSET+2+max_k)*(2*sizeof(void *)+sizeof(int)));

  if (gmres->q_preallocate) {
    gmres->vv_allocated   = VEC_OFFSET + 2 + max_k;
    ierr = VecDuplicateVecs(VEC_RHS,gmres->vv_allocated,&gmres->user_work[0]);CHKERRQ(ierr);
    PLogObjectParents(ksp,gmres->vv_allocated,gmres->user_work[0]);
    gmres->mwork_alloc[0] = gmres->vv_allocated;
    gmres->nwork_alloc    = 1;
    for (k=0; k<gmres->vv_allocated; k++) {
      gmres->vecs[k] = gmres->user_work[0][k];
    }
  } else {
    gmres->vv_allocated    = 5;
    ierr = VecDuplicateVecs(ksp->vec_rhs,5,&gmres->user_work[0]);CHKERRQ(ierr);
    PLogObjectParents(ksp,5,gmres->user_work[0]);
    gmres->mwork_alloc[0]  = 5;
    gmres->nwork_alloc     = 1;
    for (k=0; k<gmres->vv_allocated; k++) {
      gmres->vecs[k] = gmres->user_work[0][k];
    }
  }
  PetscFunctionReturn(0);
}

/* 
    This routine computes the initial residual
 */
#undef __FUNC__  
#define __FUNC__ /*<a name="GMRESResidual"></a>*/"GMRESResidual"
static int GMRESResidual(KSP ksp)
{
  KSP_GMRES    *gmres = (KSP_GMRES *)(ksp->data);
  Scalar       mone = -1.0;
  Mat          Amat,Pmat;
  MatStructure pflag;
  int          ierr;

  PetscFunctionBegin;
  ierr = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);CHKERRQ(ierr);
  /* compute initial residual: f - M*x */
  /* (inv(B)*A)*x or (A*inv(B)*B)*x into dest */
  if (ksp->pc_side == PC_RIGHT) {
    /* we want A * inv(B) * B * x, or just a * x for the first step */
    /* a*x into temp */
    ierr = KSP_MatMult(ksp,Amat,VEC_SOLN,VEC_TEMP);CHKERRQ(ierr);
  } else {
    /* else we do inv(B) * A * x */
    ierr = KSP_PCApplyBAorAB(ksp,ksp->B,ksp->pc_side,VEC_SOLN,VEC_TEMP,VEC_TEMP_MATOP);CHKERRQ(ierr);
  }
  /* This is an extra copy for the right-inverse case */
  ierr = VecCopy(VEC_BINVF,VEC_VV(0));CHKERRQ(ierr);
  ierr = VecAXPY(&mone,VEC_TEMP,VEC_VV(0));CHKERRQ(ierr);
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
#undef __FUNC__  
#define __FUNC__ /*<a name="GMREScycle"></a>*/"GMREScycle"
int GMREScycle(int *itcount,KSP ksp)
{
  KSP_GMRES  *gmres = (KSP_GMRES *)(ksp->data);
  PetscReal  res_norm,res;
  PetscReal  hapbnd,tt;
  Scalar     tmp;
  int        ierr,it = 0;
  int        max_k = gmres->max_k,max_it = ksp->max_it;
  PetscTruth hapend = PETSC_FALSE;

  PetscFunctionBegin;

  ierr   = VecNorm(VEC_VV(0),NORM_2,&res_norm);CHKERRQ(ierr);
  res    = res_norm;
  *RS(0) = res_norm;

  /* check for the convergence */
  if (!res) {
    if (itcount) *itcount = 0;
    ksp->reason = KSP_CONVERGED_RTOL;
    PetscFunctionReturn(0);
  }

  /* scale VEC_VV (the initial residual) */
  tmp = 1.0/res_norm; ierr = VecScale(&tmp,VEC_VV(0));CHKERRQ(ierr);

  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->rnorm = res;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
  gmres->it = (it - 1);
  ierr = (*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  while (!ksp->reason && it < max_k && ksp->its < max_it) {
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
    ierr = VecNorm(VEC_VV(it+1),NORM_2,&tt);CHKERRQ(ierr);
    /* save the magnitude */
    *HH(it+1,it)    = tt;
    *HES(it+1,it)   = tt;

    /* check for the happy breakdown */
    hapbnd  = gmres->epsabs * PetscAbsScalar(*HH(it,it) / *RS(it));
    if (hapbnd > gmres->haptol) hapbnd = gmres->haptol;
    if (tt > hapbnd) {
      tmp = 1.0/tt; ierr = VecScale(&tmp,VEC_VV(it+1));CHKERRQ(ierr);
    } else {
      PLogInfo(ksp,"Detected happy breakdown, current hapbnd = %g tt = %g\n",hapbnd,tt);
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
        SETERRQ1(0,0,"You reached the happy break down, but convergence was not indicated. Residual norm = %",res);
      }
      break;
    }
  }
  KSPLogResidualHistory(ksp,res);

  /*
     Monitor if we know that we will not return for a restart */
  if (ksp->reason || ksp->its >= max_it) {
    KSPMonitor(ksp,ksp->its,res);
  }

  if (itcount) *itcount    = it;


  /*
    Down here we have to solve for the "best" coefficients of the Krylov
    columns, add the solution values together, and possibly unwind the
    preconditioning from the solution
   */
  /* Form the solution (or the solution so far) */
  ierr = BuildGmresSoln(RS(0),VEC_SOLN,VEC_SOLN,ksp,it-1);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="KSPSolve_GMRES"></a>*/"KSPSolve_GMRES"
int KSPSolve_GMRES(KSP ksp,int *outits)
{
  int       ierr,its,itcount;
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;

  if (ksp->calc_sings && !gmres->Rsvd) {
    SETERRQ(1,1,"Must call KSPSetComputeSingularValues() before KSPSetUp() is called");
  }

  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its = 0;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);

  itcount  = 0;
  /* Save binv*f */
  if (ksp->pc_side == PC_LEFT) {
    /* inv(b)*f */
    ierr = KSP_PCApply(ksp,ksp->B,VEC_RHS,VEC_BINVF);CHKERRQ(ierr);
  } else if (ksp->pc_side == PC_RIGHT) {
    ierr = VecCopy(VEC_RHS,VEC_BINVF);CHKERRQ(ierr);
  }
  /* Compute the initial (preconditioned) residual */
  if (!ksp->guess_zero) {
    ierr = GMRESResidual(ksp);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(VEC_BINVF,VEC_VV(0));CHKERRQ(ierr);
  }
    
  ierr    = GMREScycle(&its,ksp);CHKERRQ(ierr);
  itcount += its;
  while (!ksp->reason) {
    ierr     = GMRESResidual(ksp);CHKERRQ(ierr);
    if (itcount >= ksp->max_it) break;
    /* need another check to make sure that gmres breaks out 
       at precisely the number of iterations chosen */
    ierr     = GMREScycle(&its,ksp);CHKERRQ(ierr);
    itcount += its;  
  }
  /* mark lack of convergence  */
  if (itcount >= ksp->max_it) {
    itcount--;
    ksp->reason = KSP_DIVERGED_ITS;
  }

  *outits = itcount;  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="KSPDestroy_GMRES"></a>*/"KSPDestroy_GMRES" 
int KSPDestroy_GMRES(KSP ksp)
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
#undef __FUNC__  
#define __FUNC__ /*<a name="BuildGmresSoln"></a>*/"BuildGmresSoln"
static int BuildGmresSoln(Scalar* nrs,Vec vs,Vec vdest,KSP ksp,int it)
{
  Scalar    tt,zero = 0.0,one = 1.0;
  int       ierr,ii,k,j;
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);

  PetscFunctionBegin;
  /* Solve for solution vector that minimizes the residual */

  /* If it is < 0, no gmres steps have been performed */
  if (it < 0) {
    if (vdest != vs) {
      ierr = VecCopy(vs,vdest);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  if (*HH(it,it) == 0.0) SETERRQ1(1,1,"HH(it,it) is identically zero; RS(it) = %g",*RS(it));
  if (*HH(it,it) != 0.0) {
    nrs[it] = *RS(it) / *HH(it,it);
  } else {
    nrs[it] = 0.0;
  }
  for (ii=1; ii<=it; ii++) {
    k   = it - ii;
    tt  = *RS(k);
    for (j=k+1; j<=it; j++) tt  = tt - *HH(k,j) * nrs[j];
    nrs[k]   = tt / *HH(k,k);
  }

  /* Accumulate the correction to the solution of the preconditioned problem in TEMP */
  ierr = VecSet(&zero,VEC_TEMP);CHKERRQ(ierr);
  ierr = VecMAXPY(it+1,nrs,VEC_TEMP,&VEC_VV(0));CHKERRQ(ierr);

  /* If we preconditioned on the right, we need to solve for the correction to
     the unpreconditioned problem */
  if (ksp->pc_side == PC_RIGHT) {
    if (vdest != vs) {
      ierr = KSP_PCApply(ksp,ksp->B,VEC_TEMP,vdest);CHKERRQ(ierr);
      ierr = VecAXPY(&one,vs,vdest);CHKERRQ(ierr);
    } else {
      ierr = KSP_PCApply(ksp,ksp->B,VEC_TEMP,VEC_TEMP_MATOP);CHKERRQ(ierr);
      ierr = VecAXPY(&one,VEC_TEMP_MATOP,vdest);CHKERRQ(ierr);
    }
  } else if (ksp->pc_side == PC_LEFT) {
    if (vdest != vs) {
      ierr = VecCopy(VEC_TEMP,vdest);CHKERRQ(ierr);
      ierr = VecAXPY(&one,vs,vdest);CHKERRQ(ierr);
    } else {
      ierr = VecAXPY(&one,VEC_TEMP,vdest);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
/*
   Do the scalar work for the orthogonalization.  Return new residual.
 */
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"GMRESUpdateHessenberg"
static int GMRESUpdateHessenberg(KSP ksp,int it,PetscTruth hapend,PetscReal *res)
{
  Scalar    *hh,*cc,*ss,tt;
  int       j;
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);

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
    if (tt == 0.0) {SETERRQ(PETSC_ERR_KSP_BRKDWN,0,"Your matrix or preconditioner is the null operator");}
    *cc       = *hh / tt;
    *ss       = *(hh+1) / tt;
    *RS(it+1) = - (*ss * *RS(it));
#if defined(PETSC_USE_COMPLEX)
    *RS(it)   = PetscConj(*cc) * *RS(it);
    *hh       = PetscConj(*cc) * *hh + *ss * *(hh+1);
#else
    *RS(it)   = *cc * *RS(it);
    *hh       = *cc * *hh + *ss * *(hh+1);
#endif
    *res      = PetscAbsScalar(*RS(it+1));
  } else {
    /* happy breakdown: HH(it+1, it) = 0, therfore we don't need to apply 
            another rotation matrix (so RH doesn't change).  The new residual is 
            always the new sine term times the residual from last time (RS(it)), 
            but now the new sine rotation would be zero...so the residual should
            be zero...so we will multiply "zero" by the last residual.  This might
            not be exactly what we want to do here -could just return "zero". */
 
    *res = PetscAbsScalar(gmres->epsabs * *RS(it));
  }
  PetscFunctionReturn(0);
}
/*
   This routine allocates more work vectors, starting from VEC_VV(it).
 */
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"GMRESGetNewVectors" 
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
  PLogObjectParents(ksp,nalloc,gmres->user_work[nwork]);CHKPTRQ(gmres->user_work[nwork]);
  gmres->mwork_alloc[nwork] = nalloc;
  for (k=0; k<nalloc; k++) {
    gmres->vecs[it+VEC_OFFSET+k] = gmres->user_work[nwork][k];
  }
  gmres->nwork_alloc++;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPBuildSolution_GMRES"
int KSPBuildSolution_GMRES(KSP ksp,Vec  ptr,Vec *result)
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data; 
  int       ierr;

  PetscFunctionBegin;
  if (!ptr) {
    if (!gmres->sol_temp) {
      ierr = VecDuplicate(ksp->vec_sol,&gmres->sol_temp);CHKERRQ(ierr);
      PLogObjectParent(ksp,gmres->sol_temp);
    }
    ptr = gmres->sol_temp;
  }
  if (!gmres->nrs) {
    /* allocate the work area */
    gmres->nrs = (Scalar *)PetscMalloc(gmres->max_k*sizeof(Scalar));
    PLogObjectMemory(ksp,gmres->max_k*sizeof(Scalar));
  }

  ierr = BuildGmresSoln(gmres->nrs,VEC_SOLN,ptr,ksp,gmres->it);CHKERRQ(ierr);
  *result = ptr; PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPView_GMRES" 
int KSPView_GMRES(KSP ksp,Viewer viewer)
{
  KSP_GMRES  *gmres = (KSP_GMRES *)ksp->data; 
  char       *cstr;
  int        ierr;
  PetscTruth isascii,isstring;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,STRING_VIEWER,&isstring);CHKERRQ(ierr);
  if (gmres->orthog == KSPGMRESUnmodifiedGramSchmidtOrthogonalization) {
    cstr = "Unmodified Gram-Schmidt Orthogonalization";
  } else if (gmres->orthog == KSPGMRESModifiedGramSchmidtOrthogonalization) {
    cstr = "Modified Gram-Schmidt Orthogonalization";
  } else if (gmres->orthog == KSPGMRESIROrthogonalization) {
    cstr = "Unmodified Gram-Schmidt + 1 step Iterative Refinement Orthogonalization";
  } else {
    cstr = "unknown orthogonalization";
  }
  if (isascii) {
    ierr = ViewerASCIIPrintf(viewer,"  GMRES: restart=%d, using %s\n",gmres->max_k,cstr);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = ViewerStringSPrintf(viewer,"%s restart %d",cstr,gmres->max_k);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for KSP GMRES",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPPrintHelp_GMRES"
static int KSPPrintHelp_GMRES(KSP ksp,char *p)
{
  int ierr;

  PetscFunctionBegin;
  ierr = (*PetscHelpPrintf)(ksp->comm," Options for GMRES method:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_gmres_restart <num>: GMRES restart, defaults to 30\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_gmres_unmodifiedgramschmidt: use alternative orthogonalization\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_gmres_modifiedgramschmidt: use alternative orthogonalization\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_gmres_irorthog: (default) use iterative refinement in orthogonalization\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_gmres_preallocate: preallocate GMRES work vectors\n",p);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPGMRESKrylovMonitor"
/*@C
   KSPGMRESKrylovMonitor - Calls VecView() for each direction in the 
   GMRES accumulated Krylov space.

   Collective on KSP

   Input Parameters:
+  ksp - the KSP context
.  its - iteration number
.  fgnorm - 2-norm of residual (or gradient)
-  a viewers object created with ViewersCreate()

   Level: intermediate

.keywords: KSP, nonlinear, vector, monitor, view, Krylov space

.seealso: KSPSetMonitor(), KSPDefaultMonitor(), VecView(), ViewersCreate(), ViewersDestroy()
@*/
int KSPGMRESKrylovMonitor(KSP ksp,int its,PetscReal fgnorm,void *dummy)
{
  Viewers   viewers = (Viewers)dummy;
  KSP_GMRES *gmres = (KSP_GMRES*)ksp->data;
  int       ierr;
  Vec       x;
  Viewer    viewer;

  PetscFunctionBegin;
  ierr = ViewersGetViewer(viewers,gmres->it+1,&viewer);CHKERRQ(ierr);
  ierr = ViewerSetType(viewer,DRAW_VIEWER);CHKERRQ(ierr);

  x      = VEC_VV(gmres->it+1);
  ierr   = VecView(x,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_AMS)
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPOptionsPublish_GMRES"
int KSPOptionsPublish_GMRES(KSP ksp)
{
  int       ierr;
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;
  ierr = OptionsSelectBegin(ksp->comm,ksp->prefix,"KSP GMRES Options");CHKERRQ(ierr);
  ierr = OptionsSelectInt(ksp->comm,"-ksp_gmres_restart","Number of Krylov search directions",gmres->max_k);CHKERRQ(ierr);
  ierr = OptionsSelectName(ksp->comm,"-ksp_gmres_preallocate","Preallocate all Krylov vectors");CHKERRQ(ierr);
  ierr = OptionsSelectName(ksp->comm,"-ksp_gmres_unmodifiedgramschmidt","Use classical (unmodified) Gram-Schmidt (fast)");CHKERRQ(ierr);
  ierr = OptionsSelectName(ksp->comm,"-ksp_gmres_modifiedgramschmidt","Use modified Gram-Schmidt (slow but more stable)");CHKERRQ(ierr);
  ierr = OptionsSelectName(ksp->comm,"-ksp_gmres_irorthog","Use classical Gram-Schmidt with iterative refinement");CHKERRQ(ierr);
  ierr = OptionsSelectName(ksp->comm,"-ksp_gmres_krylov_monitor","Graphically plot the Krylov directions");CHKERRQ(ierr);
  ierr = OptionsSelectEnd(ksp->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPSetFromOptions_GMRES"
int KSPSetFromOptions_GMRES(KSP ksp)
{
  int        ierr,restart;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = OptionsGetInt(ksp->prefix,"-ksp_gmres_restart",&restart,&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPGMRESSetRestart(ksp,restart);CHKERRQ(ierr); }
  ierr = OptionsHasName(ksp->prefix,"-ksp_gmres_preallocate",&flg);CHKERRQ(ierr);
  if (flg) {ierr = KSPGMRESSetPreAllocateVectors(ksp);CHKERRQ(ierr);}
  ierr = OptionsHasName(ksp->prefix,"-ksp_gmres_unmodifiedgramschmidt",&flg);CHKERRQ(ierr);
  if (flg) {ierr = KSPGMRESSetOrthogonalization(ksp,KSPGMRESUnmodifiedGramSchmidtOrthogonalization);CHKERRQ(ierr);}
  ierr = OptionsHasName(ksp->prefix,"-ksp_gmres_modifiedgramschmidt",&flg);CHKERRQ(ierr);
  if (flg) {ierr = KSPGMRESSetOrthogonalization(ksp,KSPGMRESModifiedGramSchmidtOrthogonalization);CHKERRQ(ierr);}
  ierr = OptionsHasName(ksp->prefix,"-ksp_gmres_irorthog",&flg);CHKERRQ(ierr);
  if (flg) {ierr = KSPGMRESSetOrthogonalization(ksp,KSPGMRESIROrthogonalization);CHKERRQ(ierr);}
  ierr = OptionsHasName(ksp->prefix,"-ksp_gmres_krylov_monitor",&flg);CHKERRQ(ierr);
  if (flg) {
    Viewers viewers;
    ierr = ViewersCreate(ksp->comm,&viewers);CHKERRQ(ierr);
    ierr = KSPSetMonitor(ksp,KSPGMRESKrylovMonitor,viewers,(int (*)(void*))ViewersDestroy);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

EXTERN int KSPComputeExtremeSingularValues_GMRES(KSP,PetscReal *,PetscReal *);
EXTERN int KSPComputeEigenvalues_GMRES(KSP,int,PetscReal *,PetscReal *,int *);


EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPGMRESSetRestart_GMRES" 
int KSPGMRESSetRestart_GMRES(KSP ksp,int max_k)
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;
  gmres->max_k = max_k;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPGMRESSetOrthogonalization_GMRES" 
int KSPGMRESSetOrthogonalization_GMRES(KSP ksp,int (*fcn)(KSP,int))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ((KSP_GMRES *)ksp->data)->orthog = fcn;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPGMRESSetPreAllocateVectors_GMRES" 
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
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPCreate_GMRES"
int KSPCreate_GMRES(KSP ksp)
{
  KSP_GMRES *gmres;
  int       ierr;

  PetscFunctionBegin;
  gmres = (KSP_GMRES*)PetscMalloc(sizeof(KSP_GMRES));CHKPTRQ(gmres);
  ierr  = PetscMemzero(gmres,sizeof(KSP_GMRES));CHKERRQ(ierr);
  PLogObjectMemory(ksp,sizeof(KSP_GMRES));
  ksp->data                              = (void*)gmres;
  ksp->ops->buildsolution                = KSPBuildSolution_GMRES;

  ksp->ops->setup                        = KSPSetUp_GMRES;
  ksp->ops->solve                        = KSPSolve_GMRES;
  ksp->ops->destroy                      = KSPDestroy_GMRES;
  ksp->ops->view                         = KSPView_GMRES;
  ksp->ops->printhelp                    = KSPPrintHelp_GMRES;
  ksp->ops->setfromoptions               = KSPSetFromOptions_GMRES;
  ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_GMRES;
  ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_GMRES;
#if defined(PETSC_HAVE_AMS)
  ksp->ops->publishoptions               = KSPOptionsPublish_GMRES;
#endif

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetPreAllocateVectors_C",
                                    "KSPGMRESSetPreAllocateVectors_GMRES",
                                     KSPGMRESSetPreAllocateVectors_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetOrthogonalization_C",
                                    "KSPGMRESSetOrthogonalization_GMRES",
                                     KSPGMRESSetOrthogonalization_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetRestart_C",
                                    "KSPGMRESSetRestart_GMRES",
                                     KSPGMRESSetRestart_GMRES);CHKERRQ(ierr);

  gmres->haptol              = 1.0e-8;
  gmres->epsabs              = 1.0e-8;
  gmres->q_preallocate       = 0;
  gmres->delta_allocate      = GMRES_DELTA_DIRECTIONS;
  gmres->orthog              = KSPGMRESUnmodifiedGramSchmidtOrthogonalization;
  gmres->nrs                 = 0;
  gmres->sol_temp            = 0;
  gmres->max_k               = GMRES_DEFAULT_MAXK;
  gmres->Rsvd                = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

