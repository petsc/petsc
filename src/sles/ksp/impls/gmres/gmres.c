/*$Id: gmres.c,v 1.130 1999/10/24 14:03:14 bsmith Exp bsmith $*/

/*
    This file implements GMRES (a Generalized Minimal Residual) method.  
    Reference:  Saad and Schultz, 1986.

    The solver may be called recursively as long as all of the user-supplied
    routines can. This routine is meant to be compatible with execution on a
    parallel processor.  As such, it expects to be given routines for 
    all operations as well as a user-defined pointer to a distributed
    data structure.  THIS IS A DATA-STRUCTURE NEUTRAL IMPLEMENTATION.
  
    A context variable is used to hold internal data (the Hessenberg
    matrix and various parameters).

    Here are the routines that must be provided.  The generic parameters
    are:
	 ksp   - Iterative context.  See the generic iterative method
	         information

    The calling sequence is the same as for all of the iterative methods.
    The special values (specific to GMRES) are:

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

#include "src/sles/ksp/impls/gmres/gmresp.h"       /*I  "ksp.h"  I*/
#define GMRES_DELTA_DIRECTIONS 10
#define GMRES_DEFAULT_MAXK     30
static int    GMRESGetNewVectors( KSP ,int );
static int    GMRESUpdateHessenberg( KSP , int,double * );
static int    BuildGmresSoln(Scalar* ,Vec,Vec ,KSP, int);

#undef __FUNC__  
#define __FUNC__ "KSPSetUp_GMRES"
int    KSPSetUp_GMRES(KSP ksp )
{
  unsigned  int size, hh, hes, rs, cc;
  int       ierr,  max_k, k;
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

  gmres->hh_origin  = (Scalar *) PetscMalloc(size);CHKPTRQ(gmres->hh_origin);
  PLogObjectMemory(ksp,size);
  gmres->hes_origin = gmres->hh_origin + hh;
  gmres->rs_origin  = gmres->hes_origin + hes;
  gmres->cc_origin  = gmres->rs_origin + rs;
  gmres->ss_origin  = gmres->cc_origin + cc;

  if (ksp->calc_sings) {
    /* Allocate workspace to hold Hessenberg matrix needed by Eispack */
    size = (max_k + 3)*(max_k + 9)*sizeof(Scalar);
    gmres->Rsvd = (Scalar *) PetscMalloc(size);CHKPTRQ(gmres->Rsvd);
    gmres->Dsvd = (double *) PetscMalloc(5*(max_k+2)*sizeof(double));CHKPTRQ(gmres->Dsvd);
    PLogObjectMemory(ksp,size+5*(max_k+2)*sizeof(double));
  }

  /* Allocate array to hold pointers to user vectors.  Note that we need
   4 + max_k + 1 (since we need it+1 vectors, and it <= max_k) */
  gmres->vecs = (Vec *) PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void *));CHKPTRQ(gmres->vecs);
  gmres->vecs_allocated = VEC_OFFSET + 2 + max_k;
  gmres->user_work   = (Vec **)PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void *));CHKPTRQ(gmres->user_work);
  gmres->mwork_alloc = (int *) PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(int));CHKPTRQ(gmres->mwork_alloc);
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
    ierr = VecDuplicateVecs(ksp->vec_rhs, 5,    &gmres->user_work[0]);CHKERRQ(ierr);
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
    This routine computes the initial residual without making any assumptions
    about the solution.
 */
#undef __FUNC__  
#define __FUNC__ "GMRESResidual"
static int GMRESResidual(  KSP ksp,int restart )
{
  KSP_GMRES    *gmres = (KSP_GMRES *)(ksp->data);
  Scalar       mone = -1.0;
  Mat          Amat, Pmat;
  MatStructure pflag;
  int          ierr;

  PetscFunctionBegin;
  ierr = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);CHKERRQ(ierr);
  /* compute initial residual: f - M*x */
  /* (inv(b)*a)*x or (a*inv(b)*b)*x into dest */
  if (ksp->pc_side == PC_RIGHT) {
    /* we want a * binv * b * x, or just a * x for the first step */
    /* a*x into temp */
    ierr = KSP_MatMult(ksp,Amat,VEC_SOLN,VEC_TEMP );CHKERRQ(ierr);
  } else {
    /* else we do binv * a * x */
    ierr = KSP_PCApplyBAorAB(ksp,ksp->B,ksp->pc_side,VEC_SOLN,VEC_TEMP,VEC_TEMP_MATOP);CHKERRQ(ierr);
  }
  /* This is an extra copy for the right-inverse case */
  ierr = VecCopy( VEC_BINVF, VEC_VV(gmres->nprestart) );CHKERRQ(ierr);
  ierr = VecAXPY( &mone, VEC_TEMP, VEC_VV(gmres->nprestart) );CHKERRQ(ierr);
      /* inv(b)(f - a*x) into dest */
  PetscFunctionReturn(0);
}

/*
    Run gmres, possibly with restart.  Return residual history if requested.
    input parameters:
.        restart - 1 if restarting gmres, 0 otherwise
.	gmres  - structure containing parameters and work areas
.	itsSoFar- total number of iterations so far (from previous cycles)

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
#define __FUNC__ "GMREScycle"
int GMREScycle(int *  itcount, int itsSoFar,int restart,KSP ksp,int *converged )
{
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  double    res_norm, res;
  double    hapbnd,tt;
  Scalar    tmp;
  int       ierr, it;
  int       max_k = gmres->max_k, max_it = ksp->max_it;

  /* Note that hapend is ignored in the code */

  PetscFunctionBegin;

  /*
    Number of pseudo iterations since last restart is the number of prestart directions
  */
  it         = gmres->nprestart;
  *converged = 0;


  if (it > 0) {
    /* orthogonalize input against previous directions and update Hessenberg matrix */

    /* update hessenberg matrix and do Gram-Schmidt */
    ierr = (*gmres->orthog)(  ksp, it-1 );CHKERRQ(ierr);

    /* vv(i) . vv(i) */
    ierr = VecNorm(VEC_VV(it),NORM_2,&tt);CHKERRQ(ierr);
    /* save the magnitude */
    *HH(it,it-1)    = tt;
    *HES(it,it-1)   = tt;

    /* check for the convergence */
    if (!tt) {
      if (itcount) *itcount = 0;
      *converged = 1;
      PetscFunctionReturn(0);
    }
    tmp = 1.0/tt; ierr = VecScale( &tmp, VEC_VV(it) );CHKERRQ(ierr);

    ierr = GMRESUpdateHessenberg( ksp, it-1, &res );CHKERRQ(ierr);
  } else {

    ierr   = VecNorm(VEC_VV(0),NORM_2,&res_norm);CHKERRQ(ierr);
    res    = res_norm;
    *RS(0) = res_norm;

    /* check for the convergence */
    if (!res) {
      if (itcount) *itcount = 0;
      *converged = 1;
      PetscFunctionReturn(0);
    }

    /* scale VEC_VV (the initial residual) */
    tmp = 1.0/res_norm; ierr = VecScale(&tmp , VEC_VV(0) );CHKERRQ(ierr);
  }

  if (!restart) {
    ksp->ttol = PetscMax(ksp->rtol*res_norm,ksp->atol);
  }
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->rnorm = res;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
  gmres->it = (it - 1);
  while (!(*converged = (*ksp->converged)(ksp,ksp->its,res,ksp->cnvP))
           && it < max_k && ksp->its < max_it) {
    KSPLogResidualHistory(ksp,res);
    gmres->it = (it - 1);
    KSPMonitor(ksp,ksp->its,res); 
    if (gmres->vv_allocated <= it + VEC_OFFSET + 1) {
      ierr = GMRESGetNewVectors(  ksp, it+1 );CHKERRQ(ierr);
    }
    ierr = KSP_PCApplyBAorAB(ksp,ksp->B,ksp->pc_side,VEC_VV(it),VEC_VV(1+it),VEC_TEMP_MATOP);CHKERRQ(ierr);

    /* update hessenberg matrix and do Gram-Schmidt */
    ierr = (*gmres->orthog)(  ksp, it );CHKERRQ(ierr);

    /* vv(i+1) . vv(i+1) */
    ierr = VecNorm(VEC_VV(it+1),NORM_2,&tt);CHKERRQ(ierr);
    /* save the magnitude */
    *HH(it+1,it)    = tt;
    *HES(it+1,it)   = tt;

    /* check for the happy breakdown */
    hapbnd  = gmres->epsabs * PetscAbsScalar( *HH(it,it) / *RS(it) );
    if (hapbnd > gmres->haptol) hapbnd = gmres->haptol;
    if (tt > hapbnd) {
        tmp = 1.0/tt; ierr = VecScale( &tmp, VEC_VV(it+1) );CHKERRQ(ierr);
    } else {
        /* We SHOULD probably abort the gmres step
           here.  This happens when the solution is exactly reached. */
      ; /* hapend = 1;   */
    }
    ierr = GMRESUpdateHessenberg( ksp, it, &res );CHKERRQ(ierr);
    it++;
    gmres->it  = (it-1);  /* For converged */
    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = res;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
  }
  KSPLogResidualHistory(ksp,res);

  /*
     Monitor if we know that we will not return for a restart */
  if (*converged || ksp->its >= max_it) {
    KSPMonitor( ksp,  ksp->its, res );
  }

  if (itcount) *itcount    = it;


  /* Didn't go in any direction, current solution is correct */
  if (it == gmres->nprestart) {
    *converged = 1;
    PetscFunctionReturn(0);
  }

  /*
    Down here we have to solve for the "best" coefficients of the Krylov
    columns, add the solution values together, and possibly unwind the
    preconditioning from the solution
   */
  /* Form the solution (or the solution so far) */
  ierr = BuildGmresSoln(RS(0),VEC_SOLN,VEC_SOLN,ksp,it-1);CHKERRQ(ierr);

  /* set the prestart counter */
  if (gmres->nprestart_requested > 0 && gmres->nprestart == 0) {
    /* 
       Cut off to make sure number of directions is less than or equal
       number of directions actually computed
    */
    gmres->nprestart = PetscMin(it-1,gmres->nprestart_requested);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSolve_GMRES"
int KSPSolve_GMRES(KSP ksp,int *outits )
{
  int       ierr, restart, its, itcount, converged;
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its = 0;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);

  restart  = 0;
  itcount  = 0;
  /* Save binv*f */
  if (ksp->pc_side == PC_LEFT) {
    /* inv(b)*f */
    ierr = KSP_PCApply(ksp,ksp->B, VEC_RHS, VEC_BINVF );CHKERRQ(ierr);
  } else if (ksp->pc_side == PC_RIGHT) {
    ierr = VecCopy( VEC_RHS, VEC_BINVF );CHKERRQ(ierr);
  }
  /* Compute the initial (preconditioned) residual */
  if (!ksp->guess_zero) {
    ierr = GMRESResidual(  ksp, restart );CHKERRQ(ierr);
  } else {
    ierr = VecCopy( VEC_BINVF, VEC_VV(gmres->nprestart) );CHKERRQ(ierr);
  }
    
  ierr    = GMREScycle(&its, itcount, restart, ksp, &converged);CHKERRQ(ierr);
  itcount += its;
  while (!converged) {
    restart  = 1;
    ierr     = GMRESResidual(  ksp, restart);CHKERRQ(ierr);
    if (itcount >= ksp->max_it) break;
    /* need another check to make sure that gmres breaks out 
       at precisely the number of iterations chosen */
    ierr     = GMREScycle(&its, itcount, restart, ksp, &converged);CHKERRQ(ierr);
    itcount += its;  
  }
  /* mark lack of convergence with negative the number of iterations */
  if (itcount >= ksp->max_it) itcount = -itcount;

  *outits = itcount;  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPDestroy_GMRES" 
int KSPDestroy_GMRES(KSP ksp)
{
  KSP_GMRES *gmres = (KSP_GMRES *) ksp->data;
  int       i,ierr;

  PetscFunctionBegin;
  /* Free the Hessenberg matrix */
  if (gmres->hh_origin) {ierr = PetscFree( gmres->hh_origin );CHKERRQ(ierr);}

  /* Free the pointer to user variables */
  if (gmres->vecs) {ierr = PetscFree( gmres->vecs );CHKERRQ(ierr);}

  /* free work vectors */
  for (i=0; i<gmres->nwork_alloc; i++) {
    ierr = VecDestroyVecs(gmres->user_work[i], gmres->mwork_alloc[i] );CHKERRQ(ierr);
  }
  if (gmres->user_work)  {ierr = PetscFree( gmres->user_work );CHKERRQ(ierr);}
  if (gmres->mwork_alloc) {ierr = PetscFree( gmres->mwork_alloc );CHKERRQ(ierr);}
  if (gmres->nrs) {ierr = PetscFree( gmres->nrs );CHKERRQ(ierr);}
  if (gmres->sol_temp) {ierr = VecDestroy(gmres->sol_temp);CHKERRQ(ierr);}
  if (gmres->Rsvd) {ierr = PetscFree(gmres->Rsvd);CHKERRQ(ierr);}
  if (gmres->Dsvd) {ierr = PetscFree(gmres->Dsvd);CHKERRQ(ierr);}
  ierr = PetscFree( gmres ); CHKERRQ(ierr);
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
#define __FUNC__ "BuildGmresSoln"
static int BuildGmresSoln(Scalar* nrs,Vec vs,Vec vdest,KSP ksp, int it )
{
  Scalar    tt, zero = 0.0, one = 1.0;
  int       ierr, ii, k, j;
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);

  PetscFunctionBegin;
  /* Solve for solution vector that minimizes the residual */

  /* If it is < 0, no gmres steps have been performed */
  if (it < 0) {
    if (vdest != vs) {
      ierr = VecCopy( vs, vdest );CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  nrs[it] = *RS(it) / *HH(it,it);
  for (ii=1; ii<=it; ii++) {
    k   = it - ii;
    tt  = *RS(k);
    for (j=k+1; j<=it; j++) tt  = tt - *HH(k,j) * nrs[j];
    nrs[k]   = tt / *HH(k,k);
  }

  /* Accumulate the correction to the solution of the preconditioned problem in TEMP */
  ierr = VecSet( &zero, VEC_TEMP );CHKERRQ(ierr);
  ierr = VecMAXPY(it+1, nrs, VEC_TEMP, &VEC_VV(0) );CHKERRQ(ierr);

  /* If we preconditioned on the right, we need to solve for the correction to
     the unpreconditioned problem */
  if (ksp->pc_side == PC_RIGHT) {
    if (vdest != vs) {
      ierr = KSP_PCApply(ksp,ksp->B, VEC_TEMP, vdest );CHKERRQ(ierr);
      ierr = VecAXPY( &one, vs, vdest );CHKERRQ(ierr);
    } else {
      ierr = KSP_PCApply(ksp,ksp->B,VEC_TEMP,VEC_TEMP_MATOP);CHKERRQ(ierr);
      ierr = VecAXPY(&one,VEC_TEMP_MATOP,vdest);CHKERRQ(ierr);
    }
  } else if (ksp->pc_side == PC_LEFT) {
    if (vdest != vs) {
      ierr = VecCopy( VEC_TEMP, vdest );CHKERRQ(ierr);
      ierr = VecAXPY( &one, vs, vdest );CHKERRQ(ierr);
    } else {
      ierr = VecAXPY( &one, VEC_TEMP, vdest );CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
/*
   Do the scalar work for the orthogonalization.  Return new residual.
 */
#undef __FUNC__  
#define __FUNC__ "GMRESUpdateHessenberg"
static int GMRESUpdateHessenberg( KSP ksp, int it, double *res )
{
  Scalar    *hh, *cc, *ss, tt;
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
    *hh = *cc++ * *hh - ( *ss++ * tt );
  }

  /*
    compute the new plane rotation, and apply it to:
     1) the right-hand-side of the Hessenberg system
     2) the new column of the Hessenberg matrix
    thus obtaining the updated value of the residual
  */
#if defined(PETSC_USE_COMPLEX)
  tt        = PetscSqrtScalar( PetscConj(*hh) * *hh + PetscConj(*(hh+1)) * *(hh+1) );
#else
  tt        = PetscSqrtScalar( *hh * *hh + *(hh+1) * *(hh+1) );
#endif
  if (tt == 0.0) {SETERRQ(PETSC_ERR_KSP_BRKDWN,0,"Your matrix or preconditioner is the null operator");}
  *cc       = *hh / tt;
  *ss       = *(hh+1) / tt;
  *RS(it+1) = - ( *ss * *RS(it) );
#if defined(PETSC_USE_COMPLEX)
  *RS(it)   = PetscConj(*cc) * *RS(it);
  *hh       = PetscConj(*cc) * *hh + *ss * *(hh+1);
#else
  *RS(it)   = *cc * *RS(it);
  *hh       = *cc * *hh + *ss * *(hh+1);
#endif
  *res      = PetscAbsScalar( *RS(it+1) );
  PetscFunctionReturn(0);
}
/*
   This routine allocates more work vectors, starting from VEC_VV(it).
 */
#undef __FUNC__  
#define __FUNC__ "GMRESGetNewVectors" 
static int GMRESGetNewVectors( KSP ksp,int it )
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;
  int       nwork = gmres->nwork_alloc,k, nalloc,ierr;

  PetscFunctionBegin;
  nalloc = gmres->delta_allocate;
  /* Adjust the number to allocate to make sure that we don't exceed the
    number of available slots */
  if (it + VEC_OFFSET + nalloc >= gmres->vecs_allocated)
      nalloc = gmres->vecs_allocated - it - VEC_OFFSET;
  /*CHKPTRQ(nalloc); */
  if (nalloc == 0) PetscFunctionReturn(0);

  gmres->vv_allocated += nalloc;
  ierr = VecDuplicateVecs(ksp->vec_rhs, nalloc,&gmres->user_work[nwork] );CHKERRQ(ierr);
  PLogObjectParents(ksp,nalloc,gmres->user_work[nwork]);CHKPTRQ(gmres->user_work[nwork]);
  gmres->mwork_alloc[nwork] = nalloc;
  for (k=0; k<nalloc; k++) {
    gmres->vecs[it+VEC_OFFSET+k] = gmres->user_work[nwork][k];
  }
  gmres->nwork_alloc++;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPBuildSolution_GMRES"
int KSPBuildSolution_GMRES(KSP ksp,Vec  ptr,Vec *result )
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data; 
  int       ierr;

  PetscFunctionBegin;
  if (ptr == 0) {
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
#define __FUNC__ "KSPView_GMRES" 
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
    if (gmres->nprestart > 0) {
      ierr = ViewerASCIIPrintf(viewer,"  GMRES: using prestart=%d\n",gmres->nprestart);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = ViewerStringSPrintf(viewer,"%s restart %d",cstr,gmres->max_k);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for KSP GMRES",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPPrintHelp_GMRES"
static int KSPPrintHelp_GMRES(KSP ksp,char *p)
{
  int ierr;

  PetscFunctionBegin;
  ierr = (*PetscHelpPrintf)(ksp->comm," Options for GMRES method:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_gmres_restart <num>: GMRES restart, defaults to 30\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_gmres_prestart <num>: GMRES prestart, defaults to 0\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_gmres_unmodifiedgramschmidt: use alternative orthogonalization\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_gmres_modifiedgramschmidt: use alternative orthogonalization\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_gmres_irorthog: (default) use iterative refinement in orthogonalization\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_gmres_preallocate: preallocate GMRES work vectors\n",p);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGMRESKrylovMonitor"
/*@C
    KSPGMRESKrylovMonitor- Calls VecView() for each direction in the 
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
int KSPGMRESKrylovMonitor(KSP ksp,int its,double fgnorm,void *dummy)
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

#undef __FUNC__  
#define __FUNC__ "KSPSetFromOptions_GMRES"
int KSPSetFromOptions_GMRES(KSP ksp)
{
  int        ierr,restart,prestart;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = OptionsGetInt(ksp->prefix,"-ksp_gmres_restart",&restart,&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPGMRESSetRestart(ksp,restart);CHKERRQ(ierr); }
  ierr = OptionsHasName(ksp->prefix,"-ksp_gmres_preallocate", &flg);CHKERRQ(ierr);
  if (flg) {ierr = KSPGMRESSetPreAllocateVectors(ksp);CHKERRQ(ierr);}
  ierr = OptionsHasName(ksp->prefix,"-ksp_gmres_unmodifiedgramschmidt",&flg);CHKERRQ(ierr);
  if (flg) {ierr = KSPGMRESSetOrthogonalization(ksp,KSPGMRESUnmodifiedGramSchmidtOrthogonalization);CHKERRQ(ierr);}
  ierr = OptionsHasName(ksp->prefix,"-ksp_gmres_modifiedgramschmidt",&flg);CHKERRQ(ierr);
  if (flg) {ierr = KSPGMRESSetOrthogonalization(ksp,KSPGMRESModifiedGramSchmidtOrthogonalization);CHKERRQ(ierr);}
  ierr = OptionsGetInt(ksp->prefix,"-ksp_gmres_prestart",&prestart,&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPGMRESPrestartSet(ksp,prestart);CHKERRQ(ierr); }
  ierr = OptionsHasName(ksp->prefix,"-ksp_gmres_irorthog",&flg);CHKERRQ(ierr);
  if (flg) {ierr = KSPGMRESSetOrthogonalization(ksp, KSPGMRESIROrthogonalization);CHKERRQ(ierr);}
  ierr = OptionsHasName(ksp->prefix,"-ksp_gmres_krylov_monitor",&flg);CHKERRQ(ierr);
  if (flg) {
    Viewers viewers;
    ierr = ViewersCreate(ksp->comm,&viewers);CHKERRQ(ierr);
    ierr = KSPSetMonitor(ksp,KSPGMRESKrylovMonitor,viewers,(int (*)(void*))ViewersDestroy);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

extern int KSPComputeExtremeSingularValues_GMRES(KSP,double *,double *);
extern int KSPComputeEigenvalues_GMRES(KSP,int,double *,double *,int *);
extern int KSPDefaultConverged_GMRES(KSP,int,double,void*);

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "KSPGMRESPrestartSet_GMRES" 
int KSPGMRESPrestartSet_GMRES(KSP ksp,int pre)
{
  KSP_GMRES *gmres;

  PetscFunctionBegin;
  gmres                      = (KSP_GMRES *)ksp->data;
  if (pre > gmres->max_k-1) {
    SETERRQ(1,1,"Prestart count is too large for current restart");
  }
  gmres->nprestart_requested = pre;
  gmres->nprestart           = 0; /*reset this so that it will be set after the first solve*/
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "KSPGMRESSetRestart_GMRES" 
int KSPGMRESSetRestart_GMRES(KSP ksp,int max_k )
{
  KSP_GMRES *gmres;

  PetscFunctionBegin;
  gmres = (KSP_GMRES *)ksp->data;
  gmres->max_k = max_k;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "KSPGMRESSetOrthogonalization_GMRES" 
int KSPGMRESSetOrthogonalization_GMRES( KSP ksp,int (*fcn)(KSP,int) )
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ((KSP_GMRES *)ksp->data)->orthog = fcn;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "KSPGMRESSetPreAllocateVectors_GMRES" 
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
#define __FUNC__ "KSPCreate_GMRES"
int KSPCreate_GMRES(KSP ksp)
{
  KSP_GMRES *gmres;
  int       ierr;

  PetscFunctionBegin;
  gmres = (KSP_GMRES*) PetscMalloc(sizeof(KSP_GMRES));CHKPTRQ(gmres);
  ierr  = PetscMemzero(gmres,sizeof(KSP_GMRES));CHKERRQ(ierr);
  PLogObjectMemory(ksp,sizeof(KSP_GMRES));
  ksp->data                              = (void *) gmres;
  ksp->converged                         = KSPDefaultConverged_GMRES;
  ksp->ops->buildsolution                = KSPBuildSolution_GMRES;

  ksp->ops->setup                        = KSPSetUp_GMRES;
  ksp->ops->solve                        = KSPSolve_GMRES;
  ksp->ops->destroy                      = KSPDestroy_GMRES;
  ksp->ops->view                         = KSPView_GMRES;
  ksp->ops->printhelp                    = KSPPrintHelp_GMRES;
  ksp->ops->setfromoptions               = KSPSetFromOptions_GMRES;
  ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_GMRES;
  ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_GMRES;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetPreAllocateVectors_C",
                                    "KSPGMRESSetPreAllocateVectors_GMRES",
                                     (void*)KSPGMRESSetPreAllocateVectors_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetOrthogonalization_C",
                                    "KSPGMRESSetOrthogonalization_GMRES",
                                     (void*)KSPGMRESSetOrthogonalization_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetRestart_C",
                                     "KSPGMRESSetRestart_GMRES",
                                    (void*)KSPGMRESSetRestart_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESPrestartSet_C",
                                     "KSPGMRESPrestartSet_GMRES",
                                    (void*)KSPGMRESPrestartSet_GMRES);CHKERRQ(ierr);

  gmres->haptol              = 1.0e-8;
  gmres->epsabs              = 1.0e-8;
  gmres->q_preallocate       = 0;
  gmres->delta_allocate      = GMRES_DELTA_DIRECTIONS;
  gmres->orthog              = KSPGMRESIROrthogonalization;
  gmres->nrs                 = 0;
  gmres->sol_temp            = 0;
  gmres->max_k               = GMRES_DEFAULT_MAXK;
  gmres->Rsvd                = 0;
  gmres->nprestart           = 0;
  gmres->nprestart_requested = 0;
  ksp->guess_zero            = 1; 
  PetscFunctionReturn(0);
}
EXTERN_C_END

