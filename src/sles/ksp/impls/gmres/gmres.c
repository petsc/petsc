
#ifndef lint
static char vcid[] = "$Id: gmres.c,v 1.53 1996/01/26 04:32:43 bsmith Exp bsmith $";
#endif

/*
    This implements gmres.  It may be called recurrsively as long as 
    all of the user-supplied routines can. 

    This routine is meant to be compatible with execution on a parallel
    processor.  As such, it expects to be given routines for 
    all operations as well as a user-defined pointer to a distributed
    data structure.  THIS IS A DATA-STRUCTURE NEUTRAL IMPLEMENTATION.
  
    A context variable is used to hold internal data (the Hessenberg
    matrix and various parameters).

    Here are the routines that must be provided.  The generic parameters
    are:
	 itP   - Iterative context.  See the generic iterative method
	         information

    Special routines needed only for gmres:

    void   orthog(  itP, it )
        perform the orthogonalization of the vectors VV to VV+it.  A 
        basic version of this, defined in terms of vdot and maxpy, is
        available (in borthog.c) called void GMRESBasicOrthog;
	The user may use this routine to try alternate approaches.

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

#include <math.h>
#include <stdio.h>
#include "gmresp.h"       /*I  "ksp.h"  I*/
#include "pinclude/pviewer.h"
#define GMRES_DELTA_DIRECTIONS 10
#define GMRES_DEFAULT_MAXK     30
static int    GMRESGetNewVectors( KSP ,int );
static int    GMRESUpdateHessenberg( KSP , int,double * );
static int    BuildGmresSoln(Scalar* ,Vec,Vec ,KSP, int);

static int    KSPSetUp_GMRES(KSP itP )
{
  unsigned  int size, hh, hes, rs, cc;
  int       ierr,  max_k, k;
  KSP_GMRES *gmresP = (KSP_GMRES *)itP->data;

  if (itP->pc_side == PC_SYMMETRIC)
    {SETERRQ(2,"KSPSetUp_GMRES:no symmetric preconditioning for KSPGMRES");}
  if ((ierr = KSPCheckDef( itP ))) return ierr;
  max_k         = gmresP->max_k;
  hh            = (max_k + 2) * (max_k + 1);
  hes           = (max_k + 1) * (max_k + 1);
  rs            = (max_k + 2);
  cc            = (max_k + 1);
  size          = (hh + hes + rs + 2*cc) * sizeof(Scalar);

  gmresP->hh_origin  = (Scalar *) PetscMalloc(size);CHKPTRQ(gmresP->hh_origin);
  PLogObjectMemory(itP,size);
  gmresP->hes_origin = gmresP->hh_origin + hh;
  gmresP->rs_origin  = gmresP->hes_origin + hes;
  gmresP->cc_origin  = gmresP->rs_origin + rs;
  gmresP->ss_origin  = gmresP->cc_origin + cc;

  /* Allocate array to hold pointers to user vectors.  Note that we need
   4 + max_k + 1 (since we need it+1 vectors, and it <= max_k) */
  gmresP->vecs = (Vec *) PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void *));
  CHKPTRQ(gmresP->vecs);
  gmresP->vecs_allocated = VEC_OFFSET + 2 + max_k;
  gmresP->user_work = (Vec **)PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void *));
  CHKPTRQ(gmresP->user_work);
  gmresP->mwork_alloc = (int *) PetscMalloc( (VEC_OFFSET+2+max_k)*sizeof(int) );
  CHKPTRQ(gmresP->mwork_alloc);
  PLogObjectMemory(itP,(VEC_OFFSET+2+max_k)*(2*sizeof(void *)+sizeof(int)));

  if (gmresP->q_preallocate) {
    gmresP->vv_allocated   = VEC_OFFSET + 2 + max_k;
    ierr = VecDuplicateVecs(itP->vec_rhs,gmresP->vv_allocated,&gmresP->user_work[0]);
    CHKERRQ(ierr);
    PLogObjectParents(itP,gmresP->vv_allocated,gmresP->user_work[0]);
    gmresP->mwork_alloc[0] = gmresP->vv_allocated;
    gmresP->nwork_alloc    = 1;
    for (k=0; k<gmresP->vv_allocated; k++)
	gmresP->vecs[k] = gmresP->user_work[0][k];
  }
  else {
    gmresP->vv_allocated    = 5;
    ierr = VecDuplicateVecs(itP->vec_rhs, 5,    &gmresP->user_work[0]); CHKERRQ(ierr);
    PLogObjectParents(itP,5,gmresP->user_work[0]);
    gmresP->mwork_alloc[0]  = 5;
    gmresP->nwork_alloc     = 1;
    for (k=0; k<gmresP->vv_allocated; k++)
	gmresP->vecs[k] = gmresP->user_work[0][k];
  }
  return 0;
}
/* 
    This routine computes the initial residual without making any assumptions
    about the solution.
 */
static int GMRESResidual(  KSP itP,int restart )
{
  KSP_GMRES    *gmresP = (KSP_GMRES *)(itP->data);
  Scalar       mone = -1.0;
  Mat          Amat, Pmat;
  MatStructure pflag;
  int          ierr;

  ierr = PCGetOperators(itP->B,&Amat,&Pmat,&pflag); CHKERRQ(ierr);
  /* compute initial residual: f - M*x */
  /* (inv(b)*a)*x or (a*inv(b)*b)*x into dest */
  if (itP->pc_side == PC_RIGHT) {
    /* we want a * binv * b * x, or just a * x for the first step */
    /* a*x into temp */
    ierr = MatMult(Amat,VEC_SOLN,VEC_TEMP ); CHKERRQ(ierr);
  }
  else {
    /* else we do binv * a * x */
    ierr = PCApplyBAorAB(itP->B,itP->pc_side,VEC_SOLN,VEC_TEMP,
                         VEC_TEMP_MATOP ); CHKERRQ(ierr);
  }
  /* This is an extra copy for the right-inverse case */
  ierr = VecCopy( VEC_BINVF, VEC_VV(0) ); CHKERRQ(ierr);
  ierr = VecAXPY( &mone, VEC_TEMP, VEC_VV(0) ); CHKERRQ(ierr);
      /* inv(b)(f - a*x) into dest */
  return 0;
}
/*
    Run gmres, possibly with restart.  Return residual history if requested.
    input parameters:
.        restart - 1 if restarting gmres, 0 otherwise
.	gmresP  - structure containing parameters and work areas
.	itsSoFar- total number of iterations so far (from previous cycles)

    output parameters:
.        nres    - residuals (from preconditioned system) at each step.
                  If restarting, consider passing nres+it.  If null, 
                  ignored
.        itcount - number of iterations used.  nres[0] to nres[itcount]
                  are defined.  If null, ignored.
    Returns:
    0 on success, 1 on failure (did not converge)

    Notes:
    On entry, the value in vector VEC_VV(0) should be the initial residual
    (this allows shortcuts where the initial preconditioned residual is 0).
 */
int GMREScycle(int *  itcount, int itsSoFar,int restart,KSP itP )
{
  double    res_norm, res, rtol;
  Scalar    tmp;
  int       hist_len= itP->res_hist_size, cerr, ierr;
  double    hapbnd,*nres = itP->residual_history,tt;
  /* Note that hapend is ignored in the code */
  int       it, hapend, converged;
  KSP_GMRES *gmresP = (KSP_GMRES *)(itP->data);
  int       max_k = gmresP->max_k, max_it = itP->max_it;

  /* Question: on restart, compute the residual?  No; provide a restart 
     driver */

  it = 0;

  /* dest . dest */
  ierr = VecNorm(VEC_VV(0),NORM_2,&res_norm); CHKERRQ(-ierr);
  res    = res_norm;
  *RS(0) = res_norm;

  /* Do-nothing case: */
  if (res_norm == 0.0) {
    if (itcount) *itcount = 0;
    return 0;
  }
  /* scale VEC_VV (the initial residual) */
  tmp = 1.0/res_norm; ierr = VecScale(&tmp , VEC_VV(0) ); CHKERRQ(-ierr);

  if (!restart) {
    rtol      = itP->rtol * res_norm;
    itP->ttol = (itP->atol > rtol) ? itP->atol : rtol;
  }
  rtol= itP->ttol;
  gmresP->it = (it-1);  /* For converged */
  while (!(converged = cerr = (*itP->converged)(itP,it+itsSoFar,res,itP->cnvP))
           && it < max_k && it + itsSoFar < max_it) {
    if (nres && hist_len > it + itsSoFar) nres[it+itsSoFar]   = res;
    if (itP->monitor) {
	gmresP->it = (it - 1);
        ierr = (*itP->monitor)(itP,it + itsSoFar,res,itP->monP);CHKERRQ(-ierr);
	}
    if (gmresP->vv_allocated <= it + VEC_OFFSET + 1) {
	/* get more vectors */
	ierr = GMRESGetNewVectors(  itP, it+1 );CHKERRQ(-ierr);
	}
    ierr = PCApplyBAorAB(itP->B,itP->pc_side,VEC_VV(it),VEC_VV(1+it),
                         VEC_TEMP_MATOP); CHKERRQ(-ierr);

    /* update hessenberg matrix and do Gram-Schmidt */
    (*gmresP->orthog)(  itP, it );

    /* vv(i+1) . vv(i+1) */
    ierr = VecNorm(VEC_VV(it+1),NORM_2,&tt); CHKERRQ(-ierr);
    /* save the magnitude */
    *HH(it+1,it)    = tt;
    *HES(it+1,it)   = tt;

    /* check for the happy breakdown */
    hapbnd  = gmresP->epsabs * PetscAbsScalar( *HH(it,it) / *RS(it) );
    if (hapbnd > gmresP->haptol) hapbnd = gmresP->haptol;
    if (tt > hapbnd) {
        tmp = 1.0/tt; ierr = VecScale( &tmp, VEC_VV(it+1) ); CHKERRQ(-ierr);
    }
    else {
        /* We SHOULD probably abort the gmres step
           here.  This happens when the solution is exactly reached. */
      hapend = 1;
    }
    ierr = GMRESUpdateHessenberg( itP, it, &res ); CHKERRQ(-ierr);
    it++;
    gmresP->it = (it-1);  /* For converged */
  }
  if (nres && hist_len > it + itsSoFar) nres[it + itsSoFar]   = res; 
  if (nres) 
    itP->res_act_size = (hist_len < it + itsSoFar) ? hist_len : it + itsSoFar + 1;
  if (itP->monitor) {
    gmresP->it = it - 1;
    ierr = (*itP->monitor)( itP,  it + itsSoFar, res, itP->monP ); CHKERRQ(-ierr);
  }
  if (itcount) *itcount    = it;

  /*
    Down here we have to solve for the "best" coefficients of the Krylov
    columns, add the solution values together, and possibly unwind the
    preconditioning from the solution
   */
  if (it == 0) {
    /* exited at the top before doing ANYTHING */
    return 0;
  }

  /* Form the solution (or the solution so far) */
  ierr = BuildGmresSoln(RS(0),VEC_SOLN,VEC_SOLN,itP,it-1); CHKERRQ(-ierr);

  /* Return correct status (Failed on iteration test (failed to converge)) */
  return !converged;
}

static int KSPSolve_GMRES(KSP itP,int *outits )
{
  int       ierr, restart, its, itcount;
  KSP_GMRES *gmresP = (KSP_GMRES *)itP->data;

  restart = 0;
  itcount = 0;
  /* Save binv*f */
  if (itP->pc_side == PC_LEFT) {
    /* inv(b)*f */
    ierr = PCApply(itP->B, VEC_RHS, VEC_BINVF ); CHKERRQ(ierr);
  }
  else if (itP->pc_side == PC_RIGHT) {
    ierr = VecCopy( VEC_RHS, VEC_BINVF ); CHKERRQ(ierr);
  }
  /* Compute the initial (preconditioned) residual */
  if (!itP->guess_zero) {
    if ((ierr=GMRESResidual(  itP, restart ))) return ierr;
  }
  else {
    ierr = VecCopy( VEC_BINVF, VEC_VV(0) ); CHKERRQ(ierr);
  }
    
  while ((ierr = GMREScycle(  &its, itcount, restart, itP ))) {
    if (ierr < 0) SETERRQ(1,0);
    restart = 1;
    itcount += its;
    if ((ierr = GMRESResidual(  itP, restart ))) return ierr;
    if (itcount > itP->max_it) break;
    /* need another check to make sure that gmres breaks out 
       at precisely the number of iterations chosen */
  }
  itcount += its;      /* add in last call to GMREScycle */
  *outits = itcount;  return 0;
}

static int KSPAdjustWork_GMRES(KSP itP )
{
  KSP_GMRES *gmresP;
  int       i,ierr;

  if ( itP->adjust_work_vectors ) {
    gmresP = (KSP_GMRES *) itP->data;
    for (i=0; i<gmresP->vv_allocated; i++) {
      ierr = (*itP->adjust_work_vectors)(itP,gmresP->user_work[i],gmresP->mwork_alloc[i]); 
      CHKERRQ(ierr);
    }  
  }
  return 0;
}

static int KSPDestroy_GMRES(PetscObject obj)
{
  KSP       itP = (KSP) obj;
  KSP_GMRES *gmresP = (KSP_GMRES *) itP->data;
  int       i;

  /* Free the Hessenberg matrix */
  if (gmresP->hh_origin) PetscFree( gmresP->hh_origin );

  /* Free the pointer to user variables */
  if (gmresP->vecs) PetscFree( gmresP->vecs );

  /* free work vectors */
  for (i=0; i<gmresP->nwork_alloc; i++) {
    VecDestroyVecs(gmresP->user_work[i], gmresP->mwork_alloc[i] );
  }
  if (gmresP->user_work)  PetscFree( gmresP->user_work );
  if (gmresP->mwork_alloc) PetscFree( gmresP->mwork_alloc );
  if (gmresP->nrs) PetscFree( gmresP->nrs );
  if (gmresP->sol_temp) VecDestroy(gmresP->sol_temp);
  PetscFree( gmresP ); 
  return 0;
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
static int BuildGmresSoln(Scalar* nrs,Vec vs,Vec vdest,KSP itP, int it )
{
  Scalar    tt, zero = 0.0, one = 1.0;
  int       ierr, ii, k, j;
  KSP_GMRES *gmresP = (KSP_GMRES *)(itP->data);

  /* Solve for solution vector that minimizes the residual */

  /* If it is < 0, no gmres steps have been performed */
  if (it < 0) {
    if (vdest != vs) {
      ierr = VecCopy( vs, vdest ); CHKERRQ(ierr);
    }
    return 0;
  }
  nrs[it] = *RS(it) / *HH(it,it);
  for (ii=1; ii<=it; ii++) {
    k   = it - ii;
    tt  = *RS(k);
    for (j=k+1; j<=it; j++) tt  = tt - *HH(k,j) * nrs[j];
    nrs[k]   = tt / *HH(k,k);
  }

  /* Accumulate the correction to the solution of the preconditioned problem
    in TEMP */
  ierr = VecSet( &zero, VEC_TEMP ); CHKERRQ(ierr);
  ierr = VecMAXPY(it+1, nrs, VEC_TEMP, &VEC_VV(0) ); CHKERRQ(ierr);

  /* If we preconditioned on the right, we need to solve for the correction to
     the unpreconditioned problem */
  if (itP->pc_side == PC_RIGHT) {
    if (vdest != vs) {
      ierr = PCApply(itP->B, VEC_TEMP, vdest ); CHKERRQ(ierr);
      ierr = VecAXPY( &one, vs, vdest ); CHKERRQ(ierr);
    }
    else {
      ierr = PCApply(itP->B,VEC_TEMP,VEC_TEMP_MATOP); CHKERRQ(ierr);
      ierr = VecAXPY(&one,VEC_TEMP_MATOP,vdest); CHKERRQ(ierr);
    }
  }
  else if (itP->pc_side == PC_LEFT) {
    if (vdest != vs) {
      ierr = VecCopy( VEC_TEMP, vdest ); CHKERRQ(ierr);
      ierr = VecAXPY( &one, vs, vdest ); CHKERRQ(ierr);
    }
    else {
      ierr = VecAXPY( &one, VEC_TEMP, vdest ); CHKERRQ(ierr);
    }
  }
  return 0;
}
/*
   Do the scalar work for the orthogonalization.  Return new residual.
 */
static int GMRESUpdateHessenberg( KSP itP, int it, double *res )
{
  Scalar    *hh, *cc, *ss, tt;
  int       j;
  KSP_GMRES *gmresP = (KSP_GMRES *)(itP->data);

  hh  = HH(0,it);
  cc  = CC(0);
  ss  = SS(0);

  /* Apply all the previously computed plane rotations to the new column
     of the Hessenberg matrix */
  for (j=1; j<=it; j++) {
    tt  = *hh;
    *hh = *cc * tt + *ss * *(hh+1);
    hh++;
    *hh = *cc++ * *hh - ( *ss++ * tt );
  }

  /*
    compute the new plane rotation, and apply it to:
     1) the right hand side of the Hessenberg system
     2) the new column of the Hessenberg matrix
    thus obtaining the updated value of the residual
  */
  tt        = sqrt( *hh * *hh + *(hh+1) * *(hh+1) );
  if (tt == 0.0) {SETERRQ(1,"KSPSolve_GMRES:bad A or B operator, are you sure it is !0?");}
  *cc       = *hh / tt;
  *ss       = *(hh+1) / tt;
  *RS(it+1) = - ( *ss * *RS(it) );
  *RS(it)   = *cc * *RS(it);
  *hh       = *cc * *hh + *ss * *(hh+1);
  *res = PetscAbsScalar( *RS(it+1) );
  return 0;
}
/*
   This routine allocates more work vectors, starting from VEC_VV(it).
 */
static int GMRESGetNewVectors( KSP itP,int it )
{
  KSP_GMRES *gmresP = (KSP_GMRES *)itP->data;
  int       nwork = gmresP->nwork_alloc,k, nalloc;

  nalloc = gmresP->delta_allocate;
  /* Adjust the number to allocate to make sure that we don't exceed the
    number of available slots */
  if (it + VEC_OFFSET + nalloc >= gmresP->vecs_allocated)
      nalloc = gmresP->vecs_allocated - it - VEC_OFFSET;
  /* CHKPTRQ(nalloc); */
  if (nalloc == 0) return 0;

  gmresP->vv_allocated += nalloc;
  VecDuplicateVecs(itP->vec_rhs, nalloc,&gmresP->user_work[nwork] );
  PLogObjectParents(itP,nalloc,gmresP->user_work[nwork]);CHKPTRQ(gmresP->user_work[nwork]);
  gmresP->mwork_alloc[nwork] = nalloc;
  for (k=0; k<nalloc; k++) {
    gmresP->vecs[it+VEC_OFFSET+k] = gmresP->user_work[nwork][k];
  }
  gmresP->nwork_alloc++;
  return 0;
}

/*@
    KSPGMRESSetRestart - Sets the number of search directions 
    for GMRES before restart.

    Input Parameters:
.   itP - the iterative context
.   max_k - the number of directions

    Options Database Key:
$   -ksp_gmres_restart  max_k

    Note:
    The default value of max_k = 10.

.keywords: GMRES, set, restart

.seealso: KSPGMRESSetUseUnmodifiedGramSchmidt()
@*/
int KSPGMRESSetRestart(KSP itP,int max_k )
{
  KSP_GMRES *gmresP;
  PETSCVALIDHEADERSPECIFIC(itP,KSP_COOKIE);
  gmresP = (KSP_GMRES *)itP->data;
  if (itP->type != KSPGMRES) return 0;
  gmresP->max_k = max_k;
  return 0;
}

int KSPDefaultConverged_GMRES(KSP itP,int n,double rnorm,void *dummy)
{
  if ( rnorm <= itP->ttol ) return(1);
  else return(0);
}

static int KSPBuildSolution_GMRES(KSP itP,Vec  ptr,Vec *result )
{
  KSP_GMRES *gmresP = (KSP_GMRES *)itP->data; 
  int       ierr;

  if (ptr == 0) {
    if (!gmresP->sol_temp) {
      ierr = VecDuplicate(itP->vec_sol,&gmresP->sol_temp); CHKERRQ(ierr);
      PLogObjectParent(itP,gmresP->sol_temp);
    }
    ptr = gmresP->sol_temp;
  }
  if (!gmresP->nrs) {
    /* allocate the work area */
    gmresP->nrs = (Scalar *)PetscMalloc((unsigned)(gmresP->max_k*sizeof(Scalar)));
    PLogObjectMemory(itP,gmresP->max_k*sizeof(Scalar));
  }

  ierr = BuildGmresSoln(gmresP->nrs,VEC_SOLN,ptr,itP,gmresP->it); CHKERRQ(ierr);
  *result = ptr; return 0;
}

/*@C
  KSPGMRESSetOrthogRoutine - Sets the orthogonalization routine used by GMRES.

  Input Parameters:
.   itP   - iterative context obtained from KSPCreate
.   fcn   - Orthogonalization function.  

  Notes:
  The functions KSPGMRESBasicOrthog and KSPGMRESUnmodifiedOrthog are predefined.
  The default is KSPGMRESBasicOrthog; KSPGMRESUnmodifiedOrthog is a simple 
  Gram-Schmidt (NOT modified Gram-Schmidt).  The KSPGMRESUnmodifiedOrthog is 
  NOT recommended; however, for some problems, particularly when using 
  parallel distributed vectors, this may be significantly faster.

  The routine KSPGMRESIROrthog is an interative refinement version of 
  KSPGMRESUnmodifiedOrthog.  It may be more numerically stable than
  KSPGMRESUnmodifiedOrthog on parallel systems.  
@*/
int KSPGMRESSetOrthogRoutine( KSP itP,int (*fcn)(KSP,int) )
{
  PETSCVALIDHEADERSPECIFIC(itP,KSP_COOKIE);
  if (itP->type == KSPGMRES) {
    ((KSP_GMRES *)itP->data)->orthog = fcn;
  }
  return 0;
}

static int KSPView_GMRES(PetscObject obj,Viewer viewer)
{
  KSP       itP = (KSP)obj;
  KSP_GMRES *gmresP = (KSP_GMRES *)itP->data; 
  FILE      *fd;
  char      *cstring;
  int       ierr;

  ierr = ViewerFileGetPointer(viewer,&fd); CHKERRQ(ierr);

  if (gmresP->orthog == KSPGMRESUnmodifiedOrthog) 
    cstring = "GMRESUnmodifiedOrthog";
  else if (gmresP->orthog == KSPGMRESBasicOrthog) 
    cstring = "GMRESBasicOrthog";
  else if (gmresP->orthog == KSPGMRESIROrthog) 
    cstring = "GMRESIROrthog";
  else 
    cstring = "unknown";
  MPIU_fprintf(itP->comm,fd,"    GMRES: restart=%d, orthogonalization routine is %s\n",
               gmresP->max_k,cstring);
  return 0;
}

int KSPCreate_GMRES(KSP itP)
{
  KSP_GMRES *gmresP;

  gmresP = (KSP_GMRES*) PetscMalloc(sizeof(KSP_GMRES)); CHKPTRQ(gmresP);
  PetscMemzero(gmresP,sizeof(KSP_GMRES));
  PLogObjectMemory(itP,sizeof(KSP_GMRES));
  itP->data              = (void *) gmresP;
  itP->type              = KSPGMRES;
  itP->converged         = KSPDefaultConverged_GMRES;
  itP->buildsolution     = KSPBuildSolution_GMRES;

  itP->setup             = KSPSetUp_GMRES;
  itP->solver            = KSPSolve_GMRES;
  itP->adjustwork        = KSPAdjustWork_GMRES;
  itP->destroy           = KSPDestroy_GMRES;
  itP->view              = KSPView_GMRES;

  gmresP->haptol         = 1.0e-8;
  gmresP->epsabs         = 1.0e-8;
  gmresP->q_preallocate  = 0;
  gmresP->delta_allocate = GMRES_DELTA_DIRECTIONS;
  gmresP->orthog         = KSPGMRESBasicOrthog;
  gmresP->nrs            = 0;
  gmresP->sol_temp       = 0;
  gmresP->max_k          = GMRES_DEFAULT_MAXK;
  return 0;
}

/*@
     KSPGMRESSetPreAllocateVectors - Causes GMRES to preallocate all its
         needed work vectors at initial setup rather then the default which 
         is to allocate them in chunks when needed.

  Input Paramter:
.    ksp - the Krylov subspace context

@*/
int KSPGMRESSetPreAllocateVectors(KSP ksp)
{
  KSP_GMRES *gmresP;

  if (ksp->type != KSPGMRES) return 0;
  gmresP = (KSP_GMRES *)ksp->data;
  gmresP->q_preallocate = 1;
  return 0;
}

