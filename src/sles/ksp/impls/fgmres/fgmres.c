/* $Id: fgmres.c,v 1.12 2000/02/02 20:09:53 bsmith Exp bsmith $ */

/*
    This file implements FGMRES (a Generalized Minimal Residual) method.  
    Reference:  Saad, 1993.

    Preconditioning:  It the preconditioner is constant then this fgmres
    code is equivalent to RIGHT-PRECONDITIONED GMRES.

    Restarts:  Restarts are basically solves with x0 not equal to zero.
 
       Contributed by Allison Baker

*/

#include "src/sles/ksp/impls/fgmres/fgmresp.h"       /*I  "ksp.h"  I*/
#define FGMRES_DELTA_DIRECTIONS 10
#define FGMRES_DEFAULT_MAXK     30
static int    FGMRESGetNewVectors(KSP,int);
static int    FGMRESUpdateHessenberg(KSP,int,PetscTruth,PetscReal *);
static int    BuildFgmresSoln(Scalar*,Vec,Vec,KSP,int);

/*

    KSPSetUp_FGMRES - Sets up the workspace needed by fgmres.

    This is called once, usually automatically by SLESSolve() or SLESSetUp(),
    but can be called directly by KSPSetUp().

*/
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"KSPSetUp_FGMRES"
int    KSPSetUp_FGMRES(KSP ksp)
{
  unsigned  int size,hh,hes,rs,cc;
  int           ierr,max_k,k;
  KSP_FGMRES    *fgmres = (KSP_FGMRES *)ksp->data;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(2,0,"no symmetric preconditioning for KSPFGMRES");
  }
  max_k         = fgmres->max_k;
  hh            = (max_k + 2) * (max_k + 1);
  hes           = (max_k + 1) * (max_k + 1);
  rs            = (max_k + 2);
  cc            = (max_k + 1);  /* SS and CC are the same size */
  size          = (hh + hes + rs + 2*cc) * sizeof(Scalar);

  /* Allocate space and set pointers to beginning */
  fgmres->hh_origin  = (Scalar*)PetscMalloc(size);CHKPTRQ(fgmres->hh_origin);
  PLogObjectMemory(ksp,size);                      /* HH - modified (by plane 
                                                      rotations) hessenburg */
  fgmres->hes_origin = fgmres->hh_origin + hh;     /* HES - unmodified hessenburg */
  fgmres->rs_origin  = fgmres->hes_origin + hes;   /* RS - the right-hand-side of the 
                                                      Hessenberg system */
  fgmres->cc_origin  = fgmres->rs_origin + rs;     /* CC - cosines for rotations */
  fgmres->ss_origin  = fgmres->cc_origin + cc;     /* SS - sines for rotations */

  if (ksp->calc_sings) {
    /* Allocate workspace to hold Hessenberg matrix needed by Eispack */
    size = (max_k + 3)*(max_k + 9)*sizeof(Scalar);
    fgmres->Rsvd = (Scalar*)PetscMalloc(size);CHKPTRQ(fgmres->Rsvd);
    fgmres->Dsvd = (PetscReal*)PetscMalloc(5*(max_k+2)*sizeof(PetscReal));CHKPTRQ(fgmres->Dsvd);
    PLogObjectMemory(ksp,size+5*(max_k+2)*sizeof(PetscReal));
  }

  /* Allocate array to hold pointers to user vectors.  Note that we need
   4 + max_k + 1 (since we need it+1 vectors, and it <= max_k) */
  fgmres->vecs = (Vec*)PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void *));CHKPTRQ(fgmres->vecs);
  fgmres->vecs_allocated = VEC_OFFSET + 2 + max_k;
  fgmres->user_work   = (Vec **)PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void *));CHKPTRQ(fgmres->user_work);
  fgmres->mwork_alloc = (int*)PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(int));CHKPTRQ(fgmres->mwork_alloc);
  PLogObjectMemory(ksp,(VEC_OFFSET+2+max_k)*(2*sizeof(void *)+sizeof(int)));

  /* New for FGMRES - Allocate array to hold pointers to preconditioned 
     vectors - same sizes as user vectors above */
  fgmres->prevecs = (Vec*)PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void *));CHKPTRQ(fgmres->prevecs);
  fgmres->prevecs_user_work  = (Vec **)PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void *));CHKPTRQ(fgmres->prevecs_user_work);
  PLogObjectMemory(ksp,(VEC_OFFSET+2+max_k)*(2*sizeof(void *)));


  /* if q_preallocate = 0 then only allocate one "chunck" of space (for 
     5 vectors) - additional will then be allocated from FGMREScycle() 
     as needed.  Otherwise, allocate all of the space that could be needed */
  if (fgmres->q_preallocate) {
    fgmres->vv_allocated   = VEC_OFFSET + 2 + max_k;
  } else {
    fgmres->vv_allocated    = 5;
  }

  /* space for work vectors */
  ierr = VecDuplicateVecs(VEC_RHS,fgmres->vv_allocated,&fgmres->user_work[0]);CHKERRQ(ierr);
  PLogObjectParents(ksp,fgmres->vv_allocated,fgmres->user_work[0]);
  for (k=0; k < fgmres->vv_allocated; k++) {
    fgmres->vecs[k] = fgmres->user_work[0][k];
  } 

  /* space for preconditioned vectors */
  ierr = VecDuplicateVecs(VEC_RHS,fgmres->vv_allocated,&fgmres->prevecs_user_work[0]);CHKERRQ(ierr);
  PLogObjectParents(ksp,fgmres->vv_allocated,fgmres->prevecs_user_work[0]);
  for (k=0; k < fgmres->vv_allocated; k++) {
    fgmres->prevecs[k] = fgmres->prevecs_user_work[0][k];
  } 

  /* specify how many work vectors have been allocated in this 
     chunck" (the first one) */
  fgmres->mwork_alloc[0] = fgmres->vv_allocated;
  fgmres->nwork_alloc    = 1;

  PetscFunctionReturn(0);
}

/* 

    FGMRESResidual - This routine computes the initial residual (NOT 
                     PRECONDITIONED) without
                     making any assumptions about the solution.  

*/
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"FGMRESResidual"
static int FGMRESResidual(KSP ksp)
{
  KSP_FGMRES   *fgmres = (KSP_FGMRES *)(ksp->data);
  Scalar       mone = -1.0;
  Mat          Amat,Pmat;
  MatStructure pflag;
  int          ierr;

  PetscFunctionBegin;
  ierr = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);CHKERRQ(ierr);

  /* put A*x into VEC_TEMP */
  ierr = MatMult(Amat,VEC_SOLN,VEC_TEMP);CHKERRQ(ierr);
  /* now put residual (-a*x + f) into vec_vv(0) */
  ierr = VecWAXPY(&mone,VEC_TEMP,VEC_RHS,VEC_VV(0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*

    FGMRESCycle - Run fgmres, possibly with restart.  Return residual 
                  history if requested.

    input parameters:
.	 fgmres  - structure containing parameters and work areas

    output parameters:
.        itcount - number of iterations used.  If null, ignored.
.        converged - 0 if not converged

		  
    Notes:
    On entry, the value in vector VEC_VV(0) should be 
    the initial residual.


 */
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"FGMREScycle"
int FGMREScycle(int *itcount,KSP ksp)
{

  KSP_FGMRES   *fgmres = (KSP_FGMRES *)(ksp->data);
  PetscReal    res_norm;             
  PetscReal    hapbnd,tt;
  Scalar       zero = 0.0;
  Scalar       tmp;
  PetscTruth   hapend = PETSC_FALSE;  /* indicates happy breakdown ending */
  int          ierr;
  int          loc_it;                /* local count of # of dir. in Krylov space */ 
  int          max_k = fgmres->max_k; /* max # of directions Krylov space */
  int          max_it = ksp->max_it;  /* max # of overall iterations for the method */ 
  Mat          Amat,Pmat;
  MatStructure pflag;

  PetscFunctionBegin;

  /* Number of pseudo iterations since last restart is the number 
     of prestart directions */
  loc_it = 0;

  /* initial residual is in VEC_VV(0)  - compute its norm*/ 
  ierr   = VecNorm(VEC_VV(0),NORM_2,&res_norm);CHKERRQ(ierr);

  /* first entry in right-hand-side of hessenberg system is just 
     the initial residual norm */
  *RS(0) = res_norm;

  /* check for the convergence - maybe the current guess is good enough */
  ierr = (*ksp->converged)(ksp,ksp->its,res_norm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) {
    if (itcount) *itcount = 0;
    PetscFunctionReturn(0);
  }

  /* scale VEC_VV (the initial residual) */
  tmp = 1.0/res_norm; ierr = VecScale(&tmp,VEC_VV(0));CHKERRQ(ierr);

  /* FYI: AMS calls are for memory snooper */
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->rnorm = res_norm;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);


  /* note: (fgmres->it) is always set one less than (loc_it) It is used in 
     KSPBUILDSolution_FGMRES, where it is passed to BuildFGmresSoln.  
     Note that when BuildFGmresSoln is called from this function, 
     (loc_it -1) is passed, so the two are equivalent */
  fgmres->it = (loc_it - 1);
   
  /* MAIN ITERATION LOOP BEGINNING*/
  /* keep iterating until we have converged OR generated the max number
     of directions OR reached the max number of iterations for the method */ 
  ierr = (*ksp->converged)(ksp,ksp->its,res_norm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  while (!ksp->reason && loc_it < max_k && ksp->its < max_it) {
    KSPLogResidualHistory(ksp,res_norm);
    fgmres->it = (loc_it - 1);
    KSPMonitor(ksp,ksp->its,res_norm); 

    /* see if more space is needed for work vectors */
    if (fgmres->vv_allocated <= loc_it + VEC_OFFSET + 1) {
      ierr = FGMRESGetNewVectors(ksp,loc_it+1);CHKERRQ(ierr);
      /* (loc_it+1) is passed in as number of the first vector that should
         be allocated */
    }

    /* CHANGE THE PRECONDITIONER? */ 
    /* ModifyPC is the callback function that can be used to
       change the PC or its attributes before its applied */
    (*fgmres->modifypc)(ksp,ksp->its,loc_it,res_norm,fgmres->modifyctx);
   
  
    /* apply PRECONDITIONER to direction vector and store with 
       preconditioned vectors in prevec */
    ierr = PCApply(ksp->B,VEC_VV(loc_it),PREVEC(loc_it));CHKERRQ(ierr);
     
    ierr = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);CHKERRQ(ierr);
    /* Multiply preconditioned vector by operator - put in VEC_VV(loc_it+1) */
    ierr = MatMult(Amat,PREVEC(loc_it),VEC_VV(1+loc_it));CHKERRQ(ierr);

 
    /* update hessenberg matrix and do Gram-Schmidt - new direction is in
       VEC_VV(1+loc_it)*/
    ierr = (*fgmres->orthog)(ksp,loc_it);CHKERRQ(ierr);

    /* new entry in hessenburg is the 2-norm of our new direction */
    ierr = VecNorm(VEC_VV(loc_it+1),NORM_2,&tt);CHKERRQ(ierr);
    *HH(loc_it+1,loc_it)   = tt;
    *HES(loc_it+1,loc_it)  = tt;

    /* Happy Breakdown Check */
    hapbnd  = fgmres->epsabs * PetscAbsScalar(*HH(loc_it,loc_it) / *RS(loc_it));
    /* RS(loc_it) contains the res_norm from the last iteration  */
    hapbnd = PetscMin(fgmres->haptol,hapbnd);
    /*if (hapbnd > fgmres->haptol) hapbnd = fgmres->haptol;*/
    if (tt > hapbnd) {
        tmp = 1.0/tt; 
        /* scale new direction by its norm */
        ierr = VecScale(&tmp,VEC_VV(loc_it+1));CHKERRQ(ierr);
    } else {
        /* This happens when the solution is exactly reached. */
        /* So there is no new direction... */
          ierr   = VecSet(&zero,VEC_TEMP);CHKERRQ(ierr); /* set VEC_TEMP to 0 */
          hapend = PETSC_TRUE;
    }
    /* note that for FGMRES we could get HES(loc_it+1, loc_it)  = 0 and the
       current solution would not be exact if HES was singular.  Note that 
       HH non-singular implies that HES is no singular, and HES is guaranteed
       to be nonsingular when PREVECS are linearly independent and A is 
       nonsingular (in GMRES, the nonsingularity of A implies the nonsingularity 
       of HES). So we should really add a check to verify that HES is nonsingular.*/

 
    /* Now apply rotations to new col of hessenberg (and right side of system), 
       calculate new rotation, and get new residual norm at the same time*/
    ierr = FGMRESUpdateHessenberg(ksp,loc_it,hapend,&res_norm);CHKERRQ(ierr);
    loc_it++;
    fgmres->it  = (loc_it-1);  /* Add this here in case it has converged */
 
    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = res_norm;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);

    ierr = (*ksp->converged)(ksp,ksp->its,res_norm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);

    /* Catch error in happy breakdown and signal convergence and break from loop */
    if (hapend) {
      if (!ksp->reason) {
        SETERRQ(0,0,"You reached the happy break down,but convergence was not indicated.");
      }
      break;
    }
  }
  /* END OF ITERATION LOOP */

  KSPLogResidualHistory(ksp,res_norm);

  /*
     Monitor if we know that we will not return for a restart */
  if (ksp->reason || ksp->its >= max_it) {
    KSPMonitor(ksp,ksp->its,res_norm);
  }

  if (itcount) *itcount    = loc_it;

  /*
    Down here we have to solve for the "best" coefficients of the Krylov
    columns, add the solution values together, and possibly unwind the
    preconditioning from the solution
   */
 
  /* Form the solution (or the solution so far) */
  /* Note: must pass in (loc_it-1) for iteration count so that BuildFgmresSoln
     properly navigates */

  ierr = BuildFgmresSoln(RS(0),VEC_SOLN,VEC_SOLN,ksp,loc_it-1);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*  
    KSPSolve_FGMRES - This routine applies the FGMRES method.


   Input Parameter:
.     ksp - the Krylov space object that was set to use fgmres

   Output Parameter:
.     outits - number of iterations used

*/
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"KSPSolve_FGMRES"

int KSPSolve_FGMRES(KSP ksp,int *outits)
{
  int        ierr;
  int        cycle_its; /* iterations done in a call to FGMREScycle */
  int        itcount;   /* running total of iterations, incl. those in restarts */
  KSP_FGMRES *fgmres = (KSP_FGMRES *)ksp->data;

  PetscFunctionBegin;

  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its = 0;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);

  /* initialize */
  itcount  = 0;

  /* Compute the initial (NOT preconditioned) residual */
  if (!ksp->guess_zero) {
    ierr = FGMRESResidual(ksp);CHKERRQ(ierr);
  } else { /* guess is 0 so residual is F (which is in VEC_RHS) */
    ierr = VecCopy(VEC_RHS,VEC_VV(0));CHKERRQ(ierr);
  }
  /* now the residual is in VEC_VV(0) - which is what 
     FGMREScycle expects... */
  
  ierr    = FGMREScycle(&cycle_its,ksp);CHKERRQ(ierr);
  itcount += cycle_its;
  while (!ksp->reason) {
    ierr     = FGMRESResidual(ksp);CHKERRQ(ierr);
    if (itcount >= ksp->max_it) break;
    /* need another check to make sure that fgmres breaks out 
       at precisely the number of iterations chosen */
    ierr     = FGMREScycle(&cycle_its,ksp);CHKERRQ(ierr);
    itcount += cycle_its;  
  }
  /* mark lack of convergence with negative the number of iterations */
  if (itcount >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;

  *outits = itcount;
  PetscFunctionReturn(0);
}

/*

   KSPDestroy_FGMRES - Frees all memory space used by the Krylov method.

*/
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"KSPDestroy_FGMRES" 
int KSPDestroy_FGMRES(KSP ksp)
{
  KSP_FGMRES *fgmres = (KSP_FGMRES*)ksp->data;
  int       i,ierr;

  PetscFunctionBegin;
  /* Free the Hessenberg matrices */
  if (fgmres->hh_origin) {ierr = PetscFree(fgmres->hh_origin);CHKERRQ(ierr);}

  /* Free pointers to user variables */
  if (fgmres->vecs) {ierr = PetscFree(fgmres->vecs);CHKERRQ(ierr);}
  if (fgmres->prevecs) {ierr = PetscFree (fgmres->prevecs);CHKERRQ(ierr);}

  /* free work vectors */
  for (i=0; i < fgmres->nwork_alloc; i++) {
    ierr = VecDestroyVecs(fgmres->user_work[i],fgmres->mwork_alloc[i]);CHKERRQ(ierr);
  }
  if (fgmres->user_work)  {ierr = PetscFree(fgmres->user_work);CHKERRQ(ierr);}

  for (i=0; i < fgmres->nwork_alloc; i++) {
    ierr = VecDestroyVecs(fgmres->prevecs_user_work[i],fgmres->mwork_alloc[i]);CHKERRQ(ierr);
  }
  if (fgmres->prevecs_user_work) {ierr = PetscFree(fgmres->prevecs_user_work);CHKERRQ(ierr);}

  if (fgmres->mwork_alloc) {ierr = PetscFree(fgmres->mwork_alloc);CHKERRQ(ierr);}
  if (fgmres->nrs) {ierr = PetscFree(fgmres->nrs);CHKERRQ(ierr);}
  if (fgmres->sol_temp) {ierr = VecDestroy(fgmres->sol_temp);CHKERRQ(ierr);}
  if (fgmres->Rsvd) {ierr = PetscFree(fgmres->Rsvd);CHKERRQ(ierr);}
  if (fgmres->Dsvd) {ierr = PetscFree(fgmres->Dsvd);CHKERRQ(ierr);}
  if (fgmres->modifydestroy) {
    ierr = (*fgmres->modifydestroy)(fgmres->modifyctx);CHKERRQ(ierr);
  }
  ierr = PetscFree(fgmres);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    BuildFgmresSoln - create the solution from the starting vector and the
                      current iterates.

    Input parameters:
        nrs - work area of size it + 1.
	vguess  - index of initial guess
	vdest - index of result.  Note that vguess may == vdest (replace
	        guess with the solution).
        it - HH upper triangular part is a block of size (it+1) x (it+1)  

     This is an internal routine that knows about the FGMRES internals.
 */
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"BuildFgmresSoln"
static int BuildFgmresSoln(Scalar* nrs,Vec vguess,Vec vdest,KSP ksp,int it)
{
  Scalar     tt,zero = 0.0,one = 1.0;
  int        ierr,ii,k,j;
  KSP_FGMRES *fgmres = (KSP_FGMRES *)(ksp->data);

  PetscFunctionBegin;
  /* Solve for solution vector that minimizes the residual */

  /* If it is < 0, no fgmres steps have been performed */
  if (it < 0) {
    if (vdest != vguess) {
      ierr = VecCopy(vguess,vdest);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

  /* so fgmres steps HAVE been performed */

  /* solve the upper triangular system - RS is the right side and HH is 
     the upper triangular matrix  - put soln in nrs */
  nrs[it] = *RS(it) / *HH(it,it);
  for (ii=1; ii<=it; ii++) {
    k   = it - ii;
    tt  = *RS(k);
    for (j=k+1; j<=it; j++) tt  = tt - *HH(k,j) * nrs[j];
    nrs[k]   = tt / *HH(k,k);
  }

  /* Accumulate the correction to the soln of the preconditioned prob. in 
     VEC_TEMP - note that we use the preconditioned vectors  */
  ierr = VecSet(&zero,VEC_TEMP);CHKERRQ(ierr); /* set VEC_TEMP components to 0 */
  ierr = VecMAXPY(it+1,nrs,VEC_TEMP,&PREVEC(0));CHKERRQ(ierr); 

  /* put updated solution into vdest.*/
  if (vdest != vguess) {
    ierr = VecCopy(VEC_TEMP,vdest);CHKERRQ(ierr);
    ierr = VecAXPY(&one,vguess,vdest);CHKERRQ(ierr);
  } else  {/* replace guess with solution */
    ierr = VecAXPY(&one,VEC_TEMP,vdest);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*

    FGMRESUpdateHessenberg - Do the scalar work for the orthogonalization.  
                            Return new residual.

    input parameters:

.        ksp -    Krylov space object
.	 it  -    plane rotations are applied to the (it+1)th column of the 
                  modified hessenberg (i.e. HH(:,it))
.        hapend - PETSC_FALSE not happy breakdown ending.

    output parameters:
.        res - the new residual
	
 */
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"FGMRESUpdateHessenberg"
static int FGMRESUpdateHessenberg(KSP ksp,int it,PetscTruth hapend,PetscReal *res)
{
  Scalar     *hh,*cc,*ss,tt;
  int        j;
  KSP_FGMRES *fgmres = (KSP_FGMRES *)(ksp->data);

  PetscFunctionBegin;
  hh  = HH(0,it);  /* pointer to beginning of column to update - so 
                      incrementing hh "steps down" the (it+1)th col of HH*/ 
  cc  = CC(0);     /* beginning of cosine rotations */ 
  ss  = SS(0);     /* beginning of sine rotations */

  /* Apply all the previously computed plane rotations to the new column
     of the Hessenberg matrix */
  /* Note: this uses the rotation [conj(c)  s ; -s   c], c= cos(theta), s= sin(theta),
     and some refs have [c   s ; -conj(s)  c] (don't be confused!) */

  for (j=1; j<=it; j++) {
    tt  = *hh;
#if defined(PETSC_USE_COMPLEX)
    *hh = PetscConj(*cc) * tt + *ss * *(hh+1);
#else
    *hh = *cc * tt + *ss * *(hh+1);
#endif
    hh++;
    *hh = *cc++ * *hh - (*ss++ * tt);
    /* hh, cc, and ss have all been incremented one by end of loop */
  }

  /*
    compute the new plane rotation, and apply it to:
     1) the right-hand-side of the Hessenberg system (RS)
        note: it affects RS(it) and RS(it+1)
     2) the new column of the Hessenberg matrix
        note: it affects HH(it,it) which is currently pointed to 
        by hh and HH(it+1, it) (*(hh+1))  
    thus obtaining the updated value of the residual...
  */

  /* compute new plane rotation */

  if (!hapend) {
#if defined(PETSC_USE_COMPLEX)
    tt        = PetscSqrtScalar(PetscConj(*hh) * *hh + PetscConj(*(hh+1)) * *(hh+1));
#else
    tt        = PetscSqrtScalar(*hh * *hh + *(hh+1) * *(hh+1));
#endif
    if (tt == 0.0) {SETERRQ(PETSC_ERR_KSP_BRKDWN,0,"Your matrix or preconditioner is the null operator");}
    *cc       = *hh / tt;   /* new cosine value */
    *ss       = *(hh+1) / tt;  /* new sine value */

    /* apply to 1) and 2) */
    *RS(it+1) = - (*ss * *RS(it));
#if defined(PETSC_USE_COMPLEX)
    *RS(it)   = PetscConj(*cc) * *RS(it);
    *hh       = PetscConj(*cc) * *hh + *ss * *(hh+1);
#else
    *RS(it)   = *cc * *RS(it);
    *hh       = *cc * *hh + *ss * *(hh+1);
#endif

    /* residual is the last element (it+1) of right-hand side! */
    *res      = PetscAbsScalar(*RS(it+1));

  } else { /* happy breakdown: HH(it+1, it) = 0, therfore we don't need to apply 
            another rotation matrix (so RH doesn't change).  The new residual is 
            always the new sine term times the residual from last time (RS(it)), 
            but now the new sine rotation would be zero...so the residual should
            be zero...so we will multiply "zero" by the last residual.  This might
            not be exactly what we want to do here -could just return "zero". */
 
    *res = PetscAbsScalar(fgmres->epsabs * *RS(it));
  }
  PetscFunctionReturn(0);
}

/*

   FGMRESGetNewVectors - This routine allocates more work vectors, starting from 
                         VEC_VV(it), and more preconditioned work vectors, starting 
                         from PREVEC(i).

*/
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"FGMRESGetNewVectors" 
static int FGMRESGetNewVectors(KSP ksp,int it)
{
  KSP_FGMRES *fgmres = (KSP_FGMRES *)ksp->data;
  int        nwork = fgmres->nwork_alloc; /* number of work vector chunks allocated */
  int        nalloc;                      /* number to allocate */
  int        k,ierr;
 
  PetscFunctionBegin;
  nalloc = fgmres->delta_allocate; /* number of vectors to allocate 
                                      in a single chunk */

  /* Adjust the number to allocate to make sure that we don't exceed the
     number of available slots (fgmres->vecs_allocated)*/
  if (it + VEC_OFFSET + nalloc >= fgmres->vecs_allocated){
    nalloc = fgmres->vecs_allocated - it - VEC_OFFSET;
  }
  if (!nalloc) PetscFunctionReturn(0);

  fgmres->vv_allocated += nalloc; /* vv_allocated is the number of vectors allocated */

  /* work vectors */
  ierr = VecDuplicateVecs(VEC_RHS,nalloc,&fgmres->user_work[nwork]);CHKERRQ(ierr);
  PLogObjectParents(ksp,nalloc,fgmres->user_work[nwork]); 
  for (k=0; k < nalloc; k++) {
    fgmres->vecs[it+VEC_OFFSET+k] = fgmres->user_work[nwork][k];
  }
  /* specify size of chunk allocated */
  fgmres->mwork_alloc[nwork] = nalloc;

  /* preconditioned vectors */
  ierr = VecDuplicateVecs(VEC_RHS,nalloc,&fgmres->prevecs_user_work[nwork]);CHKERRQ(ierr);
  PLogObjectParents(ksp,nalloc,fgmres->prevecs_user_work[nwork]);CHKPTRQ(fgmres->prevecs_user_work[nwork]);
  for (k=0; k < nalloc; k++) {
    fgmres->prevecs[it+VEC_OFFSET+k] = fgmres->prevecs_user_work[nwork][k];
  } 

  /* increment the number of work vector chunks */
  fgmres->nwork_alloc++;
  PetscFunctionReturn(0);
}

/* 

   KSPBuildSolution_FGMRES

     Input Parameter:
.     ksp - the Krylov space object
.     ptr-

   Output Parameter:
.     result - the solution

   Note: this calls BuildFgmresSoln - the same function that FGMREScycle
   calls directly.  

*/
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"KSPBuildSolution_FGMRES"
int KSPBuildSolution_FGMRES(KSP ksp,Vec ptr,Vec *result)
{
  KSP_FGMRES *fgmres = (KSP_FGMRES *)ksp->data; 
  int        ierr;

  PetscFunctionBegin;
  if (!ptr) {
    if (!fgmres->sol_temp) {
      ierr = VecDuplicate(ksp->vec_sol,&fgmres->sol_temp);CHKERRQ(ierr);
      PLogObjectParent(ksp,fgmres->sol_temp);
    }
    ptr = fgmres->sol_temp;
  }
  if (!fgmres->nrs) {
    /* allocate the work area */
    fgmres->nrs = (Scalar *)PetscMalloc(fgmres->max_k*sizeof(Scalar));CHKPTRQ(fgmres->nrs);
    PLogObjectMemory(ksp,fgmres->max_k*sizeof(Scalar));
  }
 
  ierr = BuildFgmresSoln(fgmres->nrs,VEC_SOLN,ptr,ksp,fgmres->it);CHKERRQ(ierr);
  *result = ptr; 
  
  PetscFunctionReturn(0);
}

/*

   KSPView_FGMRES -Prints information about the current Krylov method 
                  being used.

 */
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"KSPView_FGMRES" 
int KSPView_FGMRES(KSP ksp,Viewer viewer)
{
  KSP_FGMRES   *fgmres = (KSP_FGMRES *)ksp->data; 
  char         *cstr;
  int          ierr;
  PetscTruth   isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (fgmres->orthog == KSPGMRESUnmodifiedGramSchmidtOrthogonalization) {
      cstr = "Unmodified Gram-Schmidt Orthogonalization";
    } else if (fgmres->orthog == KSPGMRESModifiedGramSchmidtOrthogonalization) {
      cstr = "Modified Gram-Schmidt Orthogonalization";
    } else if (fgmres->orthog == KSPGMRESIROrthogonalization) {
      cstr = "Unmodified Gram-Schmidt + 1 step Iterative Refinement Orthogonalization";
    } else {
      cstr = "unknown orthogonalization";
    }
    ierr = ViewerASCIIPrintf(viewer,"  FGMRES: restart=%d, using %s\n",fgmres->max_k,cstr);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }
  PetscFunctionReturn(0);
}

/*

   KSPPrint_Help_FGMRES - Prints a help message that indicates what run time 
                          options are available for this solver

*/
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"KSPPrintHelp_FGMRES"
static int KSPPrintHelp_FGMRES(KSP ksp,char *p)
{
  PetscFunctionBegin;
  (*PetscHelpPrintf)(ksp->comm," Options for FGMRES method:\n");
  (*PetscHelpPrintf)(ksp->comm,"   %sksp_gmres_restart <num>: FGMRES restart, defaults to 30\n",p);
  (*PetscHelpPrintf)(ksp->comm,"   %sksp_mres_unmodifiedgramschmidt: use alternative orthogonalization\n",p);
  (*PetscHelpPrintf)(ksp->comm,"   %sksp_gmres_modifiedgramschmidt: use alternative orthogonalization\n",p);
  (*PetscHelpPrintf)(ksp->comm,"   %sksp_gmres_irorthog: (default) use iterative refinement in orthogonalization\n",p);
  (*PetscHelpPrintf)(ksp->comm,"   %sksp_gmres_preallocate: preallocate FGMRES work vectors\n",p);
  
  (*PetscHelpPrintf)(ksp->comm,"   %sksp_fgmres_modifypcnochange: (default) do not vary the preconditioner\n",p);
  (*PetscHelpPrintf)(ksp->comm,"   %sksp_fgmres_modifypcsles: vary the SLES based preconditioner (example)\n",p);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"KSPSetFromOptions_FGMRES"
int KSPSetFromOptions_FGMRES(KSP ksp)
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
  
  ierr = OptionsHasName(ksp->prefix,"-ksp_fgmres_modifypcnochange",&flg);CHKERRQ(ierr);
  if (flg) {ierr = KSPFGMRESSetModifyPC(ksp,KSPFGMRESModifyPCNoChange,0,0);CHKERRQ(ierr);} 
  ierr = OptionsHasName(ksp->prefix,"-ksp_fgmres_modifypcsles",&flg);CHKERRQ(ierr);
  if (flg) {ierr = KSPFGMRESSetModifyPC(ksp,KSPFGMRESModifyPCSLES,0,0);CHKERRQ(ierr);} 

  PetscFunctionReturn(0);
}

extern int KSPComputeExtremeSingularValues_GMRES(KSP,PetscReal *,PetscReal *);
extern int KSPComputeEigenvalues_GMRES(KSP,int,PetscReal *,PetscReal *,int *);

EXTERN_C_BEGIN
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"KSPFGMRESSetModifyPC_FGMRES" 
int KSPFGMRESSetModifyPC_FGMRES(KSP ksp,int (*fcn)(KSP,int,int,PetscReal,void*),void *ctx,int (*d)(void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ((KSP_FGMRES *)ksp->data)->modifypc      = fcn;
  ((KSP_FGMRES *)ksp->data)->modifydestroy = d;
  ((KSP_FGMRES *)ksp->data)->modifyctx     = ctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
extern int KSPGMRESSetPreAllocateVectors_GMRES(KSP);
extern int KSPGMRESSetRestart_GMRES(KSP,int);
extern int KSPGMRESSetOrthogonalization_GMRES(KSP,int (*)(KSP,int));
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"KSPCreate_FGMRES"
int KSPCreate_FGMRES(KSP ksp)
{
  KSP_FGMRES *fgmres;
  int        ierr;

  PetscFunctionBegin;
  fgmres = (KSP_FGMRES*)PetscMalloc(sizeof(KSP_FGMRES));CHKPTRQ(fgmres);
  PetscMemzero(fgmres,sizeof(KSP_FGMRES));
  PLogObjectMemory(ksp,sizeof(KSP_FGMRES));
  ksp->data                              = (void*)fgmres;
  ksp->ops->buildsolution                = KSPBuildSolution_FGMRES;

  ksp->ops->setup                        = KSPSetUp_FGMRES;
  ksp->ops->solve                        = KSPSolve_FGMRES;
  ksp->ops->destroy                      = KSPDestroy_FGMRES;
  ksp->ops->view                         = KSPView_FGMRES;
  ksp->ops->printhelp                    = KSPPrintHelp_FGMRES;
  ksp->ops->setfromoptions               = KSPSetFromOptions_FGMRES;
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
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPFGMRESSetModifyPC_C",
                                    "KSPFGMRESSetModifyPC_FGMRES",
                                   (void*)KSPFGMRESSetModifyPC_FGMRES);CHKERRQ(ierr);


  fgmres->haptol              = 1.0e-8;
  fgmres->epsabs              = 1.0e-8;
  fgmres->q_preallocate       = 0;
  fgmres->delta_allocate      = FGMRES_DELTA_DIRECTIONS;
  fgmres->orthog              = KSPGMRESIROrthogonalization;
  fgmres->nrs                 = 0;
  fgmres->sol_temp            = 0;
  fgmres->max_k               = FGMRES_DEFAULT_MAXK;
  fgmres->Rsvd                = 0;
  fgmres->modifypc            = KSPFGMRESModifyPCNoChange;
  fgmres->modifyctx           = PETSC_NULL;
  fgmres->modifydestroy       = PETSC_NULL;

  PetscFunctionReturn(0);
}
EXTERN_C_END
