
#include <../src/ksp/ksp/impls/gmres/lgmres/lgmresimpl.h>   /*I petscksp.h I*/

#define LGMRES_DELTA_DIRECTIONS 10
#define LGMRES_DEFAULT_MAXK     30
#define LGMRES_DEFAULT_AUGDIM   2 /*default number of augmentation vectors */
static PetscErrorCode    KSPLGMRESGetNewVectors(KSP,PetscInt);
static PetscErrorCode    KSPLGMRESUpdateHessenberg(KSP,PetscInt,PetscBool,PetscReal*);
static PetscErrorCode    KSPLGMRESBuildSoln(PetscScalar*,Vec,Vec,KSP,PetscInt);

PetscErrorCode  KSPLGMRESSetAugDim(KSP ksp, PetscInt dim)
{
  PetscFunctionBegin;
  CHKERRQ(PetscTryMethod((ksp),"KSPLGMRESSetAugDim_C",(KSP,PetscInt),(ksp,dim)));
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPLGMRESSetConstant(KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(PetscTryMethod((ksp),"KSPLGMRESSetConstant_C",(KSP),(ksp)));
  PetscFunctionReturn(0);
}

/*
    KSPSetUp_LGMRES - Sets up the workspace needed by lgmres.

    This is called once, usually automatically by KSPSolve() or KSPSetUp(),
    but can be called directly by KSPSetUp().

*/
PetscErrorCode    KSPSetUp_LGMRES(KSP ksp)
{
  PetscInt       max_k,k, aug_dim;
  KSP_LGMRES     *lgmres = (KSP_LGMRES*)ksp->data;

  PetscFunctionBegin;
  max_k   = lgmres->max_k;
  aug_dim = lgmres->aug_dim;
  CHKERRQ(KSPSetUp_GMRES(ksp));

  /* need array of pointers to augvecs*/
  CHKERRQ(PetscMalloc1(2*aug_dim + AUG_OFFSET,&lgmres->augvecs));

  lgmres->aug_vecs_allocated = 2 *aug_dim + AUG_OFFSET;

  CHKERRQ(PetscMalloc1(2*aug_dim + AUG_OFFSET,&lgmres->augvecs_user_work));
  CHKERRQ(PetscMalloc1(aug_dim,&lgmres->aug_order));
  CHKERRQ(PetscLogObjectMemory((PetscObject)ksp,(aug_dim)*(4*sizeof(void*) + sizeof(PetscInt)) + AUG_OFFSET*2*sizeof(void*)));

  /*  for now we will preallocate the augvecs - because aug_dim << restart
     ... also keep in mind that we need to keep augvecs from cycle to cycle*/
  lgmres->aug_vv_allocated = 2* aug_dim + AUG_OFFSET;
  lgmres->augwork_alloc    =  2* aug_dim + AUG_OFFSET;

  CHKERRQ(KSPCreateVecs(ksp,lgmres->aug_vv_allocated,&lgmres->augvecs_user_work[0],0,NULL));
  CHKERRQ(PetscMalloc1(max_k+1,&lgmres->hwork));
  CHKERRQ(PetscLogObjectParents(ksp,lgmres->aug_vv_allocated,lgmres->augvecs_user_work[0]));
  for (k=0; k<lgmres->aug_vv_allocated; k++) {
    lgmres->augvecs[k] = lgmres->augvecs_user_work[0][k];
  }
  PetscFunctionReturn(0);
}

/*

    KSPLGMRESCycle - Run lgmres, possibly with restart.  Return residual
                  history if requested.

    input parameters:
.        lgmres  - structure containing parameters and work areas

    output parameters:
.        nres    - residuals (from preconditioned system) at each step.
                  If restarting, consider passing nres+it.  If null,
                  ignored
.        itcount - number of iterations used.   nres[0] to nres[itcount]
                  are defined.  If null, ignored.  If null, ignored.
.        converged - 0 if not converged

    Notes:
    On entry, the value in vector VEC_VV(0) should be
    the initial residual.

 */
PetscErrorCode KSPLGMRESCycle(PetscInt *itcount,KSP ksp)
{
  KSP_LGMRES     *lgmres = (KSP_LGMRES*)(ksp->data);
  PetscReal      res_norm, res;
  PetscReal      hapbnd, tt;
  PetscScalar    tmp;
  PetscBool      hapend = PETSC_FALSE;  /* indicates happy breakdown ending */
  PetscInt       loc_it;                /* local count of # of dir. in Krylov space */
  PetscInt       max_k  = lgmres->max_k; /* max approx space size */
  PetscInt       max_it = ksp->max_it;  /* max # of overall iterations for the method */

  /* LGMRES_MOD - new variables*/
  PetscInt    aug_dim = lgmres->aug_dim;
  PetscInt    spot    = 0;
  PetscInt    order   = 0;
  PetscInt    it_arnoldi;                /* number of arnoldi steps to take */
  PetscInt    it_total;                  /* total number of its to take (=approx space size)*/
  PetscInt    ii, jj;
  PetscReal   tmp_norm;
  PetscScalar inv_tmp_norm;
  PetscScalar *avec;

  PetscFunctionBegin;
  /* Number of pseudo iterations since last restart is the number
     of prestart directions */
  loc_it = 0;

  /* LGMRES_MOD: determine number of arnoldi steps to take */
  /* if approx_constant then we keep the space the same size even if
     we don't have the full number of aug vectors yet*/
  if (lgmres->approx_constant) it_arnoldi = max_k - lgmres->aug_ct;
  else it_arnoldi = max_k - aug_dim;

  it_total =  it_arnoldi + lgmres->aug_ct;

  /* initial residual is in VEC_VV(0)  - compute its norm*/
  CHKERRQ(VecNorm(VEC_VV(0),NORM_2,&res_norm));
  KSPCheckNorm(ksp,res_norm);
  res  = res_norm;

  /* first entry in right-hand-side of hessenberg system is just
     the initial residual norm */
  *GRS(0) = res_norm;

  /* check for the convergence */
  if (!res) {
    if (itcount) *itcount = 0;
    ksp->reason = KSP_CONVERGED_ATOL;
    CHKERRQ(PetscInfo(ksp,"Converged due to zero residual norm on entry\n"));
    PetscFunctionReturn(0);
  }

  /* scale VEC_VV (the initial residual) */
  tmp = 1.0/res_norm; CHKERRQ(VecScale(VEC_VV(0),tmp));

  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = res;
  else ksp->rnorm = 0.0;

  /* note: (lgmres->it) is always set one less than (loc_it) It is used in
     KSPBUILDSolution_LGMRES, where it is passed to KSPLGMRESBuildSoln.
     Note that when KSPLGMRESBuildSoln is called from this function,
     (loc_it -1) is passed, so the two are equivalent */
  lgmres->it = (loc_it - 1);

  /* MAIN ITERATION LOOP BEGINNING*/

  /* keep iterating until we have converged OR generated the max number
     of directions OR reached the max number of iterations for the method */
  CHKERRQ((*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP));

  while (!ksp->reason && loc_it < it_total && ksp->its < max_it) { /* LGMRES_MOD: changed to it_total */
    CHKERRQ(KSPLogResidualHistory(ksp,res));
    lgmres->it = (loc_it - 1);
    CHKERRQ(KSPMonitor(ksp,ksp->its,res));

    /* see if more space is needed for work vectors */
    if (lgmres->vv_allocated <= loc_it + VEC_OFFSET + 1) {
      CHKERRQ(KSPLGMRESGetNewVectors(ksp,loc_it+1));
      /* (loc_it+1) is passed in as number of the first vector that should
          be allocated */
    }

    /*LGMRES_MOD: decide whether this is an arnoldi step or an aug step */
    if (loc_it < it_arnoldi) { /* Arnoldi */
      CHKERRQ(KSP_PCApplyBAorAB(ksp,VEC_VV(loc_it),VEC_VV(1+loc_it),VEC_TEMP_MATOP));
    } else { /*aug step */
      order = loc_it - it_arnoldi + 1; /* which aug step */
      for (ii=0; ii<aug_dim; ii++) {
        if (lgmres->aug_order[ii] == order) {
          spot = ii;
          break; /* must have this because there will be duplicates before aug_ct = aug_dim */
        }
      }

      CHKERRQ(VecCopy(A_AUGVEC(spot), VEC_VV(1+loc_it)));
      /*note: an alternate implementation choice would be to only save the AUGVECS and
        not A_AUGVEC and then apply the PC here to the augvec */
    }

    /* update hessenberg matrix and do Gram-Schmidt - new direction is in
       VEC_VV(1+loc_it)*/
    CHKERRQ((*lgmres->orthog)(ksp,loc_it));

    /* new entry in hessenburg is the 2-norm of our new direction */
    CHKERRQ(VecNorm(VEC_VV(loc_it+1),NORM_2,&tt));

    *HH(loc_it+1,loc_it)  = tt;
    *HES(loc_it+1,loc_it) = tt;

    /* check for the happy breakdown */
    hapbnd = PetscAbsScalar(tt / *GRS(loc_it)); /* GRS(loc_it) contains the res_norm from the last iteration  */
    if (hapbnd > lgmres->haptol) hapbnd = lgmres->haptol;
    if (tt > hapbnd) {
      tmp  = 1.0/tt;
      CHKERRQ(VecScale(VEC_VV(loc_it+1),tmp)); /* scale new direction by its norm */
    } else {
      CHKERRQ(PetscInfo(ksp,"Detected happy breakdown, current hapbnd = %g tt = %g\n",(double)hapbnd,(double)tt));
      hapend = PETSC_TRUE;
    }

    /* Now apply rotations to new col of hessenberg (and right side of system),
       calculate new rotation, and get new residual norm at the same time*/
    CHKERRQ(KSPLGMRESUpdateHessenberg(ksp,loc_it,hapend,&res));
    if (ksp->reason) break;

    loc_it++;
    lgmres->it = (loc_it-1);   /* Add this here in case it has converged */

    CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->its++;
    if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = res;
    else ksp->rnorm = 0.0;
    CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));

    CHKERRQ((*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP));

    /* Catch error in happy breakdown and signal convergence and break from loop */
    if (hapend) {
      if (!ksp->reason) {
        PetscCheckFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"You reached the happy break down, but convergence was not indicated. Residual norm = %g",(double)res);
        else {
          ksp->reason = KSP_DIVERGED_BREAKDOWN;
          break;
        }
      }
    }
  }
  /* END OF ITERATION LOOP */
  CHKERRQ(KSPLogResidualHistory(ksp,res));

  /* Monitor if we know that we will not return for a restart */
  if (ksp->reason || ksp->its >= max_it) {
    CHKERRQ(KSPMonitor(ksp, ksp->its, res));
  }

  if (itcount) *itcount = loc_it;

  /*
    Down here we have to solve for the "best" coefficients of the Krylov
    columns, add the solution values together, and possibly unwind the
    preconditioning from the solution
   */

  /* Form the solution (or the solution so far) */
  /* Note: must pass in (loc_it-1) for iteration count so that KSPLGMRESBuildSoln
     properly navigates */

  CHKERRQ(KSPLGMRESBuildSoln(GRS(0),ksp->vec_sol,ksp->vec_sol,ksp,loc_it-1));

  /* LGMRES_MOD collect aug vector and A*augvector for future restarts -
     only if we will be restarting (i.e. this cycle performed it_total
     iterations)  */
  if (!ksp->reason && ksp->its < max_it && aug_dim > 0) {

    /*AUG_TEMP contains the new augmentation vector (assigned in  KSPLGMRESBuildSoln) */
    if (!lgmres->aug_ct) {
      spot = 0;
      lgmres->aug_ct++;
    } else if (lgmres->aug_ct < aug_dim) {
      spot = lgmres->aug_ct;
      lgmres->aug_ct++;
    } else { /* truncate */
      for (ii=0; ii<aug_dim; ii++) {
        if (lgmres->aug_order[ii] == aug_dim) spot = ii;
      }
    }

    CHKERRQ(VecCopy(AUG_TEMP, AUGVEC(spot)));
    /*need to normalize */
    CHKERRQ(VecNorm(AUGVEC(spot), NORM_2, &tmp_norm));

    inv_tmp_norm = 1.0/tmp_norm;

    CHKERRQ(VecScale(AUGVEC(spot),inv_tmp_norm));

    /*set new aug vector to order 1  - move all others back one */
    for (ii=0; ii < aug_dim; ii++) AUG_ORDER(ii)++;
    AUG_ORDER(spot) = 1;

    /*now add the A*aug vector to A_AUGVEC(spot)  - this is independ. of preconditioning type*/
    /* want V*H*y - y is in GRS, V is in VEC_VV and H is in HES */

    /* first do H+*y */
    avec = lgmres->hwork;
    CHKERRQ(PetscArrayzero(avec,it_total+1));
    for (ii=0; ii < it_total + 1; ii++) {
      for (jj=0; jj <= ii+1 && jj < it_total+1; jj++) {
        avec[jj] += *HES(jj ,ii) * *GRS(ii);
      }
    }

    /*now multiply result by V+ */
    CHKERRQ(VecSet(VEC_TEMP,0.0));
    CHKERRQ(VecMAXPY(VEC_TEMP, it_total+1, avec, &VEC_VV(0))); /*answer is in VEC_TEMP*/

    /*copy answer to aug location  and scale*/
    CHKERRQ(VecCopy(VEC_TEMP,  A_AUGVEC(spot)));
    CHKERRQ(VecScale(A_AUGVEC(spot),inv_tmp_norm));
  }
  PetscFunctionReturn(0);
}

/*
    KSPSolve_LGMRES - This routine applies the LGMRES method.

   Input Parameter:
.     ksp - the Krylov space object that was set to use lgmres

   Output Parameter:
.     outits - number of iterations used

*/

PetscErrorCode KSPSolve_LGMRES(KSP ksp)
{
  PetscInt       cycle_its; /* iterations done in a call to KSPLGMRESCycle */
  PetscInt       itcount;   /* running total of iterations, incl. those in restarts */
  KSP_LGMRES     *lgmres    = (KSP_LGMRES*)ksp->data;
  PetscBool      guess_zero = ksp->guess_zero;
  PetscInt       ii;        /*LGMRES_MOD variable */

  PetscFunctionBegin;
  PetscCheckFalse(ksp->calc_sings && !lgmres->Rsvd,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ORDER,"Must call KSPSetComputeSingularValues() before KSPSetUp() is called");

  CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));

  ksp->its        = 0;
  lgmres->aug_ct  = 0;
  lgmres->matvecs = 0;

  CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));

  /* initialize */
  itcount     = 0;
  /*LGMRES_MOD*/
  for (ii=0; ii<lgmres->aug_dim; ii++) lgmres->aug_order[ii] = 0;

  while (!ksp->reason) {
    /* calc residual - puts in VEC_VV(0) */
    CHKERRQ(KSPInitialResidual(ksp,ksp->vec_sol,VEC_TEMP,VEC_TEMP_MATOP,VEC_VV(0),ksp->vec_rhs));
    CHKERRQ(KSPLGMRESCycle(&cycle_its,ksp));
    itcount += cycle_its;
    if (itcount >= ksp->max_it) {
      if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
      break;
    }
    ksp->guess_zero = PETSC_FALSE; /* every future call to KSPInitialResidual() will have nonzero guess */
  }
  ksp->guess_zero = guess_zero; /* restore if user provided nonzero initial guess */
  PetscFunctionReturn(0);
}

/*

   KSPDestroy_LGMRES - Frees all memory space used by the Krylov method.

*/
PetscErrorCode KSPDestroy_LGMRES(KSP ksp)
{
  KSP_LGMRES     *lgmres = (KSP_LGMRES*)ksp->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(lgmres->augvecs));
  if (lgmres->augwork_alloc) {
    CHKERRQ(VecDestroyVecs(lgmres->augwork_alloc,&lgmres->augvecs_user_work[0]));
  }
  CHKERRQ(PetscFree(lgmres->augvecs_user_work));
  CHKERRQ(PetscFree(lgmres->aug_order));
  CHKERRQ(PetscFree(lgmres->hwork));
  CHKERRQ(KSPDestroy_GMRES(ksp));
  PetscFunctionReturn(0);
}

/*
    KSPLGMRESBuildSoln - create the solution from the starting vector and the
                      current iterates.

    Input parameters:
        nrs - work area of size it + 1.
        vguess  - index of initial guess
        vdest - index of result.  Note that vguess may == vdest (replace
                guess with the solution).
        it - HH upper triangular part is a block of size (it+1) x (it+1)

     This is an internal routine that knows about the LGMRES internals.
 */
static PetscErrorCode KSPLGMRESBuildSoln(PetscScalar *nrs,Vec vguess,Vec vdest,KSP ksp,PetscInt it)
{
  PetscScalar    tt;
  PetscInt       ii,k,j;
  KSP_LGMRES     *lgmres = (KSP_LGMRES*)(ksp->data);
  /*LGMRES_MOD */
  PetscInt it_arnoldi, it_aug;
  PetscInt jj, spot = 0;

  PetscFunctionBegin;
  /* Solve for solution vector that minimizes the residual */

  /* If it is < 0, no lgmres steps have been performed */
  if (it < 0) {
    CHKERRQ(VecCopy(vguess,vdest)); /* VecCopy() is smart, exists immediately if vguess == vdest */
    PetscFunctionReturn(0);
  }

  /* so (it+1) lgmres steps HAVE been performed */

  /* LGMRES_MOD - determine if we need to use augvecs for the soln  - do not assume that
     this is called after the total its allowed for an approx space */
  if (lgmres->approx_constant) {
    it_arnoldi = lgmres->max_k - lgmres->aug_ct;
  } else {
    it_arnoldi = lgmres->max_k - lgmres->aug_dim;
  }
  if (it_arnoldi >= it +1) {
    it_aug     = 0;
    it_arnoldi = it+1;
  } else {
    it_aug = (it + 1) - it_arnoldi;
  }

  /* now it_arnoldi indicates the number of matvecs that took place */
  lgmres->matvecs += it_arnoldi;

  /* solve the upper triangular system - GRS is the right side and HH is
     the upper triangular matrix  - put soln in nrs */
  PetscCheckFalse(*HH(it,it) == 0.0,PETSC_COMM_SELF,PETSC_ERR_CONV_FAILED,"HH(it,it) is identically zero; it = %D GRS(it) = %g",it,(double)PetscAbsScalar(*GRS(it)));
  if (*HH(it,it) != 0.0) {
    nrs[it] = *GRS(it) / *HH(it,it);
  } else {
    nrs[it] = 0.0;
  }

  for (ii=1; ii<=it; ii++) {
    k  = it - ii;
    tt = *GRS(k);
    for (j=k+1; j<=it; j++) tt = tt - *HH(k,j) * nrs[j];
    nrs[k] = tt / *HH(k,k);
  }

  /* Accumulate the correction to the soln of the preconditioned prob. in VEC_TEMP */
  CHKERRQ(VecSet(VEC_TEMP,0.0)); /* set VEC_TEMP components to 0 */

  /*LGMRES_MOD - if augmenting has happened we need to form the solution
    using the augvecs */
  if (!it_aug) { /* all its are from arnoldi */
    CHKERRQ(VecMAXPY(VEC_TEMP,it+1,nrs,&VEC_VV(0)));
  } else { /*use aug vecs */
    /*first do regular krylov directions */
    CHKERRQ(VecMAXPY(VEC_TEMP,it_arnoldi,nrs,&VEC_VV(0)));
    /*now add augmented portions - add contribution of aug vectors one at a time*/

    for (ii=0; ii<it_aug; ii++) {
      for (jj=0; jj<lgmres->aug_dim; jj++) {
        if (lgmres->aug_order[jj] == (ii+1)) {
          spot = jj;
          break; /* must have this because there will be duplicates before aug_ct = aug_dim */
        }
      }
      CHKERRQ(VecAXPY(VEC_TEMP,nrs[it_arnoldi+ii],AUGVEC(spot)));
    }
  }
  /* now VEC_TEMP is what we want to keep for augmenting purposes - grab before the
     preconditioner is "unwound" from right-precondtioning*/
  CHKERRQ(VecCopy(VEC_TEMP, AUG_TEMP));

  CHKERRQ(KSPUnwindPreconditioner(ksp,VEC_TEMP,VEC_TEMP_MATOP));

  /* add solution to previous solution */
  /* put updated solution into vdest.*/
  CHKERRQ(VecCopy(vguess,vdest));
  CHKERRQ(VecAXPY(vdest,1.0,VEC_TEMP));
  PetscFunctionReturn(0);
}

/*

    KSPLGMRESUpdateHessenberg - Do the scalar work for the orthogonalization.
                            Return new residual.

    input parameters:

.        ksp -    Krylov space object
.        it  -    plane rotations are applied to the (it+1)th column of the
                  modified hessenberg (i.e. HH(:,it))
.        hapend - PETSC_FALSE not happy breakdown ending.

    output parameters:
.        res - the new residual

 */
static PetscErrorCode KSPLGMRESUpdateHessenberg(KSP ksp,PetscInt it,PetscBool hapend,PetscReal *res)
{
  PetscScalar *hh,*cc,*ss,tt;
  PetscInt    j;
  KSP_LGMRES  *lgmres = (KSP_LGMRES*)(ksp->data);

  PetscFunctionBegin;
  hh = HH(0,it);   /* pointer to beginning of column to update - so
                      incrementing hh "steps down" the (it+1)th col of HH*/
  cc = CC(0);      /* beginning of cosine rotations */
  ss = SS(0);      /* beginning of sine rotations */

  /* Apply all the previously computed plane rotations to the new column
     of the Hessenberg matrix */
  /* Note: this uses the rotation [conj(c)  s ; -s   c], c= cos(theta), s= sin(theta) */

  for (j=1; j<=it; j++) {
    tt  = *hh;
    *hh = PetscConj(*cc) * tt + *ss * *(hh+1);
    hh++;
    *hh = *cc++ * *hh - (*ss++ * tt);
    /* hh, cc, and ss have all been incremented one by end of loop */
  }

  /*
    compute the new plane rotation, and apply it to:
     1) the right-hand-side of the Hessenberg system (GRS)
        note: it affects GRS(it) and GRS(it+1)
     2) the new column of the Hessenberg matrix
        note: it affects HH(it,it) which is currently pointed to
        by hh and HH(it+1, it) (*(hh+1))
    thus obtaining the updated value of the residual...
  */

  /* compute new plane rotation */

  if (!hapend) {
    tt = PetscSqrtScalar(PetscConj(*hh) * *hh + PetscConj(*(hh+1)) * *(hh+1));
    if (tt == 0.0) {
      ksp->reason = KSP_DIVERGED_NULL;
      PetscFunctionReturn(0);
    }
    *cc = *hh / tt;         /* new cosine value */
    *ss = *(hh+1) / tt;        /* new sine value */

    /* apply to 1) and 2) */
    *GRS(it+1) = -(*ss * *GRS(it));
    *GRS(it)   = PetscConj(*cc) * *GRS(it);
    *hh        = PetscConj(*cc) * *hh + *ss * *(hh+1);

    /* residual is the last element (it+1) of right-hand side! */
    *res = PetscAbsScalar(*GRS(it+1));

  } else { /* happy breakdown: HH(it+1, it) = 0, therefore we don't need to apply
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

   KSPLGMRESGetNewVectors - This routine allocates more work vectors, starting from
                         VEC_VV(it)

*/
static PetscErrorCode KSPLGMRESGetNewVectors(KSP ksp,PetscInt it)
{
  KSP_LGMRES     *lgmres = (KSP_LGMRES*)ksp->data;
  PetscInt       nwork   = lgmres->nwork_alloc; /* number of work vector chunks allocated */
  PetscInt       nalloc;                      /* number to allocate */
  PetscInt       k;

  PetscFunctionBegin;
  nalloc = lgmres->delta_allocate; /* number of vectors to allocate
                                      in a single chunk */

  /* Adjust the number to allocate to make sure that we don't exceed the
     number of available slots (lgmres->vecs_allocated)*/
  if (it + VEC_OFFSET + nalloc >= lgmres->vecs_allocated) {
    nalloc = lgmres->vecs_allocated - it - VEC_OFFSET;
  }
  if (!nalloc) PetscFunctionReturn(0);

  lgmres->vv_allocated += nalloc; /* vv_allocated is the number of vectors allocated */

  /* work vectors */
  CHKERRQ(KSPCreateVecs(ksp,nalloc,&lgmres->user_work[nwork],0,NULL));
  CHKERRQ(PetscLogObjectParents(ksp,nalloc,lgmres->user_work[nwork]));
  /* specify size of chunk allocated */
  lgmres->mwork_alloc[nwork] = nalloc;

  for (k=0; k < nalloc; k++) {
    lgmres->vecs[it+VEC_OFFSET+k] = lgmres->user_work[nwork][k];
  }

  /* LGMRES_MOD - for now we are preallocating the augmentation vectors */

  /* increment the number of work vector chunks */
  lgmres->nwork_alloc++;
  PetscFunctionReturn(0);
}

/*

   KSPBuildSolution_LGMRES

     Input Parameter:
.     ksp - the Krylov space object
.     ptr-

   Output Parameter:
.     result - the solution

   Note: this calls KSPLGMRESBuildSoln - the same function that KSPLGMRESCycle
   calls directly.

*/
PetscErrorCode KSPBuildSolution_LGMRES(KSP ksp,Vec ptr,Vec *result)
{
  KSP_LGMRES     *lgmres = (KSP_LGMRES*)ksp->data;

  PetscFunctionBegin;
  if (!ptr) {
    if (!lgmres->sol_temp) {
      CHKERRQ(VecDuplicate(ksp->vec_sol,&lgmres->sol_temp));
      CHKERRQ(PetscLogObjectParent((PetscObject)ksp,(PetscObject)lgmres->sol_temp));
    }
    ptr = lgmres->sol_temp;
  }
  if (!lgmres->nrs) {
    /* allocate the work area */
    CHKERRQ(PetscMalloc1(lgmres->max_k,&lgmres->nrs));
    CHKERRQ(PetscLogObjectMemory((PetscObject)ksp,lgmres->max_k*sizeof(PetscScalar)));
  }

  CHKERRQ(KSPLGMRESBuildSoln(lgmres->nrs,ksp->vec_sol,ptr,ksp,lgmres->it));
  if (result) *result = ptr;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPView_LGMRES(KSP ksp,PetscViewer viewer)
{
  KSP_LGMRES     *lgmres = (KSP_LGMRES*)ksp->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(KSPView_GMRES(ksp,viewer));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    /*LGMRES_MOD */
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  aug. dimension=%D\n",lgmres->aug_dim));
    if (lgmres->approx_constant) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  approx. space size was kept constant.\n"));
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  number of matvecs=%D\n",lgmres->matvecs));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetFromOptions_LGMRES(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscInt       aug;
  KSP_LGMRES     *lgmres = (KSP_LGMRES*) ksp->data;
  PetscBool      flg     = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(KSPSetFromOptions_GMRES(PetscOptionsObject,ksp));
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"KSP LGMRES Options"));
  CHKERRQ(PetscOptionsBool("-ksp_lgmres_constant","Use constant approx. space size","KSPGMRESSetConstant",lgmres->approx_constant,&lgmres->approx_constant,NULL));
  CHKERRQ(PetscOptionsInt("-ksp_lgmres_augment","Number of error approximations to augment the Krylov space with","KSPLGMRESSetAugDim",lgmres->aug_dim,&aug,&flg));
  if (flg) CHKERRQ(KSPLGMRESSetAugDim(ksp,aug));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

/*functions for extra lgmres options here*/
static PetscErrorCode  KSPLGMRESSetConstant_LGMRES(KSP ksp)
{
  KSP_LGMRES *lgmres = (KSP_LGMRES*)ksp->data;

  PetscFunctionBegin;
  lgmres->approx_constant = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPLGMRESSetAugDim_LGMRES(KSP ksp,PetscInt aug_dim)
{
  KSP_LGMRES *lgmres = (KSP_LGMRES*)ksp->data;

  PetscFunctionBegin;
  PetscCheckFalse(aug_dim < 0,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Augmentation dimension must be positive");
  PetscCheckFalse(aug_dim > (lgmres->max_k -1),PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Augmentation dimension must be <= (restart size-1)");
  lgmres->aug_dim = aug_dim;
  PetscFunctionReturn(0);
}

/* end new lgmres functions */

/*MC
    KSPLGMRES - Augments the standard GMRES approximation space with approximations to
                the error from previous restart cycles.

  Options Database Keys:
+   -ksp_gmres_restart <restart> - total approximation space size (Krylov directions + error approximations)
.   -ksp_gmres_haptol <tol> - sets the tolerance for "happy ending" (exact convergence)
.   -ksp_gmres_preallocate - preallocate all the Krylov search directions initially (otherwise groups of
                            vectors are allocated as needed)
.   -ksp_gmres_classicalgramschmidt - use classical (unmodified) Gram-Schmidt to orthogonalize against the Krylov space (fast) (the default)
.   -ksp_gmres_modifiedgramschmidt - use modified Gram-Schmidt in the orthogonalization (more stable, but slower)
.   -ksp_gmres_cgs_refinement_type <refine_never,refine_ifneeded,refine_always> - determine if iterative refinement is used to increase the
                                  stability of the classical Gram-Schmidt  orthogonalization.
.   -ksp_gmres_krylov_monitor - plot the Krylov space generated
.   -ksp_lgmres_augment <k> - number of error approximations to augment the Krylov space with
-   -ksp_lgmres_constant - use a constant approx. space size (only affects restart cycles < num. error approx.(k), i.e. the first k restarts)

    To run LGMRES(m, k) as described in the above paper, use:
       -ksp_gmres_restart <m+k>
       -ksp_lgmres_augment <k>

  Level: beginner

   Notes:
    Supports both left and right preconditioning, but not symmetric.

   References:
.  * - A. H. Baker, E.R. Jessup, and T.A. Manteuffel. A technique for accelerating the convergence of restarted GMRES. SIAM Journal on Matrix Analysis and Applications, 26 (2005).

  Developer Notes:
    This object is subclassed off of KSPGMRES

  Contributed by: Allison Baker

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPFGMRES, KSPGMRES,
          KSPGMRESSetRestart(), KSPGMRESSetHapTol(), KSPGMRESSetPreAllocateVectors(), KSPGMRESSetOrthogonalization(), KSPGMRESGetOrthogonalization(),
          KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESModifiedGramSchmidtOrthogonalization(),
          KSPGMRESCGSRefinementType, KSPGMRESSetCGSRefinementType(), KSPGMRESGetCGSRefinementType(), KSPGMRESMonitorKrylov(), KSPLGMRESSetAugDim(),
          KSPGMRESSetConstant()

M*/

PETSC_EXTERN PetscErrorCode KSPCreate_LGMRES(KSP ksp)
{
  KSP_LGMRES     *lgmres;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(ksp,&lgmres));

  ksp->data               = (void*)lgmres;
  ksp->ops->buildsolution = KSPBuildSolution_LGMRES;

  ksp->ops->setup                        = KSPSetUp_LGMRES;
  ksp->ops->solve                        = KSPSolve_LGMRES;
  ksp->ops->destroy                      = KSPDestroy_LGMRES;
  ksp->ops->view                         = KSPView_LGMRES;
  ksp->ops->setfromoptions               = KSPSetFromOptions_LGMRES;
  ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_GMRES;
  ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_GMRES;

  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1));

  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetPreAllocateVectors_C",KSPGMRESSetPreAllocateVectors_GMRES));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetOrthogonalization_C",KSPGMRESSetOrthogonalization_GMRES));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESGetOrthogonalization_C",KSPGMRESGetOrthogonalization_GMRES));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetRestart_C",KSPGMRESSetRestart_GMRES));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESGetRestart_C",KSPGMRESGetRestart_GMRES));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetHapTol_C",KSPGMRESSetHapTol_GMRES));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetCGSRefinementType_C",KSPGMRESSetCGSRefinementType_GMRES));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESGetCGSRefinementType_C",KSPGMRESGetCGSRefinementType_GMRES));

  /*LGMRES_MOD add extra functions here - like the one to set num of aug vectors */
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPLGMRESSetConstant_C",KSPLGMRESSetConstant_LGMRES));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPLGMRESSetAugDim_C",KSPLGMRESSetAugDim_LGMRES));

  /*defaults */
  lgmres->haptol         = 1.0e-30;
  lgmres->q_preallocate  = 0;
  lgmres->delta_allocate = LGMRES_DELTA_DIRECTIONS;
  lgmres->orthog         = KSPGMRESClassicalGramSchmidtOrthogonalization;
  lgmres->nrs            = NULL;
  lgmres->sol_temp       = NULL;
  lgmres->max_k          = LGMRES_DEFAULT_MAXK;
  lgmres->Rsvd           = NULL;
  lgmres->cgstype        = KSP_GMRES_CGS_REFINE_NEVER;
  lgmres->orthogwork     = NULL;

  /*LGMRES_MOD - new defaults */
  lgmres->aug_dim         = LGMRES_DEFAULT_AUGDIM;
  lgmres->aug_ct          = 0;     /* start with no aug vectors */
  lgmres->approx_constant = PETSC_FALSE;
  lgmres->matvecs         = 0;
  PetscFunctionReturn(0);
}
