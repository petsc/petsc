/* Defines the basic SNES object */
#include <../src/snes/impls/ngmres/snesngmres.h>
#include <petscblaslapack.h>




#undef __FUNCT__
#define __FUNCT__ "SNESReset_NGMRES"
PetscErrorCode SNESReset_NGMRES(SNES snes)
{
  SNES_NGMRES   *ngmres = (SNES_NGMRES*) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(ngmres->msize, &ngmres->rdot);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ngmres->msize, &ngmres->xdot);CHKERRQ(ierr);
  if (snes->work) {ierr = VecDestroyVecs(snes->nwork, &snes->work);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_NGMRES"
PetscErrorCode SNESDestroy_NGMRES(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESReset_NGMRES(snes);CHKERRQ(ierr);
  if (snes->data) {
    SNES_NGMRES * ngmres = (SNES_NGMRES *)snes->data;
    ierr = PetscFree5(ngmres->h, ngmres->beta, ngmres->xi, ngmres->r_norms, ngmres->q);CHKERRQ(ierr);
    ierr = PetscFree(ngmres->s);CHKERRQ(ierr);
#if PETSC_USE_COMPLEX
    ierr = PetscFree(ngmres->rwork);
#endif
    ierr = PetscFree(ngmres->work);
  }
  ierr = PetscFree(snes->data);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_NGMRES"
PetscErrorCode SNESSetUp_NGMRES(SNES snes)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES *) snes->data;
  PetscInt       msize,hsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  msize         = ngmres->msize;  /* restart size */
  hsize         = msize * msize;


  /* explicit least squares minimization solve */
  ierr = PetscMalloc5(hsize,PetscScalar,&ngmres->h,
		      msize,PetscScalar,&ngmres->beta,
		      msize,PetscScalar,&ngmres->xi,
		      msize,PetscReal,&ngmres->r_norms,
		      hsize,PetscScalar,&ngmres->q);CHKERRQ(ierr);
  ngmres->nrhs = 1;
  ngmres->lda = msize;
  ngmres->ldb = msize;
  ierr = PetscMalloc(msize*sizeof(PetscScalar),&ngmres->s);CHKERRQ(ierr);
  
  ierr = PetscMemzero(ngmres->h,hsize*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(ngmres->q,hsize*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(ngmres->xi,msize*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(ngmres->beta,msize*sizeof(PetscScalar));CHKERRQ(ierr);
  
  ngmres->lwork = 12*msize;
#if PETSC_USE_COMPLEX
  ierr = PetscMalloc(sizeof(PetscReal)*ngmres->lwork,&ngmres->rwork);
#endif
  ierr = PetscMalloc(sizeof(PetscScalar)*ngmres->lwork,&ngmres->work);

  ierr = VecDuplicateVecs(snes->vec_sol, ngmres->msize, &ngmres->xdot);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(snes->vec_sol, ngmres->msize, &ngmres->rdot);CHKERRQ(ierr);
  ierr = SNESDefaultGetWork(snes, 3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_NGMRES"
PetscErrorCode SNESSetFromOptions_NGMRES(SNES snes)
{
  SNES_NGMRES   *ngmres = (SNES_NGMRES *) snes->data;
  PetscErrorCode ierr;
  PetscBool      debug;
  
  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES NGMRES options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_ngmres_m", "Number of directions", "SNES", ngmres->msize, &ngmres->msize, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_ngmres_restart", "Maximum iterations before restart.", "SNES", ngmres->k_rmax, &ngmres->k_rmax, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_ngmres_monitor", "Monitor actions of NGMRES", "SNES", ngmres->monitor ? PETSC_TRUE: PETSC_FALSE, &debug, PETSC_NULL);CHKERRQ(ierr); 
  if (debug) {
    ngmres->monitor = PETSC_VIEWER_STDOUT_(((PetscObject)snes)->comm);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-snes_ngmres_gammaA", "Residual selection constant", "SNES", ngmres->gammaA, &ngmres->gammaA, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_ngmres_gammaC", "Residual restart constant", "SNES", ngmres->gammaC, &ngmres->gammaC, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_ngmres_epsilonB", "Difference selection constant", "SNES", ngmres->epsilonB, &ngmres->epsilonB, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_ngmres_deltaB", "Difference residual selection constant", "SNES", ngmres->deltaB, &ngmres->deltaB, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if ((ngmres->gammaA > ngmres->gammaC) && (ngmres->gammaC > 2.)) ngmres->gammaC = ngmres->gammaA;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESView_NGMRES"
PetscErrorCode SNESView_NGMRES(SNES snes, PetscViewer viewer)
{
  SNES_NGMRES   *ngmres = (SNES_NGMRES *) snes->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "  Size of space %d\n", ngmres->msize);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  Maximum iterations before restart %d\n", ngmres->k_rmax);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESSolve_NGMRES"

PetscErrorCode SNESSolve_NGMRES(SNES snes)
{
  SNES           pc;
  SNES_NGMRES   *ngmres = (SNES_NGMRES *) snes->data;

  
  
  /* present solution, residual, and preconditioned residual */
  Vec            x, r, b, d;
  Vec            x_A, r_A;

  /* previous iterations to construct the subspace */
  Vec            *rdot = ngmres->rdot;
  Vec            *xdot = ngmres->xdot;

  /* coefficients and RHS to the minimization problem */
  PetscScalar    *beta = ngmres->beta;
  PetscScalar    *xi = ngmres->xi;
  PetscReal      r_norm, r_A_norm;
  PetscReal      nu;
  PetscScalar    alph_total = 0.;
  PetscScalar    qentry;
  PetscInt       i, j, k, k_restart, l, ivec;

  /* solution selection data */
  PetscBool      selectA, selectRestart;
  PetscReal      d_norm, d_min_norm, d_cur_norm;
  PetscReal      r_min_norm;

  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* variable initialization */
  snes->reason  = SNES_CONVERGED_ITERATING;
  x             = snes->vec_sol;
  r             = snes->vec_func;
  b             = snes->vec_rhs;
  x_A           = snes->vec_sol_update;
  r_A           = snes->work[0];
  d             = snes->work[1];
  r             = snes->work[2];

  ierr = SNESGetPC(snes, &pc);CHKERRQ(ierr);
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);

  /* initialization */

  /* r = F(x) */
  ierr = SNESComputeFunction(snes, x, r);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }

  /* nu = (r, r) */
  ierr = VecNorm(r, NORM_2, &r_norm);CHKERRQ(ierr);
  r_min_norm = r_norm;
  nu = r_norm*r_norm;
  if (PetscIsInfOrNanReal(r_norm)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FP, "Infinite or not-a-number generated in function evaluation");

  /* q_{00} = nu  */
  Q(0,0) = nu;
  ngmres->r_norms[0] = r_norm;
  /* rdot[0] = r */
  ierr = VecCopy(x, xdot[0]);CHKERRQ(ierr);
  ierr = VecCopy(r, rdot[0]);CHKERRQ(ierr);

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = r_norm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes, r_norm, 0);
  ierr = SNESMonitor(snes, 0, r_norm);CHKERRQ(ierr);
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,r_norm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  k_restart = 1;
  l = 1;
  for (k=1; k<snes->max_its; k++) {

    /* select which vector of the stored subspace will be updated */
    ivec = k_restart % ngmres->msize; /* replace the last used part of the subspace */

    /* Computation of x^M */
    ierr = SNESSolve(pc, b, x);CHKERRQ(ierr);
    /* r = F(x) */
    ierr = SNESComputeFunction(snes, x, r);CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &r_norm);CHKERRQ(ierr);
    /* nu = (r, r) */
    ngmres->r_norms[ivec] = r_norm;
    nu = r_norm*r_norm;    
    if (r_min_norm > r_norm) r_min_norm = r_norm;  /* the minimum norm is now of r^M */

    /* construct the right hand side and xi factors */
    for (i = 0; i < l; i++) {
      VecDot(rdot[i], r, &xi[i]);
      beta[i] = nu - xi[i]; 
    }

    /* construct h */
    for (j = 0; j < l; j++) {
      for (i = 0; i < l; i++) {
	H(i, j) = Q(i, j) - xi[i] - xi[j] + nu;
      }
    }
#ifdef PETSC_MISSING_LAPACK_GELSS
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "NGMRES with LS requires the LAPACK GELSS routine."); 
#else
    ngmres->m = PetscBLASIntCast(l);
    ngmres->n = PetscBLASIntCast(l);
    ngmres->info = PetscBLASIntCast(0);
    ngmres->rcond = -1.;
#ifdef PETSC_USE_COMPLEX
    LAPACKgelss_(&ngmres->m,
		 &ngmres->n,
		 &ngmres->nrhs,
		 ngmres->h,
		 &ngmres->lda,
		 ngmres->beta,
		 &ngmres->ldb,
		 ngmres->s,
		 &ngmres->rcond,
		 &ngmres->rank,
		 ngmres->work,
		 &ngmres->lwork,
		 ngmres->rwork,
		 &ngmres->info);
#else
    LAPACKgelss_(&ngmres->m,
		 &ngmres->n,
		 &ngmres->nrhs,
		 ngmres->h,
		 &ngmres->lda,
		 ngmres->beta,
		 &ngmres->ldb,
		 ngmres->s,
		 &ngmres->rcond,
		 &ngmres->rank,
		 ngmres->work,
		 &ngmres->lwork,
		 &ngmres->info);
#endif
    if (ngmres->info < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to GELSS");
    if (ngmres->info > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"SVD failed to converge");
#endif

    alph_total = 0.;
    for (i = 0; i < l; i++) {
      alph_total += beta[i];
    }
    ierr = VecCopy(x, x_A);CHKERRQ(ierr);
    ierr = VecScale(x_A, 1. - alph_total);CHKERRQ(ierr);

    for(i=0;i<l;i++){
      ierr= VecAXPY(x_A, beta[i], xdot[i]);CHKERRQ(ierr);
    }
    ierr = SNESComputeFunction(snes, x_A, r_A);CHKERRQ(ierr);
    ierr = VecNorm(r_A, NORM_2, &r_A_norm);CHKERRQ(ierr);

    selectA = PETSC_TRUE;
    /* Conditions for choosing the accelerated answer */

    /* Criterion A -- the norm of the function isn't increased above the minimum by too much */
    if (r_A_norm >= ngmres->gammaA*r_min_norm) {
      selectA = PETSC_FALSE;
    }
    
    /* Criterion B -- the choice of x^A isn't too close to some other choice */
    ierr=VecCopy(x_A,d);CHKERRQ(ierr);   
    ierr=VecAXPY(d,-1,x);CHKERRQ(ierr);   
    ierr=VecNorm(d,NORM_2,&d_norm);CHKERRQ(ierr);     
    d_min_norm = -1.0;
    for(i=0;i<l;i++) {
      ierr=VecCopy(x_A,d);CHKERRQ(ierr);   
      ierr=VecAXPY(d,-1,xdot[i]);CHKERRQ(ierr);   
      ierr=VecNorm(d,NORM_2,&d_cur_norm);CHKERRQ(ierr);        
      if((d_cur_norm < d_min_norm) || (d_min_norm < 0.0)) d_min_norm = d_cur_norm;
    }
    if (ngmres->epsilonB*d_norm<d_min_norm || sqrt(r_norm)<ngmres->deltaB*sqrt(r_min_norm)) {
    } else {
      selectA=PETSC_FALSE;
    }


    if (selectA) {
      if (ngmres->monitor) {
	ierr = PetscViewerASCIIPrintf(ngmres->monitor, "picked r_A, ||r_A||_2 = %e, ||r_M||_2 = %e\n", r_A_norm, r_norm);CHKERRQ(ierr);
      }
      /* copy it over */
      r_norm = r_A_norm;
      nu = r_norm*r_norm;
      ierr = VecCopy(r_A, r);CHKERRQ(ierr);
      ierr = VecCopy(x_A, x);CHKERRQ(ierr);
    } else {
      if (ngmres->monitor) {
	ierr = PetscViewerASCIIPrintf(ngmres->monitor, "picked r_M, ||r_A||_2 = %e, ||r_M||_2 = %e\n", r_A_norm, r_norm);CHKERRQ(ierr);
      }
    }

    selectRestart = PETSC_FALSE;
    
    /* maximum iteration criterion */
    if (k_restart > ngmres->k_rmax) {
      selectRestart = PETSC_TRUE;
    }

    /* difference stagnation restart */
    if 	((ngmres->epsilonB*d_norm > d_min_norm) && (sqrt(r_A_norm) > ngmres->deltaB*sqrt(r_min_norm))) {
      if (ngmres->monitor) {
	ierr = PetscViewerASCIIPrintf(ngmres->monitor, "difference restart: %e > %e\n", ngmres->epsilonB*d_norm, d_min_norm);CHKERRQ(ierr);
      }
      selectRestart = PETSC_TRUE;
    }
    
    /* residual stagnation restart */
    if (sqrt(r_A_norm) > ngmres->gammaC*sqrt(r_min_norm)) {
      if (ngmres->monitor) {
	ierr = PetscViewerASCIIPrintf(ngmres->monitor, "residual restart: %e > %e\n", sqrt(r_A_norm), ngmres->gammaC*sqrt(r_min_norm));CHKERRQ(ierr);
      }
      selectRestart = PETSC_TRUE;
    }

    if (selectRestart) {
      if (ngmres->monitor){
	ierr = PetscViewerASCIIPrintf(ngmres->monitor, "Restarted at iteration %d\n", k_restart);CHKERRQ(ierr);
      }
      k_restart = 1;
      l = 1;
      /* q_{00} = nu */
      ngmres->r_norms[0] = r_norm;
      nu = r_norm*r_norm;
      Q(0,0) = nu;
      /* rdot[0] = r */
      ierr = VecCopy(x, xdot[0]);CHKERRQ(ierr);
      ierr = VecCopy(r, rdot[0]);CHKERRQ(ierr);
    } else {
      /* select the current size of the subspace */
      if (l < ngmres->msize) {
	l++;
      }
      k_restart++;
      /* place the current entry in the list of previous entries */
      ierr = VecCopy(r, rdot[ivec]);CHKERRQ(ierr);
      ierr = VecCopy(x, xdot[ivec]);CHKERRQ(ierr);
      ngmres->r_norms[ivec] = r_norm;
      if (r_min_norm > r_norm) r_min_norm = r_norm;  /* the minimum norm is now of r^A */
      for (i = 0; i < l; i++) {
	VecDot(r, rdot[i], &qentry);
	Q(i, ivec) = qentry;
	Q(ivec, i) = qentry;
      }
    }
    
    SNESLogConvHistory(snes, r_norm, k);
    ierr = SNESMonitor(snes, k, r_norm);CHKERRQ(ierr);

    snes->iter =k;
    ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,r_norm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);
  }
  snes->reason = SNES_DIVERGED_MAX_IT;
  PetscFunctionReturn(0);
}

/*MC
  SNESNGMRES - The Nonlinear Generalized Minimum Residual (NGMRES) method of Oosterlee and Washio.

   Level: beginner

   "Krylov Subspace Acceleration of Nonlinear Multigrid with Application to Recirculating Flows", C. W. Oosterlee and T. Washio,
   SIAM Journal on Scientific Computing, 21(5), 2000.

   This is also the same as the algorithm called Anderson acceleration introduced in "D. G. Anderson. Iterative procedures for nonlinear integral equations. 
   J. Assoc. Comput. Mach., 12:547–560, 1965."

.seealso: SNESCreate(), SNES, SNESSetType(), SNESType (for list of available types)
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_NGMRES"
PetscErrorCode SNESCreate_NGMRES(SNES snes)
{
  SNES_NGMRES   *ngmres;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_NGMRES;
  snes->ops->setup          = SNESSetUp_NGMRES;
  snes->ops->setfromoptions = SNESSetFromOptions_NGMRES;
  snes->ops->view           = SNESView_NGMRES;
  snes->ops->solve          = SNESSolve_NGMRES;
  snes->ops->reset          = SNESReset_NGMRES;

  snes->usesksp             = PETSC_FALSE;

  ierr = PetscNewLog(snes, SNES_NGMRES, &ngmres);CHKERRQ(ierr);
  snes->data = (void*) ngmres;
  ngmres->msize = 10;

  ngmres->gammaA   = 2.;
  ngmres->gammaC   = 2.;
  ngmres->deltaB   = 0.9;
  ngmres->epsilonB = 0.1;
  ngmres->k_rmax   = 200;

  ierr = SNESGetPC(snes, &snes->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
