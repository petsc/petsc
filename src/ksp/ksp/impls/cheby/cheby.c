
#include <petsc-private/kspimpl.h>                    /*I "petscksp.h" I*/
#include <../src/ksp/ksp/impls/cheby/chebyshevimpl.h>

#undef __FUNCT__  
#define __FUNCT__ "KSPReset_Chebyshev"
PetscErrorCode KSPReset_Chebyshev(KSP ksp)
{
  KSP_Chebyshev *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPReset(cheb->kspest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_Chebyshev"
PetscErrorCode KSPSetUp_Chebyshev(KSP ksp)
{
  KSP_Chebyshev *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDefaultGetWork(ksp,3);CHKERRQ(ierr);
  if (cheb->emin == 0. || cheb->emax == 0.) { /* We need to estimate eigenvalues */
    ierr = KSPChebyshevSetEstimateEigenvalues(ksp,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPChebyshevSetEigenvalues_Chebyshev"
PETSC_EXTERN_C PetscErrorCode KSPChebyshevSetEigenvalues_Chebyshev(KSP ksp,PetscReal emax,PetscReal emin)
{
  KSP_Chebyshev *chebyshevP = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (emax <= emin) SETERRQ2(((PetscObject)ksp)->comm,PETSC_ERR_ARG_INCOMP,"Maximum eigenvalue must be larger than minimum: max %g min %G",emax,emin);
  if (emax*emin <= 0.0) SETERRQ2(((PetscObject)ksp)->comm,PETSC_ERR_ARG_INCOMP,"Both eigenvalues must be of the same sign: max %G min %G",emax,emin);
  chebyshevP->emax = emax;
  chebyshevP->emin = emin;
  ierr = KSPChebyshevSetEstimateEigenvalues(ksp,0.,0.,0.,0.);CHKERRQ(ierr); /* Destroy any estimation setup */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPChebyshevSetEstimateEigenvalues_Chebyshev"
PETSC_EXTERN_C PetscErrorCode KSPChebyshevSetEstimateEigenvalues_Chebyshev(KSP ksp,PetscReal a,PetscReal b,PetscReal c,PetscReal d)
{
  KSP_Chebyshev *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a != 0.0 || b != 0.0 || c != 0.0 || d != 0.0) {
    if (!cheb->kspest) {
      PetscBool nonzero;

      ierr = KSPCreate(((PetscObject)ksp)->comm,&cheb->kspest);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)cheb->kspest,(PetscObject)ksp,1);CHKERRQ(ierr);

      ierr = KSPGetPC(cheb->kspest,&cheb->pcnone);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)cheb->pcnone);CHKERRQ(ierr);
      ierr = PCSetType(cheb->pcnone,PCNONE);CHKERRQ(ierr);
      ierr = KSPSetPC(cheb->kspest,ksp->pc);CHKERRQ(ierr);

      ierr = KSPGetInitialGuessNonzero(ksp,&nonzero);CHKERRQ(ierr);
      ierr = KSPSetInitialGuessNonzero(cheb->kspest,nonzero);CHKERRQ(ierr);
      ierr = KSPSetComputeSingularValues(cheb->kspest,PETSC_TRUE);CHKERRQ(ierr);

      /* Estimate with a fixed number of iterations */
      ierr = KSPSetConvergenceTest(cheb->kspest,KSPSkipConverged,0,0);CHKERRQ(ierr);
      ierr = KSPSetNormType(cheb->kspest,KSP_NORM_NONE);CHKERRQ(ierr);
      ierr = KSPSetTolerances(cheb->kspest,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,10);CHKERRQ(ierr);
    }
    if (a >= 0) cheb->tform[0] = a;
    if (b >= 0) cheb->tform[1] = b;
    if (c >= 0) cheb->tform[2] = c;
    if (d >= 0) cheb->tform[3] = d;
    cheb->estimate_current = PETSC_FALSE;
  } else {
    ierr = KSPDestroy(&cheb->kspest);CHKERRQ(ierr);
    ierr = PCDestroy(&cheb->pcnone);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPChebyshevSetNewMatrix_Chebyshev"
PETSC_EXTERN_C PetscErrorCode  KSPChebyshevSetNewMatrix_Chebyshev(KSP ksp)
{
  KSP_Chebyshev *cheb = (KSP_Chebyshev*)ksp->data;

  PetscFunctionBegin;
  cheb->estimate_current = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPChebyshevSetEigenvalues"
/*@
   KSPChebyshevSetEigenvalues - Sets estimates for the extreme eigenvalues
   of the preconditioned problem.

   Logically Collective on KSP

   Input Parameters:
+  ksp - the Krylov space context
-  emax, emin - the eigenvalue estimates

  Options Database:
.  -ksp_chebyshev_eigenvalues emin,emax

   Note: If you run with the Krylov method of KSPCG with the option -ksp_monitor_singular_value it will 
    for that given matrix and preconditioner an estimate of the extreme eigenvalues.

   Level: intermediate

.keywords: KSP, Chebyshev, set, eigenvalues
@*/
PetscErrorCode  KSPChebyshevSetEigenvalues(KSP ksp,PetscReal emax,PetscReal emin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveReal(ksp,emax,2);
  PetscValidLogicalCollectiveReal(ksp,emin,3);
  ierr = PetscTryMethod(ksp,"KSPChebyshevSetEigenvalues_C",(KSP,PetscReal,PetscReal),(ksp,emax,emin));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPChebyshevSetEstimateEigenvalues"
/*@
   KSPChebyshevSetEstimateEigenvalues - Automatically estimate the eigenvalues to use for Chebyshev

   Logically Collective on KSP

   Input Parameters:
+  ksp - the Krylov space context
.  a - multiple of min eigenvalue estimate to use for min Chebyshev bound (or PETSC_DECIDE)
.  b - multiple of max eigenvalue estimate to use for min Chebyshev bound (or PETSC_DECIDE)
.  c - multiple of min eigenvalue estimate to use for max Chebyshev bound (or PETSC_DECIDE)
-  d - multiple of max eigenvalue estimate to use for max Chebyshev bound (or PETSC_DECIDE)

  Options Database:
.  -ksp_chebyshev_estimate_eigenvalues a,b,c,d

   Notes:
   The Chebyshev bounds are estimated using
.vb
   minbound = a*minest + b*maxest
   maxbound = c*minest + d*maxest
.ve
   The default configuration targets the upper part of the spectrum for use as a multigrid smoother, so only the maximum eigenvalue estimate is used.
   The minimum eigenvalue estimate obtained by Krylov iteration is typically not accurate until the method has converged.

   If 0.0 is passed for all transform arguments (a,b,c,d), eigenvalue estimation is disabled.

   The default transform is (0,0.1; 0,1.1) which targets the "upper" part of the spectrum, as desirable for use with multigrid.

   Level: intermediate

.keywords: KSP, Chebyshev, set, eigenvalues, PCMG
@*/
PetscErrorCode KSPChebyshevSetEstimateEigenvalues(KSP ksp,PetscReal a,PetscReal b,PetscReal c,PetscReal d)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveReal(ksp,a,2);
  PetscValidLogicalCollectiveReal(ksp,b,3);
  PetscValidLogicalCollectiveReal(ksp,c,4);
  PetscValidLogicalCollectiveReal(ksp,d,5);
  ierr = PetscTryMethod(ksp,"KSPChebyshevSetEstimateEigenvalues_C",(KSP,PetscReal,PetscReal,PetscReal,PetscReal),(ksp,a,b,c,d));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPChebyshevSetNewMatrix"
/*@
   KSPChebyshevSetNewMatrix - Indicates that the matrix has changed, causes eigenvalue estimates to be recomputed if appropriate.

   Logically Collective on KSP

   Input Parameter:
.  ksp - the Krylov space context

   Level: developer

.keywords: KSP, Chebyshev, set, eigenvalues

.seealso: KSPChebyshevSetEstimateEigenvalues()
@*/
PetscErrorCode KSPChebyshevSetNewMatrix(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscTryMethod(ksp,"KSPChebyshevSetNewMatrix_C",(KSP),(ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions_Chebyshev"
PetscErrorCode KSPSetFromOptions_Chebyshev(KSP ksp)
{
  KSP_Chebyshev *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       two = 2,four = 4;
  PetscReal      tform[4] = {PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE};
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP Chebyshev Options");CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-ksp_chebyshev_eigenvalues","extreme eigenvalues","KSPChebyshevSetEigenvalues",&cheb->emin,&two,0);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-ksp_chebyshev_estimate_eigenvalues","estimate eigenvalues using a Krylov method, then use this transform for Chebyshev eigenvalue bounds","KSPChebyshevSetEstimateEigenvalues",tform,&four,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (four) {
    case 0:
      ierr = KSPChebyshevSetEstimateEigenvalues(ksp,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
      break;
    case 2:                     /* Base everything on the max eigenvalues */
      ierr = KSPChebyshevSetEstimateEigenvalues(ksp,PETSC_DECIDE,tform[0],PETSC_DECIDE,tform[1]);CHKERRQ(ierr);
      break;
    case 4:                     /* Use the full 2x2 linear transformation */
      ierr = KSPChebyshevSetEstimateEigenvalues(ksp,tform[0],tform[1],tform[2],tform[3]);CHKERRQ(ierr);
      break;
    default: SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_INCOMP,"Must specify either 0, 2, or 4 parameters for eigenvalue estimation");
    }
  }

  /* 
   Use hybrid Chebyshev. 
   Ref: "A hybrid Chebyshev Krylov-subspace algorithm for solving nonsymmetric systems of linear equations", 
         Howard Elman and Y. Saad and P. E. Saylor, SIAM Journal on Scientific and Statistical Computing, 1986.
   */
  ierr = PetscOptionsBool("-ksp_chebyshev_hybrid","Use hybrid Chebyshev","",cheb->hybrid,&cheb->hybrid,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_chebyshev_hybrid_chebysteps","Number of Chebyshev steps in hybrid Chebyshev","",cheb->chebysteps,&cheb->chebysteps,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_chebyshev_hybrid_purification","Use purification in hybrid Chebyshev","",cheb->purification,&cheb->purification,PETSC_NULL);CHKERRQ(ierr);

  if (cheb->kspest) {
    /* Mask the PC so that PCSetFromOptions does not do anything */
    ierr = KSPSetPC(cheb->kspest,cheb->pcnone);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(cheb->kspest,((PetscObject)ksp)->prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(cheb->kspest,"est_");CHKERRQ(ierr);
    if (!((PetscObject)cheb->kspest)->type_name) {
      ierr = KSPSetType(cheb->kspest,KSPGMRES);CHKERRQ(ierr);
    }
    ierr = KSPSetFromOptions(cheb->kspest);CHKERRQ(ierr);
    ierr = KSPSetPC(cheb->kspest,ksp->pc);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_Chebyshev"
PetscErrorCode KSPSolve_Chebyshev(KSP ksp)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       k,kp1,km1,maxit,ktmp,i;
  PetscScalar    alpha,omegaprod,mu,omega,Gamma,c[3],scale;
  PetscReal      rnorm = 0.0;
  Vec            sol_orig,b,p[3],r;
  Mat            Amat,Pmat;
  MatStructure   pflag;
  PetscBool      diagonalscale,hybrid=cheb->hybrid;
  PetscInt       purification=cheb->purification;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  if (cheb->kspest && !cheb->estimate_current) { 
    PetscReal max,min;
    Vec       X;
    
    if (hybrid && purification){
      X = ksp->vec_sol;
    } else {
      X = ksp->work[0];
    }
    ierr = KSPSolve(cheb->kspest,ksp->vec_rhs,X);CHKERRQ(ierr);
    if (hybrid){
      cheb->its = 0; /* initialize Chebyshev iteration associated to kspest */
      ierr = KSPSetInitialGuessNonzero(cheb->kspest,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      if (ksp->guess_zero) {
        ierr = VecZeroEntries(X);CHKERRQ(ierr);
      }
    } 
    ierr = KSPComputeExtremeSingularValues(cheb->kspest,&max,&min);CHKERRQ(ierr);
    cheb->emin = cheb->tform[0]*min + cheb->tform[1]*max;
    cheb->emax = cheb->tform[2]*min + cheb->tform[3]*max;
    cheb->estimate_current = PETSC_TRUE;
  }

  ksp->its = 0;
  ierr     = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);
  maxit    = ksp->max_it;

  /* These three point to the three active solutions, we
     rotate these three at each solution update */
  km1 = 0; k = 1; kp1 = 2;
  sol_orig = ksp->vec_sol; /* ksp->vec_sol will be asigned to rotating vector p[k], thus save its address */
  b       = ksp->vec_rhs;
  p[km1]  = sol_orig;
  p[k]    = ksp->work[0];
  p[kp1]  = ksp->work[1];
  r       = ksp->work[2];

  /* use scale*B as our preconditioner */
  scale  = 2.0/(cheb->emax + cheb->emin);

  /*   -alpha <=  scale*lambda(B^{-1}A) <= alpha   */
  alpha  = 1.0 - scale*(cheb->emin); 
  Gamma  = 1.0;
  mu     = 1.0/alpha; 
  omegaprod = 2.0/alpha;

  c[km1] = 1.0;
  c[k]   = mu;

  if (!ksp->guess_zero || (hybrid && cheb->its== 0)) {
    ierr = KSP_MatMult(ksp,Amat,p[km1],r);CHKERRQ(ierr);     /*  r = b - A*p[km1] */
    ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(b,r);CHKERRQ(ierr);
  }
                  
  ierr = KSP_PCApply(ksp,r,p[k]);CHKERRQ(ierr);  /* p[k] = scale B^{-1}r + p[km1] */
  ierr = VecAYPX(p[k],scale,p[km1]);CHKERRQ(ierr);   

  for (i=0; i<maxit; i++) {
    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    if (hybrid && cheb->its && (cheb->its%cheb->chebysteps==0)){
      /* Adaptive step: update eigenvalue estimate - does not seem to improve convergence */
      PetscReal max,min;
      Vec       X = ksp->vec_sol; /* = previous p[k] */
      
      if (purification <= 1){ /* no purification here */
        X = p[km1]; /* a tmp vector, != ksp->vec_sol */
      } 

      ierr = VecCopy(p[k],X);CHKERRQ(ierr); /* p[k] = previous p[kp1] */
      ierr = KSPSolve(cheb->kspest,ksp->vec_rhs,X);CHKERRQ(ierr); 
      ierr = KSPComputeExtremeSingularValues(cheb->kspest,&max,&min);CHKERRQ(ierr);
      cheb->emin = cheb->tform[0]*min + cheb->tform[1]*max;
      cheb->emax = cheb->tform[2]*min + cheb->tform[3]*max;
      cheb->estimate_current = PETSC_TRUE;
      if (purification <= 1){ /* no purification here */
        X    = ksp->vec_sol;
        ierr = VecCopy(p[k],X);CHKERRQ(ierr);
      }

      b      = ksp->vec_rhs;
      p[km1] = X; 
      scale  = 2.0/(cheb->emax + cheb->emin);
      alpha  = 1.0 - scale*(cheb->emin); 
      mu     = 1.0/alpha; 
      omegaprod = 2.0/alpha;
    
      c[km1] = 1.0;
      c[k]   = mu; 
      ierr = KSP_MatMult(ksp,Amat,p[km1],r);CHKERRQ(ierr);     /*  r = b - A*p[km1]    */
      ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);

      ierr = KSP_PCApply(ksp,r,p[k]);CHKERRQ(ierr);  /* p[k] = scale B^{-1}r + p[km1] */
      ierr = VecAYPX(p[k],scale,p[km1]);CHKERRQ(ierr);    
    }

    ksp->its++;
    cheb->its++;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
    c[kp1] = 2.0*mu*c[k] - c[km1];
    omega = omegaprod*c[k]/c[kp1];

    ierr = KSP_MatMult(ksp,Amat,p[k],r);CHKERRQ(ierr);                 /*  r = b - Ap[k]    */
    ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);                       
    ierr = KSP_PCApply(ksp,r,p[kp1]);CHKERRQ(ierr);             /*  p[kp1] = B^{-1}r  */

    /* calculate residual norm if requested */
    if (ksp->normtype != KSP_NORM_NONE || ksp->numbermonitors) {
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);}
      else {ierr = VecNorm(p[kp1],NORM_2,&rnorm);CHKERRQ(ierr);}
      ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
      ksp->rnorm                              = rnorm;
      ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
      ksp->vec_sol = p[k]; 
      KSPLogResidualHistory(ksp,rnorm);
      ierr = KSPMonitor(ksp,i,rnorm);CHKERRQ(ierr);
      ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
      if (ksp->reason) break;
    }

    /* y^{k+1} = omega(y^{k} - y^{k-1} + Gamma*r^{k}) + y^{k-1} */
    ierr = VecScale(p[kp1],omega*Gamma*scale);CHKERRQ(ierr);
    ierr = VecAXPY(p[kp1],1.0-omega,p[km1]);CHKERRQ(ierr);
    ierr = VecAXPY(p[kp1],omega,p[k]);CHKERRQ(ierr);

    ktmp = km1;
    km1  = k;
    k    = kp1;
    kp1  = ktmp;
  }
  if (!ksp->reason) {
    if (ksp->normtype != KSP_NORM_NONE) {
      ierr = KSP_MatMult(ksp,Amat,p[k],r);CHKERRQ(ierr);       /*  r = b - Ap[k]    */
      ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
	ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);
      } else {
	ierr = KSP_PCApply(ksp,r,p[kp1]);CHKERRQ(ierr); /* p[kp1] = B^{-1}r */
	ierr = VecNorm(p[kp1],NORM_2,&rnorm);CHKERRQ(ierr);
      }
      ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
      ksp->rnorm = rnorm;
      ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
      ksp->vec_sol = p[k]; 
      KSPLogResidualHistory(ksp,rnorm);
      ierr = KSPMonitor(ksp,i,rnorm);CHKERRQ(ierr);
    }
    if (ksp->its >= ksp->max_it) {
      if (ksp->normtype != KSP_NORM_NONE) {
	ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
	if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
      } else { 
	ksp->reason = KSP_CONVERGED_ITS;
      }
    }
  }

  /* make sure solution is in vector x */
  ksp->vec_sol = sol_orig;
  if (k) {
    ierr = VecCopy(p[k],sol_orig);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPView_Chebyshev" 
PetscErrorCode KSPView_Chebyshev(KSP ksp,PetscViewer viewer)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Chebyshev: eigenvalue estimates:  min = %G, max = %G\n",cheb->emin,cheb->emax);CHKERRQ(ierr);
    if (cheb->kspest) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Chebyshev: estimated using:  [%G %G; %G %G]\n",cheb->tform[0],cheb->tform[1],cheb->tform[2],cheb->tform[3]);CHKERRQ(ierr);
      if (cheb->hybrid){ /* display info about hybrid options being used */
        ierr = PetscViewerASCIIPrintf(viewer,"  Chebyshev: hybrid is used, chebysteps %D, purification %D\n",cheb->chebysteps,cheb->purification);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = KSPView(cheb->kspest,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for KSP Chebyshev",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_Chebyshev"
PetscErrorCode KSPDestroy_Chebyshev(KSP ksp)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDestroy(&cheb->kspest);CHKERRQ(ierr);
  ierr = PCDestroy(&cheb->pcnone);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPChebyshevSetEigenvalues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPChebyshevSetEstimateEigenvalues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPChebyshevSetNewMatrix_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = KSPDefaultDestroy(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPCHEBYSHEV - The preconditioned Chebyshev iterative method

   Options Database Keys:
+   -ksp_chebyshev_eigenvalues <emin,emax> - set approximations to the smallest and largest eigenvalues
                  of the preconditioned operator. If these are accurate you will get much faster convergence.
-   -ksp_chebyshev_estimate_eigenvalues <a,b,c,d> - estimate eigenvalues using a Krylov method, then use this
                  transform for Chebyshev eigenvalue bounds (KSPChebyshevSetEstimateEigenvalues)


   Level: beginner

   Notes: The Chebyshev method requires both the matrix and preconditioner to 
          be symmetric positive (semi) definite.
          Only support for left preconditioning.

          Chebyshev is configured as a smoother by default, targetting the "upper" part of the spectrum.
          The user should call KSPChebyshevSetEigenvalues() if they have eigenvalue estimates.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPChebyshevSetEigenvalues(), KSPChebyshevSetEstimateEigenvalues(), KSPRICHARDSON, KSPCG, PCMG

M*/

#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_Chebyshev"
PETSC_EXTERN_C PetscErrorCode KSPCreate_Chebyshev(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_Chebyshev  *chebyshevP;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_Chebyshev,&chebyshevP);CHKERRQ(ierr);

  ksp->data                      = (void*)chebyshevP;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,1);CHKERRQ(ierr);

  chebyshevP->emin               = 0.;
  chebyshevP->emax               = 0.;

  chebyshevP->tform[0]           = 0.0;
  chebyshevP->tform[1]           = 0.1;
  chebyshevP->tform[2]           = 0;
  chebyshevP->tform[3]           = 1.1;

  chebyshevP->hybrid             = PETSC_FALSE;
  chebyshevP->chebysteps         = 20000;
  chebyshevP->its                = 0;
  chebyshevP->purification       = 0; /* no purification */

  ksp->ops->setup                = KSPSetUp_Chebyshev;
  ksp->ops->solve                = KSPSolve_Chebyshev;
  ksp->ops->destroy              = KSPDestroy_Chebyshev;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions       = KSPSetFromOptions_Chebyshev;
  ksp->ops->view                 = KSPView_Chebyshev;
  ksp->ops->reset                = KSPReset_Chebyshev;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPChebyshevSetEigenvalues_C",
                                    "KSPChebyshevSetEigenvalues_Chebyshev",
                                    KSPChebyshevSetEigenvalues_Chebyshev);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPChebyshevSetEstimateEigenvalues_C",
                                    "KSPChebyshevSetEstimateEigenvalues_Chebyshev",
                                    KSPChebyshevSetEstimateEigenvalues_Chebyshev);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPChebyshevSetNewMatrix_C",
                                    "KSPChebyshevSetNewMatrix_Chebyshev",
                                    KSPChebyshevSetNewMatrix_Chebyshev);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
