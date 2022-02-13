
#include <../src/snes/impls/vi/ss/vissimpl.h> /*I "petscsnes.h" I*/

/*
  SNESVIComputeMeritFunction - Evaluates the merit function for the mixed complementarity problem.

  Input Parameter:
. phi - the semismooth function

  Output Parameter:
. merit - the merit function
. phinorm - ||phi||

  Notes:
  The merit function for the mixed complementarity problem is defined as
     merit = 0.5*phi^T*phi
*/
static PetscErrorCode SNESVIComputeMeritFunction(Vec phi, PetscReal *merit,PetscReal *phinorm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNormBegin(phi,NORM_2,phinorm);CHKERRQ(ierr);
  ierr = VecNormEnd(phi,NORM_2,phinorm);CHKERRQ(ierr);

  *merit = 0.5*(*phinorm)*(*phinorm);
  PetscFunctionReturn(0);
}

static inline PetscScalar Phi(PetscScalar a,PetscScalar b)
{
  return a + b - PetscSqrtScalar(a*a + b*b);
}

static inline PetscScalar DPhi(PetscScalar a,PetscScalar b)
{
  if ((PetscAbsScalar(a) >= 1.e-6) || (PetscAbsScalar(b) >= 1.e-6)) return 1.0 - a/ PetscSqrtScalar(a*a + b*b);
  else return .5;
}

/*
   SNESVIComputeFunction - Reformulates a system of nonlinear equations in mixed complementarity form to a system of nonlinear equations in semismooth form.

   Input Parameters:
.  snes - the SNES context
.  X - current iterate
.  functx - user defined function context

   Output Parameters:
.  phi - Semismooth function

*/
static PetscErrorCode SNESVIComputeFunction(SNES snes,Vec X,Vec phi,void *functx)
{
  PetscErrorCode    ierr;
  SNES_VINEWTONSSLS *vi = (SNES_VINEWTONSSLS*)snes->data;
  Vec               Xl  = snes->xl,Xu = snes->xu,F = snes->vec_func;
  PetscScalar       *phi_arr,*f_arr,*l,*u;
  const PetscScalar *x_arr;
  PetscInt          i,nlocal;

  PetscFunctionBegin;
  ierr = (*vi->computeuserfunction)(snes,X,F,functx);CHKERRQ(ierr);
  ierr = VecGetLocalSize(X,&nlocal);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x_arr);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f_arr);CHKERRQ(ierr);
  ierr = VecGetArray(Xl,&l);CHKERRQ(ierr);
  ierr = VecGetArray(Xu,&u);CHKERRQ(ierr);
  ierr = VecGetArray(phi,&phi_arr);CHKERRQ(ierr);

  for (i=0; i < nlocal; i++) {
    if ((PetscRealPart(l[i]) <= PETSC_NINFINITY) && (PetscRealPart(u[i]) >= PETSC_INFINITY)) { /* no constraints on variable */
      phi_arr[i] = f_arr[i];
    } else if (PetscRealPart(l[i]) <= PETSC_NINFINITY) {                      /* upper bound on variable only */
      phi_arr[i] = -Phi(u[i] - x_arr[i],-f_arr[i]);
    } else if (PetscRealPart(u[i]) >= PETSC_INFINITY) {                       /* lower bound on variable only */
      phi_arr[i] = Phi(x_arr[i] - l[i],f_arr[i]);
    } else if (l[i] == u[i]) {
      phi_arr[i] = l[i] - x_arr[i];
    } else {                                                /* both bounds on variable */
      phi_arr[i] = Phi(x_arr[i] - l[i],-Phi(u[i] - x_arr[i],-f_arr[i]));
    }
  }

  ierr = VecRestoreArrayRead(X,&x_arr);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f_arr);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xl,&l);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xu,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(phi,&phi_arr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   SNESVIComputeBsubdifferentialVectors - Computes the diagonal shift (Da) and row scaling (Db) vectors needed for the
                                          the semismooth jacobian.
*/
PetscErrorCode SNESVIComputeBsubdifferentialVectors(SNES snes,Vec X,Vec F,Mat jac,Vec Da,Vec Db)
{
  PetscErrorCode ierr;
  PetscScalar    *l,*u,*x,*f,*da,*db,da1,da2,db1,db2;
  PetscInt       i,nlocal;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = VecGetArray(snes->xl,&l);CHKERRQ(ierr);
  ierr = VecGetArray(snes->xu,&u);CHKERRQ(ierr);
  ierr = VecGetArray(Da,&da);CHKERRQ(ierr);
  ierr = VecGetArray(Db,&db);CHKERRQ(ierr);
  ierr = VecGetLocalSize(X,&nlocal);CHKERRQ(ierr);

  for (i=0; i< nlocal; i++) {
    if ((PetscRealPart(l[i]) <= PETSC_NINFINITY) && (PetscRealPart(u[i]) >= PETSC_INFINITY)) { /* no constraints on variable */
      da[i] = 0;
      db[i] = 1;
    } else if (PetscRealPart(l[i]) <= PETSC_NINFINITY) {                     /* upper bound on variable only */
      da[i] = DPhi(u[i] - x[i], -f[i]);
      db[i] = DPhi(-f[i],u[i] - x[i]);
    } else if (PetscRealPart(u[i]) >= PETSC_INFINITY) {                      /* lower bound on variable only */
      da[i] = DPhi(x[i] - l[i], f[i]);
      db[i] = DPhi(f[i],x[i] - l[i]);
    } else if (l[i] == u[i]) {                              /* fixed variable */
      da[i] = 1;
      db[i] = 0;
    } else {                                                /* upper and lower bounds on variable */
      da1   = DPhi(x[i] - l[i], -Phi(u[i] - x[i], -f[i]));
      db1   = DPhi(-Phi(u[i] - x[i], -f[i]),x[i] - l[i]);
      da2   = DPhi(u[i] - x[i], -f[i]);
      db2   = DPhi(-f[i],u[i] - x[i]);
      da[i] = da1 + db1*da2;
      db[i] = db1*db2;
    }
  }

  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = VecRestoreArray(snes->xl,&l);CHKERRQ(ierr);
  ierr = VecRestoreArray(snes->xu,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(Da,&da);CHKERRQ(ierr);
  ierr = VecRestoreArray(Db,&db);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   SNESVIComputeJacobian - Computes the jacobian of the semismooth function.The Jacobian for the semismooth function is an element of the B-subdifferential of the Fischer-Burmeister function for complementarity problems.

   Input Parameters:
.  Da       - Diagonal shift vector for the semismooth jacobian.
.  Db       - Row scaling vector for the semismooth jacobian.

   Output Parameters:
.  jac      - semismooth jacobian
.  jac_pre  - optional preconditioning matrix

   Notes:
   The semismooth jacobian matrix is given by
   jac = Da + Db*jacfun
   where Db is the row scaling matrix stored as a vector,
         Da is the diagonal perturbation matrix stored as a vector
   and   jacfun is the jacobian of the original nonlinear function.
*/
PetscErrorCode SNESVIComputeJacobian(Mat jac, Mat jac_pre,Vec Da, Vec Db)
{
  PetscErrorCode ierr;

  /* Do row scaling  and add diagonal perturbation */
  PetscFunctionBegin;
  ierr = MatDiagonalScale(jac,Db,NULL);CHKERRQ(ierr);
  ierr = MatDiagonalSet(jac,Da,ADD_VALUES);CHKERRQ(ierr);
  if (jac != jac_pre) { /* If jac and jac_pre are different */
    ierr = MatDiagonalScale(jac_pre,Db,NULL);CHKERRQ(ierr);
    ierr = MatDiagonalSet(jac_pre,Da,ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   SNESVIComputeMeritFunctionGradient - Computes the gradient of the merit function psi.

   Input Parameters:
   phi - semismooth function.
   H   - semismooth jacobian

   Output Parameters:
   dpsi - merit function gradient

   Notes:
  The merit function gradient is computed as follows
        dpsi = H^T*phi
*/
PetscErrorCode SNESVIComputeMeritFunctionGradient(Mat H, Vec phi, Vec dpsi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultTranspose(H,phi,dpsi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   SNESSolve_VINEWTONSSLS - Solves the complementarity problem with a semismooth Newton
   method using a line search.

   Input Parameters:
.  snes - the SNES context

   Application Interface Routine: SNESSolve()

   Notes:
   This implements essentially a semismooth Newton method with a
   line search. The default line search does not do any line search
   but rather takes a full Newton step.

   Developer Note: the code in this file should be slightly modified so that this routine need not exist and the SNESSolve_NEWTONLS() routine is called directly with the appropriate wrapped function and Jacobian evaluations

*/
PetscErrorCode SNESSolve_VINEWTONSSLS(SNES snes)
{
  SNES_VINEWTONSSLS    *vi = (SNES_VINEWTONSSLS*)snes->data;
  PetscErrorCode       ierr;
  PetscInt             maxits,i,lits;
  SNESLineSearchReason lssucceed;
  PetscReal            gnorm,xnorm=0,ynorm;
  Vec                  Y,X,F;
  KSPConvergedReason   kspreason;
  DM                   dm;
  DMSNES               sdm;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);

  vi->computeuserfunction   = sdm->ops->computefunction;
  sdm->ops->computefunction = SNESVIComputeFunction;

  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;               /* maximum number of iterations */
  X      = snes->vec_sol;               /* solution vector */
  F      = snes->vec_func;              /* residual vector */
  Y      = snes->work[0];               /* work vectors */

  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.0;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);

  ierr = SNESVIProjectOntoBounds(snes,X);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,vi->phi);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason              = SNES_DIVERGED_FUNCTION_DOMAIN;
    sdm->ops->computefunction = vi->computeuserfunction;
    PetscFunctionReturn(0);
  }
  /* Compute Merit function */
  ierr = SNESVIComputeMeritFunction(vi->phi,&vi->merit,&vi->phinorm);CHKERRQ(ierr);

  ierr = VecNormBegin(X,NORM_2,&xnorm);CHKERRQ(ierr);        /* xnorm <- ||x||  */
  ierr = VecNormEnd(X,NORM_2,&xnorm);CHKERRQ(ierr);
  SNESCheckFunctionNorm(snes,vi->merit);

  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->norm = vi->phinorm;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  ierr       = SNESLogConvergenceHistory(snes,vi->phinorm,0);CHKERRQ(ierr);
  ierr       = SNESMonitor(snes,0,vi->phinorm);CHKERRQ(ierr);

  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,vi->phinorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) {
    sdm->ops->computefunction = vi->computeuserfunction;
    PetscFunctionReturn(0);
  }

  for (i=0; i<maxits; i++) {

    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }

    /* Solve J Y = Phi, where J is the semismooth jacobian */

    /* Get the jacobian -- note that the function must be the original function for snes_fd and snes_fd_color to work for this*/
    sdm->ops->computefunction = vi->computeuserfunction;
    ierr                      = SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
    SNESCheckJacobianDomainerror(snes);
    sdm->ops->computefunction = SNESVIComputeFunction;

    /* Get the diagonal shift and row scaling vectors */
    ierr = SNESVIComputeBsubdifferentialVectors(snes,X,F,snes->jacobian,vi->Da,vi->Db);CHKERRQ(ierr);
    /* Compute the semismooth jacobian */
    ierr = SNESVIComputeJacobian(snes->jacobian,snes->jacobian_pre,vi->Da,vi->Db);CHKERRQ(ierr);
    /* Compute the merit function gradient */
    ierr = SNESVIComputeMeritFunctionGradient(snes->jacobian,vi->phi,vi->dpsi);CHKERRQ(ierr);
    ierr = KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
    ierr = KSPSolve(snes->ksp,vi->phi,Y);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(snes->ksp,&kspreason);CHKERRQ(ierr);

    if (kspreason < 0) {
      if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
        ierr         = PetscInfo(snes,"iter=%D, number linear solve failures %D greater than current SNES allowed, stopping solve\n",snes->iter,snes->numLinearSolveFailures);CHKERRQ(ierr);
        snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
        break;
      }
    }
    ierr              = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);
    snes->linear_its += lits;
    ierr              = PetscInfo(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits);CHKERRQ(ierr);
    /*
    if (snes->ops->precheck) {
      PetscBool changed_y = PETSC_FALSE;
      ierr = (*snes->ops->precheck)(snes,X,Y,snes->precheck,&changed_y);CHKERRQ(ierr);
    }

    if (PetscLogPrintInfo) {
      ierr = SNESVICheckResidual_Private(snes,snes->jacobian,F,Y,G,W);CHKERRQ(ierr);
    }
    */
    /* Compute a (scaled) negative update in the line search routine:
         Y <- X - lambda*Y
       and evaluate G = function(Y) (depends on the line search).
    */
    ierr  = VecCopy(Y,snes->vec_sol_update);CHKERRQ(ierr);
    ynorm = 1; gnorm = vi->phinorm;
    ierr = SNESLineSearchApply(snes->linesearch, X, vi->phi, &gnorm, Y);CHKERRQ(ierr);
    ierr = SNESLineSearchGetReason(snes->linesearch, &lssucceed);CHKERRQ(ierr);
    ierr = SNESLineSearchGetNorms(snes->linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);
    ierr = PetscInfo(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",(double)vi->phinorm,(double)gnorm,(double)ynorm,(int)lssucceed);CHKERRQ(ierr);
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;
    if (snes->domainerror) {
      snes->reason              = SNES_DIVERGED_FUNCTION_DOMAIN;
      sdm->ops->computefunction = vi->computeuserfunction;
      PetscFunctionReturn(0);
    }
    if (lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
        PetscBool ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        ierr         = SNESVICheckLocalMin_Private(snes,snes->jacobian,vi->phi,X,gnorm,&ismin);CHKERRQ(ierr);
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
    }
    /* Update function and solution vectors */
    vi->phinorm = gnorm;
    vi->merit   = 0.5*vi->phinorm*vi->phinorm;
    /* Monitor convergence */
    ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = vi->phinorm;
    snes->xnorm = xnorm;
    snes->ynorm = ynorm;
    ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr       = SNESLogConvergenceHistory(snes,snes->norm,lits);CHKERRQ(ierr);
    ierr       = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence, xnorm = || X || */
    if (snes->ops->converged != SNESConvergedSkip) { ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr); }
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,vi->phinorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  if (i == maxits) {
    ierr = PetscInfo(snes,"Maximum number of iterations has been reached: %D\n",maxits);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  sdm->ops->computefunction = vi->computeuserfunction;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   SNESSetUp_VINEWTONSSLS - Sets up the internal data structures for the later use
   of the SNES nonlinear solver.

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESSetUp()

   Notes:
   For basic use of the SNES solvers, the user need not explicitly call
   SNESSetUp(), since these actions will automatically occur during
   the call to SNESSolve().
 */
PetscErrorCode SNESSetUp_VINEWTONSSLS(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_VINEWTONSSLS *vi = (SNES_VINEWTONSSLS*) snes->data;

  PetscFunctionBegin;
  ierr = SNESSetUp_VI(snes);CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_sol, &vi->dpsi);CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_sol, &vi->phi);CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_sol, &vi->Da);CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_sol, &vi->Db);CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_sol, &vi->z);CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_sol, &vi->t);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
PetscErrorCode SNESReset_VINEWTONSSLS(SNES snes)
{
  SNES_VINEWTONSSLS *vi = (SNES_VINEWTONSSLS*) snes->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = SNESReset_VI(snes);CHKERRQ(ierr);
  ierr = VecDestroy(&vi->dpsi);CHKERRQ(ierr);
  ierr = VecDestroy(&vi->phi);CHKERRQ(ierr);
  ierr = VecDestroy(&vi->Da);CHKERRQ(ierr);
  ierr = VecDestroy(&vi->Db);CHKERRQ(ierr);
  ierr = VecDestroy(&vi->z);CHKERRQ(ierr);
  ierr = VecDestroy(&vi->t);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   SNESSetFromOptions_VINEWTONSSLS - Sets various parameters for the SNESVI method.

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESSetFromOptions()
*/
static PetscErrorCode SNESSetFromOptions_VINEWTONSSLS(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESSetFromOptions_VI(PetscOptionsObject,snes);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"SNES semismooth method options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
      SNESVINEWTONSSLS - Semi-smooth solver for variational inequalities based on Newton's method

   Options Database:
+   -snes_type <vinewtonssls,vinewtonrsls> a semi-smooth solver, a reduced space active set method
-   -snes_vi_monitor - prints the number of active constraints at each iteration.

   Level: beginner

   References:
+  1. -  T. S. Munson, F. Facchinei, M. C. Ferris, A. Fischer, and C. Kanzow. The semismooth
     algorithm for large scale complementarity problems. INFORMS Journal on Computing, 13 (2001).
-  2. -  T. S. Munson, and S. Benson. Flexible Complementarity Solvers for Large Scale
     Applications, Optimization Methods and Software, 21 (2006).

.seealso:  SNESVISetVariableBounds(), SNESVISetComputeVariableBounds(), SNESCreate(), SNES, SNESSetType(), SNESVINEWTONRSLS, SNESNEWTONTR, SNESLineSearchSetType(),SNESLineSearchSetPostCheck(), SNESLineSearchSetPreCheck()

M*/
PETSC_EXTERN PetscErrorCode SNESCreate_VINEWTONSSLS(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_VINEWTONSSLS *vi;
  SNESLineSearch    linesearch;

  PetscFunctionBegin;
  snes->ops->reset          = SNESReset_VINEWTONSSLS;
  snes->ops->setup          = SNESSetUp_VINEWTONSSLS;
  snes->ops->solve          = SNESSolve_VINEWTONSSLS;
  snes->ops->destroy        = SNESDestroy_VI;
  snes->ops->setfromoptions = SNESSetFromOptions_VINEWTONSSLS;
  snes->ops->view           = NULL;

  snes->usesksp = PETSC_TRUE;
  snes->usesnpc = PETSC_FALSE;

  ierr = SNESGetLineSearch(snes, &linesearch);CHKERRQ(ierr);
  if (!((PetscObject)linesearch)->type_name) {
    ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHBT);CHKERRQ(ierr);
    ierr = SNESLineSearchBTSetAlpha(linesearch, 0.0);CHKERRQ(ierr);
  }

  snes->alwayscomputesfinalresidual = PETSC_FALSE;

  ierr       = PetscNewLog(snes,&vi);CHKERRQ(ierr);
  snes->data = (void*)vi;

  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESVISetVariableBounds_C",SNESVISetVariableBounds_VI);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESVISetComputeVariableBounds_C",SNESVISetComputeVariableBounds_VI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

