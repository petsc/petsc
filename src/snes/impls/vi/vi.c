#define PETSCSNES_DLL

#include "../src/snes/impls/vi/viimpl.h"
#include "../include/private/kspimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorVI"
PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorVI(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
{
  PetscErrorCode          ierr;
  SNES_VI                 *vi = (SNES_VI*)snes->data;
  PetscViewerASCIIMonitor viewer = (PetscViewerASCIIMonitor) dummy;
  const PetscScalar       *x,*xl,*xu,*f;
  PetscInt                i,n,act = 0;
  PetscReal               rnorm,fnorm;

  PetscFunctionBegin;
  if (!dummy) {
    ierr = PetscViewerASCIIMonitorCreate(((PetscObject)snes)->comm,"stdout",0,&viewer);CHKERRQ(ierr);
  }
  ierr = VecGetLocalSize(snes->vec_sol,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vi->xl,&xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vi->xu,&xu);CHKERRQ(ierr);
  ierr = VecGetArrayRead(snes->vec_sol,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(snes->vec_func,&f);CHKERRQ(ierr);
  
  rnorm = 0.0;
  for (i=0; i<n; i++) {
    if (((x[i] > xl[i] + 1.e-8 || (f[i] < 0.0)) && ((x[i] < xu[i] - 1.e-8) || f[i] > 0.0))) rnorm += f[i]*f[i];
    else act++;
  }
  ierr = VecRestoreArrayRead(snes->vec_func,&f);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vi->xl,&xl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vi->xu,&xu);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(snes->vec_sol,&x);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&rnorm,&fnorm,1,MPIU_REAL,MPI_SUM,((PetscObject)snes)->comm);CHKERRQ(ierr);
  fnorm = sqrt(fnorm);
  ierr = PetscViewerASCIIMonitorPrintf(viewer,"%3D SNES VI Function norm %14.12e Active constraints %D\n",its,fnorm,act);CHKERRQ(ierr);
  if (!dummy) {
    ierr = PetscViewerASCIIMonitorDestroy(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
     Checks if J^T F = 0 which implies we've found a local minimum of the norm of the function,
    || F(u) ||_2 but not a zero, F(u) = 0. In the case when one cannot compute J^T F we use the fact that
    0 = (J^T F)^T W = F^T J W iff W not in the null space of J. Thanks for Jorge More 
    for this trick. One assumes that the probability that W is in the null space of J is very, very small.
*/ 
#undef __FUNCT__  
#define __FUNCT__ "SNESVICheckLocalMin_Private"
PetscErrorCode SNESVICheckLocalMin_Private(SNES snes,Mat A,Vec F,Vec W,PetscReal fnorm,PetscBool *ismin)
{
  PetscReal      a1;
  PetscErrorCode ierr;
  PetscBool     hastranspose;

  PetscFunctionBegin;
  *ismin = PETSC_FALSE;
  ierr = MatHasOperation(A,MATOP_MULT_TRANSPOSE,&hastranspose);CHKERRQ(ierr);
  if (hastranspose) {
    /* Compute || J^T F|| */
    ierr = MatMultTranspose(A,F,W);CHKERRQ(ierr);
    ierr = VecNorm(W,NORM_2,&a1);CHKERRQ(ierr);
    ierr = PetscInfo1(snes,"|| J^T F|| %G near zero implies found a local minimum\n",a1/fnorm);CHKERRQ(ierr);
    if (a1/fnorm < 1.e-4) *ismin = PETSC_TRUE;
  } else {
    Vec         work;
    PetscScalar result;
    PetscReal   wnorm;

    ierr = VecSetRandom(W,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecNorm(W,NORM_2,&wnorm);CHKERRQ(ierr);
    ierr = VecDuplicate(W,&work);CHKERRQ(ierr);
    ierr = MatMult(A,W,work);CHKERRQ(ierr);
    ierr = VecDot(F,work,&result);CHKERRQ(ierr);
    ierr = VecDestroy(work);CHKERRQ(ierr);
    a1   = PetscAbsScalar(result)/(fnorm*wnorm);
    ierr = PetscInfo1(snes,"(F^T J random)/(|| F ||*||random|| %G near zero implies found a local minimum\n",a1);CHKERRQ(ierr);
    if (a1 < 1.e-4) *ismin = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*
     Checks if J^T(F - J*X) = 0 
*/ 
#undef __FUNCT__  
#define __FUNCT__ "SNESVICheckResidual_Private"
PetscErrorCode SNESVICheckResidual_Private(SNES snes,Mat A,Vec F,Vec X,Vec W1,Vec W2)
{
  PetscReal      a1,a2;
  PetscErrorCode ierr;
  PetscBool     hastranspose;

  PetscFunctionBegin;
  ierr = MatHasOperation(A,MATOP_MULT_TRANSPOSE,&hastranspose);CHKERRQ(ierr);
  if (hastranspose) {
    ierr = MatMult(A,X,W1);CHKERRQ(ierr);
    ierr = VecAXPY(W1,-1.0,F);CHKERRQ(ierr);

    /* Compute || J^T W|| */
    ierr = MatMultTranspose(A,W1,W2);CHKERRQ(ierr);
    ierr = VecNorm(W1,NORM_2,&a1);CHKERRQ(ierr);
    ierr = VecNorm(W2,NORM_2,&a2);CHKERRQ(ierr);
    if (a1 != 0.0) {
      ierr = PetscInfo1(snes,"||J^T(F-Ax)||/||F-AX|| %G near zero implies inconsistent rhs\n",a2/a1);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
  SNESDefaultConverged_VI - Checks the convergence of the semismooth newton algorithm.

  Notes:
  The convergence criterion currently implemented is
  merit < abstol
  merit < rtol*merit_initial
*/
#undef __FUNCT__
#define __FUNCT__ "SNESDefaultConverged_VI"
PetscErrorCode SNESDefaultConverged_VI(SNES snes,PetscInt it,PetscReal xnorm,PetscReal gradnorm,PetscReal merit,SNESConvergedReason *reason,void *dummy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(reason,6);
  
  *reason = SNES_CONVERGED_ITERATING;

  if (!it) {
    /* set parameter for default relative tolerance convergence test */
    snes->ttol = merit*snes->rtol;
  }
  if (merit != merit) {
    ierr = PetscInfo(snes,"Failed to converged, function norm is NaN\n");CHKERRQ(ierr);
    *reason = SNES_DIVERGED_FNORM_NAN;
  } else if (merit < snes->abstol) {
    ierr = PetscInfo2(snes,"Converged due to merit function %G < %G\n",merit,snes->abstol);CHKERRQ(ierr);
    *reason = SNES_CONVERGED_FNORM_ABS;
  } else if (snes->nfuncs >= snes->max_funcs) {
    ierr = PetscInfo2(snes,"Exceeded maximum number of function evaluations: %D > %D\n",snes->nfuncs,snes->max_funcs);CHKERRQ(ierr);
    *reason = SNES_DIVERGED_FUNCTION_COUNT;
  }

  if (it && !*reason) {
    if (merit < snes->ttol) {
      ierr = PetscInfo2(snes,"Converged due to merit function %G < %G (relative tolerance)\n",merit,snes->ttol);CHKERRQ(ierr);
      *reason = SNES_CONVERGED_FNORM_RELATIVE;
    }
  } 
  PetscFunctionReturn(0);
}

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
#undef __FUNCT__
#define __FUNCT__ "SNESVIComputeMeritFunction"
static PetscErrorCode SNESVIComputeMeritFunction(Vec phi, PetscReal* merit,PetscReal* phinorm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNormBegin(phi,NORM_2,phinorm);
  ierr = VecNormEnd(phi,NORM_2,phinorm);

  *merit = 0.5*(*phinorm)*(*phinorm);
  PetscFunctionReturn(0);
}

/*
  ComputeFischerFunction - Computes the semismooth fischer burmeister function for a mixed complementarity equation.

  Notes:
  The Fischer-Burmeister function is defined as
       ff(a,b) := sqrt(a*a + b*b) - a - b                                                            
  and is used reformulate a complementarity equation  as a semismooth equation.    
*/

#undef __FUNCT__
#define __FUNCT__ "ComputeFischerFunction"
static PetscErrorCode ComputeFischerFunction(PetscScalar a, PetscScalar b, PetscScalar* ff)
{
  PetscFunctionBegin;
  *ff = sqrt(a*a + b*b) - a - b;
  PetscFunctionReturn(0);
}
  
/* 
   SNESVIComputeFunction - Reformulates a system of nonlinear equations in mixed complementarity form to a system of nonlinear equations in semismooth form. 

   Input Parameters:  
.  snes - the SNES context
.  x - current iterate
.  functx - user defined function context

   Output Parameters:
.  phi - Semismooth function

   Notes:
   The result of this function is done by cases:
+  l[i] == -infinity, u[i] == infinity  -- phi[i] = -f[i]
.  l[i] == -infinity, u[i] finite       -- phi[i] = ss(u[i]-x[i], -f[i])                             
.  l[i] finite,       u[i] == infinity  -- phi[i] = ss(x[i]-l[i],  f[i])                              
.  l[i] finite < u[i] finite -- phi[i] = phi(x[i]-l[i], ss(u[i]-x[i], -f[u])) 
-  otherwise l[i] == u[i] -- phi[i] = l[i] - x[i]
   ss is the semismoothing function used to reformulate the nonlinear equation in complementarity
   form to semismooth form

*/
#undef __FUNCT__
#define __FUNCT__ "SNESVIComputeFunction"
static PetscErrorCode SNESVIComputeFunction(SNES snes,Vec X,Vec phi,void* functx)
{
  PetscErrorCode  ierr;
  SNES_VI       *vi = (SNES_VI*)snes->data;
  Vec             Xl = vi->xl,Xu = vi->xu,F = snes->vec_func;
  PetscScalar     *phi_arr,*x_arr,*f_arr,*l,*u,t;
  PetscInt        i,nlocal;

  PetscFunctionBegin;
  ierr = (*vi->computeuserfunction)(snes,X,F,functx);CHKERRQ(ierr);
  ierr = VecGetLocalSize(X,&nlocal);CHKERRQ(ierr);

  ierr = VecGetArray(X,&x_arr);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f_arr);CHKERRQ(ierr);
  ierr = VecGetArray(Xl,&l);CHKERRQ(ierr);
  ierr = VecGetArray(Xu,&u);CHKERRQ(ierr);
  ierr = VecGetArray(phi,&phi_arr);CHKERRQ(ierr);

  for (i=0;i < nlocal;i++) {
    if ((l[i] <= PETSC_VI_NINF) && (u[i] >= PETSC_VI_INF)) {
      phi_arr[i] = -f_arr[i];
    }
    else if (l[i] <= PETSC_VI_NINF) {
      t = u[i] - x_arr[i];
      ierr = ComputeFischerFunction(t,-f_arr[i],&phi_arr[i]);CHKERRQ(ierr);
      phi_arr[i] = -phi_arr[i];
    }
    else if (u[i] >= PETSC_VI_INF) {
      t = x_arr[i] - l[i];
      ierr = ComputeFischerFunction(t,f_arr[i],&phi_arr[i]);CHKERRQ(ierr);
    }
    else if (l[i] == u[i]) {
      phi_arr[i] = l[i] - x_arr[i];
    }
    else {
      t = u[i] - x_arr[i];
      ierr = ComputeFischerFunction(t,-f_arr[i],&phi_arr[i]);
      t = x_arr[i] - l[i];
      ierr = ComputeFischerFunction(t,phi_arr[i],&phi_arr[i]);
    }
  }
  
  ierr = VecRestoreArray(X,&x_arr);CHKERRQ(ierr);
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
#undef __FUNCT__
#define __FUNCT__ "SNESVIComputeBsubdifferentialVectors"
PetscErrorCode SNESVIComputeBsubdifferentialVectors(SNES snes,Vec X,Vec F,Mat jac,Vec Da,Vec Db)
{
  PetscErrorCode ierr;
  SNES_VI      *vi = (SNES_VI*)snes->data;
  PetscScalar    *l,*u,*x,*f,*da,*db,*z,*t,t1,t2,ci,di,ei;
  PetscInt       i,nlocal;

  PetscFunctionBegin;

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = VecGetArray(vi->xl,&l);CHKERRQ(ierr);
  ierr = VecGetArray(vi->xu,&u);CHKERRQ(ierr);
  ierr = VecGetArray(Da,&da);CHKERRQ(ierr);
  ierr = VecGetArray(Db,&db);CHKERRQ(ierr);
  ierr = VecGetArray(vi->z,&z);CHKERRQ(ierr);
  
  ierr = VecGetLocalSize(X,&nlocal);CHKERRQ(ierr);
  /* Set the elements of the vector z:
     z[i] = 1 if (x[i] - l[i],f[i]) = (0,0) or (u[i] - x[i],f[i]) = (0,0)
     else z[i] = 0
  */
  for(i=0;i < nlocal;i++) {
    da[i] = db[i] = z[i] = 0;
    if(PetscAbsScalar(f[i]) <= vi->const_tol) {
      if ((l[i] > PETSC_VI_NINF) && (PetscAbsScalar(x[i]-l[i]) <= vi->const_tol)) {
	da[i] = 1;
	z[i]  = 1;
      }
      if ((u[i] < PETSC_VI_INF) && (PetscAbsScalar(u[i]-x[i]) <= vi->const_tol)) {
	db[i] = 1;
	z[i]  = 1;
      }
    }
  }
  ierr = VecRestoreArray(vi->z,&z);CHKERRQ(ierr);
  ierr = MatMult(jac,vi->z,vi->t);CHKERRQ(ierr);
  ierr = VecGetArray(vi->t,&t);CHKERRQ(ierr);
  /* Compute the elements of the diagonal perturbation vector Da and row scaling vector Db */
  for(i=0;i< nlocal;i++) {
    /* Free variables */
    if ((l[i] <= PETSC_VI_NINF) && (u[i] >= PETSC_VI_INF)) {
      da[i] = 0; db[i] = -1;
    }
    /* lower bounded variables */
    else if (u[i] >= PETSC_VI_INF) {
      if (da[i] >= 1) {
	t2 = PetscScalarNorm(1,t[i]);
	da[i] = 1/t2 - 1;
	db[i] = t[i]/t2 - 1;
      } else {
	t1 = x[i] - l[i];
	t2 = PetscScalarNorm(t1,f[i]);
	da[i] = t1/t2 - 1;
	db[i] = f[i]/t2 - 1;
      }
    }
    /* upper bounded variables */
    else if (l[i] <= PETSC_VI_NINF) {
      if (db[i] >= 1) {
	t2 = PetscScalarNorm(1,t[i]);
	da[i] = -1/t2 -1;
	db[i] = -t[i]/t2 - 1;
      }
      else {
	t1 = u[i] - x[i];
	t2 = PetscScalarNorm(t1,f[i]);
	da[i] = t1/t2 - 1;
	db[i] = -f[i]/t2 - 1;
      }
    }
    /* Fixed variables */
    else if (l[i] == u[i]) {
      da[i] = -1;
      db[i] = 0;
    }
    /* Box constrained variables */
    else {
      if (db[i] >= 1) {
	t2 = PetscScalarNorm(1,t[i]);
	ci = 1/t2 + 1;
	di = t[i]/t2 + 1;
      }
      else {
	t1 = x[i] - u[i];
	t2 = PetscScalarNorm(t1,f[i]);
	ci = t1/t2 + 1;
	di = f[i]/t2 + 1;
      }
      
      if (da[i] >= 1) {
	t1 = ci + di*t[i];
	t2 = PetscScalarNorm(1,t1);
	t1 = t1/t2 - 1;
	t2 = 1/t2  - 1;
      }
      else {
	ierr = ComputeFischerFunction(u[i]-x[i],-f[i],&ei);CHKERRQ(ierr);
	t2 = PetscScalarNorm(x[i]-l[i],ei);
	t1 = ei/t2 - 1;
	t2 = (x[i] - l[i])/t2 - 1;
      }

      da[i] = t2 + t1*ci;
      db[i] = t1*di;
    }
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = VecRestoreArray(vi->xl,&l);CHKERRQ(ierr);
  ierr = VecRestoreArray(vi->xu,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(Da,&da);CHKERRQ(ierr);
  ierr = VecRestoreArray(Db,&db);CHKERRQ(ierr);
  ierr = VecRestoreArray(vi->t,&t);CHKERRQ(ierr);

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
#undef __FUNCT__
#define __FUNCT__ "SNESVIComputeJacobian"
PetscErrorCode SNESVIComputeJacobian(Mat jac, Mat jac_pre,Vec Da, Vec Db)
{
  PetscErrorCode ierr;
  
  /* Do row scaling  and add diagonal perturbation */
  ierr = MatDiagonalScale(jac,Db,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatDiagonalSet(jac,Da,ADD_VALUES);CHKERRQ(ierr);
  if (jac != jac_pre) { /* If jac and jac_pre are different */
    ierr = MatDiagonalScale(jac_pre,Db,PETSC_NULL);
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
#undef __FUNCT__
#define __FUNCT__ "SNESVIComputeMeritFunctionGradient"
PetscErrorCode SNESVIComputeMeritFunctionGradient(Mat H, Vec phi, Vec dpsi)
{
  PetscErrorCode ierr;
    
  PetscFunctionBegin;
  ierr = MatMultTranspose(H,phi,dpsi);
  PetscFunctionReturn(0);
}
  
/*
   SNESVIAdjustInitialGuess - Readjusts the initial guess to the SNES solver supplied by the user so that the initial guess lies inside the feasible region .

   Input Parameters:
.  lb - lower bound.
.  ub - upper bound.

   Output Parameters:
.  X - the readjusted initial guess.

   Notes:
   The readjusted initial guess X[i] = max(max(min(l[i],X[i]),min(X[i],u[i])),min(l[i],u[i]))
*/
#undef __FUNCT__
#define __FUNCT__ "SNESVIAdjustInitialGuess"
PetscErrorCode SNESVIAdjustInitialGuess(Vec X, Vec lb, Vec ub)
{
  PetscErrorCode ierr;
  PetscInt       i,nlocal;
  PetscScalar    *x,*l,*u;

  PetscFunctionBegin;

  ierr = VecGetLocalSize(X,&nlocal);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(lb,&l);CHKERRQ(ierr);
  ierr = VecGetArray(ub,&u);CHKERRQ(ierr);

  for(i = 0; i < nlocal; i++) {
    x[i] = PetscMax(PetscMax(PetscMin(x[i],l[i]),PetscMin(x[i],u[i])),PetscMin(l[i],u[i]));
  }

  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(lb,&l);CHKERRQ(ierr);
  ierr = VecRestoreArray(ub,&u);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*  -------------------------------------------------------------------- 

     This file implements a semismooth truncated Newton method with a line search,
     for solving a system of nonlinear equations in complementarity form, using the KSP, Vec, 
     and Mat interfaces for linear solvers, vectors, and matrices, 
     respectively.

     The following basic routines are required for each nonlinear solver:
          SNESCreate_XXX()          - Creates a nonlinear solver context
          SNESSetFromOptions_XXX()  - Sets runtime options
          SNESSolve_XXX()           - Solves the nonlinear system
          SNESDestroy_XXX()         - Destroys the nonlinear solver context
     The suffix "_XXX" denotes a particular implementation, in this case
     we use _VI (e.g., SNESCreate_VI, SNESSolve_VI) for solving
     systems of nonlinear equations with a line search (LS) method.
     These routines are actually called via the common user interface
     routines SNESCreate(), SNESSetFromOptions(), SNESSolve(), and 
     SNESDestroy(), so the application code interface remains identical 
     for all nonlinear solvers.

     Another key routine is:
          SNESSetUp_XXX()           - Prepares for the use of a nonlinear solver
     by setting data structures and options.   The interface routine SNESSetUp()
     is not usually called directly by the user, but instead is called by
     SNESSolve() if necessary.

     Additional basic routines are:
          SNESView_XXX()            - Prints details of runtime options that
                                      have actually been used.
     These are called by application codes via the interface routines
     SNESView().

     The various types of solvers (preconditioners, Krylov subspace methods,
     nonlinear solvers, timesteppers) are all organized similarly, so the
     above description applies to these categories also.  

    -------------------------------------------------------------------- */
/*
   SNESSolveVI_SS - Solves the complementarity problem with a semismooth Newton
   method using a line search.

   Input Parameters:
.  snes - the SNES context

   Output Parameter:
.  outits - number of iterations until termination

   Application Interface Routine: SNESSolve()

   Notes:
   This implements essentially a semismooth Newton method with a
   line search.  By default a cubic backtracking line search 
   is employed, as described in the text "Numerical Methods for
   Unconstrained Optimization and Nonlinear Equations" by Dennis 
   and Schnabel.
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESSolveVI_SS"
PetscErrorCode SNESSolveVI_SS(SNES snes)
{ 
  SNES_VI            *vi = (SNES_VI*)snes->data;
  PetscErrorCode     ierr;
  PetscInt           maxits,i,lits;
  PetscBool          lssucceed;
  MatStructure       flg = DIFFERENT_NONZERO_PATTERN;
  PetscReal          gnorm,xnorm=0,ynorm;
  Vec                Y,X,F,G,W;
  KSPConvergedReason kspreason;

  PetscFunctionBegin;
  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  maxits	= snes->max_its;	/* maximum number of iterations */
  X		= snes->vec_sol;	/* solution vector */
  F		= snes->vec_func;	/* residual vector */
  Y		= snes->work[0];	/* work vectors */
  G		= snes->work[1];
  W		= snes->work[2];

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.0;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);

  ierr = SNESVIAdjustInitialGuess(X,vi->xl,vi->xu);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,vi->phi);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
   /* Compute Merit function */
  ierr = SNESVIComputeMeritFunction(vi->phi,&vi->merit,&vi->phinorm);CHKERRQ(ierr);

  ierr = VecNormBegin(X,NORM_2,&xnorm);CHKERRQ(ierr);	/* xnorm <- ||x||  */
  ierr = VecNormEnd(X,NORM_2,&xnorm);CHKERRQ(ierr);
  if PetscIsInfOrNanReal(vi->merit) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = vi->phinorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,vi->phinorm,0);
  SNESMonitor(snes,0,vi->phinorm);

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = vi->phinorm*snes->rtol;
  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,vi->phinorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  for (i=0; i<maxits; i++) {

    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
 
    /* Solve J Y = Phi, where J is the semismooth jacobian */
    /* Get the nonlinear function jacobian */
    ierr = SNESComputeJacobian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);
    /* Get the diagonal shift and row scaling vectors */
    ierr = SNESVIComputeBsubdifferentialVectors(snes,X,F,snes->jacobian,vi->Da,vi->Db);CHKERRQ(ierr);
    /* Compute the semismooth jacobian */
    ierr = SNESVIComputeJacobian(snes->jacobian,snes->jacobian_pre,vi->Da,vi->Db);CHKERRQ(ierr);
    /* Compute the merit function gradient */
    ierr = SNESVIComputeMeritFunctionGradient(snes->jacobian,vi->phi,vi->dpsi);CHKERRQ(ierr);
    ierr = KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre,flg);CHKERRQ(ierr);
    ierr = SNES_KSPSolve(snes,snes->ksp,vi->phi,Y);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(snes->ksp,&kspreason);CHKERRQ(ierr);

    if (kspreason < 0) {
      if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
        ierr = PetscInfo2(snes,"iter=%D, number linear solve failures %D greater than current SNES allowed, stopping solve\n",snes->iter,snes->numLinearSolveFailures);CHKERRQ(ierr);
        snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
        break;
      }
    }
    ierr = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);
    snes->linear_its += lits;
    ierr = PetscInfo2(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits);CHKERRQ(ierr);
    /*
    if (vi->precheckstep) {
      PetscBool changed_y = PETSC_FALSE;
      ierr = (*vi->precheckstep)(snes,X,Y,vi->precheck,&changed_y);CHKERRQ(ierr);
    }

    if (PetscLogPrintInfo){
      ierr = SNESVICheckResidual_Private(snes,snes->jacobian,F,Y,G,W);CHKERRQ(ierr);
    }
    */
    /* Compute a (scaled) negative update in the line search routine: 
         Y <- X - lambda*Y 
       and evaluate G = function(Y) (depends on the line search). 
    */
    ierr = VecCopy(Y,snes->vec_sol_update);CHKERRQ(ierr);
    ynorm = 1; gnorm = vi->phinorm;
    ierr = (*vi->LineSearch)(snes,vi->lsP,X,vi->phi,G,Y,W,vi->phinorm,xnorm,&ynorm,&gnorm,&lssucceed);CHKERRQ(ierr);
    ierr = PetscInfo4(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",vi->phinorm,gnorm,ynorm,(int)lssucceed);CHKERRQ(ierr);
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
    if (!lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
	PetscBool ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        ierr = SNESVICheckLocalMin_Private(snes,snes->jacobian,G,W,gnorm,&ismin);CHKERRQ(ierr);
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
    }
    /* Update function and solution vectors */
    vi->phinorm = gnorm;
    vi->merit = 0.5*vi->phinorm*vi->phinorm;
    ierr = VecCopy(G,vi->phi);CHKERRQ(ierr);
    ierr = VecCopy(W,X);CHKERRQ(ierr);
    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = vi->phinorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,lits);
    SNESMonitor(snes,snes->iter,snes->norm);
    /* Test for convergence, xnorm = || X || */
    if (snes->ops->converged != SNESSkipConverged) { ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr); }
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,vi->phinorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  if (i == maxits) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",maxits);CHKERRQ(ierr);
    if(!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESVICreateIndexSets_AS"
PetscErrorCode SNESVICreateIndexSets_AS(SNES snes,Vec Db,PetscReal thresh,IS* ISact,IS* ISinact)
{
  PetscErrorCode ierr;
  PetscInt       i,nlocal,ilow,ihigh,nloc_isact=0,nloc_isinact=0;
  PetscInt       *idx_act,*idx_inact,i1=0,i2=0;
  PetscScalar    *db;

  PetscFunctionBegin;

  ierr = VecGetLocalSize(Db,&nlocal);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(Db,&ilow,&ihigh);CHKERRQ(ierr);
  ierr = VecGetArray(Db,&db);CHKERRQ(ierr);
  /* Compute the sizes of the active and inactive sets */
  for (i=0; i < nlocal;i++) {
    if (PetscAbsScalar(db[i]) <= thresh) nloc_isact++;
    else nloc_isinact++;
  }
  ierr = PetscMalloc(nloc_isact*sizeof(PetscInt),&idx_act);CHKERRQ(ierr);
  ierr = PetscMalloc(nloc_isinact*sizeof(PetscInt),&idx_inact);CHKERRQ(ierr);

  /* Creating the indexing arrays */
  for(i=0; i < nlocal; i++) {
    if (PetscAbsScalar(db[i]) <= thresh) idx_act[i1++] = ilow+i;
    else idx_inact[i2++] = ilow+i;
  }

  /* Create the index sets */
  ierr = ISCreateGeneral(((PetscObject)snes)->comm,nloc_isact,idx_act,PETSC_COPY_VALUES,ISact);CHKERRQ(ierr);
  ierr = ISCreateGeneral(((PetscObject)snes)->comm,nloc_isinact,idx_inact,PETSC_COPY_VALUES,ISinact);CHKERRQ(ierr);

  ierr = VecRestoreArray(Db,&db);CHKERRQ(ierr);
  ierr = PetscFree(idx_act);CHKERRQ(ierr);
  ierr = PetscFree(idx_inact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESVICreateIndexSets_RS"
PetscErrorCode SNESVICreateIndexSets_RS(SNES snes,Vec X,Vec Xl, Vec Xu,IS* ISact,IS* ISinact)
{
  PetscErrorCode ierr;
  SNES_VI        *vi = (SNES_VI*)snes->data;
  PetscInt       i,nlocal,ilow,ihigh,nloc_isact=0,nloc_isinact=0;
  PetscInt       *idx_act,*idx_inact,i1=0,i2=0;
  PetscScalar    *x,*l,*u,*f;
  Vec            F = snes->vec_func;

  PetscFunctionBegin;

  ierr = VecGetLocalSize(X,&nlocal);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(X,&ilow,&ihigh);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Xl,&l);CHKERRQ(ierr);
  ierr = VecGetArray(Xu,&u);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  /* Compute the sizes of the active and inactive sets */
  for (i=0; i < nlocal;i++) {
    if (PetscAbsScalar(x[i] - l[i]) <= vi->const_tol && f[i] > vi->const_tol) nloc_isact++;
    else if (PetscAbsScalar(u[i] - x[i]) <= vi->const_tol && f[i] < vi->const_tol) nloc_isact++;
    else nloc_isinact++;
  }
  ierr = PetscMalloc(nloc_isact*sizeof(PetscInt),&idx_act);CHKERRQ(ierr);
  ierr = PetscMalloc(nloc_isinact*sizeof(PetscInt),&idx_inact);CHKERRQ(ierr);

  /* Creating the indexing arrays */
  for(i=0; i < nlocal; i++) {
    if (PetscAbsScalar(x[i] - l[i]) <= vi->const_tol && f[i] > vi->const_tol) idx_act[i1++] = ilow+i;
    else if (PetscAbsScalar(u[i] - x[i]) <= vi->const_tol && f[i] < vi->const_tol) idx_act[i1++] = ilow+i;
    else idx_inact[i2++] = ilow+i;
  }

  /* Create the index sets */
  ierr = ISCreateGeneral(((PetscObject)snes)->comm,nloc_isact,idx_act,PETSC_COPY_VALUES,ISact);CHKERRQ(ierr);
  ierr = ISCreateGeneral(((PetscObject)snes)->comm,nloc_isinact,idx_inact,PETSC_COPY_VALUES,ISinact);CHKERRQ(ierr);

  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xl,&l);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xu,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = PetscFree(idx_act);CHKERRQ(ierr);
  ierr = PetscFree(idx_inact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create active and inactive set vectors. The local size of this vector is set and petsc computes the global size */
#undef __FUNCT__
#define __FUNCT__ "SNESVICreateVectors_AS"
PetscErrorCode SNESVICreateVectors_AS(SNES snes,PetscInt n,Vec* newv)
{
  PetscErrorCode ierr;
  Vec            v;

  PetscFunctionBegin;
  ierr = VecCreate(((PetscObject)snes)->comm,&v);CHKERRQ(ierr);
  ierr = VecSetSizes(v,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(v);CHKERRQ(ierr);
  *newv = v;

  PetscFunctionReturn(0);
}

/* Resets the snes PC and KSP when the active set sizes change */
#undef __FUNCT__
#define __FUNCT__ "SNESVIResetPCandKSP"
PetscErrorCode SNESVIResetPCandKSP(SNES snes,Mat Amat,Mat Pmat)
{
  PetscErrorCode ierr;
  KSP kspnew,snesksp;
  PC  pcnew;
  const MatSolverPackage stype;
  
  PetscFunctionBegin;
  /* The active and inactive set sizes have changed so need to create a new snes->ksp object */
  ierr = SNESGetKSP(snes,&snesksp);CHKERRQ(ierr);
  ierr = KSPCreate(((PetscObject)snes)->comm,&kspnew);CHKERRQ(ierr);
  /* Copy over snes->ksp info */
  kspnew->pc_side = snesksp->pc_side;
  kspnew->rtol    = snesksp->rtol;
  kspnew->abstol    = snesksp->abstol;
  kspnew->max_it  = snesksp->max_it;
  ierr = KSPSetType(kspnew,((PetscObject)snesksp)->type_name);CHKERRQ(ierr);
  ierr = KSPGetPC(kspnew,&pcnew);CHKERRQ(ierr);
  ierr = PCSetType(kspnew->pc,((PetscObject)snesksp->pc)->type_name);CHKERRQ(ierr);
  ierr = PCSetOperators(kspnew->pc,Amat,Pmat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PCFactorGetMatSolverPackage(snesksp->pc,&stype);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(kspnew->pc,stype);CHKERRQ(ierr);
  ierr = KSPDestroy(snesksp);CHKERRQ(ierr);
  snes->ksp = kspnew;
  ierr = PetscLogObjectParent(snes,kspnew);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(kspnew);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* Variational Inequality solver using active set method */
#undef __FUNCT__  
#define __FUNCT__ "SNESSolveVI_AS"
PetscErrorCode SNESSolveVI_AS(SNES snes)
{ 
  SNES_VI          *vi = (SNES_VI*)snes->data;
  PetscErrorCode     ierr;
  PetscInt           maxits,i,lits,Nis_act=0;
  PetscBool         lssucceed;
  MatStructure       flg = DIFFERENT_NONZERO_PATTERN;
  PetscReal          gnorm,xnorm=0,ynorm;
  Vec                Y,X,F,G,W;
  KSPConvergedReason kspreason;

  PetscFunctionBegin;
  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  maxits	= snes->max_its;	/* maximum number of iterations */
  X		= snes->vec_sol;	/* solution vector */
  F		= snes->vec_func;	/* residual vector */
  Y		= snes->work[0];	/* work vectors */
  G		= snes->work[1];
  W		= snes->work[2];

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.0;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);

  ierr = SNESVIAdjustInitialGuess(X,vi->xl,vi->xu);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,vi->phi);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
   /* Compute Merit function */
  ierr = SNESVIComputeMeritFunction(vi->phi,&vi->merit,&vi->phinorm);CHKERRQ(ierr);

  ierr = VecNormBegin(X,NORM_2,&xnorm);CHKERRQ(ierr);	/* xnorm <- ||x||  */
  ierr = VecNormEnd(X,NORM_2,&xnorm);CHKERRQ(ierr);
  if PetscIsInfOrNanReal(vi->merit) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = vi->phinorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,vi->phinorm,0);
  SNESMonitor(snes,0,vi->phinorm);

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = vi->phinorm*snes->rtol;
  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,vi->phinorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  for (i=0; i<maxits; i++) {

    IS                 IS_act,IS_inact; /* _act -> active set _inact -> inactive set */
    PetscReal          thresh,J_norm1;
    VecScatter         scat_act,scat_inact;
    PetscInt           nis_act,nis_inact,Nis_act_prev;
    Vec                Da_act,Da_inact,Db_inact;
    Vec                Y_act,Y_inact,phi_act,phi_inact;
    Mat                jac_inact_inact,jac_inact_act,prejac_inact_inact;

    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
    ierr = SNESComputeJacobian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);
    /* Compute the threshold value for creating active and inactive sets */
    ierr = MatNorm(snes->jacobian,NORM_1,&J_norm1);CHKERRQ(ierr);
    thresh = PetscMin(vi->merit,1e-2)/(1+J_norm1);

    /* Compute B-subdifferential vectors Da and Db */
    ierr = SNESVIComputeBsubdifferentialVectors(snes,X,F,snes->jacobian,vi->Da,vi->Db);CHKERRQ(ierr);

    /* Create active and inactive index sets */
    ierr = SNESVICreateIndexSets_AS(snes,vi->Db,thresh,&IS_act,&IS_inact);CHKERRQ(ierr);

    Nis_act_prev = Nis_act;
    /* Get sizes of active and inactive sets */
    ierr = ISGetLocalSize(IS_act,&nis_act);CHKERRQ(ierr);
    ierr = ISGetLocalSize(IS_inact,&nis_inact);CHKERRQ(ierr);
    ierr = ISGetSize(IS_act,&Nis_act);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"Size of active set = %d, size of inactive set = %d\n",nis_act,nis_inact);CHKERRQ(ierr);

    /* Create active and inactive set vectors */
    ierr = SNESVICreateVectors_AS(snes,nis_act,&Da_act);CHKERRQ(ierr);
    ierr = SNESVICreateVectors_AS(snes,nis_inact,&Da_inact);CHKERRQ(ierr);
    ierr = SNESVICreateVectors_AS(snes,nis_inact,&Db_inact);CHKERRQ(ierr);
    ierr = SNESVICreateVectors_AS(snes,nis_act,&phi_act);CHKERRQ(ierr);
    ierr = SNESVICreateVectors_AS(snes,nis_inact,&phi_inact);CHKERRQ(ierr);
    ierr = SNESVICreateVectors_AS(snes,nis_act,&Y_act);CHKERRQ(ierr);
    ierr = SNESVICreateVectors_AS(snes,nis_inact,&Y_inact);CHKERRQ(ierr);

    /* Create inactive set submatrices */
    ierr = MatGetSubMatrix(snes->jacobian,IS_inact,IS_act,MAT_INITIAL_MATRIX,&jac_inact_act);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(snes->jacobian,IS_inact,IS_inact,MAT_INITIAL_MATRIX,&jac_inact_inact);CHKERRQ(ierr);

    /* Create scatter contexts */
    ierr = VecScatterCreate(vi->Da,IS_act,Da_act,PETSC_NULL,&scat_act);CHKERRQ(ierr);
    ierr = VecScatterCreate(vi->Da,IS_inact,Da_inact,PETSC_NULL,&scat_inact);CHKERRQ(ierr);

    /* Do a vec scatter to active and inactive set vectors */
    ierr = VecScatterBegin(scat_act,vi->Da,Da_act,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_act,vi->Da,Da_act,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  
    ierr = VecScatterBegin(scat_inact,vi->Da,Da_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_inact,vi->Da,Da_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  
    ierr = VecScatterBegin(scat_inact,vi->Db,Db_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_inact,vi->Db,Db_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    ierr = VecScatterBegin(scat_act,vi->phi,phi_act,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_act,vi->phi,phi_act,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    ierr = VecScatterBegin(scat_inact,vi->phi,phi_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_inact,vi->phi,phi_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    ierr = VecScatterBegin(scat_act,Y,Y_act,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_act,Y,Y_act,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    ierr = VecScatterBegin(scat_inact,Y,Y_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_inact,Y,Y_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    
    /* Active set direction */
    ierr = VecPointwiseDivide(Y_act,phi_act,Da_act);CHKERRQ(ierr);
    /* inactive set jacobian and preconditioner */
    ierr = VecPointwiseDivide(Da_inact,Da_inact,Db_inact);CHKERRQ(ierr);
    ierr = MatDiagonalSet(jac_inact_inact,Da_inact,ADD_VALUES);CHKERRQ(ierr);
    if (snes->jacobian != snes->jacobian_pre) {
      ierr = MatGetSubMatrix(snes->jacobian_pre,IS_inact,IS_inact,MAT_INITIAL_MATRIX,&prejac_inact_inact);CHKERRQ(ierr);
      ierr = MatDiagonalSet(prejac_inact_inact,Da_inact,ADD_VALUES);CHKERRQ(ierr);
    } else prejac_inact_inact = jac_inact_inact;

    /* right hand side */
    ierr = VecPointwiseDivide(phi_inact,phi_inact,Db_inact);CHKERRQ(ierr);
    ierr = MatMult(jac_inact_act,Y_act,Db_inact);CHKERRQ(ierr);
    ierr = VecAXPY(phi_inact,-1.0,Db_inact);CHKERRQ(ierr);

    if ((i != 0) && (Nis_act != Nis_act_prev)) {
      ierr = SNESVIResetPCandKSP(snes,jac_inact_inact,prejac_inact_inact);CHKERRQ(ierr);
    }
    
    ierr = KSPSetOperators(snes->ksp,jac_inact_inact,prejac_inact_inact,flg);CHKERRQ(ierr);
    ierr = SNES_KSPSolve(snes,snes->ksp,phi_inact,Y_inact);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(snes->ksp,&kspreason);CHKERRQ(ierr);
    if (kspreason < 0) {
      if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
        ierr = PetscInfo2(snes,"iter=%D, number linear solve failures %D greater than current SNES allowed, stopping solve\n",snes->iter,snes->numLinearSolveFailures);CHKERRQ(ierr);
        snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
        break;
      }
     }

    ierr = VecScatterBegin(scat_act,Y_act,Y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_act,Y_act,Y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(scat_inact,Y_inact,Y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_inact,Y_inact,Y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

    ierr = VecDestroy(Da_act);CHKERRQ(ierr);
    ierr = VecDestroy(Da_inact);CHKERRQ(ierr);
    ierr = VecDestroy(Db_inact);CHKERRQ(ierr);
    ierr = VecDestroy(phi_act);CHKERRQ(ierr);
    ierr = VecDestroy(phi_inact);CHKERRQ(ierr);
    ierr = VecDestroy(Y_act);CHKERRQ(ierr);
    ierr = VecDestroy(Y_inact);CHKERRQ(ierr);
    ierr = VecScatterDestroy(scat_act);CHKERRQ(ierr);
    ierr = VecScatterDestroy(scat_inact);CHKERRQ(ierr);
    ierr = ISDestroy(IS_act);CHKERRQ(ierr);
    ierr = ISDestroy(IS_inact);CHKERRQ(ierr);
    ierr = MatDestroy(jac_inact_act);CHKERRQ(ierr);
    ierr = MatDestroy(jac_inact_inact);CHKERRQ(ierr);
    if (snes->jacobian != snes->jacobian_pre) {
      ierr = MatDestroy(prejac_inact_inact);CHKERRQ(ierr);
    }

    ierr = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);
    snes->linear_its += lits;
    ierr = PetscInfo2(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits);CHKERRQ(ierr);
    /*
    if (vi->precheckstep) {
      PetscBool changed_y = PETSC_FALSE;
      ierr = (*vi->precheckstep)(snes,X,Y,vi->precheck,&changed_y);CHKERRQ(ierr);
    }

    if (PetscLogPrintInfo){
      ierr = SNESVICheckResidual_Private(snes,snes->jacobian,F,Y,G,W);CHKERRQ(ierr);
    }
    */
    /* Compute a (scaled) negative update in the line search routine: 
         Y <- X - lambda*Y 
       and evaluate G = function(Y) (depends on the line search). 
    */
    ierr = VecCopy(Y,snes->vec_sol_update);CHKERRQ(ierr);
    ynorm = 1; gnorm = vi->phinorm;
    ierr = SNESVIComputeJacobian(snes->jacobian,snes->jacobian_pre,vi->Da,vi->Db);CHKERRQ(ierr);
    ierr = SNESVIComputeMeritFunctionGradient(snes->jacobian,vi->phi,vi->dpsi);CHKERRQ(ierr);
    ierr = (*vi->LineSearch)(snes,vi->lsP,X,vi->phi,G,Y,W,vi->phinorm,xnorm,&ynorm,&gnorm,&lssucceed);CHKERRQ(ierr);
    ierr = PetscInfo4(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",vi->phinorm,gnorm,ynorm,(int)lssucceed);CHKERRQ(ierr);
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
    if (!lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
	PetscBool ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        ierr = SNESVICheckLocalMin_Private(snes,snes->jacobian,G,W,gnorm,&ismin);CHKERRQ(ierr);
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
    }
    /* Update function and solution vectors */
    vi->phinorm = gnorm;
    vi->merit = 0.5*vi->phinorm*vi->phinorm;
    ierr = VecCopy(G,vi->phi);CHKERRQ(ierr);
    ierr = VecCopy(W,X);CHKERRQ(ierr);
    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = vi->phinorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,lits);
    SNESMonitor(snes,snes->iter,snes->norm);
    /* Test for convergence, xnorm = || X || */
    if (snes->ops->converged != SNESSkipConverged) { ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr); }
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,vi->phinorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  if (i == maxits) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",maxits);CHKERRQ(ierr);
    if(!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

/* Variational Inequality solver using active set method */
#undef __FUNCT__  
#define __FUNCT__ "SNESSolveVI_RS"
PetscErrorCode SNESSolveVI_RS(SNES snes)
{ 
  SNES_VI          *vi = (SNES_VI*)snes->data;
  PetscErrorCode     ierr;
  PetscInt           maxits,i,lits,Nis_act=0;
  PetscBool         lssucceed;
  MatStructure       flg = DIFFERENT_NONZERO_PATTERN;
  PetscReal          gnorm,xnorm=0,ynorm;
  Vec                Y,X,F,G,W;
  KSPConvergedReason kspreason;

  PetscFunctionBegin;
  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  maxits	= snes->max_its;	/* maximum number of iterations */
  X		= snes->vec_sol;	/* solution vector */
  F		= snes->vec_func;	/* residual vector */
  Y		= snes->work[0];	/* work vectors */
  G		= snes->work[1];
  W		= snes->work[2];

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.0;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);

  ierr = SNESVIAdjustInitialGuess(X,vi->xl,vi->xu);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,vi->phi);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
   /* Compute Merit function */
  ierr = SNESVIComputeMeritFunction(vi->phi,&vi->merit,&vi->phinorm);CHKERRQ(ierr);

  ierr = VecNormBegin(X,NORM_2,&xnorm);CHKERRQ(ierr);	/* xnorm <- ||x||  */
  ierr = VecNormEnd(X,NORM_2,&xnorm);CHKERRQ(ierr);
  if PetscIsInfOrNanReal(vi->merit) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = vi->phinorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,vi->phinorm,0);
  SNESMonitor(snes,0,vi->phinorm);

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = vi->phinorm*snes->rtol;
  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,vi->phinorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  for (i=0; i<maxits; i++) {

    IS                 IS_act,IS_inact; /* _act -> active set _inact -> inactive set */
    VecScatter         scat_act,scat_inact;
    PetscInt           nis_act,nis_inact,Nis_act_prev;
    Vec                Y_act,Y_inact,phi_inact;
    Mat                jac_inact_inact,prejac_inact_inact;

    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
    ierr = SNESComputeJacobian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);
    ierr = SNESVIComputeBsubdifferentialVectors(snes,X,F,snes->jacobian,vi->Da,vi->Db);CHKERRQ(ierr);
    ierr = SNESVIComputeJacobian(snes->jacobian,snes->jacobian_pre,vi->Da,vi->Db);CHKERRQ(ierr);
    ierr = SNESVIComputeMeritFunctionGradient(snes->jacobian,vi->phi,vi->dpsi);CHKERRQ(ierr);
    /* Create active and inactive index sets */
    ierr = SNESVICreateIndexSets_RS(snes,X,vi->xl,vi->xu,&IS_act,&IS_inact);CHKERRQ(ierr);

    Nis_act_prev = Nis_act;
    /* Get sizes of active and inactive sets */
    ierr = ISGetLocalSize(IS_act,&nis_act);CHKERRQ(ierr);
    ierr = ISGetLocalSize(IS_inact,&nis_inact);CHKERRQ(ierr);
    ierr = ISGetSize(IS_act,&Nis_act);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"Size of active set = %d, size of inactive set = %d\n",nis_act,nis_inact);CHKERRQ(ierr);

    /* Create active and inactive set vectors */
    ierr = SNESVICreateVectors_AS(snes,nis_inact,&phi_inact);CHKERRQ(ierr);
    ierr = SNESVICreateVectors_AS(snes,nis_act,&Y_act);CHKERRQ(ierr);
    ierr = SNESVICreateVectors_AS(snes,nis_inact,&Y_inact);CHKERRQ(ierr);

    /* Create inactive set submatrices */
    ierr = MatGetSubMatrix(snes->jacobian,IS_inact,IS_inact,MAT_INITIAL_MATRIX,&jac_inact_inact);CHKERRQ(ierr);

    /* Create scatter contexts */
    ierr = VecScatterCreate(Y,IS_act,Y_act,PETSC_NULL,&scat_act);CHKERRQ(ierr);
    ierr = VecScatterCreate(Y,IS_inact,Y_inact,PETSC_NULL,&scat_inact);CHKERRQ(ierr);

    /* Do a vec scatter to active and inactive set vectors */
    ierr = VecScatterBegin(scat_inact,vi->phi,phi_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_inact,vi->phi,phi_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    ierr = VecScatterBegin(scat_act,Y,Y_act,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_act,Y,Y_act,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    ierr = VecScatterBegin(scat_inact,Y,Y_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_inact,Y,Y_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    
    /* Active set direction = 0*/
    ierr = VecSet(Y_act,0);CHKERRQ(ierr);
    if (snes->jacobian != snes->jacobian_pre) {
      ierr = MatGetSubMatrix(snes->jacobian_pre,IS_inact,IS_inact,MAT_INITIAL_MATRIX,&prejac_inact_inact);CHKERRQ(ierr);
    } else prejac_inact_inact = jac_inact_inact;

    if ((i != 0) && (Nis_act != Nis_act_prev)) {
      ierr = SNESVIResetPCandKSP(snes,jac_inact_inact,prejac_inact_inact);CHKERRQ(ierr);
    }
    
    ierr = KSPSetOperators(snes->ksp,jac_inact_inact,prejac_inact_inact,flg);CHKERRQ(ierr);
    ierr = SNES_KSPSolve(snes,snes->ksp,phi_inact,Y_inact);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(snes->ksp,&kspreason);CHKERRQ(ierr);
    if (kspreason < 0) {
      if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
        ierr = PetscInfo2(snes,"iter=%D, number linear solve failures %D greater than current SNES allowed, stopping solve\n",snes->iter,snes->numLinearSolveFailures);CHKERRQ(ierr);
        snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
        break;
      }
     }

    ierr = VecScatterBegin(scat_act,Y_act,Y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_act,Y_act,Y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(scat_inact,Y_inact,Y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_inact,Y_inact,Y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

    ierr = VecDestroy(phi_inact);CHKERRQ(ierr);
    ierr = VecDestroy(Y_act);CHKERRQ(ierr);
    ierr = VecDestroy(Y_inact);CHKERRQ(ierr);
    ierr = VecScatterDestroy(scat_act);CHKERRQ(ierr);
    ierr = VecScatterDestroy(scat_inact);CHKERRQ(ierr);
    ierr = ISDestroy(IS_act);CHKERRQ(ierr);
    ierr = ISDestroy(IS_inact);CHKERRQ(ierr);
    ierr = MatDestroy(jac_inact_inact);CHKERRQ(ierr);
    if (snes->jacobian != snes->jacobian_pre) {
      ierr = MatDestroy(prejac_inact_inact);CHKERRQ(ierr);
    }

    ierr = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);
    snes->linear_its += lits;
    ierr = PetscInfo2(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits);CHKERRQ(ierr);
    /*
    if (vi->precheckstep) {
      PetscBool changed_y = PETSC_FALSE;
      ierr = (*vi->precheckstep)(snes,X,Y,vi->precheck,&changed_y);CHKERRQ(ierr);
    }

    if (PetscLogPrintInfo){
      ierr = SNESVICheckResidual_Private(snes,snes->jacobian,F,Y,G,W);CHKERRQ(ierr);
    }
    */
    /* Compute a (scaled) negative update in the line search routine: 
         Y <- X - lambda*Y 
       and evaluate G = function(Y) (depends on the line search). 
    */
    ierr = VecCopy(Y,snes->vec_sol_update);CHKERRQ(ierr);
    ynorm = 1; gnorm = vi->phinorm;
    ierr = (*vi->LineSearch)(snes,vi->lsP,X,vi->phi,G,Y,W,vi->phinorm,xnorm,&ynorm,&gnorm,&lssucceed);CHKERRQ(ierr);
    ierr = PetscInfo4(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",vi->phinorm,gnorm,ynorm,(int)lssucceed);CHKERRQ(ierr);
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
    if (!lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
	PetscBool ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        ierr = SNESVICheckLocalMin_Private(snes,snes->jacobian,G,W,gnorm,&ismin);CHKERRQ(ierr);
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
    }
    /* Update function and solution vectors */
    vi->phinorm = gnorm;
    vi->merit = 0.5*vi->phinorm*vi->phinorm;
    ierr = VecCopy(G,vi->phi);CHKERRQ(ierr);
    ierr = VecCopy(W,X);CHKERRQ(ierr);
    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = vi->phinorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,lits);
    SNESMonitor(snes,snes->iter,snes->norm);
    /* Test for convergence, xnorm = || X || */
    if (snes->ops->converged != SNESSkipConverged) { ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr); }
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,vi->phinorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  if (i == maxits) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",maxits);CHKERRQ(ierr);
    if(!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   SNESSetUp_VI - Sets up the internal data structures for the later use
   of the SNESVI nonlinear solver.

   Input Parameter:
.  snes - the SNES context
.  x - the solution vector

   Application Interface Routine: SNESSetUp()

   Notes:
   For basic use of the SNES solvers, the user need not explicitly call
   SNESSetUp(), since these actions will automatically occur during
   the call to SNESSolve().
 */
#undef __FUNCT__  
#define __FUNCT__ "SNESSetUp_VI"
PetscErrorCode SNESSetUp_VI(SNES snes)
{
  PetscErrorCode ierr;
  SNES_VI      *vi = (SNES_VI*) snes->data;
  PetscInt       i_start[3],i_end[3];

  PetscFunctionBegin;
  if (!snes->vec_sol_update) {
    ierr = VecDuplicate(snes->vec_sol,&snes->vec_sol_update);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(snes,snes->vec_sol_update);CHKERRQ(ierr);
  }
  if (!snes->work) {
    snes->nwork = 3;
    ierr = VecDuplicateVecs(snes->vec_sol,snes->nwork,&snes->work);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(snes,snes->nwork,snes->work);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(snes->vec_sol, &vi->dpsi);CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_sol, &vi->phi); CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_sol, &vi->Da); CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_sol, &vi->Db); CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_sol, &vi->z);CHKERRQ(ierr);
  ierr = VecDuplicate(snes->vec_sol, &vi->t); CHKERRQ(ierr);

  /* If the lower and upper bound on variables are not set, set it to
     -Inf and Inf */
  if (!vi->xl && !vi->xu) {
    vi->usersetxbounds = PETSC_FALSE;
    ierr = VecDuplicate(snes->vec_sol, &vi->xl); CHKERRQ(ierr);
    ierr = VecSet(vi->xl,PETSC_VI_NINF);CHKERRQ(ierr);
    ierr = VecDuplicate(snes->vec_sol, &vi->xu); CHKERRQ(ierr);
    ierr = VecSet(vi->xu,PETSC_VI_INF);CHKERRQ(ierr);
  } else {
    /* Check if lower bound, upper bound and solution vector distribution across the processors is identical */
    ierr = VecGetOwnershipRange(snes->vec_sol,i_start,i_end);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(vi->xl,i_start+1,i_end+1);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(vi->xu,i_start+2,i_end+2);CHKERRQ(ierr);
    if ((i_start[0] != i_start[1]) || (i_start[0] != i_start[2]) || (i_end[0] != i_end[1]) || (i_end[0] != i_end[2]))
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Distribution of lower bound, upper bound and the solution vector should be identical across all the processors.");
  }

  vi->computeuserfunction = snes->ops->computefunction;
  snes->ops->computefunction = SNESVIComputeFunction;

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   SNESDestroy_VI - Destroys the private SNES_VI context that was created
   with SNESCreate_VI().

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESDestroy()
 */
#undef __FUNCT__  
#define __FUNCT__ "SNESDestroy_VI"
PetscErrorCode SNESDestroy_VI(SNES snes)
{
  SNES_VI        *vi = (SNES_VI*) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (snes->vec_sol_update) {
    ierr = VecDestroy(snes->vec_sol_update);CHKERRQ(ierr);
    snes->vec_sol_update = PETSC_NULL;
  }
  if (snes->nwork) {
    ierr = VecDestroyVecs(snes->work,snes->nwork);CHKERRQ(ierr);
    snes->nwork = 0;
    snes->work  = PETSC_NULL;
  }

  /* clear vectors */
  ierr = VecDestroy(vi->dpsi);CHKERRQ(ierr);
  ierr = VecDestroy(vi->phi); CHKERRQ(ierr);
  ierr = VecDestroy(vi->Da); CHKERRQ(ierr);
  ierr = VecDestroy(vi->Db); CHKERRQ(ierr);
  ierr = VecDestroy(vi->z); CHKERRQ(ierr);
  ierr = VecDestroy(vi->t); CHKERRQ(ierr);
  if (!vi->usersetxbounds) {
    ierr = VecDestroy(vi->xl); CHKERRQ(ierr);
    ierr = VecDestroy(vi->xu); CHKERRQ(ierr);
  }
  if (vi->lsmonitor) {
    ierr = PetscViewerASCIIMonitorDestroy(vi->lsmonitor);CHKERRQ(ierr);
  } 
  ierr = PetscFree(snes->data);CHKERRQ(ierr);

  /* clear composed functions */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSet_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetMonitor_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchNo_VI"

/*
  This routine is a copy of SNESLineSearchNo routine in snes/impls/ls/ls.c

*/
PetscErrorCode SNESLineSearchNo_VI(SNES snes,void *lsctx,Vec x,Vec f,Vec g,Vec y,Vec w,PetscReal fnorm,PetscReal xnorm,PetscReal *ynorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscErrorCode ierr;
  SNES_VI        *vi = (SNES_VI*)snes->data;
  PetscBool     changed_w = PETSC_FALSE,changed_y = PETSC_FALSE;

  PetscFunctionBegin;
  *flag = PETSC_TRUE; 
  ierr = PetscLogEventBegin(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,ynorm);CHKERRQ(ierr);         /* ynorm = || y || */
  ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);            /* w <- x - y   */
  if (vi->postcheckstep) {
   ierr = (*vi->postcheckstep)(snes,x,y,w,vi->postcheck,&changed_y,&changed_w);CHKERRQ(ierr);
  }
  if (changed_y) {
    ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);            /* w <- x - y   */
  }
  ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  if (!snes->domainerror) {
    ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);  /* gnorm = || g || */
    if PetscIsInfOrNanReal(*gnorm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  }
  if (vi->lsmonitor) {
    ierr = PetscViewerASCIIMonitorPrintf(vi->lsmonitor,"    Line search: Using full step: fnorm %G gnorm %G\n",fnorm,*gnorm);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchNoNorms_VI"

/*
  This routine is a copy of SNESLineSearchNoNorms in snes/impls/ls/ls.c
*/
PetscErrorCode SNESLineSearchNoNorms_VI(SNES snes,void *lsctx,Vec x,Vec f,Vec g,Vec y,Vec w,PetscReal fnorm,PetscReal xnorm,PetscReal *ynorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscErrorCode ierr;
  SNES_VI        *vi = (SNES_VI*)snes->data;
  PetscBool     changed_w = PETSC_FALSE,changed_y = PETSC_FALSE;

  PetscFunctionBegin;
  *flag = PETSC_TRUE; 
  ierr = PetscLogEventBegin(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);            /* w <- x - y      */
  if (vi->postcheckstep) {
   ierr = (*vi->postcheckstep)(snes,x,y,w,vi->postcheck,&changed_y,&changed_w);CHKERRQ(ierr);
  }
  if (changed_y) {
    ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);            /* w <- x - y   */
  }
  
  /* don't evaluate function the last time through */
  if (snes->iter < snes->max_its-1) {
    ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchCubic_VI"
/*
  This routine is a copy of SNESLineSearchCubic in snes/impls/ls/ls.c
*/
PetscErrorCode SNESLineSearchCubic_VI(SNES snes,void *lsctx,Vec x,Vec f,Vec g,Vec y,Vec w,PetscReal fnorm,PetscReal xnorm,PetscReal *ynorm,PetscReal *gnorm,PetscBool *flag)
{
  /* 
     Note that for line search purposes we work with with the related
     minimization problem:
        min  z(x):  R^n -> R,
     where z(x) = .5 * fnorm*fnorm, and fnorm = || f ||_2.
     This function z(x) is same as the merit function for the complementarity problem.
   */
        
  PetscReal      initslope,lambdaprev,gnormprev,a,b,d,t1,t2,rellength;
  PetscReal      minlambda,lambda,lambdatemp;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    cinitslope;
#endif
  PetscErrorCode ierr;
  PetscInt       count;
  SNES_VI      *vi = (SNES_VI*)snes->data;
  PetscBool     changed_w = PETSC_FALSE,changed_y = PETSC_FALSE;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  *flag   = PETSC_TRUE;

  ierr = VecNorm(y,NORM_2,ynorm);CHKERRQ(ierr);
  if (*ynorm == 0.0) {
    if (vi->lsmonitor) {
      ierr = PetscViewerASCIIMonitorPrintf(vi->lsmonitor,"    Line search: Initial direction and size is 0\n");CHKERRQ(ierr);
    }
    *gnorm = fnorm;
    ierr   = VecCopy(x,w);CHKERRQ(ierr);
    ierr   = VecCopy(f,g);CHKERRQ(ierr);
    *flag  = PETSC_FALSE;
    goto theend1;
  }
  if (*ynorm > vi->maxstep) {	/* Step too big, so scale back */
    if (vi->lsmonitor) {
      ierr = PetscViewerASCIIMonitorPrintf(vi->lsmonitor,"    Line search: Scaling step by %G old ynorm %G\n",vi->maxstep/(*ynorm),*ynorm);CHKERRQ(ierr);
    }
    ierr = VecScale(y,vi->maxstep/(*ynorm));CHKERRQ(ierr);
    *ynorm = vi->maxstep;
  }
  ierr      = VecMaxPointwiseDivide(y,x,&rellength);CHKERRQ(ierr);
  minlambda = vi->minlambda/rellength;
#if defined(PETSC_USE_COMPLEX)
  ierr = VecDot(vi->dpsi,y,&cinitslope);CHKERRQ(ierr);
  initslope = PetscRealPart(cinitslope);
#else
  ierr = VecDot(vi->dpsi,y,&initslope);CHKERRQ(ierr);
#endif
  if (initslope > 0.0)  initslope = -initslope;
  if (initslope == 0.0) initslope = -1.0;

  ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);
  if (snes->nfuncs >= snes->max_funcs) {
    ierr  = PetscInfo(snes,"Exceeded maximum function evaluations, while checking full step length!\n");CHKERRQ(ierr);
    *flag = PETSC_FALSE;
    snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
    goto theend1;
  }
  ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  if (snes->domainerror) {
    ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
  if PetscIsInfOrNanReal(*gnorm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  ierr = PetscInfo2(snes,"Initial fnorm %G gnorm %G\n",fnorm,*gnorm);CHKERRQ(ierr);
  if (.5*(*gnorm)*(*gnorm) <= .5*fnorm*fnorm + vi->alpha*initslope) { /* Sufficient reduction */
    if (vi->lsmonitor) {
      ierr = PetscViewerASCIIMonitorPrintf(vi->lsmonitor,"    Line search: Using full step: fnorm %G gnorm %G\n",fnorm,*gnorm);CHKERRQ(ierr);
    }
    goto theend1;
  }

  /* Fit points with quadratic */
  lambda     = 1.0;
  lambdatemp = -initslope/((*gnorm)*(*gnorm) - fnorm*fnorm - 2.0*initslope);
  lambdaprev = lambda;
  gnormprev  = *gnorm;
  if (lambdatemp > .5*lambda)  lambdatemp = .5*lambda;
  if (lambdatemp <= .1*lambda) lambda = .1*lambda; 
  else                         lambda = lambdatemp;

  ierr  = VecWAXPY(w,-lambda,y,x);CHKERRQ(ierr);
  if (snes->nfuncs >= snes->max_funcs) {
    ierr  = PetscInfo1(snes,"Exceeded maximum function evaluations, while attempting quadratic backtracking! %D \n",snes->nfuncs);CHKERRQ(ierr);
    *flag = PETSC_FALSE;
    snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
    goto theend1;
  }
  ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  if (snes->domainerror) {
    ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
  if PetscIsInfOrNanReal(*gnorm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  if (vi->lsmonitor) {
    ierr = PetscViewerASCIIMonitorPrintf(vi->lsmonitor,"    Line search: gnorm after quadratic fit %G\n",*gnorm);CHKERRQ(ierr);
  }
  if (.5*(*gnorm)*(*gnorm) < .5*fnorm*fnorm + lambda*vi->alpha*initslope) { /* sufficient reduction */
    if (vi->lsmonitor) {
      ierr = PetscViewerASCIIMonitorPrintf(vi->lsmonitor,"    Line search: Quadratically determined step, lambda=%18.16e\n",lambda);CHKERRQ(ierr);
    }
    goto theend1;
  }

  /* Fit points with cubic */
  count = 1;
  while (PETSC_TRUE) {
    if (lambda <= minlambda) { 
      if (vi->lsmonitor) {
	ierr = PetscViewerASCIIMonitorPrintf(vi->lsmonitor,"    Line search: unable to find good step length! After %D tries \n",count);CHKERRQ(ierr);
	ierr = PetscViewerASCIIMonitorPrintf(vi->lsmonitor,"    Line search: fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, minlambda=%18.16e, lambda=%18.16e, initial slope=%18.16e\n",fnorm,*gnorm,*ynorm,minlambda,lambda,initslope);CHKERRQ(ierr);
      }
      *flag = PETSC_FALSE; 
      break;
    }
    t1 = .5*((*gnorm)*(*gnorm) - fnorm*fnorm) - lambda*initslope;
    t2 = .5*(gnormprev*gnormprev  - fnorm*fnorm) - lambdaprev*initslope;
    a  = (t1/(lambda*lambda) - t2/(lambdaprev*lambdaprev))/(lambda-lambdaprev);
    b  = (-lambdaprev*t1/(lambda*lambda) + lambda*t2/(lambdaprev*lambdaprev))/(lambda-lambdaprev);
    d  = b*b - 3*a*initslope;
    if (d < 0.0) d = 0.0;
    if (a == 0.0) {
      lambdatemp = -initslope/(2.0*b);
    } else {
      lambdatemp = (-b + sqrt(d))/(3.0*a);
    }
    lambdaprev = lambda;
    gnormprev  = *gnorm;
    if (lambdatemp > .5*lambda)  lambdatemp = .5*lambda;
    if (lambdatemp <= .1*lambda) lambda     = .1*lambda;
    else                         lambda     = lambdatemp;
    ierr  = VecWAXPY(w,-lambda,y,x);CHKERRQ(ierr);
    if (snes->nfuncs >= snes->max_funcs) {
      ierr = PetscInfo1(snes,"Exceeded maximum function evaluations, while looking for good step length! %D \n",count);CHKERRQ(ierr);
      ierr = PetscInfo5(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lambda=%18.16e, initial slope=%18.16e\n",fnorm,*gnorm,*ynorm,lambda,initslope);CHKERRQ(ierr);
      *flag = PETSC_FALSE;
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      break;
    }
    ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
    if (snes->domainerror) {
      ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
    if PetscIsInfOrNanReal(*gnorm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
    if (.5*(*gnorm)*(*gnorm) < .5*fnorm*fnorm + lambda*vi->alpha*initslope) { /* is reduction enough? */
      if (vi->lsmonitor) {
	ierr = PetscPrintf(comm,"    Line search: Cubically determined step, current gnorm %G lambda=%18.16e\n",*gnorm,lambda);CHKERRQ(ierr);
      }
      break;
    } else {
      if (vi->lsmonitor) {
        ierr = PetscPrintf(comm,"    Line search: Cubic step no good, shrinking lambda, current gnorm %G lambda=%18.16e\n",*gnorm,lambda);CHKERRQ(ierr);
      }
    }
    count++;
  }
  theend1:
  /* Optional user-defined check for line search step validity */
  if (vi->postcheckstep && *flag) {
    ierr = (*vi->postcheckstep)(snes,x,y,w,vi->postcheck,&changed_y,&changed_w);CHKERRQ(ierr);
    if (changed_y) {
      ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);
    }
    if (changed_y || changed_w) { /* recompute the function if the step has changed */
      ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
      if (snes->domainerror) {
        ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
      ierr = VecNormBegin(g,NORM_2,gnorm);CHKERRQ(ierr);
      if PetscIsInfOrNanReal(*gnorm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
      ierr = VecNormBegin(y,NORM_2,ynorm);CHKERRQ(ierr);
      ierr = VecNormEnd(g,NORM_2,gnorm);CHKERRQ(ierr);
      ierr = VecNormEnd(y,NORM_2,ynorm);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchQuadratic_VI"
/*
  This routine is a copy of SNESLineSearchQuadratic in snes/impls/ls/ls.c
*/
PetscErrorCode SNESLineSearchQuadratic_VI(SNES snes,void *lsctx,Vec x,Vec f,Vec g,Vec y,Vec w,PetscReal fnorm,PetscReal xnorm,PetscReal *ynorm,PetscReal *gnorm,PetscBool *flag)
{
  /* 
     Note that for line search purposes we work with with the related
     minimization problem:
        min  z(x):  R^n -> R,
     where z(x) = .5 * fnorm*fnorm,and fnorm = || f ||_2.
     z(x) is the same as the merit function for the complementarity problem
   */
  PetscReal      initslope,minlambda,lambda,lambdatemp,rellength;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    cinitslope;
#endif
  PetscErrorCode ierr;
  PetscInt       count;
  SNES_VI        *vi = (SNES_VI*)snes->data;
  PetscBool     changed_w = PETSC_FALSE,changed_y = PETSC_FALSE;

  PetscFunctionBegin;
  ierr    = PetscLogEventBegin(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  *flag   = PETSC_TRUE;

  ierr = VecNorm(y,NORM_2,ynorm);CHKERRQ(ierr);
  if (*ynorm == 0.0) {
    if (vi->lsmonitor) {
      ierr = PetscViewerASCIIMonitorPrintf(vi->lsmonitor,"Line search: Direction and size is 0\n");CHKERRQ(ierr);
    }
    *gnorm = fnorm;
    ierr   = VecCopy(x,w);CHKERRQ(ierr);
    ierr   = VecCopy(f,g);CHKERRQ(ierr);
    *flag  = PETSC_FALSE;
    goto theend2;
  }
  if (*ynorm > vi->maxstep) {	/* Step too big, so scale back */
    ierr   = VecScale(y,vi->maxstep/(*ynorm));CHKERRQ(ierr);
    *ynorm = vi->maxstep;
  }
  ierr      = VecMaxPointwiseDivide(y,x,&rellength);CHKERRQ(ierr);
  minlambda = vi->minlambda/rellength;
#if defined(PETSC_USE_COMPLEX)
  ierr      = VecDot(vi->dpsi,y,&cinitslope);CHKERRQ(ierr);
  initslope = PetscRealPart(cinitslope);
#else
  ierr = VecDot(vi->dpsi,y,&initslope);CHKERRQ(ierr);
#endif
  if (initslope > 0.0)  initslope = -initslope;
  if (initslope == 0.0) initslope = -1.0;
  ierr = PetscInfo1(snes,"Initslope %G \n",initslope);CHKERRQ(ierr);

  ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);
  if (snes->nfuncs >= snes->max_funcs) {
    ierr  = PetscInfo(snes,"Exceeded maximum function evaluations, while checking full step length!\n");CHKERRQ(ierr);
    *flag = PETSC_FALSE;
    snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
    goto theend2;
  }
  ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  if (snes->domainerror) {
    ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
  if PetscIsInfOrNanReal(*gnorm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  if (.5*(*gnorm)*(*gnorm) <= .5*fnorm*fnorm + vi->alpha*initslope) { /* Sufficient reduction */
    if (vi->lsmonitor) {
      ierr = PetscViewerASCIIMonitorPrintf(vi->lsmonitor,"    Line search: Using full step: fnorm %G gnorm %G\n",fnorm,*gnorm);CHKERRQ(ierr);
    }
    goto theend2;
  }

  /* Fit points with quadratic */
  lambda = 1.0;
  count = 1;
  while (PETSC_TRUE) {
    if (lambda <= minlambda) { /* bad luck; use full step */
      if (vi->lsmonitor) {
        ierr = PetscViewerASCIIMonitorPrintf(vi->lsmonitor,"Line search: Unable to find good step length! %D \n",count);CHKERRQ(ierr);
        ierr = PetscViewerASCIIMonitorPrintf(vi->lsmonitor,"Line search: fnorm=%G, gnorm=%G, ynorm=%G, lambda=%G, initial slope=%G\n",fnorm,*gnorm,*ynorm,lambda,initslope);CHKERRQ(ierr);
      }
      ierr = VecCopy(x,w);CHKERRQ(ierr);
      *flag = PETSC_FALSE;
      break;
    }
    lambdatemp = -initslope/((*gnorm)*(*gnorm) - fnorm*fnorm - 2.0*initslope);
    if (lambdatemp > .5*lambda)  lambdatemp = .5*lambda;
    if (lambdatemp <= .1*lambda) lambda     = .1*lambda; 
    else                         lambda     = lambdatemp;
    
    ierr = VecWAXPY(w,-lambda,y,x);CHKERRQ(ierr);
    if (snes->nfuncs >= snes->max_funcs) {
      ierr  = PetscInfo1(snes,"Exceeded maximum function evaluations, while looking for good step length! %D \n",count);CHKERRQ(ierr);
      ierr  = PetscInfo5(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lambda=%18.16e, initial slope=%18.16e\n",fnorm,*gnorm,*ynorm,lambda,initslope);CHKERRQ(ierr);
      *flag = PETSC_FALSE;
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      break;
    }
    ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
    if (snes->domainerror) {
      ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
    if PetscIsInfOrNanReal(*gnorm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
    if (.5*(*gnorm)*(*gnorm) < .5*fnorm*fnorm + lambda*vi->alpha*initslope) { /* sufficient reduction */
      if (vi->lsmonitor) {
        ierr = PetscViewerASCIIMonitorPrintf(vi->lsmonitor,"    Line Search: Quadratically determined step, lambda=%G\n",lambda);CHKERRQ(ierr);
      }
      break;
    }
    count++;
  }
  theend2:
  /* Optional user-defined check for line search step validity */
  if (vi->postcheckstep) {
    ierr = (*vi->postcheckstep)(snes,x,y,w,vi->postcheck,&changed_y,&changed_w);CHKERRQ(ierr);
    if (changed_y) {
      ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);
    }
    if (changed_y || changed_w) { /* recompute the function if the step has changed */
      ierr = SNESComputeFunction(snes,w,g);
      if (snes->domainerror) {
        ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
      ierr = VecNormBegin(g,NORM_2,gnorm);CHKERRQ(ierr);
      ierr = VecNormBegin(y,NORM_2,ynorm);CHKERRQ(ierr);
      ierr = VecNormEnd(g,NORM_2,gnorm);CHKERRQ(ierr);
      ierr = VecNormEnd(y,NORM_2,ynorm);CHKERRQ(ierr);
      if PetscIsInfOrNanReal(*gnorm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
    }
  }
  ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef PetscErrorCode (*FCN2)(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscBool*); /* force argument to next function to not be extern C*/
/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchSet_VI"
PetscErrorCode PETSCSNES_DLLEXPORT SNESLineSearchSet_VI(SNES snes,FCN2 func,void *lsctx)
{
  PetscFunctionBegin;
  ((SNES_VI *)(snes->data))->LineSearch = func;
  ((SNES_VI *)(snes->data))->lsP        = lsctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchSetMonitor_VI"
PetscErrorCode PETSCSNES_DLLEXPORT SNESLineSearchSetMonitor_VI(SNES snes,PetscBool flg)
{
  SNES_VI        *vi = (SNES_VI*)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (flg && !vi->lsmonitor) {
    ierr = PetscViewerASCIIMonitorCreate(((PetscObject)snes)->comm,"stdout",((PetscObject)snes)->tablevel,&vi->lsmonitor);CHKERRQ(ierr);
  } else if (!flg && vi->lsmonitor) {
    ierr = PetscViewerASCIIMonitorDestroy(vi->lsmonitor);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
   SNESView_VI - Prints info from the SNESVI data structure.

   Input Parameters:
.  SNES - the SNES context
.  viewer - visualization context

   Application Interface Routine: SNESView()
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESView_VI"
static PetscErrorCode SNESView_VI(SNES snes,PetscViewer viewer)
{
  SNES_VI        *vi = (SNES_VI *)snes->data;
  const char     *cstr,*tstr;
  PetscErrorCode ierr;
  PetscBool     iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (vi->LineSearch == SNESLineSearchNo_VI)             cstr = "SNESLineSearchNo";
    else if (vi->LineSearch == SNESLineSearchQuadratic_VI) cstr = "SNESLineSearchQuadratic";
    else if (vi->LineSearch == SNESLineSearchCubic_VI)     cstr = "SNESLineSearchCubic";
    else                                                cstr = "unknown";
    if (snes->ops->solve == SNESSolveVI_SS)      tstr = "Semismooth";
    else if (snes->ops->solve == SNESSolveVI_AS)  tstr = "Active Set";
    else if (snes->ops->solve == SNESSolveVI_RS) tstr = "Reduced Space";
    else                                         tstr = "unknown";
    ierr = PetscViewerASCIIPrintf(viewer,"  VI algorithm: %s\n",tstr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  line search variant: %s\n",cstr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  alpha=%G, maxstep=%G, minlambda=%G\n",vi->alpha,vi->maxstep,vi->minlambda);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported for SNES EQ VI",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/*
   SNESVISetVariableBounds - Sets the lower and upper bounds for the solution vector. xl <= x <= xu.

   Input Parameters:
.  snes - the SNES context.
.  xl   - lower bound.
.  xu   - upper bound.

   Notes:
   If this routine is not called then the lower and upper bounds are set to 
   PETSC_VI_INF and PETSC_VI_NINF respectively during SNESSetUp().

*/

#undef __FUNCT__
#define __FUNCT__ "SNESVISetVariableBounds"
PetscErrorCode SNESVISetVariableBounds(SNES snes, Vec xl, Vec xu)
{
  SNES_VI        *vi = (SNES_VI*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(xl,VEC_CLASSID,2);
  PetscValidHeaderSpecific(xu,VEC_CLASSID,3);

  /* Check if SNESSetFunction is called */
  if(!snes->vec_func) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call SNESSetFunction() first");

  /* Check if the vector sizes are compatible for lower and upper bounds */
  if (xl->map->N != snes->vec_func->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector lengths lower bound = %D solution vector = %D",xl->map->N,snes->vec_func->map->N);
  if (xu->map->N != snes->vec_func->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector lengths: upper bound = %D solution vector = %D",xu->map->N,snes->vec_func->map->N);
  vi->usersetxbounds = PETSC_TRUE;
  vi->xl = xl;
  vi->xu = xu;

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   SNESSetFromOptions_VI - Sets various parameters for the SNESVI method.

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESSetFromOptions()
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESSetFromOptions_VI"
static PetscErrorCode SNESSetFromOptions_VI(SNES snes)
{
  SNES_VI        *vi = (SNES_VI *)snes->data;
  const char     *lses[] = {"basic","basicnonorms","quadratic","cubic"};
  const char     *vies[] = {"ss","as","rs"};
  PetscErrorCode ierr;
  PetscInt       indx;
  PetscBool      flg,set,flg2;

  PetscFunctionBegin;
    ierr = PetscOptionsHead("SNES semismooth method options");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-snes_vi_monitor","Monitor all non-active variables","None",PETSC_FALSE,&flg,0);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESMonitorSet(snes,SNESMonitorVI,0,0);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-snes_vi_alpha","Function norm must decrease by","None",vi->alpha,&vi->alpha,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_vi_maxstep","Step must be less than","None",vi->maxstep,&vi->maxstep,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_vi_minlambda","Minimum lambda allowed","None",vi->minlambda,&vi->minlambda,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_vi_const_tol","constraint tolerance","None",vi->const_tol,&vi->const_tol,0);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-snes_vi_lsmonitor","Print progress of line searches","SNESLineSearchSetMonitor",vi->lsmonitor ? PETSC_TRUE : PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
    if (set) {ierr = SNESLineSearchSetMonitor(snes,flg);CHKERRQ(ierr);}
    ierr = PetscOptionsEList("-snes_vi_type","Semismooth algorithm used","",vies,3,"ss",&indx,&flg2);CHKERRQ(ierr);
    if (flg2) {
      switch (indx) {
      case 0:
	snes->ops->solve = SNESSolveVI_SS;
	break;
      case 1:
	snes->ops->solve = SNESSolveVI_AS;
	break;
      case 2:
        snes->ops->solve = SNESSolveVI_RS;
        break;
      }
    }
    ierr = PetscOptionsEList("-snes_vi_ls","Line search used","SNESLineSearchSet",lses,4,"cubic",&indx,&flg);CHKERRQ(ierr);
    if (flg) {
      switch (indx) {
      case 0:
        ierr = SNESLineSearchSet(snes,SNESLineSearchNo_VI,PETSC_NULL);CHKERRQ(ierr);
        break;
      case 1:
        ierr = SNESLineSearchSet(snes,SNESLineSearchNoNorms_VI,PETSC_NULL);CHKERRQ(ierr);
        break;
      case 2:
        ierr = SNESLineSearchSet(snes,SNESLineSearchQuadratic_VI,PETSC_NULL);CHKERRQ(ierr);
        break;
      case 3:
        ierr = SNESLineSearchSet(snes,SNESLineSearchCubic_VI,PETSC_NULL);CHKERRQ(ierr);
        break;
      }
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*MC
      SNESVI - Semismooth newton method based nonlinear solver that uses a line search

   Options Database:
+   -snes_vi [cubic,quadratic,basic,basicnonorms] - Selects line search
.   -snes_vi_alpha <alpha> - Sets alpha
.   -snes_vi_maxstep <maxstep> - Sets the maximum stepsize the line search will use (if the 2-norm(y) > maxstep then scale y to be y = (maxstep/2-norm(y)) *y)
.   -snes_vi_minlambda <minlambda>  - Sets the minimum lambda the line search will use  minlambda / max_i ( y[i]/x[i] )
-   -snes_vi_monitor - print information about progress of line searches 


   Level: beginner

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESTR, SNESLineSearchSet(), 
           SNESLineSearchSetPostCheck(), SNESLineSearchNo(), SNESLineSearchCubic(), SNESLineSearchQuadratic(), 
           SNESLineSearchSet(), SNESLineSearchNoNorms(), SNESLineSearchSetPreCheck(), SNESLineSearchSetParams(), SNESLineSearchGetParams()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESCreate_VI"
PetscErrorCode PETSCSNES_DLLEXPORT SNESCreate_VI(SNES snes)
{
  PetscErrorCode ierr;
  SNES_VI      *vi;

  PetscFunctionBegin;
  snes->ops->setup	     = SNESSetUp_VI;
  snes->ops->solve	     = SNESSolveVI_SS;
  snes->ops->destroy	     = SNESDestroy_VI;
  snes->ops->setfromoptions  = SNESSetFromOptions_VI;
  snes->ops->view            = SNESView_VI;
  snes->ops->converged       = SNESDefaultConverged_VI;

  ierr                   = PetscNewLog(snes,SNES_VI,&vi);CHKERRQ(ierr);
  snes->data    	 = (void*)vi;
  vi->alpha		 = 1.e-4;
  vi->maxstep		 = 1.e8;
  vi->minlambda         = 1.e-12;
  vi->LineSearch        = SNESLineSearchCubic_VI;
  vi->lsP               = PETSC_NULL;
  vi->postcheckstep     = PETSC_NULL;
  vi->postcheck         = PETSC_NULL;
  vi->precheckstep      = PETSC_NULL;
  vi->precheck          = PETSC_NULL;
  vi->const_tol         =  2.2204460492503131e-16;
  vi->computessfunction = ComputeFischerFunction; 

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetMonitor_C","SNESLineSearchSetMonitor_VI",SNESLineSearchSetMonitor_VI);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSet_C","SNESLineSearchSet_VI",SNESLineSearchSet_VI);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END
