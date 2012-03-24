
#include <petsc-private/snesimpl.h>

typedef struct {
  PetscBool jacobian_start; /* start with Bi = Jii */
  PetscInt  n_restart;       /* after n iterations, set Bi_n = Jii(x)_n */
  PetscReal alpha;           /* SOR mixing parameter */
} SORQNContext;

#undef __FUNCT__  
#define __FUNCT__ "SNESSolve_SORQN"
static PetscErrorCode SNESSolve_SORQN(SNES snes)
{
  PetscErrorCode     ierr;
  SORQNContext *     sorqn;
  PetscReal          f_norm, f_norm_old;
  MatStructure       flg = DIFFERENT_NONZERO_PATTERN;
  Vec                X, F, dX, Y, B;
  PetscInt           i, j;
  PetscInt           rs, re;
  PetscScalar        dX_i, Y_i;

  PetscFunctionBegin;
  snes->reason = SNES_CONVERGED_ITERATING;
  sorqn = (SORQNContext *)snes->data;

  X = snes->vec_sol;
  F = snes->vec_func;
  Y  = snes->work[1];
  dX = snes->vec_sol_update;

  /* the diagonal of the approximate jacobian (broyden update) */
  B  = snes->work[0];

  ierr = VecGetOwnershipRange(X, &rs, &re);CHKERRQ(ierr);

  /* initial residual */ 
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }

  /* set the initial guess for the broyden update */
  if (sorqn->jacobian_start) {
    ierr = SNESComputeJacobian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);
    ierr = MatGetDiagonal(snes->jacobian, B);CHKERRQ(ierr);
  } else {
    ierr = VecSet(B, 1.0);CHKERRQ(ierr);
  }
  
  /* take the initial residual F and residual norm*/
  ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
  ierr = VecNorm(F, NORM_2, &f_norm);CHKERRQ(ierr);
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = f_norm;
  snes->iter = 0;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes, f_norm, 0);
  ierr = SNESMonitor(snes, 0, f_norm);CHKERRQ(ierr);
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,f_norm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);
  for (i = 1; i < snes->max_its; i++) {
    f_norm_old = f_norm;
    /* Compute the change in X */
    ierr = VecCopy(F, dX);CHKERRQ(ierr);
    ierr = VecScale(dX, sorqn->alpha);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(dX, dX, B);CHKERRQ(ierr);
    ierr = VecAXPY(X, -1.0, dX);CHKERRQ(ierr);
    
    /* Compute the update for B */
    ierr = VecCopy(F, Y);CHKERRQ(ierr);
    ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr); /* r^{i+1} = F(x^i) */
    ierr = VecNorm(F, NORM_2, &f_norm);CHKERRQ(ierr);
    /*check the convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i;
    snes->norm = f_norm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes, f_norm, i);
    ierr = SNESMonitor(snes, i, f_norm);CHKERRQ(ierr);
    ierr = (*snes->ops->converged)(snes,i,0.0,0.0,f_norm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) {
      PetscFunctionReturn(0);
    }
    ierr = VecAXPBY(Y, -1.0, 1.0, F);CHKERRQ(ierr); /* y^i = r^{i} - r^{i-1} */
    
    /*CHOICE: either restart, or continue doing diagonal rank-one updates */
    
    if (i % sorqn->n_restart == 0 || f_norm > 2.0*f_norm_old) {
      if (sorqn->jacobian_start) {
	ierr = SNESComputeJacobian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);
	ierr = MatGetDiagonal(snes->jacobian, B);CHKERRQ(ierr);
      } else {
	ierr = VecSet(B, 1.0);CHKERRQ(ierr);
      }      
    } else {
      for (j = rs; j < re; j++) {
	ierr = VecGetValues(dX, 1, &j, &dX_i);CHKERRQ(ierr);
	ierr = VecGetValues(Y, 1, &j, &Y_i);CHKERRQ(ierr);
#ifdef PETSC_USE_COMPLEX
	if (PetscAbs(PetscRealPart(dX_i)) > 1e-18) {
	  Y_i = Y_i / dX_i;
	  if (PetscAbs(PetscRealPart(Y_i)) > 1e-6) {
	    ierr = VecSetValues(B, 1, &j, &Y_i, INSERT_VALUES);CHKERRQ(ierr);
	  }
	}
#else
	if (PetscAbs(dX_i) > 1e-18) {
	  Y_i = Y_i / dX_i;
	  if (PetscAbs(Y_i) > 1e-6) {
	    ierr = VecSetValues(B, 1, &j, &Y_i, INSERT_VALUES);CHKERRQ(ierr);
	  }
	}
#endif
      }
      ierr = VecAssemblyBegin(B);
      ierr = VecAssemblyEnd(B);
    }
  }
  if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetUp_SORQN"
static PetscErrorCode SNESSetUp_SORQN(SNES snes)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = SNESDefaultGetWork(snes,2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESReset_SORQN"
static PetscErrorCode SNESReset_SORQN(SNES snes)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESDestroy_SORQN"
static PetscErrorCode SNESDestroy_SORQN(SNES snes)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = SNESReset_SORQN(snes);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetFromOptions_SORQN"
static PetscErrorCode SNESSetFromOptions_SORQN(SNES snes)
{

  PetscErrorCode ierr;
  SORQNContext * sorqn;

  PetscFunctionBegin;

  sorqn = (SORQNContext *)snes->data;

  ierr = PetscOptionsHead("SNES SOR-QN options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_sorqn_jacobian_start", "Start Quasi-Newton with actual Jacobian", "SNES", sorqn->jacobian_start, &sorqn->jacobian_start, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_sorqn_alpha", "SOR mixing parameter", "SNES", sorqn->alpha, &sorqn->alpha, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_sorqn_restart", "Iterations before Newton restart", "SNES", sorqn->n_restart, &sorqn->n_restart, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
      SNESSORQN - Nonlinear solver based upon the SOR-Quasi-Newton method.
      Reference:

      Martinez, J.M., SOR-Secant methods, 
      SIAM Journal on Numerical Analysis, Vol. 31, No. 1 (Feb. 1994), SIAM

      This implementation is still very experimental, and needs to be modified to use
      the inner quasi-Newton iteration on blocks of unknowns.

   Level: beginner

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESLS, SNESTR
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESCreate_SORQN"
PetscErrorCode  SNESCreate_SORQN(SNES snes)
{
  
  PetscErrorCode ierr;
  SORQNContext * sorqn;  

  PetscFunctionBegin;
  snes->ops->setup           = SNESSetUp_SORQN;
  snes->ops->solve           = SNESSolve_SORQN;
  snes->ops->destroy         = SNESDestroy_SORQN;
  snes->ops->setfromoptions  = SNESSetFromOptions_SORQN;
  snes->ops->view            = 0;
  snes->ops->reset           = SNESReset_SORQN;

  snes->usesksp             = PETSC_FALSE;

  ierr = PetscNewLog(snes, SORQNContext, &sorqn);CHKERRQ(ierr);
  snes->data = (void *) sorqn;
  sorqn->jacobian_start = PETSC_FALSE;
  sorqn->alpha = 0.25;
  sorqn->n_restart = 10;
  PetscFunctionReturn(0);
}
EXTERN_C_END
