/*
    Context for bound-constrained nonlinear conjugate gradient method
 */


#ifndef __TAO_BNCG_H
#define __TAO_BNCG_H

#include <petsc/private/taoimpl.h>

typedef struct {
    Vec G_old, X_old, W, work;
    Vec unprojected_gradient, unprojected_gradient_old;
    IS  active_lower, active_upper, active_fixed, active_idx, inactive_idx, inactive_old, new_inactives;
    Vec inactive_grad, inactive_step;
    
    PetscInt  as_type;
    PetscReal as_step, as_tol;

    PetscReal f;
    PetscReal rho, pow;
    PetscReal eta;         /*  Restart tolerance */
    
    PetscBool recycle;

    PetscInt cg_type;           /*  Formula to use */

    PetscInt ls_fails, resets, broken_ortho, descent_error;
} TAO_BNCG;

#endif /* ifndef __TAO_BNCG_H */

PETSC_INTERN PetscErrorCode TaoBNCGEstimateActiveSet(Tao, PetscInt);
PETSC_INTERN PetscErrorCode TaoBNCGBoundStep(Tao, PetscInt, Vec);
PETSC_EXTERN PetscErrorCode TaoBNCGSetRecycleFlag(Tao, PetscBool);
