/*
    Context for bound-constrained nonlinear conjugate gradient method
 */


#ifndef __TAO_BNCG_H
#define __TAO_BNCG_H

#include <petsc/private/taoimpl.h>

typedef struct {
    Vec G_old;
    Vec X_old;
    Vec unprojected_gradient;
    Vec unprojected_gradient_old;
    IS  inactive_set;
    Vec inactive_grad, inactive_step;

    PetscReal rho, pow;
    PetscReal eta;         /*  Restart tolerance */
    PetscReal delta_max;   /*  Minimum value for scaling */
    PetscReal delta_min;   /*  Maximum value for scaling */

    PetscInt cg_type;           /*  Formula to use */

    PetscInt ls_fails, resets, broken_ortho, descent_error;
} TAO_BNCG;

#endif /* ifndef __TAO_BNCG_H */

PETSC_INTERN PetscErrorCode TaoBNCGResetStepForNewInactives(Tao, Vec);
