/*
    Context for conjugate gradient method (unconstrained minimization)
 */

#ifndef __TAO_CG_H
#define __TAO_CG_H

#include <petsc/private/taoimpl.h>

typedef struct {
  Vec G_old;
  Vec X_old;
  Vec W; /*  work vector */

  PetscReal eta;       /*  Restart tolerance */
  PetscReal delta_max; /*  Minimum value for scaling */
  PetscReal delta_min; /*  Maximum value for scaling */

  /*  The algorithm restarts when the gradient at the current point g_k,
       and the gradient of the previous point, g_{k-1}, satisfy the
       following inequality:

            abs(inner(g_k, g_{k-1})) > eta * norm(g_k, 2)^2. */

  PetscInt ngradsteps;  /*  Number of gradient steps */
  PetscInt nresetsteps; /*  Number of reset steps */

  PetscInt cg_type; /*  Formula to use */
} TAO_CG;

#endif /* ifndef __TAO_CG_H */
