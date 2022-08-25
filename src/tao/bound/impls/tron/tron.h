#ifndef __TAO_TRON_H
#define __TAO_TRON_H

#include <petsc/private/taoimpl.h>
#include <petscis.h>

typedef struct {
  /* Parameters */
  PetscReal pg_ftol;
  PetscReal actred;
  PetscReal f_new;

  PetscReal eta1, eta2, eta3, eta4;
  PetscReal sigma1, sigma2, sigma3;

  PetscInt maxgpits;

  /* Problem variables, vectors and index sets */
  PetscReal stepsize;
  PetscReal pgstepsize;

  /* Problem statistics */

  PetscInt  n;     /* Dimension of the Problem */
  PetscReal delta; /* Trust region size */
  PetscReal gnorm;
  PetscReal f;

  PetscInt total_gp_its;
  PetscInt gp_iterates;

  Vec X_New;
  Vec G_New;
  Vec Work;

  /* Subvectors and submatrices */
  Vec DXFree;
  Vec R;
  Vec rmask;
  Vec diag;
  Mat H_sub;
  Mat Hpre_sub;

  IS         Free_Local; /* Indices of local variables equal to lower bound */
  VecScatter scatter;

  PetscInt n_free; /* Number of free variables */
  PetscInt n_free_last;

} TAO_TRON;

#endif
