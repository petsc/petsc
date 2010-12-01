#ifndef __TAO_TRON_H
#define __TAO_TRON_H

#include "private/taosolver_impl.h"
#include "petscis.h"

typedef struct {

  /* Parameters */
  PetscReal pg_ftol;
  PetscReal actred;
  PetscReal f_new;
 
  PetscReal eta1,eta2,eta3,eta4;
  PetscReal sigma1,sigma2,sigma3;

  PetscInt maxgpits;

  /* Problem variables, vectors and index sets */
  PetscReal stepsize;
  PetscReal pgstepsize;

  /* Problem statistics */

  PetscInt n;   /* Dimension of the Problem */
  PetscReal delta;  /* Trust region size */
  PetscReal gnorm;
  PetscReal f;

  PetscInt total_cgits;
  PetscInt cg_iterates;
  PetscInt total_gp_its;
  PetscInt gp_iterates;
  PetscInt cgits;

  Vec DXFree;
  Vec R;

  Vec X;
  Vec G;
  Vec PG;

  Vec X_New;
  Vec G_New;
  Vec Work;
  
  Mat H_sub;
  Mat Hpre_sub;

  
  IS Free_Local;  /* Indices of local variables equal to lower bound */
  VecScatter Lower_Local;  /* Indices of local variables equal to lower bound */
  VecScatter Upper_Local;  /* Indices of local variables equal to lower bound */

  PetscInt n_free;       /* Number of free variables */
  PetscInt n_upper;
  PetscInt n_lower;
  PetscInt n_bind;       /* Number of binding varibles */
  PetscInt subset_type;

} TAO_TRON;

#endif

