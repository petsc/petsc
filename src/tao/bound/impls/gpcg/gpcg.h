#ifndef __TAO_GPCG_H
#define __TAO_GPCG_H
#include <petsc/private/taoimpl.h>
#include <petsctaolinesearch.h>

typedef struct{

  /* Parameters */
  PetscReal pg_ftol;
  PetscReal actred;
  PetscReal f_new;
  PetscReal minstep;
  PetscReal stepsize;
  PetscReal gnorm;

  PetscReal sigma1,sigma2,sigma3;

  PetscInt maxgpits;

  /* Problem variables, vectors and index sets */

  /* Problem statistics */

  PetscInt n;   /* Dimension of the Problem */

  PetscInt total_cgits;
  PetscInt cg_iterates;
  PetscInt total_gp_its;
  PetscInt gp_iterates;
  PetscInt cgits;

  Vec G_New;
  Vec DXFree;
  Vec R;
  Vec DX;
  Vec X;
  Vec X_New;
  Vec G, PG;
  Vec Work;

  Mat H;
  Vec B;
  PetscReal c;

  PetscReal f;
  PetscReal step;
  Mat Hsub;
  Mat Hsub_pre;

  IS Free_Local;  /* Indices of local variables equal to lower bound */
  IS TT;  /* Indices of local variables equal to upper bound */

  PetscInt n_free;       /* Number of free variables */
  PetscInt n_upper;
  PetscInt n_lower;
  PetscInt n_bind;       /* Number of binding varibles */
  PetscInt ksp_type;
  PetscInt subset_type;
}TAO_GPCG;



#endif






