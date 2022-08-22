#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

/*
  Limited-memory Broyden's method for approximating the inverse of
  a Jacobian.
*/

typedef struct {
  Vec       *P, *Q;
  PetscBool  allocated, needP, needQ;
  PetscReal *yty, *yts;
  PetscReal *sts, *stq;
} Mat_Brdn;
