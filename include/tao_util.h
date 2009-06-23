#ifndef __TAO_UTIL_H
#define __TAO_UTIL_H
#include "petscvec.h"

PetscErrorCode VecPow(Vec Vec1, PetscReal p);
PetscErrorCode VecMedian(Vec Vec1, Vec Vec2, Vec Vec3, Vec VMedian);

PetscErrorCode VecBoundGradientProjection(Vec, Vec, Vec, Vec, Vec);
PetscErrorCode VecStepBoundInfo(Vec,Vec,Vec,Vec,PetscReal*, PetscReal*,PetscReal*);

#endif /* defined __TAOUTIL_H */
