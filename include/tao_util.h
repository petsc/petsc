#ifndef __TAOUTIL_H
#define __TAOUTIL_H
#include "petscvec.h"

PetscErrorCode TaoVecPow(Vec Vec1, PetscReal p);
PetscErrorCode TaoVecMedian(Vec Vec1, Vec Vec2, Vec Vec3, Vec VMedian);

#endif /* defined __TAOUTIL_H */
