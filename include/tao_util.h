#ifndef __TAO_UTIL_H
#define __TAO_UTIL_H
#include "petscvec.h"

PetscErrorCode VecPow(Vec Vec1, PetscReal p);
PetscErrorCode VecMedian(Vec Vec1, Vec Vec2, Vec Vec3, Vec VMedian);
PetscErrorCode VecCompare(Vec, Vec, PetscTruth *);
PetscErrorCode VecMedian(Vec, Vec, Vec, Vec);
PetscErrorCode VecFischer(Vec, Vec, Vec, Vec, Vec);
PetscErrorCode VecSFischer(Vec, Vec, Vec, Vec, PetscScalar, Vec);

PetscErrorCode VecBoundGradientProjection(Vec, Vec, Vec, Vec, Vec);
PetscErrorCode VecStepBoundInfo(Vec,Vec,Vec,Vec,PetscReal*, PetscReal*,PetscReal*);
PetscErrorCode VecStepMax(Vec, Vec, PetscReal *);


/*
  TaoSubset - Object used to manage subsets of vector and matrices

  Level: beginner
  
.seealso: TaoSubsetCreate()
*/



#define TAOSUBSET_SINGLEPROCESSOR 0
#define TAOSUBSET_NOREDISTRIBUTE  1
#define TAOSUBSET_REDISTRIBUTE    2
#define TAOSUBSET_MASK            3
#define TAOSUBSET_MATRIXFREE      4
#define TAOSUBSET_TYPES           5

const char *TAOSUBSET[64] = {
    "singleprocessor", "noredistribute", "redistribute", "mask", "matrixfree"
};

PetscErrorCode VecWhichBetween(Vec, Vec, Vec, IS *);
PetscErrorCode VecWhichBetweenOrEqual(Vec, Vec, Vec, IS *);
PetscErrorCode VecWhichGreaterThan(Vec, Vec, IS * );
PetscErrorCode VecWhichLessThan(Vec, Vec, IS *);
PetscErrorCode VecWhichEqual(Vec, Vec, IS *);

PetscErrorCode VecGetSubVec(Vec, IS, Vec*);
PetscErrorCode VecReducedXPY(Vec, Vec, IS); 

#endif /* defined __TAOUTIL_H */
