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



/*
  TaoSubset - Object used to manage subsets of vector and matrices

  Level: beginner
  
.seealso: TaoSubsetCreate()
*/

#define TaoSubsetType const char*
#define TAOSUBSET_SINGLEPROCESSOR "singleprocessor"
#define TAOSUBSET_NOREDISTRIBUTE "noredistribute"
#define TAOSUBSET_REDISTRIBUTE   "redistribute"
#define TAOSUBSET_MASK           "mask"
#define TAOSUBSET_MATRIXFREE     "matrixfree"

PetscErrorCode VecWhichBetween(Vec, Vec, Vec, IS *);
PetscErrorCode VecWhichBetweenOrEqual(Vec, Vec, Vec, IS *);
PetscErrorCode VecWhichGreaterThan(Vec, Vec, IS * );
PetscErrorCode VecWhichLessThan(Vec, Vec, IS *);
PetscErrorCode VecWhichEqual(Vec, Vec, IS *);

PetscErrorCode VecGetSubVec(Vec, IS, Vec*);
PetscErrorCode VecReducedXPY(Vec, Vec, IS); 

#endif /* defined __TAOUTIL_H */
