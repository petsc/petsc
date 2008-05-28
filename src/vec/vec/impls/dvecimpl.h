/* 
   This should not be included in users code.

  Includes definition of structure for seqential double precision vectors

  These are shared by dvec1.c dvec2.c dvec3.c bvec1.c bvec2.c 
  pvectors/pvec.c pvectors/pbvec.c 
*/

#ifndef __DVECIMPL 
#define __DVECIMPL

#include "private/vecimpl.h"

typedef struct { 
  VECHEADER
} Vec_Seq;

EXTERN PetscErrorCode VecMDot_Seq(Vec,PetscInt,const Vec[],PetscScalar *);
EXTERN PetscErrorCode VecMTDot_Seq(Vec,PetscInt,const Vec[],PetscScalar *);
EXTERN PetscErrorCode VecMin_Seq(Vec,PetscInt*,PetscReal *);
EXTERN PetscErrorCode VecSet_Seq(Vec,PetscScalar);
EXTERN PetscErrorCode VecSetRandom_Seq(Vec,PetscRandom);
EXTERN PetscErrorCode VecMAXPY_Seq(Vec,PetscInt,const PetscScalar *,Vec *);
EXTERN PetscErrorCode VecAYPX_Seq(Vec,PetscScalar,Vec);
EXTERN PetscErrorCode VecWAXPY_Seq(Vec,PetscScalar,Vec,Vec);
EXTERN PetscErrorCode VecAXPBYPCZ_Seq(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);
EXTERN PetscErrorCode VecPointwiseMult_Seq(Vec,Vec,Vec);
EXTERN PetscErrorCode VecPointwiseMax_Seq(Vec,Vec,Vec);
EXTERN PetscErrorCode VecPointwiseMaxAbs_Seq(Vec,Vec,Vec);
EXTERN PetscErrorCode VecPointwiseMin_Seq(Vec,Vec,Vec);
EXTERN PetscErrorCode VecPointwiseDivide_Seq(Vec,Vec,Vec);
EXTERN PetscErrorCode VecMaxPointwiseDivide_Seq(Vec,Vec,PetscReal*);
EXTERN PetscErrorCode VecGetArray_Seq(Vec,PetscScalar *[]);
EXTERN PetscErrorCode VecRestoreArray_Seq(Vec,PetscScalar *[]);
EXTERN PetscErrorCode VecPlaceArray_Seq(Vec,const PetscScalar *);
EXTERN PetscErrorCode VecResetArray_Seq(Vec);
EXTERN PetscErrorCode VecReplaceArray_Seq(Vec,const PetscScalar *);
EXTERN PetscErrorCode VecGetSize_Seq(Vec,PetscInt *);
EXTERN PetscErrorCode VecDot_Seq(Vec,Vec,PetscScalar *);
EXTERN PetscErrorCode VecTDot_Seq(Vec,Vec,PetscScalar *);
EXTERN PetscErrorCode VecScale_Seq(Vec,PetscScalar);
EXTERN PetscErrorCode VecCopy_Seq(Vec,Vec);
EXTERN PetscErrorCode VecSwap_Seq(Vec,Vec);
EXTERN PetscErrorCode VecAXPY_Seq(Vec,PetscScalar,Vec);
EXTERN PetscErrorCode VecAXPBY_Seq(Vec,PetscScalar,PetscScalar,Vec);
EXTERN PetscErrorCode VecMax_Seq(Vec,PetscInt*,PetscReal *);
EXTERN PetscErrorCode VecDuplicate_Seq(Vec,Vec *);
EXTERN PetscErrorCode VecSetOption_Seq(Vec,VecOption,PetscTruth);
EXTERN PetscErrorCode VecConjugate_Seq(Vec);
EXTERN PetscErrorCode VecNorm_Seq(Vec,NormType,PetscReal*);
#endif
