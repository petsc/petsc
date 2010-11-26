/* 
   This should not be included in users code.

  Includes definition of structure for seqential vectors

  These are shared by dvec1.c dvec2.c dvec3.c bvec1.c bvec2.c pvec.c pbvec.c 
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
EXTERN PetscErrorCode VecMAXPY_Seq(Vec,PetscInt,const PetscScalar *,Vec *);
EXTERN PetscErrorCode VecAYPX_Seq(Vec,PetscScalar,Vec);
EXTERN PetscErrorCode VecWAXPY_Seq(Vec,PetscScalar,Vec,Vec);
EXTERN PetscErrorCode VecAXPBYPCZ_Seq(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);
EXTERN PetscErrorCode VecMaxPointwiseDivide_Seq(Vec,Vec,PetscReal*);
EXTERN PetscErrorCode VecPlaceArray_Seq(Vec,const PetscScalar *);
EXTERN PetscErrorCode VecReplaceArray_Seq(Vec,const PetscScalar *);
EXTERN PetscErrorCode VecDot_Seq(Vec,Vec,PetscScalar *);
EXTERN PetscErrorCode VecTDot_Seq(Vec,Vec,PetscScalar *);
EXTERN PetscErrorCode VecScale_Seq(Vec,PetscScalar);
EXTERN PetscErrorCode VecAXPY_Seq(Vec,PetscScalar,Vec);
EXTERN PetscErrorCode VecAXPBY_Seq(Vec,PetscScalar,PetscScalar,Vec);
EXTERN PetscErrorCode VecMax_Seq(Vec,PetscInt*,PetscReal *);
EXTERN PetscErrorCode VecNorm_Seq(Vec,NormType,PetscReal*);
EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_Seq(Vec);
EXTERN_C_END
EXTERN PetscErrorCode VecCreate_Seq_Private(Vec,const PetscScalar[]);

#endif
