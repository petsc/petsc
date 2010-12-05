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

extern PetscErrorCode VecMDot_Seq(Vec,PetscInt,const Vec[],PetscScalar *);
extern PetscErrorCode VecMTDot_Seq(Vec,PetscInt,const Vec[],PetscScalar *);
extern PetscErrorCode VecMin_Seq(Vec,PetscInt*,PetscReal *);
extern PetscErrorCode VecSet_Seq(Vec,PetscScalar);
extern PetscErrorCode VecMAXPY_Seq(Vec,PetscInt,const PetscScalar *,Vec *);
extern PetscErrorCode VecAYPX_Seq(Vec,PetscScalar,Vec);
extern PetscErrorCode VecWAXPY_Seq(Vec,PetscScalar,Vec,Vec);
extern PetscErrorCode VecAXPBYPCZ_Seq(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);
extern PetscErrorCode VecMaxPointwiseDivide_Seq(Vec,Vec,PetscReal*);
extern PetscErrorCode VecPlaceArray_Seq(Vec,const PetscScalar *);
extern PetscErrorCode VecReplaceArray_Seq(Vec,const PetscScalar *);
extern PetscErrorCode VecDot_Seq(Vec,Vec,PetscScalar *);
extern PetscErrorCode VecTDot_Seq(Vec,Vec,PetscScalar *);
extern PetscErrorCode VecScale_Seq(Vec,PetscScalar);
extern PetscErrorCode VecAXPY_Seq(Vec,PetscScalar,Vec);
extern PetscErrorCode VecAXPBY_Seq(Vec,PetscScalar,PetscScalar,Vec);
extern PetscErrorCode VecMax_Seq(Vec,PetscInt*,PetscReal *);
extern PetscErrorCode VecNorm_Seq(Vec,NormType,PetscReal*);
EXTERN_C_BEGIN
extern PetscErrorCode  VecCreate_Seq(Vec);
EXTERN_C_END
extern PetscErrorCode VecCreate_Seq_Private(Vec,const PetscScalar[]);

#endif
