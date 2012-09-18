/*
   This should not be included in users code.

  Includes definition of structure for seqential vectors

  These are shared by dvec1.c dvec2.c dvec3.c bvec1.c bvec2.c pvec.c pbvec.c
*/

#ifndef __DVECIMPL
#define __DVECIMPL

#include <petsc-private/vecimpl.h>

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
extern PetscErrorCode VecResetArray_Seq(Vec);
extern PetscErrorCode VecReplaceArray_Seq(Vec,const PetscScalar *);
extern PetscErrorCode VecDot_Seq(Vec,Vec,PetscScalar *);
extern PetscErrorCode VecTDot_Seq(Vec,Vec,PetscScalar *);
extern PetscErrorCode VecScale_Seq(Vec,PetscScalar);
extern PetscErrorCode VecAXPY_Seq(Vec,PetscScalar,Vec);
extern PetscErrorCode VecAXPBY_Seq(Vec,PetscScalar,PetscScalar,Vec);
extern PetscErrorCode VecMax_Seq(Vec,PetscInt*,PetscReal *);
extern PetscErrorCode VecNorm_Seq(Vec,NormType,PetscReal*);
extern PetscErrorCode VecDestroy_Seq(Vec);
extern PetscErrorCode VecDuplicate_Seq(Vec,Vec*);
extern PetscErrorCode VecSetOption_Seq(Vec,VecOption,PetscBool);
extern PetscErrorCode VecGetValues_Seq(Vec,PetscInt,const PetscInt*,PetscScalar*);
extern PetscErrorCode VecSetValues_Seq(Vec,PetscInt,const PetscInt*,const PetscScalar*,InsertMode);
extern PetscErrorCode VecSetValuesBlocked_Seq(Vec,PetscInt,const PetscInt*,const PetscScalar*,InsertMode);
extern PetscErrorCode VecView_Seq(Vec,PetscViewer);
extern PetscErrorCode VecGetSize_Seq(Vec,PetscInt*);
extern PetscErrorCode VecCopy_Seq(Vec,Vec);
extern PetscErrorCode VecSwap_Seq(Vec,Vec);
extern PetscErrorCode VecConjugate_Seq(Vec);
extern PetscErrorCode VecSetRandom_Seq(Vec,PetscRandom);
extern PetscErrorCode VecPointwiseMult_Seq(Vec,Vec,Vec);
extern PetscErrorCode VecPointwiseMax_Seq(Vec,Vec,Vec);
extern PetscErrorCode VecPointwiseMaxAbs_Seq(Vec,Vec,Vec);
extern PetscErrorCode VecPointwiseMin_Seq(Vec,Vec,Vec);
extern PetscErrorCode VecPointwiseDivide_Seq(Vec,Vec,Vec);

EXTERN_C_BEGIN
extern PetscErrorCode  VecCreate_Seq(Vec);
EXTERN_C_END
extern PetscErrorCode VecCreate_Seq_Private(Vec,const PetscScalar[]);

#endif
