/* $Id: dvecimpl.h,v 1.23 2001/09/11 16:31:56 bsmith Exp $ */
/* 
   This should not be included in users code.

  Includes definition of structure for seqential double precision vectors

  These are shared by dvec1.c dvec2.c dvec3.c bvec1.c bvec2.c 
  pvectors/pvec.c pvectors/pbvec.c 
*/

#ifndef __DVECIMPL 
#define __DVECIMPL

#include "vecimpl.h"

typedef struct { 
  VECHEADER
} Vec_Seq;

EXTERN int VecMDot_Seq(int,Vec,const Vec[],PetscScalar *);
EXTERN int VecMTDot_Seq(int,Vec,const Vec[],PetscScalar *);
EXTERN int VecMin_Seq(Vec,int*,PetscReal *);
EXTERN int VecSet_Seq(const PetscScalar*,Vec);
EXTERN int VecSetRandom_Seq(PetscRandom,Vec);
EXTERN int VecMAXPY_Seq(int,const PetscScalar *,Vec,Vec *);
EXTERN int VecAYPX_Seq(const PetscScalar *,Vec,Vec);
EXTERN int VecWAXPY_Seq(const PetscScalar*,Vec,Vec,Vec);
EXTERN int VecPointwiseMult_Seq(Vec,Vec,Vec);
EXTERN int VecPointwiseDivide_Seq(Vec,Vec,Vec);
EXTERN int VecMaxPointwiseDivide_Seq(Vec,Vec,PetscReal*);
EXTERN int VecGetArray_Seq(Vec,PetscScalar *[]);
EXTERN int VecRestoreArray_Seq(Vec,PetscScalar *[]);
EXTERN int VecPlaceArray_Seq(Vec,const PetscScalar *);
EXTERN int VecResetArray_Seq(Vec);
EXTERN int VecReplaceArray_Seq(Vec,const PetscScalar *);
EXTERN int VecGetSize_Seq(Vec,int *);
EXTERN int VecDot_Seq(Vec,Vec,PetscScalar *);
EXTERN int VecTDot_Seq(Vec,Vec,PetscScalar *);
EXTERN int VecScale_Seq(const PetscScalar *,Vec);
EXTERN int VecCopy_Seq(Vec,Vec);
EXTERN int VecSwap_Seq(Vec,Vec);
EXTERN int VecAXPY_Seq(const PetscScalar *,Vec,Vec);
EXTERN int VecAXPBY_Seq(const PetscScalar *,const PetscScalar *,Vec,Vec);
EXTERN int VecMax_Seq(Vec,int*,PetscReal *);
EXTERN int VecDuplicate_Seq(Vec,Vec *);
EXTERN int VecConjugate_Seq(Vec);
EXTERN int VecNorm_Seq(Vec,NormType,PetscReal*);
#endif
