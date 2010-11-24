#include <stdlib.h>

#include <petsc.h>
#include <petscvec.h>
#include <private/vecimpl.h>

#include "vecnestimpl.h"

/*
 Implements a basic block vector implementation.

 Allows the definition of a block vector of the form
 {X}^T = { {x1},{x2},...,{xn} }^T
 where {x1} is a vector of any type.

 It is possible some methods implemented will not support
 nested block vectors. That is {x1} cannot be of type "block".
 More testing needs to be done to verify this.
*/


/* operation prototypes */
EXTERN PetscErrorCode VecAssemblyBegin_Nest(Vec v);
EXTERN PetscErrorCode VecAssemblyEnd_Nest(Vec v);
EXTERN PetscErrorCode VecDestroy_Nest(Vec v);
EXTERN PetscErrorCode VecSetUp_Nest(Vec V);
EXTERN PetscErrorCode VecCopy_Nest(Vec x,Vec y);
EXTERN PetscErrorCode VecDuplicate_Nest(Vec x,Vec *y);
EXTERN PetscErrorCode VecDot_Nest(Vec x,Vec y,PetscScalar *val);
EXTERN PetscErrorCode VecTDot_Nest(Vec x,Vec y,PetscScalar *val);
EXTERN PetscErrorCode VecDot_Nest(Vec x,Vec y,PetscScalar *val);
EXTERN PetscErrorCode VecTDot_Nest(Vec x,Vec y,PetscScalar *val);
EXTERN PetscErrorCode VecAXPY_Nest(Vec y,PetscScalar alpha,Vec x);
EXTERN PetscErrorCode VecAYPX_Nest(Vec y,PetscScalar alpha,Vec x);
EXTERN PetscErrorCode VecAXPBY_Nest(Vec y,PetscScalar alpha,PetscScalar beta,Vec x);
EXTERN PetscErrorCode VecScale_Nest(Vec x,PetscScalar alpha);
EXTERN PetscErrorCode VecPointwiseMult_Nest(Vec w,Vec x,Vec y);
EXTERN PetscErrorCode VecPointwiseDivide_Nest(Vec w,Vec x,Vec y);
EXTERN PetscErrorCode VecReciprocal_Nest(Vec x);
EXTERN PetscErrorCode VecNorm_Nest(Vec xin,NormType type,PetscReal* z);
EXTERN PetscErrorCode VecMAXPY_Nest(Vec y,PetscInt nv,const PetscScalar alpha[],Vec *x);
EXTERN PetscErrorCode VecMDot_Nest(Vec x,PetscInt nv,const Vec y[],PetscScalar *val);
EXTERN PetscErrorCode VecMTDot_Nest(Vec x,PetscInt nv,const Vec y[],PetscScalar *val);
EXTERN PetscErrorCode VecSet_Nest(Vec x,PetscScalar alpha);
EXTERN PetscErrorCode VecConjugate_Nest(Vec x);
EXTERN PetscErrorCode VecSwap_Nest(Vec x,Vec y);
EXTERN PetscErrorCode VecWAXPY_Nest(Vec w,PetscScalar alpha,Vec x,Vec y);
EXTERN PetscErrorCode VecMax_Nest(Vec x,PetscInt *p,PetscReal *max);
EXTERN PetscErrorCode VecMin_Nest(Vec x,PetscInt *p,PetscReal *min);
EXTERN PetscErrorCode VecView_Nest(Vec x,PetscViewer viewer);
EXTERN PetscErrorCode VecGetSize_Nest(Vec x,PetscInt *size);
EXTERN PetscErrorCode VecMaxPointwiseDivide_Nest(Vec x,Vec y,PetscReal *max);

EXTERN_C_BEGIN
PetscErrorCode PETSCVEC_DLLEXPORT VecNestSetSubVecs_Nest(Vec,PetscInt,const PetscInt[],const Vec[]);
PetscErrorCode PETSCVEC_DLLEXPORT VecNestSetSubVec_Nest(Vec,const PetscInt,const Vec);
PetscErrorCode PETSCVEC_DLLEXPORT VecNestGetSubVec_Nest(Vec,PetscInt,Vec*);
PetscErrorCode PETSCVEC_DLLEXPORT VecNestGetSubVecs_Nest(Vec,PetscInt*,Vec**);
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "VecNestSetOps_Private"
static PetscErrorCode VecNestSetOps_Private(struct _VecOps *ops)
{
  PetscFunctionBegin;

  /* 0 */
  ops->duplicate               = VecDuplicate_Nest;
  ops->duplicatevecs           = VecDuplicateVecs_Default;
  ops->destroyvecs             = VecDestroyVecs_Default;
  ops->dot                     = VecDot_Nest;
  ops->mdot                    = VecMDot_Nest;

  /* 5 */
  ops->norm                    = VecNorm_Nest;
  ops->tdot                    = VecTDot_Nest;
  ops->mtdot                   = VecMTDot_Nest;
  ops->scale                   = VecScale_Nest;
  ops->copy                    = VecCopy_Nest;

  /* 10 */
  ops->set                     = VecSet_Nest;
  ops->swap                    = VecSwap_Nest;
  ops->axpy                    = VecAXPY_Nest;
  ops->axpby                   = VecAXPBY_Nest;
  ops->maxpy                   = VecMAXPY_Nest;

  /* 15 */
  ops->aypx                    = VecAYPX_Nest;
  ops->waxpy                   = VecWAXPY_Nest;
  ops->axpbypcz                = 0;
  ops->pointwisemult           = VecPointwiseMult_Nest;
  ops->pointwisedivide         = VecPointwiseDivide_Nest;
  /* 20 */
  ops->setvalues               = 0;
  ops->assemblybegin           = VecAssemblyBegin_Nest; /* VecAssemblyBegin_Empty */
  ops->assemblyend             = VecAssemblyEnd_Nest; /* VecAssemblyEnd_Empty */
  ops->getarray                = 0;
  ops->getsize                 = VecGetSize_Nest; /* VecGetSize_Empty */

  /* 25 */
  ops->getlocalsize            = VecGetSize_Nest; /* VecGetLocalSize_Empty */
  ops->restorearray            = 0;
  ops->max                     = VecMax_Nest;
  ops->min                     = VecMin_Nest;
  ops->setrandom               = 0;

  /* 30 */
  ops->setoption               = 0;    /* VecSetOption_Empty */
  ops->setvaluesblocked        = 0;
  ops->destroy                 = VecDestroy_Nest;
  ops->view                    = VecView_Nest;
  ops->placearray              = 0;

  /* 35 */
  ops->replacearray            = 0;
  ops->dot_local               = VecDot_Nest; /* VecDotLocal_Empty */
  ops->tdot_local              = VecTDot_Nest; /* VecTDotLocal_Empty */
  ops->norm_local              = VecNorm_Nest; /* VecNormLocal_Empty */
  ops->mdot_local              = VecMDot_Nest; /* VecMDotLocal_Empty */

  /* 40 */
  ops->mtdot_local             = VecMTDot_Nest; /* VecMTDotLocal_Empty */
  ops->load                    = 0;
  ops->reciprocal              = VecReciprocal_Nest;
  ops->conjugate               = VecConjugate_Nest;
  ops->setlocaltoglobalmapping = 0; /* VecSetLocalToGlobalMapping_Empty */

  /* 45 */
  ops->setvalueslocal          = 0;      /* VecSetValuesLocal_Empty */
  ops->resetarray              = 0;
  ops->setfromoptions          = 0;  /* VecSetFromOptions_Empty */
  ops->maxpointwisedivide      = VecMaxPointwiseDivide_Nest;
  ops->load                    = 0;  /* VecLoad_Empty */

  /* 50 */
  ops->pointwisemax            = 0;
  ops->pointwisemaxabs         = 0;
  ops->pointwisemin            = 0;
  ops->getvalues               = 0;
  ops->sqrt                    = 0;

  /* 55 */
  ops->abs                     = 0;
  ops->exp                     = 0;
  ops->shift                   = 0;
  ops->create                  = 0;
  ops->stridegather            = 0;

  /* 60 */
  ops->stridescatter           = 0;
  ops->dotnorm2                = 0;

  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecCreate_Nest"
PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_Nest(Vec V)
{
  Vec_Nest       *s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* allocate and set pointer for implememtation data */
  ierr = PetscMalloc(sizeof(Vec_Nest),&s);CHKERRQ(ierr);
  ierr = PetscMemzero(s,sizeof(Vec_Nest));CHKERRQ(ierr);
  V->data          = (void*)s;
  s->setup_called  = PETSC_FALSE;
  s->nb            = -1;
  s->v             = PETSC_NULL;

  ierr = VecNestSetOps_Private(V->ops);CHKERRQ(ierr);
  V->petscnative     = PETSC_TRUE;

  ierr = PetscObjectChangeTypeName((PetscObject)V,VECNEST);CHKERRQ(ierr);

  ierr = VecSetUp_Nest(V);CHKERRQ(ierr);

  /* expose block api's */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)V,"VecNestSetSubVec_C","VecNestSetSubVec_Nest",VecNestSetSubVec_Nest);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)V,"VecNestSetSubVecs_C","VecNestSetSubVecs_Nest",VecNestSetSubVecs_Nest);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)V,"VecNestGetSubVec_C","VecNestGetSubVec_Nest",VecNestGetSubVec_Nest);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)V,"VecNestGetSubVecs_C","VecNestGetSubVecs_Nest",VecNestGetSubVecs_Nest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

