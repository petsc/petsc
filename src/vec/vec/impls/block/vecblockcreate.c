#include <stdlib.h>

#include <petsc.h>
#include <petscvec.h>
#include <private/vecimpl.h>

#include "vecblockimpl.h"

/*
 Implements a basic block vector implementation.
 
 Allows the definition of a block vector of the form
 {X}^T = { {x1}, {x2}, ..., {xn} }^T
 where {x1} is a vector of any type.
 
 It is possible some methods implemented will not support 
 nested block vectors. That is {x1} cannot be of type "block".
 More testing needs to be done to verify this. 
*/


/* operation prototypes */
EXTERN PetscErrorCode VecAssemblyBegin_Block( Vec v );
EXTERN PetscErrorCode VecAssemblyEnd_Block( Vec v );
EXTERN PetscErrorCode VecDestroy_Block( Vec v );
EXTERN PetscErrorCode VecSetUp_Block( Vec V );
EXTERN PetscErrorCode VecCopy_Block(Vec x,Vec y);
EXTERN PetscErrorCode VecDuplicate_Block( Vec x, Vec *y );
EXTERN PetscErrorCode VecDot_Block( Vec x, Vec y, PetscScalar *val );
EXTERN PetscErrorCode VecTDot_Block( Vec x, Vec y, PetscScalar *val );
EXTERN PetscErrorCode VecDot_Block( Vec x, Vec y, PetscScalar *val );
EXTERN PetscErrorCode VecTDot_Block( Vec x, Vec y, PetscScalar *val );
EXTERN PetscErrorCode VecAXPY_Block( Vec y, PetscScalar alpha, Vec x );
EXTERN PetscErrorCode VecAYPX_Block( Vec y, PetscScalar alpha, Vec x );
EXTERN PetscErrorCode VecAXPBY_Block( Vec y, PetscScalar alpha, PetscScalar beta, Vec x );
EXTERN PetscErrorCode VecScale_Block( Vec x, PetscScalar alpha );
EXTERN PetscErrorCode VecPointwiseMult_Block( Vec w, Vec x, Vec y );
EXTERN PetscErrorCode VecPointwiseDivide_Block( Vec w, Vec x, Vec y );
EXTERN PetscErrorCode VecReciprocal_Block( Vec x );
EXTERN PetscErrorCode VecNorm_Block( Vec xin, NormType type, PetscReal* z );
EXTERN PetscErrorCode VecMAXPY_Block( Vec y, PetscInt nv, const PetscScalar alpha[], Vec *x );
EXTERN PetscErrorCode VecMDot_Block( Vec x, PetscInt nv, const Vec y[], PetscScalar *val );
EXTERN PetscErrorCode VecMTDot_Block( Vec x, PetscInt nv, const Vec y[], PetscScalar *val );
EXTERN PetscErrorCode VecSet_Block( Vec x, PetscScalar alpha );
EXTERN PetscErrorCode VecConjugate_Block( Vec x );
EXTERN PetscErrorCode VecSwap_Block( Vec x, Vec y );
EXTERN PetscErrorCode VecWAXPY_Block( Vec w, PetscScalar alpha, Vec x, Vec y );
EXTERN PetscErrorCode VecMax_Block( Vec x, PetscInt *p, PetscReal *max );
EXTERN PetscErrorCode VecMin_Block( Vec x, PetscInt *p, PetscReal *min );
EXTERN PetscErrorCode VecView_Block( Vec x, PetscViewer viewer );
EXTERN PetscErrorCode VecGetSize_Block(Vec x,PetscInt *size);
EXTERN PetscErrorCode VecMaxPointwiseDivide_Block(Vec x,Vec y,PetscReal *max);


/* constructor */
#undef __FUNCT__  
#define __FUNCT__ "PETSc_VecBlock_SetOps"
PetscErrorCode PETSc_VecBlock_SetOps( struct _VecOps *ops )
{
	PetscFunctionBegin;

	/* 0 */
	ops->duplicate    =  VecDuplicate_Block;
	ops->duplicatevecs = VecDuplicateVecs_Default;
	ops->destroyvecs   = VecDestroyVecs_Default;
	ops->dot           = VecDot_Block;
	ops->mdot          = VecMDot_Block;
	
	/* 5 */         
	ops->norm  = VecNorm_Block;
	ops->tdot  = VecTDot_Block;
	ops->mtdot = VecMTDot_Block;
	ops->scale = VecScale_Block;
	ops->copy  = VecCopy_Block;
	
	/* 10 */        
	ops->set   = VecSet_Block;
	ops->swap  = VecSwap_Block;
	ops->axpy  = VecAXPY_Block;
	ops->axpby = VecAXPBY_Block;
	ops->maxpy = VecMAXPY_Block;
	
	/* 15 */        
	ops->aypx            = VecAYPX_Block;
	ops->waxpy           = VecWAXPY_Block;
  ops->axpbypcz        = 0;
	ops->pointwisemult   = VecPointwiseMult_Block;
	ops->pointwisedivide = VecPointwiseDivide_Block;
	/* 20 */ 
	ops->setvalues     = 0;
	ops->assemblybegin = VecAssemblyBegin_Block; //VecAssemblyBegin_Empty;
	ops->assemblyend   = VecAssemblyEnd_Block; //VecAssemblyEnd_Empty;
	ops->getarray      = 0;
	ops->getsize       = VecGetSize_Block; //VecGetSize_Empty;
	
	/* 25 */                        
	ops->getlocalsize = VecGetSize_Block; //VecGetLocalSize_Empty;
	ops->restorearray = 0;
	ops->max          = VecMax_Block;
	ops->min          = VecMin_Block;
	ops->setrandom    = 0;
	
	/* 30 */                        
	ops->setoption        = 0; //VecSetOption_Empty;
	ops->setvaluesblocked = 0;
	ops->destroy          = VecDestroy_Block;
	ops->view             = VecView_Block;
	ops->placearray       = 0;
	
	/* 35 */                        
	ops->replacearray = 0;
	ops->dot_local    = VecDot_Block; //VecDotLocal_Empty;
	ops->tdot_local   = VecTDot_Block; //VecTDotLocal_Empty;
	ops->norm_local   = VecNorm_Block; //VecNormLocal_Empty;
	ops->mdot_local   = VecMDot_Block; //VecMDotLocal_Empty;
	
	/* 40 */                        
	ops->mtdot_local    = VecMTDot_Block; //VecMTDotLocal_Empty;
	ops->load           = 0;
	ops->reciprocal     = VecReciprocal_Block;
	ops->conjugate      = VecConjugate_Block;
	ops->setlocaltoglobalmapping = 0; //VecSetLocalToGlobalMapping_Empty;
	
	/* 45 */                        
  ops->setvalueslocal          = 0; //VecSetValuesLocal_Empty;
	ops->resetarray              = 0;
	ops->setfromoptions          = 0; //VecSetFromOptions_Empty;
	ops->maxpointwisedivide      = VecMaxPointwiseDivide_Block;
	ops->load            = 0; //VecLoad_Empty;
	
	/* 50 */                        
	ops->pointwisemax    = 0;
	ops->pointwisemaxabs = 0;
	ops->pointwisemin    = 0;
	ops->getvalues       = 0;
	ops->sqrt            = 0;

	/* 55 */
	ops->abs          = 0;
	ops->exp          = 0;
	ops->shift        = 0;
	ops->create       = 0;
	ops->stridegather = 0;

	/* 60 */
	ops->stridescatter = 0;
	ops->dotnorm2      = 0;
	
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_Block"
PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_Block( Vec V )
{
	Vec_Block      *s;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	/* allocate and set pointer for implememtation data */
	ierr = PetscMalloc( sizeof(Vec_Block), &s );CHKERRQ(ierr);
	ierr = PetscMemzero( s, sizeof(Vec_Block) );CHKERRQ(ierr);
	V->data          = (void*)s;
	s->setup_called  = PETSC_FALSE;
	s->nb            = -1;
	s->v             = PETSC_NULL;
	
	ierr = PETSc_VecBlock_SetOps( V->ops );
	V->petscnative     = PETSC_TRUE;

	ierr = PetscObjectChangeTypeName((PetscObject)V,"block");CHKERRQ(ierr);
	
	ierr = VecSetUp_Block( V );CHKERRQ(ierr);
	
	/* expose block api's */
	ierr = PetscObjectComposeFunctionDynamic((PetscObject)V,
																					 "VecBlockSetSubVec_C",
																					 "VecBlockSetSubVec_Block",
																					 VecBlockSetSubVec_Block);CHKERRQ(ierr);

	ierr = PetscObjectComposeFunctionDynamic((PetscObject)V,
																					 "VecBlockSetSubVecs_C",
																					 "VecBlockSetSubVecs_Block",
																					 VecBlockSetSubVecs_Block);CHKERRQ(ierr);

	ierr = PetscObjectComposeFunctionDynamic((PetscObject)V,
																					 "VecBlockGetSubVec_C",
																					 "VecBlockGetSubVec_Block",
																					 VecBlockGetSubVec_Block);CHKERRQ(ierr);
	
	ierr = PetscObjectComposeFunctionDynamic((PetscObject)V,
																					 "VecBlockGetSubVecs_C",
																					 "VecBlockGetSubVecs_Block",
																					 VecBlockGetSubVecs_Block);CHKERRQ(ierr);
	
	
	PetscFunctionReturn(0);
}
EXTERN_C_END

