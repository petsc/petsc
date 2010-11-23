
#include <stdlib.h>

#include <petsc.h>
#include <petscvec.h>
#include <private/vecimpl.h>

#include "vecblockimpl.h"
#include "vecblockprivate.h"


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecBlockSetSubVecs_Block"
PetscErrorCode PETSCVEC_DLLEXPORT VecBlockSetSubVecs_Block(Vec V,PetscInt m,const PetscInt idxm[],const Vec vec[])
{
  Vec_Block *b = (Vec_Block*)V->data;;
  PetscInt i;
  PetscInt row;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  for( i=0; i<m; i++ ) {
    row = idxm[i];
    if( row >= b->nb ) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,b->nb-1);
    if( b->v[row] == PETSC_NULL ) {
      ierr = PetscObjectReference( (PetscObject)vec[i] );CHKERRQ(ierr);
      b->v[row] = vec[i];
    }
    else {
      ierr = PetscObjectReference( (PetscObject)vec[i] );CHKERRQ(ierr);
      ierr = VecDestroy(b->v[row]);CHKERRQ(ierr);
      b->v[row] = vec[i];
    }
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "VecBlockSetSubVecs"
PetscErrorCode PETSCVEC_DLLEXPORT VecBlockSetSubVecs(Vec V,PetscInt m,const PetscInt idxm[],const Vec vec[])
{
  PetscErrorCode ierr,(*f)(Vec,PetscInt,const PetscInt*,const Vec*);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)V,"VecBlockSetSubVecs_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(V,m,idxm,vec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecBlockSetSubVec_Block"
PetscErrorCode PETSCVEC_DLLEXPORT VecBlockSetSubVec_Block(Vec V,const PetscInt idxm,const Vec vec)
{
  PetscErrorCode ierr;
  ierr = VecBlockSetSubVecs_Block(V,1,&idxm,&vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "VecBlockSetSubVec"
PetscErrorCode PETSCVEC_DLLEXPORT VecBlockSetSubVec(Vec V,const PetscInt idxm,const Vec vec)
{
  PetscErrorCode ierr,(*f)(Vec,const PetscInt,const Vec);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)V,"VecBlockSetSubVec_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(V,idxm,vec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "_VecBlockGetValues"
PetscErrorCode PETSCVEC_DLLEXPORT _VecBlockGetValues(Vec x,PetscInt m,const PetscInt idxm[],Vec vec[])
{
  Vec_Block *b = (Vec_Block*)x->data;
  PetscInt i;
  PetscInt row;
  PetscFunctionBegin;

  if (!m ) PetscFunctionReturn(0);
  for( i=0; i<m; i++ ) {
    row = idxm[i];
    if( row >= b->nb ) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,b->nb-1);
    vec[ i ] = b->v[row];
  }

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecBlockGetSubVec_Block"
PetscErrorCode PETSCVEC_DLLEXPORT VecBlockGetSubVec_Block(Vec X,PetscInt idxm,Vec *sx)
{
  _VecBlockGetValues( X, 1,&idxm, sx );
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "VecBlockGetSubVec"
PetscErrorCode PETSCVEC_DLLEXPORT VecBlockGetSubVec( Vec X, PetscInt idxm, Vec *sx )
{
  PetscErrorCode ierr,(*f)(Vec,PetscInt,Vec*);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)X,"VecBlockGetSubVec_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(X,idxm,sx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecBlockGetSubVecs_Block"
PetscErrorCode PETSCVEC_DLLEXPORT VecBlockGetSubVecs_Block(Vec X,PetscInt *N,Vec **sx)
{
  Vec_Block *b;
  b = (Vec_Block*)X->data;
  *N  = b->nb;
  *sx = b->v;

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "VecBlockGetSubVecs"
PetscErrorCode PETSCVEC_DLLEXPORT VecBlockGetSubVecs( Vec X, PetscInt *N, Vec **sx )
{
  PetscErrorCode ierr,(*f)(Vec,PetscInt*,Vec**);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)X,"VecBlockGetSubVecs_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(X,N,sx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

