
#include <stdlib.h>

#include <petsc.h>
#include <petscvec.h>
#include <private/vecimpl.h>

#include "vecnestimpl.h"

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecNestSetSubVecs_Nest"
PetscErrorCode PETSCVEC_DLLEXPORT VecNestSetSubVecs_Nest(Vec V,PetscInt m,const PetscInt idxm[],const Vec vec[])
{
  Vec_Nest       *b = (Vec_Nest*)V->data;;
  PetscInt       i;
  PetscInt       row;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    row = idxm[i];
    if (row >= b->nb) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,b->nb-1);
    if (b->v[row] == PETSC_NULL) {
      ierr = PetscObjectReference((PetscObject)vec[i]);CHKERRQ(ierr);
      b->v[row] = vec[i];
    }
    else {
      ierr = PetscObjectReference((PetscObject)vec[i]);CHKERRQ(ierr);
      ierr = VecDestroy(b->v[row]);CHKERRQ(ierr);
      b->v[row] = vec[i];
    }
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "VecNestSetSubVecs"
PetscErrorCode PETSCVEC_DLLEXPORT VecNestSetSubVecs(Vec V,PetscInt m,const PetscInt idxm[],const Vec vec[])
{
  PetscErrorCode ierr,(*f)(Vec,PetscInt,const PetscInt*,const Vec*);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)V,"VecNestSetSubVecs_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(V,m,idxm,vec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecNestSetSubVec_Nest"
PetscErrorCode PETSCVEC_DLLEXPORT VecNestSetSubVec_Nest(Vec V,const PetscInt idxm,const Vec vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNestSetSubVecs_Nest(V,1,&idxm,&vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "VecNestSetSubVec"
PetscErrorCode PETSCVEC_DLLEXPORT VecNestSetSubVec(Vec V,const PetscInt idxm,const Vec vec)
{
  PetscErrorCode ierr,(*f)(Vec,const PetscInt,const Vec);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)V,"VecNestSetSubVec_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(V,idxm,vec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecNestGetSubVecs_Private"
static PetscErrorCode VecNestGetSubVecs_Private(Vec x,PetscInt m,const PetscInt idxm[],Vec vec[])
{
  Vec_Nest  *b = (Vec_Nest*)x->data;
  PetscInt  i;
  PetscInt  row;

  PetscFunctionBegin;
  if (!m) PetscFunctionReturn(0);
  for (i=0; i<m; i++) {
    row = idxm[i];
    if (row >= b->nb) SETERRQ2(((PetscObject)x)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,b->nb-1);
    vec[i] = b->v[row];
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecNestGetSubVec_Nest"
PetscErrorCode PETSCVEC_DLLEXPORT VecNestGetSubVec_Nest(Vec X,PetscInt idxm,Vec *sx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNestGetSubVecs_Private(X,1,&idxm,sx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "VecNestGetSubVec"
PetscErrorCode PETSCVEC_DLLEXPORT VecNestGetSubVec(Vec X,PetscInt idxm,Vec *sx)
{
  PetscErrorCode ierr,(*f)(Vec,PetscInt,Vec*);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)X,"VecNestGetSubVec_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(X,idxm,sx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecNestGetSubVecs_Nest"
PetscErrorCode PETSCVEC_DLLEXPORT VecNestGetSubVecs_Nest(Vec X,PetscInt *N,Vec **sx)
{
  Vec_Nest  *b;

  PetscFunctionBegin;
  b = (Vec_Nest*)X->data;
  *N  = b->nb;
  *sx = b->v;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "VecNestGetSubVecs"
PetscErrorCode PETSCVEC_DLLEXPORT VecNestGetSubVecs(Vec X,PetscInt *N,Vec **sx)
{
  PetscErrorCode ierr,(*f)(Vec,PetscInt*,Vec**);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)X,"VecNestGetSubVecs_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(X,N,sx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

