
#include <stdio.h>
#include <stdlib.h>

#include <petsc.h>
#include <petscvec.h>
#include <private/vecimpl.h>

#include "vecblockimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "PETSc_VecBlock_Check2"
PetscErrorCode PETSc_VecBlock_Check2(Vec x,Vec y)
{
  Vec_Block *bx = (Vec_Block*)x->data;
  Vec_Block *by = (Vec_Block*)y->data;

  PetscFunctionBegin;
  if (!bx->setup_called) SETERRQ(((PetscObject)x)->comm,PETSC_ERR_ARG_WRONG,"Block vector x not setup.");
  if (!by->setup_called) SETERRQ(((PetscObject)x)->comm,PETSC_ERR_ARG_WRONG,"Block vector y not setup.");
  if (bx->nb != by->nb) SETERRQ(((PetscObject)x)->comm,PETSC_ERR_ARG_WRONG,"Block vectors have different numbers of blocks.");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PETSc_VecBlock_Check3"
PetscErrorCode PETSc_VecBlock_Check3(Vec w,Vec x,Vec y)
{
  Vec_Block *bw = (Vec_Block*)w->data;
  Vec_Block *bx = (Vec_Block*)x->data;
  Vec_Block *by = (Vec_Block*)y->data;

  PetscFunctionBegin;
  if (!bw->setup_called) SETERRQ(((PetscObject)w)->comm,PETSC_ERR_ARG_WRONG,"Block vector w not setup.");
  if (!bx->setup_called) SETERRQ(((PetscObject)w)->comm,PETSC_ERR_ARG_WRONG,"Block vector x not setup.");
  if (!by->setup_called) SETERRQ(((PetscObject)w)->comm,PETSC_ERR_ARG_WRONG,"Block vector y not setup.");

  if (bx->nb != by->nb) SETERRQ(((PetscObject)w)->comm,PETSC_ERR_ARG_WRONG,"Block vectors x and y have different numbers of blocks.");
  if (bx->nb != bw->nb) SETERRQ(((PetscObject)w)->comm,PETSC_ERR_ARG_WRONG,"Block vectors x and w have different numbers of blocks.");
  if (bw->nb != by->nb) SETERRQ(((PetscObject)w)->comm,PETSC_ERR_ARG_WRONG,"Block vectors w and y have different numbers of blocks.");
  PetscFunctionReturn(0);
}
