#include "petscvec.h"
#include "petscmat.h"
#include "taosubset.h"

#undef __FUNCT__
#define __FUNCT__ "TaoCreateSubset"
PetscErrorCode TAOSOLVER_DLLEXPORT TaoCreateSubset(TaoSolver tao, PetscInt subsettype, IndexSet *is) 
{
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);
    
    
    PetscFunctionReturn(0);
}

