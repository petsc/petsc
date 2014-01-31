#include <petscvec.h>
#include <petscmat.h>
#include <taosubset.h>

#undef __FUNCT__
#define __FUNCT__ "TaoCreateSubset"
PetscErrorCode TaoCreateSubset(Tao tao, PetscInt subsettype, IndexSet *is)
{
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAO_CLASSID,1);


    PetscFunctionReturn(0);
}

