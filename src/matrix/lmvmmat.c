#include "lmvmmat.h"   /*I "lmvmmat.h" */

#undef __FUNCT__
#define __FUNCT__ "MatCreateLMVM"
/*@C
  MatCreateLMVM - Creates a limited memory matrix for lmvm algorithms.

  Collective on v

  Input Parameters:
  . v - PETSc Vec of arbitrary type

  Output Parameters:
  . M - New matrix

  Level: developer

@*/
int MatCreateLMVM(Vec v)
{
    MPI_Comm comm = v->comm;
    MatLMVMCtx ctx;
    PetscErrorCode info;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(v,VEC_COOKIE,1);
    

    
    
    
}

  
  
