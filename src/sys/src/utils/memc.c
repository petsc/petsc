
#ifndef lint
static char vcid[] = "$Id: memc.c,v 1.1 1995/09/30 15:45:12 bsmith Exp bsmith $";
#endif
/*
    We define the memory operations here. The reason we just don't use 
  the standard memory routines in the PETSc code is that on some machines 
  they are broken.

*/
#include "petsc.h"        /*I  "petsc.h"   I*/
#include <memory.h>
#include "pinclude/petscfix.h"

void PetscMemcpy(void *a, void *b,int n)
{
  memcpy((char*)(a),(char*)(b),n);
}

void PetscZero(void *a,int n)
{
  memset((char*)(a),0,n);
}


