
#ifndef lint
static char vcid[] = "$Id: memc.c,v 1.3 1995/10/17 03:15:36 curfman Exp bsmith $";
#endif
/*
    We define the memory operations here. The reason we just don't use 
  the standard memory routines in the PETSc code is that on some machines 
  they are broken.

*/
#include "petsc.h"        /*I  "petsc.h"   I*/
#include <memory.h>
#include "pinclude/petscfix.h"

/*@
   PetscMemcpy - Copies n bytes, beginning at location b to the space
   beginning at location a.

   Input Parameters:
.  a - pointer to copy space
.  b - pointer to initial memory space
.  n - length (in bytes) of space to copy

   Note:
   This routine is analogous to memcpy().

.keywords: Petsc, copy, memory

.seealso: PetscMemcpy()
@*/
void PetscMemcpy(void *a,void *b,int n)
{
  memcpy((char*)(a),(char*)(b),n);
}

/*@
   PetscMemzero - Zeros the specified memory.

   Input Parameters:
.  a - pointer to beginning memory location
.  n - length (in bytes) of memory to initialize

   Note:
   This routine is analogous to memset().

.keywords: Petsc, zero, initialize, memory

.seealso: PetscMemcpy()
@*/
void PetscMemzero(void *a,int n)
{
  memset((char*)(a),0,n);
}


