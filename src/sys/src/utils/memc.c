
#ifndef lint
static char vcid[] = "$Id: memc.c,v 1.21 1997/02/22 02:23:29 bsmith Exp bsmith $";
#endif
/*
    We define the memory operations here. The reason we just don't use 
  the standard memory routines in the PETSc code is that on some machines 
  they are broken.

*/
#include "petsc.h"        /*I  "petsc.h"   I*/
/*
    On the IBM Rs6000 using the Gnu G++ compiler you may have to include 
  <string.h> instead of <memory.h> 
*/
#include <memory.h>
#include "pinclude/petscfix.h"

#undef __FUNC__  
#define __FUNC__ "PetscMemcpy" /* ADIC Ignore */
/*@C
   PetscMemcpy - Copies n bytes, beginning at location b, to the space
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

#undef __FUNC__  
#define __FUNC__ "PetscMemzero" /* ADIC Ignore */
/*@C
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

#undef __FUNC__  
#define __FUNC__ "PetscMemcmp" /* ADIC Ignore */
/*@C
   PetscMemcmp - Compares two byte streams in memory.

   Input Parameters:
.  str1 - Pointer to the first byte stream
.  str2 - Pointer to the second byte stream
.  len  - The length of the byte stream
         (both str1 and str2 are assumed to be of length 'len')

   Output Parameters:
.  returns integer less than, equal to, or 
   greater than 0, according to whether str11 is 
   less than, equal to, or greater than str2.

   Note: 
   This routine is anologous to memcmp()
@*/
int PetscMemcmp(void * str1, void *str2, int len)
{
  return memcmp((char *)str1, (char *)str2, len);
}




