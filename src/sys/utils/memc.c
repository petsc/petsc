
/*
    We define the memory operations here. The reason we just do not use
  the standard memory routines in the PETSc code is that on some machines
  they are broken.

*/
#include <petscsys.h>        /*I  "petscsys.h"   I*/
#include <petscbt.h>
#include <../src/sys/utils/ftn-kernels/fcopy.h>

#undef __FUNCT__
#define __FUNCT__ "PetscMemcmp"
/*@
   PetscMemcmp - Compares two byte streams in memory.

   Not Collective

   Input Parameters:
+  str1 - Pointer to the first byte stream
.  str2 - Pointer to the second byte stream
-  len  - The length of the byte stream
         (both str1 and str2 are assumed to be of length len)

   Output Parameters:
.   e - PETSC_TRUE if equal else PETSC_FALSE.

   Level: intermediate

   Note:
   This routine is anologous to memcmp()
@*/
PetscErrorCode  PetscMemcmp(const void *str1,const void *str2,size_t len,PetscBool  *e)
{
  int r;

  PetscFunctionBegin;
  if (len > 0 && !str1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Trying to compare at a null pointer");
  if (len > 0 && !str2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Trying to compare at a null pointer");
  r = memcmp((char *)str1,(char *)str2,len);
  if (!r) *e = PETSC_TRUE;
  else    *e = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscMemmove"
/*@
   PetscMemmove - Copies n bytes, beginning at location b, to the space
   beginning at location a. Copying  between regions that overlap will
   take place correctly.

   Not Collective

   Input Parameters:
+  b - pointer to initial memory space
-  n - length (in bytes) of space to copy

   Output Parameter:
.  a - pointer to copy space

   Level: intermediate

   Note:
   This routine is analogous to memmove().

   Since b can overlap with a, b cannot be declared as const

   Concepts: memory^copying with overlap
   Concepts: copying^memory with overlap

.seealso: PetscMemcpy()
@*/
PetscErrorCode  PetscMemmove(void *a,void *b,size_t n)
{
  PetscFunctionBegin;
  if (n > 0 && !a) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Trying to copy to null pointer");
  if (n > 0 && !b) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Trying to copy from a null pointer");
#if !defined(PETSC_HAVE_MEMMOVE)
  if (a < b) {
    if (a <= b - n) {
      memcpy(a,b,n);
    } else {
      memcpy(a,b,(int)(b - a));
      PetscMemmove(b,b + (int)(b - a),n - (int)(b - a));
    }
  }  else {
    if (b <= a - n) {
      memcpy(a,b,n);
    } else {
      memcpy(b + n,b + (n - (int)(a - b)),(int)(a - b));
      PetscMemmove(a,b,n - (int)(a - b));
    }
  }
#else
  memmove((char*)(a),(char*)(b),n);
#endif
  PetscFunctionReturn(0);
}




