#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: memc.c,v 1.37 1997/09/17 18:55:39 balay Exp balay $";
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
#if defined(HAVE_STRINGS_H)
#include <strings.h>
#endif
#if defined(HAVE_STRING_H)
#include <string.h>
#endif
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"

#undef __FUNC__  
#define __FUNC__ "PetscMemcpy"
/*@C
   PetscMemcpy - Copies n bytes, beginning at location b, to the space
   beginning at location a. The two memory regions CANNOT overlap, use
   PetscMemmove() in that case.

   Input Parameters:
.  b - pointer to initial memory space
.  n - length (in bytes) of space to copy

   Output Parameter:
.  a - pointer to copy space

   Note:
   This routine is analogous to memcpy().

.keywords: Petsc, copy, memory

.seealso: PetscMemmove()

@*/
int PetscMemcpy(void *a,void *b,int n)
{
  unsigned long al = (unsigned long) a, bl = (unsigned long) b;
  unsigned long nl = (unsigned long) n;

  if ((al > bl && (al - bl) < nl) || (bl - al) < nl) {
    SETERRQ(1,1,"Memory regions overlap: use PetscMemmov instead");
  }
  memcpy((char*)(a),(char*)(b),n);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscMemzero"
/*@C
   PetscMemzero - Zeros the specified memory.

   Input Parameters:
.  a - pointer to beginning memory location
.  n - length (in bytes) of memory to initialize

.keywords: Petsc, zero, initialize, memory

.seealso: PetscMemcpy()
@*/
int PetscMemzero(void *a,int n)
{
#if defined(PREFER_BZERO)
  bzero((char *)a,n);
#else
  memset((char*)a,0,n);
#endif
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscMemcmp"
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

#undef __FUNC__  
#define __FUNC__ "PetscMemmove"
/*@C
   PetscMemmove - Copies n bytes, beginning at location b, to the space
   beginning at location a. Copying  between regions that overlap will
   take place correctly.

   Input Parameters:
.  b - pointer to initial memory space
.  n - length (in bytes) of space to copy

   Output Parameter:
.  a - pointer to copy space

   Note:
   This routine is analogous to memmove().

   Contributed by: Matthew Knepley

.keywords: Petsc, copy, memory

.seealso: PetscMemcpy()
@*/
int PetscMemmove(void *a,void *b,int n)
{
#if !defined(HAVE_MEMMOVE)
  if (a < b) {
    if (a <= b - n) {
      memcpy(a, b, n);
    } else {
      memcpy(a, b, (int) (b - a));
      PetscMemmove(b, b + (int) (b - a), n - (int) (b - a));
    }
  }  else {
    if (b <= a - n) {
      memcpy(a, b, n);
    } else {
      memcpy(b + n, b + (n - (int) (a - b)), (int) (a - b));
      PetscMemmove(a, b, n - (int) (a - b));
    }
  }
#else
  memmove((char*)(a),(char*)(b),n);
#endif
  return 0;
}




