/*$Id: memc.c,v 1.69 2001/09/07 20:08:33 bsmith Exp $*/
/*
    We define the memory operations here. The reason we just do not use 
  the standard memory routines in the PETSc code is that on some machines 
  they are broken.

*/
#include "petsc.h"        /*I  "petsc.h"   I*/
/*
    On the IBM Rs6000 using the Gnu G++ compiler you may have to include 
  <string.h> instead of <memory.h> 
*/
#include <memory.h>
#if defined(PETSC_HAVE_STRINGS_H)
#include <strings.h>
#endif
#if defined(PETSC_HAVE_STRING_H)
#include <string.h>
#endif
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "petscfix.h"
#include "petscbt.h"
#if defined(PETSC_PREFER_DCOPY_FOR_MEMCPY)
#include "petscblaslapack.h"
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscMemcpy"
/*@C
   PetscMemcpy - Copies n bytes, beginning at location b, to the space
   beginning at location a. The two memory regions CANNOT overlap, use
   PetscMemmove() in that case.

   Not Collective

   Input Parameters:
+  b - pointer to initial memory space
-  n - length (in bytes) of space to copy

   Output Parameter:
.  a - pointer to copy space

   Level: intermediate

   Compile Option:
    PETSC_PREFER_DCOPY_FOR_MEMCPY will cause the BLAS dcopy() routine to be used 
   for memory copies on double precision values.

   Note:
   This routine is analogous to memcpy().

  Concepts: memory^copying
  Concepts: copying^memory
  
.seealso: PetscMemmove()

@*/
int PetscMemcpy(void *a,const void *b,int n)
{
  unsigned long al = (unsigned long) a,bl = (unsigned long) b;
  unsigned long nl = (unsigned long) n;

  PetscFunctionBegin;
  if (a != b) {
#if !defined(PETSC_HAVE_CRAY90_POINTER)
    if ((al > bl && (al - bl) < nl) || (bl - al) < nl) {
      SETERRQ(PETSC_ERR_ARG_INCOMP,"Memory regions overlap: either use PetscMemmov()\n\
              or make sure your copy regions and lengths are correct");
    }
#endif
#if defined(PETSC_PREFER_DCOPY_FOR_MEMCPY)
#  if defined(HAVE_DOUBLE_ALIGN)
    if (!(((long) a) % 8) && !(n % 8)) {
#  else
    if (!(((long) a) % 4) && !(n % 8)) {
#endif
      int one = 1;
      int len = n/sizeof(PetscScalar);
      BLcopy_(&len,(PetscScalar *)a,&one,(PetscScalar *)b,&one);
    } else {
      memcpy((char*)(a),(char*)(b),n);
    }
#else
    memcpy((char*)(a),(char*)(b),n);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscBitMemcpy"
/*@C
   PetscBitMemcpy - Copies an amount of data. This can include bit data.

   Not Collective

   Input Parameters:
+  b - pointer to initial memory space
.  bi - offset of initial memory space (in elementary chunk sizes)
.  bs - length (in elementary chunk sizes) of space to copy
-  dtype - datatype, for example, PETSC_INT, PETSC_DOUBLE, PETSC_LOGICAL

   Output Parameters:
+  a - pointer to result memory space
-  ai - offset of result memory space (in elementary chunk sizes)

   Level: intermediate

   Note:
   This routine is analogous to PetscMemcpy(), except when the data type is 
   PETSC_LOGICAL.

   Concepts: memory^comparing
   Concepts: comparing^memory

.seealso: PetscMemmove(), PetscMemcpy()

@*/
int PetscBitMemcpy(void *a,int ai,const void *b,int bi,int bs,PetscDataType dtype)
{
  char *aa = (char *)a,*bb = (char *)b;
  int  dsize,ierr;

  PetscFunctionBegin;
  if (dtype != PETSC_LOGICAL) {
    ierr = PetscDataTypeGetSize(dtype,&dsize);CHKERRQ(ierr);
    ierr = PetscMemcpy(aa+ai*dsize,bb+bi*dsize,bs*dsize);CHKERRQ(ierr);
  } else {
    PetscBT at = (PetscBT) a,bt = (PetscBT) b;
    int i;
    for (i=0; i<bs; i++) {
      if (PetscBTLookup(bt,bi+i)) PetscBTSet(at,ai+i);
      else                        PetscBTClear(at,ai+i);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMemzero"
/*@C
   PetscMemzero - Zeros the specified memory.

   Not Collective

   Input Parameters:
+  a - pointer to beginning memory location
-  n - length (in bytes) of memory to initialize

   Level: intermediate

   Compile Option:
   PETSC_PREFER_BZERO - on certain machines (the IBM RS6000) the bzero() routine happens
  to be faster than the memset() routine. This flag causes the bzero() routine to be used.

   Concepts: memory^zeroing
   Concepts: zeroing^memory

.seealso: PetscMemcpy()
@*/
int PetscMemzero(void *a,int n)
{
  PetscFunctionBegin;
  if (n < 0) SETERRQ(1,"Memory length must be >= 0");
  if (n > 0) {
#if defined(PETSC_PREFER_BZERO)
    bzero((char *)a,n);
#else
    memset((char*)a,0,n);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMemcmp"
/*@C
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
int PetscMemcmp(const void *str1,const void *str2,int len,PetscTruth *e)
{
  int r;

  PetscFunctionBegin;
  r = memcmp((char *)str1,(char *)str2,len);
  if (!r) *e = PETSC_TRUE;
  else    *e = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMemmove"
/*@C
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

   Contributed by: Matthew Knepley

   Concepts: memory^copying with overlap
   Concepts: copying^memory with overlap

.seealso: PetscMemcpy()
@*/
int PetscMemmove(void *a,void *b,int n)
{
  PetscFunctionBegin;
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




