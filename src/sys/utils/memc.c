#define PETSC_DLL

/*
    We define the memory operations here. The reason we just do not use 
  the standard memory routines in the PETSc code is that on some machines 
  they are broken.

*/
#include "petsc.h"        /*I  "petsc.h"   I*/
#include "src/inline/axpy.h"

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
/*@
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
    PETSC_PREFER_COPY_FOR_MEMCPY will cause C code to be used 
                                  for memory copies on double precision values.
    PETSC_PREFER_FORTRAN_FORMEMCPY will cause Fortran code to be used 
                                  for memory copies on double precision values.

   Note:
   This routine is analogous to memcpy().

  Concepts: memory^copying
  Concepts: copying^memory
  
.seealso: PetscMemmove()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscMemcpy(void *a,const void *b,size_t n)
{
  unsigned long al = (unsigned long) a,bl = (unsigned long) b;
  unsigned long nl = (unsigned long) n;

  PetscFunctionBegin;
  if (n > 0 && !b) SETERRQ(PETSC_ERR_ARG_NULL,"Trying to copy from a null pointer");
  if (n > 0 && !a) SETERRQ(PETSC_ERR_ARG_NULL,"Trying to copy to a null pointer");
  if (a != b) {
#if !defined(PETSC_HAVE_CRAY90_POINTER)
    if ((al > bl && (al - bl) < nl) || (bl - al) < nl) {
      SETERRQ3(PETSC_ERR_ARG_INCOMP,"Memory regions overlap: either use PetscMemmov()\n\
              or make sure your copy regions and lengths are correct. \n\
              Length (bytes) %ld first address %ld second address %ld",nl,al,bl);
    }
#endif
#if (defined(PETSC_PREFER_DCOPY_FOR_MEMCPY) || defined(PETSC_PREFER_COPY_FOR_MEMCPY) || defined(PETSC_PREFER_FORTRAN_FORMEMCPY))
   if (!(((long) a) % sizeof(PetscScalar)) && !(n % sizeof(PetscScalar))) {
      size_t len = n/sizeof(PetscScalar);
#if defined(PETSC_PREFER_DCOPY_FOR_MEMCPY)
      PetscBLASInt blen = (PetscBLASInt) len,one = 1;
      BLAScopy_(&blen,(PetscScalar *)b,&one,(PetscScalar *)a,&one);
#elif defined(PETSC_PREFER_FORTRAN_FORMEMCPY)
      fortrancopy_(&len,(PetscScalar*)b,(PetscScalar*)a); 
#else
      size_t      i;
      PetscScalar *x = (PetscScalar*)b, *y = (PetscScalar*)a;
      for (i=0; i<len; i++) y[i] = x[i];
#endif
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
PetscErrorCode PETSC_DLLEXPORT PetscBitMemcpy(void *a,PetscInt ai,const void *b,PetscInt bi,PetscInt bs,PetscDataType dtype)
{
  char           *aa = (char *)a,*bb = (char *)b;
  PetscInt       dsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (bs > 0 && !b) SETERRQ(PETSC_ERR_ARG_NULL,"Trying to copy from a null pointer");
  if (bs > 0 && !a) SETERRQ(PETSC_ERR_ARG_NULL,"Trying to copy to a null pointer");
  if (dtype != PETSC_LOGICAL) {
    ierr = PetscDataTypeGetSize(dtype,&dsize);CHKERRQ(ierr);
    ierr = PetscMemcpy(aa+ai*dsize,bb+bi*dsize,bs*dsize);CHKERRQ(ierr);
  } else {
    PetscBT  at = (PetscBT) a;
    PetscBT  bt = (PetscBT) b;
    PetscInt i;
    for (i=0; i<bs; i++) {
      if (PetscBTLookup(bt,bi+i)) {ierr = PetscBTSet(at,ai+i);CHKERRQ(ierr);}
      else                        {ierr = PetscBTClear(at,ai+i);CHKERRQ(ierr);}
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMemzero"
/*@
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
PetscErrorCode PETSC_DLLEXPORT PetscMemzero(void *a,size_t n)
{
  PetscFunctionBegin;
  if (n > 0) {
    if (!a) SETERRQ(PETSC_ERR_ARG_NULL,"Trying to zero at a null pointer");
#if defined(PETSC_PREFER_ZERO_FOR_MEMZERO)
    if (!(((long) a) % sizeof(PetscScalar)) && !(n % sizeof(PetscScalar))) {
      size_t      i,len = n/sizeof(PetscScalar);
      PetscScalar *x = (PetscScalar*)a;
      for (i=0; i<len; i++) x[i] = 0.0;
    } else {
#elif defined(PETSC_PREFER_FORTRAN_FOR_MEMZERO)
    if (!(((long) a) % sizeof(PetscScalar)) && !(n % sizeof(PetscScalar))) {
      PetscInt len = n/sizeof(PetscScalar);
      fortranzero_(&len,(PetscScalar*)a);
    } else {
#endif
#if defined(PETSC_PREFER_BZERO)
      bzero((char *)a,n);
#else
      memset((char*)a,0,n);
#endif
#if defined(PETSC_PREFER_ZERO_FOR_MEMZERO) || defined(PETSC_PREFER_FORTRAN_FOR_MEMZERO)
    }
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
PetscErrorCode PETSC_DLLEXPORT PetscMemcmp(const void *str1,const void *str2,size_t len,PetscTruth *e)
{
  int r;

  PetscFunctionBegin;
  if (len > 0 && !str1) SETERRQ(PETSC_ERR_ARG_NULL,"Trying to compare at a null pointer");
  if (len > 0 && !str2) SETERRQ(PETSC_ERR_ARG_NULL,"Trying to compare at a null pointer");
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

   Concepts: memory^copying with overlap
   Concepts: copying^memory with overlap

.seealso: PetscMemcpy()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscMemmove(void *a,void *b,size_t n)
{
  PetscFunctionBegin;
  if (n > 0 && !a) SETERRQ(PETSC_ERR_ARG_NULL,"Trying to copy to null pointer");
  if (n > 0 && !b) SETERRQ(PETSC_ERR_ARG_NULL,"Trying to copy from a null pointer");
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




