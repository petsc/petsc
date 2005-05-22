
#include <stdio.h>
#include "petscsys.h"
#include "src/sys/src/viewer/impls/socket/socket.h"
#include "mex.h"


/*
   TAKEN from src/sys/src/fileio/sysio.c The swap byte routines are 
  included here because the Matlab programs that use this do NOT
  link to the PETSc libraries.
*/
#include <errno.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif

#if !defined(PETSC_WORDS_BIGENDIAN)
/*
  SYByteSwapInt - Swap bytes in an integer
*/
#undef __FUNCT__  
#define __FUNCT__ "SYByteSwapInt"
void SYByteSwapInt(int *buff,int n)
{
  int  i,j,tmp;
  char *ptr1,*ptr2 = (char*)&tmp;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<sizeof(int); i++) {
      ptr2[i] = ptr1[sizeof(int)-1-i];
    }
    buff[j] = tmp;
  }
}
/*
  SYByteSwapShort - Swap bytes in a short
*/
#undef __FUNCT__  
#define __FUNCT__ "SYByteSwapShort"
void SYByteSwapShort(short *buff,int n)
{
  int   i,j;
  short tmp;
  char  *ptr1,*ptr2 = (char*)&tmp;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<sizeof(short); i++) {
      ptr2[i] = ptr1[sizeof(int)-1-i];
    }
    buff[j] = tmp;
  }
}
/*
  SYByteSwapScalar - Swap bytes in a double
  Complex is dealt with as if array of double twice as long.
*/
#undef __FUNCT__  
#define __FUNCT__ "SYByteSwapScalar"
void SYByteSwapScalar(PetscScalar *buff,int n)
{
  int    i,j;
  double tmp,*buff1 = (double*)buff;
  char   *ptr1,*ptr2 = (char*)&tmp;
#if defined(PETSC_USE_COMPLEX)
  n *= 2;
#endif
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff1 + j);
    for (i=0; i<sizeof(double); i++) {
      ptr2[i] = ptr1[sizeof(double)-1-i];
    }
    buff1[j] = tmp;
  }
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscBinaryRead"
/*
    PetscBinaryRead - Reads from a binary file.

  Input Parameters:
.   fd - the file
.   n  - the number of items to read 
.   type - the type of items to read (PETSC_INT or PETSC_SCALAR)

  Output Parameters:
.   p - the buffer

  Notes: does byte swapping to work on all machines.
*/
PetscErrorCode PetscBinaryRead(int fd,void *p,int n,PetscDataType type)
{

  int  maxblock,wsize,err;
  char *pp = (char*)p;
#if !defined(PETSC_WORDS_BIGENDIAN)
  int  ntmp = n; 
  void *ptmp = p; 
#endif

  maxblock = 65536;
  if (type == PETSC_INT)         n *= sizeof(int);
  else if (type == PETSC_SCALAR) n *= sizeof(PetscScalar);
  else if (type == PETSC_SHORT)  n *= sizeof(short);
  else printf("PetscBinaryRead: Unknown type");
  
  while (n) {
    wsize = (n < maxblock) ? n : maxblock;
    err = read(fd,pp,wsize);
#if !defined(PETSC_MISSING_ERRNO_EINTR)
    if (err < 0 && errno == EINTR) continue;
#endif
    if (!err && wsize > 0) return 1;
    if (err < 0) {
      perror("error reading");
      return err;
    }
    n  -= err;
    pp += err;
  }
#if !defined(PETSC_WORDS_BIGENDIAN)
  if (type == PETSC_INT) SYByteSwapInt((int*)ptmp,ntmp);
  else if (type == PETSC_SCALAR) SYByteSwapScalar((PetscScalar*)ptmp,ntmp);
  else if (type == PETSC_SHORT) SYByteSwapShort((short*)ptmp,ntmp);
#endif

  return 0;
}













