#include <petscsys.h>
#include <../src/sys/classes/viewer/impls/socket/socket.h>


/*
   TAKEN from src/sys/fileio/sysio.c The swap byte routines are
  included here because the MATLAB programs that use this do NOT
  link to the PETSc libraries.
*/
#include <errno.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif

/*
  SYByteSwapInt - Swap bytes in an integer
*/
void SYByteSwapInt(int *buff,int n)
{
  int  i,j,tmp;
  char *ptr1,*ptr2 = (char*)&tmp;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<(int)sizeof(int); i++) ptr2[i] = ptr1[sizeof(int)-1-i];
    buff[j] = tmp;
  }
}
/*
  SYByteSwapShort - Swap bytes in a short
*/
void SYByteSwapShort(short *buff,int n)
{
  int   i,j;
  short tmp;
  char  *ptr1,*ptr2 = (char*)&tmp;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<(int)sizeof(short); i++) ptr2[i] = ptr1[sizeof(int)-1-i];
    buff[j] = tmp;
  }
}
/*
  SYByteSwapScalar - Swap bytes in a double
  Complex is dealt with as if array of double twice as long.
*/
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
    for (i=0; i<(int)sizeof(double); i++) ptr2[i] = ptr1[sizeof(double)-1-i];
    buff1[j] = tmp;
  }
}

#define PETSC_MEX_ERROR(a) {fprintf(stdout,"sread: %s \n",a); return PETSC_ERR_SYS;}

/*
    PetscBinaryRead - Reads from a socket, called from MATLAB

  Input Parameters:
.   fd - the file
.   n  - the number of items to read
.   type - the type of items to read (PETSC_INT or PETSC_SCALAR)

  Output Parameters:
.   p - the buffer

  Notes:
    does byte swapping to work on all machines.
*/
PetscErrorCode PetscBinaryRead(int fd,void *p,int n,int *dummy, PetscDataType type)
{

  int  maxblock,wsize,err;
  char *pp = (char*)p;
  int  ntmp  = n;
  void *ptmp = p;

  maxblock = 65536;
  if (type == PETSC_INT)         n *= sizeof(int);
  else if (type == PETSC_SCALAR) n *= sizeof(PetscScalar);
  else if (type == PETSC_SHORT)  n *= sizeof(short);
  else if (type == PETSC_CHAR)   n *= sizeof(char);
  else PETSC_MEX_ERROR("PetscBinaryRead: Unknown type");


  while (n) {
    wsize = (n < maxblock) ? n : maxblock;
    err   = read(fd,pp,wsize);
#if !defined(PETSC_MISSING_ERRNO_EINTR)
    if (err < 0 && errno == EINTR) continue;
#endif
    if (!err && wsize > 0) return 1;
    if (err < 0) PETSC_MEX_ERROR("Error reading from socket\n");
    n  -= err;
    pp += err;
  }

  if(!PetscBinaryBigEndian()) {
    if (type == PETSC_INT) SYByteSwapInt((int*)ptmp,ntmp);
    else if (type == PETSC_SCALAR) SYByteSwapScalar((PetscScalar*)ptmp,ntmp);
    else if (type == PETSC_SHORT) SYByteSwapShort((short*)ptmp,ntmp);
  }
  return 0;
}

/*
    PetscBinaryWrite - Writes to a socket, called from MATLAB

  Input Parameters:
.   fd - the file
.   n  - the number of items to read
.   p - the data
.   type - the type of items to read (PETSC_INT or PETSC_SCALAR)


  Notes:
    does byte swapping to work on all machines.
*/
PetscErrorCode PetscBinaryWrite(int fd,const void *p,int n,PetscDataType type)
{

  int  maxblock,wsize,err,retv=0;
  char *pp = (char*)p;
  int  ntmp  = n;
  void *ptmp = (void*)p;

  maxblock = 65536;
  if (type == PETSC_INT)         n *= sizeof(int);
  else if (type == PETSC_SCALAR) n *= sizeof(PetscScalar);
  else if (type == PETSC_SHORT)  n *= sizeof(short);
  else if (type == PETSC_CHAR)   n *= sizeof(char);
  else PETSC_MEX_ERROR("PetscBinaryRead: Unknown type");

  if(!PetscBinaryBigEndian()) {
    /* make sure data is in correct byte ordering before sending  */
    if (type == PETSC_INT) SYByteSwapInt((int*)ptmp,ntmp);
    else if (type == PETSC_SCALAR) SYByteSwapScalar((PetscScalar*)ptmp,ntmp);
    else if (type == PETSC_SHORT) SYByteSwapShort((short*)ptmp,ntmp);
  }

  while (n) {
    wsize = (n < maxblock) ? n : maxblock;
    err   = write(fd,pp,wsize);
#if !defined(PETSC_MISSING_ERRNO_EINTR)
    if (err < 0 && errno == EINTR) continue;
#endif
    if (!err && wsize > 0) { retv = 1; break; };
    if (err < 0) break;
    n  -= err;
    pp += err;
  }

  if(!PetscBinaryBigEndian()) {
    /* swap the data back if we swapped it before sending it */
    if (type == PETSC_INT) SYByteSwapInt((int*)ptmp,ntmp);
    else if (type == PETSC_SCALAR) SYByteSwapScalar((PetscScalar*)ptmp,ntmp);
    else if (type == PETSC_SHORT) SYByteSwapShort((short*)ptmp,ntmp);
  }

  if (err < 0) PETSC_MEX_ERROR("Error writing to socket\n");
  return retv;
}
