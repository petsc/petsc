#ifndef lint
static char vcid[] = "$Id: sysio.c,v 1.2 1995/08/27 13:44:24 bsmith Exp bsmith $";
#endif

/* 
   This file contains simple binary read/write routines that 
   are based on SYSafeRead and SYSafeWrite in old-style PETSc.

 */

#include "petsc.h"
#include "sysio.h"
#include <sys/errno.h>
#include <unistd.h>
/*
   Cray T3D cannot find errno!
*/
#if defined(PARCH_t3d)
int errno = 0;
#else
extern int errno;
#endif

#if defined(HAVE_SWAPPED_BYTES)
/*
  SYByteSwapInt - Swap bytes in an integer

  Input Parameters:
. buff - buffer
. n    - number of integers
*/
void SYByteSwapInt(int *buff,int n)
{
  int  i,j,tmp;
  char *ptr1,*ptr2 = (char *) &tmp;
  for ( j=0; j<n; j++ ) {
    ptr1 = (char *) (&buff[j]);
    for (i=0; i<sizeof(int); i++) {
      ptr2[i] = ptr1[sizeof(int)-1-i];
    }
    buff[j] = tmp;
  }
}
/*
  SYByteSwapShort - Swap bytes in a short

  Input Parameters:
. buff - buffer
. n    - number of shorts
*/
void SYByteSwapShort(short *buff,int n)
{
  int  i,j;
  short tmp;
  char *ptr1,*ptr2 = (char *) &tmp;
  for ( j=0; j<n; j++ ) {
    ptr1 = (char *) (&buff[j]);
    for (i=0; i<sizeof(short); i++) {
      ptr2[i] = ptr1[sizeof(int)-1-i];
    }
    buff[j] = tmp;
  }
}
/*
  SYByteSwapDouble - Swap bytes in a double

  Input Parameters:
. buff - buffer
. n    - number of doubles
*/
void SYByteSwapScalar(Scalar *buff,int n)
{
  int    i,j;
  double tmp,*ptr3, *buff1 = (double *) buff;
  char   *ptr1,*ptr2 = (char *) &tmp;
#if defined(PETSC_COMPLEX)
  n *= 2;
#endif
  for ( j=0; j<n; j++ ) {
    ptr3 = &buff1[j];
    ptr1 = (char *) ptr3;
    for (i=0; i<sizeof(double); i++) {
      ptr2[i] = ptr1[sizeof(double)-1-i];
    }
    buff[j] = tmp;
  }
}
#endif

int SYRead(int fd,char *p,int n,SYIOType type)
{

  int  maxblock, wsize, err;
#if defined(HAVE_SWAPPED_BYTES)
  int  ntmp = n; 
  char *ptmp = p; 
#endif

  maxblock = 65536;
  while (n) {
    wsize = (n < maxblock) ? n : maxblock;
    err = read( fd, p, wsize );
    if (err < 0 && errno == EINTR) continue;
    if (err == 0 && wsize > 0) return 1;
    if (err < 0) SETERRQ(1,"Error in reading from file");
    n -= err;
    p += err;
  }
#if defined(HAVE_SWAPPED_BYTES)
  if (type == SYINT) SYByteSwapInt((int*)ptmp,ntmp/sizeof(int));
  else if (type == SYSCALAR) SYByteSwapScalar((Scalar*)ptmp,ntmp/sizeof(double));
#endif

  return 0;
}
/* -------------------------------------------------------------------- */
int SYWrite(int fd,char *p,int n,SYIOType type,int istemp)
{
  int err, maxblock, wsize;
#if defined(HAVE_SWAPPED_BYTES)
  int  ntmp  = n; 
  char *ptmp = p; 
#endif

  maxblock = 65536;
  /* Write the data in blocks of 65536 (some systems don't like large writes;*/

#if defined(HAVE_SWAPPED_BYTES)
  if (type == SYINT) SYByteSwapInt((int*)ptmp,ntmp/sizeof(int));
  else if (type == SYSCALAR) SYByteSwapScalar((Scalar*)ptmp,ntmp/sizeof(double));
#endif

  while (n) {
    wsize = (n < maxblock) ? n : maxblock;
    err = write( fd, p, wsize );
    if (err < 0 && errno == EINTR) continue;
    if (err != wsize) SETERRQ(n,"Error in writing to file.");
    n -= wsize;
    p += wsize;
  }

#if defined(HAVE_SWAPPED_BYTES)
  if (!istemp) {
    if (type == SYINT) SYByteSwapInt((int*)ptmp,ntmp/sizeof(int));
    else if (type == SYSCALAR) SYByteSwapScalar((Scalar*)ptmp,ntmp/sizeof(double));
  }
#endif

  return 0;
}
/* -------------------------------------------------------------------- */
/* 
   SYReadBuffer - Reads data from a file fd at:

   startloc + (nr+1) * sizeof(int) + (so-1)*sizeof(int)        into ja
   startloc + (nr+1 + nz)* sizeof(int) + (so-1)*sizeof(double) into a

   For a first pass, we just seek and read.  Eventually, we want to buffer 
   and reuse (should make buffers lie on 4k boundaries relative to 0, not
   to startloc)

   An fd = -1 clears the existing buffers 

   Returns 1 on failure, 0 on success OR fd == -1
 */
int SYReadBuffer(int fd,long startloc,int n,char *p,SYIOType type)
{
  if (fd < 0) return 0;
  lseek( fd, startloc, SEEK_SET );
  if ((SYRead(fd,p,n,type))) SETERRQ(1,"SYReadBuffer: Failure on SYRead.");
  return 0;
}
