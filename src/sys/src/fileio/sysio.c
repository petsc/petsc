#ifndef lint
static char vcid[] = "$Id: sysio.c,v 1.4 1995/09/30 19:27:41 bsmith Exp bsmith $";
#endif

/* 
   This file contains simple binary read/write routines.
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
*/
void SYByteSwapInt(int *buff,int n)
{
  int  i,j,tmp;
  char *ptr1,*ptr2 = (char *) &tmp;
  for ( j=0; j<n; j++ ) {
    ptr1 = (char *) (buff + j);
    for (i=0; i<sizeof(int); i++) {
      ptr2[i] = ptr1[sizeof(int)-1-i];
    }
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
  char  *ptr1,*ptr2 = (char *) &tmp;
  for ( j=0; j<n; j++ ) {
    ptr1 = (char *) (buff + j);
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
void SYByteSwapScalar(Scalar *buff,int n)
{
  int    i,j;
  double tmp,*buff1 = (double *) buff;
  char   *ptr1,*ptr2 = (char *) &tmp;
#if defined(PETSC_COMPLEX)
  n *= 2;
#endif
  for ( j=0; j<n; j++ ) {
    ptr1 = (char *) (buff1 + j);
    for (i=0; i<sizeof(double); i++) {
      ptr2[i] = ptr1[sizeof(double)-1-i];
    }
    buff1[j] = tmp;
  }
}
#endif

/*@
    SYRead - Reads from a binary file.

  Input Parameters:
.   fd - the file
.   n  - the number of items to read 
.   type - the type of items to read (SYINT or SYSCALAR)

  Output Parameters:
.   p - the buffer

  Notes: does byte swapping to work on all machines.
@*/
int SYRead(int fd,void *p,int n,SYIOType type)
{

  int  maxblock, wsize, err;
  char *pp = (char *) p;
#if defined(HAVE_SWAPPED_BYTES)
  int  ntmp = n; 
  void *ptmp = p; 
#endif

  maxblock = 65536;
  if (type == SYINT)         n *= sizeof(int);
  else if (type == SYSCALAR) n *= sizeof(Scalar);
  else if (type == SYSHORT)  n *= sizeof(short);
  else SETERRQ(1,"SYRead:Unknown type");
  
  while (n) {
    wsize = (n < maxblock) ? n : maxblock;
    err = read( fd, pp, wsize );
    if (err < 0 && errno == EINTR) continue;
    if (err == 0 && wsize > 0) return 1;
    if (err < 0) SETERRQ(1,"SYRead:Error reading from file");
    n  -= err;
    pp += err;
  }
#if defined(HAVE_SWAPPED_BYTES)
  if (type == SYINT) SYByteSwapInt((int*)ptmp,ntmp);
  else if (type == SYSCALAR) SYByteSwapScalar((Scalar*)ptmp,ntmp);
  else if (type == SYSHORT) SYByteSwapShort((short*)ptmp,ntmp);
#endif

  return 0;
}
/* -------------------------------------------------------------------- */
/*@
    SYWrite - Writes to a binary file.

  Input Parameters:
.   fd - the file
.   p - the buffer
.   n  - the number of items to read 
.   type - the type of items to read (SYINT or SYSCALAR)

  Notes: does byte swapping to work on all machines.
@*/
int SYWrite(int fd,void *p,int n,SYIOType type,int istemp)
{
  int  err, maxblock, wsize;
  char *pp = (char *) p;
#if defined(HAVE_SWAPPED_BYTES)
  int  ntmp  = n; 
  void *ptmp = p; 
#endif

  maxblock = 65536;
  /* Write the data in blocks of 65536 (some systems don't like large writes;*/

#if defined(HAVE_SWAPPED_BYTES)
  if (type == SYINT) SYByteSwapInt((int*)ptmp,ntmp);
  else if (type == SYSCALAR) SYByteSwapScalar((Scalar*)ptmp,ntmp);
  else if (type == SYSHORT) SYByteSwapShort((short*)ptmp,ntmp);
#endif

  if (type == SYINT)         n *= sizeof(int);
  else if (type == SYSCALAR) n *= sizeof(Scalar);
  else if (type == SYSHORT)  n *= sizeof(short);
  else SETERRQ(1,"SYWrite:Unknown type");

  while (n) {
    wsize = (n < maxblock) ? n : maxblock;
    err = write( fd, pp, wsize );
    if (err < 0 && errno == EINTR) continue;
    if (err != wsize) SETERRQ(n,"SYWrite:Error writing to file.");
    n  -= wsize;
    pp += wsize;
  }

#if defined(HAVE_SWAPPED_BYTES)
  if (!istemp) {
    if (type == SYINT) SYByteSwapInt((int*)ptmp,ntmp);
    else if (type == SYSCALAR) SYByteSwapScalar((Scalar*)ptmp,ntmp);
    else if (type == SYSHORT) SYByteSwapShort((short*)ptmp,ntmp);
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
int SYReadBuffer(int fd,long startloc,int n,void *p,SYIOType type)
{
  int ierr;
  if (fd < 0) return 0;
  lseek( fd, startloc, SEEK_SET );
  ierr = SYRead(fd,p,n,type); CHKERRQ(ierr);
  return 0;
}
