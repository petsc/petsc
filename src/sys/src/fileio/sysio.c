#ifndef lint
static char vcid[] = "$Id: sysio.c,v 1.9 1996/01/30 04:51:07 bsmith Exp bsmith $";
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
  int  i,j,tmp =0;
  int  *tptr = &tmp;          /* Need to access tmp indirectly to get */
                                /* arround the bug in DEC-ALPHA compilers*/
  char *ptr1,*ptr2 = (char *) &tmp;

  for ( j=0; j<n; j++ ) {
    ptr1 = (char *) (buff + j);
    for (i=0; i<sizeof(int); i++) {
      ptr2[i] = ptr1[sizeof(int)-1-i];
    }
    buff[j] = *tptr;
  }
}
/* --------------------------------------------------------- */
/*
  SYByteSwapShort - Swap bytes in a short
*/
void SYByteSwapShort(short *buff,int n)
{
  int   i,j;
  short tmp;
  short *tptr = &tmp;           /* take care pf bug in DEC-ALPHA g++ */
  char  *ptr1,*ptr2 = (char *) &tmp;
  for ( j=0; j<n; j++ ) {
    ptr1 = (char *) (buff + j);
    for (i=0; i<sizeof(short); i++) {
      ptr2[i] = ptr1[sizeof(int)-1-i];
    }
    buff[j] = *tptr;
  }
}
/* --------------------------------------------------------- */
/*
  SYByteSwapScalar - Swap bytes in a double
  Complex is dealt with as if array of double twice as long.
*/
void SYByteSwapScalar(Scalar *buff,int n)
{
  int    i,j;
  double tmp,*buff1 = (double *) buff;
  double *tptr = &tmp;          /* take care pf bug in DEC-ALPHA g++ */
  char   *ptr1,*ptr2 = (char *) &tmp;
#if defined(PETSC_COMPLEX)
  n *= 2;
#endif
  for ( j=0; j<n; j++ ) {
    ptr1 = (char *) (buff1 + j);
    for (i=0; i<sizeof(double); i++) {
      ptr2[i] = ptr1[sizeof(double)-1-i];
    }
    buff1[j] = *tptr;
  }
}
#endif
/* --------------------------------------------------------- */
/*@C
   SYRead - Reads from a binary file.

   Input Parameters:
.  fd - the file
.  n  - the number of items to read 
.  type - the type of items to read (SYINT or SYSCALAR)

   Output Parameters:
.  p - the buffer

   Notes: 
   SYRead() uses byte swapping to work on all machines.

   SYRead() is not supported in Fortran.

.keywords: binary, input, read

.seealso: SYWrite()
@*/
int SYRead(int fd,void *p,int n,SYIOType type)
{

  int  maxblock, wsize, err, m = n;
  char *pp = (char *) p;
#if defined(HAVE_SWAPPED_BYTES) || defined(HAVE_64BIT_INT)
  void *ptmp = p; 
#endif

  maxblock = 65536;
#if defined(HAVE_64BIT_INT)
  if (type == SYINT){
    /* 
       integers on the Cray T#d are 64 bits so we read the 
       32 bits from the file and then extend them into 
       ints
    */
    m   *= sizeof(short);
    pp   = (char *) PetscMalloc(m); CHKPTRQ(pp);
    ptmp = (void*) pp;
  }
#else
  if (type == SYINT)         m *= sizeof(int);
#endif
  else if (type == SYSCALAR) m *= sizeof(Scalar);
  else if (type == SYSHORT)  m *= sizeof(short);
  else SETERRQ(1,"SYRead:Unknown type");
  
  while (m) {
    wsize = (m < maxblock) ? m : maxblock;
    err = read( fd, pp, wsize );
    if (err < 0 && errno == EINTR) continue;
    if (err == 0 && wsize > 0) return 1;
    if (err < 0) SETERRQ(1,"SYRead:Error reading from file");
    m  -= err;
    pp += err;
  }
#if defined(HAVE_SWAPPED_BYTES)
  if (type == SYINT) SYByteSwapInt((int*)ptmp,n);
  else if (type == SYSCALAR) SYByteSwapScalar((Scalar*)ptmp,n);
  else if (type == SYSHORT) SYByteSwapShort((short*)ptmp,n);
#endif

#if defined(HAVE_64BIT_INT)
  if (type == SYINT){
    /* 
       integers on the Cray T#d are 64 bits so we read the 
       32 bits from the file and then extend them into 
       ints
    */
    int   *p_int = (int *) p,i;
    short *p_short = (short *)ptmp;
    for ( i=0; i<n; i++ ) {
      p_int[i] = (int) p_short[i];
    }
    PetscFree(ptmp);
  }
#endif

  return 0;
}
/* --------------------------------------------------------- */
/*@C
   SYWrite - Writes to a binary file.

   Input Parameters:
.  fd - the file
.  p - the buffer
.  n  - the number of items to read 
.  type - the type of items to read (SYINT or SYSCALAR)

   Notes: 
   SYWrite() uses byte swapping to work on all machines.

   SYWrite() is not supported in Fortran.

.keywords: binary, output, write

.seealso: SYRead()
@*/
int SYWrite(int fd,void *p,int n,SYIOType type,int istemp)
{
  int  err, maxblock, wsize,m = n;
  char *pp = (char *) p;
#if defined(HAVE_SWAPPED_BYTES) || defined(HAVE_64BIT_INT)
  void *ptmp = p; 
#endif

  maxblock = 65536;

#if defined(HAVE_SWAPPED_BYTES)
  if (type == SYINT) SYByteSwapInt((int*)ptmp,n);
  else if (type == SYSCALAR) SYByteSwapScalar((Scalar*)ptmp,n);
  else if (type == SYSHORT) SYByteSwapShort((short*)ptmp,n);
#endif

#if defined(HAVE_64BIT_INT)
  if (type == SYINT){
    /* 
       integers on the Cray T#d are 64 bits so we copy the big
      integers into a short array and write those out.
    */
    int   *p_int = (int *) p,i;
    short *p_short;
    m       *= sizeof(short);
    pp      = (char *) PetscMalloc(m); CHKPTRQ(pp);
    ptmp    = (void*) pp;
    p_short = (short *) pp;

    for ( i=0; i<n; i++ ) {
      p_short[i] = (short) p_int[i];
    }
  }
#else
  if (type == SYINT)         m *= sizeof(int);
#endif
  else if (type == SYSCALAR) m *= sizeof(Scalar);
  else if (type == SYSHORT)  m *= sizeof(short);
  else SETERRQ(1,"SYWrite:Unknown type");

  while (m) {
    wsize = (m < maxblock) ? m : maxblock;
    err = write( fd, pp, wsize );
    if (err < 0 && errno == EINTR) continue;
    if (err != wsize) SETERRQ(n,"SYWrite:Error writing to file.");
    m -= wsize;
    pp += wsize;
  }

#if defined(HAVE_SWAPPED_BYTES)
  if (!istemp) {
    if (type == SYSCALAR) SYByteSwapScalar((Scalar*)ptmp,n);
    else if (type == SYSHORT) SYByteSwapShort((short*)ptmp,n);
    else if (type == SYINT) SYByteSwapInt((int*)ptmp,n);
  }
#endif

#if defined(HAVE_64BIT_INT)
  if (type == SYINT){
    PetscFree(ptmp);
  }
#endif

  return 0;
}
