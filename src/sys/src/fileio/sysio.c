#ifndef lint
static char vcid[] = "$Id: sysio.c,v 1.1 1995/08/17 20:46:21 curfman Exp bsmith $";
#endif

/* 
   This file contains simple binary read/write routines that 
   are based on SYSafeRead and SYSafeWrite in old-style PETSc.

   Byte swapping is not included here, but it will be incorporated soon
   Also, the variants for different architectures (npux, MSDOS, cray) 
   are not included.
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

/* Should expand to be SYSafeRead/SYSafeWrite.  See tools.n/syetem/safewr.c */
int SYRead(int fd,char *p,int n,SYIOType type)
{
/* Note:  type is not yet used. */

  int maxblock, wsize, err;
/*  int ntmp = n; */
/*  char *ptmp = p; */

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
  return 0;
}
/* -------------------------------------------------------------------- */
int SYWrite(int fd,char *p,int n,SYIOType type,int istemp)
{
/* Note:  type and istemp are not yet used. */

  int err, maxblock, wsize;
/*  int  ntmp  = n; */
/*  char *ptmp = p; */

  maxblock = 65536;
  /* Write the data in blocks of 65536 (some systems don't like large writes;
   SunOS is a particular example) */

  while (n) {
    wsize = (n < maxblock) ? n : maxblock;
    err = write( fd, p, wsize );
    if (err < 0 && errno == EINTR) continue;
    if (err != wsize) SETERRQ(n,"Error in writing to file.");
    n -= wsize;
    p += wsize;
  }
  return 0;
}
/* -------------------------------------------------------------------- */
/* based on old PETSc SpiReadBuffer */
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
