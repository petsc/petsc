#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: receive.c,v 1.12 1999/05/12 03:26:02 bsmith Exp bsmith $";
#endif
/*
 
  This is a MATLAB Mex program which waits at a particular 
  portnumber until a matrix arrives, it then returns to 
  matlab with that matrix.

  Usage: A = receive(portnumber);  portnumber obtained with openport();
 
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92

  Since this is called from Matlab it cannot be compiled with C++.
*/

#include <stdio.h>
#include "sys.h"
#include "src/sys/src/viewer/impls/socket/socket.h"
#include "mex.h"
extern int ReceiveSparseMatrix(Matrix **,int);
extern int ReceiveIntDenseMatrix(Matrix **,int);

#define ERROR(a) {fprintf(stderr,"RECEIVE: %s \n",a); return ;}
/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "mexFunction"
void mexFunction(int nlhs, Matrix *plhs[], int nrhs, Matrix *prhs[])
{
 int    type,t;

  /* check output parameters */
  if (nlhs != 1) ERROR("Receive requires one output argument.");

  if (nrhs == 0) ERROR("Receive requires one input argument.");
  t = (int) *mxGetPr(prhs[0]);

  /* get type of matrix */
  if (PetscBinaryRead(t,&type,1,PETSC_INT))   ERROR("reading type"); 

  if (type == DENSEREAL) ReceiveDenseMatrix(plhs,t);
  if (type == DENSEINT) ReceiveDenseIntMatrix(plhs,t);
  if (type == DENSECHARACTER) {
    if (ReceiveDenseMatrix(plhs,t)) return;
    /* mxSetDispMode(plhs[0],1); */
  }
  if (type == SPARSEREAL) ReceiveSparseMatrix(plhs,t); 
  return;
}

/*
   TAKEN from src/sys/src/sysio.c The swap byte routines are 
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
#undef __FUNC__  
#define __FUNC__ "SYByteSwapInt"
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
#undef __FUNC__  
#define __FUNC__ "SYByteSwapShort"
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
#undef __FUNC__  
#define __FUNC__ "SYByteSwapScalar"
void SYByteSwapScalar(Scalar *buff,int n)
{
  int    i,j;
  double tmp,*buff1 = (double *) buff;
  char   *ptr1,*ptr2 = (char *) &tmp;
#if defined(PETSC_USE_COMPLEX)
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

#undef __FUNC__  
#define __FUNC__ "PetscBinaryRead"
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
int PetscBinaryRead(int fd,void *p,int n,PetscDataType type)
{

  int  maxblock, wsize, err;
  char *pp = (char *) p;
#if !defined(PETSC_WORDS_BIGENDIAN)
  int  ntmp = n; 
  void *ptmp = p; 
#endif

  maxblock = 65536;
  if (type == PETSC_INT)         n *= sizeof(int);
  else if (type == PETSC_SCALAR) n *= sizeof(Scalar);
  else if (type == PETSC_SHORT)  n *= sizeof(short);
  else printf("PetscBinaryRead: Unknown type");
  
  while (n) {
    wsize = (n < maxblock) ? n : maxblock;
    err = read( fd, pp, wsize );
    if (err < 0 && errno == EINTR) continue;
    if (err == 0 && wsize > 0) return 1;
    if (err < 0) {
      perror("error reading");
      return err;
    }
    n  -= err;
    pp += err;
  }
#if !defined(PETSC_WORDS_BIGENDIAN)
  if (type == PETSC_INT) SYByteSwapInt((int*)ptmp,ntmp);
  else if (type == PETSC_SCALAR) SYByteSwapScalar((Scalar*)ptmp,ntmp);
  else if (type == PETSC_SHORT) SYByteSwapShort((short*)ptmp,ntmp);
#endif

  return 0;
}













