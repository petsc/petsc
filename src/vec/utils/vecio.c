#ifndef lint
static char vcid[] = "$Id: gcreatev.c,v 1.19 1995/07/05 17:22:58 bsmith Exp $";
#endif

/* 
   This file contains simple binary read/write routines for vectors.
 */

#include "petsc.h"
#include "vec.h"
#include "sysio.h"
#include <sys/errno.h>
#include <unistd.h>
extern int errno;

/* @ 
  VecLoadBinary - Loads a vector that is stored in binary.

  Input Parameters:
. comm - MPI communicator
. fd - file descriptor
. outtype - type of output vector
. ind - index set of local vector indices

  Output Parameter:
. newvec - the newly loaded vector

  Notes:
  Currently, the input file must contain the full global vector, as
  written by the routine VecViewBinary().  Only those vector indices that
  are specified by the index set "ind" are read into the local vector
  segment on a given processor. 
@ */  

int VecLoadBinary(MPI_Comm comm,int fd,VecType outtype,IS ind,Vec *newvec)
{
  int    i, rows, ierr, lsize, gsize, *pind, low, high, iglobal;
  Vec    vec, tempvec;
  Scalar *avec;
  VecType intype = VECSEQ;

  if (intype != VECSEQ) SETERRQ(1,"Only sequential format supported.");

  /* Read vector header */
  ierr = SYRead(fd,(char *)&rows,sizeof(int),0); CHKERRQ(ierr);

  /* Read vector contents */
  if (!ind) {
    if (outtype == VECSEQ) {
      ierr = VecCreateSequential(MPI_COMM_SELF,rows,&vec); CHKERRQ(ierr);
      ierr = VecGetArray(vec,&avec); CHKERRQ(ierr);
      ierr = SYRead(fd,(char *)avec,rows*sizeof(Scalar),0); CHKERRQ(ierr);
    } else SETERRQ(1,"Must specify index set for parallel input.");
  }
  else {
    /* We MUST change this to allow reading partial vector */
    ierr = ISGetLocalSize(ind,&lsize); CHKERRQ(ierr);
    MPI_Allreduce(&lsize,&gsize,1,MPI_INT,MPI_SUM,comm);
    if (gsize != rows) SETERRQ(1,"Incompatible parallel vector length.");
    if (outtype == VECSEQ) {
      ierr = VecCreateSequential(MPI_COMM_SELF,rows,&vec); CHKERRQ(ierr);
      ierr = VecGetArray(vec,&avec); CHKERRQ(ierr);
      ierr = SYRead(fd,(char *)avec,rows*sizeof(Scalar),0); CHKERRQ(ierr);
    } else if (outtype == VECMPI) {
      ierr = VecCreateMPI(comm,lsize,rows,&vec); CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(vec,&low,&high);
      ierr = VecCreateSequential(MPI_COMM_SELF,rows,&tempvec); CHKERRQ(ierr);
      ierr = VecGetArray(tempvec,&avec); CHKERRQ(ierr);
      ierr = SYRead(fd,(char *)avec,rows*sizeof(Scalar),0); CHKERRQ(ierr);
      ierr = ISGetIndices(ind,&pind); CHKERRQ(ierr);
      for (i=0; i<lsize; i++) {
        iglobal = i + low;
        ierr = VecSetValues(vec,1,&iglobal,&avec[pind[i]],INSERTVALUES);
        CHKERRQ(ierr);
      }
      ierr = VecDestroy(tempvec); CHKERRQ(ierr);
    } else {
     SETERRQ(1,"Only VECSEQ and VECMPI output vectors are supported.");
    }
  }
  ierr = VecAssemblyBegin(vec); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec); CHKERRQ(ierr);

/* { Viewer viewer;
    ViewerFileOpenSync("vec.in",comm,&viewer);
    VecView(vec,viewer);
  }  */
  *newvec = vec;
  return 0;
}
/* ------------------------------------------------------------- */
#include "vec/vecimpl.h"

int VecViewBinary(Vec v,int fd)
{
  int    length, ierr;
  Scalar *va;

  if (v->type != VECSEQ) SETERRQ(1,"Only VECSEQ currently supported.");
  ierr = VecGetArray(v,&va); CHKERRQ(ierr);
  ierr = VecGetSize(v,&length); CHKERRQ(ierr);

  /* Write vector header */
  ierr = SYWrite(fd,(char *)&length,sizeof(int),0,0); CHKERRQ(ierr);

  /* Write vector contents */
  ierr = SYWrite(fd,(char *)va,length*sizeof(Scalar),0,0); CHKERRQ(ierr);
  return 0;
}
