#ifndef lint
static char vcid[] = "$Id: vecio.c,v 1.6 1995/09/05 16:16:07 curfman Exp curfman $";
#endif

/* 
   This file contains simple binary input routines for vectors.  The
   analogous output routines are within each vector implementation's 
   VecView (with viewer types BIN_FILE_VIEWER and BIN_FILES_VIEWER).
 */

#include "petsc.h"
#include "vec/vecimpl.h"
#include "sysio.h"
#include "pinclude/pviewer.h"

/*@ 
  VecLoad - Loads a vector that has been stored in binary format
  with VecView().

  Input Parameters:
. comm - MPI communicator
. viewer - binary file viewer, obtained from ViewerFileOpenBinary()
. outtype - type of output vector
. ind - optional index set of local vector indices (or 0 for loading
  the entire vector on each processor)

  Output Parameter:
. newvec - the newly loaded vector

  Notes:
  Currently, the input file must contain the full global vector, as
  written by the routine VecView().  Only those vector indices that
  are specified by the index set "ind" are read into the local vector
  segment on a given processor. 
@*/  
int VecLoad(MPI_Comm comm,Viewer bview,VecType outtype,IS ind,Vec *newvec)
{
  int    i, rows, ierr, lsize, gsize, *pind, low, high, iglobal, type, fd;
  Vec    vec, tempvec;
  Scalar *avec;
  PetscObject vobj = (PetscObject) bview;

  PETSCVALIDHEADERSPECIFIC(vobj,VIEWER_COOKIE);
  if (vobj->type != BIN_FILES_VIEWER && vobj->type != BIN_FILES_VIEWER)
   SETERRQ(1,"VecLoad: Invalid viewer; open viewer with ViewerFileOpenBinary().");
  fd = ViewerFileGetDescriptor_Private(bview);

  /* Read vector header.  Should this really be the full header? */
  ierr = SYRead(fd,(char *)&type,sizeof(int),SYINT); CHKERRQ(ierr);
  if ((VecType)type != VECSEQ)
    SETERRQ(1,"VecLoadBinary: Only VECSEQ input format supported.");
  ierr = SYRead(fd,(char *)&rows,sizeof(int),SYINT); CHKERRQ(ierr);

  /* Read vector contents */
  if (!ind) {
    if (outtype == VECSEQ) {
      ierr = VecCreateSequential(MPI_COMM_SELF,rows,&vec); CHKERRQ(ierr);
      ierr = VecGetArray(vec,&avec); CHKERRQ(ierr);
      ierr = SYRead(fd,(char *)avec,rows*sizeof(Scalar),SYSCALAR); CHKERRQ(ierr);
      ierr = VecRestoreArray(vec,&avec); CHKERRQ(ierr);
      ierr = VecAssemblyBegin(vec); CHKERRQ(ierr);
      ierr = VecAssemblyEnd(vec); CHKERRQ(ierr);
    } else SETERRQ(1,"Must specify index set for parallel input.");
  }
  else {
    /* We should change this to allow reading partial vector */
    ierr = ISGetLocalSize(ind,&lsize); CHKERRQ(ierr);
    MPI_Allreduce(&lsize,&gsize,1,MPI_INT,MPI_SUM,comm);
    if (gsize != rows) SETERRQ(1,"Incompatible parallel vector length.");
    if (outtype == VECSEQ) {
      ierr = VecCreateSequential(MPI_COMM_SELF,rows,&vec); CHKERRQ(ierr);
      ierr = VecGetArray(vec,&avec); CHKERRQ(ierr);
      ierr = SYRead(fd,(char *)avec,rows*sizeof(Scalar),SYSCALAR); CHKERRQ(ierr);
      ierr = VecAssemblyBegin(vec); CHKERRQ(ierr);
      ierr = VecAssemblyEnd(vec); CHKERRQ(ierr);
      ierr = VecRestoreArray(vec,&avec); CHKERRQ(ierr);
    } else if (outtype == VECMPI) {
      ierr = VecCreateMPI(comm,lsize,rows,&vec); CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(vec,&low,&high);
      ierr = VecCreateSequential(MPI_COMM_SELF,rows,&tempvec); CHKERRQ(ierr);
      ierr = VecGetArray(tempvec,&avec); CHKERRQ(ierr);
      ierr = SYRead(fd,(char *)avec,rows*sizeof(Scalar),SYSCALAR); CHKERRQ(ierr);
      ierr = ISGetIndices(ind,&pind); CHKERRQ(ierr);
      for (i=0; i<lsize; i++) {
        iglobal = i + low;
        ierr = VecSetValues(vec,1,&iglobal,&avec[pind[i]],INSERTVALUES);
        CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(tempvec,&avec); CHKERRQ(ierr);
      ierr = VecDestroy(tempvec); CHKERRQ(ierr);
      ierr = ISRestoreIndices(ind,&pind); CHKERRQ(ierr);
      ierr = VecAssemblyBegin(vec); CHKERRQ(ierr);
      ierr = VecAssemblyEnd(vec); CHKERRQ(ierr);
    } else {
     SETERRQ(1,"Only VECSEQ and VECMPI output vectors are supported.");
    }
  }
  *newvec = vec;
  return 0;
}
