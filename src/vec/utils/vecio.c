#ifndef lint
static char vcid[] = "$Id: vecio.c,v 1.8 1995/09/06 03:04:14 bsmith Exp bsmith $";
#endif

/* 
   This file contains simple binary input routines for vectors.  The
   analogous output routines are within each vector implementation's 
   VecView (with viewer types BIN_FILE_VIEWER and BIN_FILES_VIEWER).
 */

#include "petsc.h"
#include "vec/impls/mpi/pvecimpl.h"
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
int VecLoad(Viewer bview,IS ind,Vec *newvec)
{
  int         i,rows,ierr,lsize, gsize, *pind, low, high, iglobal, type, fd;
  Vec         vec, tempvec;
  Vec_MPI     *v;
  Scalar      *avec;
  int         mytid,numtid,n;
  PetscObject obj = (PetscObject) bview;
  MPI_Comm    comm;
  MPI_Request *requests;
  MPI_Status  status,*statuses;

  PETSCVALIDHEADERSPECIFIC(obj,VIEWER_COOKIE);
  if (obj->type != BIN_FILE_VIEWER)
   SETERRQ(1,"VecLoad:Invalid viewer;open viewer with ViewerFileOpenBinary()");
  ierr = ViewerFileGetDescriptor_Private(bview,&fd); CHKERRQ(ierr);
  
  PetscObjectGetComm(obj,&comm);
  MPI_Comm_rank(comm,&mytid);
  MPI_Comm_size(comm,&numtid);

  if (!ind) {
    if (!mytid) {
      /* Read vector header. */
      ierr = SYRead(fd,(char *)&type,sizeof(int),SYINT); CHKERRQ(ierr);
      if ((VecType)type != VEC_COOKIE)
           SETERRQ(1,"VecLoadBinary: Trying to read a non-vector object");
      ierr = SYRead(fd,(char *)&rows,sizeof(int),SYINT); CHKERRQ(ierr);
      MPI_Bcast(&rows,1,MPI_INT,0,comm);
      ierr = VecCreate(comm,rows,&vec); CHKERRQ(ierr);
      v = (Vec_MPI*) vec->data;
      ierr = VecGetArray(vec,&avec); CHKERRQ(ierr);
      ierr = SYRead(fd,(char *)avec,v->n*sizeof(Scalar),SYSCALAR);
      CHKERRQ(ierr);
      ierr = VecRestoreArray(vec,&avec); CHKERRQ(ierr);

      if (numtid > 1) {
        /* read in other chuncks and send to other processors */
        /* determine maximum chunck owned by other */
        n = 1;
        for ( i=1; i<numtid; i++ ) {
          n = PETSCMAX(n,v->ownership[i] - v->ownership[i-1]);
        }
        avec = (Scalar *) PETSCMALLOC( n*sizeof(Scalar) ); CHKPTRQ(avec);
        requests = (MPI_Request *) PETSCMALLOC((numtid-1)*sizeof(MPI_Request)); 
        CHKPTRQ(requests);
        statuses = (MPI_Status *) PETSCMALLOC((numtid-1)*sizeof(MPI_Status)); 
        CHKPTRQ(statuses);
        for ( i=1; i<numtid; i++ ) {
          n = v->ownership[i+1]-v->ownership[i];
          ierr = SYRead(fd,(char *)avec,n*sizeof(Scalar),SYSCALAR);
          CHKERRQ(ierr);
          MPI_Isend(avec,n,MPIU_SCALAR,i,vec->tag,vec->comm,requests+i-1);
        }
        MPI_Waitall(numtid-1,requests,statuses);
        PETSCFREE(avec); PETSCFREE(requests); PETSCFREE(statuses);
      }
    } else {
      MPI_Bcast(&rows,1,MPI_INT,0,comm);
      ierr = VecCreate(comm,rows,&vec); CHKERRQ(ierr);
      VecGetLocalSize(vec,&n);CHKERRQ(ierr); 
      VecGetArray(vec,&avec); CHKERRQ(ierr);
      MPI_Recv(avec,n,MPIU_SCALAR,0,vec->tag,vec->comm,&status);
      ierr = VecRestoreArray(vec,&avec); CHKERRQ(ierr);
    }
  }
  else {
    ierr = ISGetLocalSize(ind,&lsize); CHKERRQ(ierr);
    MPI_Allreduce(&lsize,&gsize,1,MPI_INT,MPI_SUM,comm);
    if (gsize != rows) SETERRQ(1,"Incompatible parallel vector length.");
    SETERRQ(1,"No support yet for permuted vectors");
  }
  *newvec = vec;
  return 0;
}
