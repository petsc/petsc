#ifndef lint
static char vcid[] = "$Id: vecio.c,v 1.14 1995/09/30 19:26:26 bsmith Exp bsmith $";
#endif

/* 
   This file contains simple binary input routines for vectors.  The
   analogous output routines are within each vector implementation's 
   VecView (with viewer types BINARY_FILE_VIEWER)
 */

#include "petsc.h"
#include "vec/impls/mpi/pvecimpl.h"
#include "sysio.h"
#include "pinclude/pviewer.h"

/*@C 
  VecLoad - Loads a vector that has been stored in binary format
  with VecView().

  Input Parameters:
. comm - MPI communicator
. viewer - binary file viewer, obtained from ViewerFileOpenBinary()

  Output Parameter:
. newvec - the newly loaded vector

  Notes:
  Currently, the input file must contain the full global vector, as
  written by the routine VecView().  Only those vector indices that
  are specified by the index set "ind" are read into the local vector
  segment on a given processor. 
@*/  
int VecLoad(Viewer bview,Vec *newvec)
{
  int         i, rows, ierr, type, fd;
  Vec         vec;
  Vec_MPI     *v;
  Scalar      *avec;
  int         mytid,numtid,n;
  PetscObject obj = (PetscObject) bview;
  MPI_Comm    comm;
  MPI_Request *requests;
  MPI_Status  status,*statuses;

  PETSCVALIDHEADERSPECIFIC(obj,VIEWER_COOKIE);
  if (obj->type != BINARY_FILE_VIEWER)
   SETERRQ(1,"VecLoad:Invalid viewer;open viewer with ViewerFileOpenBinary()");
  PLogEventBegin(VEC_Load,bview,0,0,0);
  ierr = ViewerFileGetDescriptor_Private(bview,&fd); CHKERRQ(ierr);
  
  PetscObjectGetComm(obj,&comm);
  MPI_Comm_rank(comm,&mytid);
  MPI_Comm_size(comm,&numtid);

  if (!mytid) {
    /* Read vector header. */
    ierr = SYRead(fd,&type,1,SYINT); CHKERRQ(ierr);
    if ((VecType)type != VEC_COOKIE) SETERRQ(1,"VecLoadBinary: Non-vector object");
    ierr = SYRead(fd,&rows,1,SYINT); CHKERRQ(ierr);
    MPI_Bcast(&rows,1,MPI_INT,0,comm);
    ierr = VecCreate(comm,rows,&vec); CHKERRQ(ierr);
    v = (Vec_MPI*) vec->data;
    ierr = VecGetArray(vec,&avec); CHKERRQ(ierr);
    ierr = SYRead(fd,avec,v->n,SYSCALAR);CHKERRQ(ierr);
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
        ierr = SYRead(fd,avec,n,SYSCALAR);
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
  *newvec = vec;
  PLogEventEnd(VEC_Load,bview,0,0,0);
  return 0;
}
