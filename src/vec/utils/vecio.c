#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: vecio.c,v 1.51 1999/03/11 16:16:10 bsmith Exp bsmith $";
#endif

/* 
   This file contains simple binary input routines for vectors.  The
   analogous output routines are within each vector implementation's 
   VecView (with viewer types BINARY_VIEWER)
 */

#include "petsc.h"
#include "sys.h"
#include "vec.h"         /*I  "ve.h"  I*/

#undef __FUNC__  
#define __FUNC__ "VecLoad"
/*@C 
  VecLoad - Loads a vector that has been stored in binary format
  with VecView().

  Collective on Viewer 

  Input Parameters:
. viewer - binary file viewer, obtained from ViewerBinaryOpen()

  Output Parameter:
. newvec - the newly loaded vector

   Level: intermediate

  Notes:
  The input file must contain the full global vector, as
  written by the routine VecView().

  Notes for advanced users:
  Most users should not need to know the details of the binary storage
  format, since VecLoad() and VecView() completely hide these details.
  But for anyone who's interested, the standard binary matrix storage
  format is
.vb
     int    VEC_COOKIE
     int    number of rows
     Scalar *values of all nonzeros
.ve

   Note for Cray users, the int's stored in the binary file are 32 bit
integers; not 64 as they are represented in the memory, so if you
write your own routines to read/write these binary files from the Cray
you need to adjust the integer sizes that you read in, see
PetscReadBinary() and PetscWriteBinary() to see how this may be
done.

   In addition, PETSc automatically does the byte swapping for
machines that store the bytes reversed, e.g.  DEC alpha, freebsd,
linux, nt and the paragon; thus if you write your own binary
read/write routines you have to swap the bytes; see PetscReadBinary()
and PetscWriteBinary() to see how this may be done.

.keywords: vector, load, binary, input

.seealso: ViewerBinaryOpen(), VecView(), MatLoad(), VecLoadIntoVector() 
@*/  
int VecLoad(Viewer viewer,Vec *newvec)
{
  int         i, rows, ierr, type, fd,rank,size,n,*range,tag,bs,flag;
  Vec         vec;
  Scalar      *avec;
  MPI_Comm    comm;
  MPI_Request request;
  MPI_Status  status;
  ViewerType  vtype;
  Map         map;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (PetscStrcmp(vtype,BINARY_VIEWER)) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must be binary viewer");
  PLogEventBegin(VEC_Load,viewer,0,0,0);
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);

  if (!rank) {
    /* Read vector header. */
    ierr = PetscBinaryRead(fd,&type,1,PETSC_INT); CHKERRQ(ierr);
    if (type != VEC_COOKIE) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Non-vector object");
    ierr = PetscBinaryRead(fd,&rows,1,PETSC_INT); CHKERRQ(ierr);
    ierr = MPI_Bcast(&rows,1,MPI_INT,0,comm);CHKERRQ(ierr);
    ierr = VecCreate(comm,PETSC_DECIDE,rows,&vec); CHKERRQ(ierr);
    ierr = OptionsGetInt(PETSC_NULL,"-vecload_block_size",&bs,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = VecSetBlockSize(vec,bs);CHKERRQ(ierr);
    }
    ierr = VecSetFromOptions(vec); CHKERRQ(ierr);
    ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
    ierr = VecGetArray(vec,&avec); CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,avec,n,PETSC_SCALAR);CHKERRQ(ierr);
    ierr = VecRestoreArray(vec,&avec); CHKERRQ(ierr);

    if (size > 1) {
      /* read in other chuncks and send to other processors */
      /* determine maximum chunck owned by other */
      ierr = VecGetMap(vec,&map);CHKERRQ(ierr);
      ierr = MapGetGlobalRange(map,&range);CHKERRQ(ierr);
      n = 1;
      for ( i=1; i<size; i++ ) {
        n = PetscMax(n,range[i] - range[i-1]);
      }
      avec     = (Scalar *) PetscMalloc( n*sizeof(Scalar) ); CHKPTRQ(avec);
      ierr     = PetscObjectGetNewTag((PetscObject)viewer,&tag);CHKERRQ(ierr);
      for ( i=1; i<size; i++ ) {
        n    = range[i+1] - range[i];
        ierr = PetscBinaryRead(fd,avec,n,PETSC_SCALAR);CHKERRQ(ierr);
        ierr = MPI_Isend(avec,n,MPIU_SCALAR,i,tag,comm,&request);CHKERRQ(ierr);
        ierr = MPI_Wait(&request,&status);CHKERRQ(ierr);
      }
      PetscFree(avec);
    }
  } else {
    ierr = MPI_Bcast(&rows,1,MPI_INT,0,comm);CHKERRQ(ierr);
    ierr = VecCreate(comm,PETSC_DECIDE,rows,&vec); CHKERRQ(ierr);
    ierr = VecSetFromOptions(vec); CHKERRQ(ierr);
    ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr); 
    ierr = PetscObjectGetNewTag((PetscObject)viewer,&tag);CHKERRQ(ierr);
    ierr = VecGetArray(vec,&avec); CHKERRQ(ierr);
    ierr = MPI_Recv(avec,n,MPIU_SCALAR,0,tag,comm,&status);CHKERRQ(ierr);
    ierr = VecRestoreArray(vec,&avec); CHKERRQ(ierr);
  }
  *newvec = vec;
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  PLogEventEnd(VEC_Load,viewer,0,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecLoadIntoVector_Default"
int VecLoadIntoVector_Default(Viewer viewer,Vec vec)
{
  int         i, rows, ierr, type, fd,rank,size,n,*range,tag,bs,flag;
  Scalar      *avec;
  MPI_Comm    comm;
  MPI_Request request;
  MPI_Status  status;
  ViewerType  vtype;
  Map         map;

  PetscFunctionBegin;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (PetscStrcmp(vtype,BINARY_VIEWER)) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must be binary viewer");
  PLogEventBegin(VEC_Load,viewer,vec,0,0);
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);

  if (!rank) {
    /* Read vector header. */
    ierr = PetscBinaryRead(fd,&type,1,PETSC_INT); CHKERRQ(ierr);
    if (type != VEC_COOKIE) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Non-vector object");
    ierr = PetscBinaryRead(fd,&rows,1,PETSC_INT); CHKERRQ(ierr);
    ierr = VecGetSize(vec,&n);CHKERRQ(ierr);
    if (n != rows) SETERRQ(1,1,"Vector in file different length then input vector");
    ierr = MPI_Bcast(&rows,1,MPI_INT,0,comm);CHKERRQ(ierr);

    ierr = OptionsGetInt(PETSC_NULL,"-vecload_block_size",&bs,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = VecSetBlockSize(vec,bs);CHKERRQ(ierr);
    }
    ierr = VecSetFromOptions(vec); CHKERRQ(ierr);
    ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
    ierr = VecGetArray(vec,&avec); CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,avec,n,PETSC_SCALAR);CHKERRQ(ierr);
    ierr = VecRestoreArray(vec,&avec); CHKERRQ(ierr);

    if (size > 1) {
      /* read in other chuncks and send to other processors */
      /* determine maximum chunck owned by other */
      ierr = VecGetMap(vec,&map);CHKERRQ(ierr);
      ierr = MapGetGlobalRange(map,&range);CHKERRQ(ierr);
      n = 1;
      for ( i=1; i<size; i++ ) {
        n = PetscMax(n,range[i] - range[i-1]);
      }
      avec     = (Scalar *) PetscMalloc( n*sizeof(Scalar) ); CHKPTRQ(avec);
      ierr     = PetscObjectGetNewTag((PetscObject)viewer,&tag);CHKERRQ(ierr);
      for ( i=1; i<size; i++ ) {
        n    = range[i+1] - range[i];
        ierr = PetscBinaryRead(fd,avec,n,PETSC_SCALAR);CHKERRQ(ierr);
        ierr = MPI_Isend(avec,n,MPIU_SCALAR,i,tag,comm,&request);CHKERRQ(ierr);
        ierr = MPI_Wait(&request,&status);CHKERRQ(ierr);
      }
      PetscFree(avec);
    }
  } else {
    ierr = MPI_Bcast(&rows,1,MPI_INT,0,comm);CHKERRQ(ierr);
    ierr = VecSetFromOptions(vec); CHKERRQ(ierr);
    ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr); 
    ierr = PetscObjectGetNewTag((PetscObject)viewer,&tag);CHKERRQ(ierr);
    ierr = VecGetArray(vec,&avec); CHKERRQ(ierr);
    ierr = MPI_Recv(avec,n,MPIU_SCALAR,0,tag,comm,&status);CHKERRQ(ierr);
    ierr = VecRestoreArray(vec,&avec); CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  PLogEventEnd(VEC_Load,viewer,vec,0,0);
  PetscFunctionReturn(0);
}

