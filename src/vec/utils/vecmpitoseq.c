#include "src/vec/vecimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "VecConvertMPIToSeqAll"
/*@C
  VecConvertMPIToSeqAll - make available all the values of
  an MPIVEC on all processors as a SEQVEC

  Collective

  Input Parameter: 
.  vin  - input MPIVEC

  Output Parameter:
.  vout - output SEQVEC

  Level: intermediate

  Notes: Each processor will have all the values
.seealso VecConvertMPIToMPIZero
@*/
int VecConvertMPIToSeqAll(Vec vin,Vec *vout)
{

  int        ierr,N;
  IS         is;
  VecScatter ctx;

  PetscFunctionBegin;

  /* Check if vin is of type VECMPI ????????? */
  PetscValidHeaderSpecific(vin,VEC_COOKIE);
  PetscValidType(vin);

  /* Create seq vec on each proc, with the same size of the original mpi vec */
  ierr = VecGetSize(vin,&N);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,N,vout);CHKERRQ(ierr);
  /* Create the VecScatter ctx with the communication info */
  ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&is);CHKERRQ(ierr);
  ierr = VecScatterCreate(vin,is,*vout,is,&ctx);CHKERRQ(ierr);
  /* Now trasfer the values into the seq vector */
  ierr = VecScatterBegin(vin,*vout,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(vin,*vout,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRQ(ierr);

  ierr = ISDestroy(is);CHKERRQ(ierr);
  ierr = VecScatterDestroy(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecConvertMPIToMPIZero"
/*@C
  VecConvertMPIToMPIZero - make available all the values of
  an MPIVEC on processor zero as an MPIVEC

  Collective on Vec

  Input Parameter: 
.  vin  - input MPIVEC

  Output Parameter:
.  vout - output MPIVEC, with values only on processor zero.

  Level: intermediate

.seealso VecConvertMPIToSeqAll
@*/
int VecConvertMPIToMPIZero(Vec vin,Vec *vout)
{

  int        ierr,rank,N;
  IS         is;
  VecScatter ctx;

  PetscFunctionBegin;

  /* Check if vin is of type VECMPI ????????? */
  PetscValidHeaderSpecific(vin,VEC_COOKIE);
  PetscValidType(vin);

  /* Create seq vec on each proc, with the same size of the original mpi vec */
  ierr = VecGetSize(vin,&N);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(vin->comm,&rank);CHKERRQ(ierr);

  if (!rank) {
    ierr = VecCreateMPI(vin->comm,N,N,vout);
  } else {
    ierr = VecCreateMPI(vin->comm,0,N,vout);
  }

  /* Create the VecScatter ctx with the communication info */
  ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&is);CHKERRQ(ierr);
  ierr = VecScatterCreate(vin,is,*vout,is,&ctx);CHKERRQ(ierr);
  /* Now trasfer the values into the new layout */
  ierr = VecScatterBegin(vin,*vout,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(vin,*vout,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRQ(ierr);
  
  ierr = ISDestroy(is);CHKERRQ(ierr);
  ierr = VecScatterDestroy(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
