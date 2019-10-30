
#include <petsc/private/petscimpl.h>              /*I "petscsys.h" I*/

/* Logging support */
PetscLogEvent PETSC_Barrier;

static int hash(const char *str)
{
  unsigned int c,hash = 5381;

  while ((c = *str++)) hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
  return hash;
}

/*
   This is used by MPIU_Allreduce() to insure that all callers originated from the same place in the PETSc code
*/
PetscErrorCode PetscAllreduceBarrierCheck(MPI_Comm comm,PetscMPIInt ctn,int line,const char *func,const char *file)
{
  PetscMPIInt err;
  PetscMPIInt b1[6],b2[6];

  b1[0] = -(PetscMPIInt)line;       b1[1] = -b1[0];
  b1[2] = -(PetscMPIInt)hash(func); b1[3] = -b1[2];
  b1[4] = -(PetscMPIInt)ctn;        b1[5] = -b1[4];
  err = MPI_Allreduce(b1,b2,6,MPI_INT,MPI_MAX,comm);
  if (err) return PetscError(PETSC_COMM_SELF,line,func,file,PETSC_ERR_LIB,PETSC_ERROR_INITIAL,"MPI_Allreduce() failed with error code %d",err);
  if (-b2[0] != b2[1]) return PetscError(PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"MPI_Allreduce() called in different locations (code lines) on different processors");
  if (-b2[2] != b2[3]) return PetscError(PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"MPI_Allreduce() called in different locations (functions) on different processors");
  if (-b2[4] != b2[5]) return PetscError(PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"MPI_Allreduce() called with different counts %d on different processors",ctn);
  return 0;
}

/*@C
    PetscBarrier - Blocks until this routine is executed by all
                   processors owning the object obj.

   Input Parameters:
.  obj - PETSc object  (Mat, Vec, IS, SNES etc...)
        The object be caste with a (PetscObject). NULL can be used to indicate the barrier should be across MPI_COMM_WORLD

  Level: intermediate

  Notes:
  This routine calls MPI_Barrier with the communicator of the PETSc Object obj

  Fortran Usage:
    You may pass PETSC_NULL_VEC or any other PETSc null object, such as PETSC_NULL_MAT, to indicate the barrier should be
    across MPI_COMM_WORLD. You can also pass in any PETSc object, Vec, Mat, etc

@*/
PetscErrorCode  PetscBarrier(PetscObject obj)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  if (obj) PetscValidHeader(obj,1);
  ierr = PetscLogEventBegin(PETSC_Barrier,obj,0,0,0);CHKERRQ(ierr);
  if (obj) {
    ierr = PetscObjectGetComm(obj,&comm);CHKERRQ(ierr);
  } else comm = PETSC_COMM_WORLD;
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSC_Barrier,obj,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

