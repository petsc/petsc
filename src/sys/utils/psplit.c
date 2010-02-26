#define PETSC_DLL

#include "petscsys.h"           /*I    "petscsys.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscSplitOwnershipBlock"
/*@
    PetscSplitOwnershipBlock - Given a global (or local) length determines a local 
        (or global) length via a simple formula. Splits so each processors local size
        is divisible by the block size.

   Collective on MPI_Comm (if N is PETSC_DECIDE)

   Input Parameters:
+    comm - MPI communicator that shares the object being divided
.    bs - block size
.    n - local length (or PETSC_DECIDE to have it set)
-    N - global length (or PETSC_DECIDE)

  Level: developer

   Notes:
     n and N cannot be both PETSC_DECIDE

     If one processor calls this with N of PETSC_DECIDE then all processors
     must, otherwise the program will hang.

.seealso: PetscSplitOwnership()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscSplitOwnershipBlock(MPI_Comm comm,PetscInt bs,PetscInt *n,PetscInt *N)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;

  PetscFunctionBegin;
  if (*N == PETSC_DECIDE && *n == PETSC_DECIDE) SETERRQ(PETSC_ERR_ARG_INCOMP,"Both n and N cannot be PETSC_DECIDE");

  if (*N == PETSC_DECIDE) { 
    if (*n % bs != 0) SETERRQ2(PETSC_ERR_ARG_INCOMP,"local size %D not divisible by block size %D",*n,bs);
    ierr = MPI_Allreduce(n,N,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  } else if (*n == PETSC_DECIDE) { 
    PetscInt Nbs = *N/bs;
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr); 
    *n = bs*(Nbs/size + ((Nbs % size) > rank));
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscSplitOwnership"
/*@
    PetscSplitOwnership - Given a global (or local) length determines a local 
        (or global) length via a simple formula

   Collective on MPI_Comm (if N is PETSC_DECIDE)

   Input Parameters:
+    comm - MPI communicator that shares the object being divided
.    n - local length (or PETSC_DECIDE to have it set)
-    N - global length (or PETSC_DECIDE)

  Level: developer

   Notes:
     n and N cannot be both PETSC_DECIDE

     If one processor calls this with N of PETSC_DECIDE then all processors
     must, otherwise the program will hang.

.seealso: PetscSplitOwnershipBlock()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscSplitOwnership(MPI_Comm comm,PetscInt *n,PetscInt *N)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;

  PetscFunctionBegin;
  if (*N == PETSC_DECIDE && *n == PETSC_DECIDE) SETERRQ(PETSC_ERR_ARG_INCOMP,"Both n and N cannot be PETSC_DECIDE\n  likely a call to VecSetSizes() or MatSetSizes() is wrong.\nSee http://www.mcs.anl.gov/petsc/petsc-as/documentation/troubleshooting.html#PetscSplitOwnership");

  if (*N == PETSC_DECIDE) { 
    ierr = MPI_Allreduce(n,N,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  } else if (*n == PETSC_DECIDE) { 
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr); 
    *n = *N/size + ((*N % size) > rank);
#if defined(PETSC_USE_DEBUG)
  } else {
    PetscInt tmp;
    ierr = MPI_Allreduce(n,&tmp,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
    if (tmp != *N) SETERRQ3(PETSC_ERR_ARG_SIZ,"Sum of local lengths %D does not equal global length %D, my local length %D\n  likely a call to VecSetSizes() or MatSetSizes() is wrong.\nSee http://www.mcs.anl.gov/petsc/petsc-as/documentation/troubleshooting.html#PetscSplitOwnership",tmp,*N,*n);
#endif
  }
  PetscFunctionReturn(0);
}

