/*$Id: psplit.c,v 1.16 2001/03/23 23:20:45 balay Exp $*/

#include "petsc.h"           /*I    "petsc.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscSplitOwnershipBlock"
/*@C
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
int PetscSplitOwnershipBlock(MPI_Comm comm,int bs,int *n,int *N)
{
  int ierr,size,rank;

  PetscFunctionBegin;
  if (*N == PETSC_DECIDE && *n == PETSC_DECIDE) SETERRQ(1,"Both n and N cannot be PETSC_DECIDE");

  if (*N == PETSC_DECIDE) { 
    if (*n % bs != 0) SETERRQ2(1,"local size %d not divisible by block size %d",*n,bs);
    ierr = MPI_Allreduce(n,N,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  } else if (*n == PETSC_DECIDE) { 
    int Nbs = *N/bs;
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr); 
    *n = bs*(Nbs/size + ((Nbs % size) > rank));
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscSplitOwnership"
/*@C
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
int PetscSplitOwnership(MPI_Comm comm,int *n,int *N)
{
  int ierr,size,rank;

  PetscFunctionBegin;
  if (*N == PETSC_DECIDE && *n == PETSC_DECIDE) SETERRQ(1,"Both n and N cannot be PETSC_DECIDE");

  if (*N == PETSC_DECIDE) { 
    ierr = MPI_Allreduce(n,N,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  } else if (*n == PETSC_DECIDE) { 
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr); 
    *n = *N/size + ((*N % size) > rank);
#if defined(PETSC_USE_BOPT_g)
  } else {
    int tmp;
    ierr = MPI_Allreduce(n,&tmp,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
    if (tmp != *N) SETERRQ3(1,"Sum of local lengths %d does not equal global length %d, my local length %d",tmp,*N,*n);
#endif
  }

  PetscFunctionReturn(0);
}

