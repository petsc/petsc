#define PETSC_DLL
/*
     Provides utility routines for split MPI communicator.
*/
#include "petscsys.h"    /*I   "petscsys.h"    I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscSubcommDestroy"
PetscErrorCode PETSCMAT_DLLEXPORT PetscSubcommDestroy(PetscSubcomm psubcomm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_free(&psubcomm->dupparent);CHKERRQ(ierr);
  ierr = MPI_Comm_free(&psubcomm->comm);CHKERRQ(ierr);
  ierr = PetscFree(psubcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSubcommCreate"
/*@C
  PetscSubcommCreate - Create a PetscSubcomm context.

   Collective on MPI_Comm

   Input Parameter:
+  comm - MPI communicator
-  nsubcomm - the number of subcommunicators to be created

   Output Parameter:
.  psubcomm - location to store the PetscSubcomm context


   Notes:
   To avoid data scattering from subcomm back to original comm, we create subcommunicators 
   by iteratively taking a process into a subcommunicator. 
   Example: size=4, nsubcomm=(*psubcomm)->n=3
     comm=(*psubcomm)->parent:
      rank:     [0]  [1]  [2]  [3]
      color:     0    1    2    0

     subcomm=(*psubcomm)->comm:
      subrank:  [0]  [0]  [0]  [1]    

     dupcomm=(*psubcomm)->dupparent:
      duprank:  [0]  [2]  [3]  [1]

     Here, subcomm[color = 0] has subsize=2, owns process [0] and [3]
           subcomm[color = 1] has subsize=1, owns process [1]
           subcomm[color = 2] has subsize=1, owns process [2]
           dupcomm has same number of processes as comm, and its duprank maps
           processes in subcomm contiguously into a 1d array:
            duprank: [0] [1]      [2]         [3]
            rank:    [0] [3]      [1]         [2]
                    subcomm[0] subcomm[1]  subcomm[2]

   Level: advanced

.keywords: communicator, create

.seealso: PetscSubcommDestroy()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT PetscSubcommCreate(MPI_Comm comm,PetscInt nsubcomm,PetscSubcomm *psubcomm)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size,*subsize,duprank,subrank;
  PetscInt       np_subcomm,nleftover,i,j,color;
  MPI_Comm       subcomm=0,dupcomm=0;
  PetscSubcomm   psubcomm_tmp;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (nsubcomm < 1 || nsubcomm > size) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Num of subcommunicators %D cannot be < 1 or > input comm size %D",nsubcomm,size);

  /* get size of each subcommunicator */
  ierr = PetscMalloc((1+nsubcomm)*sizeof(PetscMPIInt),&subsize);CHKERRQ(ierr);
  np_subcomm = size/nsubcomm;
  nleftover  = size - nsubcomm*np_subcomm;
  for (i=0; i<nsubcomm; i++){
    subsize[i] = np_subcomm;
    if (i<nleftover) subsize[i]++;
  }

  /* find color for this proc */
  color   = rank%nsubcomm;
  subrank = rank/nsubcomm;

  ierr = MPI_Comm_split(comm,color,subrank,&subcomm);CHKERRQ(ierr);

  j = 0; duprank = 0;
  for (i=0; i<nsubcomm; i++){
    if (j == color){
      duprank += subrank;
      break;
    }
    duprank += subsize[i]; j++;
  }
  ierr = PetscFree(subsize);CHKERRQ(ierr);
 
  /* create dupcomm with same size as comm, but its rank, duprank, maps subcomm's contiguously into dupcomm */   
  ierr = MPI_Comm_split(comm,0,duprank,&dupcomm);CHKERRQ(ierr);
 
  ierr = PetscNew(struct _n_PetscSubcomm,&psubcomm_tmp);CHKERRQ(ierr);
  psubcomm_tmp->parent    = comm;
  psubcomm_tmp->dupparent = dupcomm;
  psubcomm_tmp->comm      = subcomm;
  psubcomm_tmp->n         = nsubcomm;
  psubcomm_tmp->color     = color;
  *psubcomm = psubcomm_tmp;
  PetscFunctionReturn(0);
}
