
/*
     Provides utility routines for split MPI communicator.
*/
#include <petscsys.h>    /*I   "petscsys.h"    I*/
#include <petsc-private/threadcommimpl.h> /* Petsc_ThreadComm_keyval */

const char *const PetscSubcommTypes[] = {"GENERAL","CONTIGUOUS","INTERLACED","PetscSubcommType","PETSC_SUBCOMM_",0};

extern PetscErrorCode PetscSubcommCreate_contiguous(PetscSubcomm);
extern PetscErrorCode PetscSubcommCreate_interlaced(PetscSubcomm);

#undef __FUNCT__
#define __FUNCT__ "PetscSubcommSetNumber"
/*@C
  PetscSubcommSetNumber - Set total number of subcommunicators.

   Collective on MPI_Comm

   Input Parameter:
+  psubcomm - PetscSubcomm context
-  nsubcomm - the total number of subcommunicators in psubcomm

   Level: advanced

.keywords: communicator

.seealso: PetscSubcommCreate(),PetscSubcommDestroy(),PetscSubcommSetType(),PetscSubcommSetTypeGeneral()
@*/
PetscErrorCode  PetscSubcommSetNumber(PetscSubcomm psubcomm,PetscInt nsubcomm)
{
  PetscErrorCode ierr;
  MPI_Comm       comm=psubcomm->parent;
  PetscMPIInt    rank,size;

  PetscFunctionBegin;
  if (!psubcomm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"PetscSubcomm is not created. Call PetscSubcommCreate() first");
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (nsubcomm < 1 || nsubcomm > size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "Num of subcommunicators %D cannot be < 1 or > input comm size %D",nsubcomm,size);

  psubcomm->n = nsubcomm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSubcommSetType"
/*@C
  PetscSubcommSetType - Set type of subcommunicators.

   Collective on MPI_Comm

   Input Parameter:
+  psubcomm - PetscSubcomm context
-  subcommtype - subcommunicator type, PETSC_SUBCOMM_CONTIGUOUS,PETSC_SUBCOMM_INTERLACED

   Level: advanced

.keywords: communicator

.seealso: PetscSubcommCreate(),PetscSubcommDestroy(),PetscSubcommSetNumber(),PetscSubcommSetTypeGeneral()
@*/
PetscErrorCode  PetscSubcommSetType(PetscSubcomm psubcomm,PetscSubcommType subcommtype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!psubcomm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"PetscSubcomm is not created. Call PetscSubcommCreate()");
  if (psubcomm->n < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"number of subcommunicators %D is incorrect. Call PetscSubcommSetNumber()",psubcomm->n);

  if (subcommtype == PETSC_SUBCOMM_CONTIGUOUS){
    ierr = PetscSubcommCreate_contiguous(psubcomm);CHKERRQ(ierr);
  } else if (subcommtype == PETSC_SUBCOMM_INTERLACED){
    ierr = PetscSubcommCreate_interlaced(psubcomm);CHKERRQ(ierr);
  } else SETERRQ1(psubcomm->parent,PETSC_ERR_SUP,"PetscSubcommType %D is not supported yet",subcommtype);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSubcommSetTypeGeneral"
/*@C
  PetscSubcommSetTypeGeneral - Set type of subcommunicators from user's specifications

   Collective on MPI_Comm

   Input Parameter:
+  psubcomm - PetscSubcomm context
.  color   - control of subset assignment (nonnegative integer). Processes with the same color are in the same subcommunicator.
.  subrank - rank in the subcommunicator
-  duprank - rank in the dupparent (see PetscSubcomm)

   Level: advanced

.keywords: communicator, create

.seealso: PetscSubcommCreate(),PetscSubcommDestroy(),PetscSubcommSetNumber(),PetscSubcommSetType()
@*/
PetscErrorCode  PetscSubcommSetTypeGeneral(PetscSubcomm psubcomm,PetscMPIInt color,PetscMPIInt subrank,PetscMPIInt duprank)
{
  PetscErrorCode ierr;
  MPI_Comm       subcomm=0,dupcomm=0,comm=psubcomm->parent;
  PetscMPIInt    size;

  PetscFunctionBegin;
  if (!psubcomm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"PetscSubcomm is not created. Call PetscSubcommCreate()");
  if (psubcomm->n < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"number of subcommunicators %D is incorrect. Call PetscSubcommSetNumber()",psubcomm->n);

  ierr = MPI_Comm_split(comm,color,subrank,&subcomm);CHKERRQ(ierr);

  /* create dupcomm with same size as comm, but its rank, duprank, maps subcomm's contiguously into dupcomm
     if duprank is not a valid number, then dupcomm is not created - not all applications require dupcomm! */
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (duprank == PETSC_DECIDE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"duprank==PETSC_DECIDE is not supported yet");
  else if (duprank >= 0 && duprank < size){
    ierr = MPI_Comm_split(comm,0,duprank,&dupcomm);CHKERRQ(ierr);
  }
  ierr = PetscCommDuplicate(dupcomm,&psubcomm->dupparent,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscCommDuplicate(subcomm,&psubcomm->comm,PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_free(&dupcomm);CHKERRQ(ierr);
  ierr = MPI_Comm_free(&subcomm);CHKERRQ(ierr);
  psubcomm->color     = color;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSubcommDestroy"
PetscErrorCode  PetscSubcommDestroy(PetscSubcomm *psubcomm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*psubcomm) PetscFunctionReturn(0);
  ierr = PetscCommDestroy(&(*psubcomm)->dupparent);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&(*psubcomm)->comm);CHKERRQ(ierr);
  ierr = PetscFree((*psubcomm));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSubcommCreate"
/*@C
  PetscSubcommCreate - Create a PetscSubcomm context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  psubcomm - location to store the PetscSubcomm context

   Level: advanced

.keywords: communicator, create

.seealso: PetscSubcommDestroy()
@*/
PetscErrorCode  PetscSubcommCreate(MPI_Comm comm,PetscSubcomm *psubcomm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _n_PetscSubcomm,psubcomm);CHKERRQ(ierr);
  (*psubcomm)->parent = comm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSubcommCreate_contiguous"
PetscErrorCode PetscSubcommCreate_contiguous(PetscSubcomm psubcomm)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size,*subsize,duprank=-1,subrank=-1;
  PetscInt       np_subcomm,nleftover,i,color=-1,rankstart,nsubcomm=psubcomm->n;
  MPI_Comm       subcomm=0,dupcomm=0,comm=psubcomm->parent;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* get size of each subcommunicator */
  ierr = PetscMalloc((1+nsubcomm)*sizeof(PetscMPIInt),&subsize);CHKERRQ(ierr);
  np_subcomm = size/nsubcomm;
  nleftover  = size - nsubcomm*np_subcomm;
  for (i=0; i<nsubcomm; i++){
    subsize[i] = np_subcomm;
    if (i<nleftover) subsize[i]++;
  }

  /* get color and subrank of this proc */
  rankstart = 0;
  for (i=0; i<nsubcomm; i++){
    if ( rank >= rankstart && rank < rankstart+subsize[i]) {
      color   = i;
      subrank = rank - rankstart;
      duprank = rank;
      break;
    } else {
      rankstart += subsize[i];
    }
  }
  ierr = PetscFree(subsize);CHKERRQ(ierr);

  ierr = MPI_Comm_split(comm,color,subrank,&subcomm);CHKERRQ(ierr);

  /* create dupcomm with same size as comm, but its rank, duprank, maps subcomm's contiguously into dupcomm */
  ierr = MPI_Comm_split(comm,0,duprank,&dupcomm);CHKERRQ(ierr);

  ierr = PetscCommDuplicate(dupcomm,&psubcomm->dupparent,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscCommDuplicate(subcomm,&psubcomm->comm,PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_free(&dupcomm);CHKERRQ(ierr);
  ierr = MPI_Comm_free(&subcomm);CHKERRQ(ierr);
  psubcomm->color     = color;

  {
    PetscThreadComm tcomm;
    ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
    ierr = MPI_Attr_put(psubcomm->dupparent,Petsc_ThreadComm_keyval,tcomm);CHKERRQ(ierr);
    tcomm->refct++;
    ierr = MPI_Attr_put(psubcomm->comm,Petsc_ThreadComm_keyval,tcomm);CHKERRQ(ierr);
    tcomm->refct++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSubcommCreate_interlaced"
/*
   Note:
   In PCREDUNDANT, to avoid data scattering from subcomm back to original comm, we create subcommunicators
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
*/

PetscErrorCode PetscSubcommCreate_interlaced(PetscSubcomm psubcomm)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size,*subsize,duprank,subrank;
  PetscInt       np_subcomm,nleftover,i,j,color,nsubcomm=psubcomm->n;
  MPI_Comm       subcomm=0,dupcomm=0,comm=psubcomm->parent;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

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

  ierr = PetscCommDuplicate(dupcomm,&psubcomm->dupparent,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscCommDuplicate(subcomm,&psubcomm->comm,PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_free(&dupcomm);CHKERRQ(ierr);
  ierr = MPI_Comm_free(&subcomm);CHKERRQ(ierr);
  psubcomm->color     = color;

  {
    PetscThreadComm tcomm;
    ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
    ierr = MPI_Attr_put(psubcomm->dupparent,Petsc_ThreadComm_keyval,tcomm);CHKERRQ(ierr);
    tcomm->refct++;
    ierr = MPI_Attr_put(psubcomm->comm,Petsc_ThreadComm_keyval,tcomm);CHKERRQ(ierr);
    tcomm->refct++;
  }
  PetscFunctionReturn(0);
}


