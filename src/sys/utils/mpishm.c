#include <petscsys.h>        /*I  "petscsys.h"  I*/
#include <petsc/private/petscimpl.h>

struct _n_PetscShmComm {
  PetscMPIInt *globranks;       /* global ranks of each rank in the shared memory communicator */
  PetscMPIInt shmsize;          /* size of the shared memory communicator */
  MPI_Comm    globcomm,shmcomm; /* global communicator and shared memory communicator (a sub-communicator of the former) */
};

/*
   Private routine to delete internal tag/name shared memory communicator when a communicator is freed.

   This is called by MPI, not by users. This is called by MPI_Comm_free() when the communicator that has this  data as an attribute is freed.

   Note: this is declared extern "C" because it is passed to MPI_Comm_create_keyval()

*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelComm_Shm(MPI_Comm comm,PetscMPIInt keyval,void *val,void *extra_state)
{
  PetscErrorCode  ierr;
  PetscShmComm p = (PetscShmComm)val;

  PetscFunctionBegin;
  ierr = PetscInfo1(0,"Deleting shared memory subcommunicator in a MPI_Comm %ld\n",(long)comm);CHKERRMPI(ierr);
  ierr = MPI_Comm_free(&p->shmcomm);CHKERRMPI(ierr);
  ierr = PetscFree(p->globranks);CHKERRMPI(ierr);
  ierr = PetscFree(val);CHKERRMPI(ierr);
  PetscFunctionReturn(MPI_SUCCESS);
}

/*@C
    PetscShmCommGet - Given a PETSc communicator returns a communicator of all ranks that share a common memory


    Collective on comm.

    Input Parameter:
.   globcomm - MPI_Comm

    Output Parameter:
.   pshmcomm - the PETSc shared memory communicator object

    Level: developer

    Notes:
    This should be called only with an PetscCommDuplicate() communictor

           When used with MPICH, MPICH must be configured with --download-mpich-device=ch3:nemesis

    Concepts: MPI subcomm^numbering

@*/
PetscErrorCode PetscShmCommGet(MPI_Comm globcomm,PetscShmComm *pshmcomm)
{
#ifdef PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY
  PetscErrorCode   ierr;
  MPI_Group        globgroup,shmgroup;
  PetscMPIInt      *shmranks,i,flg;
  PetscCommCounter *counter;

  PetscFunctionBegin;
  ierr = MPI_Comm_get_attr(globcomm,Petsc_Counter_keyval,&counter,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(globcomm,PETSC_ERR_ARG_CORRUPT,"Bad MPI communicator supplied; must be a PETSc communicator");

  ierr = MPI_Comm_get_attr(globcomm,Petsc_ShmComm_keyval,pshmcomm,&flg);CHKERRQ(ierr);
  if (flg) PetscFunctionReturn(0);

  ierr        = PetscNew(pshmcomm);CHKERRQ(ierr);
  (*pshmcomm)->globcomm = globcomm;

  ierr = MPI_Comm_split_type(globcomm, MPI_COMM_TYPE_SHARED,0, MPI_INFO_NULL,&(*pshmcomm)->shmcomm);CHKERRQ(ierr);

  ierr = MPI_Comm_size((*pshmcomm)->shmcomm,&(*pshmcomm)->shmsize);CHKERRQ(ierr);
  ierr = MPI_Comm_group(globcomm, &globgroup);CHKERRQ(ierr);
  ierr = MPI_Comm_group((*pshmcomm)->shmcomm, &shmgroup);CHKERRQ(ierr);
  ierr = PetscMalloc1((*pshmcomm)->shmsize,&shmranks);CHKERRQ(ierr);
  ierr = PetscMalloc1((*pshmcomm)->shmsize,&(*pshmcomm)->globranks);CHKERRQ(ierr);
  for (i=0; i<(*pshmcomm)->shmsize; i++) shmranks[i] = i;
  ierr = MPI_Group_translate_ranks(shmgroup, (*pshmcomm)->shmsize, shmranks, globgroup, (*pshmcomm)->globranks);CHKERRQ(ierr);
  ierr = PetscFree(shmranks);CHKERRQ(ierr);
  ierr = MPI_Group_free(&globgroup);CHKERRQ(ierr);
  ierr = MPI_Group_free(&shmgroup);CHKERRQ(ierr);

  for (i=0; i<(*pshmcomm)->shmsize; i++) {
    ierr = PetscInfo2(NULL,"Shared memory rank %d global rank %d\n",i,(*pshmcomm)->globranks[i]);CHKERRQ(ierr);
  }
  ierr = MPI_Comm_set_attr(globcomm,Petsc_ShmComm_keyval,*pshmcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  SETERRQ(globcomm, PETSC_ERR_SUP, "Shared memory communicators need MPI-3 package support.\nPlease upgrade your MPI or reconfigure with --download-mpich.");
#endif
}

/*@C
    PetscShmCommGlobalToLocal - Given a global rank returns the local rank in the shared memory communicator

    Input Parameters:
+   pshmcomm - the shared memory communicator object
-   grank    - the global rank

    Output Parameter:
.   lrank - the local rank, or MPI_PROC_NULL if it does not exist

    Level: developer

    Developer Notes:
    Assumes the pshmcomm->globranks[] is sorted

    It may be better to rewrite this to map multiple global ranks to local in the same function call

    Concepts: MPI subcomm^numbering

@*/
PetscErrorCode PetscShmCommGlobalToLocal(PetscShmComm pshmcomm,PetscMPIInt grank,PetscMPIInt *lrank)
{
  PetscMPIInt    low,high,t,i;
  PetscBool      flg = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *lrank = MPI_PROC_NULL;
  if (grank < pshmcomm->globranks[0]) PetscFunctionReturn(0);
  if (grank > pshmcomm->globranks[pshmcomm->shmsize-1]) PetscFunctionReturn(0);
  ierr = PetscOptionsGetBool(NULL,NULL,"-noshared",&flg,NULL);CHKERRQ(ierr);
  if (flg) PetscFunctionReturn(0);
  low  = 0;
  high = pshmcomm->shmsize;
  while (high-low > 5) {
    t = (low+high)/2;
    if (pshmcomm->globranks[t] > grank) high = t;
    else low = t;
  }
  for (i=low; i<high; i++) {
    if (pshmcomm->globranks[i] > grank) PetscFunctionReturn(0);
    if (pshmcomm->globranks[i] == grank) {
      *lrank = i;
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscShmCommLocalToGlobal - Given a local rank in the shared memory communicator returns the global rank

    Input Parameters:
+   pshmcomm - the shared memory communicator object
-   lrank    - the local rank in the shared memory communicator

    Output Parameter:
.   grank - the global rank in the global communicator where the shared memory communicator is built

    Level: developer

    Concepts: MPI subcomm^numbering
@*/
PetscErrorCode PetscShmCommLocalToGlobal(PetscShmComm pshmcomm,PetscMPIInt lrank,PetscMPIInt *grank)
{
  PetscFunctionBegin;
#ifdef PETSC_USE_DEBUG
  {
    PetscErrorCode ierr;
    if (lrank < 0 || lrank >= pshmcomm->shmsize) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"No rank %D in the shared memory communicator",lrank);CHKERRQ(ierr); }
  }
#endif
  *grank = pshmcomm->globranks[lrank];
  PetscFunctionReturn(0);
}

/*@C
    PetscShmCommGetMpiShmComm - Returns the MPI communicator that represents all processes with common shared memory

    Input Parameter:
.   pshmcomm - PetscShmComm object obtained with PetscShmCommGet()

    Output Parameter:
.   comm     - the MPI communicator

    Level: developer

@*/
PetscErrorCode PetscShmCommGetMpiShmComm(PetscShmComm pshmcomm,MPI_Comm *comm)
{
  PetscFunctionBegin;
  *comm = pshmcomm->shmcomm;
  PetscFunctionReturn(0);
}

