#include <petscsys.h> /*I  "petscsys.h"  I*/
#include <petsc/private/petscimpl.h>

struct _n_PetscShmComm {
  PetscMPIInt *globranks;         /* global ranks of each rank in the shared memory communicator */
  PetscMPIInt  shmsize;           /* size of the shared memory communicator */
  MPI_Comm     globcomm, shmcomm; /* global communicator and shared memory communicator (a sub-communicator of the former) */
};

/*
   Private routine to delete internal shared memory communicator when a communicator is freed.

   This is called by MPI, not by users. This is called by MPI_Comm_free() when the communicator that has this  data as an attribute is freed.

   Note: this is declared extern "C" because it is passed to MPI_Comm_create_keyval()

*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_ShmComm_Attr_DeleteFn(MPI_Comm comm, PetscMPIInt keyval, void *val, void *extra_state)
{
  PetscShmComm p = (PetscShmComm)val;

  PetscFunctionBegin;
  PetscCallReturnMPI(PetscInfo(NULL, "Deleting shared memory subcommunicator in a MPI_Comm %ld\n", (long)comm));
  PetscCallMPIReturnMPI(MPI_Comm_free(&p->shmcomm));
  PetscCallReturnMPI(PetscFree(p->globranks));
  PetscCallReturnMPI(PetscFree(val));
  PetscFunctionReturn(MPI_SUCCESS);
}

#ifdef PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY
  /* Data structures to support freeing comms created in PetscShmCommGet().
  Since we predict communicators passed to PetscShmCommGet() are very likely
  either a PETSc inner communicator or an MPI communicator with a linked PETSc
  inner communicator, we use a simple static array to store dupped communicators
  on rare cases otherwise.
 */
  #define MAX_SHMCOMM_DUPPED_COMMS 16
static PetscInt       num_dupped_comms = 0;
static MPI_Comm       shmcomm_dupped_comms[MAX_SHMCOMM_DUPPED_COMMS];
static PetscErrorCode PetscShmCommDestroyDuppedComms(void)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < num_dupped_comms; i++) PetscCall(PetscCommDestroy(&shmcomm_dupped_comms[i]));
  num_dupped_comms = 0; /* reset so that PETSc can be reinitialized */
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*@C
  PetscShmCommGet - Returns a sub-communicator of all ranks that share a common memory

  Collective.

  Input Parameter:
. globcomm - `MPI_Comm`, which can be a user `MPI_Comm` or a PETSc inner `MPI_Comm`

  Output Parameter:
. pshmcomm - the PETSc shared memory communicator object

  Level: developer

  Note:
  When used with MPICH, MPICH must be configured with `--download-mpich-device=ch3:nemesis`

.seealso: `PetscShmCommGlobalToLocal()`, `PetscShmCommLocalToGlobal()`, `PetscShmCommGetMpiShmComm()`
@*/
PetscErrorCode PetscShmCommGet(MPI_Comm globcomm, PetscShmComm *pshmcomm)
{
#ifdef PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY
  MPI_Group         globgroup, shmgroup;
  PetscMPIInt      *shmranks, i, flg;
  PetscCommCounter *counter;

  PetscFunctionBegin;
  PetscAssertPointer(pshmcomm, 2);
  /* Get a PETSc inner comm, since we always want to stash pshmcomm on PETSc inner comms */
  PetscCallMPI(MPI_Comm_get_attr(globcomm, Petsc_Counter_keyval, &counter, &flg));
  if (!flg) { /* globcomm is not a PETSc comm */
    union
    {
      MPI_Comm comm;
      void    *ptr;
    } ucomm;
    /* check if globcomm already has a linked PETSc inner comm */
    PetscCallMPI(MPI_Comm_get_attr(globcomm, Petsc_InnerComm_keyval, &ucomm, &flg));
    if (!flg) {
      /* globcomm does not have a linked PETSc inner comm, so we create one and replace globcomm with it */
      PetscCheck(num_dupped_comms < MAX_SHMCOMM_DUPPED_COMMS, globcomm, PETSC_ERR_PLIB, "PetscShmCommGet() is trying to dup more than %d MPI_Comms", MAX_SHMCOMM_DUPPED_COMMS);
      PetscCall(PetscCommDuplicate(globcomm, &globcomm, NULL));
      /* Register a function to free the dupped PETSc comms at PetscFinalize() at the first time */
      if (num_dupped_comms == 0) PetscCall(PetscRegisterFinalize(PetscShmCommDestroyDuppedComms));
      shmcomm_dupped_comms[num_dupped_comms] = globcomm;
      num_dupped_comms++;
    } else {
      /* otherwise, we pull out the inner comm and use it as globcomm */
      globcomm = ucomm.comm;
    }
  }

  /* Check if globcomm already has an attached pshmcomm. If no, create one */
  PetscCallMPI(MPI_Comm_get_attr(globcomm, Petsc_ShmComm_keyval, pshmcomm, &flg));
  if (flg) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscNew(pshmcomm));
  (*pshmcomm)->globcomm = globcomm;

  PetscCallMPI(MPI_Comm_split_type(globcomm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &(*pshmcomm)->shmcomm));

  PetscCallMPI(MPI_Comm_size((*pshmcomm)->shmcomm, &(*pshmcomm)->shmsize));
  PetscCallMPI(MPI_Comm_group(globcomm, &globgroup));
  PetscCallMPI(MPI_Comm_group((*pshmcomm)->shmcomm, &shmgroup));
  PetscCall(PetscMalloc1((*pshmcomm)->shmsize, &shmranks));
  PetscCall(PetscMalloc1((*pshmcomm)->shmsize, &(*pshmcomm)->globranks));
  for (i = 0; i < (*pshmcomm)->shmsize; i++) shmranks[i] = i;
  PetscCallMPI(MPI_Group_translate_ranks(shmgroup, (*pshmcomm)->shmsize, shmranks, globgroup, (*pshmcomm)->globranks));
  PetscCall(PetscFree(shmranks));
  PetscCallMPI(MPI_Group_free(&globgroup));
  PetscCallMPI(MPI_Group_free(&shmgroup));

  for (i = 0; i < (*pshmcomm)->shmsize; i++) PetscCall(PetscInfo(NULL, "Shared memory rank %d global rank %d\n", i, (*pshmcomm)->globranks[i]));
  PetscCallMPI(MPI_Comm_set_attr(globcomm, Petsc_ShmComm_keyval, *pshmcomm));
  PetscFunctionReturn(PETSC_SUCCESS);
#else
  SETERRQ(globcomm, PETSC_ERR_SUP, "Shared memory communicators need MPI-3 package support.\nPlease upgrade your MPI or reconfigure with --download-mpich.");
#endif
}

/*@C
  PetscShmCommGlobalToLocal - Given a global rank returns the local rank in the shared memory communicator

  Input Parameters:
+ pshmcomm - the shared memory communicator object
- grank    - the global rank

  Output Parameter:
. lrank - the local rank, or `MPI_PROC_NULL` if it does not exist

  Level: developer

  Developer Notes:
  Assumes the pshmcomm->globranks[] is sorted

  It may be better to rewrite this to map multiple global ranks to local in the same function call

.seealso: `PetscShmCommGet()`, `PetscShmCommLocalToGlobal()`, `PetscShmCommGetMpiShmComm()`
@*/
PetscErrorCode PetscShmCommGlobalToLocal(PetscShmComm pshmcomm, PetscMPIInt grank, PetscMPIInt *lrank)
{
  PetscMPIInt low, high, t, i;
  PetscBool   flg = PETSC_FALSE;

  PetscFunctionBegin;
  PetscAssertPointer(pshmcomm, 1);
  PetscAssertPointer(lrank, 3);
  *lrank = MPI_PROC_NULL;
  if (grank < pshmcomm->globranks[0]) PetscFunctionReturn(PETSC_SUCCESS);
  if (grank > pshmcomm->globranks[pshmcomm->shmsize - 1]) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-noshared", &flg, NULL));
  if (flg) PetscFunctionReturn(PETSC_SUCCESS);
  low  = 0;
  high = pshmcomm->shmsize;
  while (high - low > 5) {
    t = (low + high) / 2;
    if (pshmcomm->globranks[t] > grank) high = t;
    else low = t;
  }
  for (i = low; i < high; i++) {
    if (pshmcomm->globranks[i] > grank) PetscFunctionReturn(PETSC_SUCCESS);
    if (pshmcomm->globranks[i] == grank) {
      *lrank = i;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscShmCommLocalToGlobal - Given a local rank in the shared memory communicator returns the global rank

  Input Parameters:
+ pshmcomm - the shared memory communicator object
- lrank    - the local rank in the shared memory communicator

  Output Parameter:
. grank - the global rank in the global communicator where the shared memory communicator is built

  Level: developer

.seealso: `PetscShmCommGlobalToLocal()`, `PetscShmCommGet()`, `PetscShmCommGetMpiShmComm()`
@*/
PetscErrorCode PetscShmCommLocalToGlobal(PetscShmComm pshmcomm, PetscMPIInt lrank, PetscMPIInt *grank)
{
  PetscFunctionBegin;
  PetscAssertPointer(pshmcomm, 1);
  PetscAssertPointer(grank, 3);
  PetscCheck(lrank >= 0 && lrank < pshmcomm->shmsize, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No rank %d in the shared memory communicator", lrank);
  *grank = pshmcomm->globranks[lrank];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscShmCommGetMpiShmComm - Returns the MPI communicator that represents all processes with common shared memory

  Input Parameter:
. pshmcomm - PetscShmComm object obtained with PetscShmCommGet()

  Output Parameter:
. comm - the MPI communicator

  Level: developer

.seealso: `PetscShmCommGlobalToLocal()`, `PetscShmCommGet()`, `PetscShmCommLocalToGlobal()`
@*/
PetscErrorCode PetscShmCommGetMpiShmComm(PetscShmComm pshmcomm, MPI_Comm *comm)
{
  PetscFunctionBegin;
  PetscAssertPointer(pshmcomm, 1);
  PetscAssertPointer(comm, 2);
  *comm = pshmcomm->shmcomm;
  PetscFunctionReturn(PETSC_SUCCESS);
}
