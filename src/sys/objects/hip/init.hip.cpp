/*

   This file defines the HIP initialization of PETSc

*/

#include <petscsys.h>        /*I  "petscsys.h"   I*/
#include <petsc/private/petscimpl.h>
#include <petschipblas.h>



/*
     PetscHIPInitializeLogView - Initializes the HIP device for the case when -log_view is called
     This is to do costly hip runtime initialization early so that not to distort the timing later.
@*/
PETSC_EXTERN PetscErrorCode PetscHIPInitializeLogView(MPI_Comm comm, PetscInt device)
{
  PetscErrorCode        ierr;
  hipError_t            cerr;
  int                   devId,devCount=0;
  PetscMPIInt           rank;

  PetscFunctionBegin;
  devCount = 0;
  cerr = hipGetDeviceCount(&devCount);
  hipGetLastError(); /* Reset the last error */
  if (cerr == hipSuccess && devCount >= 1) { /* There are GPU(s) */
    devId = 0;
    if (devCount > 1) { /* Decide which GPU to init when there are multiple GPUs */
      cerr = hipSetDeviceFlags(hipDeviceMapHost);
      hipGetLastError(); /* Reset the last error */
      if (cerr == hipSuccess) { /* It implies hip runtime has not been initialized */
        ierr  = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
        devId = rank % devCount;
        cerr  = hipSetDevice(devId);CHKERRHIP(cerr);
      } else if (cerr == hipErrorSetOnActiveProcess) {
        /* It means user initialized hip runtime outside of petsc. We respect the device choice. */
        cerr = hipGetDevice(&devId);CHKERRHIP(cerr);
      }
    }
    ierr = PetscHIPInitialize(PETSC_COMM_WORLD,(PetscInt)devId);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
