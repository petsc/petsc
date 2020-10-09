/*
 Management of CUBLAS and CUSOLVER handles
 */

#include <petscsys.h>
#include <petsc/private/petscimpl.h>
#include <petsccublas.h>

static cublasHandle_t     cublasv2handle   = NULL;
static cusolverDnHandle_t cusolverdnhandle = NULL;

/*
   Destroys the CUBLAS handle.
   This function is intended and registered for PetscFinalize - do not call manually!
 */
static PetscErrorCode PetscCUBLASDestroyHandle()
{
  cublasStatus_t cberr;

  PetscFunctionBegin;
  if (cublasv2handle) {
    cberr          = cublasDestroy(cublasv2handle);CHKERRCUBLAS(cberr);
    cublasv2handle = NULL;  /* Ensures proper reinitialization */
  }
  PetscFunctionReturn(0);
}

/*
    Initializing the cuBLAS handle can take 1/2 a second therefore
    initialize in PetscInitialize() before being timing so it does
    not distort the -log_view information
*/
PetscErrorCode PetscCUBLASInitializeHandle(void)
{
  PetscErrorCode ierr;
  cublasStatus_t cberr;

  PetscFunctionBegin;
  if (!cublasv2handle) {
    for (int i=0; i<3; i++) {
      cberr = cublasCreate(&cublasv2handle);
      if (cberr == CUBLAS_STATUS_SUCCESS) break;
      if (cberr != CUBLAS_STATUS_ALLOC_FAILED && cberr != CUBLAS_STATUS_NOT_INITIALIZED) CHKERRCUBLAS(cberr);
      if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
    }
    if (cberr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuBLAS");

    /* Make sure that the handle will be destroyed properly */
    ierr = PetscRegisterFinalize(PetscCUBLASDestroyHandle);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCUBLASGetHandle(cublasHandle_t *handle)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  if (!cublasv2handle) {ierr = PetscCUBLASInitializeHandle();CHKERRQ(ierr);}
  *handle = cublasv2handle;
  PetscFunctionReturn(0);
}

/* cusolver */
static PetscErrorCode PetscCUSOLVERDnDestroyHandle()
{
  cusolverStatus_t  cerr;

  PetscFunctionBegin;
  if (cusolverdnhandle) {
    cerr             = cusolverDnDestroy(cusolverdnhandle);CHKERRCUSOLVER(cerr);
    cusolverdnhandle = NULL;  /* Ensures proper reinitialization */
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCUSOLVERDnInitializeHandle(void)
{
  PetscErrorCode    ierr;
  cusolverStatus_t  cerr;

  PetscFunctionBegin;
  if (!cusolverdnhandle) {
    for (int i=0; i<3; i++) {
      cerr = cusolverDnCreate(&cusolverdnhandle);
      if (cerr == CUSOLVER_STATUS_SUCCESS) break;
      if (cerr != CUSOLVER_STATUS_ALLOC_FAILED) CHKERRCUSOLVER(cerr);
      if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
    }
    if (cerr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuSPARSE");
    ierr = PetscRegisterFinalize(PetscCUSOLVERDnDestroyHandle);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCUSOLVERDnGetHandle(cusolverDnHandle_t *handle)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  if (!cusolverdnhandle) {ierr = PetscCUSOLVERDnInitializeHandle();CHKERRQ(ierr);}
  *handle = cusolverdnhandle;
  PetscFunctionReturn(0);
}

