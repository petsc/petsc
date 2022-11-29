#include <petsc/private/deviceimpl.h> /*I <petscdevice.h> I*/
#include <petscdevice_cupm.h>

// REVIEW ME: this should probably return PETSC_MEMTYPE_CUDA and PETSC_MEMTYPE_HIP

/*@C
  PetscGetMemType - Query the `PetscMemType` of a pointer

  Not Collective

  Input Parameter:
. ptr - The pointer to query (may be `NULL`)

  Output Parameter:
. type - The `PetscMemType` of the pointer

  Notes:
  Currently only CUDA and HIP memtypes are supported.

  Level: intermediate

.seelso: `PetscMemType`, `PetscDeviceMalloc()`, `PetscDeviceCalloc()`, `PetscDeviceFree()`,
`PetscDeviceArrayCopy()`, `PetscDeviceArrayZero()`
@*/
PetscErrorCode PetscGetMemType(const void *ptr, PetscMemType *type)
{
  PetscFunctionBegin;
  PetscValidPointer(type, 2);
  *type = PETSC_MEMTYPE_HOST;
  if (!ptr) PetscFunctionReturn(0);
#if PetscDefined(HAVE_CUDA)
  if (PetscDeviceInitialized(PETSC_DEVICE_CUDA)) {
    cudaError_t                  cerr;
    struct cudaPointerAttributes attr;
    enum cudaMemoryType          mtype;
    cerr = cudaPointerGetAttributes(&attr, ptr); /* Do not check error since before CUDA 11.0, passing a host pointer returns cudaErrorInvalidValue */
    if (cerr) cerr = cudaGetLastError();         /* If there was an error, return it and then reset it */
  #if (CUDART_VERSION < 10000)
    mtype = attr.memoryType;
  #else
    mtype = attr.type;
  #endif
    if (cerr == cudaSuccess && mtype == cudaMemoryTypeDevice) *type = PETSC_MEMTYPE_DEVICE;
    PetscFunctionReturn(0);
  }
#endif

#if PetscDefined(HAVE_HIP)
  if (PetscDeviceInitialized(PETSC_DEVICE_HIP)) {
    hipError_t                   cerr;
    struct hipPointerAttribute_t attr;
    enum hipMemoryType           mtype;
    cerr = hipPointerGetAttributes(&attr, ptr);
    if (cerr) cerr = hipGetLastError();
    mtype = attr.memoryType;
    if (cerr == hipSuccess && mtype == hipMemoryTypeDevice) *type = PETSC_MEMTYPE_DEVICE;
  }
#endif
  PetscFunctionReturn(0);
}
