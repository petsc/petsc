#include <petsc/private/deviceimpl.h> /*I <petscdevice.h> I*/
#include <petscdevice_cupm.h>

// REVIEW ME: this should probably return PETSC_MEMTYPE_CUDA and PETSC_MEMTYPE_HIP

/*@C
  PetscGetMemType - Query the `PetscMemType` of a pointer

  Not Collective, No Fortran Support

  Input Parameter:
. ptr - The pointer to query (may be `NULL`)

  Output Parameter:
. type - The `PetscMemType` of the pointer

  Level: intermediate

  Notes:
  Currently only CUDA and HIP memtypes are supported.

  The CUDA and HIP calls needed to determine the `PetscMemType` take a non-trivial amount of time, thus for optimal GPU performance this
  routine should be used sparingly and instead the code should track the `PetscMemType` for its important arrays.

.seealso: `PetscMemType`, `PetscDeviceMalloc()`, `PetscDeviceCalloc()`, `PetscDeviceFree()`,
`PetscDeviceArrayCopy()`, `PetscDeviceArrayZero()`
@*/
PetscErrorCode PetscGetMemType(const void *ptr, PetscMemType *type)
{
  PetscFunctionBegin;
  PetscAssertPointer(type, 2);
  *type = PETSC_MEMTYPE_HOST;
  if (!ptr) PetscFunctionReturn(PETSC_SUCCESS);
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
    PetscFunctionReturn(PETSC_SUCCESS);
  }
#endif

#if PetscDefined(HAVE_HIP)
  if (PetscDeviceInitialized(PETSC_DEVICE_HIP)) {
    hipError_t                   cerr;
    struct hipPointerAttribute_t attr;
    enum hipMemoryType           mtype;
    cerr = hipPointerGetAttributes(&attr, ptr);
    if (cerr) cerr = hipGetLastError();
  #if PETSC_PKG_HIP_VERSION_GE(5, 5, 0)
    mtype = attr.type;
  #else
    mtype = attr.memoryType;
  #endif
    if (cerr == hipSuccess && mtype == hipMemoryTypeDevice) *type = PETSC_MEMTYPE_DEVICE;
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
