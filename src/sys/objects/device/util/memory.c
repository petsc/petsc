#include <petsc/private/deviceimpl.h> /*I <petscdevice.h> I*/
#include <petscdevice_cupm.h>

PetscErrorCode PetscDeviceGetAllocationAlignment_Private(size_t requested, size_t type_alignment, PetscBool host_fallback, size_t *alignment)
{
  PetscFunctionBegin;
  PetscAssertPointer(alignment, 4);
  PetscCheck(requested > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Alignment must be PETSC_DECIDE or a positive power of two, got %zu", requested);
  PetscCheck(requested == (size_t)PETSC_DECIDE || !(requested & (requested - 1)), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Alignment %zu must be a power of two", requested);
  PetscCheck(type_alignment && !(type_alignment & (type_alignment - 1)), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Type alignment %zu must be a positive power of two", type_alignment);
  if (host_fallback) {
    PetscCheck(requested == (size_t)PETSC_DECIDE || !(PETSC_MEMALIGN % requested), PETSC_COMM_SELF, PETSC_ERR_SUP, "Host allocation fallback guarantees %d-byte alignment; requested %zu-byte alignment is not supported", PETSC_MEMALIGN, requested);
    *alignment = PETSC_MEMALIGN;
  } else *alignment = requested == (size_t)PETSC_DECIDE ? type_alignment : PetscMax(requested, type_alignment);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// REVIEW ME: this should probably return PETSC_MEMTYPE_CUDA and PETSC_MEMTYPE_HIP

/*@
  PetscGetMemType - Query the `PetscMemType` of a pointer

  Not Collective, No Fortran Support

  Input Parameter:
. ptr - The pointer to query (may be `NULL`)

  Output Parameter:
. type - The `PetscMemType` of the pointer

  Level: beginner

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
    cerr = cudaPointerGetAttributes(&attr, ptr);
    if (cerr) cerr = cudaGetLastError(); /* If there was an error, return it and then reset it */
    mtype = attr.type;
    // CUDA managed memory is device accessible, so we treat it as device memory?
    if (cerr == cudaSuccess && (mtype == cudaMemoryTypeDevice || mtype == cudaMemoryTypeManaged)) *type = PETSC_MEMTYPE_DEVICE;
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
    if (cerr == hipSuccess && (mtype == hipMemoryTypeDevice || mtype == hipMemoryTypeManaged)) *type = PETSC_MEMTYPE_DEVICE;
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
