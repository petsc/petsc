#include <petsc/private/vecimpl.h> /*I "petscvec.h"  I*/

PetscFunctionList VecList = NULL;

/* compare a vector type against a list of target vector types */
static inline PetscErrorCode VecTypeCompareAny_Private(VecType srcType, PetscBool *match, const char tgtTypes[], ...)
{
  PetscBool flg = PETSC_FALSE;
  va_list   Argp;

  PetscFunctionBegin;
  PetscAssertPointer(match, 2);
  *match = PETSC_FALSE;
  va_start(Argp, tgtTypes);
  while (tgtTypes && tgtTypes[0]) {
    PetscCall(PetscStrcmp(srcType, tgtTypes, &flg));
    if (flg) {
      *match = PETSC_TRUE;
      break;
    }
    tgtTypes = va_arg(Argp, const char *);
  }
  va_end(Argp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PETSC_MAX_VECTYPE_LEN 64

/*@
  VecSetType - Builds a vector, for a particular vector implementation.

  Collective

  Input Parameters:
+ vec     - The vector object
- newType - The name of the vector type

  Options Database Key:
. -vec_type <type> - Sets the vector type; use -help for a list
                     of available types

  Level: intermediate

  Notes:
  See `VecType` for available vector types (for instance, `VECSEQ` or `VECMPI`)
  Changing a vector to a new type will retain its old value if any.

  Use `VecDuplicate()` or `VecDuplicateVecs()` to form additional vectors of the same type as an existing vector.

.seealso: [](ch_vectors), `Vec`, `VecType`, `VecGetType()`, `VecCreate()`, `VecDuplicate()`, `VecDuplicateVecs()`
@*/
PetscErrorCode VecSetType(Vec vec, VecType newType)
{
  PetscErrorCode (*r)(Vec);
  VecType      curType;
  PetscBool    match;
  PetscMPIInt  size;
  PetscBool    dstSeq = PETSC_FALSE; // type info of the new type
  MPI_Comm     comm;
  char         seqType[PETSC_MAX_VECTYPE_LEN] = {0};
  char         mpiType[PETSC_MAX_VECTYPE_LEN] = {0};
  PetscScalar *oldValue;
  PetscBool    srcStandard, dstStandard;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 1);

  PetscCall(VecGetType(vec, &curType));
  if (!curType) goto newvec; // vec's type is not set yet

  /* return if exactly the same type */
  PetscCall(PetscObjectTypeCompare((PetscObject)vec, newType, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  /* error on illegal mpi to seq conversion */
  PetscCall(PetscObjectGetComm((PetscObject)vec, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  PetscCall(PetscStrbeginswith(newType, VECSEQ, &dstSeq));
  PetscCheck(!(size > 1 && dstSeq), comm, PETSC_ERR_ARG_WRONG, "Cannot convert MPI vectors to sequential ones");

  /* return if standard => standard */
  if (size == 1) PetscCall(PetscObjectTypeCompare((PetscObject)vec, VECSEQ, &srcStandard));
  else PetscCall(PetscObjectTypeCompare((PetscObject)vec, VECMPI, &srcStandard));
  PetscCall(VecTypeCompareAny_Private(newType, &dstStandard, VECSTANDARD, VECSEQ, VECMPI, ""));
  if (srcStandard && dstStandard) PetscFunctionReturn(PETSC_SUCCESS);

  /* return if curType = "seq" | "mpi" + newType */
  PetscCall(PetscStrncpy(mpiType, "mpi", 4));
  PetscCall(PetscStrlcat(mpiType, newType, PETSC_MAX_VECTYPE_LEN));
  PetscCall(PetscStrncpy(seqType, "seq", 4));
  PetscCall(PetscStrlcat(seqType, newType, PETSC_MAX_VECTYPE_LEN));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)vec, &match, seqType, mpiType, ""));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  /* downcast VECSTANDARD to VECCUDA/HIP/KOKKOS in place. We don't do in-place upcasting
  for those vectors. At least, it is not always possible to upcast a VECCUDA to VECSTANDARD
  in place, since the host array might be pinned (i.e., allocated by cudaMallocHost()). If
  we upcast it to VECSTANDARD, we could not free the pinned array with PetscFree(), which
  is assumed for VECSTANDARD. Thus we just create a new vector, though it is expensive.
  Upcasting is rare and users are not recommended to use it.
  */
#if defined(PETSC_HAVE_CUDA)
  {
    PetscBool dstCUDA = PETSC_FALSE;
    if (!dstStandard) PetscCall(VecTypeCompareAny_Private(newType, &dstCUDA, VECCUDA, VECSEQCUDA, VECMPICUDA, ""));
    if (srcStandard && dstCUDA) {
      if (size == 1) PetscCall(VecConvert_Seq_SeqCUDA_inplace(vec));
      else PetscCall(VecConvert_MPI_MPICUDA_inplace(vec));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
#endif
#if defined(PETSC_HAVE_HIP)
  {
    PetscBool dstHIP = PETSC_FALSE;
    if (!dstStandard) PetscCall(VecTypeCompareAny_Private(newType, &dstHIP, VECHIP, VECSEQHIP, VECMPIHIP, ""));
    if (srcStandard && dstHIP) {
      if (size == 1) PetscCall(VecConvert_Seq_SeqHIP_inplace(vec));
      else PetscCall(VecConvert_MPI_MPIHIP_inplace(vec));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  {
    PetscBool dstKokkos = PETSC_FALSE;
    if (!dstStandard) PetscCall(VecTypeCompareAny_Private(newType, &dstKokkos, VECKOKKOS, VECSEQKOKKOS, VECMPIKOKKOS, ""));
    if (srcStandard && dstKokkos) {
      if (size == 1) PetscCall(VecConvert_Seq_SeqKokkos_inplace(vec));
      else PetscCall(VecConvert_MPI_MPIKokkos_inplace(vec));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
#endif

  /* Other conversion scenarios: create a new vector but retain old value */
newvec:
  PetscCall(PetscFunctionListFind(VecList, newType, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)vec), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown vector type: %s", newType);
  if (curType) { /* no need to destroy a vec without type */
    const PetscScalar *array;
    PetscCall(VecGetArrayRead(vec, &array));
    if (array) {                                       /* record the old value if any before destroy */
      PetscCall(PetscMalloc1(vec->map->n, &oldValue)); /* no need to free since we'll drop it into vec */
      PetscCall(PetscArraycpy(oldValue, array, vec->map->n));
    } else {
      oldValue = NULL;
    }
    PetscCall(VecRestoreArrayRead(vec, &array));
    PetscTryTypeMethod(vec, destroy);
    PetscCall(PetscMemzero(vec->ops, sizeof(struct _VecOps)));
    PetscCall(PetscFree(vec->defaultrandtype));
    PetscCall(PetscFree(((PetscObject)vec)->type_name)); /* free type_name to make vec clean to use, as we might call VecSetType() again */
  }

  if (vec->map->n < 0 && vec->map->N < 0) {
    vec->ops->create = r;
    vec->ops->load   = VecLoad_Default;
  } else {
    PetscCall((*r)(vec));
  }

  /* drop in the old value */
  if (curType && vec->map->n) PetscCall(VecReplaceArray(vec, oldValue));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  VecGetType - Gets the vector type name (as a string) from a `Vec`.

  Not Collective

  Input Parameter:
. vec - The vector

  Output Parameter:
. type - The `VecType` of the vector

  Level: intermediate

.seealso: [](ch_vectors), `Vec`, `VecType`, `VecCreate()`, `VecDuplicate()`, `VecDuplicateVecs()`
@*/
PetscErrorCode VecGetType(Vec vec, VecType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(VecRegisterAll());
  *type = ((PetscObject)vec)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecGetRootType_Private(Vec vec, VecType *vtype)
{
  PetscBool iscuda, iship, iskokkos, isvcl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 1);
  PetscAssertPointer(vtype, 2);
  PetscCall(PetscObjectTypeCompareAny((PetscObject)vec, &iscuda, VECCUDA, VECMPICUDA, VECSEQCUDA, ""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)vec, &iship, VECHIP, VECMPIHIP, VECSEQHIP, ""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)vec, &iskokkos, VECKOKKOS, VECMPIKOKKOS, VECSEQKOKKOS, ""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)vec, &isvcl, VECVIENNACL, VECMPIVIENNACL, VECSEQVIENNACL, ""));
  if (iscuda) {
    *vtype = VECCUDA;
  } else if (iship) {
    *vtype = VECHIP;
  } else if (iskokkos) {
    *vtype = VECKOKKOS;
  } else if (isvcl) {
    *vtype = VECVIENNACL;
  } else {
    *vtype = VECSTANDARD;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*--------------------------------------------------------------------------------------------------------------------*/

/*@C
  VecRegister -  Adds a new vector component implementation

  Not Collective, No Fortran Support

  Input Parameters:
+ sname    - The name of a new user-defined creation routine
- function - The creation routine

  Notes:
  `VecRegister()` may be called multiple times to add several user-defined vectors

  Example Usage:
.vb
    VecRegister("my_vec",MyVectorCreate);
.ve

  Then, your vector type can be chosen with the procedural interface via
.vb
    VecCreate(MPI_Comm, Vec *);
    VecSetType(Vec,"my_vector_name");
.ve
  or at runtime via the option
.vb
    -vec_type my_vector_name
.ve

  Level: advanced

.seealso: `VecRegisterAll()`, `VecRegisterDestroy()`
@*/
PetscErrorCode VecRegister(const char sname[], PetscErrorCode (*function)(Vec))
{
  PetscFunctionBegin;
  PetscCall(VecInitializePackage());
  PetscCall(PetscFunctionListAdd(&VecList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}
