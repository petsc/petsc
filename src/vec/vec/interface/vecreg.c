
#include <petsc/private/vecimpl.h> /*I "petscvec.h"  I*/

PetscFunctionList VecList              = NULL;
PetscBool         VecRegisterAllCalled = PETSC_FALSE;

/*@C
  VecSetType - Builds a vector, for a particular vector implementation.

  Collective on Vec

  Input Parameters:
+ vec    - The vector object
- method - The name of the vector type

  Options Database Key:
. -vec_type <type> - Sets the vector type; use -help for a list
                     of available types

  Notes:
  See "petsc/include/petscvec.h" for available vector types (for instance, VECSEQ, VECMPI, or VECSHARED).

  Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the same type as an existing vector.

  Level: intermediate

.seealso: `VecGetType()`, `VecCreate()`
@*/
PetscErrorCode VecSetType(Vec vec, VecType method)
{
  PetscErrorCode (*r)(Vec);
  PetscBool   match;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)vec, method, &match));
  if (match) PetscFunctionReturn(0);

  /* Return if asked for VECSTANDARD and Vec is already VECSEQ on 1 process or VECMPI on more.
     Otherwise, we free the Vec array in the call to destroy below and never reallocate it,
     since the VecType will be the same and VecSetType(v,VECSEQ) will return when called from VecCreate_Standard */
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)vec), &size));
  PetscCall(PetscStrcmp(method, VECSTANDARD, &match));
  if (match) {
    PetscCall(PetscObjectTypeCompare((PetscObject)vec, size > 1 ? VECMPI : VECSEQ, &match));
    if (match) PetscFunctionReturn(0);
  }
  /* same reasons for VECCUDA and VECVIENNACL */
#if defined(PETSC_HAVE_CUDA)
  PetscCall(PetscStrcmp(method, VECCUDA, &match));
  if (match) {
    PetscCall(PetscObjectTypeCompare((PetscObject)vec, size > 1 ? VECMPICUDA : VECSEQCUDA, &match));
    if (match) PetscFunctionReturn(0);
  }
#endif
#if defined(PETSC_HAVE_HIP)
  PetscCall(PetscStrcmp(method, VECHIP, &match));
  if (match) {
    PetscCall(PetscObjectTypeCompare((PetscObject)vec, size > 1 ? VECMPIHIP : VECSEQHIP, &match));
    if (match) PetscFunctionReturn(0);
  }
#endif
#if defined(PETSC_HAVE_VIENNACL)
  PetscCall(PetscStrcmp(method, VECVIENNACL, &match));
  if (match) {
    PetscCall(PetscObjectTypeCompare((PetscObject)vec, size > 1 ? VECMPIVIENNACL : VECSEQVIENNACL, &match));
    if (match) PetscFunctionReturn(0);
  }
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscCall(PetscStrcmp(method, VECKOKKOS, &match));
  if (match) {
    PetscCall(PetscObjectTypeCompare((PetscObject)vec, size > 1 ? VECMPIKOKKOS : VECSEQKOKKOS, &match));
    if (match) PetscFunctionReturn(0);
  }
#endif
  PetscCall(PetscFunctionListFind(VecList, method, &r));
  PetscCheck(r, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown vector type: %s", method);
  PetscTryTypeMethod(vec, destroy);
  vec->ops->destroy = NULL;
  PetscCall(PetscMemzero(vec->ops, sizeof(struct _VecOps)));
  PetscCall(PetscFree(vec->defaultrandtype));
  PetscCall(PetscStrallocpy(PETSCRANDER48, &vec->defaultrandtype));
  if (vec->map->n < 0 && vec->map->N < 0) {
    vec->ops->create = r;
    vec->ops->load   = VecLoad_Default;
  } else {
    PetscCall((*r)(vec));
  }
  PetscFunctionReturn(0);
}

/*@C
  VecGetType - Gets the vector type name (as a string) from the Vec.

  Not Collective

  Input Parameter:
. vec  - The vector

  Output Parameter:
. type - The vector type name

  Level: intermediate

.seealso: `VecSetType()`, `VecCreate()`
@*/
PetscErrorCode VecGetType(Vec vec, VecType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 1);
  PetscValidPointer(type, 2);
  PetscCall(VecRegisterAll());
  *type = ((PetscObject)vec)->type_name;
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetRootType_Private(Vec vec, VecType *vtype)
{
  PetscBool iscuda, iship, iskokkos, isvcl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 1);
  PetscValidPointer(vtype, 2);
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
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------------------------------------------------*/

/*@C
  VecRegister -  Adds a new vector component implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  VecRegister() may be called multiple times to add several user-defined vectors

  Sample usage:
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
  PetscFunctionReturn(0);
}
