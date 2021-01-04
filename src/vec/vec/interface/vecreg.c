
#include <petsc/private/vecimpl.h>    /*I "petscvec.h"  I*/

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

.seealso: VecGetType(), VecCreate()
@*/
PetscErrorCode VecSetType(Vec vec, VecType method)
{
  PetscErrorCode (*r)(Vec);
  PetscBool      match;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject) vec, method, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  /* Return if asked for VECSTANDARD and Vec is already VECSEQ on 1 process or VECMPI on more.
     Otherwise, we free the Vec array in the call to destroy below and never reallocate it,
     since the VecType will be the same and VecSetType(v,VECSEQ) will return when called from VecCreate_Standard */
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)vec),&size);CHKERRMPI(ierr);
  ierr = PetscStrcmp(method,VECSTANDARD,&match);CHKERRQ(ierr);
  if (match) {

    ierr = PetscObjectTypeCompare((PetscObject) vec, size > 1 ? VECMPI : VECSEQ, &match);CHKERRQ(ierr);
    if (match) PetscFunctionReturn(0);
  }
  /* same reasons for VECCUDA and VECVIENNACL */
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscStrcmp(method,VECCUDA,&match);CHKERRQ(ierr);
  if (match) {
    ierr = PetscObjectTypeCompare((PetscObject) vec, size > 1 ? VECMPICUDA : VECSEQCUDA, &match);CHKERRQ(ierr);
    if (match) PetscFunctionReturn(0);
  }
#endif
#if defined(PETSC_HAVE_HIP)
  ierr = PetscStrcmp(method,VECHIP,&match);CHKERRQ(ierr);
  if (match) {
    ierr = PetscObjectTypeCompare((PetscObject) vec, size > 1 ? VECMPIHIP : VECSEQHIP, &match);CHKERRQ(ierr);
    if (match) PetscFunctionReturn(0);
  }
#endif
#if defined(PETSC_HAVE_VIENNACL)
  ierr = PetscStrcmp(method,VECVIENNACL,&match);CHKERRQ(ierr);
  if (match) {
    ierr = PetscObjectTypeCompare((PetscObject) vec, size > 1 ? VECMPIVIENNACL : VECSEQVIENNACL, &match);CHKERRQ(ierr);
    if (match) PetscFunctionReturn(0);
  }
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  ierr = PetscStrcmp(method,VECKOKKOS,&match);CHKERRQ(ierr);
  if (match) {
    ierr = PetscObjectTypeCompare((PetscObject) vec, size > 1 ? VECMPIKOKKOS : VECSEQKOKKOS, &match);CHKERRQ(ierr);
    if (match) PetscFunctionReturn(0);
  }
#endif
  ierr = PetscFunctionListFind(VecList,method,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown vector type: %s", method);
  if (vec->ops->destroy) {
    ierr = (*vec->ops->destroy)(vec);CHKERRQ(ierr);
    vec->ops->destroy = NULL;
  }
  ierr = PetscMemzero(vec->ops,sizeof(struct _VecOps));CHKERRQ(ierr);
  ierr = PetscFree(vec->defaultrandtype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(PETSCRANDER48,&vec->defaultrandtype);CHKERRQ(ierr);
  if (vec->map->n < 0 && vec->map->N < 0) {
    vec->ops->create = r;
    vec->ops->load   = VecLoad_Default;
  } else {
    ierr = (*r)(vec);CHKERRQ(ierr);
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

.seealso: VecSetType(), VecCreate()
@*/
PetscErrorCode VecGetType(Vec vec, VecType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID,1);
  PetscValidPointer(type,2);
  ierr = VecRegisterAll();CHKERRQ(ierr);
  *type = ((PetscObject)vec)->type_name;
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

.seealso: VecRegisterAll(), VecRegisterDestroy()
@*/
PetscErrorCode VecRegister(const char sname[], PetscErrorCode (*function)(Vec))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&VecList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
