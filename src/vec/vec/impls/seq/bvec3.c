
/*
   Implements the sequential vectors.
*/

#include <../src/vec/vec/impls/dvecimpl.h> /*I "petscvec.h" I*/
/*MC
   VECSEQ - VECSEQ = "seq" - The basic sequential vector

   Options Database Keys:
. -vec_type seq - sets the vector type to VECSEQ during a call to VecSetFromOptions()

  Level: beginner

.seealso: `VecCreate()`, `VecSetType()`, `VecSetFromOptions()`, `VecCreateSeqWithArray()`, `VECMPI`, `VecType`, `VecCreateMPI()`, `VecCreateSeq()`
M*/

#if defined(PETSC_USE_MIXED_PRECISION)
extern PetscErrorCode VecCreate_Seq_Private(Vec, const float *);
extern PetscErrorCode VecCreate_Seq_Private(Vec, const double *);
#endif

PetscErrorCode VecCreate_Seq(Vec V)
{
  Vec_Seq     *s;
  PetscScalar *array;
  PetscInt     n = PetscMax(V->map->n, V->map->N);
  PetscMPIInt  size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)V), &size));
  PetscCheck(size <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot create VECSEQ on more than one process");
#if !defined(PETSC_USE_MIXED_PRECISION)
  PetscCall(PetscCalloc1(n, &array));
  PetscCall(VecCreate_Seq_Private(V, array));

  s                  = (Vec_Seq *)V->data;
  s->array_allocated = array;
#else
  switch (((PetscObject)V)->precision) {
  case PETSC_PRECISION_SINGLE: {
    float *aarray;

    PetscCall(PetscCalloc1(n, &aarray));
    PetscCall(VecCreate_Seq_Private(V, aarray));

    s                  = (Vec_Seq *)V->data;
    s->array_allocated = (PetscScalar *)aarray;
  } break;
  case PETSC_PRECISION_DOUBLE: {
    double *aarray;

    PetscCall(PetscCalloc1(n, &aarray));
    PetscCall(VecCreate_Seq_Private(V, aarray));

    s                  = (Vec_Seq *)V->data;
    s->array_allocated = (PetscScalar *)aarray;
  } break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)V), PETSC_ERR_SUP, "No support for mixed precision %d", (int)(((PetscObject)V)->precision));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
