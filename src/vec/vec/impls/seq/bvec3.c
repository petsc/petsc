
/*
   Implements the sequential vectors.
*/

#include <../src/vec/vec/impls/dvecimpl.h>          /*I "petscvec.h" I*/
/*MC
   VECSEQ - VECSEQ = "seq" - The basic sequential vector

   Options Database Keys:
. -vec_type seq - sets the vector type to VECSEQ during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateSeqWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateSeq()
M*/

#if defined(PETSC_USE_MIXED_PRECISION)
extern PetscErrorCode VecCreate_Seq_Private(Vec,const float*);
extern PetscErrorCode VecCreate_Seq_Private(Vec,const double*);
#endif

PETSC_EXTERN PetscErrorCode VecCreate_Seq(Vec V)
{
  Vec_Seq        *s;
  PetscScalar    *array;
  PetscErrorCode ierr;
  PetscInt       n = PetscMax(V->map->n,V->map->N);
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)V),&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQ on more than one process");
#if !defined(PETSC_USE_MIXED_PRECISION)
  ierr = PetscMalloc1(n,&array);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)V, n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecCreate_Seq_Private(V,array);CHKERRQ(ierr);

  s                  = (Vec_Seq*)V->data;
  s->array_allocated = array;

  ierr = VecSet(V,0.0);CHKERRQ(ierr);
#else
  switch (((PetscObject)V)->precision) {
  case PETSC_PRECISION_SINGLE: {
    float *aarray;

    ierr = PetscCalloc1(n,&aarray);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)V, n*sizeof(float));CHKERRQ(ierr);
    ierr = VecCreate_Seq_Private(V,aarray);CHKERRQ(ierr);

    s                  = (Vec_Seq*)V->data;
    s->array_allocated = (PetscScalar*)aarray;
  } break;
  case PETSC_PRECISION_DOUBLE: {
    double *aarray;

    ierr = PetscCalloc1(n,&aarray);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)V, n*sizeof(double));CHKERRQ(ierr);
    ierr = VecCreate_Seq_Private(V,aarray);CHKERRQ(ierr);

    s                  = (Vec_Seq*)V->data;
    s->array_allocated = (PetscScalar*)aarray;
  } break;
  default: SETERRQ1(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"No support for mixed precision %d",(int)(((PetscObject)V)->precision));
  }
#endif
  PetscFunctionReturn(0);
}
