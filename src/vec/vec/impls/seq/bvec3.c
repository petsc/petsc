
/*
   Implements the sequential vectors.
*/

#include <petsc-private/vecimpl.h>          /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petscthreadcomm.h>
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

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecCreate_Seq"
PetscErrorCode  VecCreate_Seq(Vec V)
{
  Vec_Seq        *s;
  PetscScalar    *array;
  PetscErrorCode ierr;
  PetscInt       n = PetscMax(V->map->n,V->map->N);
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)V)->comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQ on more than one process");
#if !defined(PETSC_USE_MIXED_PRECISION)
  ierr = PetscMalloc(n*sizeof(PetscScalar),&array);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(V, n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecCreate_Seq_Private(V,array);CHKERRQ(ierr);
  s    = (Vec_Seq*)V->data;
  s->array_allocated = array;
  ierr = VecSet(V,0.0);CHKERRQ(ierr);
#else
  if (((PetscObject)V)->precision == PETSC_PRECISION_SINGLE) {
    float *aarray;
    ierr = PetscMalloc(n*sizeof(float),&aarray);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(V, n*sizeof(float));CHKERRQ(ierr);
    ierr = PetscMemzero(aarray,n*sizeof(float));CHKERRQ(ierr);
    ierr = VecCreate_Seq_Private(V,aarray);CHKERRQ(ierr);
    s    = (Vec_Seq*)V->data;
    s->array_allocated = (PetscScalar*)aarray;
  } else {
    double *aarray;
    ierr = PetscMalloc(n*sizeof(double),&aarray);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(V, n*sizeof(double));CHKERRQ(ierr);
    ierr = PetscMemzero(aarray,n*sizeof(double));CHKERRQ(ierr);
    ierr = VecCreate_Seq_Private(V,aarray);CHKERRQ(ierr);
    s    = (Vec_Seq*)V->data;
    s->array_allocated = (PetscScalar*)aarray;
  }
#endif
  PetscFunctionReturn(0);
}
EXTERN_C_END
