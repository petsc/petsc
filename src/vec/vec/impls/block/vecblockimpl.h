
#ifndef VecBlock_impl_h
#define VecBlock_impl_h

#include <petsc.h>
#include <petscvec.h>

typedef struct {
  PetscInt  nb;           /* n blocks */
  Vec       *v;
  PetscBool setup_called;
} Vec_Block;

#define VecBlockCheckCompatible2(x,xarg,y,yarg) do {                    \
    PetscValidHeaderSpecific(x,VEC_CLASSID,xarg);                       \
    PetscValidHeaderSpecific(y,VEC_CLASSID,yarg);                       \
    PetscCheckSameComm(x,xarg,y,yarg);                                  \
    if (!((Vec_Block*)x->data)->setup_called) SETERRQ1(((PetscObject)x)->comm,PETSC_ERR_ARG_WRONG,"Block vector argument %D not setup.",xarg); \
    if (!((Vec_Block*)y->data)->setup_called) SETERRQ1(((PetscObject)x)->comm,PETSC_ERR_ARG_WRONG,"Block vector argument %D not setup.",yarg); \
    if (((Vec_Block*)x->data)->nb != ((Vec_Block*)y->data)->nb)         \
      SETERRQ2(((PetscObject)x)->comm,PETSC_ERR_ARG_WRONG,"Block vector arguments %D and %D have different numbers of blocks.",xarg,yarg); \
  } while (0)

#define VecBlockCheckCompatible3(x,xarg,y,yarg,z,zarg) do {             \
    PetscValidHeaderSpecific(x,VEC_CLASSID,xarg);                       \
    PetscValidHeaderSpecific(y,VEC_CLASSID,yarg);                       \
    PetscValidHeaderSpecific(z,VEC_CLASSID,zarg);                       \
    PetscCheckSameComm(x,xarg,y,yarg);                                  \
    PetscCheckSameComm(x,xarg,z,zarg);                                  \
    if (!((Vec_Block*)x->data)->setup_called) SETERRQ1(((PetscObject)w)->comm,PETSC_ERR_ARG_WRONG,"Block vector argument %D not setup.",xarg); \
    if (!((Vec_Block*)y->data)->setup_called) SETERRQ1(((PetscObject)w)->comm,PETSC_ERR_ARG_WRONG,"Block vector argument %D not setup.",yarg); \
    if (!((Vec_Block*)z->data)->setup_called) SETERRQ1(((PetscObject)w)->comm,PETSC_ERR_ARG_WRONG,"Block vector argument %D not setup.",zarg); \
    if (((Vec_Block*)x->data)->nb != ((Vec_Block*)y->data)->nb)         \
      SETERRQ2(((PetscObject)w)->comm,PETSC_ERR_ARG_WRONG,"Block vectors arguments %D and %D have different numbers of blocks.",xarg,yarg); \
    if (((Vec_Block*)x->data)->nb != ((Vec_Block*)z->data)->nb)         \
      SETERRQ2(((PetscObject)w)->comm,PETSC_ERR_ARG_WRONG,"Block vectors arguments %D and %D have different numbers of blocks.",xarg,zarg); \
  } while (0)

#endif
