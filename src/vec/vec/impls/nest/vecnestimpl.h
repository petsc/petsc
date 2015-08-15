
#if !defined(VecNest_impl_h)
#define VecNest_impl_h

#include <petsc/private/vecimpl.h>

typedef struct {
  PetscInt  nb;           /* n blocks */
  Vec       *v;
  IS        *is;
  PetscBool setup_called;
} Vec_Nest;

#define VecNestCheckCompatible2(x,xarg,y,yarg) do {                    \
    PetscValidHeaderSpecific(x,VEC_CLASSID,xarg);                       \
    PetscValidHeaderSpecific(y,VEC_CLASSID,yarg);                       \
    PetscCheckSameComm(x,xarg,y,yarg);                                  \
    if (!((Vec_Nest*)x->data)->setup_called) SETERRQ1(PetscObjectComm((PetscObject)x),PETSC_ERR_ARG_WRONG,"Nest vector argument %D not setup.",xarg); \
    if (!((Vec_Nest*)y->data)->setup_called) SETERRQ1(PetscObjectComm((PetscObject)x),PETSC_ERR_ARG_WRONG,"Nest vector argument %D not setup.",yarg); \
    if (((Vec_Nest*)x->data)->nb != ((Vec_Nest*)y->data)->nb) SETERRQ2(PetscObjectComm((PetscObject)x),PETSC_ERR_ARG_WRONG,"Nest vector arguments %D and %D have different numbers of blocks.",xarg,yarg); \
  } while (0)

#define VecNestCheckCompatible3(x,xarg,y,yarg,z,zarg) do {             \
    PetscValidHeaderSpecific(x,VEC_CLASSID,xarg);                       \
    PetscValidHeaderSpecific(y,VEC_CLASSID,yarg);                       \
    PetscValidHeaderSpecific(z,VEC_CLASSID,zarg);                       \
    PetscCheckSameComm(x,xarg,y,yarg);                                  \
    PetscCheckSameComm(x,xarg,z,zarg);                                  \
    if (!((Vec_Nest*)x->data)->setup_called) SETERRQ1(PetscObjectComm((PetscObject)w),PETSC_ERR_ARG_WRONG,"Nest vector argument %D not setup.",xarg); \
    if (!((Vec_Nest*)y->data)->setup_called) SETERRQ1(PetscObjectComm((PetscObject)w),PETSC_ERR_ARG_WRONG,"Nest vector argument %D not setup.",yarg); \
    if (!((Vec_Nest*)z->data)->setup_called) SETERRQ1(PetscObjectComm((PetscObject)w),PETSC_ERR_ARG_WRONG,"Nest vector argument %D not setup.",zarg); \
    if (((Vec_Nest*)x->data)->nb != ((Vec_Nest*)y->data)->nb) SETERRQ2(PetscObjectComm((PetscObject)w),PETSC_ERR_ARG_WRONG,"Nest vector arguments %D and %D have different numbers of blocks.",xarg,yarg); \
    if (((Vec_Nest*)x->data)->nb != ((Vec_Nest*)z->data)->nb) SETERRQ2(PetscObjectComm((PetscObject)w),PETSC_ERR_ARG_WRONG,"Nest vector arguments %D and %D have different numbers of blocks.",xarg,zarg); \
  } while (0)

#endif
