#ifndef PETSC_VECNESTIMPL_H
#define PETSC_VECNESTIMPL_H

#include <petsc/private/vecimpl.h>

typedef struct {
  PetscInt  nb; /* n blocks */
  Vec      *v;
  IS       *is;
  PetscBool setup_called;
} Vec_Nest;

#if !defined(PETSC_CLANG_STATIC_ANALYZER)
  #define VecNestCheckCompatible2(x, xarg, y, yarg) \
    do { \
      PetscValidHeaderSpecific(x, VEC_CLASSID, xarg); \
      PetscValidHeaderSpecific(y, VEC_CLASSID, yarg); \
      PetscCheckSameComm(x, xarg, y, yarg); \
      PetscCheckSameType(x, xarg, y, yarg); \
      PetscCheck(((Vec_Nest *)(x)->data)->setup_called, PetscObjectComm((PetscObject)x), PETSC_ERR_ARG_WRONG, "Nest vector argument %d not setup.", xarg); \
      PetscCheck(((Vec_Nest *)(y)->data)->setup_called, PetscObjectComm((PetscObject)y), PETSC_ERR_ARG_WRONG, "Nest vector argument %d not setup.", yarg); \
      PetscCheck(((Vec_Nest *)(x)->data)->nb == ((Vec_Nest *)(y)->data)->nb, PetscObjectComm((PetscObject)(x)), PETSC_ERR_ARG_WRONG, "Nest vector arguments %d and %d have different numbers of blocks.", xarg, yarg); \
    } while (0)

  #define VecNestCheckCompatible3(x, xarg, y, yarg, z, zarg) \
    do { \
      PetscValidHeaderSpecific(x, VEC_CLASSID, xarg); \
      PetscValidHeaderSpecific(y, VEC_CLASSID, yarg); \
      PetscValidHeaderSpecific(z, VEC_CLASSID, zarg); \
      PetscCheckSameComm(x, xarg, y, yarg); \
      PetscCheckSameType(x, xarg, y, yarg); \
      PetscCheckSameComm(x, xarg, z, zarg); \
      PetscCheckSameType(x, xarg, z, zarg); \
      PetscCheck(((Vec_Nest *)(x)->data)->setup_called, PetscObjectComm((PetscObject)(x)), PETSC_ERR_ARG_WRONG, "Nest vector argument %d not setup.", xarg); \
      PetscCheck(((Vec_Nest *)(y)->data)->setup_called, PetscObjectComm((PetscObject)(y)), PETSC_ERR_ARG_WRONG, "Nest vector argument %d not setup.", yarg); \
      PetscCheck(((Vec_Nest *)(z)->data)->setup_called, PetscObjectComm((PetscObject)(z)), PETSC_ERR_ARG_WRONG, "Nest vector argument %d not setup.", zarg); \
      PetscCheck(((Vec_Nest *)(x)->data)->nb == ((Vec_Nest *)(y)->data)->nb, PetscObjectComm((PetscObject)(x)), PETSC_ERR_ARG_WRONG, "Nest vector arguments %d and %d have different numbers of blocks.", xarg, yarg); \
      PetscCheck(((Vec_Nest *)(x)->data)->nb == ((Vec_Nest *)(z)->data)->nb, PetscObjectComm((PetscObject)(x)), PETSC_ERR_ARG_WRONG, "Nest vector arguments %d and %d have different numbers of blocks.", xarg, zarg); \
    } while (0)
#else
template <typename Tv>
void VecNestCheckCompatible2(Tv, int, Tv, int);
template <typename Tv>
void VecNestCheckCompatible3(Tv, int, Tv, int, Tv, int);
#endif

#endif // PETSC_VECNESTIMPL_H
