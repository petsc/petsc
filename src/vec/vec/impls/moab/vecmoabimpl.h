#if !defined(VecMoab_impl_h)
#define VecMoab_impl_h

#include <petsc-private/vecimpl.h>

#include <moab/Core.hpp>
#include <moab/ParallelComm.hpp>

typedef struct {
  moab::Interface    *mbint;
  moab::ParallelComm *pcomm;
  moab::Range         tag_range; /* entities to which this tag applies */
  moab::Tag           tag;
  moab::Tag           ltog_tag;
  PetscInt            tag_size;
  PetscBool           new_tag;
  PetscBool           serial;

} Vec_MOAB;

PETSC_EXTERN PetscErrorCode VecMoabCreateFromTag(moab::Interface *mbint, moab::ParallelComm *pcomm, moab::Tag tag,moab::Tag ltog_tag,moab::Range range,PetscBool serial, PetscBool destroy_tag,Vec *X);
PETSC_EXTERN PetscErrorCode VecMoabCreate(moab::Interface *mbint,moab::ParallelComm *pcomm,PetscInt tag_size,moab::Tag ltog_tag,moab::Range range,PetscBool serial,PetscBool destroy_tag,Vec *vec);
PETSC_EXTERN PetscErrorCode VecMoab_Duplicate(Vec x,Vec *y);
PETSC_EXTERN PetscErrorCode VecMoabGetTag(Vec X, moab::Tag *tag_handle);
PETSC_EXTERN PetscErrorCode VecMoabGetRange(Vec X, moab::Range *range);
PETSC_EXTERN PetscErrorCode VecMoabDestroy_Private(void *ctx);
PETSC_EXTERN PetscErrorCode VecMoabGetTagName_Private(const moab::ParallelComm *pcomm,std::string& tag_name);
#endif
