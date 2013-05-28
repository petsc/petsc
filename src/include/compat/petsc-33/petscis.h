#include <petsc-private/vecimpl.h>

static PetscClassId PETSC_SECTION_CLASSID = 0;

#define PetscSectionError do {                                          \
    PetscFunctionBegin;                                                 \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version"); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

#undef  __FUNCT__
#define __FUNCT__ "PetscSectionClone"
PetscErrorCode PetscSectionClone(PETSC_UNUSED PetscSection s,...){PetscSectionError;}
#undef  __FUNCT__
#define __FUNCT__ "PetscSectionReset"
PetscErrorCode PetscSectionReset(PETSC_UNUSED PetscSection s,...){PetscSectionError;}

#undef PetscSectionError

