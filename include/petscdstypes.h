#if !defined(PETSCDSTYPES_H)
#define PETSCDSTYPES_H

#include <petscdmlabel.h>

/*S
  PetscDS - PETSc object that manages a discrete system, which is a set of discretizations + continuum equations from a PetscWeakForm

  Level: intermediate

.seealso: PetscDSCreate(), PetscDSSetType(), PetscDSType, PetscWeakForm, PetscFECreate(), PetscFVCreate()
S*/
typedef struct _p_PetscDS *PetscDS;

/*S
  PetscWeakForm - PETSc object that manages a sets of pointwise functions defining a system of equations

  Level: intermediate

.seealso: PetscWeakFormCreate(), PetscDS, PetscFECreate(), PetscFVCreate()
S*/
typedef struct _p_PetscWeakForm *PetscWeakForm;

typedef struct _PetscHashFormKey
{
  DMLabel  label;
  PetscInt value;
  PetscInt field;
} PetscHashFormKey;

#endif
