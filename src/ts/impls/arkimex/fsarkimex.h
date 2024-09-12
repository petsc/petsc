#pragma once
PETSC_INTERN PetscErrorCode TSARKIMEXSetFastSlowSplit_ARKIMEX(TS, PetscBool);
PETSC_INTERN PetscErrorCode TSARKIMEXGetFastSlowSplit_ARKIMEX(TS, PetscBool *);
PETSC_INTERN PetscErrorCode TSHasRHSFunction(TS, PetscBool *);
