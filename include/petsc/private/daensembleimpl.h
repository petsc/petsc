#pragma once
PETSC_INTERN PetscErrorCode PetscDACreate_Ensemble(PetscDA);
PETSC_INTERN PetscErrorCode PetscDAView_Ensemble(PetscDA, PetscViewer);
PETSC_INTERN PetscErrorCode PetscDASetFromOptions_Ensemble(PetscDA, PetscOptionItems *);
PETSC_INTERN PetscErrorCode PetscDASetUp_Ensemble(PetscDA);
PETSC_INTERN PetscErrorCode PetscDAEnsembleForecast_Ensemble(PetscDA, PetscErrorCode (*)(Vec, Vec, PetscCtx), PetscCtx);
PETSC_INTERN PetscErrorCode PetscDADestroy_Ensemble(PetscDA);
extern PetscLogEvent        PetscDA_Analysis;
