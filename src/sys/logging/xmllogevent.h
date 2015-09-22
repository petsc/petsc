#if !defined(__PETSNESTEDLOGEVENTS_H)
#define __PETSNESTEDLOGEVENTS_H
PetscErrorCode PetscLogInitializeNested(void);
PetscErrorCode PetscLogFreeNested(void);
PetscErrorCode PetscLogViewNested(PetscViewer);
PetscErrorCode PetscLogSetThreshold(PetscLogDouble, PetscLogDouble *);
#endif

