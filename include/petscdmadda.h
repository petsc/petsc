#if !defined(__PETSCDMDA_H)
#define __PETSCDMDA_H

#include <petscdm.h>

PetscErrorCode  DMADDACreate(MPI_Comm,PetscInt,PetscInt*,PetscInt*,PetscInt,PetscBool *,DM*);
PetscErrorCode  DMADDASetParameters(DM,PetscInt,PetscInt*,PetscInt*,PetscInt,PetscBool*);
PetscErrorCode  DMADDASetRefinement(DM, PetscInt *,PetscInt);
PetscErrorCode  DMADDAGetCorners(DM, PetscInt **, PetscInt **);
PetscErrorCode  DMADDAGetGhostCorners(DM, PetscInt **, PetscInt **);
PetscErrorCode  DMADDAGetMatrixNS(DM, DM, MatType , Mat *);

/* functions to set values in vectors and matrices */
struct _ADDAIdx_s {
  PetscInt     *x;               /* the coordinates, user has to make sure it is the correct size! */
  PetscInt     d;                /* indexes the dof */
};
typedef struct _ADDAIdx_s ADDAIdx;

PetscErrorCode  DMADDAMatSetValues(Mat, DM, PetscInt, const ADDAIdx[], DM, PetscInt, const ADDAIdx[], const PetscScalar[], InsertMode);
PetscBool  ADDAHCiterStartup(const PetscInt, const PetscInt *const, const PetscInt *const, PetscInt *const);
PetscBool  ADDAHCiter(const PetscInt, const PetscInt *const, const PetscInt *const, PetscInt *const);

#endif
