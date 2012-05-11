/*
   Private data structure used by the BCGS method. This data structure
  must be identical to the beginning of the KSP_FBCGS data structure
  so if you CHANGE anything here you must also change it there.
*/
#if !defined(__BCGS)
#define __BCGS

#include <petsc-private/kspimpl.h>        /*I "petscksp.h" I*/

typedef struct {
  Vec  guess;  /* if using right preconditioning with nonzero initial guess must keep that around to "fix" solution */
} KSP_BCGS;

extern PetscErrorCode KSPSetFromOptions_BCGS(KSP);
extern PetscErrorCode KSPView_BCGS(KSP,PetscViewer);
extern PetscErrorCode KSPSetUp_BCGS(KSP);
extern PetscErrorCode KSPSolve_BCGS(KSP);
extern PetscErrorCode KSPBuildSolution_BCGS(KSP,Vec,Vec*);
extern PetscErrorCode KSPReset_BCGS(KSP);
extern PetscErrorCode KSPDestroy_BCGS(KSP);

#endif
