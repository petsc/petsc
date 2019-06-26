#if !defined(__DEFLATION_H)
#define __DEFLATION_H

#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/
#include <petscksp.h>

typedef struct {
  PetscBool   init;            /* do only init step - error correction of direction is omitted */
  PetscBool   correct;         /* add CP (Qr) correction to descent direction */
  PetscScalar correctfact;
  PetscInt    reductionfact;
  Mat         W,Wt,WtA,WtAW;    /* deflation space, coarse problem mats */
  KSP         WtAWinv;         /* deflation coarse problem */
  KSPType     ksptype;
  PC          pc;              /* additional preconditioner */
  Vec         work;
  Vec         *workcoarse;
  const char  *prefix;

  PCDeflationSpaceType spacetype;
  PetscInt             spacesize;
  PetscInt             lvl;
  PetscInt             maxlvl;
  PetscBool            extendsp;
} PC_Deflation;

PETSC_INTERN PetscErrorCode PCDeflationComputeSpace(PC);

#endif

