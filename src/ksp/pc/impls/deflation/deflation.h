#if !defined(__DEFLATION_H)
#define __DEFLATION_H

#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/

typedef struct {
  PetscBool init;            /* do only init step - error correction of direction is omitted */
  PetscBool pre;             /* start with x0 being the solution in the deflation space */
  PetscBool correct;         /* add CP (Qr) correction to descent direction */
  PetscBool truenorm;
  PetscBool adaptiveconv;
  PetscReal adaptiveconst;
  PetscInt  reductionfact;
  Mat       W,Wt,AW,WtAW;    /* deflation space, coarse problem mats */
  KSP       WtAWinv;         /* deflation coarse problem */
  KSPType   ksptype;
  PC        pc;              /* additional preconditioner */
  Vec       work;
  Vec       *workcoarse;

  PCDeflationSpaceType spacetype;
  PetscInt             spacesize;
  PetscInt             nestedlvl;
  PetscInt             maxnestedlvl;
  PetscBool            extendsp;
} PC_Deflation;

PETSC_INTERN PetscErrorCode PCDeflationComputeSpace(PC);

#endif

