/* 
   Private data structure for ILU/ICC/LU/Cholesky preconditioners.
*/
#if !defined(__FACTOR_H)
#define __FACTOR_H

#include <petsc-private/pcimpl.h>                /*I "petscpc.h" I*/

typedef struct {
  Mat               fact;             /* factored matrix */
  MatFactorInfo     info;
  MatOrderingType   ordering;         /* matrix reordering */
  MatSolverPackage  solvertype;
  MatFactorType     factortype;
} PC_Factor;

extern PetscErrorCode  PCFactorGetMatrix_Factor(PC,Mat*);

EXTERN_C_BEGIN
extern PetscErrorCode  PCFactorSetZeroPivot_Factor(PC,PetscReal);
extern PetscErrorCode  PCFactorSetShiftType_Factor(PC,MatFactorShiftType);
extern PetscErrorCode  PCFactorSetShiftAmount_Factor(PC,PetscReal);
extern PetscErrorCode  PCFactorSetDropTolerance_Factor(PC,PetscReal,PetscReal,PetscInt);
extern PetscErrorCode  PCFactorSetFill_Factor(PC,PetscReal);
extern PetscErrorCode  PCFactorSetMatOrderingType_Factor(PC,const MatOrderingType);
extern PetscErrorCode  PCFactorSetLevels_Factor(PC,PetscInt);
extern PetscErrorCode  PCFactorSetAllowDiagonalFill_Factor(PC);
extern PetscErrorCode  PCFactorSetPivotInBlocks_Factor(PC,PetscBool );
extern PetscErrorCode  PCFactorSetMatSolverPackage_Factor(PC,const MatSolverPackage);
extern PetscErrorCode  PCFactorSetUpMatSolverPackage_Factor(PC);
extern PetscErrorCode  PCFactorGetMatSolverPackage_Factor(PC,const MatSolverPackage*);
extern PetscErrorCode  PCFactorSetColumnPivot_Factor(PC,PetscReal);
extern PetscErrorCode  PCSetFromOptions_Factor(PC);
extern PetscErrorCode PCView_Factor(PC,PetscViewer);
EXTERN_C_END

#endif
