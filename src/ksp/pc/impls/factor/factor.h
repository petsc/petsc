/* 
   Private data structure for ILU/ICC/LU/Cholesky preconditioners.
*/
#if !defined(__FACTOR_H)
#define __FACTOR_H

#include "private/pcimpl.h"                /*I "petscpc.h" I*/

typedef struct {
  Mat               fact;             /* factored matrix */
  MatFactorInfo     info;
  MatOrderingType   ordering;         /* matrix reordering */
  MatSolverPackage  solvertype;
  MatFactorType     factortype;
} PC_Factor;

EXTERN_C_BEGIN
extern PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetZeroPivot_Factor(PC,PetscReal);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftType_Factor(PC,MatFactorShiftType);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftAmount_Factor(PC,PetscReal);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetDropTolerance_Factor(PC,PetscReal,PetscReal,PetscInt);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetFill_Factor(PC,PetscReal);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetMatOrderingType_Factor(PC,const MatOrderingType);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetLevels_Factor(PC,PetscInt);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetAllowDiagonalFill_Factor(PC);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetPivotInBlocks_Factor(PC,PetscTruth);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCFactorGetMatrix_Factor(PC,Mat*);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetMatSolverPackage_Factor(PC,const MatSolverPackage);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCFactorGetMatSolverPackage_Factor(PC,const MatSolverPackage*);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetColumnPivot_Factor(PC,PetscReal);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCSetFromOptions_Factor(PC);
extern PetscErrorCode PCView_Factor(PC,PetscViewer);
EXTERN_C_END

#endif
