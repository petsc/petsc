#define PETSCKSP_DLL

#include "../src/ksp/pc/impls/factor/factor.h"     /*I "petscpc.h"  I*/

/* ------------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetZeroPivot_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetZeroPivot_Factor(PC pc,PetscReal z)
{
  PC_Factor *ilu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  ilu->info.zeropivot = z;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftNonzero_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftNonzero_Factor(PC pc,PetscReal shift)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (shift == (PetscReal) PETSC_DECIDE) {
    dir->info.shiftnz = 1.e-12;
  } else {
    dir->info.shiftnz = shift;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftPd_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftPd_Factor(PC pc,PetscTruth shift)
{
  PC_Factor *dir = (PC_Factor*)pc->data;
 
  PetscFunctionBegin;
  if (shift) {
    dir->info.shiftpd = 1.0;
  } else {
    dir->info.shiftpd = 0.0;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetUseDropTolerance_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetUseDropTolerance_Factor(PC pc,PetscReal dt,PetscReal dtcol,PetscInt dtcount)
{
  PC_Factor         *ilu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (pc->setupcalled && (!ilu->info.usedt || ((PC_Factor*)ilu)->info.dt != dt || ((PC_Factor*)ilu)->info.dtcol != dtcol || ((PC_Factor*)ilu)->info.dtcount != dtcount)) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot change tolerance after use");
  }
  ilu->info.usedt   = PETSC_TRUE;
  ilu->info.dt      = dt;
  ilu->info.dtcol   = dtcol;
  ilu->info.dtcount = dtcount;
  ilu->info.fill    = PETSC_DEFAULT;
  PetscFunctionReturn(0);
}  
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetFill_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetFill_Factor(PC pc,PetscReal fill)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  dir->info.fill = fill;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetMatOrderingType_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetMatOrderingType_Factor(PC pc,const MatOrderingType ordering)
{
  PC_Factor      *dir = (PC_Factor*)pc->data;
  PetscErrorCode ierr;
  PetscTruth     flg;
 
  PetscFunctionBegin;
  if (!pc->setupcalled) {
     ierr = PetscStrfree(dir->ordering);CHKERRQ(ierr);
     ierr = PetscStrallocpy(ordering,&dir->ordering);CHKERRQ(ierr);
  } else {
    ierr = PetscStrcmp(dir->ordering,ordering,&flg);CHKERRQ(ierr);
    if (!flg) {
      SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot change ordering after use");
    }
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetLevels_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetLevels_Factor(PC pc,PetscInt levels)
{
  PC_Factor      *ilu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    ilu->info.levels = levels;
  } else if (ilu->info.usedt || ilu->info.levels != levels) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot change levels after use");
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetAllowDiagonalFill_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetAllowDiagonalFill_Factor(PC pc)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  dir->info.diagonal_fill = 1;
  PetscFunctionReturn(0);
}
EXTERN_C_END


/* ------------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetPivotInBlocks_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetPivotInBlocks_Factor(PC pc,PetscTruth pivot)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  dir->info.pivotinblocks = pivot ? 1.0 : 0.0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftInBlocks_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftInBlocks_Factor(PC pc,PetscReal shift)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (shift == PETSC_DEFAULT) {
    dir->info.shiftinblocks = 1.e-12;
  } else {
    dir->info.shiftinblocks = shift;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorGetMatrix_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorGetMatrix_Factor(PC pc,Mat *mat)
{
  PC_Factor *ilu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (!ilu->fact) SETERRQ(PETSC_ERR_ORDER,"Matrix not yet factored; call after KSPSetUp() or PCSetUp()");
  *mat = ilu->fact;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetMatSolverPackage_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetMatSolverPackage_Factor(PC pc,const MatSolverPackage stype)
{
  PetscErrorCode ierr;
  PC_Factor      *lu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (lu->fact) {
    const MatSolverPackage ltype;
    PetscTruth             flg;
    ierr = MatFactorGetSolverPackage(lu->fact,&ltype);CHKERRQ(ierr);
    ierr = PetscStrcmp(stype,ltype,&flg);CHKERRQ(ierr);
    if (!flg) {
      SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot change solver matrix package after PC has been setup or used");
    }
  }
  ierr = PetscStrfree(lu->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(stype,&lu->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorGetMatSolverPackage_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorGetMatSolverPackage_Factor(PC pc,const MatSolverPackage *stype)
{
  PC_Factor      *lu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  *stype = lu->solvertype;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetPivoting_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetPivoting_Factor(PC pc,PetscReal dtcol)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (dtcol < 0.0 || dtcol > 1.0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Column pivot tolerance is %G must be between 0 and 1",dtcol);
  dir->info.dtcol = dtcol;
  PetscFunctionReturn(0);
}
EXTERN_C_END
