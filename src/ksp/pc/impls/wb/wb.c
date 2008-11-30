#define PETSCKSP_DLL


#include "petscpc.h"   /*I "petscpc.h" I*/
#include "petscmg.h"   /*I "petscpc.h" I*/
#include "petscda.h"   /*I "petscda.h" I*/

extern PetscErrorCode DAGetWireBasketInterpolation(DA,Mat,Mat*);
extern PetscErrorCode DAGetFaceInterpolation(DA,Mat,Mat*);

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_WB"
PetscErrorCode PCSetUp_WB(PC pc,void *ida)
{
  PetscErrorCode ierr;
  DA             da = (DA)ida;
  Mat            A,P;

  PetscFunctionBegin;
  ierr = PCGetOperators(pc,PETSC_NULL,&A,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAGetFaceInterpolation(da,A,&P);CHKERRQ(ierr);
  ierr = PCMGSetInterpolation(pc,1,P);CHKERRQ(ierr);
  ierr = MatDestroy(P);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_WB"
PetscErrorCode PCDestroy_WB(PC pc,void *ida)
{
  PetscErrorCode ierr;
  DA             da = (DA)ida;

  PetscFunctionBegin;
  ierr = DADestroy(da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_WB_Error"
PetscErrorCode PCSetUp_WB_Error(PC pc,void *ida)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"You are using the WB preconditioner but never called PDWBSetDA()");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCWBSetDA"
PetscErrorCode PETSCKSP_DLLEXPORT PCWBSetDA(PC pc,DA da)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(da,DM_COOKIE,1);

  ierr = PCMGSetSetup(pc,PCSetUp_WB,PCDestroy_WB,da);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)da);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}


/*MC
     PCWB - Wirebasket based coarse problem two level multigrid preconditioner

.seealso:  PCMG, PCWBSetDA()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_WB"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_WB(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCSetType(pc,PCMG);CHKERRQ(ierr);
  ierr = PCMGSetLevels(pc,2,PETSC_NULL);CHKERRQ(ierr);
  ierr = PCMGSetGalerkin(pc);CHKERRQ(ierr);
  ierr = PCMGSetSetup(pc,PCSetUp_WB_Error,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
