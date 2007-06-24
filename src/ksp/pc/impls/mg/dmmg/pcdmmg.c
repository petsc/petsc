#define PETSCKSP_DLL

/*
    Defines a multigrid preconditioner that is built from a DM

*/
#include "petscmg.h"                    /*I "petscmg.h" I*/
#include "petscda.h"                    /*I "petscda.h" I*/


#undef __FUNCT__  
#define __FUNCT__ "PCDMMGSetDM"
/*@
   PCDMMGSetDM - Sets the coarsest DM that is to be used to define the interpolation/restriction
      for the multigrid preconditioner.

   Not Collective

   Input Parameter:
+  pc - the preconditioner context
-  dm - the coarsest dm

   Level: intermediate

.keywords: MG, get, levels, multigrid

.seealso: PCMG, PCMGSetLevels()
@*/
extern PetscErrorCode PCDMMGSetDM(PC pc,DM dm)
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels;
  Mat            R;
  DM             dmf;
  MPI_Comm       comm;
  PetscTruth     flg;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCMG,&flg);CHKERRQ(ierr);
  if (!flg) PetscFunctionReturn(0);

  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  ierr = PCMGGetLevels(pc,&nlevels);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);

  /* refine the DM nlevels - 1 and use it to fill up the PCMG restrictions/interpolations */
  for (i=1; i<nlevels; i++) {
    ierr = DMRefine(dm,comm,&dmf);CHKERRQ(ierr);
    ierr = DMGetInterpolation(dm,dmf,&R,PETSC_NULL);CHKERRQ(ierr);
    ierr = PCMGSetInterpolation(pc,i,R);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    dm   = dmf;
  }
  ierr = DMDestroy(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
