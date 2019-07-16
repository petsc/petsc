static char help[] = "Spot check DMStag Compatibility Checks";

#include <petscdm.h>
#include <petscdmstag.h>

#define NDMS 4

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             dms[NDMS];
  PetscInt       i;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* Two 3d DMs, with all the same parameters */
  ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,3,2,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,2,3,4,5,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dms[0]);CHKERRQ(ierr);
    ierr = DMSetUp(dms[0]);CHKERRQ(ierr);
  ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,3,2,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,2,3,4,5,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dms[1]);CHKERRQ(ierr);
    ierr = DMSetUp(dms[1]);CHKERRQ(ierr);

  /* A derived 3d DM, with a different section */
  ierr = DMStagCreateCompatibleDMStag(dms[0],0,1,0,1,&dms[2]);CHKERRQ(ierr);

  /* A DM expected to be incompatible (different stencil width) */
  ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,3,2,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,2,3,4,5,DMSTAG_STENCIL_BOX,2,NULL,NULL,NULL,&dms[3]);CHKERRQ(ierr);

  /* Check expected self-compatibility */
  for (i=0; i<NDMS; ++i) {
    PetscBool compatible,set;
    ierr = DMGetCompatibility(dms[i],dms[i],&compatible,&set);CHKERRQ(ierr);
    if (!set || !compatible) SETERRQ1(PetscObjectComm((PetscObject)dms[i]),PETSC_ERR_PLIB,"DM %D not determined compatible with itself",i);CHKERRQ(ierr);
  }

  /* Check expected compatibility */
  for (i=1; i<=2; ++i) {
    PetscBool compatible,set;
    ierr = DMGetCompatibility(dms[0],dms[i],&compatible,&set);CHKERRQ(ierr);
    if (!set || !compatible) SETERRQ2(PetscObjectComm((PetscObject)dms[i]),PETSC_ERR_PLIB,"DM %D not determined compatible with DM %d",i,0);CHKERRQ(ierr);
  }

  /* Check expected incompatibility */
  {
    PetscBool compatible,set;
    ierr = DMGetCompatibility(dms[0],dms[3],&compatible,&set);CHKERRQ(ierr);
    if (!set || compatible) SETERRQ2(PetscObjectComm((PetscObject)dms[i]),PETSC_ERR_PLIB,"DM %D not determined incompatible with DM %d",i,0);CHKERRQ(ierr);
  }

  for (i=0; i<NDMS; ++i) {
    ierr = DMDestroy(&dms[i]);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 1
      suffix: 1

   test:
      nsize: 3
      suffix: 2

TEST*/
