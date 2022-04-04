static char help[] = "Spot check DMStag Compatibility Checks";

#include <petscdm.h>
#include <petscdmstag.h>

#define NDMS 4

int main(int argc,char **argv)
{
  DM             dms[NDMS];
  PetscInt       i;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  /* Two 3d DMs, with all the same parameters */
  PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,3,2,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,2,3,4,5,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dms[0]));
    PetscCall(DMSetUp(dms[0]));
  PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,3,2,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,2,3,4,5,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dms[1]));
    PetscCall(DMSetUp(dms[1]));

  /* A derived 3d DM, with a different section */
  PetscCall(DMStagCreateCompatibleDMStag(dms[0],0,1,0,1,&dms[2]));

  /* A DM expected to be incompatible (different stencil width) */
  PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,3,2,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,2,3,4,5,DMSTAG_STENCIL_BOX,2,NULL,NULL,NULL,&dms[3]));

  /* Check expected self-compatibility */
  for (i=0; i<NDMS; ++i) {
    PetscBool compatible,set;
    PetscCall(DMGetCompatibility(dms[i],dms[i],&compatible,&set));
    PetscCheck(set && compatible,PetscObjectComm((PetscObject)dms[i]),PETSC_ERR_PLIB,"DM %D not determined compatible with itself",i);
  }

  /* Check expected compatibility */
  for (i=1; i<=2; ++i) {
    PetscBool compatible,set;
    PetscCall(DMGetCompatibility(dms[0],dms[i],&compatible,&set));
    PetscCheck(set && compatible,PetscObjectComm((PetscObject)dms[i]),PETSC_ERR_PLIB,"DM %D not determined compatible with DM %d",i,0);
  }

  /* Check expected incompatibility */
  {
    PetscBool compatible,set;
    PetscCall(DMGetCompatibility(dms[0],dms[3],&compatible,&set));
    PetscCheck(set && !compatible,PetscObjectComm((PetscObject)dms[i]),PETSC_ERR_PLIB,"DM %D not determined incompatible with DM %d",i,0);
  }

  for (i=0; i<NDMS; ++i) {
    PetscCall(DMDestroy(&dms[i]));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 1
      suffix: 1

   test:
      nsize: 3
      suffix: 2

TEST*/
