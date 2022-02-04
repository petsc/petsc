static char help[] = "Test DMStag transfer operators, on a faces-only grid.\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  DM             dm;
  PetscInt       dim;
  PetscBool      flg,dump;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Supply -dim option\n");
  if (dim == 1) PetscCall(DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,1,0,DMSTAG_STENCIL_BOX,1,NULL,&dm));
  else if (dim == 2) PetscCall(DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,4,PETSC_DECIDE,PETSC_DECIDE,0,1,0,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm));
  else if (dim == 3)
 PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,4,4,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,0,0,1,0,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm));
  else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"dim must be 1, 2, or 3");
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));

  /* Flags to dump binary or ASCII output */
  dump = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-dump",&dump,NULL));

  /* Directly create a coarsened DM and transfer operators */
  {
    DM dmCoarse;
    PetscCall(DMCoarsen(dm,MPI_COMM_NULL,&dmCoarse));
    {
      Mat Ai;
      PetscCall(DMCreateInterpolation(dmCoarse,dm,&Ai,NULL));
      if (dump) {
        PetscViewer viewer;
        PetscCall(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)dm),"matI.pbin",FILE_MODE_WRITE,&viewer));
        PetscCall(MatView(Ai,viewer));
        PetscCall(PetscViewerDestroy(&viewer));
      }
      PetscCall(MatDestroy(&Ai));
    }
    {
      Mat Ar;
      PetscCall(DMCreateRestriction(dmCoarse,dm,&Ar));
      if (dump) {
        PetscViewer viewer;
        PetscCall(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)dm),"matR.pbin",FILE_MODE_WRITE,&viewer));
        PetscCall(MatView(Ar,viewer));
        PetscCall(PetscViewerDestroy(&viewer));
      }
      PetscCall(MatDestroy(&Ar));
    }
    PetscCall(DMDestroy(&dmCoarse));
  }

  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      args: -dim 1

   test:
      suffix: 2
      nsize: 1
      args: -dim 2

TEST*/
