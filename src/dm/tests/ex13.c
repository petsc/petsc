
static char help[] = "Tests loading DM vector from file.\n\n";

/*
    ex14.c writes out the DMDA and vector read by this program.
*/

#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscInt       M = PETSC_DECIDE,N = PETSC_DECIDE;
  DM             da;
  Vec            global;
  PetscViewer    bviewer;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));

  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"daoutput",FILE_MODE_READ,&bviewer));
  CHKERRQ(DMCreate(PETSC_COMM_WORLD,&da));

  CHKERRQ(DMLoad(da,bviewer));
  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(VecLoad(global,bviewer));
  CHKERRQ(PetscViewerDestroy(&bviewer));

  CHKERRQ(VecView(global,PETSC_VIEWER_DRAW_WORLD));

  /* Free memory */
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(PetscFinalize());
  return 0;
}
