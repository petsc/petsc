
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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"daoutput",FILE_MODE_READ,&bviewer));
  PetscCall(DMCreate(PETSC_COMM_WORLD,&da));

  PetscCall(DMLoad(da,bviewer));
  PetscCall(DMCreateGlobalVector(da,&global));
  PetscCall(VecLoad(global,bviewer));
  PetscCall(PetscViewerDestroy(&bviewer));

  PetscCall(VecView(global,PETSC_VIEWER_DRAW_WORLD));

  /* Free memory */
  PetscCall(VecDestroy(&global));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
