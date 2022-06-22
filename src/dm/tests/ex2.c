
static char help[] = "Tests various 1-dimensional DMDA routines.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscMPIInt            rank;
  PetscInt               M  = 13,s=1,dof=1;
  DMBoundaryType         bx = DM_BOUNDARY_PERIODIC;
  DM                     da;
  PetscViewer            viewer;
  Vec                    local,global;
  PetscScalar            value;
  PetscDraw              draw;
  PetscBool              flg = PETSC_FALSE;
  ISLocalToGlobalMapping is;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",280,480,600,200,&viewer));
  PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
  PetscCall(PetscDrawSetDoubleBuffer(draw));

  /* Readoptions */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscOptionsGetEnum(NULL,NULL,"-wrap",DMBoundaryTypes,(PetscEnum*)&bx,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-s",&s,NULL));

  /* Create distributed array and get vectors */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,bx,M,dof,s,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMView(da,viewer));
  PetscCall(DMCreateGlobalVector(da,&global));
  PetscCall(DMCreateLocalVector(da,&local));

  value = 1;
  PetscCall(VecSet(global,value));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  value = rank+1;
  PetscCall(VecScale(global,value));

  PetscCall(VecView(global,viewer));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nGlobal Vector:\n"));
  PetscCall(VecView(global,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"\n"));

  /* Send ghost points to local vectors */
  PetscCall(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  PetscCall(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-local_print",&flg,NULL));
  if (flg) {
    PetscViewer            sviewer;

    PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nLocal Vector: processor %d\n",rank));
    PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
    PetscCall(VecView(local,sviewer));
    PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
    PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nLocal to global mapping\n"));
  PetscCall(DMGetLocalToGlobalMapping(da,&is));
  PetscCall(ISLocalToGlobalMappingView(is,PETSC_VIEWER_STDOUT_WORLD));

  /* Free memory */
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&global));
  PetscCall(VecDestroy(&local));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      args: -nox
      filter: grep -v " MPI process"
      output_file: output/ex2_1.out
      requires: x

   test:
      suffix: 2
      nsize: 3
      args: -wrap none -local_print -nox
      filter: grep -v "Vec Object: Vec"
      requires: x

   test:
      suffix: 3
      nsize: 3
      args: -wrap ghosted -local_print -nox
      filter: grep -v "Vec Object: Vec"
      requires: x

TEST*/
