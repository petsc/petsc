
static char help[] = "Tests various 1-dimensional DMDA routines.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscMPIInt            rank;
  PetscInt               M  = 13,s=1,dof=1;
  DMBoundaryType         bx = DM_BOUNDARY_PERIODIC;
  PetscErrorCode         ierr;
  DM                     da;
  PetscViewer            viewer;
  Vec                    local,global;
  PetscScalar            value;
  PetscDraw              draw;
  PetscBool              flg = PETSC_FALSE;
  ISLocalToGlobalMapping is;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",280,480,600,200,&viewer));
  CHKERRQ(PetscViewerDrawGetDraw(viewer,0,&draw));
  CHKERRQ(PetscDrawSetDoubleBuffer(draw));

  /* Readoptions */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetEnum(NULL,NULL,"-wrap",DMBoundaryTypes,(PetscEnum*)&bx,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-s",&s,NULL));

  /* Create distributed array and get vectors */
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,bx,M,dof,s,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMView(da,viewer));
  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(DMCreateLocalVector(da,&local));

  value = 1;
  CHKERRQ(VecSet(global,value));

  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  value = rank+1;
  CHKERRQ(VecScale(global,value));

  CHKERRQ(VecView(global,viewer));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nGlobal Vector:\n"));
  CHKERRQ(VecView(global,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"\n"));

  /* Send ghost points to local vectors */
  CHKERRQ(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  CHKERRQ(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-local_print",&flg,NULL));
  if (flg) {
    PetscViewer            sviewer;

    CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nLocal Vector: processor %d\n",rank));
    CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
    CHKERRQ(VecView(local,sviewer));
    CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
    CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nLocal to global mapping\n"));
  CHKERRQ(DMGetLocalToGlobalMapping(da,&is));
  CHKERRQ(ISLocalToGlobalMappingView(is,PETSC_VIEWER_STDOUT_WORLD));

  /* Free memory */
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      args: -nox
      filter: grep -v "MPI processes"
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
