
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
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",280,480,600,200,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);

  /* Readoptions */
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetEnum(NULL,NULL,"-wrap",DMBoundaryTypes,(PetscEnum*)&bx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-s",&s,NULL);CHKERRQ(ierr);

  /* Create distributed array and get vectors */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,bx,M,dof,s,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMView(da,viewer);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);

  value = 1;
  ierr  = VecSet(global,value);CHKERRQ(ierr);

  ierr  = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  value = rank+1;
  ierr  = VecScale(global,value);CHKERRQ(ierr);

  ierr = VecView(global,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nGlobal Vector:\n");CHKERRQ(ierr);
  ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"\n");CHKERRQ(ierr);

  /* Send ghost points to local vectors */
  ierr = DMGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(NULL,NULL,"-local_print",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    PetscViewer            sviewer;

    ierr = PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nLocal Vector: processor %d\n",rank);CHKERRQ(ierr);
    ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    ierr = VecView(local,sviewer);CHKERRQ(ierr);
    ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nLocal to global mapping\n");CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(da,&is);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Free memory */
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
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
