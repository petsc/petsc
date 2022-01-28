static char help[] = "Demonstrate standard DMStag operations.\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

static PetscErrorCode TestFields(DM dmstag);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             dmstag;
  PetscInt       dim;
  PetscBool      setSizes;

  /* Initialize PETSc and process command line arguments */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  dim = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);
  setSizes = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-setsizes",&setSizes,NULL);CHKERRQ(ierr);

  /* Creation (normal) */
  if (!setSizes) {
    switch (dim) {
      case 1:
        ierr = DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,3,1,1,DMSTAG_STENCIL_BOX,1,NULL,&dmstag);CHKERRQ(ierr);
        break;
      case 2:
        ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,3,2,PETSC_DECIDE,PETSC_DECIDE,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dmstag);CHKERRQ(ierr);
        break;
      case 3:
        ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,3,2,4,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dmstag);CHKERRQ(ierr);
        break;
      default:
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No support for dimension %D",dim);
    }
  } else {
    /* Creation (test providing decomp exactly)*/
    PetscMPIInt size;
    PetscInt lx[4] = {1,2,3}, ranksx = 3, mx = 6;
    PetscInt ly[3] = {4,5},   ranksy = 2, my = 9;
    PetscInt lz[2] = {6,7},   ranksz = 2, mz = 13;

    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
    switch (dim) {
      case 1:
        PetscAssertFalse(size != ranksx,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Must run on %D ranks with -dim 1 -setSizes",ranksx);
        ierr = DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,mx,1,1,DMSTAG_STENCIL_BOX,1,lx,&dmstag);CHKERRQ(ierr);
        break;
      case 2:
        PetscAssertFalse(size != ranksx * ranksy,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Must run on %D ranks with -dim 2 -setSizes",ranksx * ranksy);
        ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,mx,my,ranksx,ranksy,1,1,1,DMSTAG_STENCIL_BOX,1,lx,ly,&dmstag);CHKERRQ(ierr);
        break;
      case 3:
        PetscAssertFalse(size != ranksx * ranksy * ranksz,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Must run on %D ranks with -dim 3 -setSizes", ranksx * ranksy * ranksz);
        ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,mx,my,mz,ranksx,ranksy,ranksz,1,1,1,1,DMSTAG_STENCIL_BOX,1,lx,ly,lz,&dmstag);CHKERRQ(ierr);
        break;
      default:
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No support for dimension %D",dim);
    }
  }

  /* Setup */
  ierr = DMSetFromOptions(dmstag);CHKERRQ(ierr);
  ierr = DMSetUp(dmstag);CHKERRQ(ierr);

  /* Field Creation */
  ierr = TestFields(dmstag);CHKERRQ(ierr);

  /* Clean up and finalize PETSc */
  ierr = DMDestroy(&dmstag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

static PetscErrorCode TestFields(DM dmstag)
{
  PetscErrorCode ierr;
  Vec            vecLocal,vecGlobal;
  PetscReal      norm2;

  PetscFunctionBeginUser;
  ierr = DMCreateLocalVector(dmstag,&vecLocal);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmstag,&vecGlobal);CHKERRQ(ierr);
  ierr = VecSet(vecLocal,1.0);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmstag,vecLocal,INSERT_VALUES,vecGlobal);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmstag,vecLocal,INSERT_VALUES,vecGlobal);CHKERRQ(ierr);
  ierr = VecSet(vecGlobal,2.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmstag,vecGlobal,INSERT_VALUES,vecLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dmstag,vecGlobal,INSERT_VALUES,vecLocal);CHKERRQ(ierr);
  ierr = VecNorm(vecGlobal,NORM_2,&norm2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"2 Norm of test vector: %g\n",(double)norm2);CHKERRQ(ierr);
  ierr = VecDestroy(&vecLocal);CHKERRQ(ierr);
  ierr = VecDestroy(&vecGlobal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: basic_1
      nsize: 8
      args: -dm_view -dim 1 -stag_grid_x 37 -stag_stencil_type none -stag_stencil_width 2

   test:
      suffix: basic_2
      nsize: 14
      args: -dm_view -dim 2 -stag_grid_x 11 -stag_grid_y 7 -stag_stencil_type star

   test:
      suffix: basic_3
      nsize: 27
      args: -dm_view -dim 3 -stag_grid_x 4 -stag_grid_y 5 -stag_grid_z 6 -stag_stencil_type star -stag_ranks_x 3 -stag_ranks_y 3 -stag_ranks_z 3

   test:
      suffix: multidof_1
      nsize: 3
      args: -dm_view -dim 1 -stag_dof_0 2 -stag_dof_1 7

   test:
      suffix: multidof_2
      nsize: 9
      args: -dm_view -dim 2 -stag_grid_x 3 -stag_grid_y 3 -stag_dof_0 3 -stag_dof_1 4 -stag_dof_2 5

   test:
      suffix: multidof_3
      nsize: 27
      args: -dm_view -dim 3 -stag_grid_x 6 -stag_grid_y 5 -stag_grid_z 4 -stag_ranks_x 3 -stag_ranks_y 3 -stag_ranks_z 3 -stag_dof_0 3 -stag_dof_1 4 -stag_dof_2 2 -stag_dof_3 5

   test:
      suffix: zerodof_1
      nsize: 3
      args: -dm_view -dim 1 -stag_dof_0 0 -stag_dof_1 0

   test:
      suffix: zerodof_2
      nsize: 9
      args: -dm_view -dim 2 -stag_grid_x 3 -stag_grid_y 3 -stag_dof_0 0 -stag_dof_1 0 -stag_dof_2 0

   test:
      suffix: zerodof_3
      nsize: 27
      args: -dm_view -dim 3 -stag_grid_x 4 -stag_grid_y 5 -stag_grid_z 6 -stag_ranks_x 3 -stag_ranks_y 3 -stag_ranks_z 3 -stag_dof_0 0 -stag_dof_1 4 -stag_dof_2 0 -stag_dof_3 0

   test:
      suffix: sizes_1
      nsize: 3
      args: -dm_view -dim 1 -setSizes

   test:
      suffix: sizes_2
      nsize: 6
      args: -dm_view -dim 2 -setSizes

   test:
      suffix: sizes_3
      nsize: 12
      args: -dm_view -dim 3 -setSizes

   test:
      suffix: stencil_none_1
      nsize: 6
      args:  -dm_view -dim 2 -stag_grid_x 4 -stag_grid_y 5 -stag_stencil_type none -stag_stencil_width 0

   test:
      suffix: stencil_none_2
      nsize: 8
      args:  -dm_view -dim 3 -stag_grid_x 4 -stag_grid_y 5 -stag_grid_z 3 -stag_stencil_type none -stag_stencil_width 0

   test:
      suffix: ghosted_zerowidth_seq_1
      nsize: 1
      args:  -dm_view -dim 1 -stag_grid_x 4 -stag_boundary_type_x ghosted -stag_stencil_width 0

   test:
      suffix: ghosted_zerowidth_par_1
      nsize: 3
      args:  -dm_view -dim 1 -setsizes -stag_boundary_type_x ghosted -stag_stencil_width 0

   test:
      suffix: ghosted_zerowidth_seq_2
      nsize: 1
      args:  -dm_view -dim 2 -stag_grid_x 3 -stag_grid_y 5 -stag_boundary_type_x ghosted -stag_boundary_type_y ghosted -stag_stencil_width 0

   test:
      suffix: ghosted_zerowidth_par_2
      nsize: 6
      args:  -dm_view -dim 2 -setsizes -stag_boundary_type_x ghosted -stag_boundary_type_y ghosted -stag_stencil_width 0

   test:
      suffix: ghosted_zerowidth_seq_3
      nsize: 1
      args:  -dm_view -dim 3 -stag_grid_x 3 -stag_grid_y 5 -stag_grid_z 4 -stag_boundary_type_x ghosted -stag_boundary_type_y ghosted -stag_boundary_type_z ghosted -stag_stencil_width 0

   test:
      suffix: ghosted_zerowidth_par_3
      nsize: 12
      args:  -dm_view -dim 3 -setsizes -stag_boundary_type_x ghosted -stag_boundary_type_y ghosted -stag_boundary_type_z ghosted -stag_stencil_width 0

   testset:
      suffix: periodic_skinny_1
      nsize: 1
      args:  -dm_view -dim 1 -stag_grid_x 4 -stag_boundary_type_x periodic -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: periodic_skinny_2
      nsize: 1
      args:  -dm_view -dim 2 -stag_grid_x 4 -stag_grid_y 5 -stag_boundary_type_x periodic -stag_boundary_type_y periodic -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: periodic_skinny_3
      nsize: 1
      args:  -dm_view -dim 3 -stag_grid_x 4 -stag_grid_y 5 -stag_grid_z 3 -stag_boundary_type_x periodic -stag_boundary_type_y periodic -stag_boundary_type_z periodic -stag_stencil_width {{0 1 2}separate output}

TEST*/
