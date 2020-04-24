static char help[] = "Test DMStag 2d periodic and ghosted boundary conditions\n\n";
#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  DM              dm;
  Vec             vec,vecLocal1,vecLocal2;
  PetscScalar     *a,***a1,***a2,expected;
  PetscInt        startx,starty,nx,ny,i,j,d,is,js,dof0,dof1,dof2,dofTotal,stencilWidth,Nx,Ny;
  DMBoundaryType  boundaryTypex,boundaryTypey;
  PetscMPIInt     rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  dof0 = 1;
  dof1 = 1;
  dof2 = 1;
  stencilWidth = 2;
  ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,4,4,PETSC_DECIDE,PETSC_DECIDE,dof0,dof1,dof2,DMSTAG_STENCIL_BOX,stencilWidth,NULL,NULL,&dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);
  dofTotal = dof0 + 2*dof1 + dof2;
  ierr = DMStagGetStencilWidth(dm,&stencilWidth);CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm,&vecLocal1);CHKERRQ(ierr);
  ierr = VecDuplicate(vecLocal1,&vecLocal2);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm,&vec);CHKERRQ(ierr);
  ierr = VecSet(vec,1.0);CHKERRQ(ierr);
  ierr = VecSet(vecLocal1,0.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,vec,INSERT_VALUES,vecLocal1);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,vec,INSERT_VALUES,vecLocal1);CHKERRQ(ierr);

  ierr = DMStagGetCorners(dm,&startx,&starty,NULL,&nx,&ny,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dm,vecLocal1,&a1);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,vecLocal2,&a2);CHKERRQ(ierr);
  for (j=starty; j<starty + ny; ++j) {
    for (i=startx; i<startx + nx; ++i) {
      for (d=0; d<dofTotal; ++d) {
        if (a1[j][i][d] != 1.0) {
          ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Unexpected value %g (expecting %g)\n",rank,(double)PetscRealPart(a1[j][i][d]),1.0);CHKERRQ(ierr);
        }
        a2[j][i][d] = 0.0;
        for (js = -stencilWidth; js <= stencilWidth; ++js) {
          for (is = -stencilWidth; is <= stencilWidth; ++is) {
            a2[j][i][d] += a1[j+js][i+is][d];
          }
        }
      }
    }
  }
  ierr = DMStagVecRestoreArrayRead(dm,vecLocal1,&a1);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,vecLocal2,&a2);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(dm,vecLocal2,INSERT_VALUES,vec);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,vecLocal2,INSERT_VALUES,vec);CHKERRQ(ierr);

  /* For the all-periodic case, all values are the same . Otherwise, just check the local version */
  ierr = DMStagGetBoundaryTypes(dm,&boundaryTypex,&boundaryTypey,NULL);CHKERRQ(ierr);
  if (boundaryTypex == DM_BOUNDARY_PERIODIC && boundaryTypey == DM_BOUNDARY_PERIODIC) {
    ierr = VecGetArray(vec,&a);CHKERRQ(ierr);
    expected = 1.0; for (d=0;d<2;++d) expected *= (2*stencilWidth+1);
    for (i=0; i<ny*nx*dofTotal; ++i) {
      if (a[i] != expected) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Unexpected value %g (expecting %g)\n",rank,(double)PetscRealPart(a[i]),(double)PetscRealPart(expected));CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(vec,&a);CHKERRQ(ierr);
  } else {
    ierr = DMStagVecGetArrayRead(dm,vecLocal2,&a2);CHKERRQ(ierr);
    ierr = DMStagGetGlobalSizes(dm,&Nx,&Ny,NULL);CHKERRQ(ierr);
    if (stencilWidth > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Non-periodic check implemented assuming stencilWidth = 1");
      for (j=starty; j<starty + ny; ++j) {
        for (i=startx; i<startx + nx; ++i) {
          PetscInt  dd,extra[2];
          PetscBool bnd[2];
          bnd[0] = (PetscBool)((i == 0 || i == Nx-1) && boundaryTypex != DM_BOUNDARY_PERIODIC);
          bnd[1] = (PetscBool)((j == 0 || j == Ny-1) && boundaryTypey != DM_BOUNDARY_PERIODIC);
          extra[0] = i == Nx-1 && boundaryTypex != DM_BOUNDARY_PERIODIC ? 1 : 0;
          extra[1] = j == Ny-1 && boundaryTypey != DM_BOUNDARY_PERIODIC ? 1 : 0;
          { /* vertices */
            PetscScalar expected = 1.0;
            for (dd=0;dd<2;++dd) expected *= (bnd[dd] ? stencilWidth + 1 + extra[dd] : 2*stencilWidth + 1);
            for (d=0; d<dof0; ++d) {
              if (a2[j][i][d] != expected) {
                ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Element (%D,%D)[%D] Unexpected value %g (expecting %g)\n",rank,i,j,d,(double)PetscRealPart(a2[j][i][d]),(double)PetscRealPart(expected));CHKERRQ(ierr);
              }
            }
          }
          { /* down edges */
            PetscScalar expected = (bnd[1] ? stencilWidth + 1 + extra[1] : 2*stencilWidth + 1);
            expected *= ((bnd[0] ? 1 : 2) * stencilWidth + 1);
            for (d=dof0; d<dof0+dof1; ++d) {
              if (a2[j][i][d] != expected) {
                ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Element (%D,%D)[%D] Unexpected value %g (expecting %g)\n",rank,i,j,d,(double)PetscRealPart(a2[j][i][d]),(double)PetscRealPart(expected));CHKERRQ(ierr);
              }
            }
          }
          { /* left edges */
            PetscScalar expected = (bnd[0] ? stencilWidth + 1 + extra[0] : 2*stencilWidth + 1);
            expected *= ((bnd[1] ? 1 : 2) * stencilWidth + 1);
            for (d=dof0+dof1; d<dof0+2*dof1; ++d) {
              if (a2[j][i][d] != expected) {
                ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Element (%D,%D)[%D] Unexpected value %g (expecting %g)\n",rank,i,j,d,(double)PetscRealPart(a2[j][i][d]),(double)PetscRealPart(expected));CHKERRQ(ierr);
              }
            }
          }
          { /* elements */
            PetscScalar expected = 1.0;
            for (dd=0;dd<2;++dd) expected *= ((bnd[dd] ? 1 : 2) * stencilWidth + 1);
            for (d=dofTotal-dof2; d<dofTotal; ++d) {
              if (a2[j][i][d] != expected) {
                ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Element (%D,%D)[%D] Unexpected value %g (expecting %g)\n",rank,i,j,d,(double)PetscRealPart(a2[j][i][d]),(double)PetscRealPart(expected));CHKERRQ(ierr);
              }
            }
          }
        }
      }
    ierr = DMStagVecRestoreArrayRead(dm,vecLocal2,&a2);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&vec);CHKERRQ(ierr);
  ierr = VecDestroy(&vecLocal1);CHKERRQ(ierr);
  ierr = VecDestroy(&vecLocal2);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 4
      args: -stag_ranks_x 2 -stag_ranks_y 2 -stag_stencil_width 1 -stag_dof_2 2

   test:
      suffix: 2
      nsize: 4
      args: -stag_ranks_x 2 -stag_ranks_y 2 -stag_dof_1 2 -stag_grid_y 5

   test:
      suffix: 3
      nsize: 6
      args: -stag_ranks_x 3 -stag_ranks_y 2 -stag_dof_0 2 -stag_grid_x 6

   test:
      suffix: 4
      nsize: 6
      args: -stag_ranks_x 3 -stag_ranks_y 2 -stag_dof_0 0 -stag_dof_1 0 -stag_dof_2 0 -stag_grid_x 4 -stag_boundary_type_x ghosted -stag_boundary_type_y ghosted -stag_stencil_width 1

   test:
      suffix: 5
      nsize: 6
      args: -stag_ranks_x 3 -stag_ranks_y 2 -stag_dof_0 0 -stag_dof_1 0 -stag_dof_2 0 -stag_grid_x 4 -stag_boundary_type_x ghosted -stag_stencil_width 1

   test:
      suffix: 6
      nsize: 9
      args: -stag_dof_0 2 -stag_dof_1 2 -stag_dof_2 1 -stag_dof_2 1 -stag_boundary_type_y ghosted -stag_grid_x 9 -stag_grid_y 13 -stag_ranks_x 3 -stag_ranks_y 3 -stag_stencil_width 1

   test:
      suffix: 7
      nsize: 1
      args: -stag_dof_0 2 -stag_dof_1 2 -stag_dof_2 1 -stag_dof_2 1 stag_grid_x 9 -stag_grid_y 13 -stag_stencil_width 1

TEST*/
