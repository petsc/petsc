static char help[] = "Test DMStag 3d star stencil\n\n";
#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  DM              dm;
  Vec             vec,vecLocal1,vecLocal2;
  PetscScalar     *a,****a1,****a2,expected,sum;
  PetscInt        startx,starty,startz,nx,ny,nz,i,j,k,d,is,js,ks,dof0,dof1,dof2,dof3,dofTotal,stencilWidth,ngx,ngy,ngz;
  DMBoundaryType  boundaryTypex,boundaryTypey,boundaryTypez;
  PetscMPIInt     rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  dof0 = 1;
  dof1 = 1;
  dof2 = 1;
  dof3 = 1;
  stencilWidth = 2;
  ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,4,4,4,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof0,dof1,dof2,dof3,DMSTAG_STENCIL_STAR,stencilWidth,NULL,NULL,NULL,&dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,&dof3);CHKERRQ(ierr);
  dofTotal = dof0 + 3*dof1 + 3*dof2 + dof3;
  ierr = DMStagGetStencilWidth(dm,&stencilWidth);CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm,&vecLocal1);CHKERRQ(ierr);
  ierr = VecDuplicate(vecLocal1,&vecLocal2);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm,&vec);CHKERRQ(ierr);
  ierr = VecSet(vec,1.0);CHKERRQ(ierr);
  ierr = VecSet(vecLocal1,0.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,vec,INSERT_VALUES,vecLocal1);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,vec,INSERT_VALUES,vecLocal1);CHKERRQ(ierr);

  ierr = DMStagGetCorners(dm,&startx,&starty,&startz,&nx,&ny,&nz,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dm,vecLocal1,&a1);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,vecLocal2,&a2);CHKERRQ(ierr);
  for (k=startz; k<startz + nz; ++k) {
    for (j=starty; j<starty + ny; ++j) {
      for (i=startx; i<startx + nx; ++i) {
        for (d=0; d<dofTotal; ++d) {
          if (a1[k][j][i][d] != 1.0) {
            ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Unexpected value %g (expecting %g)\n",rank,(double)PetscRealPart(a1[k][j][i][d]),1.0);CHKERRQ(ierr);
          }
          a2[k][j][i][d] = 0.0;
          for (ks = -stencilWidth; ks <= stencilWidth; ++ks) {
            a2[k][j][i][d] += a1[k+ks][j][i][d];
          }
          for (js = -stencilWidth; js <= stencilWidth; ++js) {
            a2[k][j][i][d] += a1[k][j+js][i][d];
          }
          for (is = -stencilWidth; is <= stencilWidth; ++is) {
            a2[k][j][i][d] += a1[k][j][i+is][d];
          }
            a2[k][j][i][d] -= 2.0*a1[k][j][i][d];
        }
      }
    }
  }
  ierr = DMStagVecRestoreArrayRead(dm,vecLocal1,&a1);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,vecLocal2,&a2);CHKERRQ(ierr);

  DMLocalToGlobalBegin(dm,vecLocal2,INSERT_VALUES,vec);CHKERRQ(ierr);
  DMLocalToGlobalEnd(dm,vecLocal2,INSERT_VALUES,vec);CHKERRQ(ierr);

  /* For the all-periodic case, some additional checks */
  ierr = DMStagGetBoundaryTypes(dm,&boundaryTypex,&boundaryTypey,&boundaryTypez);CHKERRQ(ierr);
  if (boundaryTypex == DM_BOUNDARY_PERIODIC && boundaryTypey == DM_BOUNDARY_PERIODIC && boundaryTypez == DM_BOUNDARY_PERIODIC) {

    ierr = DMStagGetGhostCorners(dm,NULL,NULL,NULL,&ngx,&ngy,&ngz);CHKERRQ(ierr);
    expected = (ngx*ngy*ngz - 8*stencilWidth*stencilWidth*stencilWidth - 4*stencilWidth*stencilWidth*(nx + ny + nz))*dofTotal;
    ierr = VecSum(vecLocal1,&sum);CHKERRQ(ierr);
    if (sum != expected) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Unexpected sum of local entries %g (expected %g)\n",rank,(double)PetscRealPart(sum),(double)PetscRealPart(expected));CHKERRQ(ierr);
    }

    ierr = VecGetArray(vec,&a);CHKERRQ(ierr);
    expected = 1 + 6*stencilWidth;
    for (i=0; i<nz*ny*nx*dofTotal; ++i) {
      if (a[i] != expected) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Unexpected value %g (expecting %g)\n",rank,(double)PetscRealPart(a[i]),(double)PetscRealPart(expected));CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(vec,&a);CHKERRQ(ierr);
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
      nsize: 8
      args: -stag_ranks_x 2 -stag_ranks_y 2 -stag_ranks_z 2 -stag_stencil_width 1

   test:
      suffix: 2
      nsize: 12
      args: -stag_ranks_x 3 -stag_ranks_y 2 -stag_ranks_z 2 -stag_dof_0 2 -stag_grid_x 6

   test:
      suffix: 3
      nsize: 8
      args: -stag_dof_0 3 -stag_dof_1 2 -stag_dof_2 4 -stag_dof_3 2 -stag_stencil_width 3 -stag_grid_x 6 -stag_grid_y 6 -stag_grid_z 7

   test:
      suffix: 4
      nsize: 8
      args: -stag_stencil_width 1 -stag_grid_x 2 -stag_grid_y 2 -stag_grid_z 2 -stag_boundary_type_x ghosted

   test:
      suffix: 5
      nsize: 8
      args: -stag_stencil_width 1 -stag_grid_x 2 -stag_grid_y 2 -stag_grid_z 2 -stag_boundary_type_y ghosted

   test:
      suffix: 6
      nsize: 8
      args: -stag_stencil_width 1 -stag_grid_x 2 -stag_grid_y 2 -stag_grid_z 2 -stag_boundary_type_z ghosted -stag_dof_0 2 -stag_dof_1 3 -stag_dof_2 2 -stag_dof_3 2

   test:
      suffix: 7
      nsize: 8
      args: -stag_stencil_width 1 -stag_grid_x 3 -stag_grid_y 2 -stag_grid_z 2 -stag_boundary_type_x ghosted -stag_boundary_type_y ghosted

   test:
      suffix: 8
      nsize: 8
      args: -stag_stencil_width 1 -stag_grid_x 2 -stag_grid_y 5 -stag_grid_z 2 -stag_boundary_type_x ghosted -stag_boundary_type_z ghosted

   test:
      suffix: 9
      nsize: 8
      args: -stag_stencil_width 1 -stag_grid_x 2 -stag_grid_y 2 -stag_grid_z 3 -stag_boundary_type_y ghosted -stag_boundary_type_z ghosted

   test:
      suffix: 10
      nsize: 5
      args: -stag_stencil_width 1 -stag_grid_y 2 -stag_grid_z 3 -stag_grid_x 17 -stag_boundary_type_y ghosted -stag_boundary_type_z ghosted -stag_ranks_x 5
TEST*/
