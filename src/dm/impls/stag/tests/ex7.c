static char help[] = "Test DMStag 3d periodic and ghosted boundary conditions\n\n";
#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  DM              dm;
  Vec             vec,vecLocal1,vecLocal2;
  PetscScalar     *a,****a1,****a2,expected;
  PetscInt        startx,starty,startz,nx,ny,nz,i,j,k,d,is,js,ks,dof0,dof1,dof2,dof3,dofTotal,stencilWidth,Nx,Ny,Nz;
  DMBoundaryType  boundaryTypex,boundaryTypey,boundaryTypez;
  PetscMPIInt     rank;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  dof0 = 1;
  dof1 = 1;
  dof2 = 1;
  dof3 = 1;
  stencilWidth = 2;
  PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,4,4,4,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof0,dof1,dof2,dof3,DMSTAG_STENCIL_BOX,stencilWidth,NULL,NULL,NULL,&dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagGetDOF(dm,&dof0,&dof1,&dof2,&dof3));
  dofTotal = dof0 + 3*dof1 + 3*dof2 + dof3;
  PetscCall(DMStagGetStencilWidth(dm,&stencilWidth));

  PetscCall(DMCreateLocalVector(dm,&vecLocal1));
  PetscCall(VecDuplicate(vecLocal1,&vecLocal2));

  PetscCall(DMCreateGlobalVector(dm,&vec));
  PetscCall(VecSet(vec,1.0));
  PetscCall(VecSet(vecLocal1,0.0));
  PetscCall(DMGlobalToLocalBegin(dm,vec,INSERT_VALUES,vecLocal1));
  PetscCall(DMGlobalToLocalEnd(dm,vec,INSERT_VALUES,vecLocal1));

  PetscCall(DMStagGetCorners(dm,&startx,&starty,&startz,&nx,&ny,&nz,NULL,NULL,NULL));
  PetscCall(DMStagVecGetArrayRead(dm,vecLocal1,&a1));
  PetscCall(DMStagVecGetArray(dm,vecLocal2,&a2));
  for (k=startz; k<startz + nz; ++k) {
    for (j=starty; j<starty + ny; ++j) {
      for (i=startx; i<startx + nx; ++i) {
        for (d=0; d<dofTotal; ++d) {
          if (a1[k][j][i][d] != 1.0) {
            PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] Unexpected value %g (expecting %g)\n",rank,(double)PetscRealPart(a1[k][j][i][d]),1.0));
          }
          a2[k][j][i][d] = 0.0;
          for (ks = -stencilWidth; ks <= stencilWidth; ++ks) {
            for (js = -stencilWidth; js <= stencilWidth; ++js) {
              for (is = -stencilWidth; is <= stencilWidth; ++is) {
                a2[k][j][i][d] += a1[k+ks][j+js][i+is][d];
              }
            }
          }
        }
      }
    }
  }
  PetscCall(DMStagVecRestoreArrayRead(dm,vecLocal1,&a1));
  PetscCall(DMStagVecRestoreArray(dm,vecLocal2,&a2));

  PetscCall(DMLocalToGlobalBegin(dm,vecLocal2,INSERT_VALUES,vec));
  PetscCall(DMLocalToGlobalEnd(dm,vecLocal2,INSERT_VALUES,vec));

  /* For the all-periodic case, all values are the same . Otherwise, just check the local version */
  PetscCall(DMStagGetBoundaryTypes(dm,&boundaryTypex,&boundaryTypey,&boundaryTypez));
  if (boundaryTypex == DM_BOUNDARY_PERIODIC && boundaryTypey == DM_BOUNDARY_PERIODIC && boundaryTypez == DM_BOUNDARY_PERIODIC) {
    PetscCall(VecGetArray(vec,&a));
    expected = 1.0; for (d=0;d<3;++d) expected *= (2*stencilWidth+1);
    for (i=0; i<nz*ny*nx*dofTotal; ++i) {
      if (a[i] != expected) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] Unexpected value %g (expecting %g)\n",rank,(double)PetscRealPart(a[i]),(double)PetscRealPart(expected)));
      }
    }
    PetscCall(VecRestoreArray(vec,&a));
  } else {
    PetscCall(DMStagVecGetArrayRead(dm,vecLocal2,&a2));
    PetscCall(DMStagGetGlobalSizes(dm,&Nx,&Ny,&Nz));
    PetscCheck(stencilWidth <= 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Check implemented assuming stencilWidth = 1");
    for (k=startz; k<startz + nz; ++k) {
      for (j=starty; j<starty + ny; ++j) {
        for (i=startx; i<startx + nx; ++i) {
          PetscInt  dd,extra[3];
          PetscBool bnd[3];
          bnd[0] = (PetscBool)((i == 0 || i == Nx-1) && boundaryTypex != DM_BOUNDARY_PERIODIC);
          bnd[1] = (PetscBool)((j == 0 || j == Ny-1) && boundaryTypey != DM_BOUNDARY_PERIODIC);
          bnd[2] = (PetscBool)((k == 0 || k == Nz-1) && boundaryTypez != DM_BOUNDARY_PERIODIC);
          extra[0] = i == Nx-1 && boundaryTypex != DM_BOUNDARY_PERIODIC ? 1 : 0;
          extra[1] = j == Ny-1 && boundaryTypey != DM_BOUNDARY_PERIODIC ? 1 : 0;
          extra[2] = k == Nz-1 && boundaryTypez != DM_BOUNDARY_PERIODIC ? 1 : 0;
          { /* vertices */
            PetscScalar expected = 1.0;
            for (dd=0;dd<3;++dd) expected *= (bnd[dd] ? stencilWidth + 1 + extra[dd] : 2*stencilWidth + 1);
            for (d=0; d<dof0; ++d) {
              if (a2[k][j][i][d] != expected) {
                PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] Element (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ")[%" PetscInt_FMT "] Unexpected value %g (expecting %g)\n",rank,i,j,k,d,(double)PetscRealPart(a2[k][j][i][d]),(double)PetscRealPart(expected)));
              }
            }
          }
          { /* back down edges */
            PetscScalar expected = ((bnd[0] ? 1 : 2) * stencilWidth + 1);
            for (dd=1;dd<3;++dd) expected *= (bnd[dd] ? stencilWidth + 1 + extra[dd] : 2*stencilWidth + 1);
            for (d=dof0; d<dof0+dof1; ++d) {
              if (a2[k][j][i][d] != expected) {
                PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] Element (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ")[%" PetscInt_FMT "] Unexpected value %g (expecting %g)\n",rank,i,j,k,d,(double)PetscRealPart(a2[k][j][i][d]),(double)PetscRealPart(expected)));
              }
            }
          }
          { /* back left edges */
            PetscScalar expected = ((bnd[1] ? 1 : 2) * stencilWidth + 1);
            for (dd=0;dd<3;dd+=2) expected *= (bnd[dd] ? stencilWidth + 1 + extra[dd] : 2*stencilWidth + 1);
            for (d=dof0+dof1; d<dof0+2*dof1; ++d) {
              if (a2[k][j][i][d] != expected) {
                PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] Element (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ")[%" PetscInt_FMT "] Unexpected value %g (expecting %g)\n",rank,i,j,k,d,(double)PetscRealPart(a2[k][j][i][d]),(double)PetscRealPart(expected)));
              }
            }
          }
          { /* back faces */
            PetscScalar expected = (bnd[2] ? stencilWidth + 1 + extra[2] : 2*stencilWidth + 1);
            for (dd=0;dd<2;++dd) expected *= ((bnd[dd] ? 1 : 2) * stencilWidth + 1);
            for (d=dof0+2*dof1; d<dof0+2*dof1+dof2; ++d) {
              if (a2[k][j][i][d] != expected) {
                PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] Element (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ")[%" PetscInt_FMT "] Unexpected value %g (expecting %g)\n",rank,i,j,k,d,(double)PetscRealPart(a2[k][j][i][d]),(double)PetscRealPart(expected)));
              }
            }
          }
          { /* down left edges */
            PetscScalar expected = ((bnd[2] ? 1 : 2) * stencilWidth + 1);
            for (dd=0;dd<2;++dd) expected *= (bnd[dd] ? stencilWidth + 1 + extra[dd] : 2*stencilWidth + 1);
            for (d=dof0+2*dof1+dof2; d<dof0+3*dof1+dof2; ++d) {
              if (a2[k][j][i][d] != expected) {
                PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] Element (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ")[%" PetscInt_FMT "] Unexpected value %g (expecting %g)\n",rank,i,j,k,d,(double)PetscRealPart(a2[k][j][i][d]),(double)PetscRealPart(expected)));
              }
            }
          }
          { /* down faces */
            PetscScalar expected = (bnd[1] ? stencilWidth + 1 + extra[1] : 2*stencilWidth + 1);
            for (dd=0;dd<3;dd+=2) expected *= ((bnd[dd] ? 1 : 2) * stencilWidth + 1);
            for (d=dof0+3*dof1+dof2; d<dof0+3*dof1+2*dof2; ++d) {
              if (a2[k][j][i][d] != expected) {
                PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] Element (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ")[%" PetscInt_FMT "] Unexpected value %g (expecting %g)\n",rank,i,j,k,d,(double)PetscRealPart(a2[k][j][i][d]),(double)PetscRealPart(expected)));
              }
            }
          }
          { /* left faces */
            PetscScalar expected = (bnd[0] ? stencilWidth + 1 + extra[0] : 2*stencilWidth + 1);
            for (dd=1;dd<3;++dd) expected *= ((bnd[dd] ? 1 : 2) * stencilWidth + 1);
            for (d=dof0+3*dof1+2*dof2; d<dof0+3*dof1+3*dof2; ++d) {
              if (a2[k][j][i][d] != expected) {
                PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] Element (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ")[%" PetscInt_FMT "] Unexpected value %g (expecting %g)\n",rank,i,j,k,d,(double)PetscRealPart(a2[k][j][i][d]),(double)PetscRealPart(expected)));
              }
            }
          }
          { /* elements */
            PetscScalar expected = 1.0;
            for (dd=0;dd<3;++dd) expected *= ((bnd[dd] ? 1 : 2) * stencilWidth + 1);
            for (d=dofTotal-dof3; d<dofTotal; ++d) {
              if (a2[k][j][i][d] != expected) {
                PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] Element (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ")[%" PetscInt_FMT "] Unexpected value %g (expecting %g)\n",rank,i,j,k,d,(double)PetscRealPart(a2[k][j][i][d]),(double)PetscRealPart(expected)));
              }
            }
          }
        }
      }
    }
    PetscCall(DMStagVecRestoreArrayRead(dm,vecLocal2,&a2));
  }

  PetscCall(VecDestroy(&vec));
  PetscCall(VecDestroy(&vecLocal1));
  PetscCall(VecDestroy(&vecLocal2));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 8
      args: -stag_ranks_x 2 -stag_ranks_y 2 -stag_ranks_z 2 -stag_stencil_width 1 -stag_dof_3 2 -stag_grid_z 3

   test:
      suffix: 2
      nsize: 8
      args: -stag_ranks_x 2 -stag_ranks_y 2 -stag_ranks_z 2 -stag_dof_2 2 -stag_grid_y 5

   test:
      suffix: 3
      nsize: 12
      args: -stag_ranks_x 3 -stag_ranks_y 2 -stag_ranks_z 2 -stag_dof_0 2 -stag_grid_x 6

   test:
      suffix: 4
      nsize: 12
      args: -stag_ranks_x 3 -stag_ranks_y 2 -stag_ranks_z 2 -stag_dof_0 0 -stag_dof_1 0 -stag_dof_2 0 -stag_grid_x 4 -stag_boundary_type_x ghosted -stag_boundary_type_y ghosted -stag_boundary_type_z ghosted -stag_stencil_width 1

   test:
      suffix: 5
      nsize: 12
      args: -stag_ranks_x 3 -stag_ranks_y 2 -stag_ranks_z 2 -stag_dof_0 0 -stag_dof_1 0 -stag_dof_2 0 -stag_grid_x 4 -stag_boundary_type_x ghosted -stag_boundary_type_z ghosted -stag_stencil_width 1

   test:
      suffix: 6
      nsize: 8
      args: -stag_dof_0 3 -stag_dof_1 2 -stag_dof_2 4 -stag_dof_3 2 -stag_boundary_type_y ghosted -stag_boundary_type_z ghosted -stag_stencil_width 1
TEST*/
