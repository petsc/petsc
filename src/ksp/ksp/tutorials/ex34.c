/*T
   Concepts: KSP^solving a system of linear equations
   Concepts: KSP^Laplacian, 3d
   Processors: n
T*/

/*
Laplacian in 3D. Modeled by the partial differential equation

   div  grad u = f,  0 < x,y,z < 1,

with pure Neumann boundary conditions

   u = 0 for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.

The functions are cell-centered

This uses multigrid to solve the linear system

       Contributed by Jianming Yang <jianming-yang@uiowa.edu>
*/

static char help[] = "Solves 3D Laplacian using multigrid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

extern PetscErrorCode ComputeMatrix(KSP,Mat,Mat,void*);
extern PetscErrorCode ComputeRHS(KSP,Vec,void*);

int main(int argc,char **argv)
{
  KSP            ksp;
  DM             da;
  PetscReal      norm;
  PetscErrorCode ierr;
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs,d,dof;
  PetscScalar    Hx,Hy,Hz;
  PetscScalar    ****array;
  Vec            x,b,r;
  Mat            J;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  dof  = 1;
  ierr = PetscOptionsGetInt(NULL,NULL,"-da_dof",&dof,NULL);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,12,12,12,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,1,0,0,0,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetInterpolationType(da, DMDA_Q0);CHKERRQ(ierr);

  ierr = KSPSetDM(ksp,da);CHKERRQ(ierr);

  ierr = KSPSetComputeRHS(ksp,ComputeRHS,NULL);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix,NULL);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,NULL,NULL);CHKERRQ(ierr);
  ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&b);CHKERRQ(ierr);
  ierr = KSPGetOperators(ksp,NULL,&J);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&r);CHKERRQ(ierr);

  ierr = MatMult(J,x,r);CHKERRQ(ierr);
  ierr = VecAXPY(r,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(r,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",(double)norm);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da, 0, &mx, &my, &mz, 0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx   = 1.0 / (PetscReal)(mx);
  Hy   = 1.0 / (PetscReal)(my);
  Hz   = 1.0 / (PetscReal)(mz);
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da, x, &array);CHKERRQ(ierr);

  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        for (d=0; d<dof; d++) {
          array[k][j][i][d] -=
            PetscCosScalar(2*PETSC_PI*(((PetscReal)i+0.5)*Hx))*
            PetscCosScalar(2*PETSC_PI*(((PetscReal)j+0.5)*Hy))*
            PetscCosScalar(2*PETSC_PI*(((PetscReal)k+0.5)*Hz));
        }
      }
    }
  }
  ierr = DMDAVecRestoreArrayDOF(da, x, &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = VecNorm(x,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error norm %g\n",(double)norm);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_1,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error norm %g\n",(double)(norm/((PetscReal)(mx)*(PetscReal)(my)*(PetscReal)(mz))));CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error norm %g\n",(double)(norm/((PetscReal)(mx)*(PetscReal)(my)*(PetscReal)(mz))));CHKERRQ(ierr);

  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       d,dof,i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs;
  PetscScalar    Hx,Hy,Hz;
  PetscScalar    ****array;
  DM             da;
  MatNullSpace   nullspace;

  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, 0, &mx, &my, &mz, 0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
  Hx   = 1.0 / (PetscReal)(mx);
  Hy   = 1.0 / (PetscReal)(my);
  Hz   = 1.0 / (PetscReal)(mz);
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFWrite(da, b, &array);CHKERRQ(ierr);
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        for (d=0; d<dof; d++) {
          array[k][j][i][d] = 12 * PETSC_PI * PETSC_PI
                           * PetscCosScalar(2*PETSC_PI*(((PetscReal)i+0.5)*Hx))
                           * PetscCosScalar(2*PETSC_PI*(((PetscReal)j+0.5)*Hy))
                           * PetscCosScalar(2*PETSC_PI*(((PetscReal)k+0.5)*Hz))
                           * Hx * Hy * Hz;
        }
      }
    }
  }
  ierr = DMDAVecRestoreArrayDOFWrite(da, b, &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */

  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceRemove(nullspace,b);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix(KSP ksp, Mat J,Mat jac, void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       dof,i,j,k,d,mx,my,mz,xm,ym,zm,xs,ys,zs,num, numi, numj, numk;
  PetscScalar    v[7],Hx,Hy,Hz,HyHzdHx,HxHzdHy,HxHydHz;
  MatStencil     row, col[7];
  DM             da;
  MatNullSpace   nullspace;
  PetscBool      dump_mat = PETSC_FALSE, check_matis = PETSC_FALSE;

  PetscFunctionBeginUser;
  ierr    = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr    = DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
  Hx      = 1.0 / (PetscReal)(mx);
  Hy      = 1.0 / (PetscReal)(my);
  Hz      = 1.0 / (PetscReal)(mz);
  HyHzdHx = Hy*Hz/Hx;
  HxHzdHy = Hx*Hz/Hy;
  HxHydHz = Hx*Hy/Hz;
  ierr    = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        for (d=0; d<dof; d++) {
          row.i = i; row.j = j; row.k = k; row.c = d;
          if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1) {
            num = 0; numi=0; numj=0; numk=0;
            if (k!=0) {
              v[num]     = -HxHydHz;
              col[num].i = i;
              col[num].j = j;
              col[num].k = k-1;
              col[num].c = d;
              num++; numk++;
            }
            if (j!=0) {
              v[num]     = -HxHzdHy;
              col[num].i = i;
              col[num].j = j-1;
              col[num].k = k;
              col[num].c = d;
              num++; numj++;
              }
            if (i!=0) {
              v[num]     = -HyHzdHx;
              col[num].i = i-1;
              col[num].j = j;
              col[num].k = k;
              col[num].c = d;
              num++; numi++;
            }
            if (i!=mx-1) {
              v[num]     = -HyHzdHx;
              col[num].i = i+1;
              col[num].j = j;
              col[num].k = k;
              col[num].c = d;
              num++; numi++;
            }
            if (j!=my-1) {
              v[num]     = -HxHzdHy;
              col[num].i = i;
              col[num].j = j+1;
              col[num].k = k;
              col[num].c = d;
              num++; numj++;
            }
            if (k!=mz-1) {
              v[num]     = -HxHydHz;
              col[num].i = i;
              col[num].j = j;
              col[num].k = k+1;
              col[num].c = d;
              num++; numk++;
            }
            v[num]     = (PetscReal)(numk)*HxHydHz + (PetscReal)(numj)*HxHzdHy + (PetscReal)(numi)*HyHzdHx;
            col[num].i = i;   col[num].j = j;   col[num].k = k; col[num].c = d;
            num++;
            ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
          } else {
            v[0] = -HxHydHz;                          col[0].i = i;   col[0].j = j;   col[0].k = k-1; col[0].c = d;
            v[1] = -HxHzdHy;                          col[1].i = i;   col[1].j = j-1; col[1].k = k;   col[1].c = d;
            v[2] = -HyHzdHx;                          col[2].i = i-1; col[2].j = j;   col[2].k = k;   col[2].c = d;
            v[3] = 2.0*(HyHzdHx + HxHzdHy + HxHydHz); col[3].i = i;   col[3].j = j;   col[3].k = k;   col[3].c = d;
            v[4] = -HyHzdHx;                          col[4].i = i+1; col[4].j = j;   col[4].k = k;   col[4].c = d;
            v[5] = -HxHzdHy;                          col[5].i = i;   col[5].j = j+1; col[5].k = k;   col[5].c = d;
            v[6] = -HxHydHz;                          col[6].i = i;   col[6].j = j;   col[6].k = k+1; col[6].c = d;
            ierr = MatSetValuesStencil(jac,1,&row,7,col,v,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-dump_mat",&dump_mat,NULL);CHKERRQ(ierr);
  if (dump_mat) {
    Mat JJ;

    ierr = MatComputeOperator(jac,MATAIJ,&JJ);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = MatView(JJ,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatDestroy(&JJ);CHKERRQ(ierr);
  }
  ierr = MatViewFromOptions(jac,NULL,"-view_mat");CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-check_matis",&check_matis,NULL);CHKERRQ(ierr);
  if (check_matis) {
    void      (*f)(void);
    Mat       J2;
    MatType   jtype;
    PetscReal nrm;

    ierr = MatGetType(jac,&jtype);CHKERRQ(ierr);
    ierr = MatConvert(jac,MATIS,MAT_INITIAL_MATRIX,&J2);CHKERRQ(ierr);
    ierr = MatViewFromOptions(J2,NULL,"-view_conv");CHKERRQ(ierr);
    ierr = MatConvert(J2,jtype,MAT_INPLACE_MATRIX,&J2);CHKERRQ(ierr);
    ierr = MatGetOperation(jac,MATOP_VIEW,&f);CHKERRQ(ierr);
    ierr = MatSetOperation(J2,MATOP_VIEW,f);CHKERRQ(ierr);
    ierr = MatSetDM(J2,da);CHKERRQ(ierr);
    ierr = MatViewFromOptions(J2,NULL,"-view_conv_assembled");CHKERRQ(ierr);
    ierr = MatAXPY(J2,-1.,jac,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(J2,NORM_FROBENIUS,&nrm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error MATIS %g\n",(double)nrm);CHKERRQ(ierr);
    ierr = MatViewFromOptions(J2,NULL,"-view_conv_err");CHKERRQ(ierr);
    ierr = MatDestroy(&J2);CHKERRQ(ierr);
  }
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(J,nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -pc_type mg -pc_mg_type full -ksp_type fgmres -ksp_monitor_short -pc_mg_levels 3 -mg_coarse_pc_factor_shift_type nonzero -ksp_view

   test:
      suffix: 2
      nsize: 2
      args: -ksp_monitor_short -da_grid_x 50 -da_grid_y 50 -pc_type ksp -ksp_ksp_type cg -ksp_pc_type bjacobi -ksp_ksp_rtol 1e-1 -ksp_ksp_monitor -ksp_type pipefgmres -ksp_gmres_restart 5

   test:
      suffix: hyprestruct
      nsize: 3
      requires: hypre
      args: -ksp_type gmres -pc_type pfmg -dm_mat_type hyprestruct -ksp_monitor -da_refine 3

TEST*/
