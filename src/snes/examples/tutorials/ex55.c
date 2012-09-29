static char help[] = "Allen-Cahn-2d problem for constant mobility and triangular elements.\n\
Runtime options include:\n\
-xmin <xmin>\n\
-xmax <xmax>\n\
-ymin <ymin>\n\
-T <T>, where <T> is the end time for the time domain simulation\n\
-dt <dt>,where <dt> is the step size for the numerical integration\n\
-gamma <gamma>\n\
-theta_c <theta_c>\n\n";

/*
  Solves the linear system using a Schur complement solver based on PCLSC, solve the A00 block with hypre BoomerAMG

  ./ex55 -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_detect_saddle_point -pc_fieldsplit_type schur -pc_fieldsplit_schur_precondition self -fieldsplit_1_ksp_type fgmres -fieldsplit_1_pc_type lsc -snes_vi_monitor -ksp_monitor_true_residual -fieldsplit_ksp_monitor -fieldsplit_0_pc_type hypre -da_grid_x 65 -da_grid_y 65 -snes_atol 1.e-11  -ksp_rtol 1.e-8

  Solves the linear systems with multigrid on the entire system using  a Schur complement based smoother (which is handled by PCFIELDSPLIT)

./ex55 -ksp_type fgmres -pc_type mg -mg_levels_ksp_type fgmres -mg_levels_pc_type fieldsplit -mg_levels_pc_fieldsplit_detect_saddle_point -mg_levels_pc_fieldsplit_type schur -mg_levels_pc_fieldsplit_factorization_type full -mg_levels_pc_fieldsplit_schur_precondition user -mg_levels_fieldsplit_1_ksp_type gmres -mg_levels_fieldsplit_1_pc_type none -mg_levels_fieldsplit_0_ksp_type preonly -mg_levels_fieldsplit_0_pc_type sor -mg_levels_fieldsplit_0_pc_sor_forward -snes_vi_monitor -ksp_monitor_true_residual -pc_mg_levels 5 -pc_mg_galerkin -mg_levels_ksp_monitor -mg_levels_fieldsplit_ksp_monitor -mg_levels_ksp_max_it 2 -mg_levels_fieldsplit_ksp_max_it 5 -snes_atol 1.e-11 -mg_coarse_ksp_type preonly -mg_coarse_pc_type svd -da_grid_x 65 -da_grid_y 65 -ksp_rtol 1.e-8

 */

#include "petscsnes.h"
#include "petscdmda.h"

typedef struct{
  PetscReal   dt,T; /* Time step and end time */
  DM          da;
  Mat         M;    /* Jacobian matrix */
  Mat         M_0;
  Vec         q,u1,u2,u3,work1,work2,work3,work4;
  PetscScalar epsilon; /* physics parameters */
  PetscReal   xmin,xmax,ymin,ymax;
}AppCtx;

PetscErrorCode GetParams(AppCtx*);
PetscErrorCode SetVariableBounds(DM,Vec,Vec);
PetscErrorCode SetUpMatrices(AppCtx*);
PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
PetscErrorCode FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
PetscErrorCode SetInitialGuess(Vec,AppCtx*);
PetscErrorCode Update_q(Vec,Vec,Vec,Vec,Mat,AppCtx*);
PetscErrorCode Update_u(Vec,Vec,Vec,Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  Vec            x,r;  /* Solution and residual vectors */
  SNES           snes; /* Nonlinear solver context */
  AppCtx         user; /* Application context */
  Vec            xl,xu; /* Upper and lower bounds on variables */
  Mat            J;
  PetscScalar    t=0.0;
  PetscViewer view_out, view_q, view1;

  PetscInitialize(&argc,&argv, (char*)0, help);

  /* Get physics and time parameters */
  ierr = GetParams(&user);CHKERRQ(ierr);
  /* Create a 2D DA with dof = 2 */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,-4,-4,PETSC_DECIDE,PETSC_DECIDE,4,1,PETSC_NULL,PETSC_NULL,&user.da);CHKERRQ(ierr);
  /* Set Element type (triangular) */
  ierr = DMDASetElementType(user.da,DMDA_ELEMENT_P1);CHKERRQ(ierr);

  /* Set x and y coordinates */
  ierr = DMDASetUniformCoordinates(user.da,user.xmin,user.xmax,user.ymin,user.ymax,0.0,1.0);CHKERRQ(ierr);

  /* Get global vector x from DM and duplicate vectors r,xl,xu */
  ierr = DMCreateGlobalVector(user.da,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xl);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xu);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&user.q);CHKERRQ(ierr);

  /* Get Jacobian matrix structure from the da */
  ierr = DMCreateMatrix(user.da,MATAIJ,&user.M);CHKERRQ(ierr);
  /* Form the jacobian matrix and M_0 */
  ierr = SetUpMatrices(&user);CHKERRQ(ierr);
  ierr = MatDuplicate(user.M,MAT_DO_NOT_COPY_VALUES,&J);CHKERRQ(ierr);

  /* Create nonlinear solver context */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,user.da);CHKERRQ(ierr);

  /* Set Function evaluation and jacobian evaluation routines */
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,(void*)&user);CHKERRQ(ierr);

  /* Set the boundary conditions */
  ierr = SetVariableBounds(user.da,xl,xu);CHKERRQ(ierr);
  ierr = SNESVISetVariableBounds(snes,xl,xu);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = SetInitialGuess(x,&user);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_out",FILE_MODE_WRITE,&view_out);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_q",FILE_MODE_WRITE,&view_q);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file1",FILE_MODE_WRITE,&view1);CHKERRQ(ierr);
  /* Begin time loop */
  while(t < user.T) {
    ierr = VecView(user.u1,view1);CHKERRQ(ierr);
    ierr = VecView(user.u2,view1);CHKERRQ(ierr);
    ierr = VecView(user.u3,view1);CHKERRQ(ierr);

    ierr = Update_q(user.q,user.u1,user.u2,user.u3,user.M_0,&user);
    ierr = VecView(user.q,view_q);CHKERRQ(ierr);
    ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
    PetscInt its;
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"SNESVI solver converged at t = %5.4f in %d iterations\n",t,its);CHKERRQ(ierr);
    ierr = Update_u(user.u1,user.u2,user.u3,x);CHKERRQ(ierr);
    t = t + user.dt;
    ierr = VecView(user.u1,view_out);CHKERRQ(ierr);
    ierr = VecView(user.u2,view_out);CHKERRQ(ierr);
    ierr = VecView(user.u3,view_out);CHKERRQ(ierr);
  }



  PetscViewerDestroy(&view_out);
  PetscViewerDestroy(&view_q);
  PetscViewerDestroy(&view1);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&xl);CHKERRQ(ierr);
  ierr = VecDestroy(&xu);CHKERRQ(ierr);
  ierr = VecDestroy(&user.q);CHKERRQ(ierr);
  ierr = VecDestroy(&user.u1);CHKERRQ(ierr);
  ierr = VecDestroy(&user.u2);CHKERRQ(ierr);
  ierr = VecDestroy(&user.u3);CHKERRQ(ierr);
  ierr = VecDestroy(&user.work1);CHKERRQ(ierr);
  ierr = VecDestroy(&user.work2);CHKERRQ(ierr);
  ierr = VecDestroy(&user.work3);CHKERRQ(ierr);
  ierr = VecDestroy(&user.work4);CHKERRQ(ierr);
  ierr = MatDestroy(&user.M);CHKERRQ(ierr);
  ierr = MatDestroy(&user.M_0);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "Update_u"
PetscErrorCode Update_u(Vec u1,Vec u2,Vec u3,Vec X)
{
  PetscErrorCode ierr;
  PetscInt       i,n;
  PetscScalar    *u1_arr,*u2_arr,*u3_arr,*x_arr;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(u1,&n);CHKERRQ(ierr);
  ierr = VecGetArray(u1,&u1_arr);CHKERRQ(ierr);
  ierr = VecGetArray(u2,&u2_arr);CHKERRQ(ierr);
  ierr = VecGetArray(u3,&u3_arr);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x_arr);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    u1_arr[i] = x_arr[4*i];
    u2_arr[i] = x_arr[4*i+1];
    u3_arr[i] = x_arr[4*i+2];
  }
  ierr = VecRestoreArray(u1,&u1_arr);CHKERRQ(ierr);
  ierr = VecRestoreArray(u2,&u2_arr);CHKERRQ(ierr);
  ierr = VecRestoreArray(u3,&u3_arr);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x_arr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Update_q"
PetscErrorCode Update_q(Vec q,Vec u1,Vec u2,Vec u3,Mat M_0,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscScalar    *q_arr,*w_arr;
  PetscInt       i,n;
  //PetscViewer    view_q;

  PetscFunctionBegin;
  ierr = VecSet(user->work1,user->dt/3);CHKERRQ(ierr);
  //    ierr = VecView(user->work1,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatMult(M_0,user->work1,user->work2);CHKERRQ(ierr);
  //    ierr = VecView(user->work2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatMult(M_0,u1,user->work1);CHKERRQ(ierr);
  ierr = MatMult(M_0,u1,user->work4);CHKERRQ(ierr);
  //    ierr = VecView(u1,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //    ierr = VecView(user->work4,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecScale(user->work1,-1.0-(user->dt));CHKERRQ(ierr);
  ierr = VecAXPY(user->work1,1.0,user->work2);CHKERRQ(ierr);

  ierr = VecGetLocalSize(u1,&n);CHKERRQ(ierr);
  ierr = VecGetArray(q,&q_arr);CHKERRQ(ierr);
  ierr = VecGetArray(user->work1,&w_arr);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    q_arr[4*i] = w_arr[i];
  }
  ierr = VecRestoreArray(q,&q_arr);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->work1,&w_arr);CHKERRQ(ierr);

  ierr = MatMult(M_0,u2,user->work1);CHKERRQ(ierr);
  ierr = VecScale(user->work1,-1.0-(user->dt));CHKERRQ(ierr);
  ierr = VecAXPY(user->work1,1.0,user->work2);CHKERRQ(ierr);

  ierr = VecGetArray(q,&q_arr);CHKERRQ(ierr);
  ierr = VecGetArray(user->work1,&w_arr);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    q_arr[4*i+1] = w_arr[i];
  }
  ierr = VecRestoreArray(q,&q_arr);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->work1,&w_arr);CHKERRQ(ierr);

  ierr = MatMult(M_0,u3,user->work1);CHKERRQ(ierr);
  ierr = VecScale(user->work1,-1.0-(user->dt));CHKERRQ(ierr);
  ierr = VecAXPY(user->work1,1.0,user->work2);CHKERRQ(ierr);

  ierr = VecGetArray(q,&q_arr);CHKERRQ(ierr);
  ierr = VecGetArray(user->work1,&w_arr);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
    q_arr[4*i+2] = w_arr[i];
    q_arr[4*i+3] = 1.0;
  }
  ierr = VecRestoreArray(q,&q_arr);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->work1,&w_arr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetInitialGuess"
PetscErrorCode SetInitialGuess(Vec X,AppCtx* user)
{
        PetscErrorCode    ierr;
        PetscInt          nele,nen,n,i;
        const PetscInt    *ele;
        Vec               coords, rand1, rand2;
        const PetscScalar *_coords;
        PetscScalar       x[3],y[3];
        PetscInt          idx[3];
        PetscScalar        *xx,*w1,*w2,*u1,*u2,*u3;
        PetscViewer               view_out;

        PetscFunctionBegin;
        /* Get ghosted coordinates */
        ierr = DMGetCoordinatesLocal(user->da,&coords);CHKERRQ(ierr);
        ierr = VecDuplicate(user->u1,&rand1);
        ierr = VecDuplicate(user->u1,&rand2);
        ierr = VecSetRandom(rand1,PETSC_NULL);
        ierr = VecSetRandom(rand2,PETSC_NULL);

        ierr = VecGetLocalSize(X,&n);CHKERRQ(ierr);
        ierr = VecGetArrayRead(coords,&_coords);CHKERRQ(ierr);
        ierr = VecGetArray(X,&xx);CHKERRQ(ierr);
        ierr = VecGetArray(user->work1,&w1);
        ierr = VecGetArray(user->work2,&w2);
        ierr = VecGetArray(user->u1,&u1);
        ierr = VecGetArray(user->u2,&u2);
        ierr = VecGetArray(user->u3,&u3);

        /* Get local element info */
        ierr = DMDAGetElements(user->da,&nele,&nen,&ele);CHKERRQ(ierr);
        for (i=0;i < nele;i++) {
                idx[0] = ele[3*i]; idx[1] = ele[3*i+1]; idx[2] = ele[3*i+2];
                x[0] = _coords[2*idx[0]]; y[0] = _coords[2*idx[0]+1];
                x[1] = _coords[2*idx[1]]; y[1] = _coords[2*idx[1]+1];
                x[2] = _coords[2*idx[2]]; y[2] = _coords[2*idx[2]+1];

                PetscScalar vals1[3],vals2[3],valsrand[3];
                PetscInt r;
                for (r=0;r<3;r++) {
                        valsrand[r]=5*x[r]*(1-x[r])*y[r]*(1-y[r]);
                        if (x[r]>=0.5 && y[r]>=0.5){
                                vals1[r]=0.75;
                                vals2[r]=0.0;
                        }
                        if (x[r]>=0.5 && y[r]<0.5){
                                vals1[r]=0.0;
                                vals2[r]=0.0;
                        }
                        if (x[r]<0.5 && y[r]>=0.5){
                                vals1[r]=0.0;
                                vals2[r]=0.75;
                        }
                        if (x[r]<0.5 && y[r]<0.5){
                                vals1[r]=0.75;
                                vals2[r]=0.0;
                        }
                }

                ierr = VecSetValues(user->work1,3,idx,vals1,INSERT_VALUES);CHKERRQ(ierr);
                ierr = VecSetValues(user->work2,3,idx,vals2,INSERT_VALUES);CHKERRQ(ierr);
                ierr = VecSetValues(user->work3,3,idx,valsrand,INSERT_VALUES);CHKERRQ(ierr);
        }

        ierr = VecAssemblyBegin(user->work1);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(user->work1);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(user->work2);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(user->work2);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(user->work3);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(user->work3);CHKERRQ(ierr);

        ierr = VecAXPY(user->work1,1.0,user->work3);CHKERRQ(ierr);
        ierr = VecAXPY(user->work2,1.0,user->work3);CHKERRQ(ierr);

        for (i=0;i<n/4;i++) {
                xx[4*i] = w1[i];
                if (xx[4*i]>1) {
                        xx[4*i]=1;
                }
                xx[4*i+1] = w2[i];
                if (xx[4*i+1]>1) {
                        xx[4*i+1]=1;
                }
                if (xx[4*i]+xx[4*i+1]>1){
                        xx[4*i+1] = 1.0 - xx[4*i];
                }
                xx[4*i+2] = 1.0 - xx[4*i] - xx[4*i+1];
                xx[4*i+3] = 0.0;

                u1[i] = xx[4*i];
                u2[i] = xx[4*i+1];
                u3[i] = xx[4*i+2];
        }

        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_initial",FILE_MODE_WRITE,&view_out);CHKERRQ(ierr);
        ierr = VecView(user->u1,view_out);CHKERRQ(ierr);
        ierr = VecView(user->u2,view_out);CHKERRQ(ierr);
        ierr = VecView(user->u3,view_out);CHKERRQ(ierr);
        PetscViewerDestroy(&view_out);

        ierr = DMDARestoreElements(user->da,&nele,&nen,&ele);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(coords,&_coords);CHKERRQ(ierr);
        ierr = VecRestoreArray(X,&xx);CHKERRQ(ierr);
        ierr = VecRestoreArray(user->work2,&w1);CHKERRQ(ierr);
        ierr = VecRestoreArray(user->work4,&w2);CHKERRQ(ierr);
        ierr = VecRestoreArray(user->u1,&u1);CHKERRQ(ierr);
        ierr = VecRestoreArray(user->u2,&u2);CHKERRQ(ierr);
        ierr = VecRestoreArray(user->u3,&u3);CHKERRQ(ierr);
        ierr = VecDestroy(&rand1);CHKERRQ(ierr);
        ierr = VecDestroy(&rand2);CHKERRQ(ierr);
        PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void* ctx)
{
  PetscErrorCode ierr;
  AppCtx         *user=(AppCtx*)ctx;

  PetscFunctionBegin;
  ierr = MatMultAdd(user->M,X,user->q,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
PetscErrorCode FormJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flg,void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *user=(AppCtx*)ctx;

  PetscFunctionBegin;
  *flg = SAME_NONZERO_PATTERN;
  ierr = MatCopy(user->M,*J,*flg);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetVariableBounds"
PetscErrorCode SetVariableBounds(DM da,Vec xl,Vec xu)
{
  PetscErrorCode ierr;
  PetscScalar    ***l,***u;
  PetscInt       xs,xm,ys,ym;
  PetscInt       j,i;

  PetscFunctionBegin;

  ierr = DMDAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,xl,&l);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,xu,&u);CHKERRQ(ierr);
  for (j=ys; j < ys+ym; j++) {
    for (i=xs; i < xs+xm;i++) {
      l[j][i][0] = 0.0;
      l[j][i][1] = 0.0;
      l[j][i][2] = 0.0;
      l[j][i][3] = -SNES_VI_INF;
      u[j][i][0] = 1.0;
      u[j][i][1] = 1.0;
      u[j][i][2] = 1.0;
      u[j][i][3] = SNES_VI_INF;
    }
  }
  ierr = DMDAVecRestoreArrayDOF(da,xl,&l);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(da,xu,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetParams"
PetscErrorCode GetParams(AppCtx* user)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;

  /* Set default parameters */
  user->xmin = 0.0; user->xmax = 1.0;
  user->ymin = 0.0; user->ymax = 1.0;
  user->T = 0.2;    user->dt = 0.001;
  user->epsilon = 0.05;

  ierr = PetscOptionsGetReal(PETSC_NULL,"-xmin",&user->xmin,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-xmax",&user->xmax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-ymin",&user->ymin,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-ymax",&user->ymax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-T",&user->T,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-dt",&user->dt,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(PETSC_NULL,"-epsilon",&user->epsilon,&flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetUpMatrices"
PetscErrorCode SetUpMatrices(AppCtx* user)
{
  PetscErrorCode    ierr;
  PetscInt          nele,nen,i,j;
  const PetscInt    *ele;
  PetscScalar       dt=user->dt;
  Vec               coords;
  const PetscScalar *_coords;
  PetscScalar       x[3],y[3];
  PetscInt          idx[3];
  PetscScalar       eM_0[3][3],eM_2_odd[3][3],eM_2_even[3][3];
  Mat               M=user->M;
  PetscScalar       epsilon=user->epsilon;
  PetscScalar             hx;
  PetscInt n,Mda,Nda;
  DM               da;

  PetscFunctionBegin;
  /* Get ghosted coordinates */
  ierr = DMGetCoordinatesLocal(user->da,&coords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords,&_coords);CHKERRQ(ierr);

  /* Create the mass matrix M_0 */
  ierr = MatGetLocalSize(M,&n,PETSC_NULL);CHKERRQ(ierr);


  /* ierr = MatCreate(PETSC_COMM_WORLD,&user->M_0);CHKERRQ(ierr);*/
  ierr = DMDAGetInfo(user->da,PETSC_NULL,&Mda,&Nda,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);
  hx = 1.0/(Mda-1);
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,Mda,Nda,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,MATAIJ,&user->M_0);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  eM_0[0][0]=eM_0[1][1]=eM_0[2][2]=hx*hx/12.0;
  eM_0[0][1]=eM_0[0][2]=eM_0[1][0]=eM_0[1][2]=eM_0[2][0]=eM_0[2][1]=hx*hx/24.0;

  eM_2_odd[0][0] = eM_2_odd[0][1] = eM_2_odd[0][2] = 0.0;
  eM_2_odd[1][0] = eM_2_odd[1][1] = eM_2_odd[1][2] = 0.0;
  eM_2_odd[2][0] = eM_2_odd[2][1] = eM_2_odd[2][2] = 0.0;

  eM_2_odd[0][0]=1.0;
  eM_2_odd[1][1]=eM_2_odd[2][2]=0.5;
  eM_2_odd[0][1]=eM_2_odd[0][2]=eM_2_odd[1][0]=eM_2_odd[2][0]=-0.5;

  eM_2_even[0][0] = eM_2_even[0][1] = eM_2_even[0][2] = 0.0;
  eM_2_even[0][0] = eM_2_even[0][1] = eM_2_even[0][2] = 0.0;
  eM_2_even[0][0] = eM_2_even[0][1] = eM_2_even[0][2] = 0.0;

  eM_2_even[1][1]=1;
  eM_2_even[0][0]=eM_2_even[2][2]=0.5;
  eM_2_even[0][1]=eM_2_even[1][0]=eM_2_even[1][2]=eM_2_even[2][1]=-0.5;

  /* Get local element info */
  ierr = DMDAGetElements(user->da,&nele,&nen,&ele);CHKERRQ(ierr);
  for (i=0;i < nele;i++) {
    idx[0] = ele[3*i]; idx[1] = ele[3*i+1]; idx[2] = ele[3*i+2];
    x[0] = _coords[2*idx[0]]; y[0] = _coords[2*idx[0]+1];
    x[1] = _coords[2*idx[1]]; y[1] = _coords[2*idx[1]+1];
    x[2] = _coords[2*idx[2]]; y[2] = _coords[2*idx[2]+1];

    PetscInt    row,cols[3],r,row_M_0;
    PetscScalar vals[3],vals_M_0[3];

    for (r=0;r<3;r++) {
      row_M_0 = idx[r];

      vals_M_0[0]=eM_0[r][0];
      vals_M_0[1]=eM_0[r][1];
      vals_M_0[2]=eM_0[r][2];

      ierr = MatSetValues(user->M_0,1,&row_M_0,3,idx,vals_M_0,ADD_VALUES);CHKERRQ(ierr);

      if (y[1]==y[0]) {
        row = 4*idx[r];
        cols[0] = 4*idx[0];     vals[0] = eM_0[r][0]+dt*epsilon*epsilon*eM_2_odd[r][0];
        cols[1] = 4*idx[1];     vals[1] = eM_0[r][1]+dt*epsilon*epsilon*eM_2_odd[r][1];
        cols[2] = 4*idx[2];     vals[2] = eM_0[r][2]+dt*epsilon*epsilon*eM_2_odd[r][2];
        /* Insert values in matrix M for 1st dof */
        ierr = MatSetValuesLocal(M,1,&row,3,cols,vals,ADD_VALUES);CHKERRQ(ierr);

        row = 4*idx[r]+1;
        cols[0] = 4*idx[0]+1;   vals[0] = eM_0[r][0]+dt*epsilon*epsilon*eM_2_odd[r][0];
        cols[1] = 4*idx[1]+1;   vals[1] = eM_0[r][1]+dt*epsilon*epsilon*eM_2_odd[r][1];
        cols[2] = 4*idx[2]+1;   vals[2] = eM_0[r][2]+dt*epsilon*epsilon*eM_2_odd[r][2];
        /* Insert values in matrix M for 2nd dof */
        ierr = MatSetValuesLocal(M,1,&row,3,cols,vals,ADD_VALUES);CHKERRQ(ierr);

        row = 4*idx[r]+2;
        cols[0] = 4*idx[0]+2;   vals[0] = eM_0[r][0]+dt*epsilon*epsilon*eM_2_odd[r][0];
        cols[1] = 4*idx[1]+2;   vals[1] = eM_0[r][1]+dt*epsilon*epsilon*eM_2_odd[r][1];
        cols[2] = 4*idx[2]+2;   vals[2] = eM_0[r][2]+dt*epsilon*epsilon*eM_2_odd[r][2];
        /* Insert values in matrix M for 3nd dof */
        ierr = MatSetValuesLocal(M,1,&row,3,cols,vals,ADD_VALUES);CHKERRQ(ierr);
      }
      else{
        row = 4*idx[r];
        cols[0] = 4*idx[0];     vals[0] = eM_0[r][0]+dt*epsilon*epsilon*eM_2_even[r][0];
        cols[1] = 4*idx[1];     vals[1] = eM_0[r][1]+dt*epsilon*epsilon*eM_2_even[r][1];
        cols[2] = 4*idx[2];     vals[2] = eM_0[r][2]+dt*epsilon*epsilon*eM_2_even[r][2];
        /* Insert values in matrix M for 1st dof */
        ierr = MatSetValuesLocal(M,1,&row,3,cols,vals,ADD_VALUES);CHKERRQ(ierr);

        row = 4*idx[r]+1;
        cols[0] = 4*idx[0]+1;   vals[0] = eM_0[r][0]+dt*epsilon*epsilon*eM_2_even[r][0];
        cols[1] = 4*idx[1]+1;   vals[1] = eM_0[r][1]+dt*epsilon*epsilon*eM_2_even[r][1];
        cols[2] = 4*idx[2]+1;   vals[2] = eM_0[r][2]+dt*epsilon*epsilon*eM_2_even[r][2];
        /* Insert values in matrix M for 2nd dof */
        ierr = MatSetValuesLocal(M,1,&row,3,cols,vals,ADD_VALUES);CHKERRQ(ierr);

        row = 4*idx[r]+2;
        cols[0] = 4*idx[0]+2;   vals[0] = eM_0[r][0]+dt*epsilon*epsilon*eM_2_even[r][0];
        cols[1] = 4*idx[1]+2;   vals[1] = eM_0[r][1]+dt*epsilon*epsilon*eM_2_even[r][1];
        cols[2] = 4*idx[2]+2;   vals[2] = eM_0[r][2]+dt*epsilon*epsilon*eM_2_even[r][2];
        /* Insert values in matrix M for 3nd dof */
        ierr = MatSetValuesLocal(M,1,&row,3,cols,vals,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }

  ierr = MatAssemblyBegin(user->M_0,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->M_0,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscScalar vals[9];

  vals[0] = -1.0; vals[1] =  0.0; vals[2] =  0.0;
  vals[3] =  0.0; vals[4] = -1.0; vals[5] =  0.0;
  vals[6] =  0.0; vals[7] =  0.0; vals[8] = -1.0;


  for (j=0;j < nele;j++) {
    idx[0] = ele[3*j]; idx[1] = ele[3*j+1]; idx[2] = ele[3*j+2];

    PetscInt   r,rows[3],cols[3];
    for (r=0;r<3;r++) {

      rows[0] = 4*idx[0]+r;     cols[0] = 4*idx[0]+3;
      rows[1] = 4*idx[1]+r;   cols[1] = 4*idx[1]+3;
      rows[2] = 4*idx[2]+r;   cols[2] = 4*idx[2]+3;
      ierr = MatSetValuesLocal(M,3,rows,3,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(M,3,cols,3,rows,vals,INSERT_VALUES);CHKERRQ(ierr);

    }

  }

  ierr = DMDARestoreElements(user->da,&nele,&nen,&ele);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coords,&_coords);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);



  ierr = VecCreate(PETSC_COMM_WORLD,&user->u1);CHKERRQ(ierr);
  ierr = VecSetSizes(user->u1,n/4,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->u1);CHKERRQ(ierr);
  ierr = VecDuplicate(user->u1,&user->u2);CHKERRQ(ierr);
  ierr = VecDuplicate(user->u1,&user->u3);CHKERRQ(ierr);
  ierr = VecDuplicate(user->u1,&user->work1);CHKERRQ(ierr);
  ierr = VecDuplicate(user->u1,&user->work2);CHKERRQ(ierr);
  ierr = VecDuplicate(user->u1,&user->work3);CHKERRQ(ierr);
  ierr = VecDuplicate(user->u1,&user->work4);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
