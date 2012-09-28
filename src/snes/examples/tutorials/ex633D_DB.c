static char help[] = "3D coupled Allen-Cahn and Cahn-Hilliard equation for degenerate mobility and triangular elements.\n\
Runtime options include:\n\
-xmin <xmin>\n\
-xmax <xmax>\n\
-ymin <ymin>\n\
-T <T>, where <T> is the end time for the time domain simulation\n\
-dt <dt>,where <dt> is the step size for the numerical integration\n\
-gamma <gamma>\n\
-theta_c <theta_c>\n\n";

/*
./ex633D_DB -ksp_type fgmres -snes_vi_monitor -snes_atol 1.e-11 -snes_converged_reason -ksp_converged_reason -snes_linesearch_monitor -VG 10000000  -pc_type mg -pc_mg_galerkin -log_summary -dt .000001 -mg_coarse_pc_type svd  -ksp_monitor_true_residual -ksp_rtol 1.e-9 -snes_linesearch_type basic -T .0020 -P_casc .0005 -snes_monitor_solution -da_refine 1
./ex633D_DB -ksp_type fgmres -snes_vi_monitor -snes_atol 1.e-11 -snes_converged_reason -ksp_converged_reason -snes_linesearch_monitor -VG 10000000  -pc_type mg -pc_mg_galerkin -log_summary  -da_refine 1

 */

#include "petscsnes.h"
#include "petscdmda.h"

typedef struct{
  PetscReal   dt,T; /* Time step and end time */
  DM          da1,da1_clone,da2;
  Mat         M;    /* Jacobian matrix */
  Mat         M_0;
  Vec         q,wv,cv,wi,ci,eta,cvi,DPsiv,DPsii,DPsieta,Pv,Pi,Piv,logcv,logci,logcvi,Riv;
  Vec         work1,work2,work3,work4;
  PetscScalar Dv,Di,Evf,Eif,A,kBT,kav,kai,kaeta,Rsurf,Rbulk,L,P_casc,VG; /* physics parameters */
  PetscReal   xmin,xmax,ymin,ymax,zmin,zmax;
  PetscInt    nx;
  PetscBool   voidgrowth;
  PetscBool   degenerate;
  PetscBool   periodic;
  PetscReal   smallnumber;
  PetscReal   initv;
  PetscReal   initeta;
}AppCtx;

PetscErrorCode GetParams(AppCtx*);
PetscErrorCode SetRandomVectors(AppCtx*);
PetscErrorCode SetVariableBounds(DM,Vec,Vec);
PetscErrorCode SetUpMatrices(AppCtx*);
PetscErrorCode UpdateMatrices(AppCtx*);
PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
PetscErrorCode FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
PetscErrorCode SetInitialGuess(Vec,AppCtx*);
PetscErrorCode Update_q(AppCtx*);
PetscErrorCode Update_u(Vec,AppCtx*);
PetscErrorCode DPsi(AppCtx*);
PetscErrorCode Llog(Vec,Vec);
PetscErrorCode CheckRedundancy(SNES,IS,IS*,DM);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  Vec            x,r;  /* olution and residual vectors */
  SNES           snes; /* Nonlinear solver context */
  AppCtx         user; /* Application context */
  Vec            xl,xu; /* Upper and lower bounds on variables */
  Mat            J;
  PetscScalar    t=0.0;
  PetscViewer    view_out, view_p, view_q, view_psi, view_mat;
  PetscReal      bounds[] = {1000.0,-1000.,0.0,1.0,1000.0,-1000.0,0.0,1.0,1000.0,-1000.0};


  PetscInitialize(&argc,&argv, (char*)0, help);


  // Get physics and time parameters
  ierr = GetParams(&user);CHKERRQ(ierr);
	
  if (user.periodic) {
	  ierr = DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX, -3,-3,-3,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, 5, 1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&user.da1);CHKERRQ(ierr);
	  ierr = DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX, -3,-3,-3,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, 5, 1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&user.da1_clone);CHKERRQ(ierr);
	  ierr = DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX, -3,-3,-3,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, 1, 1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&user.da2);CHKERRQ(ierr);	
	
  } else {
	  // Create a 1D DA with dof = 5; the whole thing
	  ierr = DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX, -3,-3,-3,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, 5, 1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&user.da1);CHKERRQ(ierr);
	  ierr = DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX, -3,-3,-3,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, 5, 1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&user.da1_clone);CHKERRQ(ierr);
	  // Create a 1D DA with dof = 1; for individual componentes
	  ierr = DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX, -3,-3,-3,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, 1, 1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&user.da2);CHKERRQ(ierr);
  }
	
  // Set Element type (rectangular)
  ierr = DMDASetElementType(user.da1,DMDA_ELEMENT_Q1);CHKERRQ(ierr);
  ierr = DMDASetElementType(user.da2,DMDA_ELEMENT_Q1);CHKERRQ(ierr);
	
  // Set x and y coordinates
  ierr = DMDASetUniformCoordinates(user.da1,user.xmin,user.xmax,user.ymin,user.ymax, user.zmin,user.zmax);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(user.da2,user.xmin,user.xmax,user.ymin,user.ymax, user.zmin,user.zmax);CHKERRQ(ierr);
	
  // Get global vector x from DM (da1) and duplicate vectors r,xl,xu
  ierr = DMCreateGlobalVector(user.da1,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xl);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xu);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&user.q);CHKERRQ(ierr);


  // Get global vector user->wv from da2 and duplicate other vectors
  ierr = DMCreateGlobalVector(user.da2,&user.wv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.cv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.wi);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.ci);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.eta);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.cvi);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.DPsiv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.DPsii);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.DPsieta);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.Pv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.Pi);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.Piv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.Riv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.logcv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.logci);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.logcvi);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.work1);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.work2);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.work3);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.work4);CHKERRQ(ierr);
	
  // Get Jacobian matrix structure from the da for the entire thing, da1
  ierr = DMCreateMatrix(user.da1,MATAIJ,&user.M);CHKERRQ(ierr);
  // Get the (usual) mass matrix structure from da2
  ierr = DMCreateMatrix(user.da2,MATAIJ,&user.M_0);CHKERRQ(ierr);
	
  ierr = SetInitialGuess(x,&user);CHKERRQ(ierr);
	
  // Form the jacobian matrix and M_0
  ierr = SetUpMatrices(&user);CHKERRQ(ierr);
	
 	
  ierr = MatDuplicate(user.M,MAT_DO_NOT_COPY_VALUES,&J);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,user.da1);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,r,FormFunction,(void*)&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,(void*)&user);CHKERRQ(ierr);

  ierr = SetVariableBounds(user.da1,xl,xu);CHKERRQ(ierr);
  //ierr = SNESVISetRedundancyCheck(snes,(PetscErrorCode (*)(SNES,IS,IS*,void*))CheckRedundancy,user.da1_clone);CHKERRQ(ierr);
  ierr = SNESVISetVariableBounds(snes,xl,xu);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  	
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_out",FILE_MODE_WRITE,&view_out);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_p",FILE_MODE_WRITE,&view_p);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_q",FILE_MODE_WRITE,&view_q);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_psi",FILE_MODE_WRITE,&view_psi);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_mat",FILE_MODE_WRITE,&view_mat);CHKERRQ(ierr);

	
  /* 	
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_q",FILE_MODE_WRITE,&view_q);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_mat",FILE_MODE_WRITE,&view_mat);CHKERRQ(ierr);	
  	
  ierr = VecView(user.q,view_q);CHKERRQ(ierr);
  ierr = MatView(user.M,view_mat);CHKERRQ(ierr);	
	
  ierr = PetscViewerDestroy(&view_q);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_mat);CHKERRQ(ierr);
 */
	
 	
	
  while(t<user.T) {
    char         filename[PETSC_MAX_PATH_LEN];
    PetscScalar  a = 1.0;
    PetscInt     i;
    PetscViewer  view;


    ierr = SNESSetFunction(snes,r,FormFunction,(void*)&user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,FormJacobian,(void*)&user);CHKERRQ(ierr);

    ierr = SetRandomVectors(&user);CHKERRQ(ierr);
    ierr = DPsi(&user);CHKERRQ(ierr);
    ierr = VecView(user.DPsiv,view_psi);CHKERRQ(ierr);
    ierr = VecView(user.DPsii,view_psi);CHKERRQ(ierr);
    ierr = VecView(user.DPsieta,view_psi);CHKERRQ(ierr);

    ierr = VecView(user.Pv,view_p);CHKERRQ(ierr);
    ierr = Update_q(&user);CHKERRQ(ierr);
    ierr = VecView(user.q,view_q);CHKERRQ(ierr);
	
	  printf("after VecView\n");
    ierr = MatView(user.M,view_mat);CHKERRQ(ierr);
	
	  printf("after MatView\n");

    ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);

    ierr = VecView(x,view_out);CHKERRQ(ierr);

    PetscInt its;
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"SNESVI solver converged at t = %5.4g in %d iterations\n",t,its);CHKERRQ(ierr);

    ierr = Update_u(x,&user);CHKERRQ(ierr);
	 /*
    for (i=0; i < (int)(user.T/a) ; i++) {
      if (t/a > i - user.dt/a && t/a < i + user.dt/a) {
        sprintf(filename,"output_%f",t);
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
        ierr = VecView(user.cv,view);CHKERRQ(ierr);
        ierr = VecView(user.ci,view);CHKERRQ(ierr);
        ierr = VecView(user.eta,view);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
      }

    }
	 */
    ierr = UpdateMatrices(&user);CHKERRQ(ierr);
    t = t + user.dt;

  }



  ierr = PetscViewerDestroy(&view_out);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_p);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_q);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_psi);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_mat);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&xl);CHKERRQ(ierr);
  ierr = VecDestroy(&xu);CHKERRQ(ierr);
  ierr = VecDestroy(&user.q);CHKERRQ(ierr);
  ierr = VecDestroy(&user.wv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.cv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.wi);CHKERRQ(ierr);
  ierr = VecDestroy(&user.ci);CHKERRQ(ierr);
  ierr = VecDestroy(&user.eta);CHKERRQ(ierr);
  ierr = VecDestroy(&user.cvi);CHKERRQ(ierr);
  ierr = VecDestroy(&user.DPsiv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.DPsii);CHKERRQ(ierr);
  ierr = VecDestroy(&user.DPsieta);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Pv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Pi);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Piv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Riv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.logcv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.logci);CHKERRQ(ierr);
  ierr = VecDestroy(&user.logcvi);CHKERRQ(ierr);
  ierr = VecDestroy(&user.work1);CHKERRQ(ierr);
  ierr = VecDestroy(&user.work2);CHKERRQ(ierr);
  ierr = VecDestroy(&user.work3);CHKERRQ(ierr);
  ierr = VecDestroy(&user.work4);CHKERRQ(ierr);
  ierr = MatDestroy(&user.M);CHKERRQ(ierr);
  ierr = MatDestroy(&user.M_0);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da1);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da1_clone);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da2);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
	

	
  printf("I am finalize \n");	
  PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "Update_u"
PetscErrorCode Update_u(Vec X,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscInt       i,n;
  PetscScalar    *xx,*wv_p,*cv_p,*wi_p,*ci_p,*eta_p;


  PetscFunctionBegin;
  ierr = VecGetLocalSize(user->wv,&n);CHKERRQ(ierr);

  ierr = VecGetArray(X,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(user->wv,&wv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->wi,&wi_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->ci,&ci_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->eta,&eta_p);CHKERRQ(ierr);


  for (i=0;i<n;i++) {
    wv_p[i] = xx[5*i];
    cv_p[i] = xx[5*i+1];
    wi_p[i] = xx[5*i+2];
    ci_p[i] = xx[5*i+3];
    eta_p[i] = xx[5*i+4];
  }
  ierr = VecRestoreArray(X,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->wv,&wv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->wi,&wi_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->ci,&ci_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->eta,&eta_p);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Update_q"
PetscErrorCode Update_q(AppCtx *user)
{
  PetscErrorCode ierr;
  PetscScalar    *q_p, *w1, *w2;
  PetscInt       i,n;

  PetscFunctionBegin;

  ierr = VecPointwiseMult(user->Riv,user->eta,user->eta);CHKERRQ(ierr); //Riv = eta.^2
  ierr = VecScale(user->Riv,user->Rsurf);CHKERRQ(ierr); // Riv = Rsurf * eta.^2
  ierr = VecShift(user->Riv,user->Rbulk);CHKERRQ(ierr); // Riv = Rbulk + Rsurf * eta.^2
  ierr = VecPointwiseMult(user->Riv,user->ci,user->Riv);CHKERRQ(ierr); // Riv = (Rbulk + Rsurf * eta.^2) * ci
  ierr = VecPointwiseMult(user->Riv,user->cv,user->Riv);CHKERRQ(ierr);// Riv = (Rbulk + Rsurf * eta.^2) * ci * cv

  ierr = VecCopy(user->Riv,user->work1);CHKERRQ(ierr);
  ierr = VecScale(user->work1,user->dt);CHKERRQ(ierr); // work1 = dt * Riv
  ierr = VecAXPY(user->work1,-1.0,user->cv);CHKERRQ(ierr);// work1 = dt * Riv - cv
  ierr = MatMult(user->M_0,user->work1,user->work2);CHKERRQ(ierr); // work2 = M_0 * (dt * Riv - cv)

  ierr = VecGetArray(user->q,&q_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->work1,&w1);CHKERRQ(ierr);
  ierr = VecGetArray(user->work2,&w2);CHKERRQ(ierr);
  ierr = VecGetLocalSize(user->wv,&n);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    q_p[5*i]=w2[i];
  }

  ierr = MatMult(user->M_0,user->DPsiv,user->work1);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    q_p[5*i+1]=w1[i];
  }

  ierr = VecCopy(user->Riv,user->work1);CHKERRQ(ierr);
  ierr = VecScale(user->work1,user->dt);CHKERRQ(ierr);
  ierr = VecAXPY(user->work1,-1.0,user->ci);CHKERRQ(ierr);
  ierr = MatMult(user->M_0,user->work1,user->work2);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    q_p[5*i+2]=w2[i];
  }

  ierr = MatMult(user->M_0,user->DPsii,user->work1);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    q_p[5*i+3]=w1[i];
  }

  ierr = VecCopy(user->DPsieta,user->work1);CHKERRQ(ierr);
  ierr = VecScale(user->work1,user->L);CHKERRQ(ierr);
  ierr = VecAXPY(user->work1,-1.0/user->dt,user->eta);CHKERRQ(ierr);
  ierr = MatMult(user->M_0,user->work1,user->work2);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    q_p[5*i+4]=w2[i];
  }

  ierr = VecRestoreArray(user->q,&q_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->work1,&w1);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->work2,&w2);CHKERRQ(ierr);



  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DPsi"
PetscErrorCode DPsi(AppCtx* user)
{
  PetscErrorCode  ierr;
  PetscScalar     Evf=user->Evf,Eif=user->Eif,kBT=user->kBT,A=user->A;
  PetscScalar     *cv_p,*ci_p,*eta_p,*logcv_p,*logci_p,*logcvi_p,*DPsiv_p,*DPsii_p,*DPsieta_p;
  PetscInt        n,i;

  PetscFunctionBegin;

  ierr = VecGetLocalSize(user->cv,&n);
  ierr = VecGetArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->ci,&ci_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->eta,&eta_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->logcv,&logcv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->logci,&logci_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->logcvi,&logcvi_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->DPsiv,&DPsiv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->DPsii,&DPsii_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->DPsieta,&DPsieta_p);CHKERRQ(ierr);

  ierr = Llog(user->cv,user->logcv);CHKERRQ(ierr);
  ierr = Llog(user->ci,user->logci);CHKERRQ(ierr);

  ierr = VecCopy(user->cv,user->cvi);CHKERRQ(ierr);
  ierr = VecAXPY(user->cvi,1.0,user->ci);CHKERRQ(ierr);
  ierr = VecScale(user->cvi,-1.0);CHKERRQ(ierr);
  ierr = VecShift(user->cvi,1.0);CHKERRQ(ierr);
  ierr = Llog(user->cvi,user->logcvi);CHKERRQ(ierr);

  for (i=0;i<n;i++)
  {
    DPsiv_p[i] = (eta_p[i]-1.0)*(eta_p[i]-1.0)*( Evf + kBT*(logcv_p[i] - logcvi_p[i])) + eta_p[i]*eta_p[i]*2*A*(cv_p[i]-1);

    DPsii_p[i] = (eta_p[i]-1.0)*(eta_p[i]-1.0)*( Eif + kBT*(logci_p[i] - logcvi_p[i])) + eta_p[i]*eta_p[i]*2*A*ci_p[i] ;

    DPsieta_p[i] = 2.0*(eta_p[i]-1.0)*( Evf*cv_p[i] + Eif*ci_p[i] + kBT*( cv_p[i]* logcv_p[i] + ci_p[i]* logci_p[i] + (1-cv_p[i]-ci_p[i])*logcvi_p[i] ) ) + 2.0*eta_p[i]*A*( (cv_p[i]-1.0)*(cv_p[i]-1.0) + ci_p[i]*ci_p[i]);


  }

  ierr = VecRestoreArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->ci,&ci_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->eta,&eta_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->logcv,&logcv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->logci,&logci_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->logcvi,&logcvi_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->DPsiv,&DPsiv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->DPsii,&DPsii_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->DPsieta,&DPsieta_p);CHKERRQ(ierr);


  PetscFunctionReturn(0);

}


#undef __FUNCT__
#define __FUNCT__ "Llog"
PetscErrorCode Llog(Vec X, Vec Y)
{
  PetscErrorCode    ierr;
  PetscScalar       *x,*y;
  PetscInt          n,i;

  PetscFunctionBegin;

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = VecGetLocalSize(X,&n);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    if (x[i] < 1.0e-12) {
      y[i] = log(1.0e-12);
    }
    else {
      y[i] = log(x[i]);
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetInitialGuess"
PetscErrorCode SetInitialGuess(Vec X,AppCtx* user)
{	
	/////////////////////////// 3D //////////////////////////////
	PetscErrorCode    ierr;
	PetscInt          n,i,j,Xda,Yda,Zda;
	PetscScalar	      *xx,*cv_p,*ci_p,*wv_p,*wi_p,*eta_p;
	PetscViewer       view_out;
	// needed for the void growth case
	PetscScalar       cv_v=1.0,cv_m=0.122,ci_v=0.0,ci_m=.00069,eta_v=1.0,eta_m=0.0,h,lambda;
	PetscScalar       xmid,ymid,zmid;
	PetscInt          nele,nen,idx[8];
	const PetscInt    *ele;
	PetscScalar       x[8],y[8],z[8];
	Vec               coords;
	const PetscScalar *_coords;
	PetscViewer       view;
	PetscScalar       xwidth = user->xmax - user->xmin, ywidth = user->ymax - user->ymin, zwidth = user->zmax - user->zmin;
	
	PetscFunctionBegin;
	
	ierr = VecGetLocalSize(X,&n);CHKERRQ(ierr);
	
	if (user->voidgrowth) {
		ierr = DMDAGetInfo(user->da2,PETSC_NULL,&Xda,&Yda,&Zda,PETSC_NULL,PETSC_NULL,
						   PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
		ierr = DMGetCoordinatesLocal(user->da2,&coords);CHKERRQ(ierr);
		ierr = VecGetArrayRead(coords,&_coords);CHKERRQ(ierr);
		
		if (user->periodic) {
			h = (user->xmax-user->xmin)/Xda;
		} else {
			h = (user->xmax-user->xmin)/(Xda-1.0);
		}		
		
		xmid = (user->xmax + user->xmin)/2.0;
		ymid = (user->ymax + user->ymin)/2.0;
		zmid = (user->zmax + user->zmin)/2.0;
		lambda = 4.0*h;
		
		ierr = DMDAGetElements(user->da2,&nele,&nen,&ele);CHKERRQ(ierr); // number of local elements, number of element nodes, the local indices of the elements' vertices
		for (i=0;i < nele; i++)
		{
			for (j = 0; j<8; j++)
			{
				idx[j] = ele[8*i + j];
				x[j] = _coords[3*idx[j]];
				y[j] = _coords[3*idx[j]+1];
				z[j] = _coords[3*idx[j]+2];
			}
			
			PetscInt k;
			PetscScalar vals_cv[8],vals_ci[8],vals_eta[8],s,hhr,r;
			for (k=0; k < 8 ; k++)
			{
				s = sqrt((x[k] - xmid)*(x[k] - xmid) + (y[k] - ymid)*(y[k] - ymid) + (z[k] - zmid)*(z[k] - zmid));
				if (s < xwidth*(5.0/64.0))
				{
					vals_cv[k] = cv_v;
					vals_ci[k] = ci_v;
					vals_eta[k] = eta_v;
				}
				else if (s>= xwidth*(5.0/64.0) && s<= xwidth*(7.0/64.0) )
				{
					//r = (s - xwidth*(6.0/64.0) )/(0.5*lambda);
					r = (s - xwidth*(6.0/64.0) )/(xwidth/64.0);
					hhr = 0.25*(-r*r*r + 3*r + 2);
					vals_cv[k] = cv_m + (1.0 - hhr)*(cv_v - cv_m);
					vals_ci[k] = ci_m + (1.0 - hhr)*(ci_v - ci_m);
					vals_eta[k] = eta_m + (1.0 - hhr)*(eta_v - eta_m);
				} else
				{
					vals_cv[k] = cv_m;
					vals_ci[k] = ci_m;
					vals_eta[k] = eta_m;
				}
			}
			
			ierr = VecSetValuesLocal(user->cv,8,idx,vals_cv,INSERT_VALUES);CHKERRQ(ierr);
			ierr = VecSetValuesLocal(user->ci,8,idx,vals_ci,INSERT_VALUES);CHKERRQ(ierr);
			ierr = VecSetValuesLocal(user->eta,8,idx,vals_eta,INSERT_VALUES);CHKERRQ(ierr);
		}
		ierr = VecAssemblyBegin(user->cv);CHKERRQ(ierr);
		ierr = VecAssemblyEnd(user->cv);CHKERRQ(ierr);
		ierr = VecAssemblyBegin(user->ci);CHKERRQ(ierr);
		ierr = VecAssemblyEnd(user->ci);CHKERRQ(ierr);
		ierr = VecAssemblyBegin(user->eta);CHKERRQ(ierr);
		ierr = VecAssemblyEnd(user->eta);CHKERRQ(ierr);
		
		ierr = DMDARestoreElements(user->da2,&nele,&nen,&ele);CHKERRQ(ierr);
		ierr = VecRestoreArrayRead(coords,&_coords);CHKERRQ(ierr);

	}else {
	
	ierr = VecSet(user->cv,6.9e-4);CHKERRQ(ierr);
	ierr = VecSet(user->ci,6.9e-4);CHKERRQ(ierr);
    ierr = VecSet(user->eta,0.0);CHKERRQ(ierr);
  }

  ierr = DPsi(user);CHKERRQ(ierr);
  ierr = VecCopy(user->DPsiv,user->wv);CHKERRQ(ierr);
  ierr = VecCopy(user->DPsii,user->wi);CHKERRQ(ierr);

  ierr = VecGetArray(X,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->ci,&ci_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->eta,&eta_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->wv,&wv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->wi,&wi_p);CHKERRQ(ierr);
  for (i=0;i<n/5;i++)
  {
    xx[5*i]=wv_p[i];
    xx[5*i+1]=cv_p[i];
    xx[5*i+2]=wi_p[i];
    xx[5*i+3]=ci_p[i];
    xx[5*i+4]=eta_p[i];
  }

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_initial",FILE_MODE_WRITE,&view_out);CHKERRQ(ierr);
  ierr = VecView(user->wv,view_out);CHKERRQ(ierr);
  ierr = VecView(user->cv,view_out);CHKERRQ(ierr);
  ierr = VecView(user->wi,view_out);CHKERRQ(ierr);
  ierr = VecView(user->ci,view_out);CHKERRQ(ierr);
  ierr = VecView(user->eta,view_out);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_out);CHKERRQ(ierr);

  ierr = VecRestoreArray(X,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->ci,&ci_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->wv,&wv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->wi,&wi_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->eta,&eta_p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetRandomVectors"
PetscErrorCode SetRandomVectors(AppCtx* user)
{
  PetscErrorCode ierr;
  PetscInt       i,n,count=0;
  PetscScalar    *w1,*w2,*Pv_p,*eta_p;

  /* static PetscViewer viewer=0; */
  static PetscRandom rand = 0;
  static PetscInt    step = 0;

  PetscFunctionBegin;
  if (!rand) {
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  }

  ierr = VecSetRandom(user->work1,rand);CHKERRQ(ierr);
  ierr = VecSetRandom(user->work2,rand);CHKERRQ(ierr);
  ierr = VecGetArray(user->work1,&w1);CHKERRQ(ierr);
  ierr = VecGetArray(user->work2,&w2);CHKERRQ(ierr);
  ierr = VecGetArray(user->Pv,&Pv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->eta,&eta_p);CHKERRQ(ierr);
  ierr = VecGetLocalSize(user->work1,&n);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    if (eta_p[i]>=0.8 || w1[i]>user->P_casc){
      Pv_p[i]=0;
    }
    else
    {
      Pv_p[i]=w2[i]*user->VG;
      count = count + 1;
    }
  }
  step++;

  ierr = VecCopy(user->Pv,user->Pi);CHKERRQ(ierr);
  ierr = VecScale(user->Pi,0.9);CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->Piv,user->Pi,user->Pv);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->work1,&w1);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->work2,&w2);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Pv,&Pv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->eta,&eta_p);CHKERRQ(ierr);
  printf("Number of radiations count %d out of total mesh points n %d\n",count,n);

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
	PetscScalar    ****l,****u;
	PetscInt       xs,xm,ys,ym,zs,zm;
	PetscInt       k,j,i;
	
	PetscFunctionBegin;
	ierr = DMDAVecGetArrayDOF(da,xl,&l);CHKERRQ(ierr);
	ierr = DMDAVecGetArrayDOF(da,xu,&u);CHKERRQ(ierr);
	
	ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
	
	for (k = zs; k < zs + zm; k++)
	{
		for (j=ys; j<ys+ym; j++)
		{
			for (i=xs; i < xs+xm;i++)
			{
				l[k][j][i][0] = -SNES_VI_INF;
				l[k][j][i][1] = 0.0;
				l[k][j][i][2] = -SNES_VI_INF;
				l[k][j][i][3] = 0.0;
				l[k][j][i][4] = 0.0;
				u[k][j][i][0] = SNES_VI_INF;
				u[k][j][i][1] = 1.0;
				u[k][j][i][2] = SNES_VI_INF;
				u[k][j][i][3] = 1.0;
				u[k][j][i][4] = 1.0;
			}
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
  user->xmin = 0.0; user->xmax = 64.0;
  user->ymin = 0.0; user->ymax = 64.0;
  user->zmin = 0.0; user->zmax = 64.0;	
  user->Dv = 1.0; user->Di=1.0;
  user->Evf = 0.8; user->Eif = 0.8;
  user->A = 1.0;
  user->kBT = 0.11;
  user->kav = 1.0; user->kai = 1.0; user->kaeta = 1.0;
  user->Rsurf = 10.0; user->Rbulk = 0.0;
  user->L = 10.0; user->P_casc = 0.05;
  user->T = 1.0e-2;    user->dt = 1.0e-4;
  user->VG = 100.0;
  user->initv = .0001;
  user->initeta = 0.0;

  user->degenerate = PETSC_FALSE;
  /* void growth */
  user->voidgrowth = PETSC_FALSE;

  ierr = PetscOptionsGetReal(PETSC_NULL,"-xmin",&user->xmin,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-xmax",&user->xmax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-T",&user->T,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-dt",&user->dt,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-degenerate","Run with degenerate mobility\n","None",user->degenerate,&user->degenerate,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-smallnumber","Small number added to degenerate mobility\n","None",user->smallnumber,&user->smallnumber,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-voidgrowth","Use initial conditions for void growth\n","None",user->voidgrowth,&user->voidgrowth,&flg);CHKERRQ(ierr);

  PetscFunctionReturn(0);
 }


#undef __FUNCT__
#define __FUNCT__ "SetUpMatrices"
PetscErrorCode SetUpMatrices(AppCtx* user)
{
  PetscErrorCode    ierr;
  PetscInt          nele,nen,i,j,n;
  const PetscInt    *ele;
  PetscScalar       dt=user->dt,hx,hy,hz;

  PetscInt          idx[8];
  PetscScalar       eM_0[8][8],eM_2[8][8];
  PetscScalar       cv_sum, ci_sum;
  PetscScalar 	    tp_cv, tp_ci;
	
  Mat               M=user->M;
  Mat               M_0=user->M_0;
  PetscInt          Xda,Yda,Zda;
  PetscScalar       *cv_p,*ci_p;
  Vec               cvlocal,cilocal;

  PetscFunctionBegin;

  ierr = DMGetLocalVector(user->da2,&cvlocal);CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->da2,&cilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(user->da2,user->cv,INSERT_VALUES,cvlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->da2,user->cv,INSERT_VALUES,cvlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(user->da2,user->ci,INSERT_VALUES,cilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->da2,user->ci,INSERT_VALUES,cilocal);CHKERRQ(ierr);

  ierr = VecGetArray(cvlocal,&cv_p);CHKERRQ(ierr);
  ierr = VecGetArray(cilocal,&ci_p);CHKERRQ(ierr);

  ierr = MatGetLocalSize(M,&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(user->da1,PETSC_NULL,&Xda,&Yda,&Zda,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

	if (Xda!=Yda) {
		printf("Currently different Xda and Yda are not supported");
	}
	if (user->periodic) {
		hx = (user->xmax-user->xmin)/Xda;
		hy = (user->ymax-user->ymin)/Yda;
		hz = (user->zmax-user->zmin)/Zda;
	} else {
		hx = (user->xmax-user->xmin)/(Xda-1.0);
		hy = (user->ymax-user->ymin)/(Yda-1.0);
		hz = (user->zmax-user->zmin)/(Zda-1.0);
	}
	
	
	
	/** blocks of M_0 **/
	
	
	for (i = 0; i<8; i++)
	{
		eM_0[i][i] = 1;
	}
	eM_0[0][1] = 0.5; eM_0[0][2] = 0.25; eM_0[0][3] = 0.5; eM_0[0][4] = 0.5; eM_0[0][5] = 0.25; eM_0[0][6] = 0.125; eM_0[0][7] = 0.25;
	eM_0[1][2] = 0.5; eM_0[1][3] = 0.25; eM_0[1][4] = 0.25;eM_0[1][5] = 0.5; eM_0[1][6] = 0.25; eM_0[1][7] = 0.125;
	eM_0[2][3] = 0.5; eM_0[2][4] = 0.125;eM_0[2][5] = 0.25;eM_0[2][6] = 0.5;eM_0[2][7] = 0.25;
	eM_0[3][4] = 0.25;eM_0[3][5] = 0.125;eM_0[3][6] = 0.25;eM_0[3][7] = 0.5;
	eM_0[4][5] = 0.5; eM_0[4][6] = 0.25; eM_0[4][7] = 0.5;
	eM_0[5][6] = 0.5; eM_0[5][7] = 0.25;
	eM_0[6][7] = 0.5;
	
	for (i = 0; i<8; i++)
	{
		for (j =0; j<8; j++)
		{
			if (i>j)
			{
				eM_0[i][j] = eM_0[j][i];
			}
		}
	}
	
	for (i = 0; i<8; i++)
	{
		for (j =0; j<8; j++)
		{
			eM_0[i][j] = hx*hy*hz/27.0 * eM_0[i][j];
		}
	}
			
	/** blocks of M_2 **/
	
	for (i = 0; i<8; i++)
	{
		eM_2[i][i] = 12;
	}
	
	eM_2[0][1] = 0; eM_2[0][2] = -3; eM_2[0][3] = 0;  eM_2[0][4] = 0;  eM_2[0][5] = -3; eM_2[0][6] = -3; eM_2[0][7] = -3;
	eM_2[1][2] = 0; eM_2[1][3] = -3; eM_2[1][4] = -3; eM_2[1][5] = 0;  eM_2[1][6] = -3; eM_2[1][7] = -3;
	eM_2[2][3] = 0; eM_2[2][4] = -3; eM_2[2][5] = -3; eM_2[2][6] = 0;  eM_2[2][7] = -3;
	eM_2[3][4] = -3;eM_2[3][5] = -3; eM_2[3][6] = -3; eM_2[3][7] = 0;
	eM_2[4][5] = 0; eM_2[4][6] = -3; eM_2[4][7] = 0;
	eM_2[5][6] = 0; eM_2[5][7] = -3;
	eM_2[6][7] = 0;
	
	for (i = 0; i<8; i++)
	{
		for (j =0; j<8; j++)
		{
			if (i>j)
			{
				eM_2[i][j] = eM_2[j][i];
			}
		}
	}
	
	
	for (i = 0; i<8; i++)
	{
		for (j =0; j<8; j++)
		{
			eM_2[i][j] = hx*hy*hz/36.0 * eM_2[i][j];
		}
	}
	
	
	/* Get local element info */
  ierr = DMDAGetElements(user->da1,&nele,&nen,&ele);CHKERRQ(ierr);
  PetscInt    row,cols[16],r,row_M_0,cols3[8];
  PetscScalar vals[16],vals_M_0[8],vals3[8];
	
	
  for (i=0;i < nele;i++)
  {
	  for (j=0; j<8; j++)
	  {
		  idx[j] = ele[8*i + j];
	  }

	  for (r=0;r<8;r++)
	  {
		  row_M_0 = idx[r];
		  for (j=0; j<8; j++)
		  {
		      vals_M_0[j]=eM_0[r][j];
		  }
		  	
		
		  ierr = MatSetValuesLocal(M_0,1,&row_M_0,8,idx,vals_M_0,ADD_VALUES);CHKERRQ(ierr);
		
		
		  tp_cv = 0.0;
		  tp_ci = 0.0;
		  for (j = 0; j < 8; j++)
		  {
			  tp_cv = tp_cv + cv_p[idx[j]];
			  tp_ci = tp_ci + ci_p[idx[j]];
		  }
		
		  cv_sum = tp_cv * user->Dv/user->kBT;
		  ci_sum = tp_ci * user->Di/user->kBT;
		
		  row = 5*idx[r];
		  for (j = 0; j < 8; j++)
		  {
			  cols[j] = 5 * idx[j];
			  vals[j] = dt*eM_2[r][j]*cv_sum;
			  cols[j + 8] = 5 * idx[j] + 1;
			  vals[j + 8] = eM_0[r][j];
		  }
		
		  // Insert values in matrix M for 1st dof
		
		  ierr = MatSetValuesLocal(M,1,&row,16,cols,vals,ADD_VALUES);CHKERRQ(ierr);
		
		
		  row = 5*idx[r]+1;
		  for (j = 0; j < 8; j++)
		  {
			  cols[j] = 5 * idx[j];
			  vals[j] = -eM_0[r][j];
			  cols[j + 8] = 5 * idx[j] + 1;
			  vals[j + 8] = user->kav*eM_2[r][j];
		  }
		
		  // Insert values in matrix M for 2nd dof
		  ierr = MatSetValuesLocal(M,1,&row,16,cols,vals,ADD_VALUES);CHKERRQ(ierr);
		
		  row = 5*idx[r]+2;
		  for (j = 0; j < 8; j++)
		  {
			  cols[j] = 5 * idx[j] + 2;
			  vals[j] = dt*eM_2[r][j]*ci_sum;
			  cols[j + 8] = 5 * idx[j] + 3;
			  vals[j + 8] = eM_0[r][j];
		  }
		
		  // Insert values in matrix M for 3rd dof
		  ierr = MatSetValuesLocal(M,1,&row,16,cols,vals,ADD_VALUES);CHKERRQ(ierr);
		
		
		  row = 5*idx[r]+3;
		  for (j = 0; j < 8; j++)
		  {
			  cols[j] = 5 * idx[j] + 2;
			  vals[j] = -eM_0[r][j];
			  cols[j + 8] = 5 * idx[j] + 3;
			  vals[j + 8] = user->kai*eM_2[r][j];
		  }
		
		  // Insert values in matrix M for 4th dof
		  ierr = MatSetValuesLocal(M,1,&row,16,cols,vals,ADD_VALUES);CHKERRQ(ierr); 	
		
		  row = 5*idx[r]+4;
		  for (j = 0; j < 8; j++)
		  {
			  cols3[j] = 5 * idx[j] + 4;
			  vals3[j] = eM_0[r][j]/dt + user->L*user->kaeta*eM_2[r][j];
		  }
		
		  // Insert values in matrix M for 5th dof
		  ierr = MatSetValuesLocal(M,1,&row,8,cols3,vals3,ADD_VALUES);CHKERRQ(ierr);
	  }
  }  //


  ierr = VecRestoreArray(cvlocal,&cv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(cilocal,&ci_p);CHKERRQ(ierr);
	
  ierr = DMRestoreLocalVector(user->da2,&cvlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->da2,&cilocal);CHKERRQ(ierr);
	
  ierr = MatAssemblyBegin(M_0,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M_0,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = DMDARestoreElements(user->da1,&nele,&nen,&ele);CHKERRQ(ierr);


  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "UpdateMatrices"
PetscErrorCode UpdateMatrices(AppCtx* user)
{
  PetscErrorCode    ierr;
  PetscInt          nele,nen,i,j,n,Xda,Yda,Zda;
  const PetscInt    *ele;

  PetscInt          idx[8];
  PetscScalar       eM_2[8][8],h;
  Mat               M=user->M;
  PetscScalar       *cv_p,*ci_p,cv_sum,ci_sum;
  /* newly added */
  Vec               cvlocal,cilocal;

  PetscFunctionBegin;

  ierr = DMGetLocalVector(user->da2,&cvlocal);CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->da2,&cilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(user->da2,user->cv,INSERT_VALUES,cvlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->da2,user->cv,INSERT_VALUES,cvlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(user->da2,user->ci,INSERT_VALUES,cilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->da2,user->ci,INSERT_VALUES,cilocal);CHKERRQ(ierr);
	
  ierr = VecGetArray(cvlocal,&cv_p);CHKERRQ(ierr);
  ierr = VecGetArray(cilocal,&ci_p);CHKERRQ(ierr);

  /* Create the mass matrix M_0 */
  ierr = MatGetLocalSize(M,&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(user->da1,PETSC_NULL,&Xda,&Yda,&Zda,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

   if (Xda!=Yda) {
		printf("Currently different Xda and Yda are not supported");
	}
	if (user->periodic) {
		h = (user->xmax-user->xmin)/Xda;
	} else {
		h = (user->xmax-user->xmin)/(Xda-1.0);
	}
	

  /* Get local element info */
  ierr = DMDAGetElements(user->da1,&nele,&nen,&ele);CHKERRQ(ierr);

  for (i=0;i<nele;i++) {
	  for (j=0; j<8; j++)
	  {
		  idx[j] = ele[8*i + j];
	  }

    PetscInt     r,row,cols[8];
    PetscScalar  vals[8];

    for (r=0;r<8;r++) {
      row = 5*idx[r];
	  for (j = 0; j < 8; j++)
	  {
			cols[j] = 5*idx[j];     vals[j] = 0.0;
			
	   }
		
      /* Insert values in matrix M for 1st dof */
      ierr = MatSetValuesLocal(M,1,&row,8,cols,vals,INSERT_VALUES);CHKERRQ(ierr);

      row = 5*idx[r]+2;
	  for (j = 0; j < 8; j++)
	  {
		 cols[j] = 5*idx[j]+2;   vals[j] = 0.0;
	   }
		

      /* Insert values in matrix M for 3nd dof */
      ierr = MatSetValuesLocal(M,1,&row,8,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

	/** blocks of M_2 **/
	
	for (i = 0; i<8; i++)
	{
		eM_2[i][i] = 12;
	}
	
	eM_2[0][1] = 0; eM_2[0][2] = -3; eM_2[0][3] = 0;  eM_2[0][4] = 0;  eM_2[0][5] = -3; eM_2[0][6] = -3; eM_2[0][7] = -3;
	eM_2[1][2] = 0; eM_2[1][3] = -3; eM_2[1][4] = -3; eM_2[1][5] = 0;  eM_2[1][6] = -3; eM_2[1][7] = -3;
	eM_2[2][3] = 0; eM_2[2][4] = -3; eM_2[2][5] = -3; eM_2[2][6] = 0;  eM_2[2][7] = -3;
	eM_2[3][4] = -3;eM_2[3][5] = -3; eM_2[3][6] = -3; eM_2[3][7] = 0;
	eM_2[4][5] = 0; eM_2[4][6] = -3; eM_2[4][7] = 0;
	eM_2[5][6] = 0; eM_2[5][7] = -3;
	eM_2[6][7] = 0;
	
	for (i = 0; i<8; i++)
	{
		for (j =0; j<8; j++)
		{
			if (i>j)
			{
				eM_2[i][j] = eM_2[j][i];
			}
		}
	}
	
	
	for (i = 0; i<8; i++)
	{
		for (j =0; j<8; j++)
		{
			eM_2[i][j] = h*h*h/36.0 * eM_2[i][j];
		}
	}
	
  for (i=0;i<nele;i++) {
	  for (j=0; j<8; j++)
	  {
		  idx[j] = ele[8*i + j];
	  }
	
    PetscInt    row,cols[8],r;
    PetscScalar vals[8];

    for (r=0;r<8;r++) {

      if (user->degenerate) {
        cv_sum = (2.0*user->smallnumber + cv_p[idx[0]] + cv_p[idx[1]])*user->Dv/(2.0*user->kBT);
        ci_sum = (2.0*user->smallnumber + ci_p[idx[0]] + ci_p[idx[1]])*user->Di/(2.0*user->kBT);
      } else {
        cv_sum = user->initv*user->Dv/user->kBT;
        ci_sum = user->initv*user->Di/user->kBT;
      }

      row = 5*idx[r];
	  for (j=0; j<8; j++)
	  {
		 cols[j] = 5*idx[j];     vals[j] = user->dt*eM_2[r][j]*cv_sum;
      }
      /* Insert values in matrix M for 1st dof */
      ierr = MatSetValuesLocal(M,1,&row,8,cols,vals,ADD_VALUES);CHKERRQ(ierr);

      row = 5*idx[r]+2;
	  for (j=0; j<8; j++)
	  {	
		  cols[j] = 5*idx[j]+2;   vals[j] = user->dt*eM_2[r][j]*ci_sum;
	  }

      /* Insert values in matrix M for 3nd dof */
      ierr = MatSetValuesLocal(M,1,&row,8,cols,vals,ADD_VALUES);CHKERRQ(ierr);
    }
  }


  ierr = VecRestoreArray(cvlocal,&cv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(cilocal,&ci_p);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->da2,&cvlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->da2,&cilocal);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = DMDARestoreElements(user->da1,&nele,&nen,&ele);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


