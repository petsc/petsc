static char help[] = "2D coupled Allen-Cahn and Cahn-Hilliard equation for constant mobility. Only c_v and eta are considered.\n\
Runtime options include:\n\
-xmin <xmin>\n\
-xmax <xmax>\n\
-ymin <ymin>\n\
-T <T>, where <T> is the end time for the time domain simulation\n\
-dt <dt>,where <dt> is the step size for the numerical integration\n\
-gamma <gamma>\n\
-theta_c <theta_c>\n\n";

/*
./ex65 -ksp_type fgmres  -snes_atol 1.e-13  -da_refine 6  -VG 10   -pc_type mg -pc_mg_galerkin -log_summary -dt .000001 -mg_coarse_pc_type redundant -mg_coarse_redundant_pc_type svd  -ksp_rtol 1.e-13 -snes_linesearch_type basic -T .0020  -voidgrowth -da_refine 5 -draw_fileds 0,1,2 -dt .0001 -T 1 -da_grid_x 4 -da_grid_y 4 -periodic -snes_rtol 1.e-13 -ksp_atol 1.e-13  -snes_vi_ignore_function_sign -domain 1
 */

#include "petscsnes.h"
#include "petscdmda.h"

typedef struct{
  PetscReal   dt,T; /* Time step and end time */
  PetscReal   dtevent;  /* time scale of radiation events, roughly one event per dtevent */
  PetscInt    maxevents; /* once this number of events is reached no more events are generated */
  PetscReal   initv;
  PetscReal   initeta;
  DM          da1,da1_clone,da2;
  Mat         M;    /* Jacobian matrix */
  Mat         M_0;
  Vec         q,wv,cv,eta,DPsiv,DPsieta,logcv,logcv2,Pv,Pi,Piv;
  Vec         phi1,phi2,Phi2D_V,Sv,Si; /* for twodomain modeling */
  Vec         work1,work2;
  PetscScalar Mv,L,kaeta,kav,Evf,A,B,cv0,Sv_scalar,VG;
  PetscScalar Svr,Sir,cv_eq,ci_eq; /* for twodomain modeling */
  Vec         work3,work4; /* for twodomain modeling*/
  PetscReal   xmin,xmax,ymin,ymax;
  PetscInt    nx;
  PetscBool   graphics;
  PetscBool   periodic;
  PetscBool   lumpedmass;
  PetscBool   radiation; /* Either radiation or void growth */
  PetscInt    domain;
  PetscReal   grain; /* some bogus factor that controls the "strength" of the grain boundaries */
  PetscInt    darefine;
  PetscInt    dagrid;
}AppCtx;

PetscErrorCode GetParams(AppCtx*);
PetscErrorCode SetRandomVectors(AppCtx*,PetscReal);
PetscErrorCode SetVariableBounds(DM,Vec,Vec);
PetscErrorCode SetUpMatrices(AppCtx*);
PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
PetscErrorCode FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
PetscErrorCode SetInitialGuess(Vec,AppCtx*);
PetscErrorCode Update_q(AppCtx*);
PetscErrorCode Update_u(Vec,AppCtx*);
PetscErrorCode DPsi(AppCtx*);
PetscErrorCode Llog(Vec,Vec);
PetscErrorCode CheckRedundancy(SNES,IS,IS*,DM);
PetscErrorCode Phi(AppCtx*);
PetscErrorCode Phi_read(AppCtx*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  Vec            x,r;  /* solution and residual vectors */
  SNES           snes; /* Nonlinear solver context */
  AppCtx         user; /* Application context */
  Vec            xl,xu; /* Upper and lower bounds on variables */
  Mat            J;
  PetscScalar    t=0.0;
  //PetscViewer    view_out, view_p, view_q, view_psi, view_mat;
  PetscReal      bounds[] = {1000.0,-1000.,0.0,1.0,1000.0,-1000.0,0.0,1.0,1000.0,-1000.0};


  PetscInitialize(&argc,&argv, (char*)0, help);

  /* Get physics and time parameters */
  ierr = GetParams(&user);CHKERRQ(ierr);

  if (user.periodic) {
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX,-4,-4,PETSC_DECIDE,PETSC_DECIDE, 3, 1,PETSC_NULL,PETSC_NULL,&user.da1);CHKERRQ(ierr);
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX,-4,-4,PETSC_DECIDE,PETSC_DECIDE, 3, 1,PETSC_NULL,PETSC_NULL,&user.da1_clone);CHKERRQ(ierr);
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX,-4,-4,PETSC_DECIDE,PETSC_DECIDE, 1, 1,PETSC_NULL,PETSC_NULL,&user.da2);CHKERRQ(ierr);
  } else {
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,-4,-4,PETSC_DECIDE,PETSC_DECIDE, 3, 1,PETSC_NULL,PETSC_NULL,&user.da1);CHKERRQ(ierr);
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,-4,-4,PETSC_DECIDE,PETSC_DECIDE, 3, 1,PETSC_NULL,PETSC_NULL,&user.da1_clone);CHKERRQ(ierr);
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,-4,-4,PETSC_DECIDE,PETSC_DECIDE, 1, 1,PETSC_NULL,PETSC_NULL,&user.da2);CHKERRQ(ierr);
  }
  /* Set Element type (triangular) */
  ierr = DMDASetElementType(user.da1,DMDA_ELEMENT_P1);CHKERRQ(ierr);
  ierr = DMDASetElementType(user.da2,DMDA_ELEMENT_P1);CHKERRQ(ierr);

  /* Set x and y coordinates */
  ierr = DMDASetUniformCoordinates(user.da1,user.xmin,user.xmax,user.ymin,user.ymax,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(user.da2,user.xmin,user.xmax,user.ymin,user.ymax,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  /* Get global vector x from DM (da1) and duplicate vectors r,xl,xu */
  ierr = DMCreateGlobalVector(user.da1,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xl);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xu);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&user.q);CHKERRQ(ierr);

  /* Get global vector user->wv from da2 and duplicate other vectors */
  ierr = DMCreateGlobalVector(user.da2,&user.wv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.cv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.eta);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.DPsiv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.DPsieta);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.Pv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.Pi);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.Piv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.logcv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.logcv2);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.work1);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.work2);CHKERRQ(ierr);
  /* for twodomain modeling */
  ierr = VecDuplicate(user.wv,&user.phi1);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.phi2);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.Phi2D_V);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.Sv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.Si);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.work3);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.work4);CHKERRQ(ierr);

  /* Get Jacobian matrix structure from the da for the entire thing, da1 */
  ierr = DMCreateMatrix(user.da1,MATAIJ,&user.M);CHKERRQ(ierr);
  /* Get the (usual) mass matrix structure from da2 */
  ierr = DMCreateMatrix(user.da2,MATAIJ,&user.M_0);CHKERRQ(ierr);
  /* Form the jacobian matrix and M_0 */
  ierr = SetUpMatrices(&user);CHKERRQ(ierr);
  ierr = SetInitialGuess(x,&user);CHKERRQ(ierr);
  ierr = MatDuplicate(user.M,MAT_DO_NOT_COPY_VALUES,&J);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,user.da1);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,r,FormFunction,(void*)&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,(void*)&user);CHKERRQ(ierr);

  ierr = SetVariableBounds(user.da1,xl,xu);CHKERRQ(ierr);
  ierr = SNESVISetVariableBounds(snes,xl,xu);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  //ierr = SNESVISetRedundancyCheck(snes,(PetscErrorCode (*)(SNES,IS,IS*,void*))CheckRedundancy,user.da1_clone);CHKERRQ(ierr);
  /*
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_out",FILE_MODE_WRITE,&view_out);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_p",FILE_MODE_WRITE,&view_p);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_q",FILE_MODE_WRITE,&view_q);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_psi",FILE_MODE_WRITE,&view_psi);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_mat",FILE_MODE_WRITE,&view_mat);CHKERRQ(ierr);
   */
  /* ierr = PetscViewerDrawSetBounds(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),5,bounds);CHKERRQ(ierr); */


  if (user.graphics) {
    //ierr = PetscViewerDrawSetBounds(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),5,bounds);CHKERRQ(ierr);

    ierr = VecView(x,PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD));CHKERRQ(ierr);
  }


  /* multidomain modeling */
  if (user.domain) {
    switch (user.domain) {
    case 1:
      ierr = Phi(&user);CHKERRQ(ierr);
      break;
    case 2:
      ierr = Phi_read(&user);CHKERRQ(ierr);
      break;
    }
  }

  while(t<user.T) {

    char         filename[PETSC_MAX_PATH_LEN];
    PetscScalar  a = 1.0;
    PetscInt     i;
    //PetscViewer  view;


    ierr = SNESSetFunction(snes,r,FormFunction,(void*)&user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,FormJacobian,(void*)&user);CHKERRQ(ierr);

    if (user.radiation) {
      ierr = SetRandomVectors(&user,t);CHKERRQ(ierr);
    }
    ierr = DPsi(&user);CHKERRQ(ierr);
    //ierr = VecView(user.DPsiv,view_psi);CHKERRQ(ierr);
    //ierr = VecView(user.DPsieta,view_psi);CHKERRQ(ierr);

    ierr = Update_q(&user);CHKERRQ(ierr);
    //ierr = VecView(user.q,view_q);CHKERRQ(ierr);
    //ierr = MatView(user.M,view_mat);CHKERRQ(ierr);

    ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
    //ierr = VecView(x,view_out);CHKERRQ(ierr);


    if (user.graphics) {
      ierr = VecView(x,PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD));CHKERRQ(ierr);
    }

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
        ierr = VecView(user.eta,view);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
      }
    }
     */

    t = t + user.dt;

  }

  /*
  ierr = PetscViewerDestroy(&view_out);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_p);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_q);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_psi);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_mat);CHKERRQ(ierr);
   */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&xl);CHKERRQ(ierr);
  ierr = VecDestroy(&xu);CHKERRQ(ierr);
  ierr = VecDestroy(&user.q);CHKERRQ(ierr);
  ierr = VecDestroy(&user.wv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.cv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.eta);CHKERRQ(ierr);
  ierr = VecDestroy(&user.DPsiv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.DPsieta);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Pv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Pi);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Piv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.logcv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.logcv2);CHKERRQ(ierr);
  ierr = VecDestroy(&user.work1);CHKERRQ(ierr);
  ierr = VecDestroy(&user.work2);CHKERRQ(ierr);
  /* for twodomain modeling */
  ierr = VecDestroy(&user.phi1);CHKERRQ(ierr);
  ierr = VecDestroy(&user.phi2);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Phi2D_V);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Sv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Si);CHKERRQ(ierr);
  ierr = VecDestroy(&user.work3);CHKERRQ(ierr);
  ierr = VecDestroy(&user.work4);CHKERRQ(ierr);

  ierr = MatDestroy(&user.M);CHKERRQ(ierr);
  ierr = MatDestroy(&user.M_0);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da1);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da1_clone);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da2);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "Update_u"
PetscErrorCode Update_u(Vec X,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscInt       i,n;
  PetscScalar    *xx,*wv_p,*cv_p,*eta_p;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(user->wv,&n);CHKERRQ(ierr);

  ierr = VecGetArray(X,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(user->wv,&wv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->eta,&eta_p);CHKERRQ(ierr);


  for (i=0;i<n;i++) {
    wv_p[i] = xx[3*i];
    cv_p[i] = xx[3*i+1];
    eta_p[i] = xx[3*i+2];
  }
  ierr = VecRestoreArray(X,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->wv,&wv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->eta,&eta_p);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Update_q"
PetscErrorCode Update_q(AppCtx *user)
{
  PetscErrorCode ierr;
  PetscScalar    *q_p, *w1, *w2;
  PetscInt       n,i;

  PetscFunctionBegin;

  ierr = VecGetArray(user->q,&q_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->work1,&w1);CHKERRQ(ierr);
  ierr = VecGetArray(user->work2,&w2);CHKERRQ(ierr);
  ierr = VecGetLocalSize(user->wv,&n);CHKERRQ(ierr);


  ierr = VecSet(user->work1,0.0);CHKERRQ(ierr);
  if (user->radiation) {
    ierr = VecAXPY(user->work1,-1.0,user->Pv);CHKERRQ(ierr);
  }
  if (user->domain) {
    ierr = VecCopy(user->cv,user->work3);CHKERRQ(ierr);
    ierr = VecShift(user->work3,-1.0*user->cv_eq);CHKERRQ(ierr);
    ierr = VecCopy(user->Phi2D_V,user->work4);CHKERRQ(ierr);
    ierr = VecScale(user->work4,-1.0);CHKERRQ(ierr);
    ierr = VecShift(user->work4,1.0);CHKERRQ(ierr);
    ierr = VecPointwiseMult(user->work4,user->work4,user->work3);CHKERRQ(ierr);
    ierr = VecScale(user->work4,user->Svr);CHKERRQ(ierr);
    // Parameter tuning: user->grain
    // 5000.0 worked for refinement 2 and 5
    //10000.0 worked for refinement 2, 3, 4, 5
    ierr = VecAXPY(user->work1,user->grain,user->work4);CHKERRQ(ierr);
  }
  ierr = VecScale(user->work1,user->dt);CHKERRQ(ierr);
  ierr = VecAXPY(user->work1,-1.0,user->cv);CHKERRQ(ierr);
  ierr = MatMult(user->M_0,user->work1,user->work2);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    q_p[3*i]=w2[i];
  }

  ierr = MatMult(user->M_0,user->DPsiv,user->work1);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    q_p[3*i+1]=w1[i];
  }

  ierr = VecCopy(user->DPsieta,user->work1);CHKERRQ(ierr);
  ierr = VecScale(user->work1,user->L*user->dt);CHKERRQ(ierr);
  ierr = VecAXPY(user->work1,-1.0,user->eta);CHKERRQ(ierr);
  if (user->radiation) {
    ierr = VecAXPY(user->work1,-1.0,user->Piv);CHKERRQ(ierr);
  }
  ierr = MatMult(user->M_0,user->work1,user->work2);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    q_p[3*i+2]=w2[i];
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
  PetscScalar     Evf=user->Evf,A=user->A,B=user->B,cv0=user->cv0;
  PetscScalar     *cv_p,*eta_p,*logcv_p,*logcv2_p,*DPsiv_p,*DPsieta_p;
  PetscInt        n,i;

  PetscFunctionBegin;

  ierr = VecGetLocalSize(user->cv,&n);
  ierr = VecGetArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->eta,&eta_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->logcv,&logcv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->logcv2,&logcv2_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->DPsiv,&DPsiv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->DPsieta,&DPsieta_p);CHKERRQ(ierr);

  ierr = Llog(user->cv,user->logcv);CHKERRQ(ierr);

  ierr = VecCopy(user->cv,user->work1);CHKERRQ(ierr);
  ierr = VecScale(user->work1,-1.0);CHKERRQ(ierr);
  ierr = VecShift(user->work1,1.0);CHKERRQ(ierr);
  ierr = Llog(user->work1,user->logcv2);CHKERRQ(ierr);

  for (i=0;i<n;i++)
  {
    DPsiv_p[i] = (eta_p[i]-1.0)*(eta_p[i]-1.0)*(eta_p[i]+1.0)*(eta_p[i]+1.0)*( Evf + logcv_p[i] - logcv2_p[i]) - 2.0*A*(cv_p[i] - cv0)*eta_p[i]*(eta_p[i]+2.0)*(eta_p[i]-1.0)*(eta_p[i]-1.0) + 2.0*B*(cv_p[i] - 1.0)*eta_p[i]*eta_p[i];

    DPsieta_p[i] = 4.0*eta_p[i]*(eta_p[i]-1.0)*(eta_p[i]+1.0)*(Evf*cv_p[i] + cv_p[i]*logcv_p[i] + (1.0-cv_p[i])*logcv2_p[i] ) - A*(cv_p[i] - cv0)*(cv_p[i] - cv0)*(4.0*eta_p[i]*eta_p[i]*eta_p[i] - 6.0*eta_p[i] + 2.0) + 2.0*B*(cv_p[i]-1.0)*(cv_p[i]-1.0)*eta_p[i];

  }

  ierr = VecRestoreArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->eta,&eta_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->logcv,&logcv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->logcv2,&logcv2_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->DPsiv,&DPsiv_p);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscInt         n,i,Mda,Nda;
  PetscScalar	   *xx,*cv_p,*wv_p,*eta_p;
  //PetscViewer      view_out;
  /* needed for the void growth case */
  PetscScalar       xmid,ymid,cv_v=1.0,cv_m=user->Sv_scalar*user->cv0,eta_v=1.0,eta_m=0.0,h,lambda;
  PetscInt          nele,nen,idx[3];
  const PetscInt    *ele;
  PetscScalar       x[3],y[3];
  Vec               coords;
  const PetscScalar *_coords;
  PetscScalar       xwidth = user->xmax - user->xmin, ywidth = user->ymax - user->ymin;

  PetscFunctionBegin;

  ierr = VecGetLocalSize(X,&n);CHKERRQ(ierr);

  if (!user->radiation) {
    ierr = DMDAGetInfo(user->da2,PETSC_NULL,&Mda,&Nda,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(user->da2,&coords);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coords,&_coords);CHKERRQ(ierr);

    if (user->periodic) {
      h = (user->xmax-user->xmin)/Mda;
    } else {
      h = (user->xmax-user->xmin)/(Mda-1.0);
    }

    xmid = (user->xmax + user->xmin)/2.0;
    ymid = (user->ymax + user->ymin)/2.0;
    lambda = 4.0*h;

    ierr = DMDAGetElements(user->da2,&nele,&nen,&ele);CHKERRQ(ierr);
    for (i=0;i < nele; i++) {
      idx[0] = ele[3*i]; idx[1] = ele[3*i+1]; idx[2] = ele[3*i+2];

      x[0] = _coords[2*idx[0]]; y[0] = _coords[2*idx[0]+1];
      x[1] = _coords[2*idx[1]]; y[1] = _coords[2*idx[1]+1];
      x[2] = _coords[2*idx[2]]; y[2] = _coords[2*idx[2]+1];

      PetscInt k;
      PetscScalar vals_DDcv[3],vals_cv[3],vals_eta[3],s,hhr,r;
      for (k=0; k < 3 ; k++) {
        //s = PetscAbs(x[k] - xmid);
        s = sqrt((x[k] - xmid)*(x[k] - xmid) + (y[k] - ymid)*(y[k] - ymid));
        if (s <= xwidth*(5.0/64.0)) {
          vals_cv[k] = cv_v;
          vals_eta[k] = eta_v;
          vals_DDcv[k] = 0.0;
        } else if (s> xwidth*(5.0/64.0) && s<= xwidth*(7.0/64.0) ) {
          //r = (s - xwidth*(6.0/64.0) )/(0.5*lambda);
          r = (s - xwidth*(6.0/64.0) )/(xwidth/64.0);
          hhr = 0.25*(-r*r*r + 3*r + 2);
          vals_cv[k] = cv_m + (1.0 - hhr)*(cv_v - cv_m);
          vals_eta[k] = eta_m + (1.0 - hhr)*(eta_v - eta_m);
          vals_DDcv[k] = (cv_v - cv_m)*r*6.0/(lambda*lambda);
        } else {
          vals_cv[k] = cv_m;
          vals_eta[k] = eta_m;
          vals_DDcv[k] = 0.0;
        }
      }

      ierr = VecSetValuesLocal(user->cv,3,idx,vals_cv,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValuesLocal(user->eta,3,idx,vals_eta,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValuesLocal(user->work2,3,idx,vals_DDcv,INSERT_VALUES);CHKERRQ(ierr);

    }
    ierr = VecAssemblyBegin(user->cv);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->cv);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(user->eta);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->eta);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(user->work2);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->work2);CHKERRQ(ierr);

    ierr = DMDARestoreElements(user->da2,&nele,&nen,&ele);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(coords,&_coords);CHKERRQ(ierr);
  } else {
    //ierr = VecSet(user->cv,user->initv);CHKERRQ(ierr);
    //ierr = VecSet(user->ci,user->initv);CHKERRQ(ierr);
    ierr = VecSet(user->cv,.05);CHKERRQ(ierr);
    ierr = VecSet(user->eta,user->initeta);CHKERRQ(ierr);

  }
  ierr = DPsi(user);CHKERRQ(ierr);
  ierr = VecCopy(user->DPsiv,user->wv);CHKERRQ(ierr);
  ierr = VecAXPY(user->wv,-2.0*user->kav,user->work2);CHKERRQ(ierr);

  ierr = VecGetArray(X,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(user->wv,&wv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->eta,&eta_p);CHKERRQ(ierr);

  for (i=0;i<n/3;i++)
  {
    xx[3*i]=wv_p[i];
    xx[3*i+1]=cv_p[i];
    xx[3*i+2]=eta_p[i];
  }

  /*
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_initial",FILE_MODE_WRITE,&view_out);CHKERRQ(ierr);
  ierr = VecView(user->wv,view_out);CHKERRQ(ierr);
  ierr = VecView(user->cv,view_out);CHKERRQ(ierr);
  ierr = VecView(user->eta,view_out);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_out);CHKERRQ(ierr);
   */

  ierr = VecRestoreArray(X,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->wv,&wv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->eta,&eta_p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscReal dt,x,y,strength;
} RandomValues;


#undef __FUNCT__
#define __FUNCT__ "SetRandomVectors"
PetscErrorCode SetRandomVectors(AppCtx* user,PetscReal t)
{
  PetscErrorCode        ierr;
  static RandomValues   *randomvalues = 0;
  static PetscInt       randindex = 0,n; /* indicates how far into the randomvalues we have currently used */
  static PetscReal      randtime = 0; /* indicates time of last radiation event */
  PetscInt              i,j,M,N,cnt = 0;
  PetscInt              xs,ys,xm,ym;

  PetscFunctionBegin;
  if (!randomvalues) {
    PetscViewer viewer;
    char        filename[PETSC_MAX_PATH_LEN];
    PetscBool   flg;
    PetscInt    seed;

    ierr = PetscOptionsGetInt(PETSC_NULL,"-random_seed",&seed,&flg);CHKERRQ(ierr);
    if (flg) {
      sprintf(filename,"ex61.random.%d",(int)seed);CHKERRQ(ierr);
    } else {
      ierr = PetscStrcpy(filename,"ex61.random");CHKERRQ(ierr);
    }
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(viewer,&n,1,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscMalloc(n*sizeof(RandomValues),&randomvalues);CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(viewer,randomvalues,4*n,PETSC_DOUBLE);CHKERRQ(ierr);
    for (i=0; i<n; i++) randomvalues[i].dt = randomvalues[i].dt*user->dtevent;
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  user->maxevents = PetscMin(user->maxevents,n);

  ierr = VecSet(user->Pv,0.0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(user->da1,0,&M,&N,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(user->da1,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  while (user->maxevents > randindex && randtime + randomvalues[randindex].dt < t + user->dt) {  /* radiation event has occured since last time step */
    i = ((PetscInt) (randomvalues[randindex].x*M)) - xs;
    j = ((PetscInt) (randomvalues[randindex].y*N)) - ys;
    if (i >= 0 && i < xm && j >= 0 && j < ym) { /* point is on this process */

      /* need to make sure eta at the given point is not great than .8 */
      ierr = VecSetValueLocal(user->Pv,i  + xm*(j), randomvalues[randindex].strength*user->VG,INSERT_VALUES);CHKERRQ(ierr);
    }
    randtime += randomvalues[randindex++].dt;
    cnt++;
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of radiation events %d\n",cnt);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(user->Pv);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(user->Pv);CHKERRQ(ierr);

  ierr = VecCopy(user->Pv,user->Pi);CHKERRQ(ierr);
  ierr = VecScale(user->Pi,0.9);CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->Piv,user->Pi,user->Pv);CHKERRQ(ierr);
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
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr = DMDAVecGetArrayDOF(da,xl,&l);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,xu,&u);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  for (j=ys; j < ys+ym; j++) {
    for (i=xs; i < xs+xm;i++) {
      l[j][i][0] = -SNES_VI_INF;
      l[j][i][1] = 0.0;
      l[j][i][2] = 0.0;
      u[j][i][0] = SNES_VI_INF;
      u[j][i][1] = 1.0;
      u[j][i][2] = 1.0;
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
  user->xmin  = 0.0; user->xmax = 128.0;
  user->ymin  = 0.0; user->ymax = 128.0;
  user->Mv    = 1.0;
  user->L     = 1.0;
  user->kaeta = 1.0;
  user->kav   = 0.5;
  user->Evf   = 9.09;
  user->A     = 9.09;
  user->B     = 9.09;
  user->cv0   = 1.13e-4;
  user->Sv_scalar    = 500.0;
  user->dt    = 1.0e-5;
  user->T     = 1.0e-2;
  user->initv = .00069;
  user->initeta = .0;
  user->graphics = PETSC_FALSE;
  user->periodic = PETSC_FALSE;
  user->lumpedmass = PETSC_FALSE;
  user->radiation = PETSC_FALSE;
  /* multidomain modeling */
  user->domain = 0;
  user->grain  = 5000.0;
  user->Svr       = 0.5;
  user->Sir       = 0.5;
  user->cv_eq     = 6.9e-4;
  user->ci_eq     = 6.9e-4;
  user->VG        = 100.0;
  user->maxevents = 10;

  user->dtevent = user->dt;
  ierr = PetscOptionsReal("-dtevent","Average time between random events\n","None",user->dtevent,&user->dtevent,&flg);CHKERRQ(ierr);


  ierr = PetscOptionsGetReal(PETSC_NULL,"-xmin",&user->xmin,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-xmax",&user->xmax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-T",&user->T,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-dt",&user->dt,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-graphics","Contour plot solutions at each timestep\n","None",user->graphics,&user->graphics,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-periodic","Use periodic boundary conditions\n","None",user->periodic,&user->periodic,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-radiation","Allow radiation\n","None",user->radiation,&user->radiation,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-lumpedmass","Use lumped mass matrix\n","None",user->lumpedmass,&user->lumpedmass,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-domain","Number of domains (0=one domain, 1=two domains, 2=multidomain\n","None",user->domain,&user->domain,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-VG","Maximum increase in vacancy (or interstitial) concentration due to a cascade event","None",user->VG,&user->VG,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-grain","Some bogus factor that controls the strength of the grain boundaries, makes the picture more plausible","None",user->grain,&user->grain,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-maxevents","Maximum random events allowed\n","None",user->maxevents,&user->maxevents,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-da_refine","da refine \n","None",user->darefine,&user->darefine,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-da_grid_x","da grid x\n","None",user->dagrid,&user->dagrid,&flg);CHKERRQ(ierr);

  PetscFunctionReturn(0);
 }


#undef __FUNCT__
#define __FUNCT__ "SetUpMatrices"
PetscErrorCode SetUpMatrices(AppCtx* user)
{
  PetscErrorCode    ierr;
  PetscInt          nele,nen,i,n;
  const PetscInt    *ele;
  PetscScalar       dt=user->dt,h;

  PetscInt          idx[3];
  PetscScalar       eM_0[3][3],eM_2[3][3];
  Mat               M=user->M;
  Mat               M_0=user->M_0;
  PetscInt          Mda,Nda;


  PetscFunctionBegin;


  ierr = MatGetLocalSize(M,&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(user->da1,PETSC_NULL,&Mda,&Nda,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  if (Mda!=Nda) {
    printf("Currently different Mda and Nda are not supported");
  }
  if (user->periodic) {
    h = (user->xmax-user->xmin)/Mda;
  } else {
    h = (user->xmax-user->xmin)/(Mda-1.0);
  }
  if (user->lumpedmass) {
    eM_0[0][0] = eM_0[1][1] = eM_0[2][2] = h*h/6.0;
    eM_0[0][1] = eM_0[1][0] = eM_0[0][2] = eM_0[2][0] = eM_0[1][2] = eM_0[2][1] = 0.0;
  } else {
    eM_0[0][0] = eM_0[1][1] = eM_0[2][2] = h*h/12.0;
    eM_0[0][1] = eM_0[0][2] = eM_0[1][0] = eM_0[1][2] = eM_0[2][0] = eM_0[2][1] = h*h/24.0;
  }
  eM_2[0][0] = 1.0;
  eM_2[1][1] = eM_2[2][2] = 0.5;
  eM_2[0][1] = eM_2[0][2] = eM_2[1][0]= eM_2[2][0] = -0.5;
  eM_2[1][2] = eM_2[2][1] = 0.0;


  /* Get local element info */
  ierr = DMDAGetElements(user->da1,&nele,&nen,&ele);CHKERRQ(ierr);
  for (i=0;i < nele;i++) {

    idx[0] = ele[3*i]; idx[1] = ele[3*i+1]; idx[2] = ele[3*i+2];

    PetscInt    row,cols[6],r,row_M_0,cols3[3];
    PetscScalar vals[6],vals_M_0[3],vals3[3];

    for (r=0;r<3;r++) {
      row_M_0 = idx[r];
      vals_M_0[0]=eM_0[r][0];
      vals_M_0[1]=eM_0[r][1];
      vals_M_0[2]=eM_0[r][2];

      ierr = MatSetValuesLocal(M_0,1,&row_M_0,3,idx,vals_M_0,ADD_VALUES);CHKERRQ(ierr);

      row = 3*idx[r];
      cols[0] = 3*idx[0];     vals[0] = dt*eM_2[r][0]*user->Mv;
      cols[1] = 3*idx[1];     vals[1] = dt*eM_2[r][1]*user->Mv;
      cols[2] = 3*idx[2];     vals[2] = dt*eM_2[r][2]*user->Mv;
      cols[3] = 3*idx[0]+1;   vals[3] = eM_0[r][0];
      cols[4] = 3*idx[1]+1;   vals[4] = eM_0[r][1];
      cols[5] = 3*idx[2]+1;   vals[5] = eM_0[r][2];

      /* Insert values in matrix M for 1st dof */
      ierr = MatSetValuesLocal(M,1,&row,6,cols,vals,ADD_VALUES);CHKERRQ(ierr);

      row = 3*idx[r]+1;
      cols[0] = 3*idx[0];     vals[0] = -eM_0[r][0];
      cols[1] = 3*idx[1];     vals[1] = -eM_0[r][1];
      cols[2] = 3*idx[2];     vals[2] = -eM_0[r][2];
      cols[3] = 3*idx[0]+1;   vals[3] = 2.0*user->kav*eM_2[r][0];
      cols[4] = 3*idx[1]+1;   vals[4] = 2.0*user->kav*eM_2[r][1];
      cols[5] = 3*idx[2]+1;   vals[5] = 2.0*user->kav*eM_2[r][2];

      /* Insert values in matrix M for 2nd dof */
      ierr = MatSetValuesLocal(M,1,&row,6,cols,vals,ADD_VALUES);CHKERRQ(ierr);

      row = 3*idx[r]+2;
      cols3[0] = 3*idx[0]+2;   vals3[0] = eM_0[r][0] + user->dt*2.0*user->L*user->kaeta*eM_2[r][0];
      cols3[1] = 3*idx[1]+2;   vals3[1] = eM_0[r][1] + user->dt*2.0*user->L*user->kaeta*eM_2[r][1];
      cols3[2] = 3*idx[2]+2;   vals3[2] = eM_0[r][2] + user->dt*2.0*user->L*user->kaeta*eM_2[r][2];
      ierr = MatSetValuesLocal(M,1,&row,3,cols3,vals3,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(M_0,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M_0,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = DMDARestoreElements(user->da1,&nele,&nen,&ele);CHKERRQ(ierr);


  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckRedundancy"
PetscErrorCode CheckRedundancy(SNES snes, IS act, IS *outact, DM da)
{
  PetscErrorCode ierr;
  PetscScalar    **uin,**uout;
  Vec            UIN, UOUT;
  PetscInt       xs,xm,*outindex;
  const PetscInt *index;
  PetscInt       k,i,l,n,M,cnt=0;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,0,&M,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da,&UIN);CHKERRQ(ierr);
  ierr = VecSet(UIN,0.0);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&UOUT);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,UIN,&uin);CHKERRQ(ierr);
  ierr = ISGetIndices(act,&index);CHKERRQ(ierr);
  ierr = ISGetLocalSize(act,&n);CHKERRQ(ierr);
  for (k=0;k<n;k++){
    l = index[k]%5;
    i = index[k]/5;
    uin[i][l]=1.0;
  }
  printf("Number of active constraints before applying redundancy %d\n",n);
  ierr = ISRestoreIndices(act,&index);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(da,UIN,&uin);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,UIN,INSERT_VALUES,UOUT);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,UIN,INSERT_VALUES,UOUT);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,UOUT,&uout);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  for (i=xs; i < xs+xm;i++) {
    if (uout[i-1][1] && uout[i][1] && uout[i+1][1])
             uout[i][0] = 1.0;
    if (uout[i-1][3] && uout[i][3] && uout[i+1][3])
             uout[i][2] = 1.0;
  }

  for (i=xs; i < xs+xm;i++) {
    for (l=0;l<5;l++) {
      if (uout[i][l])
        cnt++;
    }
  }

  printf("Number of active constraints after applying redundancy %d\n",cnt);


  ierr = PetscMalloc(cnt*sizeof(PetscInt),&outindex);CHKERRQ(ierr);
  cnt = 0;

  for (i=xs; i < xs+xm;i++) {
    for (l=0;l<5;l++) {
      if (uout[i][l])
        outindex[cnt++] = 5*(i)+l;
    }
  }


  ierr = ISCreateGeneral(PETSC_COMM_WORLD,cnt,outindex,PETSC_OWN_POINTER,outact);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(da,UOUT,&uout);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&UIN);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&UOUT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Phi"
PetscErrorCode Phi(AppCtx* user)
{
  PetscErrorCode     ierr;
  PetscScalar        xmid, xqu, lambda, h,x[3],y[3];
  Vec                coords;
  const PetscScalar  *_coords;
  PetscInt           nele,nen,i,idx[3],Mda,Nda;
  const PetscInt     *ele;
  //PetscViewer        view;

  PetscFunctionBegin;

  ierr = DMDAGetInfo(user->da1,PETSC_NULL,&Mda,&Nda,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(user->da2,&coords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords,&_coords);CHKERRQ(ierr);

  h = (user->xmax - user->xmin)/Mda;
  xmid = (user->xmin + user->xmax)/2.0;
  xqu = (user->xmin + xmid)/2.0;
  lambda = 4.0*h;


  ierr = DMDAGetElements(user->da2,&nele,&nen,&ele);CHKERRQ(ierr);
  for (i=0;i < nele; i++) {
    idx[0] = ele[3*i]; idx[1] = ele[3*i+1]; idx[2] = ele[3*i+2];
    //printf("idx[0]=%d,idx[1]=%d,idx[2]=%d\n",idx[0],idx[1],idx[2]);

    x[0] = _coords[2*idx[0]]; y[0] = _coords[2*idx[0]+1];
    x[1] = _coords[2*idx[1]]; y[1] = _coords[2*idx[1]+1];
    x[2] = _coords[2*idx[2]]; y[2] = _coords[2*idx[2]+1];

    //printf("x[0]=%f,x[1]=%f,x[2]=%f\n",x[0],x[1],x[2]);
    //printf("y[0]=%f,y[1]=%f,y[2]=%f\n",y[0],y[1],y[2]);

    PetscScalar vals1[3],vals2[3],dist1,dist2,s1,r,hhr,xc1,xc2;
    PetscInt    k;

    xc1 = user->xmin;
    xc2 = xmid;

    //ierr = VecSet(user->phi1,0.0);CHKERRQ(ierr);
    for (k=0;k < 3; k++) {
      if (x[k]-xqu > 0) {
        s1 = (x[k] - xqu);
      } else {
        s1 = -(x[k] - xqu);
      }
      if (x[k] - xc1 > 0) {
        dist1 = (x[k] - xc1);
      } else {
        dist1 = -(x[k] - xc1);
      }
      if (x[k] - xc2 > 0) {
        dist2 = (x[k] - xc2);
      } else {
        dist2 = -(x[k] - xc2);
      }
      if (dist1 <= 0.5*lambda) {
        r = (x[k]-xc1)/(0.5*lambda);
        hhr = 0.25*(-r*r*r + 3.0*r + 2.0);
        vals1[k] = hhr;
      }
      else if (dist2 <= 0.5*lambda) {
        r = (x[k]-xc2)/(0.5*lambda);
        hhr = 0.25*(-r*r*r + 3.0*r + 2.0);
        vals1[k] = 1.0 - hhr;
      }
      else if (s1 <= xqu - 2.0*h) {
        vals1[k] = 1.0;
      }

      //else if ( abs(x[k]-(user->xmax-h)) < 0.1*h ) {
      else if ( (user->xmax-h)-x[k] < 0.1*h ) {
        vals1[k] = .15625;
       }
      else {
        vals1[k] = 0.0;
      }
    }

    ierr = VecSetValuesLocal(user->phi1,3,idx,vals1,INSERT_VALUES);CHKERRQ(ierr);

    xc1 = xmid;
    xc2 = user->xmax;

    //ierr = VecSet(user->phi2,0.0);CHKERRQ(ierr);
    for (k=0;k < 3; k++) {
      /*
      s1 = abs(x[k] - (xqu+xmid));
      dist1 = abs(x[k] - xc1);
      dist2 = abs(x[k] - xc2);
       */
      if (x[k]-(xqu + xmid) > 0) {
        s1 = (x[k] - (xqu + xmid));
      } else {
        s1 = -(x[k] - (xqu + xmid));
      }
      if (x[k] - xc1 > 0) {
        dist1 = (x[k] - xc1);
      } else {
        dist1 = -(x[k] - xc1);
      }
      if (x[k] - xc2 > 0) {
        dist2 = (x[k] - xc2);
      } else {
        dist2 = -(x[k] - xc2);
      }

      if (dist1 <= 0.5*lambda) {
        r = (x[k]-xc1)/(0.5*lambda);
        hhr = 0.25*(-r*r*r + 3.0*r + 2.0);
        vals2[k] = hhr;
      }
      else if (dist2 <= 0.5*lambda) {
        r = -(x[k]-xc2)/(0.5*lambda);
        hhr = 0.25*(-r*r*r + 3.0*r + 2.0);
        vals2[k] = hhr;
      }
      else if (s1 <= xqu - 2.0*h) {
        vals2[k] = 1.0;
      }

      else if ( x[k]-(user->xmin) < 0.1*h ) {
        vals2[k] = 0.5;
      }


      else if ( (x[k]-(user->xmin+h)) < 0.1*h ) {
        vals2[k] = .15625;
      }

      else {
        vals2[k] = 0.0;
      }

    }

    ierr = VecSetValuesLocal(user->phi2,3,idx,vals2,INSERT_VALUES);CHKERRQ(ierr);
    /*
    for (k=0;k < 3; k++) {
      vals_sum[k] = vals1[k]*vals1[k] + vals2[k]*vals2[k];
    }
     */
    //ierr = VecSetValuesLocal(user->Phi2D_V,3,idx,vals_sum,INSERT_VALUES);CHKERRQ(ierr);

  }

  ierr = VecAssemblyBegin(user->phi1);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(user->phi1);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(user->phi2);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(user->phi2);CHKERRQ(ierr);

  /*
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_phi",FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  ierr = VecView(user->phi1,view);CHKERRQ(ierr);
  ierr = VecView(user->phi2,view);CHKERRQ(ierr);
   */

  //ierr = VecView(user->phi1,0);CHKERRQ(ierr);
  //ierr = VecView(user->phi2,0);CHKERRQ(ierr);

  ierr = VecPointwiseMult(user->phi1,user->phi1,user->phi1);CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->phi2,user->phi2,user->phi2);CHKERRQ(ierr);
  /*
  ierr = VecView(user->phi1,view);CHKERRQ(ierr);
  ierr = VecView(user->phi2,view);CHKERRQ(ierr);
   */

  ierr = VecCopy(user->phi1,user->Phi2D_V);CHKERRQ(ierr);
  ierr = VecAXPY(user->Phi2D_V,1.0,user->phi2);CHKERRQ(ierr);
  //ierr = VecView(user->Phi2D_V,0);CHKERRQ(ierr);

  //ierr = VecView(user->Phi2D_V,view);CHKERRQ(ierr);
  //ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "Phi_read"
PetscErrorCode Phi_read(AppCtx* user)
{
  PetscErrorCode     ierr;
  PetscReal          *values;
  PetscViewer        viewer;
  PetscInt           power;

  PetscFunctionBegin;

  power = user->darefine + (PetscInt)(PetscLogScalar(user->dagrid)/PetscLogScalar(2.0));
  switch (power) {
  case 6:
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi4",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(user->Phi2D_V,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    break;
  case 7:
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi3",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(user->Phi2D_V,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    break;
  case 8:
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi2",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(user->Phi2D_V,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    break;
  case 9:
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi1",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(user->Phi2D_V,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    break;
  case 10:
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(user->Phi2D_V,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    break;
  case 11:
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phim1",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(user->Phi2D_V,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    break;
  case 12:
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phim2",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(user->Phi2D_V,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    break;
  case 13:
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phim3",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(user->Phi2D_V,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    break;
  case 14:
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phim4",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(user->Phi2D_V,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    break;
  }
  PetscFunctionReturn(0);
}
