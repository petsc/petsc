static char help[] = "2D coupled Allen-Cahn and Cahn-Hilliard equation for constant mobility and triangular elements. Use periodic boundary condidtions.\n\
Runtime options include:\n\
-xmin <xmin>\n\
-xmax <xmax>\n\
-ymin <ymin>\n\
-T <T>, where <T> is the end time for the time domain simulation\n\
-dt <dt>,where <dt> is the step size for the numerical integration\n\
-gamma <gamma>\n\
-theta_c <theta_c>\n\n";

/*
 ./ex61 -ksp_type gmres -snes_vi_monitor   -snes_atol 1.e-11  -da_refine 3  -T 0.1   -ksp_monitor_true_residual -pc_type lu -pc_factor_mat_solver_package superlu -snes_converged_reason -ksp_converged_reason  -ksp_rtol 1.e-9  -snes_ls_monitor -VG 10 -draw_fields 1,3,4 -snes_ls basic

./ex61 -ksp_type gmres -snes_vi_monitor   -snes_atol 1.e-11  -da_refine 4 -T 0.1   -ksp_monitor_true_residual -pc_type sor -snes_converged_reason -ksp_converged_reason  -ksp_rtol 1.e-9  -snes_ls_monitor -VG 10 -draw_fields 1,3,4 -snes_ls basic

./ex61 -ksp_type fgmres -snes_vi_monitor   -snes_atol 1.e-11  -da_refine 5 -T 0.1   -ksp_monitor_true_residual -snes_converged_reason -ksp_converged_reason  -ksp_rtol 1.e-9  -snes_ls_monitor -VG 10 -draw_fields 1,3,4 -snes_ls basic -pc_type mg -pc_mg_galerkin

./ex61 -ksp_type fgmres -snes_vi_monitor   -snes_atol 1.e-11  -da_refine 5 -snes_converged_reason -ksp_converged_reason   -snes_ls_monitor -VG 1 -draw_fields 1,3,4  -pc_type mg -pc_mg_galerkin -log_summary -dt .0000000000001 -mg_coarse_pc_type svd  -ksp_monitor_true_residual -ksp_rtol 1.e-9

Movie version
./ex61 -ksp_type fgmres -snes_vi_monitor   -snes_atol 1.e-11  -da_refine 6 -snes_converged_reason -ksp_converged_reason   -snes_ls_monitor -VG 10000000 -draw_fields 1,3,4  -pc_type mg -pc_mg_galerkin -log_summary -dt .000001 -mg_coarse_pc_type svd  -ksp_monitor_true_residual -ksp_rtol 1.e-9 -snes_ls basic -T .0020 -P_casc .0005

 */

/*
   Possible additions to the code. At each iteration count the number of solution elements that are at the upper bound and stop the program if large

   Is the solution at time 0 nonsense?  Looks totally different from first time step. Why does cubic line search at beginning screw it up? 

   Add command-line option for constant or degenerate mobility 
   Add command-line option for graphics at each time step

   Check time-step business; what should it be? How to check that it is good? 
   Make random right hand side forcing function proportional to time step so smaller time steps don't mean more radiation
   How does the multigrid linear solver work now?
   What happens when running with degenerate mobility

   
 */

#include "petscsnes.h"
#include "petscdmda.h"

typedef struct{
  PetscReal   dt,T; /* Time step and end time */
  DM          da1,da2;
  Mat         M;    /* Jacobian matrix */
  Mat         M_0;
  Vec         q,wv,cv,wi,ci,eta,cvi,DPsiv,DPsii,DPsieta,Pv,Pi,Piv,logcv,logci,logcvi,Rr,Riv;
  Vec         work1,work2,work3,work4;
  PetscScalar Dv,Di,Evf,Eif,A,kBT,kav,kai,kaeta,Rsurf,Rbulk,L,P_casc,VG; /* physics parameters */
  PetscReal   xmin,xmax,ymin,ymax;
  PetscInt    Mda, Nda;
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
PetscErrorCode LaplacianFiniteDifference(AppCtx*);
PetscErrorCode Llog(Vec,Vec);
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode      ierr;
  Vec                 x,r;  /* Solution and residual vectors */
  SNES                snes; /* Nonlinear solver context */
  AppCtx              user; /* Application context */
  Vec                 xl,xu; /* Upper and lower bounds on variables */
  Mat                 J;
  PetscScalar         t=0.0;
  /*  PetscViewer         view_out, view_q, view_psi, view_mat;*/
  /*  PetscViewer         view_rand;*/
  IS                  inactiveconstraints;
  PetscInt            ninactiveconstraints,N;
  SNESConvergedReason reason;
  PetscViewer         view_out, view_cv,view_eta,view_vtk_cv,view_vtk_eta;
  char                cv_filename[80],eta_filename[80];

  PetscInitialize(&argc,&argv, (char*)0, help);
  
  /* Get physics and time parameters */
  ierr = GetParams(&user);CHKERRQ(ierr);
  /* Create a 1D DA with dof = 5; the whole thing */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX, -3,-3,PETSC_DECIDE,PETSC_DECIDE, 5, 1,PETSC_NULL,PETSC_NULL,&user.da1);CHKERRQ(ierr);
 
  /* Create a 1D DA with dof = 1; for individual componentes */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX, -3,-3,PETSC_DECIDE,PETSC_DECIDE, 1, 1,PETSC_NULL,PETSC_NULL,&user.da2);CHKERRQ(ierr);


  /* Set Element type (triangular) */
  ierr = DMDASetElementType(user.da1,DMDA_ELEMENT_P1);CHKERRQ(ierr);
  ierr = DMDASetElementType(user.da2,DMDA_ELEMENT_P1);CHKERRQ(ierr);
  
  /* Set x and y coordinates */
  ierr = DMDASetUniformCoordinates(user.da1,user.xmin,user.xmax,user.ymin,user.ymax,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(user.da2,user.xmin,user.xmax,user.ymin,user.ymax,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);


  /* Get global vector x from DM (da1) and duplicate vectors r,xl,xu */
  ierr = DMCreateGlobalVector(user.da1,&x);CHKERRQ(ierr);
  ierr = VecGetSize(x,&N);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xl);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xu);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&user.q);CHKERRQ(ierr);
  
  /* Get global vector user->wv from da2 and duplicate other vectors */
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
  ierr = VecDuplicate(user.wv,&user.logcv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.logci);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.logcvi);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.work1);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.work2);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.Rr);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.Riv);CHKERRQ(ierr);
  

  /* Get Jacobian matrix structure from the da for the entire thing, da1 */
  ierr = DMGetMatrix(user.da1,MATAIJ,&user.M);CHKERRQ(ierr);
  /* Get the (usual) mass matrix structure from da2 */
  ierr = DMGetMatrix(user.da2,MATAIJ,&user.M_0);CHKERRQ(ierr);
  ierr = SetInitialGuess(x,&user);CHKERRQ(ierr);
  /* Form the jacobian matrix and M_0 */
  ierr = SetUpMatrices(&user);CHKERRQ(ierr);
  ierr = MatDuplicate(user.M,MAT_DO_NOT_COPY_VALUES,&J);CHKERRQ(ierr);
  
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,user.da1);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,r,FormFunction,(void*)&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,(void*)&user);CHKERRQ(ierr);
 

  ierr = SNESSetType(snes,SNESVI);CHKERRQ(ierr);
  ierr = SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,100,PETSC_DEFAULT);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SetVariableBounds(user.da1,xl,xu);CHKERRQ(ierr);
  ierr = SNESVISetVariableBounds(snes,xl,xu);CHKERRQ(ierr);
  
  /*  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_rand",FILE_MODE_WRITE,&view_rand);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_mat2",FILE_MODE_WRITE,&view_mat);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_q",FILE_MODE_WRITE,&view_q);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_out",FILE_MODE_WRITE,&view_out);CHKERRQ(ierr);
   ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_psi",FILE_MODE_WRITE,&view_psi);CHKERRQ(ierr);*/
 
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_out",FILE_MODE_WRITE,&view_out);CHKERRQ(ierr);
  
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_cv",FILE_MODE_WRITE,&view_cv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_eta",FILE_MODE_WRITE,&view_eta);CHKERRQ(ierr);
  

  ierr = VecView(x,PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD));CHKERRQ(ierr);  
  while (t<user.T) {
    ierr = SNESSetFunction(snes,r,FormFunction,(void*)&user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,FormJacobian,(void*)&user);CHKERRQ(ierr);

    ierr = SetRandomVectors(&user);CHKERRQ(ierr);
    /*    ierr = VecView(user.Pv,view_rand);CHKERRQ(ierr);
    ierr = VecView(user.Pi,view_rand);CHKERRQ(ierr);
     ierr = VecView(user.Piv,view_rand);CHKERRQ(ierr);*/

    ierr = DPsi(&user);CHKERRQ(ierr);
    /*    ierr = VecView(user.DPsiv,view_psi);CHKERRQ(ierr);
    ierr = VecView(user.DPsii,view_psi);CHKERRQ(ierr);
     ierr = VecView(user.DPsieta,view_psi);CHKERRQ(ierr);*/

    ierr = Update_q(&user);CHKERRQ(ierr);
    /*    ierr = VecView(user.q,view_q);CHKERRQ(ierr);*/
    /*  ierr = MatView(user.M,view_mat);CHKERRQ(ierr);*/

    
   
    sprintf(cv_filename,"file_cv_%f.vtk",t);
    sprintf(eta_filename,"file_eta_%f.vtk",t);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,cv_filename,&view_vtk_cv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,eta_filename,&view_vtk_eta);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(view_vtk_cv, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(view_vtk_eta, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = DMView(user.da2,view_vtk_cv);CHKERRQ(ierr);
    ierr = DMView(user.da2,view_vtk_eta);CHKERRQ(ierr);
    ierr = VecView(user.cv,view_cv);CHKERRQ(ierr);
    ierr = VecView(user.eta,view_eta);CHKERRQ(ierr);
    ierr = VecView(user.cv,view_vtk_cv);CHKERRQ(ierr);
    ierr = VecView(user.eta,view_vtk_eta);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view_vtk_cv);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view_vtk_eta);CHKERRQ(ierr);

        
    ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes,&reason);CHKERRQ(ierr);
    if (reason < 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Nonlinear solver failed");
    ierr = SNESVIGetInactiveSet(snes,&inactiveconstraints);CHKERRQ(ierr);
    ierr = ISGetSize(inactiveconstraints,&ninactiveconstraints);CHKERRQ(ierr);
    /* if (ninactiveconstraints < .90*N) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP,"To many active constraints, model has become non-physical"); */

    /*    ierr = VecView(x,view_out);CHKERRQ(ierr);*/
    ierr = VecView(x,PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD));CHKERRQ(ierr);
    ierr = VecView(x,PETSC_VIEWER_BINARY_(PETSC_COMM_WORLD));CHKERRQ(ierr);
    PetscInt its;
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"SNESVI solver converged at t = %g in %d iterations\n",t,its);CHKERRQ(ierr);

    ierr = Update_u(x,&user);CHKERRQ(ierr);
    ierr = UpdateMatrices(&user);CHKERRQ(ierr);
    t = t + user.dt;
  }
   
  /*  ierr = PetscViewerDestroy(&view_rand);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_mat);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_q);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_out);CHKERRQ(ierr);
   ierr = PetscViewerDestroy(&view_psi);CHKERRQ(ierr);*/
  ierr = PetscViewerDestroy(&view_out);CHKERRQ(ierr);
  
  ierr = PetscViewerDestroy(&view_cv);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_eta);CHKERRQ(ierr);
  
  
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
  ierr = VecDestroy(&user.logcv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.logci);CHKERRQ(ierr);
  ierr = VecDestroy(&user.logcvi);CHKERRQ(ierr);
  ierr = VecDestroy(&user.work1);CHKERRQ(ierr);
  ierr = VecDestroy(&user.work2);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Rr);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Riv);CHKERRQ(ierr);
  ierr = MatDestroy(&user.M);CHKERRQ(ierr);
  ierr = MatDestroy(&user.M_0);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da1);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da2);CHKERRQ(ierr);

 
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


  for(i=0;i<n;i++) {
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
  PetscScalar    *q_p,*w1,*w2;
  PetscInt       i,n;

  PetscFunctionBegin;
  
  ierr = VecPointwiseMult(user->Rr,user->eta,user->eta);CHKERRQ(ierr);
  ierr = VecScale(user->Rr,user->Rsurf);CHKERRQ(ierr);
  ierr = VecShift(user->Rr,user->Rbulk);CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->Riv,user->cv,user->ci);CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->Riv,user->Rr,user->Riv);CHKERRQ(ierr);

  ierr = VecGetArray(user->q,&q_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->work1,&w1);CHKERRQ(ierr);
  ierr = VecGetArray(user->work2,&w2);CHKERRQ(ierr);

  ierr = VecCopy(user->cv,user->work1);CHKERRQ(ierr);
  ierr = VecAXPY(user->work1,1.0,user->Pv);CHKERRQ(ierr); 
  ierr = VecScale(user->work1,-1.0);CHKERRQ(ierr);
  ierr = MatMult(user->M_0,user->work1,user->work2);CHKERRQ(ierr);
  ierr = VecGetLocalSize(user->work1,&n);CHKERRQ(ierr);
  
 for (i=0;i<n;i++) {
       q_p[5*i]=w2[i];
  }
 
  ierr = MatMult(user->M_0,user->DPsiv,user->work1);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
       q_p[5*i+1]=w1[i];
  }

  ierr = VecCopy(user->ci,user->work1);CHKERRQ(ierr);
  ierr = VecAXPY(user->work1,1.0,user->Pi);CHKERRQ(ierr);
  ierr = VecScale(user->work1,-1.0);CHKERRQ(ierr);
  ierr = MatMult(user->M_0,user->work1,user->work2);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
       q_p[5*i+2]=w2[i];
  }

  ierr = MatMult(user->M_0,user->DPsii,user->work1);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
       q_p[5*i+3]=w1[i];
  }

  ierr = VecCopy(user->eta,user->work1);CHKERRQ(ierr);
  ierr = VecScale(user->work1,-1.0/user->dt);CHKERRQ(ierr);
  ierr = VecAXPY(user->work1,user->L,user->DPsieta);CHKERRQ(ierr);
  ierr = VecAXPY(user->work1,-1.0,user->Piv);CHKERRQ(ierr);
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

  ierr = VecGetLocalSize(user->cv,&n);CHKERRQ(ierr);
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
    DPsiv_p[i] = (eta_p[i]-1.0)*(eta_p[i]-1.0)*( Evf + kBT*(logcv_p[i] - logcvi_p[i]) ) + eta_p[i]*eta_p[i]*2*A*(cv_p[i]-1);

    DPsii_p[i] = (eta_p[i]-1.0)*(eta_p[i]-1.0)*( Eif + kBT*(logci_p[i] - logcvi_p[i]) ) + eta_p[i]*eta_p[i]*2*A*ci_p[i] ;
  
    DPsieta_p[i] = 2.0*(eta_p[i]-1.0)*( Evf*cv_p[i] + Eif*ci_p[i] + kBT*( cv_p[i]* logcv_p[i] + ci_p[i]* logci_p[i] + (1-cv_p[i]-ci_p[i])*logcvi_p[i] ) ) + 2.0*eta_p[i]*A*( (cv_p[i]-1.0)*(cv_p[i]-1.0) + ci_p[i]*ci_p[i]);          
    
   
  }

  ierr = VecRestoreArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->ci,&ci_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->eta,&eta_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->logcv,&logcv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->logci,&logci_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->logcvi,&logcvi_p);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscInt          n,i;
  PetscScalar	   *xx,*cv_p,*ci_p,*wv_p,*wi_p;
  /*  PetscViewer       view; */
  PetscScalar       initv = .00069;

  PetscFunctionBegin;

  /*  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_initial",FILE_MODE_WRITE,&view);CHKERRQ(ierr);*/
  ierr = VecGetLocalSize(X,&n);CHKERRQ(ierr);

  ierr = VecSet(user->cv,initv);CHKERRQ(ierr);
  ierr = VecSet(user->ci,initv);CHKERRQ(ierr);
  ierr = VecSet(user->eta,0.0);CHKERRQ(ierr);

  ierr = DPsi(user);CHKERRQ(ierr);
  ierr = VecCopy(user->DPsiv,user->wv);CHKERRQ(ierr);
  ierr = VecCopy(user->DPsii,user->wi);CHKERRQ(ierr);

  ierr = VecGetArray(X,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->ci,&ci_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->wv,&wv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->wi,&wi_p);CHKERRQ(ierr);
  for (i=0;i<n/5;i++)
  {
    xx[5*i]=wv_p[i];
    xx[5*i+1]=cv_p[i];
    xx[5*i+2]=wi_p[i];
    xx[5*i+3]=ci_p[i];
    xx[5*i+4]=0.0;
  }

  /* ierr = VecView(user->wv,view);CHKERRQ(ierr);
  ierr = VecView(user->cv,view);CHKERRQ(ierr);
  ierr = VecView(user->wi,view);CHKERRQ(ierr);
  ierr = VecView(user->ci,view);CHKERRQ(ierr);
   ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);*/

  ierr = VecRestoreArray(X,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->ci,&ci_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->wv,&wv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->wi,&wi_p);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetRandomVectors"
PetscErrorCode SetRandomVectors(AppCtx* user)
{
  PetscErrorCode ierr;
  PetscInt       i,n,count=0;
  PetscScalar    *w1,*w2,*Pv_p,*eta_p;
  /* static PetscViewer viewer = 0; */
  static PetscRandom    rand = 0;
  static PetscInt       step = 0;

  PetscFunctionBegin;
  if (!rand) {
    PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
    PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  }
  
  ierr = VecSetRandom(user->work1,rand);CHKERRQ(ierr);
  ierr = VecSetRandom(user->work2,rand);CHKERRQ(ierr);
  ierr = VecGetArray(user->work1,&w1);CHKERRQ(ierr);
  ierr = VecGetArray(user->work2,&w2);CHKERRQ(ierr);
  ierr = VecGetArray(user->Pv,&Pv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->eta,&eta_p);CHKERRQ(ierr);
  ierr = VecGetLocalSize(user->work1,&n);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
   
    if ( eta_p[i]>=0.8 || w1[i]>user->P_casc){
      Pv_p[i]=0;
   
    }
    else
    {
      Pv_p[i]=w2[i]*user->VG*user->dt;
      count=count+1;
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
  printf("count %d n %d\n",count,n);
  /*
  if (!viewer) {
    ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL,"Random",0,0,300,300,&viewer);CHKERRQ(ierr);
  }
  ierr = VecView(user->Pv,viewer);CHKERRQ(ierr);
   */
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
  ierr = DMDAVecGetArrayDOF(da,xl,&l);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,xu,&u);CHKERRQ(ierr);
  
  ierr = DMDAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);
  
  for (j=ys; j<ys+ym; j++) {
    for(i=xs; i < xs+xm;i++) {
      l[j][i][0] = -SNES_VI_INF;
      l[j][i][1] = 0.0;
      l[j][i][2] = -SNES_VI_INF;
      l[j][i][3] = 0.0;
      l[j][i][4] = 0.0;
      u[j][i][0] = SNES_VI_INF;
      u[j][i][1] = 1.0;
      u[j][i][2] = SNES_VI_INF;
      u[j][i][3] = 1.0;
      u[j][i][4] = 1.0;
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
  user->Dv = 1.0; user->Di=4.0;
  user->Evf = 0.8; user->Eif = 1.2;
  user->A = 1.0;
  user->kBT = 0.11;
  user->kav = 1.0; user->kai = 1.0; user->kaeta = 1.0;
  user->Rsurf = 10.0; user->Rbulk = 1.0;
  user->L = 10.0; user->P_casc = 0.05;
  user->T = 1.0e-2;    user->dt = 1.0e-4;
  user->VG = 100.0;

  ierr = PetscOptionsGetReal(PETSC_NULL,"-xmin",&user->xmin,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-xmax",&user->xmax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-T",&user->T,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-dt",&user->dt,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-P_casc",&user->P_casc,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-VG",&user->VG,&flg);CHKERRQ(ierr);
   

  PetscFunctionReturn(0);
 }


#undef __FUNCT__
#define __FUNCT__ "SetUpMatrices"
PetscErrorCode SetUpMatrices(AppCtx* user)
{
  PetscErrorCode    ierr;
  PetscInt          nele,nen,i,n;
  const PetscInt    *ele;
  PetscScalar       dt=user->dt,hx,hy;
  
  PetscInt          idx[3];
  PetscScalar       eM_0[3][3],eM_2_even[3][3],eM_2_odd[3][3];
  PetscScalar       cv_sum, ci_sum;
  Mat               M=user->M;
  Mat               M_0=user->M_0;
  PetscInt          Mda=user->Mda, Nda=user->Nda;
  PetscScalar       *cv_p,*ci_p;
   
  PetscFunctionBegin;
 
  /*  ierr = MatSetOption(M,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
   ierr = MatSetOption(M_0,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);*/

  /* Create the mass matrix M_0 */
  ierr = VecGetArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->ci,&ci_p);CHKERRQ(ierr);
  ierr = MatGetLocalSize(M,&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(user->da1,PETSC_NULL,&Mda,&Nda,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  hx = (user->xmax-user->xmin)/Mda;
  hy = (user->ymax-user->ymin)/Nda;

  eM_0[0][0]=eM_0[1][1]=eM_0[2][2]=hx*hy/12.0;
  eM_0[0][1]=eM_0[0][2]=eM_0[1][0]=eM_0[1][2]=eM_0[2][0]=eM_0[2][1]=hx*hy/24.0;
 
  eM_2_odd[0][0] = 1.0;
  eM_2_odd[1][1] = eM_2_odd[2][2] = 0.5;
  eM_2_odd[0][1] = eM_2_odd[0][2] = eM_2_odd[1][0]= eM_2_odd[2][0] = -0.5;
  eM_2_odd[1][2] = eM_2_odd[2][1] = 0.0;

  eM_2_even[0][0] = 1.0;
  eM_2_even[1][1] = eM_2_even[2][2] = 0.5;
  eM_2_even[0][1] = eM_2_even[0][2] = eM_2_even[1][0]= eM_2_even[2][0] = -0.5;
  eM_2_even[1][2] = eM_2_even[2][1] = 0.0;

  /*  eM_2_even[1][1] = 1.0;
  eM_2_even[0][0] = eM_2_even[2][2] = 0.5;
  eM_2_even[0][1] = eM_2_even[1][0] = eM_2_even[1][2] = eM_2_even[2][1] = -0.5;
  eM_2_even[0][2] = eM_2_even[2][0] = 0.0;
   */

  //  for(k=0;k < Mda*Nda*2;k++) {
  ierr = DMDAGetElements(user->da1,&nele,&nen,&ele);CHKERRQ(ierr);
  for (i=0; i < nele; i++) {
    /*
    idx[0] = connect[k*3];
    idx[1] = connect[k*3+1];
    idx[2] = connect[k*3+2];
     */
    idx[0] = ele[3*i];
    idx[1] = ele[3*i+1];
    idx[2] = ele[3*i+2];

    PetscInt    row,cols[6],r,row_M_0,cols3[3];
    PetscScalar vals[6],vals_M_0[3],vals3[3];
    
    for(r=0;r<3;r++) {
      //row_M_0 = connect[k*3+r];
      row_M_0 = idx[r];

      vals_M_0[0]=eM_0[r][0];
      vals_M_0[1]=eM_0[r][1];
      vals_M_0[2]=eM_0[r][2];
      
     
      ierr = MatSetValuesLocal(M_0,1,&row_M_0,3,idx,vals_M_0,ADD_VALUES);CHKERRQ(ierr);
       
      //cv_sum = (cv_p[idx[0]] + cv_p[idx[1]] + cv_p[idx[2]])*user->Dv/(3.0*user->kBT);
      //ci_sum = (ci_p[idx[0]] + ci_p[idx[1]] + ci_p[idx[2]])*user->Di/(3.0*user->kBT);
      cv_sum = .0000069*user->Dv/user->kBT;
      ci_sum = .0000069*user->Di/user->kBT;
      

        
        row = 5*idx[r];
        cols[0] = 5*idx[0];     vals[0] = dt*eM_2_odd[r][0]*cv_sum;
        cols[1] = 5*idx[1];     vals[1] = dt*eM_2_odd[r][1]*cv_sum; 
        cols[2] = 5*idx[2];     vals[2] = dt*eM_2_odd[r][2]*cv_sum;
        cols[3] = 5*idx[0]+1;   vals[3] = eM_0[r][0];
        cols[4] = 5*idx[1]+1;   vals[4] = eM_0[r][1];
        cols[5] = 5*idx[2]+1;   vals[5] = eM_0[r][2];

        
        ierr = MatSetValuesLocal(M,1,&row,6,cols,vals,ADD_VALUES);CHKERRQ(ierr);
        
        
        row = 5*idx[r]+1;
        cols[0] = 5*idx[0];     vals[0] = -1.0*eM_0[r][0];
        cols[1] = 5*idx[1];     vals[1] = -1.0*eM_0[r][1];
        cols[2] = 5*idx[2];     vals[2] = -1.0*eM_0[r][2]; 
        cols[3] = 5*idx[0]+1;   vals[3] =  user->kav*eM_2_odd[r][0];
        cols[4] = 5*idx[1]+1;   vals[4] =  user->kav*eM_2_odd[r][1];
        cols[5] = 5*idx[2]+1;   vals[5] =  user->kav*eM_2_odd[r][2];
       
        ierr = MatSetValuesLocal(M,1,&row,6,cols,vals,ADD_VALUES);CHKERRQ(ierr);  
        
        row = 5*idx[r]+2;
        cols[0] = 5*idx[0]+2;   vals[0] =  dt*eM_2_odd[r][0]*ci_sum;
        cols[1] = 5*idx[1]+2;   vals[1] =  dt*eM_2_odd[r][1]*ci_sum;
        cols[2] = 5*idx[2]+2;   vals[2] =  dt*eM_2_odd[r][2]*ci_sum;
        cols[3] = 5*idx[0]+3;   vals[3] =  eM_0[r][0];
        cols[4] = 5*idx[1]+3;   vals[4] =  eM_0[r][1];
        cols[5] = 5*idx[2]+3;   vals[5] =  eM_0[r][2];
       
        ierr = MatSetValuesLocal(M,1,&row,6,cols,vals,ADD_VALUES);CHKERRQ(ierr); 
     
        
        row = 5*idx[r]+3;
        cols[0] = 5*idx[0]+2;   vals[0] = -1.0*eM_0[r][0];
        cols[1] = 5*idx[1]+2;   vals[1] = -1.0*eM_0[r][1];
        cols[2] = 5*idx[2]+2;   vals[2] = -1.0*eM_0[r][2];
        cols[3] = 5*idx[0]+3;   vals[3] =  user->kai*eM_2_odd[r][0];
        cols[4] = 5*idx[1]+3;   vals[4] =  user->kai*eM_2_odd[r][1];
        cols[5] = 5*idx[2]+3;   vals[5] =  user->kai*eM_2_odd[r][2];
       
        ierr = MatSetValuesLocal(M,1,&row,6,cols,vals,ADD_VALUES);CHKERRQ(ierr); 
        
        
        row = 5*idx[r]+4;
        /*
        cols3[0] = 5*idx[0]+4;   vals3[0] = eM_0[r][0]/dt + user->L*user->kaeta*dt*eM_2_odd[r][0];
        cols3[1] = 5*idx[1]+4;   vals3[1] = eM_0[r][1]/dt + user->L*user->kaeta*dt*eM_2_odd[r][1];
        cols3[2] = 5*idx[2]+4;   vals3[2] = eM_0[r][2]/dt + user->L*user->kaeta*dt*eM_2_odd[r][2];
         */
        cols3[0] = 5*idx[0]+4;   vals3[0] = eM_0[r][0]/dt + user->L*user->kaeta*eM_2_odd[r][0];
        cols3[1] = 5*idx[1]+4;   vals3[1] = eM_0[r][1]/dt + user->L*user->kaeta*eM_2_odd[r][1];
        cols3[2] = 5*idx[2]+4;   vals3[2] = eM_0[r][2]/dt + user->L*user->kaeta*eM_2_odd[r][2];
        
        ierr = MatSetValuesLocal(M,1,&row,3,cols3,vals3,ADD_VALUES);CHKERRQ(ierr);

    }
  }



  ierr = VecRestoreArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->ci,&ci_p);CHKERRQ(ierr);
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
  PetscInt          i,n,Mda,Nda,nele,nen;
  const PetscInt    *ele;
  
  PetscInt          idx[3];
  PetscScalar       eM_2_odd[3][3],eM_2_even[3][3],h,dt=user->dt;
  Mat               M=user->M;
  PetscScalar       *cv_p,*ci_p,cv_sum,ci_sum;

  PetscFunctionBegin;
 
  /* Create the mass matrix M_0 */
  ierr = MatGetLocalSize(M,&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecGetArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->ci,&ci_p);CHKERRQ(ierr);
  ierr = DMDAGetInfo(user->da1,PETSC_NULL,&Mda,&Nda,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

 
 
  h = (user->xmax-user->xmin)/Mda;

  ierr = DMDAGetElements(user->da1,&nele,&nen,&ele);CHKERRQ(ierr);

  for(i=0; i < nele; i++) {
    /*
    idx[0] = connect[k*3];
    idx[1] = connect[k*3+1];
    idx[2] = connect[k*3+2];
     */
    idx[0] = ele[3*i];
    idx[1] = ele[3*i+1];
    idx[2] = ele[3*i+2];

    PetscInt r,row,cols[3];
    PetscScalar vals[3];
    for (r=0;r<3;r++) {
      row = 5*idx[r];
      cols[0] = 5*idx[0];     vals[0] = 0.0;
      cols[1] = 5*idx[1];     vals[1] = 0.0; 
      cols[2] = 5*idx[2];     vals[2] = 0.0;

      /* Insert values in matrix M for 1st dof */
      ierr = MatSetValuesLocal(M,1,&row,3,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
        
      row = 5*idx[r]+2;
      cols[0] = 5*idx[0]+2;   vals[0] = 0.0;
      cols[1] = 5*idx[1]+2;   vals[1] = 0.0;
      cols[2] = 5*idx[2]+2;   vals[2] = 0.0;

      /* Insert values in matrix M for 3nd dof */
      ierr = MatSetValuesLocal(M,1,&row,3,cols,vals,INSERT_VALUES);CHKERRQ(ierr); 
    }
  }
  
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  eM_2_odd[0][0] = 1.0;
  eM_2_odd[1][1] = eM_2_odd[2][2] = 0.5;
  eM_2_odd[0][1] = eM_2_odd[0][2] = eM_2_odd[1][0]= eM_2_odd[2][0] = -0.5;
  eM_2_odd[1][2] = eM_2_odd[2][1] = 0.0;

  eM_2_even[0][0] = 1.0;
  eM_2_even[1][1] = eM_2_even[2][2] = 0.5;
  eM_2_even[0][1] = eM_2_even[0][2] = eM_2_even[1][0]= eM_2_even[2][0] = -0.5;
  eM_2_even[1][2] = eM_2_even[2][1] = 0.0;

  /*
  eM_2_even[1][1] = 1.0;
  eM_2_even[0][0] = eM_2_even[2][2] = 0.5;
  eM_2_even[0][1] = eM_2_even[1][0] = eM_2_even[1][2] = eM_2_even[2][1] = -0.5;
  eM_2_even[0][2] = eM_2_even[2][0] = 0.0;
   */
    
  /* Get local element info */
  //for(k=0;k < Mda*Nda*2;k++) {
  for (i=0; i < nele; i++) {
    /*
      idx[0] = connect[k*3];
      idx[1] = connect[k*3+1];
      idx[2] = connect[k*3+2];
     */
    idx[0] = ele[3*i];
    idx[1] = ele[3*i+1];
    idx[2] = ele[3*i+2];

      PetscInt    row,cols[3],r;     
      PetscScalar vals[3];
    
      for(r=0;r<3;r++) {
                      
        // cv_sum = (1.0e-3+cv_p[idx[0]] + cv_p[idx[1]] + cv_p[idx[2]])*user->Dv/(3.0*user->kBT);
        //ci_sum = (1.0e-3+ci_p[idx[0]] + ci_p[idx[1]] + ci_p[idx[2]])*user->Di/(3.0*user->kBT);
        cv_sum = .0000069*user->Dv/(user->kBT);
        ci_sum = .0000069*user->Di/user->kBT;

         
                
                row = 5*idx[r];
                cols[0] = 5*idx[0];     vals[0] = dt*eM_2_odd[r][0]*cv_sum;
                cols[1] = 5*idx[1];     vals[1] = dt*eM_2_odd[r][1]*cv_sum; 
                cols[2] = 5*idx[2];     vals[2] = dt*eM_2_odd[r][2]*cv_sum;
                          
                /* Insert values in matrix M for 1st dof */
                ierr = MatSetValuesLocal(M,1,&row,3,cols,vals,ADD_VALUES);CHKERRQ(ierr);
         
                
                row = 5*idx[r]+2;
                cols[0] = 5*idx[0]+2;   vals[0] = dt*eM_2_odd[r][0]*ci_sum;
                cols[1] = 5*idx[1]+2;   vals[1] = dt*eM_2_odd[r][1]*ci_sum;
                cols[2] = 5*idx[2]+2;   vals[2] = dt*eM_2_odd[r][2]*ci_sum;
          
                ierr = MatSetValuesLocal(M,1,&row,3,cols,vals,ADD_VALUES);CHKERRQ(ierr); 

            
           
        }
   
    }


  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->ci,&ci_p);CHKERRQ(ierr);


  PetscFunctionReturn(0);
}


