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
 ./ex61 -ksp_type gmres -snes_vi_monitor   -snes_atol 1.e-11  -da_refine 3  -T 0.1   -ksp_monitor_true_residual -pc_type lu -pc_factor_mat_solver_package superlu -snes_converged_reason -ksp_converged_reason  -ksp_rtol 1.e-9  -snes_linesearch_monitor -VG 10 -draw_fields 1,3,4 -snes_linesearch_type basic

./ex61 -ksp_type gmres -snes_vi_monitor   -snes_atol 1.e-11  -da_refine 4 -T 0.1   -ksp_monitor_true_residual -pc_type sor -snes_converged_reason -ksp_converged_reason  -ksp_rtol 1.e-9  -snes_linesearch_monitor -VG 10 -draw_fields 1,3,4 -snes_linesearch_type basic

./ex61 -ksp_type fgmres -snes_vi_monitor   -snes_atol 1.e-11  -da_refine 5 -T 0.1   -ksp_monitor_true_residual -snes_converged_reason -ksp_converged_reason  -ksp_rtol 1.e-9  -snes_linesearch_monitor -VG 10 -draw_fields 1,3,4 -snes_linesearch_type basic -pc_type mg -pc_mg_galerkin

./ex61 -ksp_type fgmres -snes_vi_monitor   -snes_atol 1.e-11  -da_refine 5 -snes_converged_reason -ksp_converged_reason   -snes_linesearch_monitor -VG 1 -draw_fields 1,3,4  -pc_type mg -pc_mg_galerkin -log_summary -dt .0000000000001 -mg_coarse_pc_type svd  -ksp_monitor_true_residual -ksp_rtol 1.e-9

Movie version
./ex61 -ksp_type fgmres -snes_vi_monitor   -snes_atol 1.e-11  -da_refine 6 -snes_converged_reason -ksp_converged_reason   -snes_linesearch_monitor -VG 10 -draw_fields 1,3,4  -pc_type mg -pc_mg_galerkin -log_summary -dt .000001 -mg_coarse_pc_type redundant -mg_coarse_redundant_pc_type svd  -ksp_monitor_true_residual -ksp_rtol 1.e-9 -snes_linesearch_type basic -T .0020 

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
  PetscReal   dtevent;  /* time scale of radiation events, roughly one event per dtevent */
  PetscInt    maxevents; /* once this number of events is reached no more events are generated */
  PetscReal   initv;    /* initial value of phase variables */
  PetscReal   initeta; 
  PetscBool   degenerate;  /* use degenerate mobility */
  PetscReal   smallnumber;
  PetscBool   graphics;
  PetscInt    domain;
  PetscBool   radiation;
  PetscBool   voidgrowth; /* use initial conditions for void growth */
  DM          da1,da2;
  Mat         M;    /* Jacobian matrix */
  Mat         M_0;
  Vec         q,wv,cv,wi,ci,eta,cvi,DPsiv,DPsii,DPsieta,Pv,Pi,Piv,logcv,logci,logcvi,Riv;
  Vec         phi1,phi2,Phi2D_V,Sv,Si; /* for twodomain modeling */
  Vec         work1,work2,work3,work4;
  PetscScalar Dv,Di,Evf,Eif,A,kBT,kav,kai,kaeta,Rsurf,Rbulk,L,VG; /* physics parameters */
  PetscScalar Svr,Sir,cv_eq,ci_eq; /* for twodomain modeling */
  PetscReal   asmallnumber; /* gets added to degenerate mobility */
  PetscReal   xmin,xmax,ymin,ymax;
  PetscInt    Mda, Nda;
  PetscViewer graphicsfile;  /* output of solution at each times step */
}AppCtx;

PetscErrorCode GetParams(AppCtx*);
PetscErrorCode SetRandomVectors(AppCtx*,PetscReal);
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
PetscErrorCode Phi(AppCtx*);
PetscErrorCode Phi_read(AppCtx*);

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
  PetscScalar         t=0.0,normq;
  /*  PetscViewer         view_out, view_q, view_psi, view_mat;*/
  /*  PetscViewer         view_rand;*/
  IS                  inactiveconstraints;
  PetscInt            ninactiveconstraints,N;
  SNESConvergedReason reason;
  /*PetscViewer         view_out, view_cv,view_eta,view_vtk_cv,view_vtk_eta;*/
  char                cv_filename[80],eta_filename[80];
  /*PetscReal           bounds[] = {1000.0,-1000.,0.0,1.0,1000.0,-1000.0,0.0,1.0,1000.0,-1000.0}; */

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
  ierr = VecDuplicate(user.wv,&user.Riv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.phi1);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.phi2);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.Phi2D_V);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.Sv);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.Si);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.work3);CHKERRQ(ierr);
  ierr = VecDuplicate(user.wv,&user.work4);CHKERRQ(ierr);
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
  ierr = SetInitialGuess(x,&user);CHKERRQ(ierr);
  /* twodomain modeling */
  if (user.domain) {
    switch (user.domain) {
    case 1:
      ierr = Phi(&user);CHKERRQ(ierr);
      break;
    case 2:
      ierr = Phi_read(&user);CHKERRQ(ierr);
      break ;
    }
  }

  /* Form the jacobian matrix and M_0 */
  ierr = SetUpMatrices(&user);CHKERRQ(ierr);
  ierr = MatDuplicate(user.M,MAT_DO_NOT_COPY_VALUES,&J);CHKERRQ(ierr);
  
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,user.da1);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,r,FormFunction,(void*)&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,(void*)&user);CHKERRQ(ierr);
 

  ierr = SetVariableBounds(user.da1,xl,xu);CHKERRQ(ierr);
  ierr = SNESVISetVariableBounds(snes,xl,xu);CHKERRQ(ierr);
  ierr = SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,100,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  
  /*  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_rand",FILE_MODE_WRITE,&view_rand);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_mat2",FILE_MODE_WRITE,&view_mat);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_q",FILE_MODE_WRITE,&view_q);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_out",FILE_MODE_WRITE,&view_out);CHKERRQ(ierr);
   ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_psi",FILE_MODE_WRITE,&view_psi);CHKERRQ(ierr);*/
 
  /*ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_out",FILE_MODE_WRITE,&view_out);CHKERRQ(ierr);
  
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_cv",FILE_MODE_WRITE,&view_cv);CHKERRQ(ierr);
   ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_eta",FILE_MODE_WRITE,&view_eta);CHKERRQ(ierr);*/
  
  /* ierr = PetscViewerDrawSetBounds(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),5,bounds);CHKERRQ(ierr); */
  if (user.graphics) {
    ierr = VecView(x,PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD));CHKERRQ(ierr);  
  }
  /*
  if (user.graphicsfile) {
    ierr = DMView(user.da1,user.graphicsfile);CHKERRQ(ierr);
    ierr = VecView(x,user.graphicsfile);CHKERRQ(ierr);  
  }
   */
  while (t<user.T) {
    ierr = SNESSetFunction(snes,r,FormFunction,(void*)&user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,FormJacobian,(void*)&user);CHKERRQ(ierr);

    ierr = SetRandomVectors(&user,t);CHKERRQ(ierr);
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
    /*    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,cv_filename,&view_vtk_cv);CHKERRQ(ierr);
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
     ierr = PetscViewerDestroy(&view_vtk_eta);CHKERRQ(ierr);*/

        
    ierr = VecNorm(user.q,NORM_2,&normq);CHKERRQ(ierr);
    printf("2-norm of q = %14.12f\n",normq);
    ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes,&reason);CHKERRQ(ierr);
    if (reason < 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Nonlinear solver failed");
    ierr = SNESVIGetInactiveSet(snes,&inactiveconstraints);CHKERRQ(ierr);
    ierr = ISGetSize(inactiveconstraints,&ninactiveconstraints);CHKERRQ(ierr);
    /* if (ninactiveconstraints < .90*N) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP,"To many active constraints, model has become non-physical"); */

    /*    ierr = VecView(x,view_out);CHKERRQ(ierr);*/
    if (user.graphics) {
      ierr = VecView(x,PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD));CHKERRQ(ierr);
    }
    /*    ierr = VecView(x,PETSC_VIEWER_BINARY_(PETSC_COMM_WORLD));CHKERRQ(ierr);*/
    PetscInt its;
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"SNESVI solver converged at t = %g in %d iterations\n",t,its);CHKERRQ(ierr);

    ierr = Update_u(x,&user);CHKERRQ(ierr);
    ierr = UpdateMatrices(&user);CHKERRQ(ierr);
    t = t + user.dt;
    /*
    if (user.graphicsfile) {
      ierr = VecView(x,user.graphicsfile);CHKERRQ(ierr);  
    }
     */
  }
   
  /*  ierr = PetscViewerDestroy(&view_rand);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_mat);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_q);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_out);CHKERRQ(ierr);
   ierr = PetscViewerDestroy(&view_psi);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_out);CHKERRQ(ierr);
  
  ierr = PetscViewerDestroy(&view_cv);CHKERRQ(ierr);
   ierr = PetscViewerDestroy(&view_eta);CHKERRQ(ierr);*/
  
  if (user.graphicsfile) {
    ierr = PetscViewerDestroy(&user.graphicsfile);CHKERRQ(ierr);  
  }
  
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
  PetscScalar    *q_p,*w1,*w2,max1;
  PetscInt       i,n;

 
  PetscFunctionBegin;
  
  ierr = VecPointwiseMult(user->Riv,user->eta,user->eta);CHKERRQ(ierr);
  ierr = VecScale(user->Riv,user->Rsurf);CHKERRQ(ierr);
  ierr = VecShift(user->Riv,user->Rbulk);CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->Riv,user->ci,user->Riv);CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->Riv,user->cv,user->Riv);CHKERRQ(ierr);
  
  ierr = VecCopy(user->Riv,user->work1);CHKERRQ(ierr);
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
    ierr = VecAXPY(user->work1,300.0,user->work4);CHKERRQ(ierr);
  }
  ierr = VecScale(user->work1,user->dt);CHKERRQ(ierr);
  ierr = VecAXPY(user->work1,-1.0,user->cv);CHKERRQ(ierr);
  ierr = MatMult(user->M_0,user->work1,user->work2);CHKERRQ(ierr);
 
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
  if (user->radiation) {
    ierr = VecAXPY(user->work1,-1.0,user->Pi);CHKERRQ(ierr);
  }
  if (user->domain) {
    ierr = VecCopy(user->ci,user->work3);CHKERRQ(ierr);
    ierr = VecShift(user->work3,-1.0*user->ci_eq);CHKERRQ(ierr);
    ierr = VecCopy(user->Phi2D_V,user->work4);CHKERRQ(ierr);
    ierr = VecScale(user->work4,-1.0);CHKERRQ(ierr);
    ierr = VecShift(user->work4,1.0);CHKERRQ(ierr);
    ierr = VecPointwiseMult(user->work4,user->work4,user->work3);CHKERRQ(ierr);
    ierr = VecScale(user->work4,user->Sir);CHKERRQ(ierr);
    ierr = VecAXPY(user->work1,300.0,user->work4);CHKERRQ(ierr);
  }
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
  if (user->radiation) {
    ierr = VecAXPY(user->work1,-1.0,user->Piv);CHKERRQ(ierr);
  }
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
  PetscInt          n,i,Mda,Nda;
  PetscScalar	   *xx,*cv_p,*ci_p,*wv_p,*wi_p,*eta;    

  /* needed for the void growth case */
  PetscScalar       xmid,ymid,cv_v=1.0,cv_m=0.122,ci_v=0.0,ci_m=.00069,eta_v=1.0,eta_m=0.0,h,lambda;
  PetscInt          nele,nen,idx[3];
  const PetscInt    *ele;
  PetscScalar       x[3],y[3];
  Vec               coords;
  const PetscScalar *_coords;
  PetscViewer       view; 
  PetscScalar       xwidth = user->xmax - user->xmin;

  PetscFunctionBegin;

  ierr = VecGetLocalSize(X,&n);CHKERRQ(ierr);

  if (user->voidgrowth) {
    ierr = DMDAGetInfo(user->da2,PETSC_NULL,&Mda,&Nda,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = DMDAGetGhostedCoordinates(user->da2,&coords);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coords,&_coords);CHKERRQ(ierr);

    h = (user->xmax-user->xmin)/Mda;
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
      PetscScalar vals_cv[3],vals_ci[3],vals_eta[3],s,hhr,r;
      for (k=0; k < 3 ; k++) {
        s = sqrt((x[k] - xmid)*(x[k] - xmid) + (y[k] - ymid)*(y[k] - ymid));
        if (s < xwidth*(5.0/64.0)) {
          vals_cv[k] = cv_v;
          vals_ci[k] = ci_v;
          vals_eta[k] = eta_v;
        } else if (s>= xwidth*(5.0/64.0) && s<= xwidth*(7.0/64.0) ) {
          //r = (s - xwidth*(6.0/64.0) )/(0.5*lambda);
          r = (s - xwidth*(6.0/64.0) )/(xwidth/64.0);
          hhr = 0.25*(-r*r*r + 3*r + 2);
          vals_cv[k] = cv_m + (1.0 - hhr)*(cv_v - cv_m);
          vals_ci[k] = ci_m + (1.0 - hhr)*(ci_v - ci_m);
          vals_eta[k] = eta_m + (1.0 - hhr)*(eta_v - eta_m);
        } else {
          vals_cv[k] = cv_m;
          vals_ci[k] = ci_m;
          vals_eta[k] = eta_m;
        }
      }
      ierr = VecSetValuesLocal(user->cv,3,idx,vals_cv,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValuesLocal(user->ci,3,idx,vals_ci,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValuesLocal(user->eta,3,idx,vals_eta,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(user->cv);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->cv);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(user->ci);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->ci);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(user->eta);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->eta);CHKERRQ(ierr);

    ierr = DMDARestoreElements(user->da2,&nele,&nen,&ele);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(coords,&_coords);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_initial",FILE_MODE_WRITE,&view);CHKERRQ(ierr);
    ierr = VecView(user->cv,view);CHKERRQ(ierr);
    ierr = VecView(user->ci,view);CHKERRQ(ierr);
    ierr = VecView(user->eta,view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  }
  else {
    //ierr = VecSet(user->cv,user->initv);CHKERRQ(ierr);
    //ierr = VecSet(user->ci,user->initv);CHKERRQ(ierr);
    ierr = VecSet(user->cv,.05);CHKERRQ(ierr);
    ierr = VecSet(user->ci,.05);CHKERRQ(ierr);
    ierr = VecSet(user->eta,user->initeta);CHKERRQ(ierr);
  }

  ierr = DPsi(user);CHKERRQ(ierr);
  ierr = VecCopy(user->DPsiv,user->wv);CHKERRQ(ierr);
  ierr = VecCopy(user->DPsii,user->wi);CHKERRQ(ierr);

  ierr = VecGetArray(X,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->ci,&ci_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->wv,&wv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->wi,&wi_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->eta,&eta);CHKERRQ(ierr);
  for (i=0;i<n/5;i++)
  {
    xx[5*i]=wv_p[i];
    xx[5*i+1]=cv_p[i];
    xx[5*i+2]=wi_p[i];
    xx[5*i+3]=ci_p[i];
    xx[5*i+4]=eta[i];
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
  ierr = VecRestoreArray(user->eta,&eta);CHKERRQ(ierr);
  
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
      ierr = VecSetValueLocal(user->Pv,i + 1 + xm*(j + 1), randomvalues[randindex].strength*user->VG,INSERT_VALUES);CHKERRQ(ierr);
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
  PetscBool      flg,graphicsfile = PETSC_FALSE;
  
  PetscFunctionBegin;
  
  /* Set default parameters */
  user->xmin = 0.0; user->xmax = 128.0;
  user->ymin = 0.0; user->ymax = 128.0;
  user->Dv    = 1.0; 
  user->Di    = 4.0;
  user->Evf   = 0.8; 
  user->Eif   = 1.2;
  user->A     = 1.0;
  user->kBT   = 0.11;
  user->kav   = 1.0; 
  user->kai   = 1.0; 
  user->kaeta = 1.0;
  user->Rsurf = 10.0; 
  user->Rbulk = 1.0;
  user->VG    = 100.0;
  user->L     = 10.0; 

  user->T          = 1.0e-2;   
  user->dt         = 1.0e-4;
  user->initv      = .00069; 
  user->initeta    = 0.0;
  user->degenerate = PETSC_FALSE;
  user->maxevents  = 10;
  user->graphics   = PETSC_TRUE;

  /* multidomain modeling */
  user->domain    =   1;
  user->Svr       = 0.5; 
  user->Sir       = 0.5;
  user->cv_eq     = 6.9e-4;
  user->ci_eq     = 6.9e-4;
  /* void growth */
  user->voidgrowth = PETSC_FALSE;

  user->radiation = PETSC_FALSE;

  /* degenerate mobility */
  user->smallnumber = 1.0e-3;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Coupled Cahn-Hillard/Allen-Cahn Equations","Phasefield");CHKERRQ(ierr);
    
  ierr = PetscOptionsInt("-domain","Number of domains (0=one domain, 1=two domains, 2=multidomain\n","None",user->domain,&user->domain,&flg);CHKERRQ(ierr);

    ierr = PetscOptionsReal("-Dv","Vacancy Diffusivity\n","None",user->Dv,&user->Dv,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-Di","Interstitial Diffusivity\n","None",user->Di,&user->Di,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-Evf","Vacancy Formation Energy\n","None",user->Evf,&user->Evf,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-Eif","Interstitial Formation energy\n","None",user->Eif,&user->Eif,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-A","???","None",user->A,&user->A,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-kBT","Boltzmann's Constant times the Absolute Temperature","None",user->kBT,&user->kBT,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-kav","???","None",user->kav,&user->kav,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-kai","???","None",user->kai,&user->kai,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-kaeta","???","None",user->kaeta,&user->kaeta,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-Rsurf","???","None",user->Rsurf,&user->Rsurf,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-Rbulk","???","None",user->Rbulk,&user->Rbulk,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-VG","Maximum increase in vacancy (or interstitial) concentration due to a cascade event","None",user->VG,&user->VG,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-L","???","None",user->L,&user->L,&flg);CHKERRQ(ierr);

    ierr = PetscOptionsReal("-initv","Initial solution of Cv and Ci","None",user->initv,&user->initv,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-initeta","Initial solution of Eta","None",user->initeta,&user->initeta,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-degenerate","Run with degenerate mobility\n","None",user->degenerate,&user->degenerate,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-smallnumber","Small number added to degenerate mobility\n","None",user->smallnumber,&user->smallnumber,&flg);CHKERRQ(ierr);

    ierr = PetscOptionsBool("-voidgrowth","Use initial conditions for void growth\n","None",user->voidgrowth,&user->voidgrowth,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-radiation","Use initial conditions for void growth\n","None",user->radiation,&user->radiation,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-xmin","Lower X coordinate of domain\n","None",user->xmin,&user->xmin,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-xmax","Upper X coordinate of domain\n","None",user->xmax,&user->xmax,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-T","Total runtime\n","None",user->T,&user->T,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-dt","Time step\n","None",user->dt,&user->dt,&flg);CHKERRQ(ierr);
    user->dtevent = user->dt;
    ierr = PetscOptionsReal("-dtevent","Average time between random events\n","None",user->dtevent,&user->dtevent,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-maxevents","Maximum random events allowed\n","None",user->maxevents,&user->maxevents,&flg);CHKERRQ(ierr);

    ierr = PetscOptionsBool("-graphics","Contour plot solutions at each timestep\n","None",user->graphics,&user->graphics,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-graphicsfile","Save solution at each timestep\n","None",graphicsfile,&graphicsfile,&flg);CHKERRQ(ierr);
    if (graphicsfile) {
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"ex61.data",FILE_MODE_WRITE,&user->graphicsfile);CHKERRQ(ierr);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);   
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
  /* newly added */
  Vec               cvlocal,cilocal;

  PetscFunctionBegin;
 
  /*  ierr = MatSetOption(M,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
   ierr = MatSetOption(M_0,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);*/

  /* new stuff */
  ierr = DMGetLocalVector(user->da2,&cvlocal);CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->da2,&cilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(user->da2,user->cv,INSERT_VALUES,cvlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->da2,user->cv,INSERT_VALUES,cvlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(user->da2,user->ci,INSERT_VALUES,cilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->da2,user->ci,INSERT_VALUES,cilocal);CHKERRQ(ierr);
  /* old stuff */
  /*
  ierr = VecGetArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->ci,&ci_p);CHKERRQ(ierr);
   */
  /* new stuff */
  ierr = VecGetArray(cvlocal,&cv_p);CHKERRQ(ierr);
  ierr = VecGetArray(cilocal,&ci_p);CHKERRQ(ierr);

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
       
      if (user->degenerate) {
        cv_sum = (cv_p[idx[0]] + cv_p[idx[1]] + cv_p[idx[2]])*user->Dv/(3.0*user->kBT);
        ci_sum = (ci_p[idx[0]] + ci_p[idx[1]] + ci_p[idx[2]])*user->Di/(3.0*user->kBT);
      } else {
        cv_sum = user->initv*user->Dv/user->kBT;
        ci_sum = user->initv*user->Di/user->kBT;
      }
      

        
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
        cols3[0] = 5*idx[0]+4;   vals3[0] = (eM_0[r][0]/dt + user->L*user->kaeta*eM_2_odd[r][0]);
        cols3[1] = 5*idx[1]+4;   vals3[1] = (eM_0[r][1]/dt + user->L*user->kaeta*eM_2_odd[r][1]);
        cols3[2] = 5*idx[2]+4;   vals3[2] = (eM_0[r][2]/dt + user->L*user->kaeta*eM_2_odd[r][2]);
        
        ierr = MatSetValuesLocal(M,1,&row,3,cols3,vals3,ADD_VALUES);CHKERRQ(ierr);

    }
  }

  /* new */
  ierr = VecRestoreArray(cvlocal,&cv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(cilocal,&ci_p);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->da2,&cvlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->da2,&cilocal);CHKERRQ(ierr);
  /* old */
  /*
  ierr = VecRestoreArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->ci,&ci_p);CHKERRQ(ierr);
   */
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
  /* newly added */
  Vec               cvlocal,cilocal;

  PetscFunctionBegin;
 
  
  ierr = MatGetLocalSize(M,&n,PETSC_NULL);CHKERRQ(ierr);
  
  /* new stuff */
  ierr = DMGetLocalVector(user->da2,&cvlocal);CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->da2,&cilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(user->da2,user->cv,INSERT_VALUES,cvlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->da2,user->cv,INSERT_VALUES,cvlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(user->da2,user->ci,INSERT_VALUES,cilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->da2,user->ci,INSERT_VALUES,cilocal);CHKERRQ(ierr);
  /* new stuff */
  ierr = VecGetArray(cvlocal,&cv_p);CHKERRQ(ierr);
  ierr = VecGetArray(cilocal,&ci_p);CHKERRQ(ierr);
  
  /* old stuff */
  /*
  ierr = VecGetArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecGetArray(user->ci,&ci_p);CHKERRQ(ierr);
   */
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
                 
      if (user->degenerate) {     
        printf("smallnumber = %14.12f\n",user->smallnumber);
        cv_sum = (user->smallnumber + cv_p[idx[0]] + cv_p[idx[1]] + cv_p[idx[2]])*user->Dv/(3.0*user->kBT);
        ci_sum = (user->smallnumber + ci_p[idx[0]] + ci_p[idx[1]] + ci_p[idx[2]])*user->Di/(3.0*user->kBT);
      } else {
        cv_sum = user->initv*user->Dv/(user->kBT);
        ci_sum = user->initv*user->Di/user->kBT;
      }

         
                
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

  /* new stuff */
  ierr = VecRestoreArray(cvlocal,&cv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(cilocal,&ci_p);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->da2,&cvlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->da2,&cilocal);CHKERRQ(ierr);
  /* old stuff */
  /*
  ierr = VecRestoreArray(user->cv,&cv_p);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->ci,&ci_p);CHKERRQ(ierr);
   */

  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


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
  PetscViewer        view;

  PetscFunctionBegin;

  ierr = DMDAGetInfo(user->da1,PETSC_NULL,&Mda,&Nda,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostedCoordinates(user->da2,&coords);CHKERRQ(ierr);
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
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"file_phi",FILE_MODE_WRITE,&view);CHKERRQ(ierr);

  ierr = VecView(user->phi1,view);CHKERRQ(ierr);
  ierr = VecView(user->phi2,view);CHKERRQ(ierr);

  
  //ierr = VecView(user->phi1,0);CHKERRQ(ierr);
  //ierr = VecView(user->phi2,0);CHKERRQ(ierr);
  
  ierr = VecPointwiseMult(user->phi1,user->phi1,user->phi1);CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->phi2,user->phi2,user->phi2);CHKERRQ(ierr);
  ierr = VecView(user->phi1,view);CHKERRQ(ierr);
  ierr = VecView(user->phi2,view);CHKERRQ(ierr);

  ierr = VecCopy(user->phi1,user->Phi2D_V);CHKERRQ(ierr);
  ierr = VecAXPY(user->Phi2D_V,1.0,user->phi2);CHKERRQ(ierr);
  //ierr = VecView(user->Phi2D_V,0);CHKERRQ(ierr);

  ierr = VecView(user->Phi2D_V,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  //  ierr = VecNorm(user->Phi2D_V,NORM_INFINITY,&max1);CHKERRQ(ierr);
  //ierr = VecMin(user->Phi2D_V,&loc1,&min1);CHKERRQ(ierr);
  //printf("norm phi = %f, min phi = %f\n",max1,min1);

  PetscFunctionReturn(0);
  
}

#undef __FUNCT__
#define __FUNCT__ "Phi_read"
PetscErrorCode Phi_read(AppCtx* user)
{
  PetscErrorCode     ierr;
  PetscReal          *values;
  PetscViewer        viewer;

  PetscFunctionBegin;
  
  ierr = VecGetArray(user->Phi2D_V,&values);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi3",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,values,16384,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
