/*$Id: itfunc.c,v 1.159 2001/08/07 03:03:45 balay Exp $*/
/*
      Interface KSP routines that the user calls.
*/

#include "src/ksp/ksp/kspimpl.h"   /*I "petscksp.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "KSPComputeExtremeSingularValues"
/*@
   KSPComputeExtremeSingularValues - Computes the extreme singular values
   for the preconditioned operator. Called after or during KSPSolve()
   (KSPSolve()).

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameters:
.  emin, emax - extreme singular values

   Notes:
   One must call KSPSetComputeSingularValues() before calling KSPSetUp() 
   (or use the option -ksp_compute_eigenvalues) in order for this routine to work correctly.

   Many users may just want to use the monitoring routine
   KSPSingularValueMonitor() (which can be set with option -ksp_singmonitor)
   to print the extreme singular values at each iteration of the linear solve.

   Level: advanced

.keywords: KSP, compute, extreme, singular, values

.seealso: KSPSetComputeSingularValues(), KSPSingularValueMonitor(), KSPComputeEigenvalues()
@*/
int KSPComputeExtremeSingularValues(KSP ksp,PetscReal *emax,PetscReal *emin)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidScalarPointer(emax);
  PetscValidScalarPointer(emin);
  if (!ksp->calc_sings) {
    SETERRQ(4,"Singular values not requested before KSPSetUp()");
  }

  if (ksp->ops->computeextremesingularvalues) {
    ierr = (*ksp->ops->computeextremesingularvalues)(ksp,emax,emin);CHKERRQ(ierr);
  } else {
    *emin = -1.0;
    *emax = -1.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPComputeEigenvalues"
/*@
   KSPComputeEigenvalues - Computes the extreme eigenvalues for the
   preconditioned operator. Called after or during KSPSolve() (KSPSolve()).

   Not Collective

   Input Parameter:
+  ksp - iterative context obtained from KSPCreate()
-  n - size of arrays r and c. The number of eigenvalues computed (neig) will, in 
       general, be less than this.

   Output Parameters:
+  r - real part of computed eigenvalues
.  c - complex part of computed eigenvalues
-  neig - number of eigenvalues computed (will be less than or equal to n)

   Options Database Keys:
+  -ksp_compute_eigenvalues - Prints eigenvalues to stdout
-  -ksp_plot_eigenvalues - Plots eigenvalues in an x-window display

   Notes:
   The number of eigenvalues estimated depends on the size of the Krylov space
   generated during the KSPSolve() (that is the KSPSolve); for example, with 
   CG it corresponds to the number of CG iterations, for GMRES it is the number 
   of GMRES iterations SINCE the last restart. Any extra space in r[] and c[]
   will be ignored.

   KSPComputeEigenvalues() does not usually provide accurate estimates; it is
   intended only for assistance in understanding the convergence of iterative 
   methods, not for eigenanalysis. 

   One must call KSPSetComputeEigenvalues() before calling KSPSetUp() 
   in order for this routine to work correctly.

   Many users may just want to use the monitoring routine
   KSPSingularValueMonitor() (which can be set with option -ksp_singmonitor)
   to print the singular values at each iteration of the linear solve.

   Level: advanced

.keywords: KSP, compute, extreme, singular, values

.seealso: KSPSetComputeSingularValues(), KSPSingularValueMonitor(), KSPComputeExtremeSingularValues()
@*/
int KSPComputeEigenvalues(KSP ksp,int n,PetscReal *r,PetscReal *c,int *neig)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidScalarPointer(r);
  PetscValidScalarPointer(c);
  if (!ksp->calc_sings) {
    SETERRQ(4,"Eigenvalues not requested before KSPSetUp()");
  }

  if (ksp->ops->computeeigenvalues) {
    ierr = (*ksp->ops->computeeigenvalues)(ksp,n,r,c,neig);CHKERRQ(ierr);
  } else {
    *neig = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUpOnBlocks"
/*@
   KSPSetUpOnBlocks - Sets up the preconditioner for each block in
   the block Jacobi, block Gauss-Seidel, and overlapping Schwarz 
   methods.

   Collective on KSP

   Input Parameter:
.  ksp - the KSP context

   Notes:
   KSPSetUpOnBlocks() is a routine that the user can optinally call for
   more precise profiling (via -log_summary) of the setup phase for these
   block preconditioners.  If the user does not call KSPSetUpOnBlocks(),
   it will automatically be called from within KSPSolve().
   
   Calling KSPSetUpOnBlocks() is the same as calling PCSetUpOnBlocks()
   on the PC context within the KSP context.

   Level: advanced

.keywords: KSP, setup, blocks

.seealso: PCSetUpOnBlocks(), KSPSetUp(), PCSetUp()
@*/
int KSPSetUpOnBlocks(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = PCSetUpOnBlocks(ksp->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp"
/*@
   KSPSetUp - Sets up the internal data structures for the
   later use of an iterative solver.

   Collective on KSP

   Input Parameter:
.  ksp   - iterative context obtained from KSPCreate()

   Level: developer

.keywords: KSP, setup

.seealso: KSPCreate(), KSPSolve(), KSPDestroy()
@*/
int KSPSetUp(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);

  /* reset the convergence flag from the previous solves */
  ksp->reason = KSP_CONVERGED_ITERATING;

  if (!ksp->type_name){
    ierr = KSPSetType(ksp,KSPGMRES);CHKERRQ(ierr);
  }

  if (ksp->setupcalled) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(KSP_SetUp,ksp,ksp->vec_rhs,ksp->vec_sol,0);CHKERRQ(ierr);
  ksp->setupcalled = 1;
  ierr = PCSetVector(ksp->B,ksp->vec_rhs);CHKERRQ(ierr);

  ierr = (*ksp->ops->setup)(ksp);CHKERRQ(ierr);

  /* scale the matrix if requested */
  if (ksp->dscale) {
    Mat mat,pmat;
    ierr = PCGetOperators(ksp->B,&mat,&pmat,PETSC_NULL);CHKERRQ(ierr);
    if (mat == pmat) {
      PetscScalar  *xx;
      int          i,n;
      PetscTruth   zeroflag = PETSC_FALSE;

      if (!ksp->diagonal) { /* allocate vector to hold diagonal */
	ierr = VecDuplicate(ksp->vec_rhs,&ksp->diagonal);CHKERRQ(ierr);
      }
      ierr = MatGetDiagonal(mat,ksp->diagonal);CHKERRQ(ierr);
      ierr = VecGetLocalSize(ksp->diagonal,&n);CHKERRQ(ierr);
      ierr = VecGetArray(ksp->diagonal,&xx);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
	if (xx[i] != 0.0) xx[i] = 1.0/sqrt(PetscAbsScalar(xx[i]));
	else {
	  xx[i]     = 1.0;
	  zeroflag  = PETSC_TRUE;
	}
      }
      ierr = VecRestoreArray(ksp->diagonal,&xx);CHKERRQ(ierr);
      if (zeroflag) {
	PetscLogInfo(ksp,"KSPSetUp:Zero detected in diagonal of matrix, using 1 at those locations\n");
      }
      ierr = MatDiagonalScale(mat,ksp->diagonal,ksp->diagonal);CHKERRQ(ierr);
      ksp->dscalefix2 = PETSC_FALSE;
    } else {
      SETERRQ(1,"No support for diagonal scaling of linear system if preconditioner matrix not actual matrix");
    }
  }
  ierr = PetscLogEventEnd(KSP_SetUp,ksp,ksp->vec_rhs,ksp->vec_sol,0);CHKERRQ(ierr);
  ierr = PCSetUp(ksp->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static char *convergedreasons[] = {"preconditioner is indefinite",                  "matrix or preconditioner is nonsymmetric",
                                   "breakdown in BICG",                             "breakdown",
                                   "residual norm increased by dtol",               "reach maximum number of iterations",
                                   "not used",                                      "not used",
                                   "never reached",                                 "not used",
                                   "residual norm decreased by relative tolerance", "residual norm decreased by absolute tolerance",
                                   "only one iteration requested",                  "negative curvature obtained in QCG",
                                   "constrained in QCG",                            "small step length reached"};

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve"
/*@
   KSPSolve - Solves linear system; usually not called directly, rather 
   it is called by a call to KSPSolve().

   Collective on KSP

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  its - number of iterations required

   Options Database Keys:
+  -ksp_compute_eigenvalues - compute preconditioned operators eigenvalues
.  -ksp_plot_eigenvalues - plot the computed eigenvalues in an X-window
.  -ksp_compute_eigenvalues_explicitly - compute the eigenvalues by forming the 
      dense operator and useing LAPACK
.  -ksp_plot_eigenvalues_explicitly - plot the explicitly computing eigenvalues
-  -ksp_view_binary - save matrix and right hand side that define linear system to the 
                      default binary viewer (can be
       read later with src/ksp/examples/tutorials/ex10.c for testing solvers)

   Notes:
   On return, the parameter "its" contains either the iteration
   number at which convergence was successfully reached or failure was detected.

   Call KSPGetConvergedReason() to determine if the solver converged or failed and 
   why.
   
   If using a direct method (e.g., via the KSP solver
   KSPPREONLY and a preconditioner such as PCLU/PCILU),
   then its=1.  See KSPSetTolerances() and KSPDefaultConverged()
   for more details.

   Understanding Convergence:
   The routines KSPSetMonitor(), KSPComputeEigenvalues(), and
   KSPComputeEigenvaluesExplicitly() provide information on additional
   options to monitor convergence and print eigenvalue information.

   Level: developer

.keywords: KSP, solve, linear system

.seealso: KSPCreate(), KSPSetUp(), KSPDestroy(), KSPSetTolerances(), KSPDefaultConverged(),
          KSPSolve(), KSPSolveTranspose()
@*/
int KSPSolve(KSP ksp) 
{
  int          ierr,rank;
  PetscTruth   flag1,flag2,flg;
  PetscScalar  zero = 0.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);

  ierr = PetscOptionsHasName(ksp->prefix,"-ksp_view_binary",&flg);CHKERRQ(ierr); 
  if (flg) {
    Mat mat;
    ierr = PCGetOperators(ksp->B,&mat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatView(mat,PETSC_VIEWER_BINARY_(ksp->comm));CHKERRQ(ierr);
    ierr = VecView(ksp->vec_rhs,PETSC_VIEWER_BINARY_(ksp->comm));CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(KSP_Solve,ksp,ksp->vec_rhs,ksp->vec_sol,0);CHKERRQ(ierr);

  /* reset the residual history list if requested */
  if (ksp->res_hist_reset) ksp->res_hist_len = 0;

  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);

  ksp->transpose_solve = PETSC_FALSE;
  ierr = PCPreSolve(ksp->B,ksp);CHKERRQ(ierr);
  /* diagonal scale RHS if called for */
  if (ksp->dscale) {
    ierr = VecPointwiseMult(ksp->diagonal,ksp->vec_rhs,ksp->vec_rhs);CHKERRQ(ierr);
    /* second time in, but matrix was scaled back to original */
    if (ksp->dscalefix && ksp->dscalefix2) {
      Mat mat;

      ierr = PCGetOperators(ksp->B,&mat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatDiagonalScale(mat,ksp->diagonal,ksp->diagonal);CHKERRQ(ierr);
    }
  }

  if (!ksp->setupcalled){ ierr = KSPSetUp(ksp);CHKERRQ(ierr);}
  if (ksp->guess_zero) { ierr = VecSet(&zero,ksp->vec_sol);CHKERRQ(ierr);}
  if (ksp->guess_knoll) {
    ierr            = PCApply(ksp->B,ksp->vec_rhs,ksp->vec_sol,PC_LEFT);CHKERRQ(ierr);
    ksp->guess_zero = PETSC_FALSE;
  }
  ierr = (*ksp->ops->solve)(ksp);CHKERRQ(ierr);
  if (!ksp->reason) {
    SETERRQ(1,"Internal error, solver returned without setting converged reason");
  }
  if (ksp->printreason) {
    if (ksp->reason > 0) {
      ierr = PetscPrintf(ksp->comm,"Linear solve converged due to %s\n",convergedreasons[ksp->reason+8]);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(ksp->comm,"Linear solve did not converge due to %s\n",convergedreasons[ksp->reason+8]);CHKERRQ(ierr);
    }
  }

  /* diagonal scale solution if called for */
  if (ksp->dscale) {
    ierr = VecPointwiseMult(ksp->diagonal,ksp->vec_sol,ksp->vec_sol);CHKERRQ(ierr);
    /* unscale right hand side and matrix */
    if (ksp->dscalefix) {
      Mat mat;

      ierr = VecReciprocal(ksp->diagonal);CHKERRQ(ierr);
      ierr = VecPointwiseMult(ksp->diagonal,ksp->vec_rhs,ksp->vec_rhs);CHKERRQ(ierr);
      ierr = PCGetOperators(ksp->B,&mat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatDiagonalScale(mat,ksp->diagonal,ksp->diagonal);CHKERRQ(ierr);
      ierr = VecReciprocal(ksp->diagonal);CHKERRQ(ierr);
      ksp->dscalefix2 = PETSC_TRUE;
    }
  }
  ierr = PCPostSolve(ksp->B,ksp);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(KSP_Solve,ksp,ksp->vec_rhs,ksp->vec_sol,0);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(ksp->comm,&rank);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(ksp->prefix,"-ksp_compute_eigenvalues",&flag1);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(ksp->prefix,"-ksp_plot_eigenvalues",&flag2);CHKERRQ(ierr);
  if (flag1 || flag2) {
    int       nits,n,i,neig;
    PetscReal *r,*c;
   
    ierr = KSPGetIterationNumber(ksp,&nits);CHKERRQ(ierr);
    n = nits+2;

    if (!n) {
      ierr = PetscPrintf(ksp->comm,"Zero iterations in solver, cannot approximate any eigenvalues\n");CHKERRQ(ierr);
    } else {
      ierr = PetscMalloc(2*n*sizeof(PetscReal),&r);CHKERRQ(ierr);
      c = r + n;
      ierr = KSPComputeEigenvalues(ksp,n,r,c,&neig);CHKERRQ(ierr);
      if (flag1) {
        ierr = PetscPrintf(ksp->comm,"Iteratively computed eigenvalues\n");CHKERRQ(ierr);
        for (i=0; i<neig; i++) {
          if (c[i] >= 0.0) {ierr = PetscPrintf(ksp->comm,"%g + %gi\n",r[i],c[i]);CHKERRQ(ierr);}
          else             {ierr = PetscPrintf(ksp->comm,"%g - %gi\n",r[i],-c[i]);CHKERRQ(ierr);}
        }
      }
      if (flag2 && !rank) {
        PetscViewer viewer;
        PetscDraw   draw;
        PetscDrawSP drawsp;

        ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Iteratively Computed Eigenvalues",
                               PETSC_DECIDE,PETSC_DECIDE,300,300,&viewer);CHKERRQ(ierr);
        ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
        ierr = PetscDrawSPCreate(draw,1,&drawsp);CHKERRQ(ierr);
        for (i=0; i<neig; i++) {
          ierr = PetscDrawSPAddPoint(drawsp,r+i,c+i);CHKERRQ(ierr);
        }
        ierr = PetscDrawSPDraw(drawsp);CHKERRQ(ierr);
        ierr = PetscDrawSPDestroy(drawsp);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
      }
      ierr = PetscFree(r);CHKERRQ(ierr);
    }
  }

  ierr = PetscOptionsHasName(ksp->prefix,"-ksp_compute_eigenvalues_explicitly",&flag1);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(ksp->prefix,"-ksp_plot_eigenvalues_explicitly",&flag2);CHKERRQ(ierr);
  if (flag1 || flag2) {
    int       n,i;
    PetscReal *r,*c;
    ierr = VecGetSize(ksp->vec_sol,&n);CHKERRQ(ierr);
    ierr = PetscMalloc(2*n*sizeof(PetscReal),&r);CHKERRQ(ierr);
    c = r + n;
    ierr = KSPComputeEigenvaluesExplicitly(ksp,n,r,c);CHKERRQ(ierr); 
    if (flag1) {
      ierr = PetscPrintf(ksp->comm,"Explicitly computed eigenvalues\n");CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        if (c[i] >= 0.0) {ierr = PetscPrintf(ksp->comm,"%g + %gi\n",r[i],c[i]);CHKERRQ(ierr);}
        else             {ierr = PetscPrintf(ksp->comm,"%g - %gi\n",r[i],-c[i]);CHKERRQ(ierr);}
      }
    }
    if (flag2 && !rank) {
      PetscViewer viewer;
      PetscDraw   draw;
      PetscDrawSP drawsp;

      ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Explicitly Computed Eigenvalues",0,320,300,300,&viewer);CHKERRQ(ierr);
      ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
      ierr = PetscDrawSPCreate(draw,1,&drawsp);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        ierr = PetscDrawSPAddPoint(drawsp,r+i,c+i);CHKERRQ(ierr);
      }
      ierr = PetscDrawSPDraw(drawsp);CHKERRQ(ierr);
      ierr = PetscDrawSPDestroy(drawsp);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    }
    ierr = PetscFree(r);CHKERRQ(ierr);
  }

  ierr = PetscOptionsHasName(ksp->prefix,"-ksp_view_operator",&flag2);CHKERRQ(ierr);
  if (flag2) {
    Mat A,B;
    ierr = PCGetOperators(ksp->B,&A,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatComputeExplicitOperator(A,&B);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(ksp->comm),PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = MatView(B,PETSC_VIEWER_STDOUT_(ksp->comm));CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_(ksp->comm));CHKERRQ(ierr);
    ierr = MatDestroy(B);CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(ksp->prefix,"-ksp_view_operator_binary",&flag2);CHKERRQ(ierr);
  if (flag2) {
    Mat A,B;
    ierr = PCGetOperators(ksp->B,&A,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatComputeExplicitOperator(A,&B);CHKERRQ(ierr);
    ierr = MatView(B,PETSC_VIEWER_BINARY_(ksp->comm));CHKERRQ(ierr);
    ierr = MatDestroy(B);CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(ksp->prefix,"-ksp_view_preconditioned_operator_binary",&flag2);CHKERRQ(ierr);
  if (flag2) {
    Mat B;
    ierr = KSPComputeExplicitOperator(ksp,&B);CHKERRQ(ierr);
    ierr = MatView(B,PETSC_VIEWER_BINARY_(ksp->comm));CHKERRQ(ierr);
    ierr = MatDestroy(B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolveTranspose"
/*@
   KSPSolveTranspose - Solves the transpose of a linear system. Usually
   accessed through KSPSolveTranspose().

   Collective on KSP

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Notes:
   On return, the parameter "its" contains either the iteration
   number at which convergence was successfully reached, or the
   negative of the iteration at which divergence or breakdown was detected.

   Currently only supported by KSPType of KSPPREONLY. This routine is usally 
   only used internally by the BiCG solver on the subblocks in BJacobi and ASM.

   Level: developer

.keywords: KSP, solve, linear system

.seealso: KSPCreate(), KSPSetUp(), KSPDestroy(), KSPSetTolerances(), KSPDefaultConverged(),
          KSPSolve()
@*/
int KSPSolveTranspose(KSP ksp)
{
  int           ierr;
  PetscScalar   zero = 0.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);

  if (!ksp->setupcalled){ ierr = KSPSetUp(ksp);CHKERRQ(ierr);}
  if (ksp->guess_zero) { ierr = VecSet(&zero,ksp->vec_sol);CHKERRQ(ierr);}
  ksp->transpose_solve = PETSC_TRUE;
  ierr = (*ksp->ops->solve)(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy"
/*@C
   KSPDestroy - Destroys KSP context.

   Collective on KSP

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Level: developer

.keywords: KSP, destroy

.seealso: KSPCreate(), KSPSetUp(), KSPSolve()
@*/
int KSPDestroy(KSP ksp)
{
  int i,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (--ksp->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(ksp);CHKERRQ(ierr);

  if (ksp->ops->destroy) {
    ierr = (*ksp->ops->destroy)(ksp);CHKERRQ(ierr);
  }
  for (i=0; i<ksp->numbermonitors; i++) {
    if (ksp->monitordestroy[i]) {
      ierr = (*ksp->monitordestroy[i])(ksp->monitorcontext[i]);CHKERRQ(ierr);
    }
  }
  ierr = PCDestroy(ksp->B);CHKERRQ(ierr);
  if (ksp->diagonal) {ierr = VecDestroy(ksp->diagonal);CHKERRQ(ierr);}
  PetscLogObjectDestroy(ksp);
  PetscHeaderDestroy(ksp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetPreconditionerSide"
/*@
    KSPSetPreconditionerSide - Sets the preconditioning side.

    Collective on KSP

    Input Parameter:
.   ksp - iterative context obtained from KSPCreate()

    Output Parameter:
.   side - the preconditioning side, where side is one of
.vb
      PC_LEFT - left preconditioning (default)
      PC_RIGHT - right preconditioning
      PC_SYMMETRIC - symmetric preconditioning
.ve

    Options Database Keys:
+   -ksp_left_pc - Sets left preconditioning
.   -ksp_right_pc - Sets right preconditioning
-   -ksp_symmetric_pc - Sets symmetric preconditioning

    Notes:
    Left preconditioning is used by default.  Symmetric preconditioning is
    currently available only for the KSPQCG method. Note, however, that
    symmetric preconditioning can be emulated by using either right or left
    preconditioning and a pre or post processing step.

    Level: intermediate

.keywords: KSP, set, right, left, symmetric, side, preconditioner, flag

.seealso: KSPGetPreconditionerSide()
@*/
int KSPSetPreconditionerSide(KSP ksp,PCSide side)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->pc_side = side;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetPreconditionerSide"
/*@C
    KSPGetPreconditionerSide - Gets the preconditioning side.

    Not Collective

    Input Parameter:
.   ksp - iterative context obtained from KSPCreate()

    Output Parameter:
.   side - the preconditioning side, where side is one of
.vb
      PC_LEFT - left preconditioning (default)
      PC_RIGHT - right preconditioning
      PC_SYMMETRIC - symmetric preconditioning
.ve

    Level: intermediate

.keywords: KSP, get, right, left, symmetric, side, preconditioner, flag

.seealso: KSPSetPreconditionerSide()
@*/
int KSPGetPreconditionerSide(KSP ksp,PCSide *side) 
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *side = ksp->pc_side;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetTolerances"
/*@
   KSPGetTolerances - Gets the relative, absolute, divergence, and maximum
   iteration tolerances used by the default KSP convergence tests. 

   Not Collective

   Input Parameter:
.  ksp - the Krylov subspace context
  
   Output Parameters:
+  rtol - the relative convergence tolerance
.  atol - the absolute convergence tolerance
.  dtol - the divergence tolerance
-  maxits - maximum number of iterations

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

   Level: intermediate

.keywords: KSP, get, tolerance, absolute, relative, divergence, convergence,
           maximum, iterations

.seealso: KSPSetTolerances()
@*/
int KSPGetTolerances(KSP ksp,PetscReal *rtol,PetscReal *atol,PetscReal *dtol,int *maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (atol)   *atol   = ksp->atol;
  if (rtol)   *rtol   = ksp->rtol;
  if (dtol)   *dtol   = ksp->divtol;
  if (maxits) *maxits = ksp->max_it;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetTolerances"
/*@
   KSPSetTolerances - Sets the relative, absolute, divergence, and maximum
   iteration tolerances used by the default KSP convergence testers. 

   Collective on KSP

   Input Parameters:
+  ksp - the Krylov subspace context
.  rtol - the relative convergence tolerance
   (relative decrease in the residual norm)
.  atol - the absolute convergence tolerance 
   (absolute size of the residual norm)
.  dtol - the divergence tolerance
   (amount residual can increase before KSPDefaultConverged()
   concludes that the method is diverging)
-  maxits - maximum number of iterations to use

   Options Database Keys:
+  -ksp_atol <atol> - Sets atol
.  -ksp_rtol <rtol> - Sets rtol
.  -ksp_divtol <dtol> - Sets dtol
-  -ksp_max_it <maxits> - Sets maxits

   Notes:
   Use PETSC_DEFAULT to retain the default value of any of the tolerances.

   See KSPDefaultConverged() for details on the use of these parameters
   in the default convergence test.  See also KSPSetConvergenceTest() 
   for setting user-defined stopping criteria.

   Level: intermediate

.keywords: KSP, set, tolerance, absolute, relative, divergence, 
           convergence, maximum, iterations

.seealso: KSPGetTolerances(), KSPDefaultConverged(), KSPSetConvergenceTest()
@*/
int KSPSetTolerances(KSP ksp,PetscReal rtol,PetscReal atol,PetscReal dtol,int maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (atol != PETSC_DEFAULT)   ksp->atol   = atol;
  if (rtol != PETSC_DEFAULT)   ksp->rtol   = rtol;
  if (dtol != PETSC_DEFAULT)   ksp->divtol = dtol;
  if (maxits != PETSC_DEFAULT) ksp->max_it = maxits;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetInitialGuessNonzero"
/*@
   KSPSetInitialGuessNonzero - Tells the iterative solver that the 
   initial guess is nonzero; otherwise KSP assumes the initial guess
   is to be zero (and thus zeros it out before solving).

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  flg - PETSC_TRUE indicates the guess is non-zero, PETSC_FALSE indicates the guess is zero

   Level: beginner

   Notes:
    If this is not called the X vector is zeroed in the call to 
KSPSolve() (or KSPSolve()).

.keywords: KSP, set, initial guess, nonzero

.seealso: KSPGetInitialGuessNonzero(), KSPSetInitialGuessKnoll(), KSPGetInitialGuessKnoll()
@*/
int KSPSetInitialGuessNonzero(KSP ksp,PetscTruth flg)
{
  PetscFunctionBegin;
  ksp->guess_zero   = (PetscTruth)!(int)flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetInitialGuessNonzero"
/*@
   KSPGetInitialGuessNonzero - Determines whether the KSP solver is using
   a zero initial guess.

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  flag - PETSC_TRUE if guess is nonzero, else PETSC_FALSE

   Level: intermediate

.keywords: KSP, set, initial guess, nonzero

.seealso: KSPSetInitialGuessNonzero(), KSPSetInitialGuessKnoll(), KSPGetInitialGuessKnoll()
@*/
int KSPGetInitialGuessNonzero(KSP ksp,PetscTruth *flag)
{
  PetscFunctionBegin;
  if (ksp->guess_zero) *flag = PETSC_FALSE;
  else                 *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetInitialGuessKnoll"
/*@
   KSPSetInitialGuessKnoll - Tells the iterative solver to use PCApply(pc,b,..) to compute the initial guess (The Knoll trick)

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  flg - PETSC_TRUE or PETSC_FALSE

   Level: advanced


.keywords: KSP, set, initial guess, nonzero

.seealso: KSPGetInitialGuessKnoll(), KSPSetInitialGuessNonzero(), KSPGetInitialGuessNonzero()
@*/
int KSPSetInitialGuessKnoll(KSP ksp,PetscTruth flg)
{
  PetscFunctionBegin;
  ksp->guess_knoll   = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetInitialGuessKnoll"
/*@
   KSPGetInitialGuessKnoll - Determines whether the KSP solver is using the Knoll trick (using PCApply(pc,b,...) to compute
     the initial guess

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  flag - PETSC_TRUE if using Knoll trick, else PETSC_FALSE

   Level: advanced

.keywords: KSP, set, initial guess, nonzero

.seealso: KSPSetInitialGuessKnoll(), KSPSetInitialGuessNonzero(), KSPGetInitialGuessNonzero()
@*/
int KSPGetInitialGuessKnoll(KSP ksp,PetscTruth *flag)
{
  PetscFunctionBegin;
  *flag = ksp->guess_knoll;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetComputeSingularValues"
/*@
   KSPSetComputeSingularValues - Sets a flag so that the extreme singular 
   values will be calculated via a Lanczos or Arnoldi process as the linear 
   system is solved.

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  flg - PETSC_TRUE or PETSC_FALSE

   Options Database Key:
.  -ksp_singmonitor - Activates KSPSetComputeSingularValues()

   Notes:
   Currently this option is not valid for all iterative methods.

   Many users may just want to use the monitoring routine
   KSPSingularValueMonitor() (which can be set with option -ksp_singmonitor)
   to print the singular values at each iteration of the linear solve.

   Level: advanced

.keywords: KSP, set, compute, singular values

.seealso: KSPComputeExtremeSingularValues(), KSPSingularValueMonitor()
@*/
int KSPSetComputeSingularValues(KSP ksp,PetscTruth flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->calc_sings  = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetComputeEigenvalues"
/*@
   KSPSetComputeEigenvalues - Sets a flag so that the extreme eigenvalues
   values will be calculated via a Lanczos or Arnoldi process as the linear 
   system is solved.

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  flg - PETSC_TRUE or PETSC_FALSE

   Notes:
   Currently this option is not valid for all iterative methods.

   Level: advanced

.keywords: KSP, set, compute, eigenvalues

.seealso: KSPComputeEigenvalues(), KSPComputeEigenvaluesExplicitly()
@*/
int KSPSetComputeEigenvalues(KSP ksp,PetscTruth flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->calc_sings  = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetRhs"
/*@
   KSPSetRhs - Sets the right-hand-side vector for the linear system to
   be solved.

   Collective on KSP and Vec

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  b   - right-hand-side vector

   Level: developer

.keywords: KSP, set, right-hand-side, rhs

.seealso: KSPGetRhs(), KSPSetSolution()
@*/
int KSPSetRhs(KSP ksp,Vec b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);
  ksp->vec_rhs    = (b);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetRhs"
/*@C
   KSPGetRhs - Gets the right-hand-side vector for the linear system to
   be solved.

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  r - right-hand-side vector

   Level: developer

.keywords: KSP, get, right-hand-side, rhs

.seealso: KSPSetRhs(), KSPGetSolution()
@*/
int KSPGetRhs(KSP ksp,Vec *r)
{   
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *r = ksp->vec_rhs; 
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "KSPSetSolution"
/*@
   KSPSetSolution - Sets the location of the solution for the 
   linear system to be solved.

   Collective on KSP and Vec

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  x   - solution vector

   Level: developer

.keywords: KSP, set, solution

.seealso: KSPSetRhs(), KSPGetSolution()
@*/
int KSPSetSolution(KSP ksp,Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  ksp->vec_sol    = (x);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetSolution" 
/*@C
   KSPGetSolution - Gets the location of the solution for the 
   linear system to be solved.  Note that this may not be where the solution
   is stored during the iterative process; see KSPBuildSolution().

   Not Collective

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameters:
.  v - solution vector

   Level: developer

.keywords: KSP, get, solution

.seealso: KSPGetRhs(), KSPSetSolution(), KSPBuildSolution()
@*/
int KSPGetSolution(KSP ksp,Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE); 
  *v = ksp->vec_sol; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetPC"
/*@
   KSPSetPC - Sets the preconditioner to be used to calculate the 
   application of the preconditioner on a vector. 

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  B   - the preconditioner object

   Notes:
   Use KSPGetPC() to retrieve the preconditioner context (for example,
   to free it at the end of the computations).

   Level: developer

.keywords: KSP, set, precondition, Binv

.seealso: KSPGetPC()
@*/
int KSPSetPC(KSP ksp,PC B)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidHeaderSpecific(B,PC_COOKIE);
  PetscCheckSameComm(ksp,B);
  if (ksp->B) {ierr = PCDestroy(ksp->B);CHKERRQ(ierr);}
  ksp->B = B;
  ierr = PetscObjectReference((PetscObject)ksp->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetPC"
/*@C
   KSPGetPC - Returns a pointer to the preconditioner context
   set with KSPSetPC().

   Not Collective

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  B - preconditioner context

   Level: developer

.keywords: KSP, get, preconditioner, Binv

.seealso: KSPSetPC()
@*/
int KSPGetPC(KSP ksp,PC *B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *B = ksp->B; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetMonitor"
/*@C
   KSPSetMonitor - Sets an ADDITIONAL function to be called at every iteration to monitor 
   the residual/error etc.
      
   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
.  monitor - pointer to function (if this is PETSC_NULL, it turns off monitoring
.  mctx    - [optional] context for private data for the
             monitor routine (use PETSC_NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context
          (may be PETSC_NULL)

   Calling Sequence of monitor:
$     monitor (KSP ksp, int it, PetscReal rnorm, void *mctx)

+  ksp - iterative context obtained from KSPCreate()
.  it - iteration number
.  rnorm - (estimated) 2-norm of (preconditioned) residual
-  mctx  - optional monitoring context, as set by KSPSetMonitor()

   Options Database Keys:
+    -ksp_monitor        - sets KSPDefaultMonitor()
.    -ksp_truemonitor    - sets KSPTrueMonitor()
.    -ksp_xmonitor       - sets line graph monitor,
                           uses KSPLGMonitorCreate()
.    -ksp_xtruemonitor   - sets line graph monitor,
                           uses KSPLGMonitorCreate()
.    -ksp_singmonitor    - sets KSPSingularValueMonitor()
-    -ksp_cancelmonitors - cancels all monitors that have
                          been hardwired into a code by 
                          calls to KSPSetMonitor(), but
                          does not cancel those set via
                          the options database.

   Notes:  
   The default is to do nothing.  To print the residual, or preconditioned 
   residual if KSPSetNormType(ksp,KSP_PRECONDITIONED_NORM) was called, use 
   KSPDefaultMonitor() as the monitoring routine, with a null monitoring 
   context. 

   Several different monitoring routines may be set by calling
   KSPSetMonitor() multiple times; all will be called in the 
   order in which they were set.

   Level: beginner

.keywords: KSP, set, monitor

.seealso: KSPDefaultMonitor(), KSPLGMonitorCreate(), KSPClearMonitor()
@*/
int KSPSetMonitor(KSP ksp,int (*monitor)(KSP,int,PetscReal,void*),void *mctx,int (*monitordestroy)(void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (ksp->numbermonitors >= MAXKSPMONITORS) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Too many KSP monitors set");
  }
  ksp->monitor[ksp->numbermonitors]           = monitor;
  ksp->monitordestroy[ksp->numbermonitors]    = monitordestroy;
  ksp->monitorcontext[ksp->numbermonitors++]  = (void*)mctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPClearMonitor"
/*@
   KSPClearMonitor - Clears all monitors for a KSP object.

   Collective on KSP

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

   Options Database Key:
.  -ksp_cancelmonitors - Cancels all monitors that have
    been hardwired into a code by calls to KSPSetMonitor(), 
    but does not cancel those set via the options database.

   Level: intermediate

.keywords: KSP, set, monitor

.seealso: KSPDefaultMonitor(), KSPLGMonitorCreate(), KSPSetMonitor()
@*/
int KSPClearMonitor(KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->numbermonitors = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetMonitorContext"
/*@C
   KSPGetMonitorContext - Gets the monitoring context, as set by 
   KSPSetMonitor() for the FIRST monitor only.

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  ctx - monitoring context

   Level: intermediate

.keywords: KSP, get, monitor, context

.seealso: KSPDefaultMonitor(), KSPLGMonitorCreate()
@*/
int KSPGetMonitorContext(KSP ksp,void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *ctx =      (ksp->monitorcontext[0]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetResidualHistory"
/*@
   KSPSetResidualHistory - Sets the array used to hold the residual history.
   If set, this array will contain the residual norms computed at each
   iteration of the solver.

   Not Collective

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
.  a   - array to hold history
.  na  - size of a
-  reset - PETSC_TRUE indicates the history counter is reset to zero
           for each new linear solve

   Level: advanced

   Notes: The array is NOT freed by PETSc so the user needs to keep track of 
           it and destroy once the KSP object is destroyed.

   If 'na' is PETSC_DECIDE or 'a' is PETSC_NULL, then a default array of
   length 1000 is allocated.

.keywords: KSP, set, residual, history, norm

.seealso: KSPGetResidualHistory()

@*/
int KSPSetResidualHistory(KSP ksp,PetscReal a[],int na,PetscTruth reset)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (na != PETSC_DECIDE && a != PETSC_NULL) {
    ksp->res_hist        = a;
    ksp->res_hist_max    = na;
  } else {
    ksp->res_hist_max    = 1000;
    ierr = PetscMalloc(ksp->res_hist_max*sizeof(PetscReal),&ksp->res_hist);CHKERRQ(ierr);
  }
  ksp->res_hist_len    = 0;
  ksp->res_hist_reset  = reset;


  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetResidualHistory"
/*@C
   KSPGetResidualHistory - Gets the array used to hold the residual history
   and the number of residuals it contains.

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameters:
+  a   - pointer to array to hold history (or PETSC_NULL)
-  na  - number of used entries in a (or PETSC_NULL)

   Level: advanced

   Notes:
     Can only be called after a KSPSetResidualHistory() otherwise a and na are set to zero

     The Fortran version of this routine has a calling sequence
$   call KSPGetResidualHistory(KSP ksp, integer na, integer ierr)

.keywords: KSP, get, residual, history, norm

.seealso: KSPGetResidualHistory()

@*/
int KSPGetResidualHistory(KSP ksp,PetscReal *a[],int *na)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (a)  *a = ksp->res_hist;
  if (na) *na = ksp->res_hist_len;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetConvergenceTest"
/*@C
   KSPSetConvergenceTest - Sets the function to be used to determine
   convergence.  

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
.  converge - pointer to int function
-  cctx    - context for private data for the convergence routine (may be null)

   Calling sequence of converge:
$     converge (KSP ksp, int it, PetscReal rnorm, KSPConvergedReason *reason,void *mctx)

+  ksp - iterative context obtained from KSPCreate()
.  it - iteration number
.  rnorm - (estimated) 2-norm of (preconditioned) residual
.  reason - the reason why it has converged or diverged
-  cctx  - optional convergence context, as set by KSPSetConvergenceTest()


   Notes:
   Must be called after the KSP type has been set so put this after
   a call to KSPSetType(), or KSPSetFromOptions().

   The default convergence test, KSPDefaultConverged(), aborts if the 
   residual grows to more than 10000 times the initial residual.

   The default is a combination of relative and absolute tolerances.  
   The residual value that is tested may be an approximation; routines 
   that need exact values should compute them.

   Level: advanced

.keywords: KSP, set, convergence, test, context

.seealso: KSPDefaultConverged(), KSPGetConvergenceContext()
@*/
int KSPSetConvergenceTest(KSP ksp,int (*converge)(KSP,int,PetscReal,KSPConvergedReason*,void*),void *cctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->converged = converge;	
  ksp->cnvP      = (void*)cctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetConvergenceContext"
/*@C
   KSPGetConvergenceContext - Gets the convergence context set with 
   KSPSetConvergenceTest().  

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  ctx - monitoring context

   Level: advanced

.keywords: KSP, get, convergence, test, context

.seealso: KSPDefaultConverged(), KSPSetConvergenceTest()
@*/
int KSPGetConvergenceContext(KSP ksp,void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *ctx = ksp->cnvP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPBuildSolution"
/*@C
   KSPBuildSolution - Builds the approximate solution in a vector provided.
   This routine is NOT commonly needed (see KSPSolve()).

   Collective on KSP

   Input Parameter:
.  ctx - iterative context obtained from KSPCreate()

   Output Parameter: 
   Provide exactly one of
+  v - location to stash solution.   
-  V - the solution is returned in this location. This vector is created 
       internally. This vector should NOT be destroyed by the user with
       VecDestroy().

   Notes:
   This routine can be used in one of two ways
.vb
      KSPBuildSolution(ksp,PETSC_NULL,&V);
   or
      KSPBuildSolution(ksp,v,PETSC_NULL); 
.ve
   In the first case an internal vector is allocated to store the solution
   (the user cannot destroy this vector). In the second case the solution
   is generated in the vector that the user provides. Note that for certain 
   methods, such as KSPCG, the second case requires a copy of the solution,
   while in the first case the call is essentially free since it simply 
   returns the vector where the solution already is stored.

   Level: advanced

.keywords: KSP, build, solution

.seealso: KSPGetSolution(), KSPBuildResidual()
@*/
int KSPBuildSolution(KSP ksp,Vec v,Vec *V)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (!V && !v) SETERRQ(PETSC_ERR_ARG_WRONG,"Must provide either v or V");
  if (!V) V = &v;
  ierr = (*ksp->ops->buildsolution)(ksp,v,V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPBuildResidual"
/*@C
   KSPBuildResidual - Builds the residual in a vector provided.

   Collective on KSP

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameters:
+  v - optional location to stash residual.  If v is not provided,
       then a location is generated.
.  t - work vector.  If not provided then one is generated.
-  V - the residual

   Notes:
   Regardless of whether or not v is provided, the residual is 
   returned in V.

   Level: advanced

.keywords: KSP, build, residual

.seealso: KSPBuildSolution()
@*/
int KSPBuildResidual(KSP ksp,Vec t,Vec v,Vec *V)
{
  int flag = 0,ierr;
  Vec w = v,tt = t;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (!w) {
    ierr = VecDuplicate(ksp->vec_rhs,&w);CHKERRQ(ierr);
    PetscLogObjectParent((PetscObject)ksp,w);
  }
  if (!tt) {
    ierr = VecDuplicate(ksp->vec_rhs,&tt);CHKERRQ(ierr); flag = 1;
    PetscLogObjectParent((PetscObject)ksp,tt);
  }
  ierr = (*ksp->ops->buildresidual)(ksp,tt,w,V);CHKERRQ(ierr);
  if (flag) {ierr = VecDestroy(tt);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetDiagonalScale"
/*@
   KSPSetDiagonalScale - Tells KSP to diagonally scale the system
     before solving. This actually CHANGES the matrix (and right hand side).

   Collective on KSP

   Input Parameter:
+  ksp - the KSP context
-  scale - PETSC_TRUE or PETSC_FALSE

   Notes:
    BE CAREFUL with this routine: it actually scales the matrix and right 
    hand side that define the system. After the system is solved the matrix
    and right hand side remain scaled.

    This routine is only used if the matrix and preconditioner matrix are
    the same thing.
 
    If you use this with the PCType Eisenstat preconditioner than you can 
    use the PCEisenstatNoDiagonalScaling() option, or -pc_eisenstat_no_diagonal_scaling
    to save some unneeded, redundant flops.

   Level: intermediate

.keywords: KSP, set, options, prefix, database

.seealso: KSPGetDiagonalScale(), KSPSetDiagonalScaleFix()
@*/
int KSPSetDiagonalScale(KSP ksp,PetscTruth scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->dscale = scale;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetDiagonalScale"
/*@C
   KSPGetDiagonalScale - Checks if KSP solver scales the matrix and
                          right hand side

   Not Collective

   Input Parameter:
.  ksp - the KSP context

   Output Parameter:
.  scale - PETSC_TRUE or PETSC_FALSE

   Notes:
    BE CAREFUL with this routine: it actually scales the matrix and right 
    hand side that define the system. After the system is solved the matrix
    and right hand side remain scaled.

    This routine is only used if the matrix and preconditioner matrix are
    the same thing.

   Level: intermediate

.keywords: KSP, set, options, prefix, database

.seealso: KSPSetDiagonalScale(), KSPSetDiagonalScaleFix()
@*/
int KSPGetDiagonalScale(KSP ksp,PetscTruth *scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *scale = ksp->dscale;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetDiagonalScaleFix"
/*@
   KSPSetDiagonalScaleFix - Tells KSP to diagonally scale the system
     back after solving.

   Collective on KSP

   Input Parameter:
+  ksp - the KSP context
-  fix - PETSC_TRUE to scale back after the system solve, PETSC_FALSE to not 
         rescale (default)

   Notes:
     Must be called after KSPSetDiagonalScale()

     Using this will slow things down, because it rescales the matrix before and
     after each linear solve. This is intended mainly for testing to allow one
     to easily get back the original system to make sure the solution computed is
     accurate enough.

    This routine is only used if the matrix and preconditioner matrix are
    the same thing.

   Level: intermediate

.keywords: KSP, set, options, prefix, database

.seealso: KSPGetDiagonalScale(), KSPSetDiagonalScale(), KSPGetDiagonalScaleFix()
@*/
int KSPSetDiagonalScaleFix(KSP ksp,PetscTruth fix)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->dscalefix = fix;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetDiagonalScaleFix"
/*@
   KSPGetDiagonalScaleFix - Determines if KSP diagonally scales the system
     back after solving.

   Collective on KSP

   Input Parameter:
.  ksp - the KSP context

   Output Parameter:
.  fix - PETSC_TRUE to scale back after the system solve, PETSC_FALSE to not 
         rescale (default)

   Notes:
     Must be called after KSPSetDiagonalScale()

     If PETSC_TRUE will slow things down, because it rescales the matrix before and
     after each linear solve. This is intended mainly for testing to allow one
     to easily get back the original system to make sure the solution computed is
     accurate enough.

    This routine is only used if the matrix and preconditioner matrix are
    the same thing.

   Level: intermediate

.keywords: KSP, set, options, prefix, database

.seealso: KSPGetDiagonalScale(), KSPSetDiagonalScale(), KSPSetDiagonalScaleFix()
@*/
int KSPGetDiagonalScaleFix(KSP ksp,PetscTruth *fix)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *fix = ksp->dscalefix;
  PetscFunctionReturn(0);
}
