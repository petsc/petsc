#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct {
  PetscBool  check_gradient;
  PetscBool  check_hessian;
  PetscBool  complete_print;
} Tao_Test;

/*
     TaoSolve_Test - Tests whether a hand computed Hessian
     matches one compute via finite differences.
*/
PetscErrorCode TaoSolve_Test(Tao tao)
{
  Mat            A = tao->hessian,B;
  Vec            x = tao->solution,g1,g2;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      nrm,gnorm,hcnorm,fdnorm,hcmax,fdmax,diffmax,diffnorm;
  PetscScalar    dot;
  MPI_Comm       comm;
  Tao_Test        *fd = (Tao_Test*)tao->data;

  PetscFunctionBegin;
  comm = ((PetscObject)tao)->comm;
  if (fd->check_gradient) {
    ierr = VecDuplicate(x,&g1);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&g2);CHKERRQ(ierr);

    ierr = PetscPrintf(comm,"Testing hand-coded gradient (hc) against finite difference gradient (fd), if the ratio ||fd - hc|| / ||hc|| is\n");CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"0 (1.e-8), the hand-coded gradient is probably correct.\n");CHKERRQ(ierr);

    if (!fd->complete_print) {
      ierr = PetscPrintf(comm,"Run with -tao_test_display to show difference\n");CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"between hand-coded and finite difference gradient.\n");CHKERRQ(ierr);
    }
    for (i=0; i<3; i++) {
      if (i == 1) {ierr = VecSet(x,-1.0);CHKERRQ(ierr);}
      else if (i == 2) {ierr = VecSet(x,1.0);CHKERRQ(ierr);}

      /* Compute both version of gradient */
      ierr = TaoComputeGradient(tao,x,g1);CHKERRQ(ierr);
      ierr = TaoDefaultComputeGradient(tao,x,g2,NULL);CHKERRQ(ierr);
      if (fd->complete_print) {
        MPI_Comm gcomm;
        PetscViewer viewer;
        ierr = PetscPrintf(comm,"Finite difference gradient\n");CHKERRQ(ierr);
        ierr = PetscObjectGetComm((PetscObject)g2,&gcomm);CHKERRQ(ierr);
        ierr = PetscViewerASCIIGetStdout(gcomm,&viewer);CHKERRQ(ierr);
        ierr = VecView(g2,viewer);CHKERRQ(ierr);
        ierr = PetscPrintf(comm,"Hand-coded gradient\n");CHKERRQ(ierr);
        ierr = PetscObjectGetComm((PetscObject)g1,&gcomm);CHKERRQ(ierr);
        ierr = PetscViewerASCIIGetStdout(gcomm,&viewer);CHKERRQ(ierr);
        ierr = VecView(g1,viewer);CHKERRQ(ierr);
        ierr = PetscPrintf(comm,"\n");CHKERRQ(ierr);
      }

      ierr = VecNorm(g2,NORM_2,&fdnorm);CHKERRQ(ierr);
      ierr = VecNorm(g1,NORM_2,&hcnorm);CHKERRQ(ierr);
      ierr = VecNorm(g2,NORM_INFINITY,&fdmax);CHKERRQ(ierr);
      ierr = VecNorm(g1,NORM_INFINITY,&hcmax);CHKERRQ(ierr);
      ierr = VecDot(g1,g2,&dot);CHKERRQ(ierr);
      ierr = VecAXPY(g2,-1.0,g1);CHKERRQ(ierr);
      ierr = VecNorm(g2,NORM_2,&diffnorm);CHKERRQ(ierr);
      ierr = VecNorm(g2,NORM_INFINITY,&diffmax);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"||fd|| %g, ||hc|| = %g, angle cosine = (fd'hc)/||fd||||hc|| = %g\n", (double)fdnorm, (double)hcnorm, (double)(PetscRealPart(dot)/(fdnorm*hcnorm)));CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"2-norm ||fd-hc||/max(||hc||,||fd||) = %g, difference ||fd-hc|| = %g\n", (double)(diffnorm/PetscMax(hcnorm,fdnorm)), (double)diffnorm);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"max-norm ||fd-hc||/max(||hc||,||fd||) = %g, difference ||fd-hc|| = %g\n", (double)(diffmax/PetscMax(hcmax,fdmax)), (double)diffmax);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&g1);CHKERRQ(ierr);
    ierr = VecDestroy(&g2);CHKERRQ(ierr);
  }

  if (fd->check_hessian) {
    if (A != tao->hessian_pre) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot test with alternative preconditioner");

    ierr = PetscPrintf(comm,"Testing hand-coded Hessian (hc) against finite difference Hessian (fd). If the ratio is\n");CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"O (1.e-8), the hand-coded Hessian is probably correct.\n");CHKERRQ(ierr);

    if (!fd->complete_print) {
      ierr = PetscPrintf(comm,"Run with -tao_test_display to show difference\n");CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"of hand-coded and finite difference Hessian.\n");CHKERRQ(ierr);
    }
    for (i=0;i<3;i++) {
      /* compute both versions of Hessian */
      ierr = TaoComputeHessian(tao,x,A,A);CHKERRQ(ierr);
      if (!i) {ierr = MatConvert(A,MATSAME,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);}
      ierr = TaoDefaultComputeHessian(tao,x,B,B,tao->user_hessP);CHKERRQ(ierr);
      if (fd->complete_print) {
        MPI_Comm    bcomm;
        PetscViewer viewer;
        ierr = PetscPrintf(comm,"Finite difference Hessian\n");CHKERRQ(ierr);
        ierr = PetscObjectGetComm((PetscObject)B,&bcomm);CHKERRQ(ierr);
        ierr = PetscViewerASCIIGetStdout(bcomm,&viewer);CHKERRQ(ierr);
        ierr = MatView(B,viewer);CHKERRQ(ierr);
      }
      /* compare */
      ierr = MatAYPX(B,-1.0,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(B,NORM_FROBENIUS,&nrm);CHKERRQ(ierr);
      ierr = MatNorm(A,NORM_FROBENIUS,&gnorm);CHKERRQ(ierr);
      if (fd->complete_print) {
        MPI_Comm    hcomm;
        PetscViewer viewer;
        ierr = PetscPrintf(comm,"Hand-coded Hessian\n");CHKERRQ(ierr);
        ierr = PetscObjectGetComm((PetscObject)B,&hcomm);CHKERRQ(ierr);
        ierr = PetscViewerASCIIGetStdout(hcomm,&viewer);CHKERRQ(ierr);
        ierr = MatView(A,viewer);CHKERRQ(ierr);
        ierr = PetscPrintf(comm,"Hand-coded minus finite difference Hessian\n");CHKERRQ(ierr);
        ierr = MatView(B,viewer);CHKERRQ(ierr);
      }
      if (!gnorm) gnorm = 1.0e-20;
      ierr = PetscPrintf(comm,"ratio ||fd-hc||/||hc|| = %g, difference ||fd-hc|| = %g\n",(double)(nrm/gnorm),(double)nrm);CHKERRQ(ierr);
    }

    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }
  tao->reason = TAO_CONVERGED_USER;
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */
PetscErrorCode TaoDestroy_Test(Tao tao)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_Test(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  Tao_Test        *fd = (Tao_Test *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Hand-coded Hessian tester options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_test_display","Display difference between hand-coded and finite difference Hessians","None",fd->complete_print,&fd->complete_print,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_test_gradient","Test Hand-coded gradient against finite-difference gradient","None",fd->check_gradient,&fd->check_gradient,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_test_hessian","Test Hand-coded hessian against finite-difference hessian","None",fd->check_hessian,&fd->check_hessian,NULL);CHKERRQ(ierr);
  if (fd->check_gradient == PETSC_FALSE && fd->check_hessian == PETSC_FALSE) {
    fd->check_gradient = PETSC_TRUE;
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
/*C
      FD_TEST - Test hand-coded Hessian against finite difference Hessian

   Options Database:
.    -tao_test_display  Display difference between approximate and hand-coded Hessian

   Level: intermediate

.seealso:  TaoCreate(), TaoSetType()

*/
PETSC_EXTERN PetscErrorCode  TaoCreate_Test(Tao  tao)
{
  Tao_Test        *fd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->setup           = 0;
  tao->ops->solve           = TaoSolve_Test;
  tao->ops->destroy         = TaoDestroy_Test;
  tao->ops->setfromoptions  = TaoSetFromOptions_Test;
  tao->ops->view            = 0;
  ierr                      = PetscNewLog(tao,&fd);CHKERRQ(ierr);
  tao->data                 = (void*)fd;
  fd->complete_print        = PETSC_FALSE;
  fd->check_gradient        = PETSC_TRUE;
  fd->check_hessian         = PETSC_FALSE;
  PetscFunctionReturn(0);
}

