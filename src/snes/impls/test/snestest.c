
#include <petsc-private/snesimpl.h>

typedef struct {
  PetscBool  complete_print;
} SNES_Test;

/*
     SNESSolve_Test - Tests whether a hand computed Jacobian 
     matches one compute via finite differences.
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESSolve_Test"
PetscErrorCode SNESSolve_Test(SNES snes)
{
  Mat            A = snes->jacobian,B;
  Vec            x = snes->vec_sol,f = snes->vec_func;
  PetscErrorCode ierr;
  PetscInt       i;
  MatStructure   flg;
  PetscReal      nrm,gnorm;
  SNES_Test      *neP = (SNES_Test*)snes->data;

  PetscFunctionBegin;

  if (A != snes->jacobian_pre) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot test with alternative preconditioner");

  ierr = PetscPrintf(((PetscObject)snes)->comm,"Testing hand-coded Jacobian, if the ratio is\n");CHKERRQ(ierr);
  ierr = PetscPrintf(((PetscObject)snes)->comm,"O(1.e-8), the hand-coded Jacobian is probably correct.\n");CHKERRQ(ierr);
  if (!neP->complete_print) {
    ierr = PetscPrintf(((PetscObject)snes)->comm,"Run with -snes_test_display to show difference\n");CHKERRQ(ierr);
    ierr = PetscPrintf(((PetscObject)snes)->comm,"of hand-coded and finite difference Jacobian.\n");CHKERRQ(ierr);
  }

  for (i=0; i<3; i++) {
    void *functx;
    static const char *const loc[] = {"user-defined state","constant state -1.0","constant state 1.0"};
    if (i == 1) {ierr = VecSet(x,-1.0);CHKERRQ(ierr);}
    else if (i == 2) {ierr = VecSet(x,1.0);CHKERRQ(ierr);}

    /* evaluate the function at this point because SNESDefaultComputeJacobianColor() assumes that the function has been evaluated and put into snes->vec_func */
    ierr = SNESComputeFunction(snes,x,f);CHKERRQ(ierr);
    if (snes->domainerror) {
      ierr = PetscPrintf(((PetscObject)snes)->comm,"Domain error at %s\n",loc[i]);CHKERRQ(ierr);
      snes->domainerror = PETSC_FALSE;
      continue;
    }

    /* compute both versions of Jacobian */
    ierr = SNESComputeJacobian(snes,x,&A,&A,&flg);CHKERRQ(ierr);
    if (!i) {
      PetscInt m,n,M,N;
      ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
      ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
      ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
      ierr = MatSetSizes(B,m,n,M,N);CHKERRQ(ierr);
      ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
      ierr = MatSetUp(B);CHKERRQ(ierr);
    }
    ierr = SNESGetFunction(snes,PETSC_NULL,PETSC_NULL,&functx);CHKERRQ(ierr);
    ierr = SNESDefaultComputeJacobian(snes,x,&B,&B,&flg,functx);CHKERRQ(ierr);
    if (neP->complete_print) {
      MPI_Comm    comm;
      PetscViewer viewer;
      ierr = PetscPrintf(((PetscObject)snes)->comm,"Finite difference Jacobian (%s)\n",loc[i]);CHKERRQ(ierr);
      ierr = PetscObjectGetComm((PetscObject)B,&comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);
      ierr = MatView(B,viewer);CHKERRQ(ierr);
    }
    /* compare */
    ierr = MatAYPX(B,-1.0,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(B,NORM_FROBENIUS,&nrm);CHKERRQ(ierr);
    ierr = MatNorm(A,NORM_FROBENIUS,&gnorm);CHKERRQ(ierr);
    if (neP->complete_print) {
      MPI_Comm    comm;
      PetscViewer viewer;
      ierr = PetscPrintf(((PetscObject)snes)->comm,"Hand-coded Jacobian (%s)\n",loc[i]);CHKERRQ(ierr);
      ierr = PetscObjectGetComm((PetscObject)B,&comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);
      ierr = MatView(A,viewer);CHKERRQ(ierr);
      ierr = PetscPrintf(((PetscObject)snes)->comm,"Hand-coded minus finite difference Jacobian (%s)\n",loc[i]);CHKERRQ(ierr);
      ierr = MatView(B,viewer);CHKERRQ(ierr);
    }
    if (!gnorm) gnorm = 1; /* just in case */
    ierr = PetscPrintf(((PetscObject)snes)->comm,"Norm of matrix ratio %g difference %g (%s)\n",(double)(nrm/gnorm),(double)nrm,loc[i]);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  /*
         Return error code cause Jacobian not good
  */
  PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
}
/* ------------------------------------------------------------ */
#undef __FUNCT__  
#define __FUNCT__ "SNESDestroy_Test"
PetscErrorCode SNESDestroy_Test(SNES snes)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetFromOptions_Test"
static PetscErrorCode SNESSetFromOptions_Test(SNES snes)
{
  SNES_Test      *ls = (SNES_Test *)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscOptionsHead("Hand-coded Jacobian tester options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_test_display","Display difference between hand-coded and finite difference Jacobians","None",ls->complete_print,&ls->complete_print,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
/*MC
      SNESTEST - Test hand-coded Jacobian against finite difference Jacobian

   Options Database:
.    -snes_test_display  Display difference between approximate and hand-coded Jacobian

   Level: intermediate

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESLS, SNESTR

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESCreate_Test"
PetscErrorCode  SNESCreate_Test(SNES  snes)
{
  SNES_Test      *neP;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->solve           = SNESSolve_Test;
  snes->ops->destroy         = SNESDestroy_Test;
  snes->ops->setfromoptions  = SNESSetFromOptions_Test;
  snes->ops->view            = 0;
  snes->ops->setup           = 0;
  snes->ops->reset           = 0;

  snes->usesksp             = PETSC_FALSE;

  ierr                  = PetscNewLog(snes,SNES_Test,&neP);CHKERRQ(ierr);
  snes->data            = (void*)neP;
  neP->complete_print   = PETSC_FALSE;
  PetscFunctionReturn(0);
}
EXTERN_C_END
