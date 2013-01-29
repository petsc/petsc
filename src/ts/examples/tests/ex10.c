static char help[] = "Solves Simple DAEs \n";

#include <petscts.h>

/*
        \dot{U} = f(U,V)
        F(U,V)  = 0


*/


#undef __FUNCT__
#define __FUNCT__ "f"
/*
   f(U,V) = U + V

*/
PetscErrorCode f(PetscReal t,Vec U,Vec V,Vec F,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecWAXPY(F,1.0,U,V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "F"
/*
   F(U,V) = U - V

*/
PetscErrorCode F(PetscReal t,Vec U,Vec V,Vec F,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecWAXPY(F,-1.0,V,U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------*/

typedef struct _p_TSDAESimple *TSDAESimple;
struct _p_TSDAESimple {
  MPI_Comm       comm;
  Vec            U,V;
  PetscErrorCode (*f)(PetscReal,Vec,Vec,Vec,void*);
  PetscErrorCode (*F)(PetscReal,Vec,Vec,Vec,void*);
  void           *fctx,*Fctx;
  void           *data;
};

#undef __FUNCT__
#define __FUNCT__ "TSDAESimpleCreate"
PetscErrorCode TSDAESimpleCreate(MPI_Comm comm,TSDAESimple *tsdae)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr           = PetscNew(struct _p_TSDAESimple,tsdae);CHKERRQ(ierr);
  (*tsdae)->comm = comm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDAESimpleSetRHSFunction"
PetscErrorCode TSDAESimpleSetRHSFunction(TSDAESimple tsdae,Vec U,PetscErrorCode (*f)(PetscReal,Vec,Vec,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tsdae->f    = f;
  tsdae->U    = U;
  ierr        = PetscObjectReference((PetscObject)U);CHKERRQ(ierr);
  tsdae->fctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDAESimpleSetIFunction"
PetscErrorCode TSDAESimpleSetIFunction(TSDAESimple tsdae,Vec V,PetscErrorCode (*F)(PetscReal,Vec,Vec,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tsdae->F    = F;
  tsdae->V    = V;
  ierr        = PetscObjectReference((PetscObject)V);CHKERRQ(ierr);
  tsdae->Fctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDAESimpleDestroy"
PetscErrorCode TSDAESimpleDestroy(TSDAESimple *tsdae)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&(*tsdae)->U);CHKERRQ(ierr);
  ierr = VecDestroy(&(*tsdae)->V);CHKERRQ(ierr);
  ierr = PetscFree(*tsdae);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDAESimpleSolve"
PetscErrorCode TSDAESimpleSolve(TSDAESimple tsdae,Vec Usolution)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDAESimpleSetFromOptions"
PetscErrorCode TSDAESimpleSetFromOptions(TSDAESimple tsdae)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------*/
/*
      Integrates the system by integrating the reduced ODE system and solving the
   algebraic constraints at each stage with a seperate SNES solve.
*/

typedef struct {
  PetscReal t;
  TS        ts;
  SNES      snes;
  Vec       U;
} TSDAESimple_Reduced;

#undef __FUNCT__
#define __FUNCT__ "TSDAESimple_Reduced_TSFunction"
/*
   Defines the RHS function that is passed to the time-integrator.

   Solves F(U,V) for V and then computes f(U,V)

*/
PetscErrorCode TSDAESimple_Reduced_TSFunction(TS ts,PetscReal t,Vec U,Vec F,void *actx)
{
  TSDAESimple         tsdae = (TSDAESimple)actx;
  TSDAESimple_Reduced *red = (TSDAESimple_Reduced *)tsdae->data;
  PetscErrorCode      ierr;

  PetscFunctionBeginUser;
  red->t = t;
  red->U = U;
  ierr   = SNESSolve(red->snes,PETSC_NULL,tsdae->V);CHKERRQ(ierr);
  ierr   = (*tsdae->f)(t,U,tsdae->V,F,tsdae->fctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDAESimple_Reduced_SNESFunction"
/*
   Defines the nonlinear function that is passed to the nonlinear solver

*/
PetscErrorCode TSDAESimple_Reduced_SNESFunction(SNES snes,Vec V,Vec F,void *actx)
{
  TSDAESimple         tsdae = (TSDAESimple)actx;
  TSDAESimple_Reduced *red = (TSDAESimple_Reduced *)tsdae->data;
  PetscErrorCode      ierr;

  PetscFunctionBeginUser;
  ierr = (*tsdae->F)(red->t,red->U,V,F,tsdae->Fctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDAESimpleSetUp_Reduced"
PetscErrorCode TSDAESimpleSetUp_Reduced(TSDAESimple tsdae)
{
  PetscErrorCode      ierr;
  TSDAESimple_Reduced *red;
  Vec                 tsrhs;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(TSDAESimple_Reduced),&red);CHKERRQ(ierr);
  tsdae->data = red;

  ierr = TSCreate(tsdae->comm,&red->ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(red->ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(red->ts,TSEULER);CHKERRQ(ierr);
  ierr = VecDuplicate(tsdae->U,&tsrhs);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(red->ts,tsrhs,TSDAESimple_Reduced_TSFunction,tsdae);CHKERRQ(ierr);
  ierr = VecDestroy(&tsrhs);CHKERRQ(ierr);

  ierr = SNESCreate(tsdae->comm,&red->snes);CHKERRQ(ierr);
  ierr = SNESSetFunction(red->snes,PETSC_NULL,TSDAESimple_Reduced_SNESFunction,tsdae);CHKERRQ(ierr);
  ierr = SNESSetJacobian(red->snes,PETSC_NULL,PETSC_NULL,SNESDefaultComputeJacobian,tsdae);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDAESimpleSolve_Reduced"
PetscErrorCode TSDAESimpleSolve_Reduced(TSDAESimple tsdae,Vec U)
{
  PetscErrorCode      ierr;
  TSDAESimple_Reduced *red = (TSDAESimple_Reduced *)tsdae->data;

  PetscFunctionBegin;
  ierr = TSSolve(red->ts,U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDAESimpleSetFromOptions_Reduced"
PetscErrorCode TSDAESimpleSetFromOptions_Reduced(TSDAESimple tsdae)
{
  PetscErrorCode      ierr;
  TSDAESimple_Reduced *red = (TSDAESimple_Reduced *)tsdae->data;

  PetscFunctionBegin;
  ierr = TSSetFromOptions(red->ts);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(red->snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDAESimpleDestroy_Reduced"
PetscErrorCode TSDAESimpleDestroy_Reduced(TSDAESimple tsdae)
{
  PetscErrorCode      ierr;
  TSDAESimple_Reduced *red = (TSDAESimple_Reduced *)tsdae->data;

  PetscFunctionBegin;
  ierr = TSDestroy(&red->ts);CHKERRQ(ierr);
  ierr = SNESDestroy(&red->snes);CHKERRQ(ierr);
  ierr = PetscFree(red);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  TSDAESimple    tsdae;
  Vec            U,V,Usolution;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = TSDAESimpleCreate(PETSC_COMM_WORLD,&tsdae);CHKERRQ(ierr);
  ierr = TSDAESimpleSetFromOptions(tsdae);CHKERRQ(ierr);

  ierr = VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&U);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&Usolution);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&V);CHKERRQ(ierr);
  ierr = TSDAESimpleSetRHSFunction(tsdae,U,f,PETSC_NULL);CHKERRQ(ierr);
  ierr = TSDAESimpleSetIFunction(tsdae,V,F,PETSC_NULL);CHKERRQ(ierr);

  ierr = VecSet(Usolution,1.0);CHKERRQ(ierr);
  ierr = TSDAESimpleSolve(tsdae,Usolution);CHKERRQ(ierr);

  ierr = TSDAESimpleSetUp_Reduced(tsdae);CHKERRQ(ierr);
  ierr = TSDAESimpleSetFromOptions_Reduced(tsdae);CHKERRQ(ierr);
  ierr = TSDAESimpleSolve_Reduced(tsdae,Usolution);CHKERRQ(ierr);
  ierr = TSDAESimpleDestroy_Reduced(tsdae);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&Usolution);CHKERRQ(ierr);
  ierr = VecDestroy(&V);CHKERRQ(ierr);
  ierr = TSDAESimpleDestroy(&tsdae);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}



