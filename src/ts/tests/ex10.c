static char help[] = "Simple wrapper object to solve DAE of the form:\n\
                             \\dot{U} = f(U,V)\n\
                             F(U,V) = 0\n\n";

#include <petscts.h>

/* ----------------------------------------------------------------------------*/

typedef struct _p_TSDAESimple *TSDAESimple;
struct _p_TSDAESimple {
  MPI_Comm       comm;
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,TSDAESimple);
  PetscErrorCode (*solve)(TSDAESimple,Vec);
  PetscErrorCode (*destroy)(TSDAESimple);
  Vec            U,V;
  PetscErrorCode (*f)(PetscReal,Vec,Vec,Vec,void*);
  PetscErrorCode (*F)(PetscReal,Vec,Vec,Vec,void*);
  void           *fctx,*Fctx;
  void           *data;
};

PetscErrorCode TSDAESimpleCreate(MPI_Comm comm,TSDAESimple *tsdae)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(tsdae));
  (*tsdae)->comm = comm;
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleSetRHSFunction(TSDAESimple tsdae,Vec U,PetscErrorCode (*f)(PetscReal,Vec,Vec,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tsdae->f    = f;
  tsdae->U    = U;
  CHKERRQ(PetscObjectReference((PetscObject)U));
  tsdae->fctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleSetIFunction(TSDAESimple tsdae,Vec V,PetscErrorCode (*F)(PetscReal,Vec,Vec,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tsdae->F    = F;
  tsdae->V    = V;
  CHKERRQ(PetscObjectReference((PetscObject)V));
  tsdae->Fctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleDestroy(TSDAESimple *tsdae)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ((*(*tsdae)->destroy)(*tsdae));
  CHKERRQ(VecDestroy(&(*tsdae)->U));
  CHKERRQ(VecDestroy(&(*tsdae)->V));
  CHKERRQ(PetscFree(*tsdae));
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleSolve(TSDAESimple tsdae,Vec Usolution)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ((*tsdae->solve)(tsdae,Usolution));
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleSetFromOptions(TSDAESimple tsdae)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ((*tsdae->setfromoptions)(PetscOptionsObject,tsdae));
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------*/
/*
      Integrates the system by integrating the reduced ODE system and solving the
   algebraic constraints at each stage with a separate SNES solve.
*/

typedef struct {
  PetscReal t;
  TS        ts;
  SNES      snes;
  Vec       U;
} TSDAESimple_Reduced;

/*
   Defines the RHS function that is passed to the time-integrator.

   Solves F(U,V) for V and then computes f(U,V)

*/
PetscErrorCode TSDAESimple_Reduced_TSFunction(TS ts,PetscReal t,Vec U,Vec F,void *actx)
{
  TSDAESimple         tsdae = (TSDAESimple)actx;
  TSDAESimple_Reduced *red = (TSDAESimple_Reduced*)tsdae->data;
  PetscErrorCode      ierr;

  PetscFunctionBeginUser;
  red->t = t;
  red->U = U;
  CHKERRQ(SNESSolve(red->snes,NULL,tsdae->V));
  CHKERRQ((*tsdae->f)(t,U,tsdae->V,F,tsdae->fctx));
  PetscFunctionReturn(0);
}

/*
   Defines the nonlinear function that is passed to the nonlinear solver

*/
PetscErrorCode TSDAESimple_Reduced_SNESFunction(SNES snes,Vec V,Vec F,void *actx)
{
  TSDAESimple         tsdae = (TSDAESimple)actx;
  TSDAESimple_Reduced *red = (TSDAESimple_Reduced*)tsdae->data;
  PetscErrorCode      ierr;

  PetscFunctionBeginUser;
  CHKERRQ((*tsdae->F)(red->t,red->U,V,F,tsdae->Fctx));
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleSolve_Reduced(TSDAESimple tsdae,Vec U)
{
  PetscErrorCode      ierr;
  TSDAESimple_Reduced *red = (TSDAESimple_Reduced*)tsdae->data;

  PetscFunctionBegin;
  CHKERRQ(TSSolve(red->ts,U));
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleSetFromOptions_Reduced(PetscOptionItems *PetscOptionsObject,TSDAESimple tsdae)
{
  PetscErrorCode      ierr;
  TSDAESimple_Reduced *red = (TSDAESimple_Reduced*)tsdae->data;

  PetscFunctionBegin;
  CHKERRQ(TSSetFromOptions(red->ts));
  CHKERRQ(SNESSetFromOptions(red->snes));
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleDestroy_Reduced(TSDAESimple tsdae)
{
  PetscErrorCode      ierr;
  TSDAESimple_Reduced *red = (TSDAESimple_Reduced*)tsdae->data;

  PetscFunctionBegin;
  CHKERRQ(TSDestroy(&red->ts));
  CHKERRQ(SNESDestroy(&red->snes));
  CHKERRQ(PetscFree(red));
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleSetUp_Reduced(TSDAESimple tsdae)
{
  PetscErrorCode      ierr;
  TSDAESimple_Reduced *red;
  Vec                 tsrhs;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&red));
  tsdae->data = red;

  tsdae->setfromoptions = TSDAESimpleSetFromOptions_Reduced;
  tsdae->solve          = TSDAESimpleSolve_Reduced;
  tsdae->destroy        = TSDAESimpleDestroy_Reduced;

  CHKERRQ(TSCreate(tsdae->comm,&red->ts));
  CHKERRQ(TSSetProblemType(red->ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(red->ts,TSEULER));
  CHKERRQ(TSSetExactFinalTime(red->ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(VecDuplicate(tsdae->U,&tsrhs));
  CHKERRQ(TSSetRHSFunction(red->ts,tsrhs,TSDAESimple_Reduced_TSFunction,tsdae));
  CHKERRQ(VecDestroy(&tsrhs));

  CHKERRQ(SNESCreate(tsdae->comm,&red->snes));
  CHKERRQ(SNESSetOptionsPrefix(red->snes,"tsdaesimple_"));
  CHKERRQ(SNESSetFunction(red->snes,NULL,TSDAESimple_Reduced_SNESFunction,tsdae));
  CHKERRQ(SNESSetJacobian(red->snes,NULL,NULL,SNESComputeJacobianDefault,tsdae));
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------*/

/*
      Integrates the system by integrating directly the entire DAE system
*/

typedef struct {
  TS         ts;
  Vec        UV,UF,VF;
  VecScatter scatterU,scatterV;
} TSDAESimple_Full;

/*
   Defines the RHS function that is passed to the time-integrator.

   f(U,V)
   0

*/
PetscErrorCode TSDAESimple_Full_TSRHSFunction(TS ts,PetscReal t,Vec UV,Vec F,void *actx)
{
  TSDAESimple      tsdae = (TSDAESimple)actx;
  TSDAESimple_Full *full = (TSDAESimple_Full*)tsdae->data;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  CHKERRQ(VecSet(F,0.0));
  CHKERRQ(VecScatterBegin(full->scatterU,UV,tsdae->U,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(full->scatterU,UV,tsdae->U,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterBegin(full->scatterV,UV,tsdae->V,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(full->scatterV,UV,tsdae->V,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ((*tsdae->f)(t,tsdae->U,tsdae->V,full->UF,tsdae->fctx));
  CHKERRQ(VecScatterBegin(full->scatterU,full->UF,F,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(full->scatterU,full->UF,F,INSERT_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

/*
   Defines the nonlinear function that is passed to the nonlinear solver

   \dot{U}
   F(U,V)

*/
PetscErrorCode TSDAESimple_Full_TSIFunction(TS ts,PetscReal t,Vec UV,Vec UVdot,Vec F,void *actx)
{
  TSDAESimple       tsdae = (TSDAESimple)actx;
  TSDAESimple_Full *full = (TSDAESimple_Full*)tsdae->data;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  CHKERRQ(VecCopy(UVdot,F));
  CHKERRQ(VecScatterBegin(full->scatterU,UV,tsdae->U,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(full->scatterU,UV,tsdae->U,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterBegin(full->scatterV,UV,tsdae->V,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(full->scatterV,UV,tsdae->V,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ((*tsdae->F)(t,tsdae->U,tsdae->V,full->VF,tsdae->Fctx));
  CHKERRQ(VecScatterBegin(full->scatterV,full->VF,F,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(full->scatterV,full->VF,F,INSERT_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleSolve_Full(TSDAESimple tsdae,Vec U)
{
  PetscErrorCode   ierr;
  TSDAESimple_Full *full = (TSDAESimple_Full*)tsdae->data;

  PetscFunctionBegin;
  CHKERRQ(VecSet(full->UV,1.0));
  CHKERRQ(VecScatterBegin(full->scatterU,U,full->UV,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(full->scatterU,U,full->UV,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(TSSolve(full->ts,full->UV));
  CHKERRQ(VecScatterBegin(full->scatterU,full->UV,U,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(full->scatterU,full->UV,U,INSERT_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleSetFromOptions_Full(PetscOptionItems *PetscOptionsObject,TSDAESimple tsdae)
{
  PetscErrorCode   ierr;
  TSDAESimple_Full *full = (TSDAESimple_Full*)tsdae->data;

  PetscFunctionBegin;
  CHKERRQ(TSSetFromOptions(full->ts));
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleDestroy_Full(TSDAESimple tsdae)
{
  PetscErrorCode   ierr;
  TSDAESimple_Full *full = (TSDAESimple_Full*)tsdae->data;

  PetscFunctionBegin;
  CHKERRQ(TSDestroy(&full->ts));
  CHKERRQ(VecDestroy(&full->UV));
  CHKERRQ(VecDestroy(&full->UF));
  CHKERRQ(VecDestroy(&full->VF));
  CHKERRQ(VecScatterDestroy(&full->scatterU));
  CHKERRQ(VecScatterDestroy(&full->scatterV));
  CHKERRQ(PetscFree(full));
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleSetUp_Full(TSDAESimple tsdae)
{
  PetscErrorCode   ierr;
  TSDAESimple_Full *full;
  Vec              tsrhs;
  PetscInt         nU,nV,UVstart;
  IS               is;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&full));
  tsdae->data = full;

  tsdae->setfromoptions = TSDAESimpleSetFromOptions_Full;
  tsdae->solve          = TSDAESimpleSolve_Full;
  tsdae->destroy        = TSDAESimpleDestroy_Full;

  CHKERRQ(TSCreate(tsdae->comm,&full->ts));
  CHKERRQ(TSSetProblemType(full->ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(full->ts,TSROSW));
  CHKERRQ(TSSetExactFinalTime(full->ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(VecDuplicate(tsdae->U,&full->UF));
  CHKERRQ(VecDuplicate(tsdae->V,&full->VF));

  CHKERRQ(VecGetLocalSize(tsdae->U,&nU));
  CHKERRQ(VecGetLocalSize(tsdae->V,&nV));
  CHKERRQ(VecCreateMPI(tsdae->comm,nU+nV,PETSC_DETERMINE,&tsrhs));
  CHKERRQ(VecDuplicate(tsrhs,&full->UV));

  CHKERRQ(VecGetOwnershipRange(tsrhs,&UVstart,NULL));
  CHKERRQ(ISCreateStride(tsdae->comm,nU,UVstart,1,&is));
  CHKERRQ(VecScatterCreate(tsdae->U,NULL,tsrhs,is,&full->scatterU));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(ISCreateStride(tsdae->comm,nV,UVstart+nU,1,&is));
  CHKERRQ(VecScatterCreate(tsdae->V,NULL,tsrhs,is,&full->scatterV));
  CHKERRQ(ISDestroy(&is));

  CHKERRQ(TSSetRHSFunction(full->ts,tsrhs,TSDAESimple_Full_TSRHSFunction,tsdae));
  CHKERRQ(TSSetIFunction(full->ts,NULL,TSDAESimple_Full_TSIFunction,tsdae));
  CHKERRQ(VecDestroy(&tsrhs));
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------*/

/*
   Simple example:   f(U,V) = U + V

*/
PetscErrorCode f(PetscReal t,Vec U,Vec V,Vec F,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  CHKERRQ(VecWAXPY(F,1.0,U,V));
  PetscFunctionReturn(0);
}

/*
   Simple example: F(U,V) = U - V

*/
PetscErrorCode F(PetscReal t,Vec U,Vec V,Vec F,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  CHKERRQ(VecWAXPY(F,-1.0,V,U));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  TSDAESimple    tsdae;
  Vec            U,V,Usolution;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(TSDAESimpleCreate(PETSC_COMM_WORLD,&tsdae));

  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&U));
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&V));
  CHKERRQ(TSDAESimpleSetRHSFunction(tsdae,U,f,NULL));
  CHKERRQ(TSDAESimpleSetIFunction(tsdae,V,F,NULL));

  CHKERRQ(VecDuplicate(U,&Usolution));
  CHKERRQ(VecSet(Usolution,1.0));

  /*  CHKERRQ(TSDAESimpleSetUp_Full(tsdae)); */
  CHKERRQ(TSDAESimpleSetUp_Reduced(tsdae));

  CHKERRQ(TSDAESimpleSetFromOptions(tsdae));
  CHKERRQ(TSDAESimpleSolve(tsdae,Usolution));
  CHKERRQ(TSDAESimpleDestroy(&tsdae));

  CHKERRQ(VecDestroy(&U));
  CHKERRQ(VecDestroy(&Usolution));
  CHKERRQ(VecDestroy(&V));
  ierr = PetscFinalize();
  return ierr;
}
