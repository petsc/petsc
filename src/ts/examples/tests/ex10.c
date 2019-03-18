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
  ierr           = PetscNew(tsdae);CHKERRQ(ierr);
  (*tsdae)->comm = comm;
  PetscFunctionReturn(0);
}

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

PetscErrorCode TSDAESimpleDestroy(TSDAESimple *tsdae)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*(*tsdae)->destroy)(*tsdae);CHKERRQ(ierr);
  ierr = VecDestroy(&(*tsdae)->U);CHKERRQ(ierr);
  ierr = VecDestroy(&(*tsdae)->V);CHKERRQ(ierr);
  ierr = PetscFree(*tsdae);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleSolve(TSDAESimple tsdae,Vec Usolution)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*tsdae->solve)(tsdae,Usolution);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleSetFromOptions(TSDAESimple tsdae)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*tsdae->setfromoptions)(PetscOptionsObject,tsdae);CHKERRQ(ierr);
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
  ierr   = SNESSolve(red->snes,NULL,tsdae->V);CHKERRQ(ierr);
  ierr   = (*tsdae->f)(t,U,tsdae->V,F,tsdae->fctx);CHKERRQ(ierr);
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
  ierr = (*tsdae->F)(red->t,red->U,V,F,tsdae->Fctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode TSDAESimpleSolve_Reduced(TSDAESimple tsdae,Vec U)
{
  PetscErrorCode      ierr;
  TSDAESimple_Reduced *red = (TSDAESimple_Reduced*)tsdae->data;

  PetscFunctionBegin;
  ierr = TSSolve(red->ts,U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleSetFromOptions_Reduced(PetscOptionItems *PetscOptionsObject,TSDAESimple tsdae)
{
  PetscErrorCode      ierr;
  TSDAESimple_Reduced *red = (TSDAESimple_Reduced*)tsdae->data;

  PetscFunctionBegin;
  ierr = TSSetFromOptions(red->ts);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(red->snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleDestroy_Reduced(TSDAESimple tsdae)
{
  PetscErrorCode      ierr;
  TSDAESimple_Reduced *red = (TSDAESimple_Reduced*)tsdae->data;

  PetscFunctionBegin;
  ierr = TSDestroy(&red->ts);CHKERRQ(ierr);
  ierr = SNESDestroy(&red->snes);CHKERRQ(ierr);
  ierr = PetscFree(red);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleSetUp_Reduced(TSDAESimple tsdae)
{
  PetscErrorCode      ierr;
  TSDAESimple_Reduced *red;
  Vec                 tsrhs;

  PetscFunctionBegin;
  ierr = PetscNew(&red);CHKERRQ(ierr);
  tsdae->data = red;

  tsdae->setfromoptions = TSDAESimpleSetFromOptions_Reduced;
  tsdae->solve          = TSDAESimpleSolve_Reduced;
  tsdae->destroy        = TSDAESimpleDestroy_Reduced;

  ierr = TSCreate(tsdae->comm,&red->ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(red->ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(red->ts,TSEULER);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(red->ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = VecDuplicate(tsdae->U,&tsrhs);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(red->ts,tsrhs,TSDAESimple_Reduced_TSFunction,tsdae);CHKERRQ(ierr);
  ierr = VecDestroy(&tsrhs);CHKERRQ(ierr);

  ierr = SNESCreate(tsdae->comm,&red->snes);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(red->snes,"tsdaesimple_");CHKERRQ(ierr);
  ierr = SNESSetFunction(red->snes,NULL,TSDAESimple_Reduced_SNESFunction,tsdae);CHKERRQ(ierr);
  ierr = SNESSetJacobian(red->snes,NULL,NULL,SNESComputeJacobianDefault,tsdae);CHKERRQ(ierr);
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
  ierr = VecSet(F,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterU,UV,tsdae->U,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterU,UV,tsdae->U,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterV,UV,tsdae->V,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterV,UV,tsdae->V,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = (*tsdae->f)(t,tsdae->U,tsdae->V,full->UF,tsdae->fctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterU,full->UF,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterU,full->UF,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
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
  ierr = VecCopy(UVdot,F);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterU,UV,tsdae->U,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterU,UV,tsdae->U,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterV,UV,tsdae->V,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterV,UV,tsdae->V,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = (*tsdae->F)(t,tsdae->U,tsdae->V,full->VF,tsdae->Fctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterV,full->VF,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterV,full->VF,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode TSDAESimpleSolve_Full(TSDAESimple tsdae,Vec U)
{
  PetscErrorCode   ierr;
  TSDAESimple_Full *full = (TSDAESimple_Full*)tsdae->data;

  PetscFunctionBegin;
  ierr = VecSet(full->UV,1.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterU,U,full->UV,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterU,U,full->UV,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = TSSolve(full->ts,full->UV);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterU,full->UV,U,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterU,full->UV,U,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleSetFromOptions_Full(PetscOptionItems *PetscOptionsObject,TSDAESimple tsdae)
{
  PetscErrorCode   ierr;
  TSDAESimple_Full *full = (TSDAESimple_Full*)tsdae->data;

  PetscFunctionBegin;
  ierr = TSSetFromOptions(full->ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSDAESimpleDestroy_Full(TSDAESimple tsdae)
{
  PetscErrorCode   ierr;
  TSDAESimple_Full *full = (TSDAESimple_Full*)tsdae->data;

  PetscFunctionBegin;
  ierr = TSDestroy(&full->ts);CHKERRQ(ierr);
  ierr = VecDestroy(&full->UV);CHKERRQ(ierr);
  ierr = VecDestroy(&full->UF);CHKERRQ(ierr);
  ierr = VecDestroy(&full->VF);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&full->scatterU);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&full->scatterV);CHKERRQ(ierr);
  ierr = PetscFree(full);CHKERRQ(ierr);
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
  ierr = PetscNew(&full);CHKERRQ(ierr);
  tsdae->data = full;

  tsdae->setfromoptions = TSDAESimpleSetFromOptions_Full;
  tsdae->solve          = TSDAESimpleSolve_Full;
  tsdae->destroy        = TSDAESimpleDestroy_Full;

  ierr = TSCreate(tsdae->comm,&full->ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(full->ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(full->ts,TSROSW);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(full->ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = VecDuplicate(tsdae->U,&full->UF);CHKERRQ(ierr);
  ierr = VecDuplicate(tsdae->V,&full->VF);CHKERRQ(ierr);

  ierr = VecGetLocalSize(tsdae->U,&nU);CHKERRQ(ierr);
  ierr = VecGetLocalSize(tsdae->V,&nV);CHKERRQ(ierr);
  ierr = VecCreateMPI(tsdae->comm,nU+nV,PETSC_DETERMINE,&tsrhs);CHKERRQ(ierr);
  ierr = VecDuplicate(tsrhs,&full->UV);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(tsrhs,&UVstart,NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(tsdae->comm,nU,UVstart,1,&is);CHKERRQ(ierr);
  ierr = VecScatterCreate(tsdae->U,NULL,tsrhs,is,&full->scatterU);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISCreateStride(tsdae->comm,nV,UVstart+nU,1,&is);CHKERRQ(ierr);
  ierr = VecScatterCreate(tsdae->V,NULL,tsrhs,is,&full->scatterV);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  ierr = TSSetRHSFunction(full->ts,tsrhs,TSDAESimple_Full_TSRHSFunction,tsdae);CHKERRQ(ierr);
  ierr = TSSetIFunction(full->ts,NULL,TSDAESimple_Full_TSIFunction,tsdae);CHKERRQ(ierr);
  ierr = VecDestroy(&tsrhs);CHKERRQ(ierr);
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
  ierr = VecWAXPY(F,1.0,U,V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Simple example: F(U,V) = U - V

*/
PetscErrorCode F(PetscReal t,Vec U,Vec V,Vec F,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecWAXPY(F,-1.0,V,U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  TSDAESimple    tsdae;
  Vec            U,V,Usolution;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = TSDAESimpleCreate(PETSC_COMM_WORLD,&tsdae);CHKERRQ(ierr);

  ierr = VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&U);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&V);CHKERRQ(ierr);
  ierr = TSDAESimpleSetRHSFunction(tsdae,U,f,NULL);CHKERRQ(ierr);
  ierr = TSDAESimpleSetIFunction(tsdae,V,F,NULL);CHKERRQ(ierr);

  ierr = VecDuplicate(U,&Usolution);CHKERRQ(ierr);
  ierr = VecSet(Usolution,1.0);CHKERRQ(ierr);

  /*  ierr = TSDAESimpleSetUp_Full(tsdae);CHKERRQ(ierr); */
  ierr = TSDAESimpleSetUp_Reduced(tsdae);CHKERRQ(ierr);

  ierr = TSDAESimpleSetFromOptions(tsdae);CHKERRQ(ierr);
  ierr = TSDAESimpleSolve(tsdae,Usolution);CHKERRQ(ierr);
  ierr = TSDAESimpleDestroy(&tsdae);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&Usolution);CHKERRQ(ierr);
  ierr = VecDestroy(&V);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



