
#include <../src/ts/impls/semiexplicit/semiexplicit.h>

/* ----------------------------------------------------------------------------*/

/*                                                                                                                                         
      Integrates the system by integrating directly the entire DAE system                                                                  
*/

typedef struct {
  TS         ts;
  Vec        UV,UF,VF;
  VecScatter scatterU,scatterV;
} TS_DAESimple_Full;

#undef __FUNCT__
#define __FUNCT__ "TSDAESimple_Full_TSRHSFunction"
/*                                                                                                                                                      
   Defines the RHS function that is passed to the time-integrator.                                                                                      
                                                                                                                                                        
   f(U,V)                                                                                                                                               
   0                                                                                                                                                    
                                                                                                                                                        
*/
PetscErrorCode TSDAESimple_Full_TSRHSFunction(TS tsinner,PetscReal t,Vec UV,Vec F,void *actx)
{
  TS                ts = (TS)actx;
  TS_DAESimple      *tsdae = (TS_DAESimple*)ts->data;
  TS_DAESimple_Full *full = (TS_DAESimple_Full*)tsdae->data;
  PetscErrorCode    ierr;
  DM                dm;
  PetscErrorCode    (*rhsfunction)(PetscReal,Vec,Vec,Vec,void*);
  void              *rhsfunctionctx;

  PetscFunctionBegin;
  ierr = VecSet(F,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterU,UV,tsdae->U,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterU,UV,tsdae->U,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterV,UV,tsdae->V,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterV,UV,tsdae->V,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr   = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr   = DMTSGetDAESimpleRHSFunction(dm,&rhsfunction,&rhsfunctionctx);CHKERRQ(ierr);
  ierr   = (*rhsfunction)(t,tsdae->U,tsdae->V,full->UF,rhsfunctionctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterU,full->UF,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterU,full->UF,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDAESimple_Full_TSIFunction"
/*                                                                                                                                                      
   Defines the nonlinear function that is passed to the nonlinear solver                                                                                
                                                                                                                                                        
   \dot{U}                                                                                                                                              
   F(U,V)                                                                                                                                               
                                                                                                                                                        
*/
PetscErrorCode TSDAESimple_Full_TSIFunction(TS tsinner,PetscReal t,Vec UV,Vec UVdot,Vec F,void *actx)
{
  TS                ts = (TS)actx;
  TS_DAESimple      *tsdae = (TS_DAESimple*)ts->data;
  TS_DAESimple_Full *full = (TS_DAESimple_Full*)tsdae->data;
  PetscErrorCode    ierr;
  DM                dm;
  PetscErrorCode    (*ifunction)(PetscReal,Vec,Vec,Vec,void*);
  void              *ifunctionctx;

  PetscFunctionBegin;
  ierr = VecCopy(UVdot,F);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterU,UV,tsdae->U,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterU,UV,tsdae->U,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterV,UV,tsdae->V,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterV,UV,tsdae->V,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetDAESimpleIFunction(dm,&ifunction,&ifunctionctx);
  ierr = (*ifunction)(t,tsdae->U,tsdae->V,full->VF,ifunctionctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterV,full->VF,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterV,full->VF,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSReset_DAESimple_Full"
PetscErrorCode TSReset_DAESimple_Full(TS ts)
{
  TS_DAESimple *dae=(TS_DAESimple*)ts->data;
  TS_DAESimple_Full *full = (TS_DAESimple_Full*)dae->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSDestroy(&full->ts);CHKERRQ(ierr);
  ierr = VecDestroy(&full->UV);CHKERRQ(ierr);
  ierr = VecDestroy(&full->UF);CHKERRQ(ierr);
  ierr = VecDestroy(&full->VF);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&full->scatterU);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&full->scatterV);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_DAESimple_Full"
PetscErrorCode TSDestroy_DAESimple_Full(TS ts)
{
  TS_DAESimple    *dae = (TS_DAESimple*)ts->data;
  TS_DAESimple_Full *full = (TS_DAESimple_Full*)dae->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_DAESimple_Full(ts);CHKERRQ(ierr);
  ierr = PetscFree(full);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_DAESimple_Full"
PetscErrorCode TSSetFromOptions_DAESimple_Full(TS ts)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSolve_DAESimple_Full"
PetscErrorCode TSSolve_DAESimple_Full(TS ts)
{
  PetscErrorCode      ierr;
  TS_DAESimple        *tsdae = (TS_DAESimple*)ts->data;
  TS_DAESimple_Full   *full = (TS_DAESimple_Full*)tsdae->data;

  PetscFunctionBegin;
  ierr = VecScatterBegin(full->scatterU,tsdae->U,full->UV,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterU,tsdae->U,full->UV,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterV,tsdae->V,full->UV,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterV,tsdae->V,full->UV,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = TSSolve(full->ts,full->UV);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterU,full->UV,tsdae->U,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterU,full->UV,tsdae->U,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(full->scatterV,full->UV,tsdae->V,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(full->scatterV,full->UV,tsdae->V,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_DAESimple_Full"
PetscErrorCode TSSetUp_DAESimple_Full(TS ts)
{
  PetscErrorCode       ierr;
  TS_DAESimple         *tsdae = (TS_DAESimple*)ts->data;
  TS_DAESimple_Full    *full = (TS_DAESimple_Full*)tsdae->data;
  Vec                  tsrhs;
  PetscInt             nU,nV,UVstart;
  IS                   is;

  PetscFunctionBegin;
  ierr = TSCreate(PetscObjectComm((PetscObject)ts),&full->ts);CHKERRQ(ierr);
  ierr = TSSetOptionsPrefix(full->ts,"dae_full_");CHKERRQ(ierr);
  ierr = TSSetProblemType(full->ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(full->ts,TSROSW);CHKERRQ(ierr);
  ierr = VecDuplicate(tsdae->U,&full->UF);CHKERRQ(ierr);
  ierr = VecDuplicate(tsdae->V,&full->VF);CHKERRQ(ierr);

  ierr = VecGetLocalSize(tsdae->U,&nU);CHKERRQ(ierr);
  ierr = VecGetLocalSize(tsdae->V,&nV);CHKERRQ(ierr);
  ierr = VecCreateMPI(PetscObjectComm((PetscObject)ts),nU+nV,PETSC_DETERMINE,&tsrhs);CHKERRQ(ierr);
  ierr = VecDuplicate(tsrhs,&full->UV);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(tsrhs,&UVstart,NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(PetscObjectComm((PetscObject)ts),nU,UVstart,1,&is);CHKERRQ(ierr);
  ierr = VecScatterCreate(tsdae->U,NULL,tsrhs,is,&full->scatterU);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISCreateStride(PetscObjectComm((PetscObject)ts),nV,UVstart+nU,1,&is);CHKERRQ(ierr);
  ierr = VecScatterCreate(tsdae->V,NULL,tsrhs,is,&full->scatterV);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(full->ts,tsrhs,TSDAESimple_Full_TSRHSFunction,ts);CHKERRQ(ierr);
  ierr = TSSetIFunction(full->ts,NULL,TSDAESimple_Full_TSIFunction,ts);CHKERRQ(ierr);
  ierr = TSSetFromOptions(full->ts);CHKERRQ(ierr);
  ierr = VecDestroy(&tsrhs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
/*MC
      TSDAESimple - Semi-explicit DAE solver

   Level: advanced

.seealso:  TSCreate(), TS, TSSetType(), TSCN, TSBEULER, TSThetaSetTheta(), TSThetaSetEndpoint()

M*/
#undef __FUNCT__
#define __FUNCT__ "TSCreate_DAESimple_Full"
PETSC_EXTERN PetscErrorCode TSCreate_DAESimple_Full(TS ts)
{
  TS_DAESimple    *tsdae;
  TS_DAESimple_Full *full;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ts->ops->reset          = TSReset_DAESimple_Full;
  ts->ops->destroy        = TSDestroy_DAESimple_Full;
  ts->ops->setup          = TSSetUp_DAESimple_Full;
  ts->ops->setfromoptions = TSSetFromOptions_DAESimple;
  ts->ops->solve          = TSSolve_DAESimple;

  ierr = PetscNewLog(ts,TS_DAESimple,&tsdae);CHKERRQ(ierr);
  ts->data = (void*)tsdae;

  tsdae->setfromoptions = TSSetFromOptions_DAESimple_Full;
  tsdae->solve          = TSSolve_DAESimple_Full;
  tsdae->destroy        = TSDestroy_DAESimple_Full;

  ierr = PetscMalloc(sizeof(TS_DAESimple_Full),&full);CHKERRQ(ierr);
  tsdae->data = full;

  PetscFunctionReturn(0);
}
