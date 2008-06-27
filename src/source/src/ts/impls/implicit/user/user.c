#define PETSCTS_DLL

/*
  Code for general, user-defined timestepping with implicit schemes.
       
  F(t^{n+1},x^{n+1}) = G(t^{n-k},x^{n-k}), k>=0
                 t^0 = t_0
		 x^0 = x_0

*/
#include "include/private/tsimpl.h"                /*I   "petscts.h"   I*/

/* -------------------------------------------------------------------------- */

/* backward compatibility hacks */

#if (PETSC_VERSION_MAJOR    == 2 &&	\
     PETSC_VERSION_MINOR    == 3 &&	\
     PETSC_VERSION_SUBMINOR == 2 &&	\
     PETSC_VERSION_RELEASE  == 1)
#define SNESGetLinearSolveIterations SNESGetNumberLinearIterations
#endif

#if (PETSC_VERSION_MAJOR    == 2 &&	\
     PETSC_VERSION_MINOR    == 3 &&	\
     PETSC_VERSION_SUBMINOR == 3 &&	\
     PETSC_VERSION_RELEASE  == 1)
#endif

/* -------------------------------------------------------------------------- */

typedef struct {

  PetscReal  utime;     /* time level t^{n+1} */
  Vec        update;    /* work vector where new solution x^{n+1} is formed */
  Vec        vec_func;  /* work vector where F(t^{n+1},x^{n+1}) is stored */
  Vec        vec_rhs;   /* work vector where G(t^{n-k},x^{n-k}) is stored */

  /* user-provided routines and context for fine-grained timestepping management */
  PetscErrorCode (*setup)     (TS,PetscReal,Vec,void*);
  PetscErrorCode (*presolve)  (TS,PetscReal,Vec,void*);
  PetscErrorCode (*postsolve) (TS,PetscReal,Vec,void*);
  PetscErrorCode (*prestep)   (TS,PetscReal,Vec,void*);
  PetscErrorCode (*poststep)  (TS,PetscReal,Vec,void*); 
  PetscErrorCode (*start)     (TS,PetscReal,Vec,Vec,void*);
  PetscErrorCode (*step)      (TS,PetscReal,Vec,Vec,void*);
  PetscErrorCode (*verify)    (TS,PetscReal,Vec,Vec,void*,PetscTruth*,PetscReal*);
  void           *userP;      /* context passed to previous routines */
 
} TS_User;

/*------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "TSView_User"
static PetscErrorCode TSView_User(TS ts,PetscViewer viewer)
{
  /*TS_User        *tsuser = (TS_User*)ts->data;*/
  PetscTruth     isascii;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) PetscFunctionReturn(0);
  /* XXX write me !!*/
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

/* The nonlinear equation that is to be solved with SNES */
#undef __FUNCT__  
#define __FUNCT__ "TSUserFunction"
static PetscErrorCode TSUserFunction(SNES snes,Vec x,Vec f,void *ctx)
{
  TS             ts      = (TS) ctx;
  TS_User        *tsuser = (TS_User*)ts->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* apply user-provided function */
  ierr = TSComputeRHSFunction(ts,tsuser->utime,x,f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*  The Jacobian needed for SNES */
#undef __FUNCT__  
#define __FUNCT__ "TSUserJacobian"
static PetscErrorCode TSUserJacobian(SNES snes,Vec x,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  TS             ts      = (TS) ctx;
  TS_User        *tsuser = (TS_User*)ts->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* construct user's Jacobian */
  ierr = TSComputeRHSJacobian(ts,tsuser->utime,x,AA,BB,str);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

#undef __FUNCT__  
#define __FUNCT__ "TSUserSetUp"
static PetscErrorCode TSUserSetUp(TS ts,PetscReal t,Vec x)
{
  TS_User        *tsuser = (TS_User*)ts->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (tsuser->setup) { ierr = (*tsuser->setup)(ts,t,x,tsuser->userP);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSUserPreSolve"
static PetscErrorCode TSUserPreSolve(TS ts,PetscReal t,Vec x)
{
  TS_User        *tsuser = (TS_User*)ts->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (tsuser->presolve) { ierr = (*tsuser->presolve)(ts,t,x,tsuser->userP);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSUserPostSolve"
static PetscErrorCode TSUserPostSolve(TS ts,PetscReal t,Vec x)
{
  TS_User        *tsuser = (TS_User*)ts->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (tsuser->postsolve) { ierr = (*tsuser->postsolve)(ts,t,x,tsuser->userP);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSUserPreStep"
static PetscErrorCode TSUserPreStep(TS ts,PetscReal t,Vec x)
{
  TS_User        *tsuser = (TS_User*)ts->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (tsuser->prestep) { ierr = (*tsuser->prestep)(ts,t,x,tsuser->userP);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSUserPostStep"
static PetscErrorCode TSUserPostStep(TS ts,PetscReal t,Vec x)
{
  TS_User        *tsuser = (TS_User*)ts->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (tsuser->poststep) { ierr = (*tsuser->poststep)(ts,t,x,tsuser->userP);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSUserStartStep"
static PetscErrorCode TSUserStartStep(TS ts,PetscReal t,Vec rhs,Vec u)
{
  TS_User        *tsuser = (TS_User*)ts->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!tsuser->start) PetscFunctionReturn(0);
  ierr = (*tsuser->start)(ts,t,rhs,u,tsuser->userP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSUserStep"
static PetscErrorCode TSUserStep(TS ts,PetscReal t,Vec rhs,Vec u)
{
  TS_User        *tsuser = (TS_User*)ts->data;
  PetscInt       its=0,lits=0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (tsuser->step) {
    ierr = (*tsuser->step)(ts,t,rhs,u,tsuser->userP);CHKERRQ(ierr);
    goto finally;
  }
  if (ts->problem_type == TS_NONLINEAR) {
#if (PETSC_VERSION_MAJOR    == 2  && \
     PETSC_VERSION_MINOR    == 3  && \
    (PETSC_VERSION_SUBMINOR == 3  || \
     PETSC_VERSION_SUBMINOR == 2) && \
     PETSC_VERSION_RELEASE  == 1)
    if (!((PetscObject)ts->snes)->type_name) {
      ierr = SNESSetType(ts->snes,SNESLS);CHKERRQ(ierr);
    }
#endif
    ierr = SNESSolve(ts->snes,rhs,u);CHKERRQ(ierr);
  } if (ts->problem_type == TS_LINEAR) {
    MatStructure str = DIFFERENT_NONZERO_PATTERN;
    SETERRQ(1, "not yet implemented"); PetscFunctionReturn(1);
    ierr = KSPSetOperators(ts->ksp,ts->A,ts->B,str);CHKERRQ(ierr);
    ierr = KSPSolve(ts->ksp,rhs,u);CHKERRQ(ierr);
  }
 finally:
  if (ts->problem_type == TS_NONLINEAR) {
    ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
    ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
  } 
  if (ts->problem_type == TS_LINEAR) {
    ierr = KSPGetIterationNumber(ts->ksp,&lits);CHKERRQ(ierr);
  }  
  ts->nonlinear_its += its; ts->linear_its += lits;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSUserVerifyStep"
static PetscErrorCode TSUserVerifyStep(TS ts,PetscReal t,Vec rhs,Vec u,PetscTruth *ok,PetscReal *dt)
{
  TS_User        *tsuser = (TS_User*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *ok = PETSC_TRUE;
  *dt = ts->time_step;
  if (!tsuser->verify) PetscFunctionReturn(0);
  ierr = (*tsuser->verify)(ts,t,rhs,u,tsuser->userP,ok,dt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_User"
static PetscErrorCode TSSetUp_User(TS ts)
{
  TS_User        *tsuser = (TS_User*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* check problem type, currently only for nonlinear */
  if (ts->problem_type == TS_NONLINEAR) { /* setup nonlinear problem */
    /* nothing to do at this point yet */
  } else if (ts->problem_type == TS_LINEAR) { /* setup linear problem */
    SETERRQ(PETSC_ERR_SUP,"Only for nonlinear problems");
  } else {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"No such problem type");
  }
  /* create work vector for solution */
  if (tsuser->update == PETSC_NULL) {
    ierr = VecDuplicate(ts->vec_sol,&tsuser->update);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(ts,tsuser->update);CHKERRQ(ierr);
  }
  /* create work vector for function evaluation */
  if (tsuser->vec_func == PETSC_NULL) {
    ierr = PetscObjectQuery((PetscObject)ts,"__rhs_funcvec__",(PetscObject *)&tsuser->vec_func);CHKERRQ(ierr);
    if (tsuser->vec_func) { ierr = PetscObjectReference((PetscObject)tsuser->vec_func);CHKERRQ(ierr); }
  }
  if (tsuser->vec_func == PETSC_NULL) {
    ierr = VecDuplicate(ts->vec_sol,&tsuser->vec_func);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(ts,tsuser->vec_func);CHKERRQ(ierr);
  }
  /* setup inner solvers */
  if (ts->problem_type == TS_NONLINEAR) { /* setup nonlinear problem */
    ierr = SNESSetFunction(ts->snes,tsuser->vec_func,TSUserFunction,ts);CHKERRQ(ierr);
    ierr = SNESSetJacobian(ts->snes,ts->A,ts->B,TSUserJacobian,ts);CHKERRQ(ierr);
  } else if (ts->problem_type == TS_LINEAR) {  /* setup linear problem */
    SETERRQ(PETSC_ERR_SUP,"Only for nonlinear problems");
  }
  /* call user-provided setup function */
  ierr = TSUserSetUp(ts,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSolve_User"
static PetscErrorCode TSSolve_User(TS ts,PetscInt *steps,PetscReal *ptime)
{
  TS_User        *tsuser = (TS_User*)ts->data;
  PetscInt       i,j;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  *steps = -ts->steps;
  *ptime = ts->ptime;
  
  /* call presolve routine */
  ierr = TSUserPreSolve(ts,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
  ierr = VecCopy(ts->vec_sol,tsuser->update);CHKERRQ(ierr);
  /* monitor solution, only if step counter is zero*/
  if (ts->steps == 0) {
    ierr = TSMonitor(ts,ts->steps,ts->ptime,tsuser->update);CHKERRQ(ierr);
  }
  for (i=0; i<ts->max_steps && ts->ptime<ts->max_time; i++) {
    PetscTruth  stepok = PETSC_TRUE;
    PetscReal   nextdt = ts->time_step;
    /* call prestep routine, only once per time step */
    /* update vector already have the previous solution */
    ierr = TSUserPreStep(ts,ts->ptime,tsuser->update);CHKERRQ(ierr); 
    for (j=0; j<10; j++) { /* XXX "10" should be setteable */
      /* for j>0 update vector lost the previous solution, restore it */
      if (j > 0) { ierr = VecCopy(ts->vec_sol,tsuser->update);CHKERRQ(ierr); }
      /* initialize time and time step */
      tsuser->utime = ts->ptime + nextdt;
      ts->time_step = nextdt;
      /* compute rhs an initial guess for step problem */
      ierr = TSUserStartStep(ts,tsuser->utime,tsuser->vec_rhs,tsuser->update);CHKERRQ(ierr);
      /* solve step problem */
      ierr = TSUserStep(ts,tsuser->utime,tsuser->vec_rhs,tsuser->update);CHKERRQ(ierr);
      /* verify step, it can be accepted/rejected, new time step is computed  */
      ierr = TSUserVerifyStep(ts,tsuser->utime,tsuser->vec_rhs,tsuser->update,&stepok,&nextdt);CHKERRQ(ierr);
      if (stepok) break;
    } 
    /* XXX should generate error if step is not OK */
    ts->time_step = tsuser->utime - ts->ptime;
    ierr = TSUserPostStep(ts,tsuser->utime,tsuser->update);CHKERRQ(ierr);
    ierr = TSMonitor(ts,ts->steps+1,tsuser->utime,tsuser->update);CHKERRQ(ierr);
    /* update solution, time, time step, and step counter */
    ierr = VecCopy(tsuser->update,ts->vec_sol);CHKERRQ(ierr);
    ts->ptime = tsuser->utime;
    ts->time_step = nextdt;
    ts->steps++;
  }
  /* call postsolve routine */
  ierr = TSUserPostSolve(ts,ts->ptime,ts->vec_sol);CHKERRQ(ierr);

  *steps += ts->steps;
  *ptime  = ts->ptime;


  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

#undef  PetscTryMethod
#define PetscTryMethod(obj,A,B,C) \
  0;{ PetscErrorCode (*f)B, __ierr; \
    __ierr = PetscObjectQueryFunction((PetscObject)obj,A,(PetscVoidStarFunction)&f);CHKERRQ(__ierr); \
    if (f) {__ierr = (*f)C;CHKERRQ(__ierr);}\
  }
#undef  PetscUseMethod
#define PetscUseMethod(obj,A,B,C) \
  0;{ PetscErrorCode (*f)B, __ierr; \
    __ierr = PetscObjectQueryFunction((PetscObject)obj,A,(PetscVoidStarFunction)&f);CHKERRQ(__ierr); \
    if (f) {__ierr = (*f)C;CHKERRQ(__ierr);}\
    else {SETERRQ1(PETSC_ERR_SUP,"Cannot locate function %s in object",A);} \
  }

#undef __FUNCT__  
#define __FUNCT__ "TSUserSetUserFunctions"
PetscErrorCode PETSCTS_DLLEXPORT TSUserSetUserFunctions(
  TS ts,
  PetscErrorCode (*setup)     (TS,PetscReal,Vec,void*),
  PetscErrorCode (*presolve)  (TS,PetscReal,Vec,void*),
  PetscErrorCode (*postsolve) (TS,PetscReal,Vec,void*),
  PetscErrorCode (*prestep)   (TS,PetscReal,Vec,void*),
  PetscErrorCode (*poststep)  (TS,PetscReal,Vec,void*),
  PetscErrorCode (*start)     (TS,PetscReal,Vec,Vec,void*),
  PetscErrorCode (*step)      (TS,PetscReal,Vec,Vec,void*),
  PetscErrorCode (*verify)    (TS,PetscReal,Vec,Vec,void*,PetscTruth*,PetscReal*),
  void            *userP)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE,1);
  ierr = PetscTryMethod(ts,
			"TSUserSetUserFunctions_C",
			(TS,
			 PetscErrorCode (*)(TS,PetscReal,Vec,void*),
			 PetscErrorCode (*)(TS,PetscReal,Vec,void*),
			 PetscErrorCode (*)(TS,PetscReal,Vec,void*),
			 PetscErrorCode (*)(TS,PetscReal,Vec,void*),
			 PetscErrorCode (*)(TS,PetscReal,Vec,void*),
			 PetscErrorCode (*)(TS,PetscReal,Vec,Vec,void*),
			 PetscErrorCode (*)(TS,PetscReal,Vec,Vec,void*),
			 PetscErrorCode (*)(TS,PetscReal,Vec,Vec,void*,PetscTruth*,PetscReal*),
			 void*),
			(ts,
			 setup,
			 presolve,
			 postsolve,
			 prestep,
			 poststep,
			 start,
			 step,
			 verify,
			 userP));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSUserGetUserFunctions"
PetscErrorCode PETSCTS_DLLEXPORT TSUserGetUserFunctions(
  TS ts,
  PetscErrorCode (**setup)     (TS,PetscReal,Vec,void*),
  PetscErrorCode (**presolve)  (TS,PetscReal,Vec,void*),
  PetscErrorCode (**postsolve) (TS,PetscReal,Vec,void*),
  PetscErrorCode (**prestep)   (TS,PetscReal,Vec,void*),
  PetscErrorCode (**poststep)  (TS,PetscReal,Vec,void*),
  PetscErrorCode (**start)     (TS,PetscReal,Vec,Vec,void*),
  PetscErrorCode (**step)      (TS,PetscReal,Vec,Vec,void*),
  PetscErrorCode (**verify)    (TS,PetscReal,Vec,Vec,void*,PetscTruth*,PetscReal*),
  void            **userP)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE,1);
  ierr = PetscUseMethod(ts, 
			"TSUserGetUserFunctions_C",
			(TS,
			 PetscErrorCode (**)(TS,PetscReal,Vec,void*),
			 PetscErrorCode (**)(TS,PetscReal,Vec,void*),
			 PetscErrorCode (**)(TS,PetscReal,Vec,void*),
			 PetscErrorCode (**)(TS,PetscReal,Vec,void*),
			 PetscErrorCode (**)(TS,PetscReal,Vec,void*),
			 PetscErrorCode (**)(TS,PetscReal,Vec,Vec,void*),
			 PetscErrorCode (**)(TS,PetscReal,Vec,Vec,void*),
			 PetscErrorCode (**)(TS,PetscReal,Vec,Vec,void*,PetscTruth*,PetscReal*),
			 void*),
			(ts,
			 setup,
			 presolve,
			 postsolve,
			 prestep,
			 poststep,
			 start,
			 step,
			 verify,
			 userP));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSUserSetUserFunctions_User"
PetscErrorCode PETSCTS_DLLEXPORT TSUserSetUserFunctions_User(
  TS ts,
  PetscErrorCode (*setup)     (TS,PetscReal,Vec,void*),
  PetscErrorCode (*presolve)  (TS,PetscReal,Vec,void*),
  PetscErrorCode (*postsolve) (TS,PetscReal,Vec,void*),
  PetscErrorCode (*prestep)   (TS,PetscReal,Vec,void*),
  PetscErrorCode (*poststep)  (TS,PetscReal,Vec,void*),
  PetscErrorCode (*start)     (TS,PetscReal,Vec,Vec,void*),
  PetscErrorCode (*step)      (TS,PetscReal,Vec,Vec,void*),
  PetscErrorCode (*verify)    (TS,PetscReal,Vec,Vec,void*,PetscTruth*,PetscReal*),
  void            *userP)
{
  TS_User *tsuser = (TS_User*)ts->data;
  PetscFunctionBegin;
  tsuser->setup     = setup;
  tsuser->presolve  = presolve;
  tsuser->postsolve = postsolve;
  tsuser->prestep   = prestep;
  tsuser->poststep  = poststep;
  tsuser->start     = start;
  tsuser->step      = step;
  tsuser->verify    = verify;
  tsuser->userP     = userP;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSUserGetUserFunctions_User"
PetscErrorCode PETSCTS_DLLEXPORT TSUserGetUserFunctions_User(
  TS ts,
  PetscErrorCode (**setup)     (TS,PetscReal,Vec,void*),
  PetscErrorCode (**presolve)  (TS,PetscReal,Vec,void*),
  PetscErrorCode (**postsolve) (TS,PetscReal,Vec,void*),
  PetscErrorCode (**prestep)   (TS,PetscReal,Vec,void*),
  PetscErrorCode (**poststep)  (TS,PetscReal,Vec,void*),
  PetscErrorCode (**start)     (TS,PetscReal,Vec,Vec,void*),
  PetscErrorCode (**step)      (TS,PetscReal,Vec,Vec,void*),
  PetscErrorCode (**verify)    (TS,PetscReal,Vec,Vec,void*,PetscTruth*,PetscReal*),
  void            **userP)
{
  TS_User *tsuser = (TS_User*)ts->data;
  PetscFunctionBegin;
  if(setup)     *setup     = tsuser->setup;
  if(presolve)  *presolve  = tsuser->presolve;
  if(postsolve) *postsolve = tsuser->postsolve;
  if(prestep)   *prestep   = tsuser->prestep;  
  if(poststep)  *poststep  = tsuser->poststep;
  if(start)     *start     = tsuser->start;
  if(step)      *step      = tsuser->step;
  if(verify)    *verify    = tsuser->verify;
  if(userP)     *userP     = tsuser->userP;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "TSDestroy_User"
static PetscErrorCode TSDestroy_User(TS ts)
{
  TS_User      *tsuser = (TS_User*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tsuser->update)   {ierr = VecDestroy(tsuser->update);CHKERRQ(ierr);}
  if (tsuser->vec_func) {ierr = VecDestroy(tsuser->vec_func);CHKERRQ(ierr);}
  if (tsuser->vec_rhs)  {ierr = VecDestroy(tsuser->vec_rhs);CHKERRQ(ierr);}
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts, "TSUserSetUserFunctions_C",
					   "", PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSUserGetUserFunctions_C",
					   "", PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscFree(tsuser);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*MC
      TS_USER - General ODE solver for nonlinear problems

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TS_BEULER, TS_PSEUDO

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSCreate_User"
PetscErrorCode PETSCTS_DLLEXPORT TSCreate_User(TS ts)
{
  TS_User        *tsuser;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ts->ops->destroy         = TSDestroy_User;
  ts->ops->view            = TSView_User;
  ts->ops->setup           = TSSetUp_User;
  ts->ops->step            = TSSolve_User;
  
  ierr = PetscNew(TS_User,&tsuser);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ts,sizeof(TS_User));CHKERRQ(ierr);
  ts->data = (void*)tsuser;

  tsuser->update   = PETSC_NULL;
  tsuser->vec_func = PETSC_NULL;
  tsuser->vec_rhs  = PETSC_NULL;

  tsuser->setup     = 0;
  tsuser->presolve  = 0;
  tsuser->postsolve = 0;
  tsuser->prestep   = 0;
  tsuser->poststep  = 0;
  tsuser->start     = 0;
  tsuser->step      = 0;
  tsuser->verify    = 0;
  tsuser->userP     = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSUserSetUserFunctions_C",
					   "TSUserSetUserFunctions_User",
					   TSUserSetUserFunctions_User);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSUserGetUserFunctions_C",
					   "TSUserGetUserFunctions_User",
					   TSUserGetUserFunctions_User);CHKERRQ(ierr);

  ts->problem_type = TS_NONLINEAR;

  if (ts->problem_type == TS_NONLINEAR) {
    ierr = SNESCreate(((PetscObject)ts)->comm,&ts->snes);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(ts,ts->snes);CHKERRQ(ierr);
  } else if (ts->problem_type == TS_LINEAR) {
    ierr = KSPCreate(((PetscObject)ts)->comm,&ts->ksp);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(ts,ts->ksp);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*------------------------------------------------------------*/
