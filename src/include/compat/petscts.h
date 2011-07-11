#ifndef _COMPAT_PETSC_TS_H
#define _COMPAT_PETSC_TS_H

#include "private/tsimpl.h"

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define PetscTS_ERR_SUP(ts)                                                 \
  PetscFunctionBegin;                                                       \
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);                                \
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version"); \
  PetscFunctionReturn(PETSC_ERR_SUP);
#endif


#if PETSC_VERSION_(3,1,0)
#define TSCN    TSCRANK_NICHOLSON
#define TSRK    TSRUNGE_KUTTA
#define TSALPHA "alpha"
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__  
#define __FUNCT__ "TSSetDM"
static PetscErrorCode TSSetDM(TS ts,DM dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(dm,DM_CLASSID,2);
  ierr = PetscObjectCompose((PetscObject)ts, "__DM__",
			    (PetscObject)dm);CHKERRQ(ierr);
  if (ts->snes) { ierr = SNESSetDM(ts->snes,dm);CHKERRQ(ierr); }
  if (ts->ksp)  { ierr = KSPSetDM(ts->ksp,dm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "TSGetDM"
static PetscErrorCode TSGetDM(TS ts,DM *dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(dm,2);
  ierr = PetscObjectQuery((PetscObject)ts, "__DM__",(PetscObject*)dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__  
#define __FUNCT__ "TSReset"
static PetscErrorCode TSReset(TS ts)
{PetscTS_ERR_SUP(ts);}
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "TSSetSolution_Compat"
static PetscErrorCode
TSSetSolution_Compat(TS ts,Vec u)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(u,VEC_CLASSID,2);
  ierr = TSSetSolution(ts,u);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)ts,"__solvec__",(PetscObject)u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define TSSetSolution TSSetSolution_Compat
#undef __FUNCT__
#define __FUNCT__ "TSSolve_Compat"
static PetscErrorCode
TSSolve_Compat(TS ts,Vec x,PetscReal *t)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (x) PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  if (t) PetscValidPointer(t,3);
  if (x) {ierr = TSSetSolution(ts,x);CHKERRQ(ierr);}
  ierr = TSSolve(ts,x);CHKERRQ(ierr);
  if (t) {ierr = TSGetTime(ts,t);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
#undef  TSSolve
#define TSSolve TSSolve_Compat
#undef __FUNCT__
#define __FUNCT__ "TSStep_Compat"
static PetscErrorCode
TSStep_Compat(TS ts)
{
  PetscInt  n;
  PetscReal t;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSStep(ts,&n,&t);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef  TSStep
#define TSStep TSStep_Compat
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__  
#define __FUNCT__ "TSAlphaSetRadius"
static PetscErrorCode TSAlphaSetRadius(TS ts,PetscReal radius)
{
  PetscErrorCode ierr,(*f)(TS,PetscReal);
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSAlphaSetRadius_C",(void(**)(void))&f);CHKERRQ(ierr);
  if (f) {ierr = (*f)(ts,radius);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "TSAlphaSetParams"
static PetscErrorCode TSAlphaSetParams(TS ts,PetscReal alpha_m,PetscReal alpha_f,PetscReal gamma)
{
  PetscErrorCode ierr,(*f)(TS,PetscReal,PetscReal,PetscReal);
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSAlphaSetParams_C",(void(**)(void))&f);CHKERRQ(ierr);
  if (f) {ierr = (*f)(ts,alpha_m,alpha_f,gamma);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "TSAlphaGetParams"
static PetscErrorCode TSAlphaGetParams(TS ts,PetscReal *alpha_m,PetscReal *alpha_f,PetscReal *gamma)
{ 
  PetscErrorCode ierr,(*f)(TS,PetscReal*,PetscReal*,PetscReal*);
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if(alpha_m) PetscValidPointer(alpha_m,2);
  if(alpha_f) PetscValidPointer(alpha_f,3);
  if(gamma)   PetscValidPointer(gamma,4);
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSAlphaGetParams_C",(void(**)(void))&f);CHKERRQ(ierr);
  if (!f) SETERRQ1(PETSC_ERR_SUP,"TS type %s",((PetscObject)ts)->type_name);
  ierr = (*f)(ts,alpha_m,alpha_f,gamma);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if (PETSC_VERSION_(3,0,0))
#define TSEULER           TS_EULER
#define TSBEULER          TS_BEULER
#define TSPSEUDO          TS_PSEUDO
#define TSCN              TS_CRANK_NICHOLSON
#define TSSUNDIALS        TS_SUNDIALS
#define TSRK              TS_RUNGE_KUTTA
#define TSPYTHON          "python"
#define TSTHETA           "theta"
#define TSGL              "gl"
#define TSSSP             "ssp"
#define TSALPHA           "alpha"
#endif

#undef __FUNCT__
#define __FUNCT__ "TSSetUpFunction_Private"
static PetscErrorCode
TSSetUpFunction_Private(TS ts,Vec r)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (r) {
    const TSType   ttype;
    TSProblemType  ptype;
    SNES           snes = 0;
    PetscErrorCode (*ffun)(SNES,Vec,Vec,void*) = 0;
    void           *fctx = 0;
    ierr = TSGetType(ts,&ttype);CHKERRQ(ierr);
    ierr = TSGetProblemType(ts,&ptype);CHKERRQ(ierr);
    if (ttype && ptype == TS_NONLINEAR) {
      ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
      ierr = SNESGetFunction(snes,0,&ffun,&fctx);CHKERRQ(ierr);
      #if !(PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0))
      if (!ffun) { ffun = SNESTSFormFunction; fctx = ts; }
      #endif
      ierr = SNESSetFunction(snes,r,ffun,fctx);CHKERRQ(ierr);
    }
  }
  if (r) {
    Vec svec = 0;
    ierr = TSGetSolution(ts,&svec);CHKERRQ(ierr);
    if (!svec) {
      ierr = VecDuplicate(r,&svec);CHKERRQ(ierr);
      ierr = TSSetSolution(ts,svec);CHKERRQ(ierr);
      ierr = VecDestroy(svec);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#if (PETSC_VERSION_(3,1,0))

#undef __FUNCT__
#define __FUNCT__ "TSSetIFunction_Compat"
static PetscErrorCode
TSSetIFunction_Compat(TS ts,Vec r,TSIFunction fun,void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (r) PetscValidHeaderSpecific(r,VEC_CLASSID,2);
  ierr = TSSetIFunction(ts,fun,ctx);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)ts,"__funvec__",(PetscObject)r);CHKERRQ(ierr);
  ierr = TSSetUpFunction_Private(ts,r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define TSSetIFunction TSSetIFunction_Compat

#undef __FUNCT__
#define __FUNCT__ "TSGetIFunction_Compat"
static PetscErrorCode
TSGetIFunction_Compat(TS ts,Vec *f,TSIFunction *fun,void **ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (f) {ierr = PetscObjectQuery((PetscObject)ts,"__funvec__",(PetscObject*)f);CHKERRQ(ierr);}
  #if PETSC_VERSION_(3,0,0)
  if (fun) *fun = NULL;
  #else
  if (fun) *fun = ts->ops->ifunction;
  #endif
  if (ctx) *ctx = ts->funP;
  PetscFunctionReturn(0);
}
#define TSGetIFunction TSGetIFunction_Compat

#undef __FUNCT__
#define __FUNCT__ "TSComputeIFunction_Compat"
static PetscErrorCode
TSComputeIFunction_Compat(TS ts,PetscReal t,Vec X,Vec Xdot,Vec Y,PetscBool imex)
{return TSComputeIFunction(ts,t,X,Xdot,Y);}
#define TSComputeIFunction TSComputeIFunction_Compat

#undef __FUNCT__
#define __FUNCT__ "TSComputeIJacobian_Compat"
static PetscErrorCode
TSComputeIJacobian_Compat(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal shift,Mat *A,Mat *B,MatStructure *flg,PetscBool imex)
{return TSComputeIJacobian(ts,t,X,Xdot,shift,A,B,flg);}
#define TSComputeIJacobian TSComputeIJacobian_Compat

#endif

typedef PetscErrorCode (*TSRHSFunction)(TS,PetscReal,Vec,Vec,void*);
typedef PetscErrorCode (*TSRHSJacobian)(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);

#undef __FUNCT__
#define __FUNCT__ "TSSetRHSFunction_Compat"
static PetscErrorCode
TSSetRHSFunction_Compat(TS ts,Vec r,TSRHSFunction fun,void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (r) PetscValidHeaderSpecific(r,VEC_CLASSID,2);
  ierr = TSSetRHSFunction(ts,fun,ctx);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)ts,"__funvec__",(PetscObject)r);CHKERRQ(ierr);
  ierr = TSSetUpFunction_Private(ts,r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define TSSetRHSFunction TSSetRHSFunction_Compat

#undef __FUNCT__
#define __FUNCT__ "TSGetRHSFunction_Compat"
static PetscErrorCode
TSGetRHSFunction_Compat(TS ts,Vec *f,TSRHSFunction *fun,void **ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (f) {ierr = PetscObjectQuery((PetscObject)ts, "__funvec__", (PetscObject*)f);CHKERRQ(ierr);}
  if (fun) *fun = ts->ops->rhsfunction;
  if (ctx) *ctx = ts->funP;
  PetscFunctionReturn(0);
}
#define TSGetRHSFunction TSGetRHSFunction_Compat

#undef __FUNCT__
#define __FUNCT__ "TSGetRHSJacobian_Compat"
static PetscErrorCode
TSGetRHSJacobian_Compat(TS ts,Mat *A,Mat *B, TSRHSJacobian *jac,void **ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSGetRHSJacobian(ts,A,B,ctx);CHKERRQ(ierr);
  if (jac) *jac = ts->ops->rhsjacobian;
  PetscFunctionReturn(0);
}
#define TSGetRHSJacobian TSGetRHSJacobian_Compat


#if (PETSC_VERSION_(3,0,0))
typedef PetscErrorCode (*TSIFunction)(TS,PetscReal,Vec,Vec,Vec,void*);
typedef PetscErrorCode (*TSIJacobian)(TS,PetscReal,Vec,Vec,PetscReal,
                                      Mat*,Mat*,MatStructure*,void*);
#undef __FUNCT__
#define __FUNCT__ "TSSetIFunction"
static PetscErrorCode TSSetIFunction(TS ts,Vec r,TSIFunction f,void *ctx)
{PetscTS_ERR_SUP(ts);}
#undef __FUNCT__
#define __FUNCT__ "TSSetIJacobian"
static PetscErrorCode TSSetIJacobian(TS ts,Mat A,Mat B,TSIJacobian j,void *ctx)
{PetscTS_ERR_SUP(ts);}
#undef __FUNCT__
#define __FUNCT__ "TSComputeIFunction"
static PetscErrorCode TSComputeIFunction(TS ts,PetscReal t,Vec x,Vec Xdot,Vec f,PetscTruth imex)
{PetscTS_ERR_SUP(ts);}
#undef __FUNCT__
#define __FUNCT__ "TSComputeIJacobian"
static PetscErrorCode TSComputeIJacobian(TS ts,
                                         PetscReal t,Vec x,Vec Xdot,PetscReal a,
                                         Mat *A,Mat *B,MatStructure *flag,PetscTruth imex)
{PetscTS_ERR_SUP(ts);}
#undef __FUNCT__
#define __FUNCT__ "TSGetIFunction"
static PetscErrorCode
TSGetIFunction(TS ts,Vec *f,TSIFunction *fun,void **ctx)
{PetscTS_ERR_SUP(ts);}
#undef __FUNCT__
#define __FUNCT__ "TSGetIJacobian"
static PetscErrorCode TSGetIJacobian(TS ts,Mat *A,Mat *B,
                                     TSIJacobian *j,void **ctx)
{PetscTS_ERR_SUP(ts);}
#endif

#if (PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "TSThetaSetTheta"
static PetscErrorCode TSThetaSetTheta(TS ts,PetscReal theta)
{
  PetscErrorCode ierr,(*f)(TS,PetscReal);
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSThetaSetTheta_C",(void(**)(void))&f);CHKERRQ(ierr);
  if (f) {ierr = (*f)(ts,theta);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TSThetaGetTheta"
static PetscErrorCode TSThetaGetTheta(TS ts,PetscReal *theta)
{
  PetscErrorCode ierr,(*f)(TS,PetscReal*);
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidPointer(theta,2);
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSThetaGetTheta_C",(void(**)(void))&f);CHKERRQ(ierr);
  if (!f) SETERRQ1(PETSC_ERR_SUP,"TS type %s",((PetscObject)ts)->type_name);
  ierr = (*f)(ts,theta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#endif /* _COMPAT_PETSC_TS_H */
