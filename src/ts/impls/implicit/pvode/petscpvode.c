/*$Id: petscpvode.c,v 1.56 2000/04/09 04:39:14 bsmith Exp bsmith $*/

#include "petsc.h"
/*
    Provides a PETSc interface to PVODE. Alan Hindmarsh's parallel ODE
   solver.
*/

#include "src/ts/impls/implicit/pvode/petscpvode.h"  /*I "ts.h" I*/    

/*
      TSPrecond_PVode - function that we provide to PVODE to
                        evaluate the preconditioner.

    Contributed by: Liyang Xu

*/
#undef __FUNC__
#define __FUNC__ /*<a name=""></a>*/"TSPrecond_PVode"
int TSPrecond_PVode(integer N,real tn,N_Vector y,N_Vector fy,bool jok,
                    bool *jcurPtr,real _gamma,N_Vector ewt,real h,
                    real uround,long int *nfePtr,void *P_data,
                    N_Vector vtemp1,N_Vector vtemp2,N_Vector vtemp3)
{
  TS           ts = (TS) P_data;
  TS_PVode     *cvode = (TS_PVode*)ts->data;
  PC           pc = cvode->pc;
  int          ierr;
  Mat          Jac = ts->B;
  Vec          tmpy = cvode->w1;
  Scalar       one = 1.0,gm;
  MatStructure str = DIFFERENT_NONZERO_PATTERN;
  
  PetscFunctionBegin;
  /* This allows us to construct preconditioners in-place if we like */
  ierr = MatSetUnfactored(Jac);CHKERRQ(ierr);

  /*
       jok - TRUE means reuse current Jacobian else recompute Jacobian
  */
  if (jok) {
    ierr     = MatCopy(cvode->pmat,Jac,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    str      = SAME_NONZERO_PATTERN;
    *jcurPtr = FALSE;
  } else {
    /* make PETSc vector tmpy point to PVODE vector y */
    ierr = VecPlaceArray(tmpy,&N_VIth(y,0));CHKERRQ(ierr);

    /* compute the Jacobian */
    ierr = TSComputeRHSJacobian(ts,ts->ptime,tmpy,&Jac,&Jac,&str);CHKERRQ(ierr);

    /* copy the Jacobian matrix */
    if (!cvode->pmat) {
      ierr = MatDuplicate(Jac,MAT_COPY_VALUES,&cvode->pmat);CHKERRQ(ierr);
      PLogObjectParent(ts,cvode->pmat); 
    }
    ierr = MatCopy(Jac,cvode->pmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

    *jcurPtr = TRUE;
  }

  /* construct I-gamma*Jac  */
  gm   = -_gamma;
  ierr = MatScale(&gm,Jac);CHKERRQ(ierr);
  ierr = MatShift(&one,Jac);CHKERRQ(ierr);
  
  ierr = PCSetOperators(pc,Jac,Jac,str);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     TSPSolve_PVode -  routine that we provide to PVode that applies the preconditioner.
      
    Contributed by: Liyang Xu

*/    
#undef __FUNC__
#define __FUNC__ /*<a name=""></a>*/"TSPSolve_PVode"
int TSPSolve_PVode(integer N,real tn,N_Vector y,N_Vector fy,N_Vector vtemp,
                   real _gamma,N_Vector ewt,real delta,long int *nfePtr,
                   N_Vector r,int lr,void *P_data,N_Vector z)
{ 
  TS       ts = (TS) P_data;
  TS_PVode *cvode = (TS_PVode*)ts->data;
  PC       pc = cvode->pc;
  Vec      rr = cvode->w1,xx = cvode->w2;
  int      ierr;

  PetscFunctionBegin;
  /*
      Make the PETSc work vectors rr and xx point to the arrays in the PVODE vectors 
  */
  ierr = VecPlaceArray(rr,&N_VIth(r,0));CHKERRQ(ierr);
  ierr = VecPlaceArray(xx,&N_VIth(z,0));CHKERRQ(ierr);

  /* 
      Solve the Px=r and put the result in xx 
  */
  ierr = PCApply(pc,rr,xx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
        TSFunction_PVode - routine that we provide to PVode that applies the right hand side.
      
    Contributed by: Liyang Xu
*/  
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSFunction_PVode"
void TSFunction_PVode(int N,double t,N_Vector y,N_Vector ydot,void *ctx)
{
  TS        ts = (TS) ctx;
  TS_PVode *cvode = (TS_PVode*)ts->data;
  Vec       tmpx = cvode->w1,tmpy = cvode->w2;
  int       ierr;

  PetscFunctionBegin;
  /*
      Make the PETSc work vectors tmpx and tmpy point to the arrays in the PVODE vectors 
  */
  ierr = VecPlaceArray(tmpx,&N_VIth(y,0));CHKERRA(ierr);
  ierr = VecPlaceArray(tmpy,&N_VIth(ydot,0));CHKERRA(ierr);

  /* now compute the right hand side function */
  ierr = TSComputeRHSFunction(ts,t,tmpx,tmpy);CHKERRA(ierr);
}

/*
       TSStep_PVode_Nonlinear - Calls PVode to integrate the ODE.

    Contributed by: Liyang Xu
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSStep_PVode_Nonlinear"
/* 
    TSStep_PVode_Nonlinear - 
  
   steps - number of time steps
   time - time that integrater is  terminated. 

*/
int TSStep_PVode_Nonlinear(TS ts,int *steps,double *time)
{
  TS_PVode  *cvode = (TS_PVode*)ts->data;
  Vec       sol = ts->vec_sol;
  int       ierr,i,max_steps = ts->max_steps,flag;
  double    t,tout;

  PetscFunctionBegin;
  /* initialize the number of steps */
  *steps = -ts->steps;
  ierr   = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  /* call CVSpgmr to use GMRES as the linear solver. */
  /* setup the ode integrator with the given preconditioner */
  CVSpgmr(cvode->mem,LEFT,cvode->gtype,cvode->restart,cvode->linear_tol,TSPrecond_PVode,TSPSolve_PVode,ts);

  tout = ts->max_time;
  for (i=0; i<max_steps; i++) {
    if (ts->ptime >= tout) break;
    ierr = VecGetArray(ts->vec_sol,&cvode->y->data);CHKERRQ(ierr);
    flag = CVode(cvode->mem,tout,cvode->y,&t,ONE_STEP);
    ierr = VecRestoreArray(ts->vec_sol,PETSC_NULL);CHKERRQ(ierr);
    if (flag != SUCCESS) SETERRQ(PETSC_ERR_LIB,0,"PVODE failed");	

    if (t > tout && cvode->exact_final_time) { 
      /* interpolate to final requested time */
      ierr = VecGetArray(ts->vec_sol,&cvode->y->data);CHKERRQ(ierr);
      flag = CVodeDky(cvode->mem,tout,0,cvode->y);
      ierr = VecRestoreArray(ts->vec_sol,PETSC_NULL);CHKERRQ(ierr);
      if (flag != SUCCESS) SETERRQ(PETSC_ERR_LIB,0,"PVODE interpolation to final time failed");	
      t = tout;
    }

    ts->time_step = t - ts->ptime;
    ts->ptime     = t;

    /*
       copy the solution from cvode->y to cvode->update and sol 
    */
    ierr = VecPlaceArray(cvode->w1,&N_VIth(cvode->y,0));CHKERRQ(ierr);
    ierr = VecCopy(cvode->w1,cvode->update);CHKERRQ(ierr);
    ierr = VecCopy(cvode->update,sol);CHKERRQ(ierr);
    
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,t,sol);CHKERRQ(ierr);
    ts->nonlinear_its = cvode->iopt[NNI];
    ts->linear_its    = cvode->iopt[SPGMR_NLI];
  }

  *steps           += ts->steps;
  *time             = t;

  PetscFunctionReturn(0);
}

/*

    Contributed by: Liyang Xu
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSDestroy_PVode"
int TSDestroy_PVode(TS ts)
{
  TS_PVode *cvode = (TS_PVode*)ts->data;
  int       ierr;

  PetscFunctionBegin;
  if (cvode->pmat)   {ierr = MatDestroy(cvode->pmat);CHKERRQ(ierr);}
  if (cvode->pc)     {ierr = PCDestroy(cvode->pc);CHKERRQ(ierr);}
  if (cvode->update) {ierr = VecDestroy(cvode->update);CHKERRQ(ierr);}
  if (cvode->func)   {ierr = VecDestroy(cvode->func);CHKERRQ(ierr);}
  if (cvode->rhs)    {ierr = VecDestroy(cvode->rhs);CHKERRQ(ierr);}
  if (cvode->w1)     {ierr = VecDestroy(cvode->w1);CHKERRQ(ierr);}
  if (cvode->w2)     {ierr = VecDestroy(cvode->w2);CHKERRQ(ierr);}
  ierr = PetscFree(cvode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*

    Contributed by: Liyang Xu
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSSetUp_PVode_Nonlinear"
int TSSetUp_PVode_Nonlinear(TS ts)
{
  TS_PVode    *cvode = (TS_PVode*)ts->data;
  int         ierr,M,locsize;
  machEnvType machEnv;

  PetscFunctionBegin;
  /* get the vector size */
  ierr = VecGetSize(ts->vec_sol,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(ts->vec_sol,&locsize);CHKERRQ(ierr);

  /* allocate the memory for machEnv */
  machEnv = PVInitMPI(ts->comm,locsize,M); 

  /* allocate the memory for N_Vec y */
  cvode->y         = N_VNew(M,machEnv); 
  ierr = VecGetArray(ts->vec_sol,&cvode->y->data);CHKERRQ(ierr);
  ierr = VecRestoreArray(ts->vec_sol,PETSC_NULL);CHKERRQ(ierr);

  /* initializing vector update and func */
  ierr = VecDuplicate(ts->vec_sol,&cvode->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&cvode->func);CHKERRQ(ierr);  
  PLogObjectParent(ts,cvode->update);
  PLogObjectParent(ts,cvode->func);

  /* 
      Create work vectors for the TSPSolve_PVode() routine. Note these are
    allocated with zero space arrays because the actual array space is provided 
    by PVode and set using VecPlaceArray().
  */
  ierr = VecCreateMPIWithArray(ts->comm,locsize,PETSC_DECIDE,0,&cvode->w1);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(ts->comm,locsize,PETSC_DECIDE,0,&cvode->w2);CHKERRQ(ierr);
  PLogObjectParent(ts,cvode->w1);
  PLogObjectParent(ts,cvode->w2);

  ierr = PCSetVector(cvode->pc,ts->vec_sol);CHKERRQ(ierr);

  /* allocate memory for PVode */
  ierr = VecGetArray(ts->vec_sol,&cvode->y->data);CHKERRQ(ierr);
  cvode->mem = CVodeMalloc(M,TSFunction_PVode,ts->ptime,cvode->y,cvode->cvode_type,
                           NEWTON,SS,&cvode->reltol,&cvode->abstol,ts,NULL,FALSE,cvode->iopt,
                           cvode->ropt,machEnv);CHKPTRQ(cvode->mem);
  ierr = VecRestoreArray(ts->vec_sol,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*

    Contributed by: Liyang Xu
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSSetFromOptions_PVode_Nonlinear"
int TSSetFromOptions_PVode_Nonlinear(TS ts)
{
  TS_PVode   *cvode = (TS_PVode*)ts->data;
  int        ierr,restart;
  char       method[128];
  double     aabs = PETSC_DECIDE,rel = PETSC_DECIDE,ltol;
  PetscTruth flag;

  PetscFunctionBegin;

  ierr = OptionsGetString(PETSC_NULL,"-ts_pvode_type",method,127,&flag);CHKERRQ(ierr);
  if (flag) {
    PetscTruth isbdf,isadams;

    ierr = PetscStrcmp(method,"bdf",&isbdf);CHKERRQ(ierr);
    ierr = PetscStrcmp(method,"adams",&isadams);CHKERRQ(ierr);
    if (isbdf) {
      ierr = TSPVodeSetType(ts,PVODE_BDF);CHKERRQ(ierr);
    } else if (isadams) {
      ierr = TSPVodeSetType(ts,PVODE_ADAMS);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown PVode type.\n");
    }
  }
  ierr = OptionsGetString(PETSC_NULL,"-ts_pvode_gramschmidt_type",method,127,&flag);CHKERRQ(ierr);
  if (flag) {
    PetscTruth ismodified,isunmodified;

    ierr = PetscStrcmp(method,"modified",&ismodified);CHKERRQ(ierr);
    ierr = PetscStrcmp(method,"unmodified",&isunmodified);CHKERRQ(ierr);
    if (ismodified) {
      ierr = TSPVodeSetGramSchmidtType(ts,PVODE_MODIFIED_GS);CHKERRQ(ierr);
    } else if (isunmodified) {
      ierr = TSPVodeSetGramSchmidtType(ts,PVODE_UNMODIFIED_GS);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown PVode Gram-Schmidt orthogonalization type \n");
    }
  }
  ierr = OptionsGetDouble(PETSC_NULL,"-ts_pvode_atol",&aabs,PETSC_NULL);CHKERRQ(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-ts_pvode_rtol",&rel,PETSC_NULL);CHKERRQ(ierr);
  ierr = TSPVodeSetTolerance(ts,aabs,rel);CHKERRQ(ierr);

  ierr = OptionsGetDouble(PETSC_NULL,"-ts_pvode_linear_tolerance",&ltol,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = TSPVodeSetLinearTolerance(ts,ltol);CHKERRQ(ierr);
  }

  ierr = OptionsGetInt(PETSC_NULL,"-ts_pvode_gmres_restart",&restart,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = TSPVodeSetGMRESRestart(ts,restart);CHKERRQ(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-ts_pvode_not_exact_final_time",&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = TSPVodeSetExactFinalTime(ts,PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = PCSetFromOptions(cvode->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*

    Contributed by: Liyang Xu
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSPrintHelp_PVode" 
int TSPrintHelp_PVode(TS ts,char *p)
{
  int      ierr;
  TS_PVode *cvode = (TS_PVode*)ts->data;

  PetscFunctionBegin;
  ierr = (*PetscHelpPrintf)(ts->comm," Options for TSPVODE integrater:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ts->comm," -ts_pvode_type <bdf,adams>: integration approach\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ts->comm," -ts_pvode_atol aabs: absolute tolerance of ODE solution\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ts->comm," -ts_pvode_rtol rel: relative tolerance of ODE solution\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ts->comm," -ts_pvode_gramschmidt_type <unmodified,modified>\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ts->comm," -ts_pvode_gmres_restart <restart_size> (also max. GMRES its)\n");CHKERRQ(ierr); 
  ierr = (*PetscHelpPrintf)(ts->comm," -ts_pvode_linear_tolerance <tol>\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ts->comm," -ts_pvode_not_exact_final_time\n");CHKERRQ(ierr); 

  ierr = PCPrintHelp(cvode->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*

    Contributed by: Liyang Xu
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSView_PVode" 
int TSView_PVode(TS ts,Viewer viewer)
{
  TS_PVode   *cvode = (TS_PVode*)ts->data;
  int        ierr;
  char       *type;
  PetscTruth isascii,isstring;

  PetscFunctionBegin;
  if (cvode->cvode_type == PVODE_ADAMS) {type = "Adams";}
  else                                  {type = "BDF: backward differentiation formula";}

  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,STRING_VIEWER,&isstring);CHKERRQ(ierr);
  if (isascii) {
    ierr = ViewerASCIIPrintf(viewer,"PVode integrater does not use SNES!\n");CHKERRQ(ierr); 
    ierr = ViewerASCIIPrintf(viewer,"PVode integrater type %s\n",type);CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"PVode abs tol %g rel tol %g\n",cvode->abstol,cvode->reltol);CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"PVode linear solver tolerance factor %g\n",cvode->linear_tol);CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"PVode GMRES max iterations (same as restart in PVODE) %d\n",cvode->restart);CHKERRQ(ierr);
    if (cvode->gtype == PVODE_MODIFIED_GS) {
      ierr = ViewerASCIIPrintf(viewer,"PVode using modified Gram-Schmidt for orthogonalization in GMRES\n");CHKERRQ(ierr);
    } else {
      ierr = ViewerASCIIPrintf(viewer,"PVode using unmodified (classical) Gram-Schmidt for orthogonalization in GMRES\n");CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = ViewerStringSPrintf(viewer,"Pvode type %s",type);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported by TS PVode",((PetscObject)viewer)->type_name);
  }
  ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PCView(cvode->pc,viewer);CHKERRQ(ierr);
  ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/* --------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNC__
#define __FUNC__ /*<a name=""></a>*/"TSPVodeSetType_Pvode"
int TSPVodeSetType_PVode(TS ts,TSPVodeType type)
{
  TS_PVode *cvode = (TS_PVode*)ts->data;
  
  PetscFunctionBegin;
  cvode->cvode_type = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__
#define __FUNC__ /*<a name=""></a>*/"TSPVodeSetGMRESRestart_PVode"
int TSPVodeSetGMRESRestart_PVode(TS ts,int restart)
{
  TS_PVode *cvode = (TS_PVode*)ts->data;
  
  PetscFunctionBegin;
  cvode->restart = restart;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__
#define __FUNC__ /*<a name=""></a>*/"TSPVodeSetLinearTolerance_PVode"
int TSPVodeSetLinearTolerance_PVode(TS ts,double tol)
{
  TS_PVode *cvode = (TS_PVode*)ts->data;
  
  PetscFunctionBegin;
  cvode->linear_tol = tol;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__
#define __FUNC__ /*<a name=""></a>*/"TSPVodeSetGramSchmidtType_PVode"
int TSPVodeSetGramSchmidtType_PVode(TS ts,TSPVodeGramSchmidtType type)
{
  TS_PVode *cvode = (TS_PVode*)ts->data;
  
  PetscFunctionBegin;
  cvode->gtype = type;

  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__
#define __FUNC__ /*<a name=""></a>*/"TSPVodeSetTolerance_PVode"
int TSPVodeSetTolerance_PVode(TS ts,double aabs,double rel)
{
  TS_PVode *cvode = (TS_PVode*)ts->data;
  
  PetscFunctionBegin;
  if (aabs != PETSC_DECIDE) cvode->abstol = aabs;
  if (rel != PETSC_DECIDE)  cvode->reltol = rel;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSPVodeGetPC_PVode"
int TSPVodeGetPC_PVode(TS ts,PC *pc)
{ 
  TS_PVode *cvode = (TS_PVode*)ts->data;

  PetscFunctionBegin;
  *pc = cvode->pc;

  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSPVodeGetIterations_PVode"
int TSPVodeGetIterations_PVode(TS ts,int *nonlin,int *lin)
{
  TS_PVode *cvode = (TS_PVode*)ts->data;
  
  PetscFunctionBegin;
  if (nonlin) *nonlin = cvode->iopt[NNI];
  if (lin)    *lin    = cvode->iopt[SPGMR_NLI];
  PetscFunctionReturn(0);
}
EXTERN_C_END
  
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSPVodeSetExactFinalTime_PVode"
int TSPVodeSetExactFinalTime_PVode(TS ts,PetscTruth s)
{
  TS_PVode *cvode = (TS_PVode*)ts->data;
  
  PetscFunctionBegin;
  cvode->exact_final_time = s;
  PetscFunctionReturn(0);
}
EXTERN_C_END
/* -------------------------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ /*<a name=""></a>*/"TSPVodeGetIterations"
/*@C
   TSPVodeGetIterations - Gets the number of nonlinear and linear iterations used so far by PVode.

   Not Collective

   Input parameters:
.    ts     - the time-step context

   Output Parameters:
+   nonlin - number of nonlinear iterations
-   lin    - number of linear iterations

   Level: advanced

.keywords: non-linear iterations, linear iterations

.seealso: TSPVodeSetType(), TSPVodeSetGMRESRestart(),
          TSPVodeSetLinearTolerance(), TSPVodeSetGramSchmidtType(), TSPVodeSetTolerance(),
          TSPVodeGetIterations(), TSPVodeSetType(), TSPVodeSetGMRESRestart(),
          TSPVodeSetLinearTolerance(), TSPVodeSetTolerance(), TSPVodeGetPC(),
          TSPVodeSetExactFinalTime()

@*/
int TSPVodeGetIterations(TS ts,int *nonlin,int *lin)
{
  int ierr,(*f)(TS,int*,int*);
  
  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPVodeGetIterations_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,nonlin,lin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ /*<a name=""></a>*/"TSPVodeSetType"
/*@
   TSPVodeSetType - Sets the method that PVode will use for integration.

   Collective on TS

   Input parameters:
+    ts     - the time-step context
-    type   - one of  PVODE_ADAMS or PVODE_BDF

    Contributed by: Liyang Xu

   Level: intermediate

.keywords: Adams, backward differentiation formula

.seealso: TSPVodeGetIterations(),  TSPVodeSetGMRESRestart(),
          TSPVodeSetLinearTolerance(), TSPVodeSetGramSchmidtType(), TSPVodeSetTolerance(),
          TSPVodeGetIterations(), TSPVodeSetType(), TSPVodeSetGMRESRestart(),
          TSPVodeSetLinearTolerance(), TSPVodeSetTolerance(), TSPVodeGetPC(),
          TSPVodeSetExactFinalTime()
@*/
int TSPVodeSetType(TS ts,TSPVodeType type)
{
  int ierr,(*f)(TS,TSPVodeType);
  
  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPVodeSetType_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ /*<a name=""></a>*/"TSPVodeSetGMRESRestart"
/*@
   TSPVodeSetGMRESRestart - Sets the dimension of the Krylov space used by 
       GMRES in the linear solver in PVODE. PVODE DOES NOT use restarted GMRES so
       this is ALSO the maximum number of GMRES steps that will be used.

   Collective on TS

   Input parameters:
+    ts      - the time-step context
-    restart - number of direction vectors (the restart size).

   Level: advanced

.keywords: GMRES, restart

.seealso: TSPVodeGetIterations(), TSPVodeSetType(), 
          TSPVodeSetLinearTolerance(), TSPVodeSetGramSchmidtType(), TSPVodeSetTolerance(),
          TSPVodeGetIterations(), TSPVodeSetType(), TSPVodeSetGMRESRestart(),
          TSPVodeSetLinearTolerance(), TSPVodeSetTolerance(), TSPVodeGetPC(),
          TSPVodeSetExactFinalTime()

@*/
int TSPVodeSetGMRESRestart(TS ts,int restart)
{
  int ierr,(*f)(TS,int);  

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPVodeSetGMRESRestart_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,restart);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ /*<a name=""></a>*/"TSPVodeSetLinearTolerance"
/*@
   TSPVodeSetLinearTolerance - Sets the tolerance used to solve the linear
       system by PVODE.

   Collective on TS

   Input parameters:
+    ts     - the time-step context
-    tol    - the factor by which the tolerance on the nonlinear solver is
             multiplied to get the tolerance on the linear solver, .05 by default.

   Level: advanced

.keywords: GMRES, linear convergence tolerance, PVODE

.seealso: TSPVodeGetIterations(), TSPVodeSetType(), TSPVodeSetGMRESRestart(),
          TSPVodeSetGramSchmidtType(), TSPVodeSetTolerance(),
          TSPVodeGetIterations(), TSPVodeSetType(), TSPVodeSetGMRESRestart(),
          TSPVodeSetLinearTolerance(), TSPVodeSetTolerance(), TSPVodeGetPC(),
          TSPVodeSetExactFinalTime()

@*/
int TSPVodeSetLinearTolerance(TS ts,double tol)
{
  int ierr,(*f)(TS,double);  
  
  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPVodeSetLinearTolerance_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,tol);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ /*<a name=""></a>*/"TSPVodeSetGramSchmidtType"
/*@
   TSPVodeSetGramSchmidtType - Sets type of orthogonalization used
        in GMRES method by PVODE linear solver.

   Collective on TS

   Input parameters:
+    ts  - the time-step context
-    type - either PVODE_MODIFIED_GS or PVODE_CLASSICAL_GS

   Level: advanced

.keywords: PVode, orthogonalization

.seealso: TSPVodeGetIterations(), TSPVodeSetType(), TSPVodeSetGMRESRestart(),
          TSPVodeSetLinearTolerance(),  TSPVodeSetTolerance(),
          TSPVodeGetIterations(), TSPVodeSetType(), TSPVodeSetGMRESRestart(),
          TSPVodeSetLinearTolerance(), TSPVodeSetTolerance(), TSPVodeGetPC(),
          TSPVodeSetExactFinalTime()

@*/
int TSPVodeSetGramSchmidtType(TS ts,TSPVodeGramSchmidtType type)
{
  int ierr,(*f)(TS,TSPVodeGramSchmidtType);  
  
  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPVodeSetGramSchmidtType_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ /*<a name=""></a>*/"TSPVodeSetTolerance"
/*@
   TSPVodeSetTolerance - Sets the absolute and relative tolerance used by 
                         PVode for error control.

   Collective on TS

   Input parameters:
+    ts  - the time-step context
.    aabs - the absolute tolerance  
-    rel - the relative tolerance

    Contributed by: Liyang Xu

   Level: intermediate

.keywords: PVode, tolerance

.seealso: TSPVodeGetIterations(), TSPVodeSetType(), TSPVodeSetGMRESRestart(),
          TSPVodeSetLinearTolerance(), TSPVodeSetGramSchmidtType(), 
          TSPVodeGetIterations(), TSPVodeSetType(), TSPVodeSetGMRESRestart(),
          TSPVodeSetLinearTolerance(), TSPVodeSetTolerance(), TSPVodeGetPC(),
          TSPVodeSetExactFinalTime()

@*/
int TSPVodeSetTolerance(TS ts,double aabs,double rel)
{
  int ierr,(*f)(TS,double,double);  
  
  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPVodeSetTolerance_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,aabs,rel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSPVodeGetPC"
/*@
   TSPVodeGetPC - Extract the PC context from a time-step context for PVode.

   Input Parameter:
.    ts - the time-step context

   Output Parameter:
.    pc - the preconditioner context

   Level: advanced

    Contributed by: Liyang Xu

.seealso: TSPVodeGetIterations(), TSPVodeSetType(), TSPVodeSetGMRESRestart(),
          TSPVodeSetLinearTolerance(), TSPVodeSetGramSchmidtType(), TSPVodeSetTolerance(),
          TSPVodeGetIterations(), TSPVodeSetType(), TSPVodeSetGMRESRestart(),
          TSPVodeSetLinearTolerance(), TSPVodeSetTolerance()
@*/
int TSPVodeGetPC(TS ts,PC *pc)
{ 
  int ierr,(*f)(TS,PC *);  

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPVodeGetPC_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,pc);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,1,"TS must be of PVode type to extract the PC");
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSPVodeSetExactFinalTime"
/*@
   TSPVodeSetExactFinalTime - Determines if PVode interpolates solution to the 
      exact final time requested by the user or just returns it at the final time
      it computed. (Defaults to true).

   Input Parameter:
+   ts - the time-step context
-   ft - PETSC_TRUE if interpolates, else PETSC_FALSE

   Level: beginner

.seealso:TSPVodeGetIterations(), TSPVodeSetType(), TSPVodeSetGMRESRestart(),
          TSPVodeSetLinearTolerance(), TSPVodeSetGramSchmidtType(), TSPVodeSetTolerance(),
          TSPVodeGetIterations(), TSPVodeSetType(), TSPVodeSetGMRESRestart(),
          TSPVodeSetLinearTolerance(), TSPVodeSetTolerance(), TSPVodeGetPC() 
@*/
int TSPVodeSetExactFinalTime(TS ts,PetscTruth ft)
{ 
  int ierr,(*f)(TS,PetscTruth);  

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPVodeSetExactFinalTime_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,ft);CHKERRQ(ierr);
  } 

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/

/*

    Contributed by: Liyang Xu
*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSCreate_PVode"
int TSCreate_PVode(TS ts)
{
  TS_PVode *cvode;
  int      ierr;

  PetscFunctionBegin;
  ts->destroy         = TSDestroy_PVode;
  ts->printhelp       = TSPrintHelp_PVode;
  ts->view            = TSView_PVode;

  if (ts->problem_type != TS_NONLINEAR) {
    SETERRQ(PETSC_ERR_SUP,0,"Only support for nonlinear problems");
  }
  ts->setup           = TSSetUp_PVode_Nonlinear;  
  ts->step            = TSStep_PVode_Nonlinear;
  ts->setfromoptions  = TSSetFromOptions_PVode_Nonlinear;

  cvode = PetscNew(TS_PVode);CHKPTRQ(cvode);
  ierr  = PetscMemzero(cvode,sizeof(TS_PVode));CHKERRQ(ierr);
  ierr  = PCCreate(ts->comm,&cvode->pc);CHKERRQ(ierr);
  PLogObjectParent(ts,cvode->pc);
  ts->data          = (void*)cvode;
  cvode->cvode_type = BDF;
  cvode->gtype      = PVODE_UNMODIFIED_GS;
  cvode->restart    = 5;
  cvode->linear_tol = .05;

  cvode->exact_final_time = PETSC_TRUE;

  /* set tolerance for PVode */
  cvode->abstol = 1e-6;
  cvode->reltol = 1e-6;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSPVodeSetType_C","TSPVodeSetType_PVode",
                    (void*)TSPVodeSetType_PVode);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSPVodeSetGMRESRestart_C","TSPVodeSetGMRESRestart_PVode",
                    (void*)TSPVodeSetGMRESRestart_PVode);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSPVodeSetLinearTolerance_C",
                    "TSPVodeSetLinearTolerance_PVode",
                    (void*)TSPVodeSetLinearTolerance_PVode);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSPVodeSetGramSchmidtType_C",
                    "TSPVodeSetGramSchmidtType_PVode",
                    (void*)TSPVodeSetGramSchmidtType_PVode);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSPVodeSetTolerance_C","TSPVodeSetTolerance_PVode",
                    (void*)TSPVodeSetTolerance_PVode);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSPVodeGetPC_C","TSPVodeGetPC_PVode",
                    (void*)TSPVodeGetPC_PVode);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSPVodeGetIterations_C","TSPVodeGetIterations_PVode",
                    (void*)TSPVodeGetIterations_PVode);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSPVodeSetExactFinalTime_C","TSPVodeSetExactFinalTime_PVode",
                    (void*)TSPVodeSetExactFinalTime_PVode);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END










