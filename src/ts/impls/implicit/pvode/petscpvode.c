#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: petscpvode.c,v 1.8 1997/10/13 21:16:19 bsmith Exp bsmith $";
#endif

/*
    Provides a PETSc interface to PVODE. Alan Hindmarsh's parallel ODE
   solver.
*/

#if defined(HAVE_PVODE)  && !defined(__cplusplus)

#include "src/ts/impls/implicit/pvode/petscpvode.h"  /*I "ts.h" I*/    

/*
      TSPrecond_PVode is the function that we provide to PVODE to
   evaluate the preconditioner.
*/
#undef __FUNC__
#define __FUNC__ "TSPrecond_PVode"
static int TSPrecond_PVode(integer N, real tn, N_Vector y, 
                           N_Vector fy, bool jok,
                           bool *jcurPtr, real _gamma, N_Vector ewt, real h,
                           real uround, long int *nfePtr, void *P_data,
                           N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3)
{
  TS           ts = (TS) P_data;
  TS_PVode     *cvode = (TS_PVode*) ts->data;
  PC           pc = cvode->pc;
  int          ierr, locsize;

  Mat          Jac;
  Vec          tmpy = cvode->w1;
  Scalar       one = 1.0, gm;
  MatStructure str = SAME_NONZERO_PATTERN;

  /* get the local size of N_Vector y */
  locsize = N_VLOCLENGTH(y);

  /* Jac and u must be set before calling PCSetUp */
  ierr = MatCreate(PETSC_COMM_WORLD,N,N,&Jac); CHKERRQ(ierr);

  if (!cvode->pmat) {
    ierr = MatCreate(MPI_COMM_WORLD,N,N,&cvode->pmat); CHKERRQ(ierr);
  }

  if (jok) {
    /* jok=TRUE: copy the Jacobian to Pcond */
    ierr = MatCopy(cvode->pmat,Jac); CHKERRQ(ierr);

    *jcurPtr = FALSE;
  }
  else {
    /* jok = FALSE: generate the Jacobian and then copy it to Pcond */
    /* convert N_Vector y to petsc Vec tmpy */
    ierr = VecPlaceArray(tmpy,&N_VIth(y,0));CHKERRQ(ierr);

    /* recompute the Jacobian */
    ierr = (*ts->rhsjacobian)(ts,ts->ptime,tmpy,&Jac,&Jac,&str,ts->jacP);CHKERRQ(ierr);

    /* copy the Jacobian matrix */
    ierr = MatCopy(Jac, cvode->pmat); CHKERRQ(ierr);

    /* set the flag */
    *jcurPtr = TRUE;
  }

  /* construct I-gamma*Jac  */
  gm = -_gamma;
  ierr = MatScale(&gm,Jac); CHKERRQ(ierr);
  ierr = MatShift(&one,Jac); CHKERRQ(ierr);
  
  /* setup the preconditioner contex */
  ierr = PCSetOperators(pc,Jac,Jac,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PCSetVector(pc,ts->vec_sol);   CHKERRQ(ierr);

  ierr = PCSetUp(pc); CHKERRQ(ierr);

  return(0);
}

/*
  TSPSolve_PVode -  routine that we provide to PVode that applies the 
  preconditioner.
      
   ---------------------------------------------------------------------
*/    
#undef __FUNC__
#define __FUNC__ "TSPSolve_PVode"
static int TSPSolve_PVode(integer N, real tn, N_Vector y, 
                          N_Vector fy, N_Vector vtemp,
                          real _gamma, N_Vector ewt, real delta, long int *nfePtr,
                          N_Vector r, int lr, void *P_data, N_Vector z)
{ 
  TS       ts = (TS) P_data;
  TS_PVode *cvode = (TS_PVode*) ts->data;
  PC       pc = cvode->pc;
  Vec      rr = cvode->w1, xx = cvode->w2;
  int      ierr;

  /*
      Make the PETSc work vectors rr and xx point to the arrays in the PVODE vectors 
  */
  ierr = VecPlaceArray(rr,&N_VIth(r,0)); CHKERRQ(ierr);
  ierr = VecPlaceArray(xx,&N_VIth(z,0)); CHKERRQ(ierr);

  /* 
      Solve the Px=r and put the result in xx 
  */
  ierr = PCApply(pc,rr,xx); CHKERRQ(ierr);

  return 0;
}

/*
    TSPSolve_PVode - routine that we provide to PVode that applies the 
  right hand side.
      
   ---------------------------------------------------------------------
*/  
#undef __FUNC__  
#define __FUNC__ "TSFunction_PVode"
static void TSFunction_PVode(int N,double t,N_Vector y,N_Vector ydot,void *ctx)
{
  TS        ts = (TS) ctx;
  TS_PVode *cvode = (TS_PVode*) ts->data;
  Vec       tmpx = cvode->w1, tmpy = cvode->w2;
  int       ierr;

  /*
      Make the PETSc work vectors tmpx and tmpy point to the arrays in the PVODE vectors 
  */
  ierr = VecPlaceArray(tmpx,&N_VIth(y,0)); CHKERRA(ierr);
  ierr = VecPlaceArray(tmpy,&N_VIth(ydot,0)); CHKERRA(ierr);

  /* now compute the right hand side function */
  ierr = TSComputeRHSFunction(ts,t,tmpx,tmpy); CHKERRA(ierr);
}

/*
  TSStep_PVode_Nonlinear - Calls PVode to integrate the ODE.

   ----------------------------------------------------------------------
*/
#undef __FUNC__  
#define __FUNC__ "TSStep_PVode_Nonlinear"
/* 
    TSStep_PVode_Nonlinear - 
  
   steps - number of time steps
   time - time that integrater is  terminated. 

*/
static int TSStep_PVode_Nonlinear(TS ts,int *steps,double *time)
{
  TS_PVode  *cvode = (TS_PVode*) ts->data;
  Vec       sol = ts->vec_sol;
  int       ierr,i,max_steps = ts->max_steps,flag;
  double    t, tout;

  /* initialize the number of steps */
  *steps = -ts->steps;
  ierr   = TSMonitor(ts,ts->steps,ts->ptime,sol); CHKERRQ(ierr);

  /* call CVSpgmr to use GMRES as the linear solver. */
  
  /* setup the ode integrator with the given preconditioner */
  CVSpgmr(cvode->mem, LEFT, MODIFIED_GS, 0, 0.0, TSPrecond_PVode,TSPSolve_PVode, ts);

  tout = ts->max_time;
  for ( i=0; i<max_steps; i++) {
    if (ts->ptime > ts->max_time) break;
    flag=CVode(cvode->mem, tout, cvode->y, &t, ONE_STEP);
    if (flag != SUCCESS) SETERRQ(1,0,"PVODE failed");	

    ts->time_step = t - ts->ptime;
    ts->ptime     = t;

    /*
       copy the solution from cvode->y to cvode->update and sol 
    */
    ierr = VecPlaceArray(cvode->w1,&N_VIth(cvode->y,0)); CHKERRQ(ierr);
    ierr = VecCopy(cvode->w1,cvode->update); CHKERRQ(ierr);
    ierr = VecCopy(cvode->update, sol); CHKERRQ(ierr);
    
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,t,sol); CHKERRQ(ierr);
  }

  /* count the number of interations, notice CVDEnse uses Newtons method 
  */
  ts->nonlinear_its = cvode->iopt[NNI];
  ts->linear_its    = 0;

  /* count the number of steps and the time reached by CVODE */
  *steps += ts->steps;
  *time  = t;

  return 0;
}

/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "TSDestroy_PVode"
static int TSDestroy_PVode(PetscObject obj )
{
  TS        ts = (TS) obj;
  TS_PVode *cvode = (TS_PVode*) ts->data;
  int       ierr;

  if (cvode->pc)     {ierr = PCDestroy(cvode->pc); CHKERRQ(ierr);}
  if (cvode->update) {ierr = VecDestroy(cvode->update); CHKERRQ(ierr);}
  if (cvode->func)   {ierr = VecDestroy(cvode->func);CHKERRQ(ierr);}
  if (cvode->rhs)    {ierr = VecDestroy(cvode->rhs);CHKERRQ(ierr);}
  if (cvode->w1)     {ierr = VecDestroy(cvode->w1);CHKERRQ(ierr);}
  if (cvode->w2)     {ierr = VecDestroy(cvode->w2);CHKERRQ(ierr);}
  PetscFree(cvode);
  return 0;
}


/*--------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "TSSetUp_PVode_Nonlinear"
static int TSSetUp_PVode_Nonlinear(TS ts)
{
  TS_PVode    *cvode = (TS_PVode*) ts->data;
  int         ierr, M, locsize;

  machEnvType machEnv;

  /* get the vector size */
  ierr = VecGetSize(ts->vec_sol,&M); CHKERRQ(ierr);
  ierr = VecGetLocalSize(ts->vec_sol,&locsize); CHKERRQ(ierr);

  /* allocate the memory for machEnv */
  machEnv = PVInitMPI(ts->comm,locsize,M); 

  /* allocate the memory for N_Vec y */
  cvode->y         = N_VNew(M,machEnv); 
  ierr = VecGetArray(ts->vec_sol,&cvode->y->data); CHKERRQ(ierr);

  /* set tolerance for PVode */
  cvode->abstol = 1e-6;
  cvode->reltol = 1e-6;


  /* initializing vector update and func */
  ierr = VecDuplicate(ts->vec_sol,&cvode->update); CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&cvode->func); CHKERRQ(ierr);  
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

  /* allocate memory for PVode */
  cvode->mem = CVodeMalloc(M,TSFunction_PVode,ts->ptime,cvode->y,
 			          cvode->cvode_method,
                                  NEWTON,SS,&cvode->reltol,
                                  &cvode->abstol,ts,NULL,FALSE,cvode->iopt,
                                  cvode->ropt,machEnv); CHKPTRQ(cvode->mem);
  return 0;
}

/*--------------------------------------------------------------------*/
#undef __FUNC__
#define __FUNC__ "TSPVodePCSetUp_PVode"
static int TSPVodePCSetUp_PVode(TS ts)
{
  TS_PVode *cvode = (TS_PVode*) ts->data;
  int      ierr;

  ierr = PCSetFromOptions(cvode->pc); CHKERRQ(ierr);

  return 0;
}

/*------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "TSSetFromOptions_PVode_Nonlinear"
static int TSSetFromOptions_PVode_Nonlinear(TS ts)
{
  int      ierr, flag;
  char     method[128];

  /*
     Allows user to set any of the PC options 
     -ts_pvode_method bdf or adams
  */
  ierr = OptionsGetString(PETSC_NULL,"-ts_pvode_method",method,127,&flag);CHKERRQ(ierr);
  
  if (flag) {
    if (PetscStrcmp(method,"bdf") == 0) {
      ierr = TSPVodeSetMethod(ts, BDF); CHKERRQ(ierr);
    }
    else if (PetscStrcmp(method,"adams") == 0) {
      ierr = TSPVodeSetMethod(ts, ADAMS); CHKERRQ(ierr);
    }
    else {
      SETERRQ(1,0,"Unknow PVode method. \n");
    }
  }
  else {
    ierr = TSPVodeSetMethod(ts, BDF); CHKERRQ(ierr); /* the default method */
  }

  /*  user set the preconditioner */
  ierr = TSPVodePCSetUp_PVode(ts); CHKERRQ(ierr); 
  
  return 0;
}

/*--------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "TSPrintHelp_PVode" 
static int TSPrintHelp_PVode(TS ts,char *p)
{

  return 0;
}

/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "TSView_PVode" 
static int TSView_PVode(PetscObject obj,Viewer viewer)
{
  TS       ts = (TS) obj;
  TS_PVode *cvode = (TS_PVode*) ts->data;
  int      ierr;
  MPI_Comm comm;
  FILE     *fd;

  ierr = PetscObjectGetComm(obj,&comm); CHKERRQ(ierr);
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  PetscFPrintf(comm,fd,"PVode integrater does not use SNES!\n");
  ierr = PCView(cvode->pc,viewer); CHKERRQ(ierr);

  return 0;
}

/* ------------------------------------------------------------ */
#undef __FUNC__  
#define __FUNC__ "TSCreate_PVode"
int TSCreate_PVode(TS ts )
{
  TS_PVode *cvode;
  int      ierr;

  ts->type 	      = TS_PVODE;
  ts->destroy         = TSDestroy_PVode;
  ts->printhelp       = TSPrintHelp_PVode;
  ts->view            = TSView_PVode;

  if (ts->problem_type != TS_NONLINEAR) {
    SETERRQ(1,0,"Only support for nonlinear problems");
  }
  ts->setup           = TSSetUp_PVode_Nonlinear;  
  ts->step            = TSStep_PVode_Nonlinear;
  ts->setfromoptions  = TSSetFromOptions_PVode_Nonlinear;

  cvode    = PetscNew(TS_PVode); CHKPTRQ(cvode);
  PetscMemzero(cvode,sizeof(TS_PVode));
  ierr     = PCCreate(ts->comm, &cvode->pc); CHKERRQ(ierr);
  PLogObjectParent(ts,cvode->pc);
  ts->data = (void *) cvode;

  return 0;
}


/*--------------------------------------------------------------------*/
#undef __FUNC__
#define __FUNC__ "TSPVodeSetMethod"
/*@
   TSPVodeSetMethod - Sets the method that PVode will use for integration.

   Input parameters:
    ts     - the time-step context
    method - one of  PVODE_ADAMS or PVODE_BDF

.keywords: Adams, backward differentiation formula
@*/
int TSPVodeSetMethod(TS ts, TSPVodeMethod method)
{
  TS_PVode *cvode = (TS_PVode*) ts->data;
  
  if (ts->type != TS_PVODE) return 0;
  cvode->cvode_method = method;
  return 0;
}

/*--------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "TSPVodeGetPC"
/*
   TSPVodeGetPC - Extract the PC context from a time-step context for PVode.

   Input Parameter:
.    ts - the time-step context

   Output Parameter:
.    pc - the preconditioner context

.seealso: 
*/
int TSPVodeGetPC(TS ts, PC *pc)
{ 
  int      ierr;
  TS_PVode *cvode = (TS_PVode*) ts->data;

  if (ts->type != TS_PVODE) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,1,"TS must be of PVode type to extract the PC");
  }
  if (!cvode->pc) {
    ierr = PCCreate(ts->comm,&cvode->pc); CHKERRQ(ierr);
  }
  *pc = cvode->pc;

  return 0;
}

#else

/* 
     A dummy function for compilers that dislike empy files.
*/
int adummyfunction()
{
  return 0;
}

#endif
