#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: petscpvode.c,v 1.2 1997/08/22 15:16:47 bsmith Exp balay $";
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
  int          ierr, i, locsize, low, high, loc;

  Mat          Jac;
  Vec          u, tmpy;
  Scalar       zero = 0.0, one = 1.0, gm, tmp;
  MatStructure str = SAME_NONZERO_PATTERN;

  /* get the local size of N_Vector y */
  locsize = N_VLOCLENGTH(y);

  /* Jac and u must be set before calling PCSetUp */
  ierr = MatCreate(MPI_COMM_WORLD,N,N,&Jac); CHKERRA(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,locsize,N,&u); CHKERRA(ierr);
  ierr = VecSet(&zero,u); CHKERRA(ierr);

  if (!cvode->pmat) {
    ierr = MatCreate(MPI_COMM_WORLD,N,N,&cvode->pmat); CHKERRA(ierr);
  }

  if (jok) {
    /* jok=TRUE: copy the Jacobian to Pcond */
    ierr = MatCopy(cvode->pmat,Jac); CHKERRA(ierr);

    *jcurPtr = FALSE;
  }
  else {
    /* jok = FALSE: generate the Jacobian and then copy it to Pcond */
    /* convert N_Vector y to petsc Vec tmpy */
    /* we are only working on the local part */
    ierr = VecCreateMPI(PETSC_COMM_WORLD,locsize,N,&tmpy); CHKERRA(ierr);
    ierr = VecGetOwnershipRange(tmpy,&low,&high); CHKERRA(ierr);
    for(i=0;i<locsize;i++) {
      loc = low+i;	 /* get the global position */
      tmp = Ith(y,i+1);  /* the local component */
      ierr = VecSetValues(tmpy,1,&loc,&tmp,INSERT_VALUES); CHKERRA(ierr);
    }
    ierr = VecAssemblyBegin(tmpy); CHKERRA(ierr);
    ierr = VecAssemblyEnd(tmpy); CHKERRA(ierr);

    /* recompute the Jacobian */
    ierr = (*ts->rhsjacobian)(ts,ts->ptime,tmpy,&Jac,&Jac,&str,ts->jacP);
    CHKERRA(ierr);

    /* copy the Jacobian matrix */
    ierr = MatCopy(Jac, cvode->pmat); CHKERRA(ierr);

    /* set the flag */
    *jcurPtr = TRUE;

    /* destroy vector tmpy */
    ierr = VecDestroy(tmpy); CHKERRA(ierr);
  }

  /* construct I-gamma*Jac  */
  gm = -_gamma;
  ierr = MatScale(&gm,Jac); CHKERRA(ierr);
  ierr = MatShift(&one,Jac); CHKERRA(ierr);
  
  /* setup the preconditioner contex */
  ierr = PCSetOperators(pc,Jac,Jac,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = PCSetVector(pc,u);   CHKERRA(ierr);
  ierr = PCSetUp(pc); CHKERRA(ierr);

  return(0);
}

/*
  TSPSolve_PVode is the routine that we provide to PVode that applies the 
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
  Vec      rr, xx;
  Scalar   tmp, *tmpary;

  int i, ierr, locsize, low, high, loc;

  /* create vector rr */
  locsize = N_VLOCLENGTH(y);
  ierr = VecCreateMPI(MPI_COMM_WORLD, locsize, N, &rr); CHKERRA(ierr);

  /* copy r to a petsc Vec rr */
  ierr = VecGetOwnershipRange(rr,&low,&high); CHKERRA(ierr);
  for(i=0; i<locsize; i++) {
    loc = low+i;
    tmp = Ith(r, i+1);
    ierr = VecSetValues(rr,1,&loc,&tmp,INSERT_VALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(rr); CHKERRA(ierr);
  ierr = VecAssemblyEnd(rr); CHKERRA(ierr);

  /* create vector xx */
  locsize = N_VLOCLENGTH(z);
  ierr = VecCreateMPI(MPI_COMM_WORLD, locsize, N, &xx); CHKERRA(ierr);

  /* solve the Px=r and put the result in xx */
  ierr = PCApply(pc,rr,xx); CHKERRA(ierr);

  /* copy xx to N_Vector z as th output */
  ierr = VecGetArray(xx, &tmpary); CHKERRA(ierr);
  for(i=0; i<locsize; i++) {
    loc = i;
    Ith(z,i+1)=tmpary[loc];
  }
  ierr = VecRestoreArray(xx,&tmpary); CHKERRA(ierr);

  /* destroy the vectors */
  ierr = VecDestroy(xx); CHKERRA(ierr);
  ierr = VecDestroy(rr); CHKERRA(ierr);

  return 0;
}

/*
  TSPSolve_PVode is the routine that we provide to PVode that applies the 
  right hand side.
      
   ---------------------------------------------------------------------
*/  
#undef __FUNC__  
#define __FUNC__ "TSFunction_PVode"
static void TSFunction_PVode(int N,double t,N_Vector y,N_Vector ydot,void *ctx)
{
  TS        ts = (TS) ctx;
  Vec       tmpx, tmpy;
  int       i, low, high, locsize, loc, ierr;
  Scalar    tmp, *Funp;

  /* get the local size of y */
  locsize = N_VLOCLENGTH(y);
  
  /* create petsc vector tmpx */
  ierr = VecCreateMPI(MPI_COMM_WORLD, locsize, N, &tmpx); CHKERRA(ierr);

  /* copy the N_vector y to tmpx */
  ierr = VecGetOwnershipRange(tmpx,&low,&high); CHKERRA(ierr);
  for( i=0; i<locsize; i++) {
    tmp = Ith(y,i+1);
    loc = low+i;
    ierr = VecSetValues(tmpx,1,&loc,&tmp,INSERT_VALUES); 
    CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(tmpx); CHKERRA(ierr);
  ierr = VecAssemblyEnd(tmpx); CHKERRA(ierr);

  /* get the local size of ydot */
  locsize = N_VLOCLENGTH(ydot);

  /* create petsc vector tmpy */
  ierr = VecCreateMPI(MPI_COMM_WORLD, locsize, N, &tmpy); CHKERRA(ierr);

  /* now compute the right hand side function */
  ierr = TSComputeRHSFunction(ts,t,tmpx,tmpy); CHKERRA(ierr);

  /* copy tmpy to the N_vector y */
  ierr = VecGetArray(tmpy, &Funp); CHKERRA(ierr);

  for (i=0; i<locsize; i++) {
    loc=i;
    Ith(ydot,i+1)=Funp[loc];
  }

  /* free the spaces */
  ierr = VecRestoreArray(tmpy,&Funp); CHKERRA(ierr);
  ierr = VecDestroy(tmpx); CHKERRA(ierr);
  ierr = VecDestroy(tmpy); CHKERRA(ierr);

}

/*
  This function calls PVode to integrate the ODE.

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
  Vec       sol = ts->vec_sol;
  int       ierr,i,j,max_steps = ts->max_steps;
  TS_PVode *cvode = (TS_PVode*) ts->data;
  int       flag, locsize, low, high, loc;
  double    t, tout;
  Scalar    tmp;

  /* initialize the number of steps */
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol); CHKERRQ(ierr);

  /* call CVSpgmr to use GMRES as the linear solver. */
  
  /* setup the ode integrator with the given preconditioner */
  CVSpgmr(cvode->mem, LEFT, MODIFIED_GS, 0, 0.0, TSPrecond_PVode,TSPSolve_PVode, ts);


  /* call PVode to solve the system */
  tout = ts->max_time;
  for ( i=0; i<max_steps; i++) {
    if (ts->ptime > ts->max_time) break;
    flag=CVode(cvode->mem, tout, cvode->y, &t, ONE_STEP);
    if (flag != SUCCESS) SETERRQ(1,0,"PVODE failed");	

    ts->time_step = t - ts->ptime;
    ts->ptime     = t;

    /* get the number of equations */
    ierr = VecGetLocalSize(cvode->update, &locsize); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(cvode->update,&low,&high); CHKERRQ(ierr);

    /* copy the solution from cvode->y to cvode->update */
    for( j=0; j<locsize; j++) {
      tmp = Ith(cvode->y,j+1);
      loc = low+j;
      ierr = VecSetValues(cvode->update,1,&loc,&tmp,INSERT_VALUES); CHKERRQ(ierr);
    }

    ierr = VecAssemblyBegin(cvode->update); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(cvode->update); CHKERRQ(ierr);

    /* copy vector cvode->update to sol */ 
    ierr = VecCopy(cvode->update, sol); CHKERRQ(ierr);
    
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,t,sol); CHKERRQ(ierr);
  }

  /* count on the number of interations, notice CVDEnse uses Newtons method 
  */
  ts->nonlinear_its = cvode->iopt[NNI];
  ts->linear_its    = 0;

  /* count on the number of steps and the time reached by CVODE */
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

  ierr = VecDestroy(cvode->update); CHKERRQ(ierr);
  if (cvode->func) {ierr = VecDestroy(cvode->func);CHKERRQ(ierr);}
  if (cvode->rhs) {ierr = VecDestroy(cvode->rhs);CHKERRQ(ierr);}
  PetscFree(cvode);
  return 0;
}


/*--------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "TSSetUp_PVode_Nonlinear"
static int TSSetUp_PVode_Nonlinear(TS ts)
{
  TS_PVode *cvode = (TS_PVode*) ts->data;
  int       ierr, M, locsize;

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

  /* allocate memory for PVode */
  cvode->mem = CVodeMalloc(M,TSFunction_PVode,ts->ptime,cvode->y,
 			          cvode->cvode_method,
                                  NEWTON,SS,&cvode->reltol,
                                  &cvode->abstol,ts,NULL,FALSE,cvode->iopt,
                                  cvode->ropt,machEnv);
  if (cvode->mem == NULL) {
    SETERRQ(1,0,"PVodeMalloc failed. \n");
    return 1;
  }

  /* initializing vector update and func */
  ierr = VecDuplicate(ts->vec_sol,&cvode->update); CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&cvode->func); CHKERRQ(ierr);  

  return 0;
}

/*--------------------------------------------------------------------*/
#undef __FUNC__
#define __FUNC__ "TSPVodePCSetUp_PVode"
static int TSPVodePCSetUp_PVode(TS ts)
{
  TS_PVode *cvode = (TS_PVode*) ts->data;
  int      ierr;

  ierr = PCSetFromOptions(cvode->pc); CHKERRA(ierr);

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

  if (ts->problem_type == TS_LINEAR) {
    SETERRQ(1,0,"Only support for nonlinear problems");
  } else if (ts->problem_type == TS_NONLINEAR) {
    ts->setup           = TSSetUp_PVode_Nonlinear;  
    ts->step            = TSStep_PVode_Nonlinear;
    ts->setfromoptions  = TSSetFromOptions_PVode_Nonlinear;
  } else SETERRQ(1,0,"No such problem");

  cvode    = PetscNew(TS_PVode); CHKPTRQ(cvode);
  PetscMemzero(cvode,sizeof(TS_PVode));
  ierr     = PCCreate(ts->comm, &cvode->pc); CHKERRQ(ierr);
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
#define __FUNC__ "TSPVodePCSetType"
/*@
     TSPVodeSetPCType - This function sets the preconditioner type for TS
                        when using PVODE.

  Input parameters:
.   ts - the time-step context
.   type - the preconditioner type, for example, PCBJACOBI

.seealso: TSPVodeGetPC()

@*/
int TSPVodeSetPCType(TS ts, PCType type)
{
  TS_PVode *cvode = (TS_PVode*) ts->data;
  int      ierr;
  MPI_Comm comm = ts->comm;

  if (ts->type != TS_PVODE) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,1,"TS must be of PVode type to set the PC type");

  /* Create a PETSc preconditioner context */
  ierr = PCCreate(comm,&cvode->pc); CHKERRA(ierr);
  ierr = PCSetType(cvode->pc,type); CHKERRA(ierr);
  ierr = PCSetFromOptions(cvode->pc); CHKERRA(ierr);

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

.seealso: TSPVodeSetPCType()
*/
int TSPVodeGetPC(TS ts, PC *pc)
{ 
  TS_PVode *cvode = (TS_PVode*) ts->data;

  if (ts->type != TS_PVODE) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,1,"TS must be of PVode type to extract the PC");
  *pc = cvode->pc;

  return 0;
}

/*
=====================================================================================
*/


/*--------------------------------------------------------------------*/
/* 
This function computes the RHS Jacobian by FD.
*/
#undef __FUNC__
#define __FUNC__ "TSPVodeSetRHSJacobian"
int TSPVodeSetRHSJacobian(TS ts,double t,Vec y,Mat* A,Mat* B,MatStructure *flag,void *ctx)
{
  int           ierr;
  SNES          snes = (SNES) ctx;
  Mat           J = *A;
  MatFDColoring fdcoloring;
  ISColoring    iscoloring;

  if (snes->jacobian->assembled) {
    /* set the jacobian structure */
    ierr = MatZeroEntries(J); CHKERRA(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
    ierr = MatCopy(snes->jacobian,J); CHKERRA(ierr);
  }
  else {
    /* compute the jacobian by the user provid function */
    ierr = (*snes->computejacobian)(snes,y,&J,&J,flag,snes->funP); CHKERRA(ierr);
  }
  ierr = MatGetColoring(J,COLORING_NATURAL,&iscoloring); CHKERRA(ierr);
  ierr = MatFDColoringCreate(J,iscoloring,&fdcoloring); CHKERRA(ierr);
  ierr = MatFDColoringSetFromOptions(fdcoloring); CHKERRA(ierr);
  ierr = SNESSetJacobian(snes,J,J,SNESDefaultComputeJacobianWithColoring,
         fdcoloring);CHKERRA(ierr);

  /* Compute the jacobian */
  ierr = SNESDefaultComputeJacobianWithColoring
         (snes,y,&J,&J,flag,
         fdcoloring);CHKERRA(ierr);
  /*
  This line causes error message, don't know why.
  ierr = ISColoringDestroy(iscoloring); CHKERRA(ierr);
  */

  *flag = SAME_NONZERO_PATTERN;

  return 0;
}

/*--------------------------------------------------------------------*/
/*
This function reads in the RHS function from SNES setting 
*/
#undef __FUNC__
#define __FUNC__ "TSPVodeSetRHSFunction"
int TSPVodeSetRHSFunction(TS ts, double t, Vec in, Vec out, void *ctx)
{
  int ierr;
  SNES          snes = (SNES) ctx;
  ierr = (*snes->computefunction)(snes, in, out, snes->funP);
  CHKERRQ(ierr);

  return 0;
}

/*--------------------------------------------------------------------*/
/*
This function computes the RHS Jacobian.
*/  
#undef __FUNC__
#define __FUNC__ "TSComputeRHSJacobianForPVODE"
int TSComputeRHSJacobianForPVODE(TS ts, double t, Vec y, Mat* A,
  Mat* B, MatStructure *flag, void *ctx)
{ 
  int ierr;
 
  /* compute the Jacobian */
  ierr = (*ts->rhsjacobian)(ts,t,y,A,B,flag,ctx);
  CHKERRA(ierr);

  return 0;
}

/*--------------------------------------------------------------------*/
/*  
This function computes the RHS Function.
*/
#undef __FUNC__
#define __FUNC__ "TSComputeRHSFunctionForPVODE"
int TSComputeRHSFunctionForPVODE(TS ts, double t, Vec x, Vec y, void* ctr)
{ 
  int ierr;

  /* compute the RHS function */
  ierr = (*ts->rhsfunction)(ts,t,x,y,ctr);
  
  return 0;
}

/*--------------------------------------------------------------------*/
/*
Extract the user data pointer
*/
#undef __FUNC__ 
#define __FUNC__ "TSPVodeGetUserData"
int TSPVodeGetUserData(TS ts, void *funP)
{
  funP = ts->funP;

  return 0;
}

#else

/* 
     A dummy function for compilers that dislike empyt files.
*/
int adummyfunction()
{
  return 0;
}

#endif
