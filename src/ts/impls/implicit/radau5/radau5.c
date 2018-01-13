/*
    Provides a PETSc interface to RADAU5 solver.

*/
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/

typedef struct {
  Vec work,workf;
} TS_Radau5;

#ifdef foo
/*
        TSFunction_Radau5 - routine that we provide to Radau5 that applies the right hand side.
*/
int TSFunction_Radau5(realtype t,N_Vector y,N_Vector ydot,void *ctx)
{
  TS             ts = (TS) ctx;
  DM             dm;
  DMTS           tsdm;
  TSIFunction    ifunction;
  MPI_Comm       comm;
  TS_Radau5    *cvode = (TS_Radau5*)ts->data;
  Vec            yy     = cvode->w1,yyd = cvode->w2,yydot = cvode->ydot;
  PetscScalar    *y_data,*ydot_data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  /* Make the PETSc work vectors yy and yyd point to the arrays in the RADAU5 vectors y and ydot respectively*/
  y_data    = (PetscScalar*) N_VGetArrayPointer(y);
  ydot_data = (PetscScalar*) N_VGetArrayPointer(ydot);
  ierr      = VecPlaceArray(yy,y_data);CHKERRABORT(comm,ierr);
  ierr      = VecPlaceArray(yyd,ydot_data);CHKERRABORT(comm,ierr);

  /* Now compute the right hand side function, via IFunction unless only the more efficient RHSFunction is set */
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMGetDMTS(dm,&tsdm);CHKERRQ(ierr);
  ierr = DMTSGetIFunction(dm,&ifunction,NULL);CHKERRQ(ierr);
  if (!ifunction) {
    ierr = TSComputeRHSFunction(ts,t,yy,yyd);CHKERRQ(ierr);
  } else {                      /* If rhsfunction is also set, this computes both parts and shifts them to the right */
    ierr = VecZeroEntries(yydot);CHKERRQ(ierr);
    ierr = TSComputeIFunction(ts,t,yy,yydot,yyd,PETSC_FALSE);CHKERRABORT(comm,ierr);
    ierr = VecScale(yyd,-1.);CHKERRQ(ierr);
  }
  ierr = VecResetArray(yy);CHKERRABORT(comm,ierr);
  ierr = VecResetArray(yyd);CHKERRABORT(comm,ierr);
  PetscFunctionReturn(0);
}
#endif

void FVPOL(int *N,double *X,double *Y,double *F,double *RPAR,void *IPAR)
{
  TS          ts = (TS) IPAR;
  TS_Radau5   *cvode = (TS_Radau5*)ts->data;
  DM          dm;
  DMTS        tsdm;
  TSIFunction ifunction;

  VecPlaceArray(cvode->work,Y);
  VecPlaceArray(cvode->workf,F);

  /* Now compute the right hand side function, via IFunction unless only the more efficient RHSFunction is set */
  TSGetDM(ts,&dm);
  DMGetDMTS(dm,&tsdm);
  DMTSGetIFunction(dm,&ifunction,NULL);
  if (!ifunction) {
    TSComputeRHSFunction(ts,*X,cvode->work,cvode->workf);
  } else {                      /* If rhsfunction is also set, this computes both parts and shifts them to the right */
    Vec yydot;

    VecDuplicate(cvode->work,&yydot);
    VecZeroEntries(yydot);
    TSComputeIFunction(ts,*X,cvode->work,yydot,cvode->workf,PETSC_FALSE);
    VecScale(cvode->workf,-1.);
    VecDestroy(&yydot);
  }

  VecResetArray(cvode->work);
  VecResetArray(cvode->workf);
}

#ifdef foo
void FVPOL(int *N,double *X,double *Y,double *F,double *RPAR,void *IPAR)
{
  F[0]=Y[1];
  F[1]=((1-Y[0]*Y[0])*Y[1]-Y[0])/(*RPAR);
}
#endif

void JVPOL(PetscInt *N,PetscScalar *X,PetscScalar *Y,PetscScalar *DFY,int *LDFY,PetscScalar *RPAR,void *IPAR)
{
  DFY[0]=0.0;
  DFY[1]=(-2.0*Y[0]*Y[1]-1.0)/(*RPAR);
  DFY[2]=1.0;
  DFY[3]=(1.0-Y[0]*Y[0])/(*RPAR);
}

void SOLOUT(int *NR,double *XOLD,double *X, double *Y,double *CONT,double *LRC,int *N,double *RPAR,void *IPAR,int *IRTRN)
{
  TS        ts = (TS) IPAR;
  TS_Radau5 *cvode = (TS_Radau5*)ts->data;

  VecPlaceArray(cvode->work,Y);
  ts->time_step = *X - *XOLD;
  TSMonitor(ts,*NR-1,*X,cvode->work);
  VecResetArray(cvode->work);
  PetscPrintf(PETSC_COMM_SELF,"X = %g Y = %g %g NSTEP = %d dt = %g\n",*X,Y[0],Y[1],*NR-1,*X - *XOLD);
}


void radau5_(int *,void*,double*,double*,double*,double*,double*,double*,int*,void*,int*,int*,int*,void*,int*,int*,int*,void*,int*,double*,int*,int*,int*,double*,void*,int*);

PetscErrorCode TSSolve_Radau5(TS ts)
{
  TS_Radau5      *cvode = (TS_Radau5*)ts->data;
  PetscErrorCode ierr;
  PetscScalar    *Y,*WORK,X,XEND,RTOL,ATOL,H,RPAR;
  PetscInt       ND,*IWORK,LWORK,LIWORK,MUJAC,MLMAS,MUMAS,IDID,ITOL;
  int            IJAC,MLJAC,IMAS,IOUT;

  PetscFunctionBegin;
  ierr = VecGetArray(ts->vec_sol,&Y);CHKERRQ(ierr);
  ierr = VecGetSize(ts->vec_sol,&ND);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,ND,NULL,&cvode->work);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,ND,NULL,&cvode->workf);CHKERRQ(ierr);

  LWORK  = 4*ND*ND+12*ND+20;
  LIWORK = 3*ND+20;

  ierr = PetscCalloc1(LWORK,&WORK);CHKERRQ(ierr);
  ierr = PetscCalloc1(LIWORK,&IWORK);CHKERRQ(ierr);

  /* C --- PARAMETER IN THE DIFFERENTIAL EQUATION */
  RPAR=1.0e-6;
  /* C --- COMPUTE THE JACOBIAN ANALYTICALLY */
  IJAC=1;
  /* C --- JACOBIAN IS A FULL MATRIX */
  MLJAC=ND;
  /* C --- DIFFERENTIAL EQUATION IS IN EXPLICIT FORM*/
  IMAS=0;
  /* C --- OUTPUT ROUTINE IS USED DURING INTEGRATION*/
  IOUT=1;
  /* C --- INITIAL VALUES*/
  X = ts->ptime;
  /* C --- ENDPOINT OF INTEGRATION */
  XEND = ts->max_time;
  /* C --- REQUIRED TOLERANCE */
  RTOL = ts->rtol;
  ATOL = ts->atol;
  ITOL=0;
  /* C --- INITIAL STEP SIZE */
  H = ts->time_step;

  /* output MUJAC MLMAS IDID */

  CHKMEMQ;
  radau5_(&ND,FVPOL,&X,Y,&XEND,&H,&RTOL,&ATOL,&ITOL,JVPOL,&IJAC,&MLJAC,&MUJAC,FVPOL,&IMAS,&MLMAS,&MUMAS,SOLOUT,&IOUT,WORK,&LWORK,IWORK,&LIWORK,&RPAR,(void*)ts,&IDID);
  CHKMEMQ;

  ierr = PetscFree(WORK);CHKERRQ(ierr);
  ierr = PetscFree(IWORK);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSDestroy_Radau5(TS ts)
{
  TS_Radau5      *cvode = (TS_Radau5*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&cvode->work);CHKERRQ(ierr);
  ierr = VecDestroy(&cvode->workf);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSSetFromOptions_Radau5(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_Radau5      *cvode = (TS_Radau5*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"RADAU5 ODE solver options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSView_Radau5(TS ts,PetscViewer viewer)
{
  TS_Radau5      *cvode = (TS_Radau5*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*MC
      TSRADAU5 - ODE solver using the RADAU5 package

    Notes: This uses its own nonlinear solver and Krylov method so PETSc SNES and KSP options do not apply

    Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSSetExactFinalTime()

M*/
PETSC_EXTERN PetscErrorCode TSCreate_Radau5(TS ts)
{
  TS_Radau5      *cvode;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->destroy        = TSDestroy_Radau5;
  ts->ops->view           = TSView_Radau5;
  ts->ops->solve          = TSSolve_Radau5;
  ts->ops->setfromoptions = TSSetFromOptions_Radau5;
  ts->default_adapt_type  = TSADAPTNONE;

  ierr = PetscNewLog(ts,&cvode);CHKERRQ(ierr);
  ts->data = (void*)cvode;
  PetscFunctionReturn(0);
}
