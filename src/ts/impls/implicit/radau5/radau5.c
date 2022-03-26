/*
    Provides a PETSc interface to RADAU5 solver.

*/
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/

typedef struct {
  Vec work,workf;
} TS_Radau5;

void FVPOL(int *N,double *X,double *Y,double *F,double *RPAR,void *IPAR)
{
  TS             ts = (TS) IPAR;
  TS_Radau5      *cvode = (TS_Radau5*)ts->data;
  DM             dm;
  DMTS           tsdm;
  TSIFunction    ifunction;
  PetscErrorCode ierr;

  PetscCallAbort(PETSC_COMM_SELF,VecPlaceArray(cvode->work,Y));
  PetscCallAbort(PETSC_COMM_SELF,VecPlaceArray(cvode->workf,F));

  /* Now compute the right hand side function, via IFunction unless only the more efficient RHSFunction is set */
  PetscCallAbort(PETSC_COMM_SELF,TSGetDM(ts,&dm));
  PetscCallAbort(PETSC_COMM_SELF,DMGetDMTS(dm,&tsdm));
  PetscCallAbort(PETSC_COMM_SELF,DMTSGetIFunction(dm,&ifunction,NULL));
  if (!ifunction) {
    PetscCallAbort(PETSC_COMM_SELF,TSComputeRHSFunction(ts,*X,cvode->work,cvode->workf));
  } else {       /* If rhsfunction is also set, this computes both parts and scale them to the right hand side */
    Vec yydot;

    PetscCallAbort(PETSC_COMM_SELF,VecDuplicate(cvode->work,&yydot));
    PetscCallAbort(PETSC_COMM_SELF,VecZeroEntries(yydot));
    PetscCallAbort(PETSC_COMM_SELF,TSComputeIFunction(ts,*X,cvode->work,yydot,cvode->workf,PETSC_FALSE));
    PetscCallAbort(PETSC_COMM_SELF,VecScale(cvode->workf,-1.));
    PetscCallAbort(PETSC_COMM_SELF,VecDestroy(&yydot));
  }

  PetscCallAbort(PETSC_COMM_SELF,VecResetArray(cvode->work));
  PetscCallAbort(PETSC_COMM_SELF,VecResetArray(cvode->workf));
}

void JVPOL(PetscInt *N,PetscScalar *X,PetscScalar *Y,PetscScalar *DFY,int *LDFY,PetscScalar *RPAR,void *IPAR)
{
  TS             ts = (TS) IPAR;
  TS_Radau5      *cvode = (TS_Radau5*)ts->data;
  Vec            yydot;
  Mat            mat;
  PetscInt       n;
  PetscErrorCode ierr;

  PetscCallAbort(PETSC_COMM_SELF,VecPlaceArray(cvode->work,Y));
  PetscCallAbort(PETSC_COMM_SELF,VecDuplicate(cvode->work,&yydot));
  PetscCallAbort(PETSC_COMM_SELF,VecGetSize(yydot,&n));
  PetscCallAbort(PETSC_COMM_SELF,MatCreateSeqDense(PETSC_COMM_SELF,n,n,DFY,&mat));
  PetscCallAbort(PETSC_COMM_SELF,VecZeroEntries(yydot));
  PetscCallAbort(PETSC_COMM_SELF,TSComputeIJacobian(ts,*X,cvode->work,yydot,0,mat,mat,PETSC_FALSE));
  PetscCallAbort(PETSC_COMM_SELF,MatScale(mat,-1.0));
  PetscCallAbort(PETSC_COMM_SELF,MatDestroy(&mat));
  PetscCallAbort(PETSC_COMM_SELF,VecDestroy(&yydot));
  PetscCallAbort(PETSC_COMM_SELF,VecResetArray(cvode->work));
}

void SOLOUT(int *NR,double *XOLD,double *X, double *Y,double *CONT,double *LRC,int *N,double *RPAR,void *IPAR,int *IRTRN)
{
  TS             ts = (TS) IPAR;
  TS_Radau5      *cvode = (TS_Radau5*)ts->data;
  PetscErrorCode ierr;

  PetscCallAbort(PETSC_COMM_SELF,VecPlaceArray(cvode->work,Y));
  ts->time_step = *X - *XOLD;
  PetscCallAbort(PETSC_COMM_SELF,TSMonitor(ts,*NR-1,*X,cvode->work));
  PetscCallAbort(PETSC_COMM_SELF,VecResetArray(cvode->work));
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
  PetscCall(VecGetArray(ts->vec_sol,&Y));
  PetscCall(VecGetSize(ts->vec_sol,&ND));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,ND,NULL,&cvode->work));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,ND,NULL,&cvode->workf));

  LWORK  = 4*ND*ND+12*ND+20;
  LIWORK = 3*ND+20;

  PetscCall(PetscCalloc2(LWORK,&WORK,LIWORK,&IWORK));

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

  /* output MUJAC MLMAS IDID; currently all ignored */

  radau5_(&ND,FVPOL,&X,Y,&XEND,&H,&RTOL,&ATOL,&ITOL,JVPOL,&IJAC,&MLJAC,&MUJAC,FVPOL,&IMAS,&MLMAS,&MUMAS,SOLOUT,&IOUT,WORK,&LWORK,IWORK,&LIWORK,&RPAR,(void*)ts,&IDID);

  PetscCall(PetscFree2(WORK,IWORK));
  PetscFunctionReturn(0);
}

PetscErrorCode TSDestroy_Radau5(TS ts)
{
  TS_Radau5      *cvode = (TS_Radau5*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&cvode->work));
  PetscCall(VecDestroy(&cvode->workf));
  PetscCall(PetscFree(ts->data));
  PetscFunctionReturn(0);
}

/*MC
      TSRADAU5 - ODE solver using the RADAU5 package

    Notes:
    This uses its own nonlinear solver and dense matrix direct solver so PETSc SNES and KSP options do not apply.
           Uses its own time-step adaptivity (but uses the TS rtol and atol, and initial timestep)
           Uses its own memory for the dense matrix storage and factorization
           Can only handle ODEs of the form \cdot{u} = -F(t,u) + G(t,u)

    Level: beginner

.seealso:  TSCreate(), TS, TSSetType()

M*/
PETSC_EXTERN PetscErrorCode TSCreate_Radau5(TS ts)
{
  TS_Radau5      *cvode;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->destroy        = TSDestroy_Radau5;
  ts->ops->solve          = TSSolve_Radau5;
  ts->default_adapt_type  = TSADAPTNONE;

  PetscCall(PetscNewLog(ts,&cvode));
  ts->data = (void*)cvode;
  PetscFunctionReturn(0);
}
