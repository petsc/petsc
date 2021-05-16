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

  ierr = VecPlaceArray(cvode->work,Y);CHKERRABORT(PETSC_COMM_SELF,ierr);
  ierr = VecPlaceArray(cvode->workf,F);CHKERRABORT(PETSC_COMM_SELF,ierr);

  /* Now compute the right hand side function, via IFunction unless only the more efficient RHSFunction is set */
  ierr = TSGetDM(ts,&dm);CHKERRABORT(PETSC_COMM_SELF,ierr);
  ierr = DMGetDMTS(dm,&tsdm);CHKERRABORT(PETSC_COMM_SELF,ierr);
  ierr = DMTSGetIFunction(dm,&ifunction,NULL);CHKERRABORT(PETSC_COMM_SELF,ierr);
  if (!ifunction) {
    ierr = TSComputeRHSFunction(ts,*X,cvode->work,cvode->workf);CHKERRABORT(PETSC_COMM_SELF,ierr);
  } else {       /* If rhsfunction is also set, this computes both parts and scale them to the right hand side */
    Vec yydot;

    ierr = VecDuplicate(cvode->work,&yydot);CHKERRABORT(PETSC_COMM_SELF,ierr);
    ierr = VecZeroEntries(yydot);CHKERRABORT(PETSC_COMM_SELF,ierr);
    ierr = TSComputeIFunction(ts,*X,cvode->work,yydot,cvode->workf,PETSC_FALSE);CHKERRABORT(PETSC_COMM_SELF,ierr);
    ierr = VecScale(cvode->workf,-1.);CHKERRABORT(PETSC_COMM_SELF,ierr);
    ierr = VecDestroy(&yydot);CHKERRABORT(PETSC_COMM_SELF,ierr);
  }

  ierr = VecResetArray(cvode->work);CHKERRABORT(PETSC_COMM_SELF,ierr);
  ierr = VecResetArray(cvode->workf);CHKERRABORT(PETSC_COMM_SELF,ierr);
}

void JVPOL(PetscInt *N,PetscScalar *X,PetscScalar *Y,PetscScalar *DFY,int *LDFY,PetscScalar *RPAR,void *IPAR)
{
  TS             ts = (TS) IPAR;
  TS_Radau5      *cvode = (TS_Radau5*)ts->data;
  Vec            yydot;
  Mat            mat;
  PetscInt       n;
  PetscErrorCode ierr;

  ierr = VecPlaceArray(cvode->work,Y);CHKERRABORT(PETSC_COMM_SELF,ierr);
  ierr = VecDuplicate(cvode->work,&yydot);CHKERRABORT(PETSC_COMM_SELF,ierr);
  ierr = VecGetSize(yydot,&n);CHKERRABORT(PETSC_COMM_SELF,ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,n,n,DFY,&mat);CHKERRABORT(PETSC_COMM_SELF,ierr);
  ierr = VecZeroEntries(yydot);CHKERRABORT(PETSC_COMM_SELF,ierr);
  ierr = TSComputeIJacobian(ts,*X,cvode->work,yydot,0,mat,mat,PETSC_FALSE);CHKERRABORT(PETSC_COMM_SELF,ierr);
  ierr = MatScale(mat,-1.0);CHKERRABORT(PETSC_COMM_SELF,ierr);
  ierr = MatDestroy(&mat);CHKERRABORT(PETSC_COMM_SELF,ierr);
  ierr = VecDestroy(&yydot);CHKERRABORT(PETSC_COMM_SELF,ierr);
  ierr = VecResetArray(cvode->work);CHKERRABORT(PETSC_COMM_SELF,ierr);
}

void SOLOUT(int *NR,double *XOLD,double *X, double *Y,double *CONT,double *LRC,int *N,double *RPAR,void *IPAR,int *IRTRN)
{
  TS             ts = (TS) IPAR;
  TS_Radau5      *cvode = (TS_Radau5*)ts->data;
  PetscErrorCode ierr;

  ierr = VecPlaceArray(cvode->work,Y);CHKERRABORT(PETSC_COMM_SELF,ierr);
  ts->time_step = *X - *XOLD;
  ierr = TSMonitor(ts,*NR-1,*X,cvode->work);CHKERRABORT(PETSC_COMM_SELF,ierr);
  ierr = VecResetArray(cvode->work);CHKERRABORT(PETSC_COMM_SELF,ierr);
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

  ierr = PetscCalloc2(LWORK,&WORK,LIWORK,&IWORK);CHKERRQ(ierr);

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

  ierr = PetscFree2(WORK,IWORK);CHKERRQ(ierr);
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

  ierr = PetscNewLog(ts,&cvode);CHKERRQ(ierr);
  ts->data = (void*)cvode;
  PetscFunctionReturn(0);
}
