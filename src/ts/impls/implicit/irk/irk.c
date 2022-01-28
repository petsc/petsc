/*
  Code for timestepping with implicit Runge-Kutta method

  Notes:
  The general system is written as

  F(t,U,Udot) = 0

*/
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#include <petscdm.h>
#include <petscdt.h>

static TSIRKType         TSIRKDefault = TSIRKGAUSS;
static PetscBool         TSIRKRegisterAllCalled;
static PetscBool         TSIRKPackageInitialized;
static PetscFunctionList TSIRKList;

struct _IRKTableau{
  PetscReal   *A,*b,*c;
  PetscScalar *A_inv,*A_inv_rowsum,*I_s;
  PetscReal   *binterp;   /* Dense output formula */
};

typedef struct _IRKTableau *IRKTableau;

typedef struct {
  char         *method_name;
  PetscInt     order;            /* Classical approximation order of the method */
  PetscInt     nstages;          /* Number of stages */
  PetscBool    stiffly_accurate;
  PetscInt     pinterp;          /* Interpolation order */
  IRKTableau   tableau;
  Vec          U0;               /* Backup vector */
  Vec          Z;                /* Combined stage vector */
  Vec          *Y;               /* States computed during the step */
  Vec          Ydot;             /* Work vector holding time derivatives during residual evaluation */
  Vec          U;                /* U is used to compute Ydot = shift(Y-U) */
  Vec          *YdotI;           /* Work vectors to hold the residual evaluation */
  Mat          TJ;               /* KAIJ matrix for the Jacobian of the combined system */
  PetscScalar  *work;            /* Scalar work */
  TSStepStatus status;
  PetscBool    rebuild_completion;
  PetscReal    ccfl;
} TS_IRK;

/*@C
   TSIRKTableauCreate - create the tableau for TSIRK and provide the entries

   Not Collective

   Input Parameters:
+  ts - timestepping context
.  nstages - number of stages, this is the dimension of the matrices below
.  A - stage coefficients (dimension nstages*nstages, row-major)
.  b - step completion table (dimension nstages)
.  c - abscissa (dimension nstages)
.  binterp - coefficients of the interpolation formula (dimension nstages)
.  A_inv - inverse of A (dimension nstages*nstages, row-major)
.  A_inv_rowsum - row sum of the inverse of A (dimension nstages)
-  I_s - identity matrix (dimension nstages*nstages)

   Level: advanced

.seealso: TSIRK, TSIRKRegister()
@*/
PetscErrorCode TSIRKTableauCreate(TS ts,PetscInt nstages,const PetscReal *A,const PetscReal *b,const PetscReal *c,const PetscReal *binterp,const PetscScalar *A_inv,const PetscScalar *A_inv_rowsum,const PetscScalar *I_s)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;
  IRKTableau     tab = irk->tableau;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  irk->order = nstages;
  ierr = PetscMalloc3(PetscSqr(nstages),&tab->A,PetscSqr(nstages),&tab->A_inv,PetscSqr(nstages),&tab->I_s);CHKERRQ(ierr);
  ierr = PetscMalloc4(nstages,&tab->b,nstages,&tab->c,nstages,&tab->binterp,nstages,&tab->A_inv_rowsum);CHKERRQ(ierr);
  ierr = PetscArraycpy(tab->A,A,PetscSqr(nstages));CHKERRQ(ierr);
  ierr = PetscArraycpy(tab->b,b,nstages);CHKERRQ(ierr);
  ierr = PetscArraycpy(tab->c,c,nstages);CHKERRQ(ierr);
  /* optional coefficient arrays */
  if (binterp) {
    ierr = PetscArraycpy(tab->binterp,binterp,nstages);CHKERRQ(ierr);
  }
  if (A_inv) {
    ierr = PetscArraycpy(tab->A_inv,A_inv,PetscSqr(nstages));CHKERRQ(ierr);
  }
  if (A_inv_rowsum) {
    ierr = PetscArraycpy(tab->A_inv_rowsum,A_inv_rowsum,nstages);CHKERRQ(ierr);
  }
  if (I_s) {
    ierr = PetscArraycpy(tab->I_s,I_s,PetscSqr(nstages));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Arrays should be freed with PetscFree3(A,b,c) */
static PetscErrorCode TSIRKCreate_Gauss(TS ts)
{
  PetscInt       nstages;
  PetscReal      *gauss_A_real,*gauss_b,*b,*gauss_c;
  PetscScalar    *gauss_A,*gauss_A_inv,*gauss_A_inv_rowsum,*I_s;
  PetscScalar    *G0,*G1;
  PetscInt       i,j;
  Mat            G0mat,G1mat,Amat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSIRKGetNumStages(ts,&nstages);CHKERRQ(ierr);
  ierr = PetscMalloc3(PetscSqr(nstages),&gauss_A_real,nstages,&gauss_b,nstages,&gauss_c);CHKERRQ(ierr);
  ierr = PetscMalloc4(PetscSqr(nstages),&gauss_A,PetscSqr(nstages),&gauss_A_inv,nstages,&gauss_A_inv_rowsum,PetscSqr(nstages),&I_s);CHKERRQ(ierr);
  ierr = PetscMalloc3(nstages,&b,PetscSqr(nstages),&G0,PetscSqr(nstages),&G1);CHKERRQ(ierr);
  ierr = PetscDTGaussQuadrature(nstages,0.,1.,gauss_c,b);CHKERRQ(ierr);
  for (i=0; i<nstages; i++) gauss_b[i] = b[i]; /* copy to possibly-complex array */

  /* A^T = G0^{-1} G1 */
  for (i=0; i<nstages; i++) {
    for (j=0; j<nstages; j++) {
      G0[i*nstages+j] = PetscPowRealInt(gauss_c[i],j);
      G1[i*nstages+j] = PetscPowRealInt(gauss_c[i],j+1)/(j+1);
    }
  }
  /* The arrays above are row-aligned, but we create dense matrices as the transpose */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nstages,nstages,G0,&G0mat);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nstages,nstages,G1,&G1mat);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nstages,nstages,gauss_A,&Amat);CHKERRQ(ierr);
  ierr = MatLUFactor(G0mat,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = MatMatSolve(G0mat,G1mat,Amat);CHKERRQ(ierr);
  ierr = MatTranspose(Amat,MAT_INPLACE_MATRIX,&Amat);CHKERRQ(ierr);
  for (i=0; i<nstages; i++)
    for (j=0; j<nstages; j++)
      gauss_A_real[i*nstages+j] = PetscRealPart(gauss_A[i*nstages+j]);

  ierr = MatDestroy(&G0mat);CHKERRQ(ierr);
  ierr = MatDestroy(&G1mat);CHKERRQ(ierr);
  ierr = MatDestroy(&Amat);CHKERRQ(ierr);
  ierr = PetscFree3(b,G0,G1);CHKERRQ(ierr);

  {/* Invert A */
    /* PETSc does not provide a routine to calculate the inverse of a general matrix.
     * To get the inverse of A, we form a sequential BAIJ matrix from it, consisting of a single block with block size
     * equal to the dimension of A, and then use MatInvertBlockDiagonal(). */
    Mat               A_baij;
    PetscInt          idxm[1]={0},idxn[1]={0};
    const PetscScalar *A_inv;

    ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,nstages,nstages,nstages,1,NULL,&A_baij);CHKERRQ(ierr);
    ierr = MatSetOption(A_baij,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetValuesBlocked(A_baij,1,idxm,1,idxn,gauss_A,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A_baij,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A_baij,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatInvertBlockDiagonal(A_baij,&A_inv);CHKERRQ(ierr);
    ierr = PetscMemcpy(gauss_A_inv,A_inv,nstages*nstages*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = MatDestroy(&A_baij);CHKERRQ(ierr);
  }

  /* Compute row sums A_inv_rowsum and identity I_s */
  for (i=0; i<nstages; i++) {
    gauss_A_inv_rowsum[i] = 0;
    for (j=0; j<nstages; j++) {
      gauss_A_inv_rowsum[i] += gauss_A_inv[i+nstages*j];
      I_s[i+nstages*j] = 1.*(i == j);
    }
  }
  ierr = TSIRKTableauCreate(ts,nstages,gauss_A_real,gauss_b,gauss_c,NULL,gauss_A_inv,gauss_A_inv_rowsum,I_s);CHKERRQ(ierr);
  ierr = PetscFree3(gauss_A_real,gauss_b,gauss_c);CHKERRQ(ierr);
  ierr = PetscFree4(gauss_A,gauss_A_inv,gauss_A_inv_rowsum,I_s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSIRKRegister -  adds a TSIRK implementation

   Not Collective

   Input Parameters:
+  sname - name of user-defined IRK scheme
-  function - function to create method context

   Notes:
   TSIRKRegister() may be called multiple times to add several user-defined families.

   Sample usage:
.vb
   TSIRKRegister("my_scheme",MySchemeCreate);
.ve

   Then, your scheme can be chosen with the procedural interface via
$     TSIRKSetType(ts,"my_scheme")
   or at runtime via the option
$     -ts_irk_type my_scheme

   Level: advanced

.seealso: TSIRKRegisterAll()
@*/
PetscErrorCode TSIRKRegister(const char sname[],PetscErrorCode (*function)(TS))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSIRKInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&TSIRKList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSIRKRegisterAll - Registers all of the implicit Runge-Kutta methods in TSIRK

  Not Collective, but should be called by all processes which will need the schemes to be registered

  Level: advanced

.seealso:  TSIRKRegisterDestroy()
@*/
PetscErrorCode TSIRKRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSIRKRegisterAllCalled) PetscFunctionReturn(0);
  TSIRKRegisterAllCalled = PETSC_TRUE;

  ierr = TSIRKRegister(TSIRKGAUSS,TSIRKCreate_Gauss);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSIRKRegisterDestroy - Frees the list of schemes that were registered by TSIRKRegister().

   Not Collective

   Level: advanced

.seealso: TSIRKRegister(), TSIRKRegisterAll()
@*/
PetscErrorCode TSIRKRegisterDestroy(void)
{
  PetscFunctionBegin;
  TSIRKRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  TSIRKInitializePackage - This function initializes everything in the TSIRK package. It is called
  from TSInitializePackage().

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode TSIRKInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSIRKPackageInitialized) PetscFunctionReturn(0);
  TSIRKPackageInitialized = PETSC_TRUE;
  ierr = TSIRKRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(TSIRKFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSIRKFinalizePackage - This function destroys everything in the TSIRK package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode TSIRKFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&TSIRKList);CHKERRQ(ierr);
  TSIRKPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*
 This function can be called before or after ts->vec_sol has been updated.
*/
static PetscErrorCode TSEvaluateStep_IRK(TS ts,PetscInt order,Vec U,PetscBool *done)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;
  IRKTableau     tab = irk->tableau;
  Vec            *YdotI = irk->YdotI;
  PetscScalar    *w = irk->work;
  PetscReal      h;
  PetscInt       j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (irk->status) {
  case TS_STEP_INCOMPLETE:
  case TS_STEP_PENDING:
    h = ts->time_step; break;
  case TS_STEP_COMPLETE:
    h = ts->ptime - ts->ptime_prev; break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Invalid TSStepStatus");
  }

  ierr = VecCopy(ts->vec_sol,U);CHKERRQ(ierr);
  for (j=0; j<irk->nstages; j++) w[j] = h*tab->b[j];
  ierr = VecMAXPY(U,irk->nstages,w,YdotI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSRollBack_IRK(TS ts)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(irk->U0,ts->vec_sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_IRK(TS ts)
{
  TS_IRK          *irk = (TS_IRK*)ts->data;
  IRKTableau      tab = irk->tableau;
  PetscScalar     *A_inv = tab->A_inv,*A_inv_rowsum = tab->A_inv_rowsum;
  const PetscInt  nstages = irk->nstages;
  SNES            snes;
  PetscInt        i,j,its,lits,bs;
  TSAdapt         adapt;
  PetscInt        rejections = 0;
  PetscBool       accept = PETSC_TRUE;
  PetscReal       next_time_step = ts->time_step;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!ts->steprollback) {
    ierr = VecCopy(ts->vec_sol,irk->U0);CHKERRQ(ierr);
  }
  ierr = VecGetBlockSize(ts->vec_sol,&bs);CHKERRQ(ierr);
  for (i=0; i<nstages; i++) {
    ierr = VecStrideScatter(ts->vec_sol,i*bs,irk->Z,INSERT_VALUES);CHKERRQ(ierr);
  }

  irk->status = TS_STEP_INCOMPLETE;
  while (!ts->reason && irk->status != TS_STEP_COMPLETE) {
    ierr = VecCopy(ts->vec_sol,irk->U);CHKERRQ(ierr);
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,irk->Z);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = SNESGetLinearSolveIterations(snes,&lits);CHKERRQ(ierr);
    ts->snes_its += its; ts->ksp_its += lits;
    ierr = VecStrideGatherAll(irk->Z,irk->Y,INSERT_VALUES);CHKERRQ(ierr);
    for (i=0; i<nstages; i++) {
      ierr = VecZeroEntries(irk->YdotI[i]);CHKERRQ(ierr);
      for (j=0; j<nstages; j++) {
        ierr = VecAXPY(irk->YdotI[i],A_inv[i+j*nstages]/ts->time_step,irk->Y[j]);CHKERRQ(ierr);
      }
      ierr = VecAXPY(irk->YdotI[i],-A_inv_rowsum[i]/ts->time_step,irk->U);CHKERRQ(ierr);
    }
    irk->status = TS_STEP_INCOMPLETE;
    ierr = TSEvaluateStep_IRK(ts,irk->order,ts->vec_sol,NULL);CHKERRQ(ierr);
    irk->status = TS_STEP_PENDING;
    ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
    ierr = TSAdaptChoose(adapt,ts,ts->time_step,NULL,&next_time_step,&accept);CHKERRQ(ierr);
    irk->status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) {
      ierr = TSRollBack_IRK(ts);CHKERRQ(ierr);
      ts->time_step = next_time_step;
      goto reject_step;
    }

    ts->ptime += ts->time_step;
    ts->time_step = next_time_step;
    break;
  reject_step:
    ts->reject++; accept = PETSC_FALSE;
    if (!ts->reason && ++rejections > ts->max_reject && ts->max_reject >= 0) {
      ts->reason = TS_DIVERGED_STEP_REJECTED;
      ierr = PetscInfo(ts,"Step=%D, step rejections %D greater than current TS allowed, stopping solve\n",ts->steps,rejections);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_IRK(TS ts,PetscReal itime,Vec U)
{
  TS_IRK          *irk = (TS_IRK*)ts->data;
  PetscInt        nstages = irk->nstages,pinterp = irk->pinterp,i,j;
  PetscReal       h;
  PetscReal       tt,t;
  PetscScalar     *bt;
  const PetscReal *B = irk->tableau->binterp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscAssertFalse(!B,PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSIRK %s does not have an interpolation formula",irk->method_name);
  switch (irk->status) {
  case TS_STEP_INCOMPLETE:
  case TS_STEP_PENDING:
    h = ts->time_step;
    t = (itime - ts->ptime)/h;
    break;
  case TS_STEP_COMPLETE:
    h = ts->ptime - ts->ptime_prev;
    t = (itime - ts->ptime)/h + 1; /* In the interval [0,1] */
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Invalid TSStepStatus");
  }
  ierr = PetscMalloc1(nstages,&bt);CHKERRQ(ierr);
  for (i=0; i<nstages; i++) bt[i] = 0;
  for (j=0,tt=t; j<pinterp; j++,tt*=t) {
    for (i=0; i<nstages; i++) {
      bt[i] += h * B[i*pinterp+j] * tt;
    }
  }
  ierr = VecMAXPY(U,nstages,bt,irk->YdotI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSIRKTableauReset(TS ts)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;
  IRKTableau     tab = irk->tableau;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tab) PetscFunctionReturn(0);
  ierr = PetscFree3(tab->A,tab->A_inv,tab->I_s);CHKERRQ(ierr);
  ierr = PetscFree4(tab->b,tab->c,tab->binterp,tab->A_inv_rowsum);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_IRK(TS ts)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSIRKTableauReset(ts);CHKERRQ(ierr);
  if (irk->tableau) {
    ierr = PetscFree(irk->tableau);CHKERRQ(ierr);
  }
  if (irk->method_name) {
    ierr = PetscFree(irk->method_name);CHKERRQ(ierr);
  }
  if (irk->work) {
    ierr = PetscFree(irk->work);CHKERRQ(ierr);
  }
  ierr = VecDestroyVecs(irk->nstages,&irk->Y);CHKERRQ(ierr);
  ierr = VecDestroyVecs(irk->nstages,&irk->YdotI);CHKERRQ(ierr);
  ierr = VecDestroy(&irk->Ydot);CHKERRQ(ierr);
  ierr = VecDestroy(&irk->Z);CHKERRQ(ierr);
  ierr = VecDestroy(&irk->U);CHKERRQ(ierr);
  ierr = VecDestroy(&irk->U0);CHKERRQ(ierr);
  ierr = MatDestroy(&irk->TJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSIRKGetVecs(TS ts,DM dm,Vec *U)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (U) {
    if (dm && dm != ts->dm) {
      ierr = DMGetNamedGlobalVector(dm,"TSIRK_U",U);CHKERRQ(ierr);
    } else *U = irk->U;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSIRKRestoreVecs(TS ts,DM dm,Vec *U)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (U) {
    if (dm && dm != ts->dm) {
      ierr = DMRestoreNamedGlobalVector(dm,"TSIRK_U",U);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
  This defines the nonlinear equations that is to be solved with SNES
    G[e\otimes t + C*dt, Z, Zdot] = 0
    Zdot = (In \otimes S)*Z - (In \otimes Se) U
  where S = 1/(dt*A)
*/
static PetscErrorCode SNESTSFormFunction_IRK(SNES snes,Vec ZC,Vec FC,TS ts)
{
  TS_IRK            *irk = (TS_IRK*)ts->data;
  IRKTableau        tab  = irk->tableau;
  const PetscInt    nstages = irk->nstages;
  const PetscReal   *c = tab->c;
  const PetscScalar *A_inv = tab->A_inv,*A_inv_rowsum = tab->A_inv_rowsum;
  DM                dm,dmsave;
  Vec               U,*YdotI = irk->YdotI,Ydot = irk->Ydot,*Y = irk->Y;
  PetscReal         h = ts->time_step;
  PetscInt          i,j;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = TSIRKGetVecs(ts,dm,&U);CHKERRQ(ierr);
  ierr = VecStrideGatherAll(ZC,Y,INSERT_VALUES);CHKERRQ(ierr);
  dmsave = ts->dm;
  ts->dm = dm;
  for (i=0; i<nstages; i++) {
    ierr = VecZeroEntries(Ydot);CHKERRQ(ierr);
    for (j=0; j<nstages; j++) {
      ierr = VecAXPY(Ydot,A_inv[j*nstages+i]/h,Y[j]);CHKERRQ(ierr);
    }
    ierr = VecAXPY(Ydot,-A_inv_rowsum[i]/h,U);CHKERRQ(ierr); /* Ydot = (S \otimes In)*Z - (Se \otimes In) U */
    ierr = TSComputeIFunction(ts,ts->ptime+ts->time_step*c[i],Y[i],Ydot,YdotI[i],PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = VecStrideScatterAll(YdotI,FC,INSERT_VALUES);CHKERRQ(ierr);
  ts->dm = dmsave;
  ierr   = TSIRKRestoreVecs(ts,dm,&U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   For explicit ODE, the Jacobian is
     JC = I_n \otimes S - J \otimes I_s
   For DAE, the Jacobian is
     JC = M_n \otimes S - J \otimes I_s
*/
static PetscErrorCode SNESTSFormJacobian_IRK(SNES snes,Vec ZC,Mat JC,Mat JCpre,TS ts)
{
  TS_IRK          *irk = (TS_IRK*)ts->data;
  IRKTableau      tab  = irk->tableau;
  const PetscInt  nstages = irk->nstages;
  const PetscReal *c = tab->c;
  DM              dm,dmsave;
  Vec             *Y = irk->Y,Ydot = irk->Ydot;
  Mat             J;
  PetscScalar     *S;
  PetscInt        i,j,bs;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  /* irk->Ydot has already been computed in SNESTSFormFunction_IRK (SNES guarantees this) */
  dmsave = ts->dm;
  ts->dm = dm;
  ierr = VecGetBlockSize(Y[nstages-1],&bs);CHKERRQ(ierr);
  if (ts->equation_type <= TS_EQ_ODE_EXPLICIT) { /* Support explicit formulas only */
    ierr = VecStrideGather(ZC,(nstages-1)*bs,Y[nstages-1],INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatKAIJGetAIJ(JC,&J);CHKERRQ(ierr);
    ierr = TSComputeIJacobian(ts,ts->ptime+ts->time_step*c[nstages-1],Y[nstages-1],Ydot,0,J,J,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatKAIJGetS(JC,NULL,NULL,&S);CHKERRQ(ierr);
    for (i=0; i<nstages; i++)
      for (j=0; j<nstages; j++)
        S[i+nstages*j] = tab->A_inv[i+nstages*j]/ts->time_step;
    ierr = MatKAIJRestoreS(JC,&S);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSIRK %s does not support implicit formula",irk->method_name); /* TODO: need the mass matrix for DAE  */
  ts->dm = dmsave;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsenHook_TSIRK(DM fine,DM coarse,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRestrictHook_TSIRK(DM fine,Mat restrct,Vec rscale,Mat inject,DM coarse,void *ctx)
{
  TS             ts = (TS)ctx;
  Vec            U,U_c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSIRKGetVecs(ts,fine,&U);CHKERRQ(ierr);
  ierr = TSIRKGetVecs(ts,coarse,&U_c);CHKERRQ(ierr);
  ierr = MatRestrict(restrct,U,U_c);CHKERRQ(ierr);
  ierr = VecPointwiseMult(U_c,rscale,U_c);CHKERRQ(ierr);
  ierr = TSIRKRestoreVecs(ts,fine,&U);CHKERRQ(ierr);
  ierr = TSIRKRestoreVecs(ts,coarse,&U_c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainHook_TSIRK(DM dm,DM subdm,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainRestrictHook_TSIRK(DM dm,VecScatter gscat,VecScatter lscat,DM subdm,void *ctx)
{
  TS             ts = (TS)ctx;
  Vec            U,U_c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSIRKGetVecs(ts,dm,&U);CHKERRQ(ierr);
  ierr = TSIRKGetVecs(ts,subdm,&U_c);CHKERRQ(ierr);

  ierr = VecScatterBegin(gscat,U,U_c,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(gscat,U,U_c,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = TSIRKRestoreVecs(ts,dm,&U);CHKERRQ(ierr);
  ierr = TSIRKRestoreVecs(ts,subdm,&U_c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_IRK(TS ts)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;
  IRKTableau     tab = irk->tableau;
  DM             dm;
  Mat            J;
  Vec            R;
  const PetscInt nstages = irk->nstages;
  PetscInt       vsize,bs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!irk->work) {
    ierr = PetscMalloc1(irk->nstages,&irk->work);CHKERRQ(ierr);
  }
  if (!irk->Y) {
    ierr = VecDuplicateVecs(ts->vec_sol,irk->nstages,&irk->Y);CHKERRQ(ierr);
  }
  if (!irk->YdotI) {
    ierr = VecDuplicateVecs(ts->vec_sol,irk->nstages,&irk->YdotI);CHKERRQ(ierr);
  }
  if (!irk->Ydot) {
    ierr = VecDuplicate(ts->vec_sol,&irk->Ydot);CHKERRQ(ierr);
  }
  if (!irk->U) {
    ierr = VecDuplicate(ts->vec_sol,&irk->U);CHKERRQ(ierr);
  }
  if (!irk->U0) {
    ierr = VecDuplicate(ts->vec_sol,&irk->U0);CHKERRQ(ierr);
  }
  if (!irk->Z) {
    ierr = VecCreate(PetscObjectComm((PetscObject)ts->vec_sol),&irk->Z);CHKERRQ(ierr);
    ierr = VecGetSize(ts->vec_sol,&vsize);CHKERRQ(ierr);
    ierr = VecSetSizes(irk->Z,PETSC_DECIDE,vsize*irk->nstages);CHKERRQ(ierr);
    ierr = VecGetBlockSize(ts->vec_sol,&bs);CHKERRQ(ierr);
    ierr = VecSetBlockSize(irk->Z,irk->nstages*bs);CHKERRQ(ierr);
    ierr = VecSetFromOptions(irk->Z);CHKERRQ(ierr);
  }
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_TSIRK,DMRestrictHook_TSIRK,ts);CHKERRQ(ierr);
  ierr = DMSubDomainHookAdd(dm,DMSubDomainHook_TSIRK,DMSubDomainRestrictHook_TSIRK,ts);CHKERRQ(ierr);

  ierr = TSGetSNES(ts,&ts->snes);CHKERRQ(ierr);
  ierr = VecDuplicate(irk->Z,&R);CHKERRQ(ierr);
  ierr = SNESSetFunction(ts->snes,R,SNESTSFormFunction,ts);CHKERRQ(ierr);
  ierr = TSGetIJacobian(ts,&J,NULL,NULL,NULL);CHKERRQ(ierr);
  if (!irk->TJ) {
    /* Create the KAIJ matrix for solving the stages */
    ierr = MatCreateKAIJ(J,nstages,nstages,tab->A_inv,tab->I_s,&irk->TJ);CHKERRQ(ierr);
  }
  ierr = SNESSetJacobian(ts->snes,irk->TJ,irk->TJ,SNESTSFormJacobian,ts);CHKERRQ(ierr);
  ierr = VecDestroy(&R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_IRK(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;
  char           tname[256] = TSIRKGAUSS;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"IRK ODE solver options");CHKERRQ(ierr);
  {
    PetscBool flg1,flg2;
    ierr = PetscOptionsInt("-ts_irk_nstages","Stages of the IRK method","TSIRKSetNumStages",irk->nstages,&irk->nstages,&flg1);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-ts_irk_type","Type of IRK method","TSIRKSetType",TSIRKList,irk->method_name[0] ? irk->method_name : tname,tname,sizeof(tname),&flg2);CHKERRQ(ierr);
    if (flg1 ||flg2 || !irk->method_name[0]) { /* Create the method tableau after nstages or method is set */
      ierr = TSIRKSetType(ts,tname);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_IRK(TS ts,PetscViewer viewer)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    IRKTableau    tab = irk->tableau;
    TSIRKType irktype;
    char          buf[512];

    ierr = TSIRKGetType(ts,&irktype);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  IRK type %s\n",irktype);CHKERRQ(ierr);
    ierr = PetscFormatRealArray(buf,sizeof(buf),"% 8.6f",irk->nstages,tab->c);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Abscissa       c = %s\n",buf);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Stiffly accurate: %s\n",irk->stiffly_accurate ? "yes" : "no");CHKERRQ(ierr);
    ierr = PetscFormatRealArray(buf,sizeof(buf),"% 8.6f",PetscSqr(irk->nstages),tab->A);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  A coefficients       A = %s\n",buf);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSLoad_IRK(TS ts,PetscViewer viewer)
{
  PetscErrorCode ierr;
  SNES           snes;
  TSAdapt        adapt;

  PetscFunctionBegin;
  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptLoad(adapt,viewer);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESLoad(snes,viewer);CHKERRQ(ierr);
  /* function and Jacobian context for SNES when used with TS is always ts object */
  ierr = SNESSetFunction(snes,NULL,NULL,ts);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,NULL,NULL,NULL,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSIRKSetType - Set the type of IRK scheme

  Logically collective

  Input Parameters:
+  ts - timestepping context
-  irktype - type of IRK scheme

  Options Database:
.  -ts_irk_type <gauss>

  Level: intermediate

.seealso: TSIRKGetType(), TSIRK, TSIRKType, TSIRKGAUSS
@*/
PetscErrorCode TSIRKSetType(TS ts,TSIRKType irktype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidCharPointer(irktype,2);
  ierr = PetscTryMethod(ts,"TSIRKSetType_C",(TS,TSIRKType),(ts,irktype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSIRKGetType - Get the type of IRK IMEX scheme

  Logically collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  irktype - type of IRK-IMEX scheme

  Level: intermediate

.seealso: TSIRKGetType()
@*/
PetscErrorCode TSIRKGetType(TS ts,TSIRKType *irktype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscUseMethod(ts,"TSIRKGetType_C",(TS,TSIRKType*),(ts,irktype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSIRKSetNumStages - Set the number of stages of IRK scheme

  Logically collective

  Input Parameters:
+  ts - timestepping context
-  nstages - number of stages of IRK scheme

  Options Database:
.  -ts_irk_nstages <int>: Number of stages

  Level: intermediate

.seealso: TSIRKGetNumStages(), TSIRK
@*/
PetscErrorCode TSIRKSetNumStages(TS ts,PetscInt nstages)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSIRKSetNumStages_C",(TS,PetscInt),(ts,nstages));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSIRKGetNumStages - Get the number of stages of IRK scheme

  Logically collective

  Input Parameters:
+  ts - timestepping context
-  nstages - number of stages of IRK scheme

  Level: intermediate

.seealso: TSIRKSetNumStages(), TSIRK
@*/
PetscErrorCode TSIRKGetNumStages(TS ts,PetscInt *nstages)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(nstages,2);
  ierr = PetscTryMethod(ts,"TSIRKGetNumStages_C",(TS,PetscInt*),(ts,nstages));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSIRKGetType_IRK(TS ts,TSIRKType *irktype)
{
  TS_IRK *irk = (TS_IRK*)ts->data;

  PetscFunctionBegin;
  *irktype = irk->method_name;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSIRKSetType_IRK(TS ts,TSIRKType irktype)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;
  PetscErrorCode ierr,(*irkcreate)(TS);

  PetscFunctionBegin;
  if (irk->method_name) {
    ierr = PetscFree(irk->method_name);CHKERRQ(ierr);
    ierr = TSIRKTableauReset(ts);CHKERRQ(ierr);
  }
  ierr = PetscFunctionListFind(TSIRKList,irktype,&irkcreate);CHKERRQ(ierr);
  PetscAssertFalse(!irkcreate,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TSIRK type \"%s\" given",irktype);
  ierr = (*irkcreate)(ts);CHKERRQ(ierr);
  ierr = PetscStrallocpy(irktype,&irk->method_name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSIRKSetNumStages_IRK(TS ts,PetscInt nstages)
{
  TS_IRK *irk = (TS_IRK*)ts->data;

  PetscFunctionBegin;
  PetscAssertFalse(nstages<=0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"input argument, %d, out of range",nstages);
  irk->nstages = nstages;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSIRKGetNumStages_IRK(TS ts,PetscInt *nstages)
{
  TS_IRK *irk = (TS_IRK*)ts->data;

  PetscFunctionBegin;
  PetscValidIntPointer(nstages,2);
  *nstages = irk->nstages;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_IRK(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_IRK(ts);CHKERRQ(ierr);
  if (ts->dm) {
    ierr = DMCoarsenHookRemove(ts->dm,DMCoarsenHook_TSIRK,DMRestrictHook_TSIRK,ts);CHKERRQ(ierr);
    ierr = DMSubDomainHookRemove(ts->dm,DMSubDomainHook_TSIRK,DMSubDomainRestrictHook_TSIRK,ts);CHKERRQ(ierr);
  }
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSIRKSetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSIRKGetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSIRKSetNumStages_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSIRKGetNumStages_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      TSIRK - ODE and DAE solver using Implicit Runge-Kutta schemes

  Notes:

  TSIRK uses the sparse Kronecker product matrix implementation of MATKAIJ to achieve good arithmetic intensity.

  Gauss-Legrendre methods are currently supported. These are A-stable symplectic methods with an arbitrary number of stages. The order of accuracy is 2s when using s stages. The default method uses three stages and thus has an order of six. The number of stages (thus order) can be set with -ts_irk_nstages or TSIRKSetNumStages().

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSIRKSetType(), TSIRKGetType(), TSIRKGAUSS, TSIRKRegister(), TSIRKSetNumStages()

M*/
PETSC_EXTERN PetscErrorCode TSCreate_IRK(TS ts)
{
  TS_IRK         *irk;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSIRKInitializePackage();CHKERRQ(ierr);

  ts->ops->reset          = TSReset_IRK;
  ts->ops->destroy        = TSDestroy_IRK;
  ts->ops->view           = TSView_IRK;
  ts->ops->load           = TSLoad_IRK;
  ts->ops->setup          = TSSetUp_IRK;
  ts->ops->step           = TSStep_IRK;
  ts->ops->interpolate    = TSInterpolate_IRK;
  ts->ops->evaluatestep   = TSEvaluateStep_IRK;
  ts->ops->rollback       = TSRollBack_IRK;
  ts->ops->setfromoptions = TSSetFromOptions_IRK;
  ts->ops->snesfunction   = SNESTSFormFunction_IRK;
  ts->ops->snesjacobian   = SNESTSFormJacobian_IRK;

  ts->usessnes = PETSC_TRUE;

  ierr = PetscNewLog(ts,&irk);CHKERRQ(ierr);
  ts->data = (void*)irk;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSIRKSetType_C",TSIRKSetType_IRK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSIRKGetType_C",TSIRKGetType_IRK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSIRKSetNumStages_C",TSIRKSetNumStages_IRK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSIRKGetNumStages_C",TSIRKGetNumStages_IRK);CHKERRQ(ierr);
  /* 3-stage IRK_Gauss is the default */
  ierr = PetscNew(&irk->tableau);CHKERRQ(ierr);
  irk->nstages = 3;
  ierr = TSIRKSetType(ts,TSIRKDefault);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
