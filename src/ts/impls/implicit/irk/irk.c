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

  PetscFunctionBegin;
  irk->order = nstages;
  CHKERRQ(PetscMalloc3(PetscSqr(nstages),&tab->A,PetscSqr(nstages),&tab->A_inv,PetscSqr(nstages),&tab->I_s));
  CHKERRQ(PetscMalloc4(nstages,&tab->b,nstages,&tab->c,nstages,&tab->binterp,nstages,&tab->A_inv_rowsum));
  CHKERRQ(PetscArraycpy(tab->A,A,PetscSqr(nstages)));
  CHKERRQ(PetscArraycpy(tab->b,b,nstages));
  CHKERRQ(PetscArraycpy(tab->c,c,nstages));
  /* optional coefficient arrays */
  if (binterp) {
    CHKERRQ(PetscArraycpy(tab->binterp,binterp,nstages));
  }
  if (A_inv) {
    CHKERRQ(PetscArraycpy(tab->A_inv,A_inv,PetscSqr(nstages)));
  }
  if (A_inv_rowsum) {
    CHKERRQ(PetscArraycpy(tab->A_inv_rowsum,A_inv_rowsum,nstages));
  }
  if (I_s) {
    CHKERRQ(PetscArraycpy(tab->I_s,I_s,PetscSqr(nstages)));
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

  PetscFunctionBegin;
  CHKERRQ(TSIRKGetNumStages(ts,&nstages));
  CHKERRQ(PetscMalloc3(PetscSqr(nstages),&gauss_A_real,nstages,&gauss_b,nstages,&gauss_c));
  CHKERRQ(PetscMalloc4(PetscSqr(nstages),&gauss_A,PetscSqr(nstages),&gauss_A_inv,nstages,&gauss_A_inv_rowsum,PetscSqr(nstages),&I_s));
  CHKERRQ(PetscMalloc3(nstages,&b,PetscSqr(nstages),&G0,PetscSqr(nstages),&G1));
  CHKERRQ(PetscDTGaussQuadrature(nstages,0.,1.,gauss_c,b));
  for (i=0; i<nstages; i++) gauss_b[i] = b[i]; /* copy to possibly-complex array */

  /* A^T = G0^{-1} G1 */
  for (i=0; i<nstages; i++) {
    for (j=0; j<nstages; j++) {
      G0[i*nstages+j] = PetscPowRealInt(gauss_c[i],j);
      G1[i*nstages+j] = PetscPowRealInt(gauss_c[i],j+1)/(j+1);
    }
  }
  /* The arrays above are row-aligned, but we create dense matrices as the transpose */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,nstages,nstages,G0,&G0mat));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,nstages,nstages,G1,&G1mat));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,nstages,nstages,gauss_A,&Amat));
  CHKERRQ(MatLUFactor(G0mat,NULL,NULL,NULL));
  CHKERRQ(MatMatSolve(G0mat,G1mat,Amat));
  CHKERRQ(MatTranspose(Amat,MAT_INPLACE_MATRIX,&Amat));
  for (i=0; i<nstages; i++)
    for (j=0; j<nstages; j++)
      gauss_A_real[i*nstages+j] = PetscRealPart(gauss_A[i*nstages+j]);

  CHKERRQ(MatDestroy(&G0mat));
  CHKERRQ(MatDestroy(&G1mat));
  CHKERRQ(MatDestroy(&Amat));
  CHKERRQ(PetscFree3(b,G0,G1));

  {/* Invert A */
    /* PETSc does not provide a routine to calculate the inverse of a general matrix.
     * To get the inverse of A, we form a sequential BAIJ matrix from it, consisting of a single block with block size
     * equal to the dimension of A, and then use MatInvertBlockDiagonal(). */
    Mat               A_baij;
    PetscInt          idxm[1]={0},idxn[1]={0};
    const PetscScalar *A_inv;

    CHKERRQ(MatCreateSeqBAIJ(PETSC_COMM_SELF,nstages,nstages,nstages,1,NULL,&A_baij));
    CHKERRQ(MatSetOption(A_baij,MAT_ROW_ORIENTED,PETSC_FALSE));
    CHKERRQ(MatSetValuesBlocked(A_baij,1,idxm,1,idxn,gauss_A,INSERT_VALUES));
    CHKERRQ(MatAssemblyBegin(A_baij,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A_baij,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatInvertBlockDiagonal(A_baij,&A_inv));
    CHKERRQ(PetscMemcpy(gauss_A_inv,A_inv,nstages*nstages*sizeof(PetscScalar)));
    CHKERRQ(MatDestroy(&A_baij));
  }

  /* Compute row sums A_inv_rowsum and identity I_s */
  for (i=0; i<nstages; i++) {
    gauss_A_inv_rowsum[i] = 0;
    for (j=0; j<nstages; j++) {
      gauss_A_inv_rowsum[i] += gauss_A_inv[i+nstages*j];
      I_s[i+nstages*j] = 1.*(i == j);
    }
  }
  CHKERRQ(TSIRKTableauCreate(ts,nstages,gauss_A_real,gauss_b,gauss_c,NULL,gauss_A_inv,gauss_A_inv_rowsum,I_s));
  CHKERRQ(PetscFree3(gauss_A_real,gauss_b,gauss_c));
  CHKERRQ(PetscFree4(gauss_A,gauss_A_inv,gauss_A_inv_rowsum,I_s));
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
  PetscFunctionBegin;
  CHKERRQ(TSIRKInitializePackage());
  CHKERRQ(PetscFunctionListAdd(&TSIRKList,sname,function));
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
  PetscFunctionBegin;
  if (TSIRKRegisterAllCalled) PetscFunctionReturn(0);
  TSIRKRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(TSIRKRegister(TSIRKGAUSS,TSIRKCreate_Gauss));
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
  PetscFunctionBegin;
  if (TSIRKPackageInitialized) PetscFunctionReturn(0);
  TSIRKPackageInitialized = PETSC_TRUE;
  CHKERRQ(TSIRKRegisterAll());
  CHKERRQ(PetscRegisterFinalize(TSIRKFinalizePackage));
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
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&TSIRKList));
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

  PetscFunctionBegin;
  switch (irk->status) {
  case TS_STEP_INCOMPLETE:
  case TS_STEP_PENDING:
    h = ts->time_step; break;
  case TS_STEP_COMPLETE:
    h = ts->ptime - ts->ptime_prev; break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Invalid TSStepStatus");
  }

  CHKERRQ(VecCopy(ts->vec_sol,U));
  for (j=0; j<irk->nstages; j++) w[j] = h*tab->b[j];
  CHKERRQ(VecMAXPY(U,irk->nstages,w,YdotI));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSRollBack_IRK(TS ts)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;

  PetscFunctionBegin;
  CHKERRQ(VecCopy(irk->U0,ts->vec_sol));
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

  PetscFunctionBegin;
  if (!ts->steprollback) {
    CHKERRQ(VecCopy(ts->vec_sol,irk->U0));
  }
  CHKERRQ(VecGetBlockSize(ts->vec_sol,&bs));
  for (i=0; i<nstages; i++) {
    CHKERRQ(VecStrideScatter(ts->vec_sol,i*bs,irk->Z,INSERT_VALUES));
  }

  irk->status = TS_STEP_INCOMPLETE;
  while (!ts->reason && irk->status != TS_STEP_COMPLETE) {
    CHKERRQ(VecCopy(ts->vec_sol,irk->U));
    CHKERRQ(TSGetSNES(ts,&snes));
    CHKERRQ(SNESSolve(snes,NULL,irk->Z));
    CHKERRQ(SNESGetIterationNumber(snes,&its));
    CHKERRQ(SNESGetLinearSolveIterations(snes,&lits));
    ts->snes_its += its; ts->ksp_its += lits;
    CHKERRQ(VecStrideGatherAll(irk->Z,irk->Y,INSERT_VALUES));
    for (i=0; i<nstages; i++) {
      CHKERRQ(VecZeroEntries(irk->YdotI[i]));
      for (j=0; j<nstages; j++) {
        CHKERRQ(VecAXPY(irk->YdotI[i],A_inv[i+j*nstages]/ts->time_step,irk->Y[j]));
      }
      CHKERRQ(VecAXPY(irk->YdotI[i],-A_inv_rowsum[i]/ts->time_step,irk->U));
    }
    irk->status = TS_STEP_INCOMPLETE;
    CHKERRQ(TSEvaluateStep_IRK(ts,irk->order,ts->vec_sol,NULL));
    irk->status = TS_STEP_PENDING;
    CHKERRQ(TSGetAdapt(ts,&adapt));
    CHKERRQ(TSAdaptChoose(adapt,ts,ts->time_step,NULL,&next_time_step,&accept));
    irk->status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) {
      CHKERRQ(TSRollBack_IRK(ts));
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
      CHKERRQ(PetscInfo(ts,"Step=%D, step rejections %D greater than current TS allowed, stopping solve\n",ts->steps,rejections));
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

  PetscFunctionBegin;
  PetscCheck(B,PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSIRK %s does not have an interpolation formula",irk->method_name);
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
  CHKERRQ(PetscMalloc1(nstages,&bt));
  for (i=0; i<nstages; i++) bt[i] = 0;
  for (j=0,tt=t; j<pinterp; j++,tt*=t) {
    for (i=0; i<nstages; i++) {
      bt[i] += h * B[i*pinterp+j] * tt;
    }
  }
  CHKERRQ(VecMAXPY(U,nstages,bt,irk->YdotI));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSIRKTableauReset(TS ts)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;
  IRKTableau     tab = irk->tableau;

  PetscFunctionBegin;
  if (!tab) PetscFunctionReturn(0);
  CHKERRQ(PetscFree3(tab->A,tab->A_inv,tab->I_s));
  CHKERRQ(PetscFree4(tab->b,tab->c,tab->binterp,tab->A_inv_rowsum));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_IRK(TS ts)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;

  PetscFunctionBegin;
  CHKERRQ(TSIRKTableauReset(ts));
  if (irk->tableau) {
    CHKERRQ(PetscFree(irk->tableau));
  }
  if (irk->method_name) {
    CHKERRQ(PetscFree(irk->method_name));
  }
  if (irk->work) {
    CHKERRQ(PetscFree(irk->work));
  }
  CHKERRQ(VecDestroyVecs(irk->nstages,&irk->Y));
  CHKERRQ(VecDestroyVecs(irk->nstages,&irk->YdotI));
  CHKERRQ(VecDestroy(&irk->Ydot));
  CHKERRQ(VecDestroy(&irk->Z));
  CHKERRQ(VecDestroy(&irk->U));
  CHKERRQ(VecDestroy(&irk->U0));
  CHKERRQ(MatDestroy(&irk->TJ));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSIRKGetVecs(TS ts,DM dm,Vec *U)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;

  PetscFunctionBegin;
  if (U) {
    if (dm && dm != ts->dm) {
      CHKERRQ(DMGetNamedGlobalVector(dm,"TSIRK_U",U));
    } else *U = irk->U;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSIRKRestoreVecs(TS ts,DM dm,Vec *U)
{
  PetscFunctionBegin;
  if (U) {
    if (dm && dm != ts->dm) {
      CHKERRQ(DMRestoreNamedGlobalVector(dm,"TSIRK_U",U));
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

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(TSIRKGetVecs(ts,dm,&U));
  CHKERRQ(VecStrideGatherAll(ZC,Y,INSERT_VALUES));
  dmsave = ts->dm;
  ts->dm = dm;
  for (i=0; i<nstages; i++) {
    CHKERRQ(VecZeroEntries(Ydot));
    for (j=0; j<nstages; j++) {
      CHKERRQ(VecAXPY(Ydot,A_inv[j*nstages+i]/h,Y[j]));
    }
    CHKERRQ(VecAXPY(Ydot,-A_inv_rowsum[i]/h,U)); /* Ydot = (S \otimes In)*Z - (Se \otimes In) U */
    CHKERRQ(TSComputeIFunction(ts,ts->ptime+ts->time_step*c[i],Y[i],Ydot,YdotI[i],PETSC_FALSE));
  }
  CHKERRQ(VecStrideScatterAll(YdotI,FC,INSERT_VALUES));
  ts->dm = dmsave;
  CHKERRQ(TSIRKRestoreVecs(ts,dm,&U));
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

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes,&dm));
  /* irk->Ydot has already been computed in SNESTSFormFunction_IRK (SNES guarantees this) */
  dmsave = ts->dm;
  ts->dm = dm;
  CHKERRQ(VecGetBlockSize(Y[nstages-1],&bs));
  if (ts->equation_type <= TS_EQ_ODE_EXPLICIT) { /* Support explicit formulas only */
    CHKERRQ(VecStrideGather(ZC,(nstages-1)*bs,Y[nstages-1],INSERT_VALUES));
    CHKERRQ(MatKAIJGetAIJ(JC,&J));
    CHKERRQ(TSComputeIJacobian(ts,ts->ptime+ts->time_step*c[nstages-1],Y[nstages-1],Ydot,0,J,J,PETSC_FALSE));
    CHKERRQ(MatKAIJGetS(JC,NULL,NULL,&S));
    for (i=0; i<nstages; i++)
      for (j=0; j<nstages; j++)
        S[i+nstages*j] = tab->A_inv[i+nstages*j]/ts->time_step;
    CHKERRQ(MatKAIJRestoreS(JC,&S));
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

  PetscFunctionBegin;
  CHKERRQ(TSIRKGetVecs(ts,fine,&U));
  CHKERRQ(TSIRKGetVecs(ts,coarse,&U_c));
  CHKERRQ(MatRestrict(restrct,U,U_c));
  CHKERRQ(VecPointwiseMult(U_c,rscale,U_c));
  CHKERRQ(TSIRKRestoreVecs(ts,fine,&U));
  CHKERRQ(TSIRKRestoreVecs(ts,coarse,&U_c));
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

  PetscFunctionBegin;
  CHKERRQ(TSIRKGetVecs(ts,dm,&U));
  CHKERRQ(TSIRKGetVecs(ts,subdm,&U_c));

  CHKERRQ(VecScatterBegin(gscat,U,U_c,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(gscat,U,U_c,INSERT_VALUES,SCATTER_FORWARD));

  CHKERRQ(TSIRKRestoreVecs(ts,dm,&U));
  CHKERRQ(TSIRKRestoreVecs(ts,subdm,&U_c));
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

  PetscFunctionBegin;
  if (!irk->work) {
    CHKERRQ(PetscMalloc1(irk->nstages,&irk->work));
  }
  if (!irk->Y) {
    CHKERRQ(VecDuplicateVecs(ts->vec_sol,irk->nstages,&irk->Y));
  }
  if (!irk->YdotI) {
    CHKERRQ(VecDuplicateVecs(ts->vec_sol,irk->nstages,&irk->YdotI));
  }
  if (!irk->Ydot) {
    CHKERRQ(VecDuplicate(ts->vec_sol,&irk->Ydot));
  }
  if (!irk->U) {
    CHKERRQ(VecDuplicate(ts->vec_sol,&irk->U));
  }
  if (!irk->U0) {
    CHKERRQ(VecDuplicate(ts->vec_sol,&irk->U0));
  }
  if (!irk->Z) {
    CHKERRQ(VecCreate(PetscObjectComm((PetscObject)ts->vec_sol),&irk->Z));
    CHKERRQ(VecGetSize(ts->vec_sol,&vsize));
    CHKERRQ(VecSetSizes(irk->Z,PETSC_DECIDE,vsize*irk->nstages));
    CHKERRQ(VecGetBlockSize(ts->vec_sol,&bs));
    CHKERRQ(VecSetBlockSize(irk->Z,irk->nstages*bs));
    CHKERRQ(VecSetFromOptions(irk->Z));
  }
  CHKERRQ(TSGetDM(ts,&dm));
  CHKERRQ(DMCoarsenHookAdd(dm,DMCoarsenHook_TSIRK,DMRestrictHook_TSIRK,ts));
  CHKERRQ(DMSubDomainHookAdd(dm,DMSubDomainHook_TSIRK,DMSubDomainRestrictHook_TSIRK,ts));

  CHKERRQ(TSGetSNES(ts,&ts->snes));
  CHKERRQ(VecDuplicate(irk->Z,&R));
  CHKERRQ(SNESSetFunction(ts->snes,R,SNESTSFormFunction,ts));
  CHKERRQ(TSGetIJacobian(ts,&J,NULL,NULL,NULL));
  if (!irk->TJ) {
    /* Create the KAIJ matrix for solving the stages */
    CHKERRQ(MatCreateKAIJ(J,nstages,nstages,tab->A_inv,tab->I_s,&irk->TJ));
  }
  CHKERRQ(SNESSetJacobian(ts->snes,irk->TJ,irk->TJ,SNESTSFormJacobian,ts));
  CHKERRQ(VecDestroy(&R));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_IRK(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;
  char           tname[256] = TSIRKGAUSS;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"IRK ODE solver options"));
  {
    PetscBool flg1,flg2;
    CHKERRQ(PetscOptionsInt("-ts_irk_nstages","Stages of the IRK method","TSIRKSetNumStages",irk->nstages,&irk->nstages,&flg1));
    CHKERRQ(PetscOptionsFList("-ts_irk_type","Type of IRK method","TSIRKSetType",TSIRKList,irk->method_name[0] ? irk->method_name : tname,tname,sizeof(tname),&flg2));
    if (flg1 ||flg2 || !irk->method_name[0]) { /* Create the method tableau after nstages or method is set */
      CHKERRQ(TSIRKSetType(ts,tname));
    }
  }
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_IRK(TS ts,PetscViewer viewer)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    IRKTableau    tab = irk->tableau;
    TSIRKType irktype;
    char          buf[512];

    CHKERRQ(TSIRKGetType(ts,&irktype));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  IRK type %s\n",irktype));
    CHKERRQ(PetscFormatRealArray(buf,sizeof(buf),"% 8.6f",irk->nstages,tab->c));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Abscissa       c = %s\n",buf));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Stiffly accurate: %s\n",irk->stiffly_accurate ? "yes" : "no"));
    CHKERRQ(PetscFormatRealArray(buf,sizeof(buf),"% 8.6f",PetscSqr(irk->nstages),tab->A));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  A coefficients       A = %s\n",buf));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSLoad_IRK(TS ts,PetscViewer viewer)
{
  SNES           snes;
  TSAdapt        adapt;

  PetscFunctionBegin;
  CHKERRQ(TSGetAdapt(ts,&adapt));
  CHKERRQ(TSAdaptLoad(adapt,viewer));
  CHKERRQ(TSGetSNES(ts,&snes));
  CHKERRQ(SNESLoad(snes,viewer));
  /* function and Jacobian context for SNES when used with TS is always ts object */
  CHKERRQ(SNESSetFunction(snes,NULL,NULL,ts));
  CHKERRQ(SNESSetJacobian(snes,NULL,NULL,NULL,ts));
  PetscFunctionReturn(0);
}

/*@C
  TSIRKSetType - Set the type of IRK scheme

  Logically collective

  Input Parameters:
+  ts - timestepping context
-  irktype - type of IRK scheme

  Options Database:
.  -ts_irk_type <gauss> - set irk type

  Level: intermediate

.seealso: TSIRKGetType(), TSIRK, TSIRKType, TSIRKGAUSS
@*/
PetscErrorCode TSIRKSetType(TS ts,TSIRKType irktype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidCharPointer(irktype,2);
  CHKERRQ(PetscTryMethod(ts,"TSIRKSetType_C",(TS,TSIRKType),(ts,irktype)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  CHKERRQ(PetscUseMethod(ts,"TSIRKGetType_C",(TS,TSIRKType*),(ts,irktype)));
  PetscFunctionReturn(0);
}

/*@C
  TSIRKSetNumStages - Set the number of stages of IRK scheme

  Logically collective

  Input Parameters:
+  ts - timestepping context
-  nstages - number of stages of IRK scheme

  Options Database:
.  -ts_irk_nstages <int> - set number of stages

  Level: intermediate

.seealso: TSIRKGetNumStages(), TSIRK
@*/
PetscErrorCode TSIRKSetNumStages(TS ts,PetscInt nstages)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  CHKERRQ(PetscTryMethod(ts,"TSIRKSetNumStages_C",(TS,PetscInt),(ts,nstages)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(nstages,2);
  CHKERRQ(PetscTryMethod(ts,"TSIRKGetNumStages_C",(TS,PetscInt*),(ts,nstages)));
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
  PetscErrorCode (*irkcreate)(TS);

  PetscFunctionBegin;
  if (irk->method_name) {
    CHKERRQ(PetscFree(irk->method_name));
    CHKERRQ(TSIRKTableauReset(ts));
  }
  CHKERRQ(PetscFunctionListFind(TSIRKList,irktype,&irkcreate));
  PetscCheck(irkcreate,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TSIRK type \"%s\" given",irktype);
  CHKERRQ((*irkcreate)(ts));
  CHKERRQ(PetscStrallocpy(irktype,&irk->method_name));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSIRKSetNumStages_IRK(TS ts,PetscInt nstages)
{
  TS_IRK *irk = (TS_IRK*)ts->data;

  PetscFunctionBegin;
  PetscCheck(nstages>0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"input argument, %d, out of range",nstages);
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
  PetscFunctionBegin;
  CHKERRQ(TSReset_IRK(ts));
  if (ts->dm) {
    CHKERRQ(DMCoarsenHookRemove(ts->dm,DMCoarsenHook_TSIRK,DMRestrictHook_TSIRK,ts));
    CHKERRQ(DMSubDomainHookRemove(ts->dm,DMSubDomainHook_TSIRK,DMSubDomainRestrictHook_TSIRK,ts));
  }
  CHKERRQ(PetscFree(ts->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSIRKSetType_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSIRKGetType_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSIRKSetNumStages_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSIRKGetNumStages_C",NULL));
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

  PetscFunctionBegin;
  CHKERRQ(TSIRKInitializePackage());

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

  CHKERRQ(PetscNewLog(ts,&irk));
  ts->data = (void*)irk;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSIRKSetType_C",TSIRKSetType_IRK));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSIRKGetType_C",TSIRKGetType_IRK));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSIRKSetNumStages_C",TSIRKSetNumStages_IRK));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSIRKGetNumStages_C",TSIRKGetNumStages_IRK));
  /* 3-stage IRK_Gauss is the default */
  CHKERRQ(PetscNew(&irk->tableau));
  irk->nstages = 3;
  CHKERRQ(TSIRKSetType(ts,TSIRKDefault));
  PetscFunctionReturn(0);
}
