#ifndef PETSC4PY_CUSTOM_H
#define PETSC4PY_CUSTOM_H

#include <petsc/private/deviceimpl.h>
#include <petsc/private/sfimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/pcimpl.h>
#include <petsc/private/kspimpl.h>
#include <petsc/private/snesimpl.h>
#include <petsc/private/tsimpl.h>
#include <petsc/private/taoimpl.h>
#include <petsc/private/deviceimpl.h>
#include <petsc/private/viewerimpl.h>

/* ---------------------------------------------------------------- */

#define PetscERROR(comm,FUNCT,n,t,msg,arg) \
        PetscError(comm,__LINE__,FUNCT,__FILE__,n,t,msg,arg)

/* ---------------------------------------------------------------- */

typedef PetscErrorCode (*PetscErrorHandlerFunction)
(MPI_Comm,int,const char*,const char*,
 PetscErrorCode,PetscErrorType,const char*,void*);
#define PetscTBEH PetscTraceBackErrorHandler

/* ---------------------------------------------------------------- */

PETSC_EXTERN PetscErrorCode (*PetscPythonMonitorSet_C)(PetscObject,const char*);

/* ---------------------------------------------------------------- */

static
PetscErrorCode PetscLogStageFindId(const char name[], PetscLogStage *stageid)
{
  PetscLogState log_state = NULL;

  PetscFunctionBegin;
  PetscAssertPointer(name,1);
  PetscAssertPointer(stageid,2);
  *stageid = -1;
  if (!(log_state = petsc_log_state)) PetscFunctionReturn(PETSC_SUCCESS); /* logging is off ? */
  PetscCall(PetscLogStageGetId(name, stageid));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode PetscLogClassFindId(const char name[], PetscClassId *classid)
{
  PetscLogState log_state = NULL;

  PetscFunctionBegin;
  PetscAssertPointer(name,1);
  PetscAssertPointer(classid,2);
  *classid = -1;
  if (!(log_state = petsc_log_state)) PetscFunctionReturn(PETSC_SUCCESS); /* logging is off ? */
  PetscCall(PetscLogClassGetClassId(name, classid));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode PetscLogEventFindId(const char name[], PetscLogEvent *eventid)
{
  PetscLogState log_state = NULL;

  PetscFunctionBegin;
  PetscAssertPointer(name,1);
  PetscAssertPointer(eventid,2);
  *eventid = -1;
  if (!(log_state = petsc_log_state)) PetscFunctionReturn(PETSC_SUCCESS); /* logging is off ? */
  PetscCall(PetscLogEventGetId(name, eventid));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode PetscLogStageFindName(PetscLogStage stageid, const char *name[])
{
  PetscLogState log_state = NULL;

  PetscFunctionBegin;
  PetscAssertPointer(name,3);
  *name = NULL;
  if (!(log_state = petsc_log_state)) PetscFunctionReturn(PETSC_SUCCESS); /* logging is off ? */
  PetscCall(PetscLogStageGetName(stageid, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode PetscLogClassFindName(PetscClassId classid, const char *name[])
{
  PetscLogState log_state = NULL;

  PetscFunctionBegin;
  PetscAssertPointer(name,3);
  *name = 0;
  if (!(log_state = petsc_log_state)) PetscFunctionReturn(PETSC_SUCCESS); /* logging is off ? */
  PetscCall(PetscLogClassIdGetName(classid, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode PetscLogEventFindName(PetscLogEvent eventid, const char *name[])
{
  PetscLogState log_state = NULL;

  PetscFunctionBegin;
  PetscAssertPointer(name,3);
  *name = 0;
  if (!(log_state = petsc_log_state)) PetscFunctionReturn(PETSC_SUCCESS); /* logging is off ? */
  PetscCall(PetscLogEventGetName(eventid, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------- */

static
PetscErrorCode PetscObjectComposedDataGetIntPy(PetscObject o, PetscInt id, PetscInt *v, PetscBool *exist)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposedDataGetInt(o,id,*v,*exist));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode PetscObjectComposedDataSetIntPy(PetscObject o, PetscInt id, PetscInt v)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposedDataSetInt(o,id,v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode PetscObjectComposedDataRegisterPy(PetscInt *id)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposedDataRegister(id));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------- */

/* The object is not used so far. I expect PETSc will sooner or later support
   a different device context for each object */
static
PetscErrorCode PetscObjectGetDeviceId(PetscObject o, PetscInt *id)
{
#if defined(PETSC_HAVE_DEVICE)
  PetscDeviceContext dctx;
  PetscDevice device;
#endif
  PetscFunctionBegin;
  PetscValidHeader(o,1);
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetDevice(dctx,&device));
  PetscCall(PetscDeviceGetDeviceId(device,id));
#else
  *id = 0;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------- */

static
PetscErrorCode VecGetCurrentMemType(Vec v, PetscMemType *m)
{
  PetscBool bound;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscAssertPointer(m,2);
  *m = PETSC_MEMTYPE_HOST;
  PetscCall(VecBoundToCPU(v,&bound));
  if (!bound) {
    VecType rtype;
    char *iscuda = NULL, *iship = NULL, *iskok = NULL;

    PetscCall(VecGetRootType_Private(v,&rtype));
    PetscCall(PetscStrstr(rtype,"cuda",&iscuda));
    PetscCall(PetscStrstr(rtype,"hip",&iship));
    PetscCall(PetscStrstr(rtype,"kokkos",&iskok));
    if (iscuda)     *m = PETSC_MEMTYPE_CUDA;
    else if (iship) *m = PETSC_MEMTYPE_HIP;
    else if (iskok) *m = PETSC_MEMTYPE_KOKKOS;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------- */

static
PetscErrorCode MatIsPreallocated(Mat A,PetscBool *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscAssertPointer(flag,2);
  *flag = A->preallocated;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode MatHasPreallocationAIJ(Mat A,PetscBool *aij,PetscBool *baij,PetscBool *sbaij,PetscBool *is)
{
  PetscErrorCodeFn *f = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscAssertPointer(aij,2);
  PetscAssertPointer(baij,3);
  PetscAssertPointer(sbaij,4);
  PetscAssertPointer(is,5);
  *aij = *baij = *sbaij = *is = PETSC_FALSE;
  if (!f) PetscCall(PetscObjectQueryFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",&f));
  if (!f) PetscCall(PetscObjectQueryFunction((PetscObject)A,"MatSeqAIJSetPreallocation_C",&f));
  if (f)  {*aij = PETSC_TRUE; goto done;}
  if (!f) PetscCall(PetscObjectQueryFunction((PetscObject)A,"MatMPIBAIJSetPreallocation_C",&f));
  if (!f) PetscCall(PetscObjectQueryFunction((PetscObject)A,"MatSeqBAIJSetPreallocation_C",&f));
  if (f)  {*baij = PETSC_TRUE; goto done;}
  if (!f) PetscCall(PetscObjectQueryFunction((PetscObject)A,"MatMPISBAIJSetPreallocation_C",&f));
  if (!f) PetscCall(PetscObjectQueryFunction((PetscObject)A,"MatSeqSBAIJSetPreallocation_C",&f));
  if (f)  {*sbaij = PETSC_TRUE; goto done;}
  if (!f) PetscCall(PetscObjectQueryFunction((PetscObject)A,"MatISSetPreallocation_C",&f));
  if (f)  {*is = PETSC_TRUE; goto done;}
 done:
  PetscFunctionReturn(PETSC_SUCCESS);
}

#ifndef MatNullSpaceFunction
typedef PetscErrorCode MatNullSpaceFunction(MatNullSpace,Vec,void*);
#endif

/* ---------------------------------------------------------------- */

static
PetscErrorCode MatFactorInfoDefaults(PetscBool incomplete,PetscBool cholesky,MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscAssertPointer(info,2);
  PetscCall(MatFactorInfoInitialize(info));
  if (incomplete) {
    info->levels         = (PetscReal)0;
    info->diagonal_fill  = (PetscReal)0;
    info->fill           = (PetscReal)1.0;
    info->usedt          = (PetscReal)0;
    info->dt             = (PetscReal)PETSC_DEFAULT;
    info->dtcount        = (PetscReal)PETSC_DEFAULT;
    info->dtcol          = (PetscReal)PETSC_DEFAULT;
    info->zeropivot      = (PetscReal)100.0*PETSC_MACHINE_EPSILON;
    info->pivotinblocks  = (PetscReal)1;
  } else {
    info->fill           = (PetscReal)5.0;
    info->dtcol          = (PetscReal)1.e-6;
    info->zeropivot      = (PetscReal)100.0*PETSC_MACHINE_EPSILON;
    info->pivotinblocks  = (PetscReal)1;
  }
  if (incomplete) {
    if (cholesky)
      info->shifttype    = (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE;
    else
      info->shifttype    = (PetscReal)MAT_SHIFT_NONZERO;
    info->shiftamount    = (PetscReal)100.0*PETSC_MACHINE_EPSILON;
  } else {
    info->shifttype      = (PetscReal)MAT_SHIFT_NONE;
    info->shiftamount    = (PetscReal)0.0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------- */

static
PetscErrorCode KSPSetIterationNumber(KSP ksp, PetscInt its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscCheck(its >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"iteration number must be nonnegative");
  ksp->its = its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode KSPSetResidualNorm(KSP ksp, PetscReal rnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscCheck(rnorm >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"residual norm must be nonnegative");
  ksp->rnorm = rnorm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode KSPConvergenceTestCall(KSP ksp, PetscInt its, PetscReal rnorm, KSPConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscAssertPointer(reason,4);
  PetscCheck(its >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"iteration number must be nonnegative");
  PetscCheck(rnorm >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"residual norm must be nonnegative");
  PetscCall((*ksp->converged)(ksp,its,rnorm,reason,ksp->cnvP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode KSPSetConvergedReason(KSP ksp, KSPConvergedReason reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ksp->reason = reason;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode KSPConverged(KSP ksp,PetscInt iter,PetscReal rnorm,KSPConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (reason) PetscAssertPointer(reason,2);
  if (!iter) ksp->rnorm0 = rnorm;
  if (!iter) {
    ksp->reason = KSP_CONVERGED_ITERATING;
    ksp->ttol = PetscMax(rnorm*ksp->rtol,ksp->abstol);
  }
  if (ksp->converged) {
    PetscCall(ksp->converged(ksp,iter,rnorm,&ksp->reason,ksp->cnvP));
  } else {
    PetscCall(KSPConvergedSkip(ksp,iter,rnorm,&ksp->reason,NULL));
    /*PetscCall(KSPConvergedDefault(ksp,iter,rnorm,&ksp->reason,NULL));*/
  }
  ksp->rnorm = rnorm;
  if (reason) *reason = ksp->reason;
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscBool prepend_custom;
  KSPConvergenceTestFn *convtest;
  PetscCtxDestroyFn    *convdestroy;
  KSPConvergenceTestFn *convtestcustom;
  void *convctx;
} KSPConvergedNativeCtx;

static
PetscErrorCode KSPConvergedNative_Private(KSP ksp, PetscInt n, PetscReal rnorm, KSPConvergedReason *reason, void *cctx)
{
  KSPConvergedNativeCtx *ctx = (KSPConvergedNativeCtx *)cctx;

  PetscFunctionBegin;
  *reason = KSP_CONVERGED_ITERATING;
  if (ctx->prepend_custom) {
    PetscCall((*ctx->convtestcustom)(ksp, n, rnorm, reason, NULL));
    if (*reason) {
      PetscCall(PetscInfo(ksp, "User provided prepended Python convergence test reason %s KSP iterations=%" PetscInt_FMT ", rnorm=%g\n", KSPConvergedReasons[*reason], n, (double)rnorm));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscCall((*ctx->convtest)(ksp, n, rnorm, reason, ctx->convctx));
  if (*reason) {
    PetscCall(PetscInfo(ksp, "Default convergence test reason %s KSP iterations=%" PetscInt_FMT ", rnorm=%g\n", KSPConvergedReasons[*reason], n, (double)rnorm));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (!ctx->prepend_custom) {
    PetscCall((*ctx->convtestcustom)(ksp, n, rnorm, reason, NULL));
    if (*reason) PetscCall(PetscInfo(ksp, "User provide appended Python convergence test reason %s KSP iterations=%" PetscInt_FMT ", rnorm=%g\n", KSPConvergedReasons[*reason], n, (double)rnorm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPConvergedNative_Destroy(void **cctx)
{
  KSPConvergedNativeCtx *ctx = (KSPConvergedNativeCtx *)*cctx;

  PetscFunctionBegin;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing context");
  if (ctx->convdestroy) PetscCall((*ctx->convdestroy)(&ctx->convctx));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode KSPAddConvergenceTest(KSP ksp, PetscErrorCode (*custom)(KSP, PetscInt, PetscReal, KSPConvergedReason *, void *), PetscBool prepend)
{
  KSPConvergedNativeCtx *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,prepend,3);
  PetscCall(PetscNew(&ctx));
  ctx->convtestcustom = custom;
  ctx->prepend_custom = prepend;
  PetscCall(KSPGetAndClearConvergenceTest(ksp, &ctx->convtest, &ctx->convctx, &ctx->convdestroy));
  PetscCall(KSPSetConvergenceTest(ksp, KSPConvergedNative_Private, ctx, KSPConvergedNative_Destroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode KSPLogHistory(KSP ksp,PetscReal rnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscCall(KSPLogResidualHistory(ksp,rnorm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------- */

static
PetscErrorCode SNESConvergenceTestCall(SNES snes, PetscInt its, PetscReal xnorm, PetscReal ynorm, PetscReal fnorm, SNESConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscAssertPointer(reason,4);
  PetscCheck(its >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"iteration number must be nonnegative");
  PetscCheck(xnorm >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"solution norm must be nonnegative");
  PetscCheck(ynorm >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"step norm must be nonnegative");
  PetscCheck(fnorm >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"function norm must be nonnegative");
  PetscUseTypeMethod(snes,converged,its,xnorm,ynorm,fnorm,reason,snes->cnvP);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode SNESLogHistory(SNES snes,PetscReal rnorm,PetscInt lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscCall(SNESLogConvergenceHistory(snes,rnorm,lits));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode SNESGetUseMFFD(SNES snes,PetscBool *flag)
{
  PetscErrorCode (*jac)(SNES,Vec,Mat,Mat,void*) = NULL;
  Mat            J = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscAssertPointer(flag,2);
  *flag = PETSC_FALSE;
  PetscCall(SNESGetJacobian(snes,&J,0,&jac,0));
  if (J) PetscCall(PetscObjectTypeCompare((PetscObject)J,MATMFFD,flag));
  else if (jac == MatMFFDComputeJacobian) *flag = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode SNESSetUseMFFD(SNES snes,PetscBool flag)
{
  const char* prefix = NULL;
  PetscBool   flg    = PETSC_FALSE;
  Vec         r      = NULL;
  Mat         A      = NULL,B = NULL,J = NULL;
  void*       funP   = NULL;
  void*       jacP   = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);

  PetscCall(SNESGetUseMFFD(snes,&flg));
  if (flg  &&  flag) PetscFunctionReturn(PETSC_SUCCESS);
  if (!flg && !flag) PetscFunctionReturn(PETSC_SUCCESS);
  if (flg  && !flag) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
            "cannot change matrix-free once it is set");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }

  PetscCall(SNESGetOptionsPrefix(snes,&prefix));
  PetscCall(SNESGetFunction(snes,&r,0,&funP));
  PetscCall(SNESGetJacobian(snes,&A,&B,0,&jacP));
  if (!r) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"SNESSetFunction() must be called first");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }
  PetscCall(MatCreateSNESMF(snes,&J));
  PetscCall(MatSetOptionsPrefix(J,prefix));
  PetscCall(MatSetFromOptions(J));
  if (!B) {
    KSP       ksp;
    PC        pc;
    PetscBool shell,python;
    PetscCall(SNESSetJacobian(snes,J,J,MatMFFDComputeJacobian,jacP));
    PetscCall(SNESGetKSP(snes,&ksp));
    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCSHELL,&shell));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCPYTHON,&python));
    if (!shell && !python) PetscCall(PCSetType(pc,PCNONE));
  } else PetscCall(SNESSetJacobian(snes,J,0,0,0));
  PetscCall(MatDestroy(&J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode SNESGetUseFDColoring(SNES snes,PetscBool *flag)
{
  PetscErrorCode (*jac)(SNES,Vec,Mat,Mat,void*) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscAssertPointer(flag,2);
  *flag = PETSC_FALSE;
  PetscCall(SNESGetJacobian(snes,0,0,&jac,0));
  if (jac == SNESComputeJacobianDefaultColor) *flag = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode SNESSetUseFDColoring(SNES snes,PetscBool flag)
{
  PetscBool      flg = PETSC_FALSE;
  PetscErrorCode (*fun)(SNES,Vec,Vec,void*) = NULL;
  void*          funP = NULL;
  Mat            A = NULL,B = NULL;
  PetscErrorCode (*jac)(SNES,Vec,Mat,Mat,void*) = NULL;
  void*          jacP = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);

  PetscCall(SNESGetUseFDColoring(snes,&flg));
  if (flg  &&  flag) PetscFunctionReturn(PETSC_SUCCESS);
  if (!flg && !flag) PetscFunctionReturn(PETSC_SUCCESS);
  if (flg  && !flag) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
            "cannot change colored finite differences once it is set");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }

  PetscCall(SNESGetFunction(snes,NULL,&fun,&funP));
  PetscCall(SNESGetJacobian(snes,&A,&B,&jac,&jacP));
  PetscCall(SNESSetJacobian(snes,A,B,SNESComputeJacobianDefaultColor,0));
  {
    DM     dm;
    DMSNES sdm;
    PetscCall(SNESGetDM(snes,&dm));
    PetscCall(DMGetDMSNES(dm,&sdm));
    PetscCall(DMSNESUnsetJacobianContext_Internal(dm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode SNESComputeUpdate(SNES snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscTryTypeMethod(snes, update, snes->iter);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode SNESGetUseKSP(SNES snes,PetscBool *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscAssertPointer(flag,2);
  *flag = snes->usesksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode SNESSetUseKSP(SNES snes,PetscBool flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveBool(snes,flag,2);
  snes->usesksp = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------- */

static
PetscErrorCode TaoConverged(Tao tao, TaoConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscAssertPointer(reason,2);
  if (tao->ops->convergencetest) {
    PetscUseTypeMethod(tao,convergencetest,tao->cnvP);
  } else {
    PetscCall(TaoDefaultConvergenceTest(tao,tao->cnvP));
  }
  *reason = tao->reason;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode TaoCheckReals(Tao tao, PetscReal f, PetscReal g)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(g),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER,"User provided compute function generated Inf or NaN");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode TaoCreateDefaultKSP(Tao tao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscCall(KSPDestroy(&tao->ksp));
  PetscCall(KSPCreate(((PetscObject)tao)->comm,&tao->ksp));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->ksp,(PetscObject)tao,1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode TaoCreateDefaultLineSearch(Tao tao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscCall(TaoLineSearchDestroy(&tao->linesearch));
  PetscCall(TaoLineSearchCreate(((PetscObject)tao)->comm,&tao->linesearch));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch,(PetscObject)tao,1));
  PetscCall(TaoLineSearchSetType(tao->linesearch,TAOLINESEARCHMT));
  PetscCall(TaoLineSearchUseTaoRoutines(tao->linesearch,tao));
  PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch,1.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode TaoHasGradientRoutine(Tao tao, PetscBool* flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscAssertPointer(flg,2);
  *flg = (PetscBool)(tao->ops->computegradient || tao->ops->computeobjectiveandgradient);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
static
PetscErrorCode TaoHasHessianRoutine(Tao tao, PetscBool* flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscAssertPointer(flg,2);
  *flg = tao->ops->computehessian;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static
PetscErrorCode TaoComputeUpdate(Tao tao, PetscReal *f)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (tao->ops->update) {
    PetscUseTypeMethod(tao,update,tao->niter,tao->user_update);
    PetscCall(TaoComputeObjective(tao,tao->solution,f));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode TaoGetVecs(Tao tao, Vec *X, Vec *G, Vec *S)
{
  PetscBool has_g;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscCall(TaoHasGradientRoutine(tao,&has_g));
  if (X) *X = tao->solution;
  if (G) {
    if (has_g && !tao->gradient) PetscCall(VecDuplicate(tao->solution,&tao->gradient));
    *G = has_g ? tao->gradient : NULL;
  }
  if (S) {
    if (has_g && !tao->stepdirection) PetscCall(VecDuplicate(tao->solution,&tao->stepdirection));
    *S = has_g ? tao->stepdirection : NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode TaoApplyLineSearch(Tao tao, PetscReal* f, PetscReal *s, TaoLineSearchConvergedReason *lsr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscAssertPointer(f,2);
  PetscAssertPointer(s,3);
  PetscCall(TaoLineSearchApply(tao->linesearch,tao->solution,f,tao->gradient,tao->stepdirection,s,lsr));
  PetscCall(TaoAddLineSearchCounts(tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------- */

static
PetscErrorCode DMDACreateND(MPI_Comm comm,
                            PetscInt dim,PetscInt dof,
                            PetscInt M,PetscInt N,PetscInt P,
                            PetscInt m,PetscInt n,PetscInt p,
                            const PetscInt lx[],const PetscInt ly[],const PetscInt lz[],
                            DMBoundaryType bx,DMBoundaryType by,DMBoundaryType bz,
                            DMDAStencilType stencil_type,PetscInt stencil_width,
                            DM *dm)
{
  DM da;

  PetscFunctionBegin;
  PetscAssertPointer(dm,18);
  PetscCall(DMDACreate(comm,&da));
  PetscCall(DMSetDimension(da,dim));
  PetscCall(DMDASetDof(da,dof));
  PetscCall(DMDASetSizes(da,M,N,P));
  PetscCall(DMDASetNumProcs(da,m,n,p));
  PetscCall(DMDASetOwnershipRanges(da,lx,ly,lz));
  PetscCall(DMDASetBoundaryType(da,bx,by,bz));
  PetscCall(DMDASetStencilType(da,stencil_type));
  PetscCall(DMDASetStencilWidth(da,stencil_width));
  *dm = (DM)da;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static
PetscErrorCode PetscDeviceReference(PetscDevice device)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceReference_Internal(device));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------- */

#endif/* PETSC4PY_CUSTOM_H*/

/*
  Local variables:
  c-basic-offset: 2
  indent-tabs-mode: nil
  End:
*/
