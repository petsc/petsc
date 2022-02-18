#ifndef PETSC4PY_CUSTOM_H
#define PETSC4PY_CUSTOM_H

#include "petsc/private/vecimpl.h"
#include "petsc/private/matimpl.h"
#include "petsc/private/kspimpl.h"
#include "petsc/private/pcimpl.h"
#include "petsc/private/snesimpl.h"
#include "petsc/private/tsimpl.h"
#include "petsc/private/taoimpl.h"
#include "petsc/private/sfimpl.h"

/* ---------------------------------------------------------------- */

#ifndef PETSC_ERR_PYTHON
#define PETSC_ERR_PYTHON ((PetscErrorCode)(-1))
#endif

/* ---------------------------------------------------------------- */

typedef PetscErrorCode (*PetscErrorHandlerFunction)
(MPI_Comm,int,const char*,const char*,
 PetscErrorCode,PetscErrorType,const char*,void*);
#define PetscTBEH PetscTraceBackErrorHandler

/* ---------------------------------------------------------------- */

#if !defined(PETSC_USE_LOG)
static PetscStageLog petsc_stageLog = NULL;
#endif

#define PetscCLASSID(stageLog,index) \
        ((stageLog)->classLog->classInfo[(index)].classid)

static PetscErrorCode
PetscLogStageFindId(const char name[], PetscLogStage *stageid)
{
  int            s;
  PetscStageLog  stageLog = 0;
  PetscBool      match = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidCharPointer(name,1);
  PetscValidIntPointer(stageid,2);
  *stageid = -1;
  if (!(stageLog=petsc_stageLog)) PetscFunctionReturn(0); /* logging is off ? */
  for (s = 0; s < stageLog->numStages; s++) {
    const char *sname = stageLog->stageInfo[s].name;
    ierr = PetscStrcasecmp(sname, name, &match);CHKERRQ(ierr);
    if (match) { *stageid = s; break; }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode
PetscLogClassFindId(const char name[], PetscClassId *classid)
{
  int            c;
  PetscStageLog  stageLog = 0;
  PetscBool      match = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidCharPointer(name,1);
  PetscValidIntPointer(classid,2);
  *classid = -1;
  if (!(stageLog=petsc_stageLog)) PetscFunctionReturn(0); /* logging is off ? */
  for (c = 0; c < stageLog->classLog->numClasses; c++) {
    const char *cname = stageLog->classLog->classInfo[c].name;
    PetscClassId id = PetscCLASSID(stageLog,c);
    ierr = PetscStrcasecmp(cname, name, &match);CHKERRQ(ierr);
    if (match) { *classid = id; break; }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode
PetscLogEventFindId(const char name[], PetscLogEvent *eventid)
{
  int            e;
  PetscStageLog  stageLog = 0;
  PetscBool      match = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidCharPointer(name,1);
  PetscValidIntPointer(eventid,2);
  *eventid = -1;
  if (!(stageLog=petsc_stageLog)) PetscFunctionReturn(0); /* logging is off ? */
  for (e = 0; e < stageLog->eventLog->numEvents; e++) {
    const char *ename = stageLog->eventLog->eventInfo[e].name;
    ierr = PetscStrcasecmp(ename, name, &match);CHKERRQ(ierr);
    if (match) { *eventid = e; break; }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode
PetscLogStageFindName(PetscLogStage stageid,
                      const char *name[])
{
  PetscStageLog stageLog = 0;
  PetscFunctionBegin;
  PetscValidPointer(name,3);
  *name = 0;
  if (!(stageLog=petsc_stageLog)) PetscFunctionReturn(0); /* logging is off ? */
  if (stageid >=0 && stageid < stageLog->numStages) {
    *name  = stageLog->stageInfo[stageid].name;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode
PetscLogClassFindName(PetscClassId classid,
                      const char *name[])
{
  int           c;
  PetscStageLog stageLog = 0;
  PetscFunctionBegin;
  PetscValidPointer(name,3);
  *name = 0;
  if (!(stageLog=petsc_stageLog)) PetscFunctionReturn(0); /* logging is off ? */
  for (c = 0; c < stageLog->classLog->numClasses; c++) {
    if (classid == PetscCLASSID(stageLog,c)) {
      *name  = stageLog->classLog->classInfo[c].name;
      break;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode
PetscLogEventFindName(PetscLogEvent eventid,
                      const char *name[])
{
  PetscStageLog stageLog = 0;
  PetscFunctionBegin;
  PetscValidPointer(name,3);
  *name = 0;
  if (!(stageLog=petsc_stageLog)) PetscFunctionReturn(0); /* logging is off ? */
  if (eventid >=0 && eventid < stageLog->eventLog->numEvents) {
    *name  = stageLog->eventLog->eventInfo[eventid].name;
  }
  PetscFunctionReturn(0);
}

#if !defined(PETSC_USE_LOG)
static PetscErrorCode
PetscLogEventGetPerfInfo(int stage,PetscLogEvent event,PetscEventPerfInfo *info)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(info,3);
  (void)stage; (void)event; /* unused */
  ierr = PetscMemzero(info,sizeof(PetscEventPerfInfo));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

/* ---------------------------------------------------------------- */

/* The object is not used so far. I expect PETSc will sooner or later support
   a different device context for each object */
static PetscErrorCode
PetscObjectGetDeviceId(PetscObject o, PetscInt *id)
{
#if defined(PETSC_HAVE_DEVICE)
  PetscDeviceContext dctx;
  PetscDevice device;
  PetscErrorCode ierr;
#endif
  PetscFunctionBegin;
  PetscValidHeader(o,1);
#if defined(PETSC_HAVE_DEVICE)
  ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetDevice(dctx,&device);CHKERRQ(ierr);
  ierr = PetscDeviceGetDeviceId(device,id);CHKERRQ(ierr);
#else
  *id = 0;
#endif
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

static inline PetscErrorCode
VecGetCurrentMemType(Vec v, PetscMemType *m)
{
  PetscErrorCode ierr;
  PetscBool bound;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(m,2);
  *m = PETSC_MEMTYPE_HOST;
  ierr = VecBoundToCPU(v,&bound);CHKERRQ(ierr);
  if (!bound) {
    VecType rtype;
    char *iscuda, *iship, *iskok;

    ierr = VecGetRootType_Private(v,&rtype);CHKERRQ(ierr);
    ierr = PetscStrstr(rtype,"cuda",&iscuda);CHKERRQ(ierr);
    ierr = PetscStrstr(rtype,"hip",&iship);CHKERRQ(ierr);
    ierr = PetscStrstr(rtype,"kokkos",&iskok);CHKERRQ(ierr);
    if (iscuda)     *m = PETSC_MEMTYPE_CUDA;
    else if (iship) *m = PETSC_MEMTYPE_HIP;
    else if (iskok) *m = PETSC_MEMTYPE_KOKKOS;
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode
VecStrideSum(Vec v, PetscInt start, PetscScalar *a)
{
  PetscInt          i,n,bs;
  const PetscScalar *x;
  PetscScalar       sum;
  MPI_Comm          comm;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidType(v,1);
  PetscValidScalarPointer(a,2);
  ierr = VecGetBlockSize(v,&bs);CHKERRQ(ierr);
  PetscCheckFalse(start <  0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                            "Negative start %" PetscInt_FMT,start);
  PetscCheckFalse(start >= bs,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,
                            "Start of stride subvector (%" PetscInt_FMT ") is too large "
                            "for block size (%" PetscInt_FMT ")",start,bs);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(v,&x);CHKERRQ(ierr);
  sum = (PetscScalar)0.0;
  for (i=start; i<n; i+=bs) sum += x[i];
  ierr = VecRestoreArrayRead(v,&x);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)v,&comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&sum,a,1,MPIU_SCALAR,MPIU_SUM,comm);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

static inline
PetscErrorCode MatIsPreallocated(Mat A,PetscBool *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(flag,2);
  *flag = A->preallocated;
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode MatHasPreallocationAIJ(Mat A,PetscBool *aij,PetscBool *baij,PetscBool *sbaij,PetscBool *is)
{
  void (*f)(void) = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(aij,2);
  PetscValidPointer(baij,3);
  PetscValidPointer(sbaij,4);
  PetscValidPointer(is,5);
  *aij = *baij = *sbaij = *is = PETSC_FALSE;
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (f)  {*aij = PETSC_TRUE; goto done;};
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (f)  {*baij = PETSC_TRUE; goto done;};
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPISBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqSBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (f)  {*sbaij = PETSC_TRUE; goto done;};
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatISSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (f)  {*is = PETSC_TRUE; goto done;};
 done:
  PetscFunctionReturn(0);
}

static inline PetscErrorCode
MatGetCurrentMemType(Mat A, PetscMemType *m)
{
  PetscErrorCode ierr;
  PetscBool bound;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(m,2);
  *m = PETSC_MEMTYPE_HOST;
  ierr = MatBoundToCPU(A,&bound);CHKERRQ(ierr);
  if (!bound) {
    VecType rtype;
    char *iscuda, *iship, *iskok;

    ierr = MatGetRootType_Private(A,&rtype);CHKERRQ(ierr);
    ierr = PetscStrstr(rtype,"cuda",&iscuda);CHKERRQ(ierr);
    ierr = PetscStrstr(rtype,"hip",&iship);CHKERRQ(ierr);
    ierr = PetscStrstr(rtype,"kokkos",&iskok);CHKERRQ(ierr);
    if (iscuda)     *m = PETSC_MEMTYPE_CUDA;
    else if (iship) *m = PETSC_MEMTYPE_HIP;
    else if (iskok) *m = PETSC_MEMTYPE_KOKKOS;
  }
  PetscFunctionReturn(0);
}

#ifndef MatNullSpaceFunction
typedef PetscErrorCode MatNullSpaceFunction(MatNullSpace,Vec,void*);
#endif

/* ---------------------------------------------------------------- */

static PetscErrorCode
MatFactorInfoDefaults(PetscBool incomplete,PetscBool cholesky,
                      MatFactorInfo *info)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(info,2);
  ierr = MatFactorInfoInitialize(info);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

static PetscErrorCode
KSPSetIterationNumber(KSP ksp, PetscInt its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscCheckFalse(its < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                       "iteration number must be nonnegative");
  ksp->its = its;
  PetscFunctionReturn(0);
}

static PetscErrorCode
KSPSetResidualNorm(KSP ksp, PetscReal rnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscCheckFalse(rnorm < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                         "residual norm must be nonnegative");
  ksp->rnorm = rnorm;
  PetscFunctionReturn(0);
}

static PetscErrorCode
KSPConvergenceTestCall(KSP ksp, PetscInt its, PetscReal rnorm, KSPConvergedReason *reason)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(reason,4);
  PetscCheckFalse(its < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                         "iteration number must be nonnegative");
  PetscCheckFalse(rnorm < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                         "residual norm must be nonnegative");
  ierr = (*ksp->converged)(ksp,its,rnorm,reason,ksp->cnvP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode
KSPSetConvergedReason(KSP ksp, KSPConvergedReason reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ksp->reason = reason;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

static PetscErrorCode
SNESConvergenceTestCall(SNES snes, PetscInt its,
                        PetscReal xnorm, PetscReal ynorm, PetscReal fnorm,
                        SNESConvergedReason *reason)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(reason,4);
  PetscCheckFalse(its < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                         "iteration number must be nonnegative");
  PetscCheckFalse(xnorm < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                         "solution norm must be nonnegative");
  PetscCheckFalse(ynorm < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                         "step norm must be nonnegative");
  PetscCheckFalse(fnorm < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                         "function norm must be nonnegative");
  ierr = (*snes->ops->converged)(snes,its,xnorm,ynorm,fnorm,reason,snes->cnvP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode
SNESGetUseMFFD(SNES snes,PetscBool *flag)
{
  PetscErrorCode (*jac)(SNES,Vec,Mat,Mat,void*) = PETSC_NULL;
  Mat            J = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(flag,2);
  *flag = PETSC_FALSE;
  ierr = SNESGetJacobian(snes,&J,0,&jac,0);CHKERRQ(ierr);
  if (J) { ierr = PetscObjectTypeCompare((PetscObject)J,MATMFFD,flag);CHKERRQ(ierr); }
  else if (jac == MatMFFDComputeJacobian) *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode
SNESSetUseMFFD(SNES snes,PetscBool flag)
{
  const char*    prefix = PETSC_NULL;
  PetscBool      flg = PETSC_FALSE;
  Vec            r = PETSC_NULL;
  Mat            A = PETSC_NULL,B = PETSC_NULL,J = PETSC_NULL;
  void*          funP = PETSC_NULL;
  void*          jacP = PETSC_NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);

  ierr = SNESGetUseMFFD(snes,&flg);CHKERRQ(ierr);
  if (flg  &&  flag) PetscFunctionReturn(0);
  if (!flg && !flag) PetscFunctionReturn(0);
  if (flg  && !flag) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
            "cannot change matrix-free once it is set");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }

  ierr = SNESGetOptionsPrefix(snes,&prefix);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,&r,0,&funP);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes,&A,&B,0,&jacP);CHKERRQ(ierr);
  if (!r) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"SNESSetFunction() must be called first");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }
  ierr = MatCreateSNESMF(snes,&J);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(J,prefix);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  if (!B) {
    KSP       ksp;
    PC        pc;
    PetscBool shell,python;
    ierr = SNESSetJacobian(snes,J,J,MatMFFDComputeJacobian,jacP);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)pc,PCSHELL,&shell);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)pc,PCPYTHON,&python);CHKERRQ(ierr);
    if (!shell && !python) { ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr); }
  } else {
    ierr = SNESSetJacobian(snes,J,0,0,0);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&J);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode
SNESGetUseFDColoring(SNES snes,PetscBool *flag)
{
  PetscErrorCode (*jac)(SNES,Vec,Mat,Mat,void*) = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(flag,2);
  *flag = PETSC_FALSE;
  ierr = SNESGetJacobian(snes,0,0,&jac,0);CHKERRQ(ierr);
  if (jac == SNESComputeJacobianDefaultColor) *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode
SNESSetUseFDColoring(SNES snes,PetscBool flag)
{
  PetscBool      flg = PETSC_FALSE;
  PetscErrorCode (*fun)(SNES,Vec,Vec,void*) = PETSC_NULL;
  void*          funP = PETSC_NULL;
  Mat            A = PETSC_NULL,B = PETSC_NULL;
  PetscErrorCode (*jac)(SNES,Vec,Mat,Mat,void*) = PETSC_NULL;
  void*          jacP = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);

  ierr = SNESGetUseFDColoring(snes,&flg);CHKERRQ(ierr);
  if (flg  &&  flag) PetscFunctionReturn(0);
  if (!flg && !flag) PetscFunctionReturn(0);
  if (flg  && !flag) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
            "cannot change colored finite differences once it is set");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }

  ierr = SNESGetFunction(snes,NULL,&fun,&funP);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes,&A,&B,&jac,&jacP);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,A,B,SNESComputeJacobianDefaultColor,0);CHKERRQ(ierr);
  {
    DM     dm;
    DMSNES sdm;
    ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
    ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
    sdm->jacobianctx = NULL;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

static PetscErrorCode
DMDACreateND(MPI_Comm comm,
             PetscInt dim,PetscInt dof,
             PetscInt M,PetscInt N,PetscInt P,
             PetscInt m,PetscInt n,PetscInt p,
             const PetscInt lx[],const PetscInt ly[],const PetscInt lz[],
             DMBoundaryType bx,DMBoundaryType by,DMBoundaryType bz,
             DMDAStencilType stencil_type,PetscInt stencil_width,
             DM *dm)
{
  DM             da;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(dm,18);
  ierr = DMDACreate(comm,&da);CHKERRQ(ierr);
  ierr = DMSetDimension(da,dim);CHKERRQ(ierr);
  ierr = DMDASetDof(da,dof);CHKERRQ(ierr);
  ierr = DMDASetSizes(da,M,N,P);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(da,m,n,p);CHKERRQ(ierr);
  ierr = DMDASetOwnershipRanges(da,lx,ly,lz);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da,bx,by,bz);CHKERRQ(ierr);
  ierr = DMDASetStencilType(da,stencil_type);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da,stencil_width);CHKERRQ(ierr);
  *dm = (DM)da;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

#endif/* PETSC4PY_CUSTOM_H*/

/*
  Local variables:
  c-basic-offset: 2
  indent-tabs-mode: nil
  End:
*/
