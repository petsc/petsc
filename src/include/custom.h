#ifndef PETSC4PY_CUSTOM_H
#define PETSC4PY_CUSTOM_H

#undef  __FUNCT__
#define __FUNCT__ "<petsc4py.PETSc>"

#if PETSC_VERSION_(3,2,0)
#include "private/vecimpl.h"
#include "private/matimpl.h"
#include "private/kspimpl.h"
#include "private/pcimpl.h"
#include "private/snesimpl.h"
#include "private/tsimpl.h"
#else
#include "petsc-private/vecimpl.h"
#include "petsc-private/matimpl.h"
#include "petsc-private/kspimpl.h"
#include "petsc-private/pcimpl.h"
#include "petsc-private/snesimpl.h"
#include "petsc-private/tsimpl.h"
#endif

/* ---------------------------------------------------------------- */

#ifndef PETSC_ERR_PYTHON
#define PETSC_ERR_PYTHON ((PetscErrorCode)(-1))
#endif

/* ---------------------------------------------------------------- */

#if PETSC_VERSION_(3,2,0)
#define petsc_stageLog _stageLog
#endif

#define PetscCLASSID(stageLog,index) \
        ((stageLog)->classLog->classInfo[(index)].classid)

#undef __FUNCT__
#define __FUNCT__ "PetscLogStageFindId"
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
  for(s = 0; s < stageLog->numStages; s++) {
    const char *sname = stageLog->stageInfo[s].name;
    ierr = PetscStrcasecmp(sname, name, &match);CHKERRQ(ierr);
    if (match) { *stageid = s; break; }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLogClassFindId"
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
  for(c = 0; c < stageLog->classLog->numClasses; c++) {
    const char *cname = stageLog->classLog->classInfo[c].name;
    PetscClassId id = PetscCLASSID(stageLog,c);
    ierr = PetscStrcasecmp(cname, name, &match);CHKERRQ(ierr);
    if (match) { *classid = id; break; }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLogEventFindId"
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
  for(e = 0; e < stageLog->eventLog->numEvents; e++) {
    const char *ename = stageLog->eventLog->eventInfo[e].name;
    ierr = PetscStrcasecmp(ename, name, &match);CHKERRQ(ierr);
    if (match) { *eventid = e; break; }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLogStageFindName"
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

#undef __FUNCT__
#define __FUNCT__ "PetscLogClassFindName"
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
  for(c = 0; c < stageLog->classLog->numClasses; c++) {
    if (classid == PetscCLASSID(stageLog,c)) {
      *name  = stageLog->classLog->classInfo[c].name;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLogEventFindName"
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

/* ---------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "VecStrideSum"
PETSC_STATIC_INLINE PetscErrorCode
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
  if (start <  0)  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                            "Negative start %D",start);
  if (start >= bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,
                            "Start of stride subvector (%D) is too large "
                            "for block size (%D)",start,bs);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(v,&x);CHKERRQ(ierr);
  sum = (PetscScalar)0.0;
  for (i=start; i<n; i+=bs) sum += x[i];
  ierr = VecRestoreArrayRead(v,&x);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)v,&comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&sum,a,1,MPIU_SCALAR,MPIU_SUM,comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "MatIsPreallocated"
static PetscErrorCode
MatIsPreallocated(Mat A,PetscBool *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(flag,2);
  *flag = A->preallocated;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateAnyAIJ"
static PetscErrorCode
MatCreateAnyAIJ(MPI_Comm comm, PetscInt bs,
                PetscInt m, PetscInt n,
                PetscInt M, PetscInt N,
                Mat *A)
{
  Mat            mat = PETSC_NULL;
  MatType        mtype = PETSC_NULL;
  PetscMPIInt    size = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(A,7);
  ierr = MatCreate(comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (bs == PETSC_DECIDE) {
    if (size > 1) mtype = (MatType)MATMPIAIJ;
    else          mtype = (MatType)MATSEQAIJ;
  } else {
    if (size > 1) mtype = (MatType)MATMPIBAIJ;
    else          mtype = (MatType)MATSEQBAIJ;
  }
  ierr = MatSetType(mat,mtype);CHKERRQ(ierr);
  if (bs != PETSC_DECIDE) {
    ierr = MatSetBlockSize(mat,bs);CHKERRQ(ierr);
  }
  *A = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateAnyAIJCRL"
static PetscErrorCode
MatCreateAnyAIJCRL(MPI_Comm comm, PetscInt bs,
                   PetscInt m, PetscInt n,
                   PetscInt M, PetscInt N,
                   Mat *A)
{
  Mat            mat = PETSC_NULL;
  MatType        mtype = PETSC_NULL;
  PetscMPIInt    size = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(A,7);
  ierr = MatCreate(comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (size > 1) mtype = (MatType)MATMPIAIJCRL;
  else          mtype = (MatType)MATSEQAIJCRL;
  ierr = MatSetType(mat,mtype);CHKERRQ(ierr);
  if (bs != PETSC_DECIDE) {
    ierr = MatSetBlockSize(mat,bs);CHKERRQ(ierr);
  }
  *A = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateAnyDense"
static PetscErrorCode
MatCreateAnyDense(MPI_Comm comm, PetscInt bs,
                  PetscInt m, PetscInt n,
                  PetscInt M, PetscInt N,
                  Mat *A)
{
  Mat            mat = PETSC_NULL;
  MatType        mtype = PETSC_NULL;
  PetscMPIInt    size = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(A,7);
  ierr = MatCreate(comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) mtype = (MatType)MATMPIDENSE;
  else          mtype = (MatType)MATSEQDENSE;
  ierr = MatSetType(mat,mtype);CHKERRQ(ierr);
  if (bs != PETSC_DECIDE) {
    ierr = MatSetBlockSize(mat,bs);CHKERRQ(ierr);
  }
  *A = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAnyAIJSetPreallocation"
static PetscErrorCode
MatAnyAIJSetPreallocation(Mat A,PetscInt bs,
                          PetscInt d_nz,const PetscInt d_nnz[],
                          PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscBool      flag = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  if (d_nnz) PetscValidIntPointer(d_nnz,3);
  if (o_nnz) PetscValidIntPointer(o_nnz,5);
  ierr = MatIsPreallocated(A,&flag);CHKERRQ(ierr);
  if (flag) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,
                    "matrix is already preallocated");
#if PETSC_VERSION_(3,2,0)
  {PetscInt bsize = A->rmap->bs;
#endif
  if (bs == PETSC_DECIDE) {
    ierr = MatSeqAIJSetPreallocation(A,d_nz,d_nnz);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(A,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSeqBAIJSetPreallocation(A,bs,d_nz,d_nnz);CHKERRQ(ierr);
    ierr = MatMPIBAIJSetPreallocation(A,bs,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  }
#if PETSC_VERSION_(3,2,0)
  ierr = MatSetUp(A);CHKERRQ(ierr);
  if (bsize > 1) {
    A->rmap->bs = A->cmap->bs = -1;
    ierr = MatSetBlockSize(A,bsize);CHKERRQ(ierr);
  }}
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAnyAIJSetPreallocationCSR"
static PetscErrorCode
MatAnyAIJSetPreallocationCSR(Mat A,PetscInt bs, const PetscInt Ii[],
                             const PetscInt Jj[], const PetscScalar V[])
{
  PetscBool      flag = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidIntPointer(Ii,3);
  PetscValidIntPointer(Jj,4);
  if (V) PetscValidScalarPointer(V,5);
  ierr = MatIsPreallocated(A,&flag);CHKERRQ(ierr);
  if (flag) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,
                    "matrix is already preallocated");
#if PETSC_VERSION_(3,2,0)
  {PetscInt bsize = A->rmap->bs;
#endif
  if (bs == PETSC_DECIDE) {
    ierr = MatSeqAIJSetPreallocationCSR(A,Ii,Jj,V);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocationCSR(A,Ii,Jj,V);CHKERRQ(ierr);
  } else {
    ierr = MatSeqBAIJSetPreallocationCSR(A,bs,Ii,Jj,V);CHKERRQ(ierr);
    ierr = MatMPIBAIJSetPreallocationCSR(A,bs,Ii,Jj,V);CHKERRQ(ierr);
  }
#if PETSC_VERSION_(3,2,0)
  ierr = MatSetUp(A);CHKERRQ(ierr);
  if (bsize > 1) {
    A->rmap->bs = A->cmap->bs = -1;
    ierr = MatSetBlockSize(A,bsize);CHKERRQ(ierr);
  }}
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAnyDenseSetPreallocation"
static PetscErrorCode
MatAnyDenseSetPreallocation(Mat mat, PetscScalar *data)
{
  PetscBool      flag = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (data) PetscValidScalarPointer(data,3);
  ierr = MatIsPreallocated(mat,&flag);CHKERRQ(ierr);
  if (flag) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,
                    "matrix is already preallocated");
#if PETSC_VERSION_(3,2,0)
  {PetscInt bsize = mat->rmap->bs;
#endif
  ierr = MatSeqDenseSetPreallocation(mat,data);CHKERRQ(ierr);
  ierr = MatMPIDenseSetPreallocation(mat,data);CHKERRQ(ierr);
#if PETSC_VERSION_(3,2,0)
  if (bsize > 1) {
    mat->rmap->bs = mat->cmap->bs = -1;
    ierr = MatSetBlockSize(mat,bsize);CHKERRQ(ierr);
  }}
#endif
  PetscFunctionReturn(0);
}

#ifndef MatNullSpaceFunction
typedef PetscErrorCode MatNullSpaceFunction(MatNullSpace,Vec,void*);
#endif

/* ---------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "MatFactorInfoDefaults"
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

#undef __FUNCT__
#define __FUNCT__ "KSPSetIterationNumber"
static PetscErrorCode
KSPSetIterationNumber(KSP ksp, PetscInt its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (its < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                       "iteration number must be nonnegative");
  ksp->its = its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetResidualNorm"
static PetscErrorCode
KSPSetResidualNorm(KSP ksp, PetscReal rnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (rnorm < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                         "residual norm must be nonnegative");
  ksp->rnorm = rnorm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPConvergenceTestCall"
static PetscErrorCode
KSPConvergenceTestCall(KSP ksp, PetscInt its, PetscReal rnorm, KSPConvergedReason *reason)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(reason,4);
  if (its < 0)   SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                         "iteration number must be nonnegative");
  if (rnorm < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                         "residual norm must be nonnegative");
  ierr = (*ksp->converged)(ksp,its,rnorm,reason,ksp->cnvP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetConvergedReason"
static PetscErrorCode
KSPSetConvergedReason(KSP ksp, KSPConvergedReason reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ksp->reason = reason;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "SNESConvergenceTestCall"
static PetscErrorCode
SNESConvergenceTestCall(SNES snes, PetscInt its,
                        PetscReal xnorm, PetscReal ynorm, PetscReal fnorm,
                        SNESConvergedReason *reason)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(reason,4);
  if (its < 0)   SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                         "iteration number must be nonnegative");
  if (xnorm < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                         "solution norm must be nonnegative");
  if (ynorm < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                         "step norm must be nonnegative");
  if (fnorm < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                         "function norm must be nonnegative");
  ierr = (*snes->ops->converged)(snes,its,xnorm,ynorm,fnorm,reason,snes->cnvP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetConvergedReason"
static PetscErrorCode
SNESSetConvergedReason(SNES snes, SNESConvergedReason reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  snes->reason = reason;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESGetUseMFFD"
static PetscErrorCode
SNESGetUseMFFD(SNES snes,PetscBool *flag)
{
  PetscErrorCode (*jac)(SNES,Vec,Mat*,Mat*,MatStructure*,void*) = PETSC_NULL;
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

#undef __FUNCT__
#define __FUNCT__ "SNESSetUseMFFD"
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
  if ( flg &&  flag) PetscFunctionReturn(0);
  if (!flg && !flag) PetscFunctionReturn(0);
  if ( flg && !flag) {
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

#if PETSC_VERSION_LE(3,3,0)
#undef  __FUNCT__
#define __FUNCT__ "SNESComputeJacobianDefaultColor"
PETSC_EXTERN PetscErrorCode SNESComputeJacobianDefaultColor(SNES snes,Vec x,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  Vec            f = NULL;
  PetscErrorCode (*fun)(SNES,Vec,Vec,void*) = NULL;
  void*          funP = NULL;
  ISColoring     iscoloring = NULL;
  MatFDColoring  color = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = SNESGetFunction(snes,&f,&fun,&funP);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)*B,"SNESMatFDColoring",(PetscObject*)&color);CHKERRQ(ierr);
  if (!color) {
    ierr = MatGetColoring(*B,MATCOLORINGSL,&iscoloring);CHKERRQ(ierr);
    ierr = MatFDColoringCreate(*B,iscoloring,&color);CHKERRQ(ierr);
    ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
    ierr = MatFDColoringSetFunction(color,(PetscErrorCode(*)(void))fun,(void*)funP);CHKERRQ(ierr);
    ierr = MatFDColoringSetFromOptions(color);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)*B,"SNESMatFDColoring",(PetscObject)color);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)color);CHKERRQ(ierr);
  }
  ierr = SNESDefaultComputeJacobianColor(snes,x,J,B,flag,(void*)color);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "SNESGetUseFDColoring"
static PetscErrorCode
SNESGetUseFDColoring(SNES snes,PetscBool *flag)
{
  PetscErrorCode (*jac)(SNES,Vec,Mat*,Mat*,MatStructure*,void*) = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(flag,2);
  *flag = PETSC_FALSE;
  ierr = SNESGetJacobian(snes,0,0,&jac,0);CHKERRQ(ierr);
  if (jac == SNESComputeJacobianDefaultColor) *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUseFDColoring"
static PetscErrorCode
SNESSetUseFDColoring(SNES snes,PetscBool flag)
{
  PetscBool      flg = PETSC_FALSE;
  PetscErrorCode (*fun)(SNES,Vec,Vec,void*) = PETSC_NULL;
  void*          funP = PETSC_NULL;
  Mat            A = PETSC_NULL,B = PETSC_NULL;
  PetscErrorCode (*jac)(SNES,Vec,Mat*,Mat*,MatStructure*,void*) = PETSC_NULL;
  void*          jacP = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);

  ierr = SNESGetUseFDColoring(snes,&flg);CHKERRQ(ierr);
  if ( flg &&  flag) PetscFunctionReturn(0);
  if (!flg && !flag) PetscFunctionReturn(0);
  if ( flg && !flag) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
            "cannot change colored finite diferences once it is set");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }

  ierr = SNESGetFunction(snes,NULL,&fun,&funP);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes,&A,&B,&jac,&jacP);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,A,B,SNESComputeJacobianDefaultColor,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "TSSetTimeStepNumber"
static PetscErrorCode
TSSetTimeStepNumber(TS ts, PetscInt step)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->steps = step;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "DMDACreateND"
static PetscErrorCode
DMDACreateND(MPI_Comm comm,
             PetscInt dim,PetscInt dof,
             PetscInt M,PetscInt N,PetscInt P,
             PetscInt m,PetscInt n,PetscInt p,
             const PetscInt lx[],const PetscInt ly[],const PetscInt lz[],
             DMDABoundaryType bx,DMDABoundaryType by,DMDABoundaryType bz,
             DMDAStencilType stencil_type,PetscInt stencil_width,
             DM *dm)
{
  DM             da;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(dm,18);
  ierr = DMDACreate(comm,&da);CHKERRQ(ierr);
  ierr = DMDASetDim(da,dim);CHKERRQ(ierr);
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

#undef  __FUNCT__
#define __FUNCT__ "<petsc4py.PETSc>"

#endif/* PETSC4PY_CUSTOM_H*/

/*
  Local variables:
  c-basic-offset: 2
  indent-tabs-mode: nil
  End:
*/
