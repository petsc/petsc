/* ---------------------------------------------------------------- */

#include "compat.h"

#include "private/vecimpl.h"
#include "private/matimpl.h"
#include "private/kspimpl.h"
#include "private/snesimpl.h"
#include "private/tsimpl.h"

/* ---------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "PetscLogStageFindId"
PETSC_STATIC_INLINE PetscErrorCode
PetscLogStageFindId(const char name[], PetscLogStage *stageid)
{
  int            s;
  StageLog       stageLog = 0;
  PetscTruth     match = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidCharPointer(name,1);
  PetscValidIntPointer(stageid,2);
  *stageid = -1;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  if (!stageLog) PetscFunctionReturn(0); /* logging is turned off ? */
  for(s = 0; s < stageLog->numStages; s++) {
    const char *sname = stageLog->stageInfo[s].name;
    ierr = PetscStrcasecmp(sname, name, &match);CHKERRQ(ierr);
    if (match) { *stageid = s; break; }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLogClassFindId"
PETSC_STATIC_INLINE PetscErrorCode
PetscLogClassFindId(const char name[], PetscCookie *classid)
{
  int            c;
  StageLog       stageLog = 0;
  PetscTruth     match = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidCharPointer(name,1);
  PetscValidIntPointer(classid,2);
  *classid = -1;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  if (!stageLog) PetscFunctionReturn(0); /* logging is turned off ? */
  for(c = 0; c < stageLog->classLog->numClasses; c++) {
    const char *cname = stageLog->classLog->classInfo[c].name;
    PetscCookie cookie = stageLog->classLog->classInfo[c].cookie;
    ierr = PetscStrcasecmp(cname, name, &match);CHKERRQ(ierr);
    if (match) { *classid = cookie; break; }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLogEventFindId"
PETSC_STATIC_INLINE PetscErrorCode
PetscLogEventFindId(const char name[], PetscLogEvent *eventid)
{
  int            e;
  StageLog       stageLog = 0;
  PetscTruth     match = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidCharPointer(name,1);
  PetscValidIntPointer(eventid,2);
  *eventid = -1;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  if (!stageLog) PetscFunctionReturn(0); /* logging is turned off ? */
  for(e = 0; e < stageLog->eventLog->numEvents; e++) {
    const char *ename = stageLog->eventLog->eventInfo[e].name;
    ierr = PetscStrcasecmp(ename, name, &match);CHKERRQ(ierr);
    if (match) { *eventid = e; break; }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLogStageFindName"
PETSC_STATIC_INLINE PetscErrorCode
PetscLogStageFindName(PetscLogStage stageid,
                      const char *name[])
{
  StageLog       stageLog = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(name,3);
  *name = 0;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  if (!stageLog) PetscFunctionReturn(0); /* logging is turned off ? */
  if (stageid >=0 && stageid < stageLog->numStages) {
    *name  = stageLog->stageInfo[stageid].name;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLogClassFindName"
PETSC_STATIC_INLINE PetscErrorCode
PetscLogClassFindName(PetscCookie classid,
                      const char *name[])
{
  int            c;
  StageLog       stageLog = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(name,3);
  *name = 0;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  if (!stageLog) PetscFunctionReturn(0); /* logging is turned off ? */
  for(c = 0; c < stageLog->classLog->numClasses; c++) {
    if (classid == stageLog->classLog->classInfo[c].cookie) {
      *name  = stageLog->classLog->classInfo[c].name;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLogEventFindName"
PETSC_STATIC_INLINE PetscErrorCode
PetscLogEventFindName(PetscLogEvent eventid,
                      const char *name[])
{
  StageLog       stageLog = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(name,3);
  *name = 0;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  if (!stageLog) PetscFunctionReturn(0); /* logging is turned off ? */
  if (eventid >=0 && eventid < stageLog->eventLog->numEvents) {
    *name  = stageLog->eventLog->eventInfo[eventid].name;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "PetscObjectGetClassName"
PETSC_STATIC_INLINE PetscErrorCode
PetscObjectGetClassName(PetscObject obj, const char *class_name[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidPointer(class_name,2);
  *class_name = obj->class_name;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "ISValid"
PETSC_STATIC_INLINE PetscErrorCode
ISValid(IS is, PetscTruth *flg)
{
  PetscFunctionBegin;
  PetscValidIntPointer(flg,2);
  if (!is)                                         *flg = PETSC_FALSE;
  else if (((PetscObject)is)->cookie != IS_COOKIE) *flg = PETSC_FALSE;
  else                                             *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISGetType"
PETSC_STATIC_INLINE PetscErrorCode
ISGetType(IS is, ISType *istype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidPointer(istype,3);
  *istype = (ISType) ((PetscObject)is)->type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGetArrayC"
PETSC_STATIC_INLINE PetscErrorCode
VecGetArrayC(Vec v, PetscScalar *a[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1);
  PetscValidType(v,1);
  PetscValidPointer(a,2);
  ierr = VecGetArray(v,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecRestoreArrayC"
PETSC_STATIC_INLINE PetscErrorCode
VecRestoreArrayC(Vec v, PetscScalar *a[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1);
  PetscValidType(v,1);
  PetscValidPointer(a,2);
  ierr = VecRestoreArray(v,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

#if PETSC_VERSION_(2,3,2)
typedef PetscMap* PetscLayout;
#define PetscGetLayout(o, m) (&(o)->m)
#define PetscSetUpLayout(o, m) PetscMapInitialize((o)->comm,&(o)->m)
#elif PETSC_VERSION_(2,3,3)
typedef PetscMap* PetscLayout;
#define PetscGetLayout(o, m) (&(o)->m)
#define PetscSetUpLayout(o, m) PetscMapSetUp(&(o)->m)
#elif PETSC_VERSION_(3,0,0)
typedef PetscMap* PetscLayout;
#define PetscGetLayout(o, m) ((o)->m)
#define PetscSetUpLayout(o, m) PetscMapSetUp((o)->m)
#else
#define PetscGetLayout(o, m) ((o)->m)
#define PetscSetUpLayout(o, m) PetscLayoutSetUp((o)->m)
#endif

#undef __FUNCT__
#define __FUNCT__ "MatBlockSize_Check"
PETSC_STATIC_INLINE PetscErrorCode
MatBlockSize_Check(Mat mat,PetscInt bs)
{
  PetscLayout rmap = 0;
  PetscLayout cmap = 0;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  rmap = PetscGetLayout(mat,rmap);
  cmap = PetscGetLayout(mat,cmap);
  if (bs < 1)
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size specified, must be positive but it is %D",bs);
  if (rmap->n != -1 && rmap->n % bs)
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Local row length %D not divisible by block size %D",rmap->n,bs);
  if (rmap->N != -1 && rmap->N % bs)
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Global row length %D not divisible by block size %D",rmap->N,bs);
  if (cmap->n != -1 && cmap->n % bs)
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Local column length %D not divisible by block size %D",cmap->n,bs);
  if (cmap->N != -1 && cmap->N % bs)
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Global column length %D not divisible by block size %D",cmap->N,bs);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatBlockSize_SetUp"
PETSC_STATIC_INLINE PetscErrorCode
MatBlockSize_SetUp(Mat mat,PetscInt bs)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscGetLayout(mat,rmap)->bs = bs;
  PetscGetLayout(mat,cmap)->bs = bs;
  ierr = PetscSetUpLayout(mat,rmap);CHKERRQ(ierr);
  ierr = PetscSetUpLayout(mat,cmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatSetBlockSize_Patch"
PETSC_STATIC_INLINE PetscErrorCode
MatSetBlockSize_Patch(Mat mat,PetscInt bs)
{
  PetscLayout rmap = mat ? PetscGetLayout(mat,rmap): 0;
  PetscLayout cmap = mat ? PetscGetLayout(mat,cmap): 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (bs < 1)
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,
             "Invalid block size specified, must be positive but it is %D",bs);
  if (mat->ops->setblocksize) {
    ierr = MatBlockSize_Check(mat,bs);CHKERRQ(ierr);
    ierr = (*mat->ops->setblocksize)(mat,bs);CHKERRQ(ierr);
    ierr = MatBlockSize_SetUp(mat,bs);CHKERRQ(ierr);
  } else if (rmap->bs == -1 && cmap->bs == -1) {
    ierr = MatBlockSize_Check(mat,bs);CHKERRQ(ierr);
    ierr = MatBlockSize_SetUp(mat,bs);CHKERRQ(ierr);
  } else if (rmap->bs != bs || cmap->bs != bs) {
    SETERRQ1(PETSC_ERR_ARG_INCOMP,
             "Cannot set/change the block size for matrix type %s",
             ((PetscObject)mat)->type_name);
  }
  PetscFunctionReturn(0);
}
#undef  MatSetBlockSize
#define MatSetBlockSize MatSetBlockSize_Patch

#undef __FUNCT__
#define __FUNCT__ "MatAnyAIJSetPreallocation"
PETSC_STATIC_INLINE PetscErrorCode
MatIsPreallocated(Mat A,PetscTruth *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidPointer(flag,2);
  *flag = A->preallocated;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateAnyAIJ"
PETSC_STATIC_INLINE PetscErrorCode
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
    ierr = MatBlockSize_Check(mat,bs);CHKERRQ(ierr);
    ierr = MatBlockSize_SetUp(mat,bs);CHKERRQ(ierr);
  }
  *A = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAnyAIJSetPreallocation"
PETSC_STATIC_INLINE PetscErrorCode
MatAnyAIJSetPreallocation(Mat A,PetscInt bs,
                          PetscInt d_nz,const PetscInt d_nnz[],
                          PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscTruth     flag = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  if (d_nnz) PetscValidIntPointer(d_nnz,3);
  if (o_nnz) PetscValidIntPointer(o_nnz,5);
  ierr = MatIsPreallocated(A,&flag);CHKERRQ(ierr);
  if (flag) { SETERRQ(PETSC_ERR_ORDER, "matrix is already preallocated"); }
  if (bs == PETSC_DECIDE) {
    ierr = MatSeqAIJSetPreallocation(A,d_nz,d_nnz);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(A,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatBlockSize_Check(A,bs);CHKERRQ(ierr);
    ierr = MatSeqBAIJSetPreallocation(A,bs,d_nz,d_nnz);CHKERRQ(ierr);
    ierr = MatMPIBAIJSetPreallocation(A,bs,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
    ierr = MatBlockSize_SetUp(A,bs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAnyAIJSetPreallocationCSR"
PETSC_STATIC_INLINE PetscErrorCode
MatAnyAIJSetPreallocationCSR(Mat A,PetscInt bs, const PetscInt Ii[],
                             const PetscInt Jj[], const PetscScalar V[])
{
  PetscTruth     flag = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  PetscValidIntPointer(Ii,3);
  PetscValidIntPointer(Jj,4);
  if (V) PetscValidScalarPointer(V,5);
  ierr = MatIsPreallocated(A,&flag);CHKERRQ(ierr);
  if (flag) { SETERRQ(PETSC_ERR_ORDER, "matrix is already preallocated"); }
  if (bs == PETSC_DECIDE) {
    ierr = MatSeqAIJSetPreallocationCSR(A,Ii,Jj,V);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocationCSR(A,Ii,Jj,V);CHKERRQ(ierr);
  } else {
    ierr = MatBlockSize_Check(A,bs);CHKERRQ(ierr);
    ierr = MatSeqBAIJSetPreallocationCSR(A,bs,Ii,Jj,V);CHKERRQ(ierr);
    ierr = MatMPIBAIJSetPreallocationCSR(A,bs,Ii,Jj,V);CHKERRQ(ierr);
    ierr = MatBlockSize_SetUp(A,bs);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateAnyDense"
PETSC_STATIC_INLINE PetscErrorCode
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
    ierr = MatBlockSize_Check(mat,bs);CHKERRQ(ierr);
    ierr = MatBlockSize_SetUp(mat,bs);CHKERRQ(ierr);
  }
  *A = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAnyDenseSetPreallocation"
PETSC_STATIC_INLINE PetscErrorCode
MatAnyDenseSetPreallocation(Mat mat, PetscInt bs, PetscScalar *data)
{
  PetscTruth     flag = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (data) PetscValidScalarPointer(data,3);
  ierr = MatIsPreallocated(mat, &flag);CHKERRQ(ierr);
  if (flag) { SETERRQ(PETSC_ERR_ORDER, "matrix is already preallocated"); }
  ierr = MatSeqDenseSetPreallocation(mat,data);CHKERRQ(ierr);
  ierr = MatMPIDenseSetPreallocation(mat,data);CHKERRQ(ierr);
  if (bs != PETSC_DECIDE) {
    ierr = MatBlockSize_Check(mat,bs);CHKERRQ(ierr);
    ierr = MatBlockSize_SetUp(mat,bs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "MatFactorInfoDefaults()"
PETSC_STATIC_INLINE PetscErrorCode
MatFactorInfoDefaults(PetscTruth incomplete, MatFactorInfo *info)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(info,2);
  ierr = MatFactorInfoInitialize(info);CHKERRQ(ierr);
  if (incomplete) {
    info->dt             = PETSC_DEFAULT;
    info->dtcount        = PETSC_DEFAULT;
    info->dtcol          = PETSC_DEFAULT;
    info->fill           = PETSC_DEFAULT;
    info->shiftnz        = 1.e-12;
    info->shiftpd        = 0.0;
    info->zeropivot      = 1.e-12;
    info->pivotinblocks  = 1.0;
#if !PETSC_VERSION_(2,3,3) && !PETSC_VERSION_(2,3,2)
    info->shiftinblocks  = 1.e-12;
#endif
  } else {
    info->dtcol          = 1.e-6;
    info->fill           = 5.0;
    info->shiftnz        = 0.0;
    info->shiftpd        = 0.0;
    info->zeropivot      = 1.e-12;
    info->pivotinblocks  = 1.0;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "KSPSetIterationNumber"
PETSC_STATIC_INLINE PetscErrorCode
KSPSetIterationNumber(KSP ksp, PetscInt its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (its < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"iteration number must be nonnegative");
  ksp->its = its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetResidualNorm"
PETSC_STATIC_INLINE PetscErrorCode
KSPSetResidualNorm(KSP ksp, PetscReal rnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (rnorm < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"residual norm must be nonnegative");
  ksp->rnorm = rnorm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPLogConvergenceHistory"
PETSC_STATIC_INLINE PetscErrorCode
KSPLogConvergenceHistory(KSP ksp, PetscInt its, PetscReal rnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (its   < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"iteration number must be nonnegative");
  if (rnorm < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"residual norm must be nonnegative");
  KSPLogResidualHistory(ksp,rnorm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorCall"
PETSC_STATIC_INLINE PetscErrorCode
KSPMonitorCall(KSP ksp, PetscInt its, PetscReal rnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (its   < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"iteration number must be nonnegative");
  if (rnorm < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"residual norm must be nonnegative");
  KSPMonitor(ksp,its,rnorm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPConvergenceTestCall"
PETSC_STATIC_INLINE PetscErrorCode
KSPConvergenceTestCall(KSP ksp, PetscInt its, PetscReal rnorm, KSPConvergedReason *reason)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidPointer(reason,4);
  if (its   < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"iteration number must be nonnegative");
  if (rnorm < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"residual norm must be nonnegative");
  ierr = (*ksp->converged)(ksp,its,rnorm,reason,ksp->cnvP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetConvergedReason"
PETSC_STATIC_INLINE PetscErrorCode
KSPSetConvergedReason(KSP ksp, KSPConvergedReason reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ksp->reason = reason;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "SNESSetIterationNumber"
PETSC_STATIC_INLINE PetscErrorCode
SNESSetIterationNumber(SNES snes, PetscInt its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (its < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"iteration number must be nonnegative");
  snes->iter = its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFunctionNorm"
PETSC_STATIC_INLINE PetscErrorCode
SNESSetFunctionNorm(SNES snes, PetscReal fnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (fnorm < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"function norm must be nonnegative");
  snes->norm = fnorm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLogResidualHistoryCall"
PETSC_STATIC_INLINE PetscErrorCode
SNESLogConvergenceHistory(SNES snes, PetscInt its, PetscReal fnorm, PetscInt lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (its   < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"iteration number must be nonnegative");
  if (fnorm < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"function norm must be nonnegative");
  SNESLogConvHistory(snes,fnorm,its);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESMonitorCall"
PETSC_STATIC_INLINE PetscErrorCode
SNESMonitorCall(SNES snes, PetscInt its, PetscReal rnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (its   < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"iteration number must be nonnegative");
  if (rnorm < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"residual norm must be nonnegative");
  SNESMonitor(snes,its,rnorm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESConvergenceTestCall"
PETSC_STATIC_INLINE PetscErrorCode
SNESConvergenceTestCall(SNES snes, PetscInt its,
                        PetscReal xnorm, PetscReal ynorm, PetscReal fnorm,
                        SNESConvergedReason *reason)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(reason,4);
  if (its   < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"iteration number must be nonnegative");
  if (xnorm < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"solution norm must be nonnegative");
  if (ynorm < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"step norm must be nonnegative");
  if (fnorm < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"function norm must be nonnegative");
  ierr = (*snes->ops->converged)(snes,its,xnorm,ynorm,fnorm,reason,snes->cnvP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetConvergedReason"
PETSC_STATIC_INLINE PetscErrorCode
SNESSetConvergedReason(SNES snes, SNESConvergedReason reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  snes->reason = reason;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFDColoringSetOptionsPrefix"
PETSC_STATIC_INLINE PetscErrorCode
MatFDColoringSetOptionsPrefix(MatFDColoring fdc, const char prefix[]) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fdc,MAT_FDCOLORING_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)fdc,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUseMFFD"
static PetscErrorCode
SNESGetUseMFFD(SNES snes,PetscTruth *flag)
{
  Mat            J = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(flag,2);
  *flag = PETSC_FALSE;
  ierr = SNESGetJacobian(snes,&J,0,0,0);CHKERRQ(ierr);
  if (J) { ierr = PetscTypeCompare((PetscObject)J,MATMFFD,flag);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUseMFFD"
PETSC_STATIC_INLINE PetscErrorCode
SNESSetUseMFFD(SNES snes,PetscTruth flag)
{
  const char*    prefix = PETSC_NULL;
  PetscTruth     flg = PETSC_FALSE;
  Vec            r = PETSC_NULL;
  Mat            A = PETSC_NULL,B = PETSC_NULL,J = PETSC_NULL;
  KSP            ksp = PETSC_NULL;
  PC             pc = PETSC_NULL;
  void*          funP = PETSC_NULL;
  void*          jacP = PETSC_NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);

  ierr = SNESGetUseMFFD(snes,&flg);CHKERRQ(ierr);
  if ( flg &&  flag) PetscFunctionReturn(0);
  if (!flg && !flag) PetscFunctionReturn(0);
  if ( flg && !flag) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"cannot change matrix-free once it is set");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }

  ierr = SNESGetFunction(snes,&r,0,&funP);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes,&A,&B,0,&jacP);CHKERRQ(ierr);
  if (r == PETSC_NULL) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"SNESSetFunction() must be called first");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }
  ierr = MatCreateSNESMF(snes,&J);CHKERRQ(ierr);
  ierr = SNESGetOptionsPrefix(snes,&prefix);CHKERRQ(ierr);
  ierr = MatMFFDSetOptionsPrefix(J,prefix);CHKERRQ(ierr);
  ierr = MatMFFDSetFromOptions(J);CHKERRQ(ierr);

  if (B == PETSC_NULL) {
    ierr = SNESSetJacobian(snes,J,J,MatMFFDComputeJacobian,jacP);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)pc,PCSHELL,&flg);CHKERRQ(ierr);
    if (!flg) { ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr); }
  } else {
    ierr = SNESSetJacobian(snes,J,0,0,0);CHKERRQ(ierr);
  }

  ierr = MatDestroy(J);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESGetUseFDColoring"
PETSC_STATIC_INLINE PetscErrorCode
SNESGetUseFDColoring(SNES snes,PetscTruth *flag)
{
  PetscErrorCode (*jac)(SNES,Vec,Mat*,Mat*,MatStructure*,void*) = PETSC_NULL;
  MatFDColoring  fdcoloring = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(flag,2);
  *flag = PETSC_FALSE;
  ierr = SNESGetJacobian(snes,0,0,&jac,(void**)&fdcoloring);CHKERRQ(ierr);
  if (jac == SNESDefaultComputeJacobianColor) *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUseFDColoring"
PETSC_STATIC_INLINE PetscErrorCode
SNESSetUseFDColoring(SNES snes,PetscTruth flag)
{
  const char*    prefix = PETSC_NULL;
  PetscTruth     flg = PETSC_FALSE;
  Vec            f = PETSC_NULL;
  PetscErrorCode (*fun)(SNES,Vec,Vec,void*) = PETSC_NULL;
  void*          funP = PETSC_NULL;
  Mat            A = PETSC_NULL,B = PETSC_NULL,J = PETSC_NULL;
  PetscErrorCode (*jac)(SNES,Vec,Mat*,Mat*,MatStructure*,void*) = PETSC_NULL;
  void*          jacP = PETSC_NULL;
  ISColoring     iscoloring = PETSC_NULL;
  MatFDColoring  fdcoloring = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);

  ierr = SNESGetUseFDColoring(snes,&flg);CHKERRQ(ierr);
  if ( flg &&  flag) PetscFunctionReturn(0);
  if (!flg && !flag) PetscFunctionReturn(0);
  if ( flg && !flag) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "cannot change colored finite diferences once it is set");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }

  ierr = SNESGetFunction(snes,&f,&fun,&funP);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes,&A,&B,&jac,&jacP);CHKERRQ(ierr);
  if (f == PETSC_NULL) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"SNESSetFunction() must be called first");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }
  if (A == PETSC_NULL && B == PETSC_NULL) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"SNESSetJacobian() must be called first");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }

  J = (B != PETSC_NULL) ? B : A;

  ierr = MatGetColoring(J,MATCOLORING_SL,&iscoloring);CHKERRQ(ierr);
  ierr = MatFDColoringCreate(J,iscoloring,&fdcoloring);CHKERRQ(ierr);
  ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);

  ierr = MatFDColoringSetFunction(fdcoloring,(PetscErrorCode (*)(void))fun,funP);
  ierr = SNESGetOptionsPrefix(snes,&prefix);CHKERRQ(ierr);
  ierr = MatFDColoringSetOptionsPrefix(fdcoloring,prefix);CHKERRQ(ierr);
  ierr = MatFDColoringSetFromOptions(fdcoloring);CHKERRQ(ierr);

  ierr = SNESSetJacobian(snes,A,B,SNESDefaultComputeJacobianColor,fdcoloring);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)snes,"fdcoloring",(PetscObject)fdcoloring);CHKERRQ(ierr);
  ierr = MatFDColoringDestroy(fdcoloring);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "TSSetRHSFunction_Ex"
static PetscErrorCode
TSSetRHSFunction_Ex(TS ts,Vec r,PetscErrorCode (*fun)(TS,PetscReal,Vec,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TSSetRHSFunction(ts,fun,ctx);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)ts,"__rhs_funcvec__",(PetscObject)r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define TSSetRHSFunction TSSetRHSFunction_Ex

#undef __FUNCT__
#define __FUNCT__ "TSSetRHSFunction_Ex"
static PetscErrorCode
TSGetRHSFunction_Ex(TS ts,Vec *f,PetscErrorCode (**fun)(TS,PetscReal,Vec,Vec,void*),void **ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (f) {ierr = PetscObjectQuery((PetscObject)ts, "__rhs_funcvec__", (PetscObject*)f);CHKERRQ(ierr); }
  if (fun) *fun = ts->ops->rhsfunction;
  if (ctx) *ctx = ts->funP;
  PetscFunctionReturn(0);
}
#define TSGetRHSFunction TSGetRHSFunction_Ex

#undef __FUNCT__
#define __FUNCT__ "TSGetRHSJacobian_Ex"
PETSC_STATIC_INLINE PetscErrorCode
TSGetRHSJacobian_Ex(TS ts,Mat *A,Mat *B,PetscErrorCode (**jac)(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*),void **ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  ierr = TSGetRHSJacobian(ts,A,B,ctx);CHKERRQ(ierr);
  if (jac) *jac = ts->ops->rhsjacobian;
  PetscFunctionReturn(0);
}
#define TSGetRHSJacobian TSGetRHSJacobian_Ex


#undef __FUNCT__
#define __FUNCT__ "TSGetUseFDColoring"
PETSC_STATIC_INLINE PetscErrorCode
TSGetUseFDColoring(TS ts,PetscTruth *flag)
{
  PetscErrorCode (*jac)(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*) = PETSC_NULL;
  MatFDColoring  fdcoloring = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidPointer(flag,2);
  *flag = PETSC_FALSE;
  ierr = TSGetRHSJacobian(ts,0,0,&jac,(void**)&fdcoloring);CHKERRQ(ierr);
  if (jac == TSDefaultComputeJacobianColor) *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetUseFDColoring"
PETSC_STATIC_INLINE PetscErrorCode
TSSetUseFDColoring(TS ts,PetscTruth flag)
{
  const char*    prefix = PETSC_NULL;
  PetscTruth     flg = PETSC_FALSE;
  Vec            f = PETSC_NULL;
  PetscErrorCode (*fun)(TS,PetscReal,Vec,Vec,void*) = PETSC_NULL;
  void*          funP = PETSC_NULL;
  Mat            A = PETSC_NULL,B = PETSC_NULL,J = PETSC_NULL;
  PetscErrorCode (*jac)(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*) = PETSC_NULL;
  void*          jacP  = PETSC_NULL;
  ISColoring     iscoloring = PETSC_NULL;
  MatFDColoring  matfdcoloring = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);

  ierr = TSGetUseFDColoring(ts,&flg);CHKERRQ(ierr);
  if ( flg &&  flag) PetscFunctionReturn(0);
  if (!flg && !flag) PetscFunctionReturn(0);
  if ( flg && !flag) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"cannot change colored finite deferences once it is set");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }
  ierr = TSGetRHSFunction(ts,&f,&fun,&funP);CHKERRQ(ierr);
  ierr = TSGetRHSJacobian(ts,&A,&B,&jac,&jacP);CHKERRQ(ierr);
  if (fun == PETSC_NULL) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"TSSetRHSFunction() must be called first");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }
  if (A == PETSC_NULL && B == PETSC_NULL) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"TSSetRHSJacobian() must be called first");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }

  J = (B != PETSC_NULL) ? B : A;
  ierr = MatGetColoring(J,MATCOLORING_SL,&iscoloring);CHKERRQ(ierr);
  ierr = MatFDColoringCreate(J,iscoloring,&matfdcoloring);CHKERRQ(ierr);
  ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);

  ierr = MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))fun,funP);
  ierr = TSGetOptionsPrefix(ts,&prefix);CHKERRQ(ierr);
  ierr = MatFDColoringSetOptionsPrefix(matfdcoloring,prefix);CHKERRQ(ierr);
  ierr = MatFDColoringSetFromOptions(matfdcoloring);CHKERRQ(ierr);

  ierr = TSSetRHSJacobian(ts,A,B,TSDefaultComputeJacobianColor,matfdcoloring);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)ts,"matfdcoloring",(PetscObject)matfdcoloring);CHKERRQ(ierr);
  ierr = MatFDColoringDestroy(matfdcoloring);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "TSSetSolution_Patch"
PETSC_STATIC_INLINE PetscErrorCode
TSSetSolution_Patch(TS ts, Vec u)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidHeaderSpecific(u,VEC_COOKIE,2);
  ierr = PetscObjectCompose((PetscObject)ts,"__solnvec__",(PetscObject)u);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef  TSSetSolution
#define TSSetSolution TSSetSolution_Patch

#undef __FUNCT__
#define __FUNCT__ "TSSolve_Patch"
PETSC_STATIC_INLINE PetscErrorCode
TSSolve_Patch(TS ts, Vec x)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (x) { ierr = TSSetSolution(ts, x); CHKERRQ(ierr); }
  ierr = TSSolve(ts,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef  TSSolve
#define TSSolve TSSolve_Patch

#undef __FUNCT__
#define __FUNCT__ "TSSetTimeStepNumber"
PETSC_STATIC_INLINE PetscErrorCode
TSSetTimeStepNumber(TS ts, PetscInt step)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  ts->steps = step;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMonitorCall"
PETSC_STATIC_INLINE PetscErrorCode
TSMonitorCall(TS ts,PetscInt step,PetscReal ptime,Vec x)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  ierr = TSMonitor(ts,step,ptime,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "AOGetType"
PETSC_STATIC_INLINE PetscErrorCode
AOGetType(AO ao, AOType *aotype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_COOKIE,1);
  PetscValidPointer(aotype,3);
  *aotype = (AOType) ((PetscObject)ao)->type;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */

#if (PETSC_VERSION_(3,0,0) || \
     PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2) )

#undef __FUNCT__
#define __FUNCT__ "DACreateND"
PETSC_STATIC_INLINE PetscErrorCode
DACreateND(MPI_Comm comm,
           PetscInt dim,PetscInt dof,
           PetscInt M,PetscInt N,PetscInt P,
           PetscInt m,PetscInt n,PetscInt p,
           const PetscInt lx[],const PetscInt ly[],const PetscInt lz[],
           DAPeriodicType wrap,DAStencilType stencil_type,PetscInt stencil_width,
           DA *da)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DACreate(comm,dim,wrap,stencil_type,
                  M,N,P,m,n,p,dof,stencil_width,
                  lx,ly,lz, da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__
#define __FUNCT__ "DACreateND"
PETSC_STATIC_INLINE PetscErrorCode
DACreateND(MPI_Comm comm,
           PetscInt dim,PetscInt dof,
           PetscInt M,PetscInt N,PetscInt P,
           PetscInt m,PetscInt n,PetscInt p,
           const PetscInt lx[],const PetscInt ly[],const PetscInt lz[],
           DAPeriodicType wrap,DAStencilType stencil_type,PetscInt stencil_width,
           DA *da)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DACreate(comm,da);CHKERRQ(ierr);
  ierr = DASetDim(*da,dim);CHKERRQ(ierr);
  ierr = DASetDof(*da,dof);CHKERRQ(ierr);
  ierr = DASetSizes(*da,M,N,P);CHKERRQ(ierr);
  ierr = DASetNumProcs(*da,m,n,p);CHKERRQ(ierr);
  ierr = DASetVertexDivision(*da,lx,ly,lz);CHKERRQ(ierr);
  ierr = DASetPeriodicity(*da,wrap);CHKERRQ(ierr);
  ierr = DASetStencilType(*da,stencil_type);CHKERRQ(ierr);
  ierr = DASetStencilWidth(*da,stencil_width);CHKERRQ(ierr);

  /* This violates the behavior for other classes, but right now users
     expect negative dimensions to be handled this way */
  /*
  ierr = DASetOptionsPrefix(*da,prefix);CHKERRQ(ierr);CHKERRQ(ierr);
  */
  ierr = DASetFromOptions(*da);CHKERRQ(ierr);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#endif

/* ---------------------------------------------------------------- */
