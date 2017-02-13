
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include "../src/vec/vec/utils/tagger/impls/simple.h"

const char *const VecTaggerCDFMethods[VECTAGGER_CDF_NUM_METHODS] = {"gather","iterative"};

typedef struct {
  VecTagger_Simple   smpl;
  PetscReal          atol;
  PetscReal          rtol;
  PetscInt           maxit;
  PetscInt           numMoments;
  VecTaggerCDFMethod method;
} VecTagger_CDF;

static PetscErrorCode VecTaggerComputeInterval_CDF_SortedArray(const PetscReal *cArray, PetscInt m, const PetscReal ints[2], PetscReal intervals[2])
{
  PetscInt    minInd, maxInd;
  PetscReal   minCDF, maxCDF;

  PetscFunctionBegin;
  minCDF = PetscMax(0., ints[0]);
  maxCDF = PetscMin(1., ints[1]);
  minInd = (PetscInt) (minCDF * m);
  maxInd = (PetscInt) (maxCDF * m);
  intervals[0] = cArray[PetscMin(minInd,m-1)];
  intervals[1] = cArray[PetscMax(minInd,maxInd-1)];
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeIntervals_CDF_Serial(VecTagger tagger,Vec vec,PetscInt bs,PetscScalar (*intervals)[2])
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;
  Vec              vComp;
  PetscInt         n, m;
  PetscInt         i;
#if defined (PETSC_USE_COMPLEX)
  PetscReal        *cReal, *cImag;
#endif
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
  m    = n/bs;
  ierr = VecCreateSeq(PETSC_COMM_SELF,m,&vComp);CHKERRQ(ierr);
#if defined (PETSC_USE_COMPLEX)
  ierr = PetscMalloc2(m,&cReal,m,&cImag);CHKERRQ(ierr);
#endif
  for (i = 0; i < bs; i++) {
    IS          isStride;
    VecScatter  vScat;
    PetscScalar *cArray;

    ierr = ISCreateStride(PETSC_COMM_SELF,m,i,bs,&isStride);CHKERRQ(ierr);
    ierr = VecScatterCreate(vec,isStride,vComp,NULL,&vScat);CHKERRQ(ierr);
    ierr = VecScatterBegin(vScat,vec,vComp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(vScat,vec,vComp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&vScat);CHKERRQ(ierr);
    ierr = ISDestroy(&isStride);CHKERRQ(ierr);

    ierr = VecGetArray(vComp,&cArray);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = PetscSortReal(m,cArray);CHKERRQ(ierr);
    ierr = VecTaggerComputeInterval_CDF_SortedArray(cArray,m,smpl->interval[i],intervals[i]);CHKERRQ(ierr);
#else
    {
      PetscInt j;
      PetscReal realInt[2], imagInt[2];
      PetscReal realIntervals[2], imagIntervals[2];

      for (j = 0; j < m; j++) {
        cReal[j] = PetscRealPart(cArray[j]);
        cImag[j] = PetscImaginaryPart(cArray[j]);
      }
      ierr = PetscSortReal(m,cReal);CHKERRQ(ierr);
      ierr = PetscSortReal(m,cImag);CHKERRQ(ierr);

      realInt[0] = PetscRealPart(smpl->interval[i][0]);
      realInt[1] = PetscRealPart(smpl->interval[i][1]);
      imagInt[0] = PetscImaginaryPart(smpl->interval[i][0]);
      imagInt[1] = PetscImaginaryPart(smpl->interval[i][1]);
      ierr = VecTaggerComputeInterval_CDF_SortedArray(cReal,m,realInt,realIntervals);CHKERRQ(ierr);
      ierr = VecTaggerComputeInterval_CDF_SortedArray(cImag,m,imagInt,imagIntervals);CHKERRQ(ierr);
      intervals[i][0] = PetscCMPLX(realIntervals[0],imagIntervals[0]);
      intervals[i][1] = PetscCMPLX(realIntervals[1],imagIntervals[1]);
    }
#endif
    ierr = VecRestoreArray(vComp,&cArray);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree2(cReal,cImag);CHKERRQ(ierr);
#endif
  ierr = VecDestroy(&vComp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeIntervals_CDF_Gather(VecTagger tagger,Vec vec,PetscInt bs,PetscScalar (*intervals)[2])
{
  Vec            gVec = NULL;
  VecScatter     vScat;
  PetscInt       rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterCreateToZero(vec,&vScat,&gVec);CHKERRQ(ierr);
  ierr = VecScatterBegin(vScat,vec,gVec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(vScat,vec,gVec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&vScat);CHKERRQ(ierr);
  ierr = MPI_Comm_rank (PetscObjectComm((PetscObject)vec),&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = VecTaggerComputeIntervals_CDF_Serial(tagger,gVec,bs,intervals);CHKERRQ(ierr);
  }
  ierr = MPI_Bcast(intervals,2*bs,MPIU_SCALAR,0,PetscObjectComm((PetscObject)vec));CHKERRQ(ierr);
  ierr = VecDestroy(&gVec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct _n_CDFStats
{
  PetscReal min;
  PetscReal max;
  PetscReal moment[3];

} CDFStats;

static void VecTaggerCDFStatsReduce(void *a, void *b, int * len, MPI_Datatype *datatype)
{
  PetscInt  i, j, N = *len;
  CDFStats *A = (CDFStats *) a;
  CDFStats *B = (CDFStats *) b;

  for (i = 0; i < N; i++) {
    B[i].min = PetscMin(A[i].min,B[i].min);
    B[i].max = PetscMax(A[i].max,B[i].max);
    for (j = 0; j < 3; j++) {
      B[i].moment[j] += A[i].moment[j];
    }
  }
}

static PetscErrorCode CDFUtilInverseEstimate(const CDFStats *stats,PetscReal cdfTarget,PetscReal *absEst)
{
  PetscReal min, max;

  PetscFunctionBegin;
  min = stats->min;
  max = stats->max;
  *absEst = min + cdfTarget * (max - min);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeInterval_CDF_SortedArray_Iterative(VecTagger tagger, MPI_Datatype statType, MPI_Op statReduce, const PetscReal *cArray, PetscInt m, const PetscReal cdfInt[2], PetscReal absInt[2])
{
  MPI_Comm       comm;
  VecTagger_CDF  *cdf;
  PetscInt       maxit, i, j, k, l, M;
  PetscInt       bounds[2][2];
  PetscInt       offsets[2];
  PetscReal      intervalLen = cdfInt[1] - cdfInt[0];
  PetscReal      rtol, atol;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  comm  = PetscObjectComm((PetscObject)tagger);
  cdf   = (VecTagger_CDF *) tagger->data;
  maxit = cdf->maxit;
  rtol  = cdf->rtol;
  atol  = cdf->atol;
  /* local range of sorted values that can contain the sought radix */
  offsets[0] = 0;
  offsets[1] = 0;
  bounds[0][0] = 0; bounds[0][1] = m;
  bounds[1][0] = 0; bounds[1][1] = m;
  ierr = VecTaggerComputeInterval_CDF_SortedArray(cArray,m,cdfInt,absInt);CHKERRQ(ierr); /* compute a local estimate of the interval */
  {
    CDFStats stats[3];

    for (i = 0; i < 2; i++) { /* compute statistics of those local estimates */
      stats[i].min = m ? absInt[i] : PETSC_MAX_REAL;
      stats[i].max = m ? absInt[i] : PETSC_MIN_REAL;
      stats[i].moment[0] = m;
      stats[i].moment[1] = m * absInt[i];
      stats[i].moment[2] = m * absInt[i] * absInt[i];
    }
    stats[2].min = PETSC_MAX_REAL;
    stats[2].max = PETSC_MAX_REAL;
    for (i = 0; i < 3; i++) {
      stats[2].moment[i] = 0.;
    }
    for (i = 0; i < m; i++) {
      PetscReal val = cArray[i];

      stats[2].min = PetscMin(stats[2].min, val);
      stats[2].max = PetscMax(stats[2].max, val);
      stats[2].moment[0] ++;
      stats[2].moment[1] += val;
      stats[2].moment[2] += val * val;
    }
    /* reduce those statistics */
    ierr = MPI_Allreduce(MPI_IN_PLACE,stats,3,statType,statReduce,comm);CHKERRQ(ierr);
    M    = (PetscInt) stats[2].moment[0];
    /* use those initial statistics to get the initial (globally agreed-upon) choices for the absolute interval bounds */
    for (i = 0; i < 2; i++) {
      ierr = CDFUtilInverseEstimate(&stats[i],cdfInt[i],&absInt[i]);CHKERRQ(ierr);
    }
  }
  /* refine the estimates by computing how close they come to the desired interval and refining */
  for (k = 0; k < maxit; k++) {
    PetscReal maxDiff = 0.;

    CDFStats stats[2][2];
    PetscInt newBounds[2][2][2];
    for (i = 0; i < 2; i++) {
      for (j = 0; j < 2; j++) {
        stats[i][j].min = PETSC_MAX_REAL;
        stats[i][j].max = PETSC_MIN_REAL;
        for (l = 0; l < 3; l++) {
          stats[i][j].moment[l] = 0.;
        }
        newBounds[i][j][0] = PetscMax(bounds[i][0],bounds[i][1]);
        newBounds[i][j][1] = PetscMin(bounds[i][0],bounds[i][1]);
      }
    }
    for (i = 0; i < 2; i++) {
      for (j = 0; j < bounds[i][1] - bounds[i][0]; j++) {
        PetscInt  thisInd = bounds[i][0] + j;
        PetscReal val     = cArray[thisInd];
        PetscInt  section;
        if (!i) {
          section = (val <  absInt[0]) ? 0 : 1;
        } else {
          section = (val <= absInt[1]) ? 0 : 1;
        }
        stats[i][section].min = PetscMin(stats[i][section].min,val);
        stats[i][section].max = PetscMax(stats[i][section].max,val);
        stats[i][section].moment[0] ++;
        stats[i][section].moment[1] += val;
        stats[i][section].moment[2] += val * val;
        newBounds[i][section][0] = PetscMin(newBounds[i][section][0],thisInd);
        newBounds[i][section][1] = PetscMax(newBounds[i][section][0],thisInd + 1);
      }
    }
    ierr = MPI_Allreduce(MPI_IN_PLACE, stats, 4, statType, statReduce, comm);CHKERRQ(ierr);
    for (i = 0; i < 2; i++) {
      PetscInt  totalLessThan = offsets[i] + stats[i][0].moment[0];
      PetscReal cdfOfAbs      = (PetscReal) totalLessThan / (PetscReal) M;
      PetscReal diff;
      PetscInt  section;

      if (cdfOfAbs == cdfInt[i]) {
        offsets[i] = totalLessThan;
        bounds[i][0] = bounds[i][1] = 0;
        continue;
      }
      if (cdfOfAbs > cdfInt[i]) { /* the correct absolute value lies in the lower section */
        section = 0;
      } else {
        section = 1;
        offsets[i] = totalLessThan;
      }
      for (j = 0; j < 2; j++) {
        bounds[i][j] = newBounds[i][section][j];
      }
      ierr = CDFUtilInverseEstimate(&stats[i][section],(cdfInt[i] - ((PetscReal) offsets[i] / (PetscReal) M))/stats[i][section].moment[0],&absInt[i]);CHKERRQ(ierr);
      diff = PetscAbs(cdfOfAbs - cdfInt[i]);
      maxDiff = PetscMax(maxDiff,diff);
    }
    if (!maxDiff) PetscFunctionReturn(0);
    if ((atol || rtol) && ((!atol) || (maxDiff <= atol)) && ((!rtol) || (maxDiff <= rtol * intervalLen))) {
      break;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeIntervals_CDF_Iterative(VecTagger tagger,Vec vec,PetscInt bs,PetscScalar (*intervals)[2])
{
  VecTagger_CDF    *cdf = (VecTagger_CDF *) tagger->data;
  VecTagger_Simple *smpl = &(cdf->smpl);
  Vec              vComp;
  PetscInt         i, N, M, n, m, rstart;
#if defined (PETSC_USE_COMPLEX)
  PetscReal        *cReal, *cImag;
#endif
  MPI_Comm         comm;
  MPI_Datatype     statType;
  MPI_Op           statReduce;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)vec);
  ierr = VecGetSize(vec,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
  M    = N/bs;
  m    = n/bs;
  ierr = VecCreateMPI(comm,m,M,&vComp);CHKERRQ(ierr);
  ierr = VecSetUp(vComp);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(vComp,&rstart,NULL);CHKERRQ(ierr);
#if defined (PETSC_USE_COMPLEX)
  ierr = PetscMalloc2(m,&cReal,m,&cImag);CHKERRQ(ierr);
#endif
  ierr = MPI_Type_contiguous(5,MPIU_REAL,&statType);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&statType);CHKERRQ(ierr);
  ierr = MPI_Op_create(VecTaggerCDFStatsReduce,1,&statReduce);CHKERRQ(ierr);
  for (i = 0; i < bs; i++) {
    IS          isStride;
    VecScatter  vScat;
    PetscScalar *cArray;

    ierr = ISCreateStride(comm,m,bs * rstart + i,bs,&isStride);CHKERRQ(ierr);
    ierr = VecScatterCreate(vec,isStride,vComp,NULL,&vScat);CHKERRQ(ierr);
    ierr = VecScatterBegin(vScat,vec,vComp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(vScat,vec,vComp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&vScat);CHKERRQ(ierr);
    ierr = ISDestroy(&isStride);CHKERRQ(ierr);

    ierr = VecGetArray(vComp,&cArray);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = PetscSortReal(m,cArray);CHKERRQ(ierr);
    ierr = VecTaggerComputeInterval_CDF_SortedArray_Iterative(tagger,statType,statReduce,cArray,m,smpl->interval[i],intervals[i]);CHKERRQ(ierr);
#else
    {
      PetscInt j;
      PetscReal realInt[2], imagInt[2];
      PetscReal realIntervals[2], imagIntervals[2];

      for (j = 0; j < m; j++) {
        cReal[j] = PetscRealPart(cArray[j]);
        cImag[j] = PetscImaginaryPart(cArray[j]);
      }
      ierr = PetscSortReal(m,cReal);CHKERRQ(ierr);
      ierr = PetscSortReal(m,cImag);CHKERRQ(ierr);

      realInt[0] = PetscRealPart(smpl->interval[i][0]);
      realInt[1] = PetscRealPart(smpl->interval[i][1]);
      imagInt[0] = PetscImaginaryPart(smpl->interval[i][0]);
      imagInt[1] = PetscImaginaryPart(smpl->interval[i][1]);
      ierr = VecTaggerComputeInterval_CDF_SortedArray_Iterative(tagger,statType,statReduce,cReal,m,realInt,realIntervals);CHKERRQ(ierr);
      ierr = VecTaggerComputeInterval_CDF_SortedArray_Iterative(tagger,statType,statReduce,cImag,m,imagInt,imagIntervals);CHKERRQ(ierr);
      intervals[i][0] = PetscCMPLX(realIntervals[0],imagIntervals[0]);
      intervals[i][1] = PetscCMPLX(realIntervals[1],imagIntervals[1]);
    }
#endif
    ierr = VecRestoreArray(vComp,&cArray);CHKERRQ(ierr);
  }
  ierr = MPI_Op_free(&statReduce);CHKERRQ(ierr);
  ierr = MPI_Type_free(&statType);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree2(cReal,cImag);CHKERRQ(ierr);
#endif
  ierr = VecDestroy(&vComp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeIntervals_CDF(VecTagger tagger,Vec vec,PetscInt *numIntervals,PetscScalar (**intervals)[2])
{
  VecTagger_CDF *cuml = (VecTagger_CDF *)tagger->data;
  PetscMPIInt          size;
  PetscInt             bs;
  PetscScalar          (*ints) [2];
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = VecTaggerGetBlockSize(tagger,&bs);CHKERRQ(ierr);
  *numIntervals = 1;
  ierr = PetscMalloc1(bs,&ints);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)tagger),&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = VecTaggerComputeIntervals_CDF_Serial(tagger,vec,bs,ints);CHKERRQ(ierr);
    *intervals = ints;
    PetscFunctionReturn(0);
  }
  switch (cuml->method) {
  case VECTAGGER_CDF_GATHER:
    ierr = VecTaggerComputeIntervals_CDF_Gather(tagger,vec,bs,ints);CHKERRQ(ierr);
    break;
  case VECTAGGER_CDF_ITERATIVE:
    ierr = VecTaggerComputeIntervals_CDF_Iterative(tagger,vec,bs,ints);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tagger),PETSC_ERR_SUP,"Unknown CDF calculation/estimation method.");
  }
  *intervals = ints;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerView_CDF(VecTagger tagger,PetscViewer viewer)
{
  VecTagger_CDF *cuml = (VecTagger_CDF *) tagger->data;
  PetscBool            iascii;
  PetscInt             size;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = VecTaggerView_Simple(tagger,viewer);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)tagger),&size);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (size > 1 && iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"CDF computation method: %s\n",VecTaggerCDFMethods[cuml->method]);CHKERRQ(ierr);
    if (cuml->method == VECTAGGER_CDF_ITERATIVE) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"max its: %D, abs tol: %g, rel tol %g\n",cuml->maxit,(double) cuml->atol,(double) cuml->rtol);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerSetFromOptions_CDF(PetscOptionItems *PetscOptionsObject,VecTagger tagger)
{
  VecTagger_CDF *cuml = (VecTagger_CDF *) tagger->data;
  PetscInt       method;
  PetscBool      set;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerSetFromOptions_Simple(PetscOptionsObject,tagger);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"VecTagger options for CDF intervals");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-vec_tagger_cdf_method","Method for computing absolute intervals from CDF intervals","VecTaggerCDFSetMethod()",VecTaggerCDFMethods,VECTAGGER_CDF_NUM_METHODS,VecTaggerCDFMethods[cuml->method],&method,&set);CHKERRQ(ierr);
  if (set) cuml->method = (VecTaggerCDFMethod) method;
  ierr = PetscOptionsInt("-vec_tagger_cdf_max_it","Maximum iterations for iterative computation of absolute intervals from CDF intervals","VecTaggerCDFIterativeSetTolerances()",cuml->maxit,&cuml->maxit,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-vec_tagger_cdf_rtol","Maximum relative tolerance for iterative computation of absolute intervals from CDF intervals","VecTaggerCDFIterativeSetTolerances()",cuml->rtol,&cuml->rtol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-vec_tagger_cdf_atol","Maximum absolute tolerance for iterative computation of absolute intervals from CDF intervals","VecTaggerCDFIterativeSetTolerances()",cuml->atol,&cuml->atol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*@C
  VecTaggerCDFSetMethod - Set the method used to compute absolute intervals from CDF intervals

  Logically Collective on VecTagger

  Level: advanced

  Input Parameters:
+ tagger - the VecTagger context
- method - the method

.seealso VecTaggerCDFMethod
@*/
PetscErrorCode VecTaggerCDFSetMethod(VecTagger tagger, VecTaggerCDFMethod method)
{
  VecTagger_CDF  *cuml = (VecTagger_CDF *) tagger->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidLogicalCollectiveInt(tagger,method,2);
  cuml->method = method;
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerCDFGetMethod - Get the method used to compute absolute intervals from CDF intervals

  Logically Collective on VecTagger

  Level: advanced

  Input Parameters:
. tagger - the VecTagger context

  Output Parameters:
. method - the method

.seealso VecTaggerCDFMethod
@*/
PetscErrorCode VecTaggerCDFGetMethod(VecTagger tagger, VecTaggerCDFMethod *method)
{
  VecTagger_CDF  *cuml = (VecTagger_CDF *) tagger->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidPointer(method,2);
  *method = cuml->method;
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerCDFIterativeSetTolerances - Set the tolerances for iterative computation of absolute intervals from CDF intervals.

  Logically Collective on VecTagger

  Input Parameters:
+ tagger - the VecTagger context
. maxit - the maximum number of iterations: 0 indicates the absolute values will be estimated from an initial guess based only on the minimum, maximum, mean, and standard deviation of the intervals.
. rtol - the acceptable relative tolerance in the absolute values from the initial guess
- atol - the acceptable absolute tolerance in the absolute values from the initial guess

  Level: advanced

.seealso: VecTaggerCDFSetMethod()
@*/
PetscErrorCode VecTaggerCDFIterativeSetTolerances(VecTagger tagger, PetscInt maxit, PetscReal rtol, PetscReal atol)
{
  VecTagger_CDF  *cuml = (VecTagger_CDF *) tagger->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidLogicalCollectiveInt(tagger,maxit,2);
  PetscValidLogicalCollectiveReal(tagger,rtol,3);
  PetscValidLogicalCollectiveReal(tagger,atol,4);
  cuml->maxit = maxit;
  cuml->rtol  = rtol;
  cuml->atol  = atol;
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerCDFIterativeGetTolerances - Get the tolerances for iterative computation of absolute intervals from CDF intervals.

  Logically Collective on VecTagger

  Input Parameters:
. tagger - the VecTagger context

  Output Parameters:
+ maxit - the maximum number of iterations: 0 indicates the absolute values will be estimated from an initial guess based only on the minimum, maximum, mean, and standard deviation of the intervals.
. rtol - the acceptable relative tolerance in the absolute values from the initial guess
- atol - the acceptable absolute tolerance in the absolute values from the initial guess

  Level: advanced

.seealso: VecTaggerCDFSetMethod()
@*/
PetscErrorCode VecTaggerCDFIterativeGetTolerances(VecTagger tagger, PetscInt *maxit, PetscReal *rtol, PetscReal *atol)
{
  VecTagger_CDF  *cuml = (VecTagger_CDF *) tagger->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  if (maxit) *maxit = cuml->maxit;
  if (rtol)  *rtol  = cuml->rtol;
  if (atol)  *atol  = cuml->atol;
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerCDFSetInterval - Set the cumulative interval (multi-dimensional box) defining the values to be tagged by the tagger, where cumulative intervals are subsets of [0,1], where 0 indicates the smallest value present in the vector and 1 indicates the largest.

  Logically Collective

  Input Arguments:
+ tagger - the VecTagger context
- interval - the interval: a blocksize list of [min,max] pairs

  Level: advanced

.seealso: VecTaggerCDFGetInterval()
@*/
PetscErrorCode VecTaggerCDFSetInterval(VecTagger tagger,PetscScalar (*interval)[2])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerSetInterval_Simple(tagger,interval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerCDFGetInterval - Get the cumulative interval (multi-dimensional box) defining the values to be tagged by the tagger, where cumulative intervals are subsets of [0,1], where 0 indicates the smallest value present in the vector and 1 indicates the largest.

  Logically Collective

  Input Arguments:
. tagger - the VecTagger context

  Output Arguments:
. interval - the interval: a blocksize list of [min,max] pairs

  Level: advanced

.seealso: VecTaggerCDFSetInterval()
@*/
PetscErrorCode VecTaggerCDFGetInterval(VecTagger tagger,const PetscScalar (**interval)[2])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerGetInterval_Simple(tagger,interval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecTaggerCreate_CDF(VecTagger tagger)
{
  VecTagger_CDF *cuml;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = VecTaggerCreate_Simple(tagger);CHKERRQ(ierr);
  ierr = PetscNewLog(tagger,&cuml);CHKERRQ(ierr);
  ierr = PetscMemcpy(&(cuml->smpl),tagger->data,sizeof(VecTagger_Simple));CHKERRQ(ierr);
  ierr = PetscFree(tagger->data);CHKERRQ(ierr);
  tagger->data = cuml;
  tagger->ops->view             = VecTaggerView_CDF;
  tagger->ops->setfromoptions   = VecTaggerSetFromOptions_CDF;
  tagger->ops->computeintervals = VecTaggerComputeIntervals_CDF;
  PetscFunctionReturn(0);
}
