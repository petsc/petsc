
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include "../src/vec/vec/utils/tagger/impls/simple.h"

const char *const VecTaggerCDFMethods[VECTAGGER_CDF_NUM_METHODS] = {"gather","iterative"};

#if !defined (PETSC_USE_COMPLEX)
typedef VecTaggerBox VecTaggerBoxReal;
#else
typedef struct {
  PetscReal min;
  PetscReal max;
} VecTaggerBoxReal;
#endif

typedef struct {
  VecTagger_Simple   smpl;
  PetscReal          atol;
  PetscReal          rtol;
  PetscInt           maxit;
  PetscInt           numMoments;
  VecTaggerCDFMethod method;
} VecTagger_CDF;

static PetscErrorCode VecTaggerComputeBox_CDF_SortedArray(const PetscReal *cArray, PetscInt m, const VecTaggerBoxReal *bxs, VecTaggerBoxReal *boxes)
{
  PetscInt    minInd, maxInd;
  PetscReal   minCDF, maxCDF;

  PetscFunctionBegin;
  minCDF = PetscMax(0., bxs->min);
  maxCDF = PetscMin(1., bxs->max);
  minInd = (PetscInt) (minCDF * m);
  maxInd = (PetscInt) (maxCDF * m);
  boxes->min = cArray[PetscMin(minInd,m-1)];
  boxes->max = cArray[PetscMax(minInd,maxInd-1)];
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeBoxes_CDF_Serial(VecTagger tagger,Vec vec,PetscInt bs,VecTaggerBox *boxes)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;
  Vec              vComp;
  PetscInt         n, m;
  PetscInt         i;
#if defined (PETSC_USE_COMPLEX)
  PetscReal        *cReal, *cImag;
#endif

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(vec,&n));
  m    = n/bs;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,m,&vComp));
#if defined (PETSC_USE_COMPLEX)
  PetscCall(PetscMalloc2(m,&cReal,m,&cImag));
#endif
  for (i = 0; i < bs; i++) {
    IS          isStride;
    VecScatter  vScat;
    PetscScalar *cArray;

    PetscCall(ISCreateStride(PETSC_COMM_SELF,m,i,bs,&isStride));
    PetscCall(VecScatterCreate(vec,isStride,vComp,NULL,&vScat));
    PetscCall(VecScatterBegin(vScat,vec,vComp,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(vScat,vec,vComp,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterDestroy(&vScat));
    PetscCall(ISDestroy(&isStride));

    PetscCall(VecGetArray(vComp,&cArray));
#if !defined(PETSC_USE_COMPLEX)
    PetscCall(PetscSortReal(m,cArray));
    PetscCall(VecTaggerComputeBox_CDF_SortedArray(cArray,m,&smpl->box[i],&boxes[i]));
#else
    {
      PetscInt         j;
      VecTaggerBoxReal realBxs, imagBxs;
      VecTaggerBoxReal realBoxes, imagBoxes;

      for (j = 0; j < m; j++) {
        cReal[j] = PetscRealPart(cArray[j]);
        cImag[j] = PetscImaginaryPart(cArray[j]);
      }
      PetscCall(PetscSortReal(m,cReal));
      PetscCall(PetscSortReal(m,cImag));

      realBxs.min = PetscRealPart(smpl->box[i].min);
      realBxs.max = PetscRealPart(smpl->box[i].max);
      imagBxs.min = PetscImaginaryPart(smpl->box[i].min);
      imagBxs.max = PetscImaginaryPart(smpl->box[i].max);
      PetscCall(VecTaggerComputeBox_CDF_SortedArray(cReal,m,&realBxs,&realBoxes));
      PetscCall(VecTaggerComputeBox_CDF_SortedArray(cImag,m,&imagBxs,&imagBoxes));
      boxes[i].min = PetscCMPLX(realBoxes.min,imagBoxes.min);
      boxes[i].max = PetscCMPLX(realBoxes.max,imagBoxes.max);
    }
#endif
    PetscCall(VecRestoreArray(vComp,&cArray));
  }
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscFree2(cReal,cImag));
#endif
  PetscCall(VecDestroy(&vComp));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeBoxes_CDF_Gather(VecTagger tagger,Vec vec,PetscInt bs,VecTaggerBox *boxes)
{
  Vec            gVec = NULL;
  VecScatter     vScat;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscCall(VecScatterCreateToZero(vec,&vScat,&gVec));
  PetscCall(VecScatterBegin(vScat,vec,gVec,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(vScat,vec,gVec,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&vScat));
  PetscCallMPI(MPI_Comm_rank (PetscObjectComm((PetscObject)vec),&rank));
  if (rank == 0) {
    PetscCall(VecTaggerComputeBoxes_CDF_Serial(tagger,gVec,bs,boxes));
  }
  PetscCallMPI(MPI_Bcast((PetscScalar *)boxes,2*bs,MPIU_SCALAR,0,PetscObjectComm((PetscObject)vec)));
  PetscCall(VecDestroy(&gVec));
  PetscFunctionReturn(0);
}

typedef struct _n_CDFStats
{
  PetscReal min;
  PetscReal max;
  PetscReal moment[3];
} CDFStats;

static void MPIAPI VecTaggerCDFStatsReduce(void *a, void *b, int * len, MPI_Datatype *datatype)
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

static PetscErrorCode VecTaggerComputeBox_CDF_SortedArray_Iterative(VecTagger tagger, MPI_Datatype statType, MPI_Op statReduce, const PetscReal *cArray, PetscInt m, const VecTaggerBoxReal *cdfBox, VecTaggerBoxReal *absBox)
{
  MPI_Comm       comm;
  VecTagger_CDF  *cdf;
  PetscInt       maxit, i, j, k, l, M;
  PetscInt       bounds[2][2];
  PetscInt       offsets[2];
  PetscReal      intervalLen = cdfBox->max - cdfBox->min;
  PetscReal      rtol, atol;

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
  PetscCall(VecTaggerComputeBox_CDF_SortedArray(cArray,m,cdfBox,absBox)); /* compute a local estimate of the interval */
  {
    CDFStats stats[3];

    for (i = 0; i < 2; i++) { /* compute statistics of those local estimates */
      PetscReal val = i ? absBox->max : absBox->min;

      stats[i].min = m ? val : PETSC_MAX_REAL;
      stats[i].max = m ? val : PETSC_MIN_REAL;
      stats[i].moment[0] = m;
      stats[i].moment[1] = m * val;
      stats[i].moment[2] = m * val * val;
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
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE,stats,3,statType,statReduce,comm));
    M    = (PetscInt) stats[2].moment[0];
    /* use those initial statistics to get the initial (globally agreed-upon) choices for the absolute box bounds */
    for (i = 0; i < 2; i++) {
      PetscCall(CDFUtilInverseEstimate(&stats[i],i ? cdfBox->max : cdfBox->min,(i ? &absBox->max : &absBox->min)));
    }
  }
  /* refine the estimates by computing how close they come to the desired box and refining */
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
          section = (val <  absBox->min) ? 0 : 1;
        } else {
          section = (val <= absBox->max) ? 0 : 1;
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
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, stats, 4, statType, statReduce, comm));
    for (i = 0; i < 2; i++) {
      PetscInt  totalLessThan = offsets[i] + stats[i][0].moment[0];
      PetscReal cdfOfAbs      = (PetscReal) totalLessThan / (PetscReal) M;
      PetscReal diff;
      PetscInt  section;

      if (cdfOfAbs == (i ? cdfBox->max : cdfBox->min)) {
        offsets[i] = totalLessThan;
        bounds[i][0] = bounds[i][1] = 0;
        continue;
      }
      if (cdfOfAbs > (i ? cdfBox->max : cdfBox->min)) { /* the correct absolute value lies in the lower section */
        section = 0;
      } else {
        section = 1;
        offsets[i] = totalLessThan;
      }
      for (j = 0; j < 2; j++) {
        bounds[i][j] = newBounds[i][section][j];
      }
      PetscCall(CDFUtilInverseEstimate(&stats[i][section],((i ? cdfBox->max : cdfBox->min) - ((PetscReal) offsets[i] / (PetscReal) M))/stats[i][section].moment[0],(i ? &absBox->max : &absBox->min)));
      diff = PetscAbs(cdfOfAbs - (i ? cdfBox->max : cdfBox->min));
      maxDiff = PetscMax(maxDiff,diff);
    }
    if (!maxDiff) PetscFunctionReturn(0);
    if ((atol || rtol) && ((!atol) || (maxDiff <= atol)) && ((!rtol) || (maxDiff <= rtol * intervalLen))) {
      break;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeBoxes_CDF_Iterative(VecTagger tagger,Vec vec,PetscInt bs,VecTaggerBox *boxes)
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

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)vec);
  PetscCall(VecGetSize(vec,&N));
  PetscCall(VecGetLocalSize(vec,&n));
  M    = N/bs;
  m    = n/bs;
  PetscCall(VecCreateMPI(comm,m,M,&vComp));
  PetscCall(VecSetUp(vComp));
  PetscCall(VecGetOwnershipRange(vComp,&rstart,NULL));
#if defined (PETSC_USE_COMPLEX)
  PetscCall(PetscMalloc2(m,&cReal,m,&cImag));
#endif
  PetscCallMPI(MPI_Type_contiguous(5,MPIU_REAL,&statType));
  PetscCallMPI(MPI_Type_commit(&statType));
  PetscCallMPI(MPI_Op_create(VecTaggerCDFStatsReduce,1,&statReduce));
  for (i = 0; i < bs; i++) {
    IS          isStride;
    VecScatter  vScat;
    PetscScalar *cArray;

    PetscCall(ISCreateStride(comm,m,bs * rstart + i,bs,&isStride));
    PetscCall(VecScatterCreate(vec,isStride,vComp,NULL,&vScat));
    PetscCall(VecScatterBegin(vScat,vec,vComp,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(vScat,vec,vComp,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterDestroy(&vScat));
    PetscCall(ISDestroy(&isStride));

    PetscCall(VecGetArray(vComp,&cArray));
#if !defined(PETSC_USE_COMPLEX)
    PetscCall(PetscSortReal(m,cArray));
    PetscCall(VecTaggerComputeBox_CDF_SortedArray_Iterative(tagger,statType,statReduce,cArray,m,&smpl->box[i],&boxes[i]));
#else
    {
      PetscInt         j;
      VecTaggerBoxReal realBxs, imagBxs;
      VecTaggerBoxReal realBoxes, imagBoxes;

      for (j = 0; j < m; j++) {
        cReal[j] = PetscRealPart(cArray[j]);
        cImag[j] = PetscImaginaryPart(cArray[j]);
      }
      PetscCall(PetscSortReal(m,cReal));
      PetscCall(PetscSortReal(m,cImag));

      realBxs.min = PetscRealPart(smpl->box[i].min);
      realBxs.max = PetscRealPart(smpl->box[i].max);
      imagBxs.min = PetscImaginaryPart(smpl->box[i].min);
      imagBxs.max = PetscImaginaryPart(smpl->box[i].max);
      PetscCall(VecTaggerComputeBox_CDF_SortedArray_Iterative(tagger,statType,statReduce,cReal,m,&realBxs,&realBoxes));
      PetscCall(VecTaggerComputeBox_CDF_SortedArray_Iterative(tagger,statType,statReduce,cImag,m,&imagBxs,&imagBoxes));
      boxes[i].min = PetscCMPLX(realBoxes.min,imagBoxes.min);
      boxes[i].max = PetscCMPLX(realBoxes.max,imagBoxes.max);
    }
#endif
    PetscCall(VecRestoreArray(vComp,&cArray));
  }
  PetscCallMPI(MPI_Op_free(&statReduce));
  PetscCallMPI(MPI_Type_free(&statType));
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscFree2(cReal,cImag));
#endif
  PetscCall(VecDestroy(&vComp));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeBoxes_CDF(VecTagger tagger,Vec vec,PetscInt *numBoxes,VecTaggerBox **boxes,PetscBool *listed)
{
  VecTagger_CDF  *cuml = (VecTagger_CDF *)tagger->data;
  PetscMPIInt    size;
  PetscInt       bs;
  VecTaggerBox   *bxs;

  PetscFunctionBegin;
  PetscCall(VecTaggerGetBlockSize(tagger,&bs));
  *numBoxes = 1;
  PetscCall(PetscMalloc1(bs,&bxs));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)tagger),&size));
  if (size == 1) {
    PetscCall(VecTaggerComputeBoxes_CDF_Serial(tagger,vec,bs,bxs));
    *boxes = bxs;
    PetscFunctionReturn(0);
  }
  switch (cuml->method) {
  case VECTAGGER_CDF_GATHER:
    PetscCall(VecTaggerComputeBoxes_CDF_Gather(tagger,vec,bs,bxs));
    break;
  case VECTAGGER_CDF_ITERATIVE:
    PetscCall(VecTaggerComputeBoxes_CDF_Iterative(tagger,vec,bs,bxs));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tagger),PETSC_ERR_SUP,"Unknown CDF calculation/estimation method.");
  }
  *boxes = bxs;
  if (listed) *listed = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerView_CDF(VecTagger tagger,PetscViewer viewer)
{
  VecTagger_CDF  *cuml = (VecTagger_CDF *) tagger->data;
  PetscBool      iascii;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCall(VecTaggerView_Simple(tagger,viewer));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)tagger),&size));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (size > 1 && iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"CDF computation method: %s\n",VecTaggerCDFMethods[cuml->method]));
    if (cuml->method == VECTAGGER_CDF_ITERATIVE) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer,"max its: %" PetscInt_FMT ", abs tol: %g, rel tol %g\n",cuml->maxit,(double) cuml->atol,(double) cuml->rtol));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerSetFromOptions_CDF(PetscOptionItems *PetscOptionsObject,VecTagger tagger)
{
  VecTagger_CDF *cuml = (VecTagger_CDF *) tagger->data;
  PetscInt       method;
  PetscBool      set;

  PetscFunctionBegin;
  PetscCall(VecTaggerSetFromOptions_Simple(PetscOptionsObject,tagger));
  PetscOptionsHeadBegin(PetscOptionsObject,"VecTagger options for CDF boxes");
  PetscCall(PetscOptionsEList("-vec_tagger_cdf_method","Method for computing absolute boxes from CDF boxes","VecTaggerCDFSetMethod()",VecTaggerCDFMethods,VECTAGGER_CDF_NUM_METHODS,VecTaggerCDFMethods[cuml->method],&method,&set));
  if (set) cuml->method = (VecTaggerCDFMethod) method;
  PetscCall(PetscOptionsInt("-vec_tagger_cdf_max_it","Maximum iterations for iterative computation of absolute boxes from CDF boxes","VecTaggerCDFIterativeSetTolerances()",cuml->maxit,&cuml->maxit,NULL));
  PetscCall(PetscOptionsReal("-vec_tagger_cdf_rtol","Maximum relative tolerance for iterative computation of absolute boxes from CDF boxes","VecTaggerCDFIterativeSetTolerances()",cuml->rtol,&cuml->rtol,NULL));
  PetscCall(PetscOptionsReal("-vec_tagger_cdf_atol","Maximum absolute tolerance for iterative computation of absolute boxes from CDF boxes","VecTaggerCDFIterativeSetTolerances()",cuml->atol,&cuml->atol,NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerCDFSetMethod - Set the method used to compute absolute boxes from CDF boxes

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
  VecTaggerCDFGetMethod - Get the method used to compute absolute boxes from CDF boxes

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
  VecTaggerCDFIterativeSetTolerances - Set the tolerances for iterative computation of absolute boxes from CDF boxes.

  Logically Collective on VecTagger

  Input Parameters:
+ tagger - the VecTagger context
. maxit - the maximum number of iterations: 0 indicates the absolute values will be estimated from an initial guess based only on the minimum, maximum, mean, and standard deviation of the box endpoints.
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
  VecTaggerCDFIterativeGetTolerances - Get the tolerances for iterative computation of absolute boxes from CDF boxes.

  Logically Collective on VecTagger

  Input Parameter:
. tagger - the VecTagger context

  Output Parameters:
+ maxit - the maximum number of iterations: 0 indicates the absolute values will be estimated from an initial guess based only on the minimum, maximum, mean, and standard deviation of the box endpoints.
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
  VecTaggerCDFSetBox - Set the cumulative box defining the values to be tagged by the tagger, where cumulative boxes are subsets of [0,1], where 0 indicates the smallest value present in the vector and 1 indicates the largest.

  Logically Collective

  Input Parameters:
+ tagger - the VecTagger context
- boxes - a blocksize array of VecTaggerBox boxes

  Level: advanced

.seealso: VecTaggerCDFGetBox()
@*/
PetscErrorCode VecTaggerCDFSetBox(VecTagger tagger,VecTaggerBox *box)
{
  PetscFunctionBegin;
  PetscCall(VecTaggerSetBox_Simple(tagger,box));
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerCDFGetBox - Get the cumulative box (multi-dimensional box) defining the values to be tagged by the tagger, where cumulative boxes are subsets of [0,1], where 0 indicates the smallest value present in the vector and 1 indicates the largest.

  Logically Collective

  Input Parameter:
. tagger - the VecTagger context

  Output Parameter:
. boxes - a blocksize array of VecTaggerBox boxes

  Level: advanced

.seealso: VecTaggerCDFSetBox()
@*/
PetscErrorCode VecTaggerCDFGetBox(VecTagger tagger,const VecTaggerBox **box)
{
  PetscFunctionBegin;
  PetscCall(VecTaggerGetBox_Simple(tagger,box));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode VecTaggerCreate_CDF(VecTagger tagger)
{
  VecTagger_CDF  *cuml;

  PetscFunctionBegin;
  PetscCall(VecTaggerCreate_Simple(tagger));
  PetscCall(PetscNewLog(tagger,&cuml));
  PetscCall(PetscMemcpy(&cuml->smpl,tagger->data,sizeof(VecTagger_Simple)));
  PetscCall(PetscFree(tagger->data));
  tagger->data = cuml;
  tagger->ops->view           = VecTaggerView_CDF;
  tagger->ops->setfromoptions = VecTaggerSetFromOptions_CDF;
  tagger->ops->computeboxes   = VecTaggerComputeBoxes_CDF;
  PetscFunctionReturn(0);
}
