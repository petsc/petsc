
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

static PetscErrorCode VecTaggerComputeIntervals_CDF_Serial(VecTagger tagger,Vec vec,PetscInt bs,PetscScalar (*intervals)[2])
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;
  Vec              vComp;
  PetscInt         n, m;
  PetscInt         i;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
  m = n/bs;
  ierr = VecCreateSeq(PETSC_COMM_SELF,m,&vComp);CHKERRQ(ierr);
  for (i = 0; i < bs; i++) {
    IS          isStride;
    VecScatter  vScat;
    PetscScalar *cArray;
    PetscInt    minInd, maxInd;
    PetscScalar minCDF, maxCDF;

    ierr = ISCreateStride(PETSC_COMM_SELF,m,i,bs,&isStride);CHKERRQ(ierr);
    ierr = VecScatterCreate(vec,isStride,vComp,NULL,&vScat);CHKERRQ(ierr);
    ierr = VecScatterBegin(vScat,vec,vComp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(vScat,vec,vComp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&vScat);CHKERRQ(ierr);
    ierr = ISDestroy(&isStride);CHKERRQ(ierr);

    ierr = VecGetArray(vComp,&cArray);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = PetscSortReal(m,cArray);CHKERRQ(ierr);
    minCDF = PetscMax(0., smpl->interval[i][0]);
    maxCDF = PetscMin(1., smpl->interval[i][1]);
    minInd = (PetscInt) (minCDF * m);
    maxInd = (PetscInt) (maxCDF * m);
    intervals[i][0] = cArray[PetscMin(minInd,m-1)];
    intervals[i][1] = cArray[PetscMin(maxInd,m-1)];
#else
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Need to implement complex sorting");
#endif
    ierr = VecRestoreArray(vComp,&cArray);CHKERRQ(ierr);
  }
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
    PetscFunctionReturn(0);
  }
  switch (cuml->method) {
  case VECTAGGER_CDF_GATHER:
    ierr = VecTaggerComputeIntervals_CDF_Gather(tagger,vec,bs,ints);CHKERRQ(ierr);
    break;
  case VECTAGGER_CDF_ITERATIVE:
    SETERRQ(PetscObjectComm((PetscObject)tagger),PETSC_ERR_SUP,"In place CDF calculation/estimation not implemented yet.");
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
      ierr = PetscViewerASCIIPrintf(viewer,"max its: %D, abs tol: %g, rel tol %g",cuml->maxit,(double) cuml->atol,(double) cuml->rtol);CHKERRQ(ierr);
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
