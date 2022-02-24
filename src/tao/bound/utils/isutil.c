#include <petsctao.h> /*I "petsctao.h" I*/
#include <petsc/private/vecimpl.h>
#include <petsc/private/taoimpl.h>
#include <../src/tao/matrix/submatfree.h>

/*@C
  TaoVecGetSubVec - Gets a subvector using the IS

  Input Parameters:
+ vfull - the full matrix
. is - the index set for the subvector
. reduced_type - the method TAO is using for subsetting (TAO_SUBSET_SUBVEC, TAO_SUBSET_MASK,  TAO_SUBSET_MATRIXFREE)
- maskvalue - the value to set the unused vector elements to (for TAO_SUBSET_MASK or TAO_SUBSET_MATRIXFREE)

  Output Parameter:
. vreduced - the subvector

  Notes:
  maskvalue should usually be 0.0, unless a pointwise divide will be used.

  Level: developer
@*/
PetscErrorCode TaoVecGetSubVec(Vec vfull, IS is, TaoSubsetType reduced_type, PetscReal maskvalue, Vec *vreduced)
{
  PetscInt       nfull,nreduced,nreduced_local,rlow,rhigh,flow,fhigh;
  PetscInt       i,nlocal;
  PetscReal      *fv,*rv;
  const PetscInt *s;
  IS             ident;
  VecType        vtype;
  VecScatter     scatter;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vfull,VEC_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);

  CHKERRQ(VecGetSize(vfull, &nfull));
  CHKERRQ(ISGetSize(is, &nreduced));

  if (nreduced == nfull) {
    CHKERRQ(VecDestroy(vreduced));
    CHKERRQ(VecDuplicate(vfull,vreduced));
    CHKERRQ(VecCopy(vfull,*vreduced));
  } else {
    switch (reduced_type) {
    case TAO_SUBSET_SUBVEC:
      CHKERRQ(VecGetType(vfull,&vtype));
      CHKERRQ(VecGetOwnershipRange(vfull,&flow,&fhigh));
      CHKERRQ(ISGetLocalSize(is,&nreduced_local));
      CHKERRQ(PetscObjectGetComm((PetscObject)vfull,&comm));
      if (*vreduced) {
        CHKERRQ(VecDestroy(vreduced));
      }
      CHKERRQ(VecCreate(comm,vreduced));
      CHKERRQ(VecSetType(*vreduced,vtype));

      CHKERRQ(VecSetSizes(*vreduced,nreduced_local,nreduced));
      CHKERRQ(VecGetOwnershipRange(*vreduced,&rlow,&rhigh));
      CHKERRQ(ISCreateStride(comm,nreduced_local,rlow,1,&ident));
      CHKERRQ(VecScatterCreate(vfull,is,*vreduced,ident,&scatter));
      CHKERRQ(VecScatterBegin(scatter,vfull,*vreduced,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(scatter,vfull,*vreduced,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&scatter));
      CHKERRQ(ISDestroy(&ident));
      break;

    case TAO_SUBSET_MASK:
    case TAO_SUBSET_MATRIXFREE:
      /* vr[i] = vf[i]   if i in is
       vr[i] = 0       otherwise */
      if (!*vreduced) {
        CHKERRQ(VecDuplicate(vfull,vreduced));
      }

      CHKERRQ(VecSet(*vreduced,maskvalue));
      CHKERRQ(ISGetLocalSize(is,&nlocal));
      CHKERRQ(VecGetOwnershipRange(vfull,&flow,&fhigh));
      CHKERRQ(VecGetArray(vfull,&fv));
      CHKERRQ(VecGetArray(*vreduced,&rv));
      CHKERRQ(ISGetIndices(is,&s));
      PetscCheck(nlocal <= (fhigh-flow),PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"IS local size %D > Vec local size %D",nlocal,fhigh-flow);
      for (i=0;i<nlocal;++i) {
        rv[s[i]-flow] = fv[s[i]-flow];
      }
      CHKERRQ(ISRestoreIndices(is,&s));
      CHKERRQ(VecRestoreArray(vfull,&fv));
      CHKERRQ(VecRestoreArray(*vreduced,&rv));
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  TaoMatGetSubMat - Gets a submatrix using the IS

  Input Parameters:
+ M - the full matrix (n x n)
. is - the index set for the submatrix (both row and column index sets need to be the same)
. v1 - work vector of dimension n, needed for TAO_SUBSET_MASK option
- subset_type <TAO_SUBSET_SUBVEC,TAO_SUBSET_MASK,TAO_SUBSET_MATRIXFREE> - the method TAO is using for subsetting

  Output Parameter:
. Msub - the submatrix

  Level: developer
@*/
PetscErrorCode TaoMatGetSubMat(Mat M, IS is, Vec v1, TaoSubsetType subset_type, Mat *Msub)
{
  PetscErrorCode ierr;
  IS             iscomp;
  PetscBool      flg = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(M,MAT_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  CHKERRQ(MatDestroy(Msub));
  switch (subset_type) {
  case TAO_SUBSET_SUBVEC:
    CHKERRQ(MatCreateSubMatrix(M, is, is, MAT_INITIAL_MATRIX, Msub));
    break;

  case TAO_SUBSET_MASK:
    /* Get Reduced Hessian
     Msub[i,j] = M[i,j] if i,j in Free_Local or i==j
     Msub[i,j] = 0      if i!=j and i or j not in Free_Local
     */
    ierr = PetscObjectOptionsBegin((PetscObject)M);CHKERRQ(ierr);
    CHKERRQ(PetscOptionsBool("-overwrite_hessian","modify the existing hessian matrix when computing submatrices","TaoSubsetType",flg,&flg,NULL));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (flg) {
      CHKERRQ(MatDuplicate(M, MAT_COPY_VALUES, Msub));
    } else {
      /* Act on hessian directly (default) */
      CHKERRQ(PetscObjectReference((PetscObject)M));
      *Msub = M;
    }
    /* Save the diagonal to temporary vector */
    CHKERRQ(MatGetDiagonal(*Msub,v1));

    /* Zero out rows and columns */
    CHKERRQ(ISComplementVec(is,v1,&iscomp));

    /* Use v1 instead of 0 here because of PETSc bug */
    CHKERRQ(MatZeroRowsColumnsIS(*Msub,iscomp,1.0,v1,v1));

    CHKERRQ(ISDestroy(&iscomp));
    break;
  case TAO_SUBSET_MATRIXFREE:
    CHKERRQ(ISComplementVec(is,v1,&iscomp));
    CHKERRQ(MatCreateSubMatrixFree(M,iscomp,iscomp,Msub));
    CHKERRQ(ISDestroy(&iscomp));
    break;
  }
  PetscFunctionReturn(0);
}

/*@C
  TaoEstimateActiveBounds - Generates index sets for variables at the lower and upper
  bounds, as well as fixed variables where lower and upper bounds equal each other.

  Input Parameters:
+ X - solution vector
. XL - lower bound vector
. XU - upper bound vector
. G - unprojected gradient
. S - step direction with which the active bounds will be estimated
. W - work vector of type and size of X
- steplen - the step length at which the active bounds will be estimated (needs to be conservative)

  Output Parameters:
+ bound_tol - tolerance for for the bound estimation
. active_lower - index set for active variables at the lower bound
. active_upper - index set for active variables at the upper bound
. active_fixed - index set for fixed variables
. active - index set for all active variables
- inactive - complementary index set for inactive variables

  Notes:
  This estimation is based on Bertsekas' method, with a built in diagonal scaling value of 1.0e-3.

  Level: developer
@*/
PetscErrorCode TaoEstimateActiveBounds(Vec X, Vec XL, Vec XU, Vec G, Vec S, Vec W, PetscReal steplen, PetscReal *bound_tol,
                                       IS *active_lower, IS *active_upper, IS *active_fixed, IS *active, IS *inactive)
{
  PetscReal                    wnorm;
  PetscReal                    zero = PetscPowReal(PETSC_MACHINE_EPSILON, 2.0/3.0);
  PetscInt                     i, n_isl=0, n_isu=0, n_isf=0, n_isa=0, n_isi=0;
  PetscInt                     N_isl, N_isu, N_isf, N_isa, N_isi;
  PetscInt                     n, low, high, nDiff;
  PetscInt                     *isl=NULL, *isu=NULL, *isf=NULL, *isa=NULL, *isi=NULL;
  const PetscScalar            *xl, *xu, *x, *g;
  MPI_Comm                     comm = PetscObjectComm((PetscObject)X);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,VEC_CLASSID,1);
  PetscValidHeaderSpecific(XL,VEC_CLASSID,2);
  PetscValidHeaderSpecific(XU,VEC_CLASSID,3);
  PetscValidHeaderSpecific(G,VEC_CLASSID,4);
  PetscValidHeaderSpecific(S,VEC_CLASSID,5);
  PetscValidHeaderSpecific(W,VEC_CLASSID,6);

  PetscValidType(X,1);
  PetscValidType(XL,2);
  PetscValidType(XU,3);
  PetscValidType(G,4);
  PetscValidType(S,5);
  PetscValidType(W,6);
  PetscCheckSameType(X,1,XL,2);
  PetscCheckSameType(X,1,XU,3);
  PetscCheckSameType(X,1,G,4);
  PetscCheckSameType(X,1,S,5);
  PetscCheckSameType(X,1,W,6);
  PetscCheckSameComm(X,1,XL,2);
  PetscCheckSameComm(X,1,XU,3);
  PetscCheckSameComm(X,1,G,4);
  PetscCheckSameComm(X,1,S,5);
  PetscCheckSameComm(X,1,W,6);
  VecCheckSameSize(X,1,XL,2);
  VecCheckSameSize(X,1,XU,3);
  VecCheckSameSize(X,1,G,4);
  VecCheckSameSize(X,1,S,5);
  VecCheckSameSize(X,1,W,6);

  /* Update the tolerance for bound detection (this is based on Bertsekas' method) */
  CHKERRQ(VecCopy(X, W));
  CHKERRQ(VecAXPBY(W, steplen, 1.0, S));
  CHKERRQ(TaoBoundSolution(W, XL, XU, 0.0, &nDiff, W));
  CHKERRQ(VecAXPBY(W, 1.0, -1.0, X));
  CHKERRQ(VecNorm(W, NORM_2, &wnorm));
  *bound_tol = PetscMin(*bound_tol, wnorm);

  CHKERRQ(VecGetOwnershipRange(X, &low, &high));
  CHKERRQ(VecGetLocalSize(X, &n));
  if (n>0) {
    CHKERRQ(VecGetArrayRead(X, &x));
    CHKERRQ(VecGetArrayRead(XL, &xl));
    CHKERRQ(VecGetArrayRead(XU, &xu));
    CHKERRQ(VecGetArrayRead(G, &g));

    /* Loop over variables and categorize the indexes */
    CHKERRQ(PetscMalloc1(n, &isl));
    CHKERRQ(PetscMalloc1(n, &isu));
    CHKERRQ(PetscMalloc1(n, &isf));
    CHKERRQ(PetscMalloc1(n, &isa));
    CHKERRQ(PetscMalloc1(n, &isi));
    for (i=0; i<n; ++i) {
      if (xl[i] == xu[i]) {
        /* Fixed variables */
        isf[n_isf]=low+i; ++n_isf;
        isa[n_isa]=low+i; ++n_isa;
      } else if ((xl[i] > PETSC_NINFINITY) && (x[i] <= xl[i] + *bound_tol) && (g[i] > zero)) {
        /* Lower bounded variables */
        isl[n_isl]=low+i; ++n_isl;
        isa[n_isa]=low+i; ++n_isa;
      } else if ((xu[i] < PETSC_INFINITY) && (x[i] >= xu[i] - *bound_tol) && (g[i] < zero)) {
        /* Upper bounded variables */
        isu[n_isu]=low+i; ++n_isu;
        isa[n_isa]=low+i; ++n_isa;
      } else {
        /* Inactive variables */
        isi[n_isi]=low+i; ++n_isi;
      }
    }

    CHKERRQ(VecRestoreArrayRead(X, &x));
    CHKERRQ(VecRestoreArrayRead(XL, &xl));
    CHKERRQ(VecRestoreArrayRead(XU, &xu));
    CHKERRQ(VecRestoreArrayRead(G, &g));
  }

  /* Clear all index sets */
  CHKERRQ(ISDestroy(active_lower));
  CHKERRQ(ISDestroy(active_upper));
  CHKERRQ(ISDestroy(active_fixed));
  CHKERRQ(ISDestroy(active));
  CHKERRQ(ISDestroy(inactive));

  /* Collect global sizes */
  CHKERRMPI(MPIU_Allreduce(&n_isl, &N_isl, 1, MPIU_INT, MPI_SUM, comm));
  CHKERRMPI(MPIU_Allreduce(&n_isu, &N_isu, 1, MPIU_INT, MPI_SUM, comm));
  CHKERRMPI(MPIU_Allreduce(&n_isf, &N_isf, 1, MPIU_INT, MPI_SUM, comm));
  CHKERRMPI(MPIU_Allreduce(&n_isa, &N_isa, 1, MPIU_INT, MPI_SUM, comm));
  CHKERRMPI(MPIU_Allreduce(&n_isi, &N_isi, 1, MPIU_INT, MPI_SUM, comm));

  /* Create index set for lower bounded variables */
  if (N_isl > 0) {
    CHKERRQ(ISCreateGeneral(comm, n_isl, isl, PETSC_OWN_POINTER, active_lower));
  } else {
    CHKERRQ(PetscFree(isl));
  }
  /* Create index set for upper bounded variables */
  if (N_isu > 0) {
    CHKERRQ(ISCreateGeneral(comm, n_isu, isu, PETSC_OWN_POINTER, active_upper));
  } else {
    CHKERRQ(PetscFree(isu));
  }
  /* Create index set for fixed variables */
  if (N_isf > 0) {
    CHKERRQ(ISCreateGeneral(comm, n_isf, isf, PETSC_OWN_POINTER, active_fixed));
  } else {
    CHKERRQ(PetscFree(isf));
  }
  /* Create index set for all actively bounded variables */
  if (N_isa > 0) {
    CHKERRQ(ISCreateGeneral(comm, n_isa, isa, PETSC_OWN_POINTER, active));
  } else {
    CHKERRQ(PetscFree(isa));
  }
  /* Create index set for all inactive variables */
  if (N_isi > 0) {
    CHKERRQ(ISCreateGeneral(comm, n_isi, isi, PETSC_OWN_POINTER, inactive));
  } else {
    CHKERRQ(PetscFree(isi));
  }

  /* Clean up and exit */
  PetscFunctionReturn(0);
}

/*@C
  TaoBoundStep - Ensures the correct zero or adjusted step direction
  values for active variables.

  Input Parameters:
+ X - solution vector
. XL - lower bound vector
. XU - upper bound vector
. active_lower - index set for lower bounded active variables
. active_upper - index set for lower bounded active variables
. active_fixed - index set for fixed active variables
- scale - amplification factor for the step that needs to be taken on actively bounded variables

  Output Parameter:
. S - step direction to be modified

  Level: developer
@*/
PetscErrorCode TaoBoundStep(Vec X, Vec XL, Vec XU, IS active_lower, IS active_upper, IS active_fixed, PetscReal scale, Vec S)
{

  Vec                          step_lower, step_upper, step_fixed;
  Vec                          x_lower, x_upper;
  Vec                          bound_lower, bound_upper;

  PetscFunctionBegin;
  /* Adjust step for variables at the estimated lower bound */
  if (active_lower) {
    CHKERRQ(VecGetSubVector(S, active_lower, &step_lower));
    CHKERRQ(VecGetSubVector(X, active_lower, &x_lower));
    CHKERRQ(VecGetSubVector(XL, active_lower, &bound_lower));
    CHKERRQ(VecCopy(bound_lower, step_lower));
    CHKERRQ(VecAXPY(step_lower, -1.0, x_lower));
    CHKERRQ(VecScale(step_lower, scale));
    CHKERRQ(VecRestoreSubVector(S, active_lower, &step_lower));
    CHKERRQ(VecRestoreSubVector(X, active_lower, &x_lower));
    CHKERRQ(VecRestoreSubVector(XL, active_lower, &bound_lower));
  }

  /* Adjust step for the variables at the estimated upper bound */
  if (active_upper) {
    CHKERRQ(VecGetSubVector(S, active_upper, &step_upper));
    CHKERRQ(VecGetSubVector(X, active_upper, &x_upper));
    CHKERRQ(VecGetSubVector(XU, active_upper, &bound_upper));
    CHKERRQ(VecCopy(bound_upper, step_upper));
    CHKERRQ(VecAXPY(step_upper, -1.0, x_upper));
    CHKERRQ(VecScale(step_upper, scale));
    CHKERRQ(VecRestoreSubVector(S, active_upper, &step_upper));
    CHKERRQ(VecRestoreSubVector(X, active_upper, &x_upper));
    CHKERRQ(VecRestoreSubVector(XU, active_upper, &bound_upper));
  }

  /* Zero out step for fixed variables */
  if (active_fixed) {
    CHKERRQ(VecGetSubVector(S, active_fixed, &step_fixed));
    CHKERRQ(VecSet(step_fixed, 0.0));
    CHKERRQ(VecRestoreSubVector(S, active_fixed, &step_fixed));
  }
  PetscFunctionReturn(0);
}

/*@C
  TaoBoundSolution - Ensures that the solution vector is snapped into the bounds within a given tolerance.

  Collective on Vec

  Input Parameters:
+ X - solution vector
. XL - lower bound vector
. XU - upper bound vector
- bound_tol - absolute tolerance in enforcing the bound

  Output Parameters:
+ nDiff - total number of vector entries that have been bounded
- Xout - modified solution vector satisfying bounds to bound_tol

  Level: developer

.seealso: TAOBNCG, TAOBNTL, TAOBNTR
@*/
PetscErrorCode TaoBoundSolution(Vec X, Vec XL, Vec XU, PetscReal bound_tol, PetscInt *nDiff, Vec Xout)
{
  PetscInt          i,n,low,high,nDiff_loc=0;
  PetscScalar       *xout;
  const PetscScalar *x,*xl,*xu;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,VEC_CLASSID,1);
  PetscValidHeaderSpecific(XL,VEC_CLASSID,2);
  PetscValidHeaderSpecific(XU,VEC_CLASSID,3);
  PetscValidHeaderSpecific(Xout,VEC_CLASSID,6);

  PetscValidType(X,1);
  PetscValidType(XL,2);
  PetscValidType(XU,3);
  PetscValidType(Xout,6);
  PetscCheckSameType(X,1,XL,2);
  PetscCheckSameType(X,1,XU,3);
  PetscCheckSameType(X,1,Xout,6);
  PetscCheckSameComm(X,1,XL,2);
  PetscCheckSameComm(X,1,XU,3);
  PetscCheckSameComm(X,1,Xout,6);
  VecCheckSameSize(X,1,XL,2);
  VecCheckSameSize(X,1,XU,3);
  VecCheckSameSize(X,1,Xout,4);

  CHKERRQ(VecGetOwnershipRange(X,&low,&high));
  CHKERRQ(VecGetLocalSize(X,&n));
  if (n>0) {
    CHKERRQ(VecGetArrayRead(X, &x));
    CHKERRQ(VecGetArrayRead(XL, &xl));
    CHKERRQ(VecGetArrayRead(XU, &xu));
    CHKERRQ(VecGetArray(Xout, &xout));

    for (i=0;i<n;++i) {
      if ((xl[i] > PETSC_NINFINITY) && (x[i] <= xl[i] + bound_tol)) {
        xout[i] = xl[i]; ++nDiff_loc;
      } else if ((xu[i] < PETSC_INFINITY) && (x[i] >= xu[i] - bound_tol)) {
        xout[i] = xu[i]; ++nDiff_loc;
      }
    }

    CHKERRQ(VecRestoreArrayRead(X, &x));
    CHKERRQ(VecRestoreArrayRead(XL, &xl));
    CHKERRQ(VecRestoreArrayRead(XU, &xu));
    CHKERRQ(VecRestoreArray(Xout, &xout));
  }
  CHKERRMPI(MPIU_Allreduce(&nDiff_loc, nDiff, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)X)));
  PetscFunctionReturn(0);
}
