#include <petsctao.h> /*I "petsctao.h" I*/
#include <petsc/private/taoimpl.h>
#include <../src/tao/matrix/submatfree.h>

/*@C
  TaoVecGetSubVec - Gets a subvector using the IS

  Input Parameters:
+ vfull - the full matrix
. is - the index set for the subvector
. reduced_type - the method TAO is using for subsetting (TAO_SUBSET_SUBVEC, TAO_SUBSET_MASK,  TAO_SUBSET_MATRIXFREE)
- maskvalue - the value to set the unused vector elements to (for TAO_SUBSET_MASK or TAO_SUBSET_MATRIXFREE)

  Output Parameters:
. vreduced - the subvector

  Notes:
  maskvalue should usually be 0.0, unless a pointwise divide will be used.

@*/
PetscErrorCode TaoVecGetSubVec(Vec vfull, IS is, TaoSubsetType reduced_type, PetscReal maskvalue, Vec *vreduced)
{
  PetscErrorCode ierr;
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

  ierr = VecGetSize(vfull, &nfull);CHKERRQ(ierr);
  ierr = ISGetSize(is, &nreduced);CHKERRQ(ierr);

  if (nreduced == nfull) {
    ierr = VecDestroy(vreduced);CHKERRQ(ierr);
    ierr = VecDuplicate(vfull,vreduced);CHKERRQ(ierr);
    ierr = VecCopy(vfull,*vreduced);CHKERRQ(ierr);
  } else {
    switch (reduced_type) {
    case TAO_SUBSET_SUBVEC:
      ierr = VecGetType(vfull,&vtype);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(vfull,&flow,&fhigh);CHKERRQ(ierr);
      ierr = ISGetLocalSize(is,&nreduced_local);CHKERRQ(ierr);
      ierr = PetscObjectGetComm((PetscObject)vfull,&comm);CHKERRQ(ierr);
      if (*vreduced) {
        ierr = VecDestroy(vreduced);CHKERRQ(ierr);
      }
      ierr = VecCreate(comm,vreduced);CHKERRQ(ierr);
      ierr = VecSetType(*vreduced,vtype);CHKERRQ(ierr);

      ierr = VecSetSizes(*vreduced,nreduced_local,nreduced);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(*vreduced,&rlow,&rhigh);CHKERRQ(ierr);
      ierr = ISCreateStride(comm,nreduced_local,rlow,1,&ident);CHKERRQ(ierr);
      ierr = VecScatterCreate(vfull,is,*vreduced,ident,&scatter);CHKERRQ(ierr);
      ierr = VecScatterBegin(scatter,vfull,*vreduced,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(scatter,vfull,*vreduced,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
      ierr = ISDestroy(&ident);CHKERRQ(ierr);
      break;

    case TAO_SUBSET_MASK:
    case TAO_SUBSET_MATRIXFREE:
      /* vr[i] = vf[i]   if i in is
       vr[i] = 0       otherwise */
      if (!*vreduced) {
        ierr = VecDuplicate(vfull,vreduced);CHKERRQ(ierr);
      }

      ierr = VecSet(*vreduced,maskvalue);CHKERRQ(ierr);
      ierr = ISGetLocalSize(is,&nlocal);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(vfull,&flow,&fhigh);CHKERRQ(ierr);
      ierr = VecGetArray(vfull,&fv);CHKERRQ(ierr);
      ierr = VecGetArray(*vreduced,&rv);CHKERRQ(ierr);
      ierr = ISGetIndices(is,&s);CHKERRQ(ierr);
      if (nlocal > (fhigh-flow)) SETERRQ2(PETSC_COMM_WORLD,1,"IS local size %d > Vec local size %d",nlocal,fhigh-flow);
      for (i=0;i<nlocal;i++) {
        rv[s[i]-flow] = fv[s[i]-flow];
      }
      ierr = ISRestoreIndices(is,&s);CHKERRQ(ierr);
      ierr = VecRestoreArray(vfull,&fv);CHKERRQ(ierr);
      ierr = VecRestoreArray(*vreduced,&rv);CHKERRQ(ierr);
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
- subset_type - the method TAO is using for subsetting (TAO_SUBSET_SUBVEC, TAO_SUBSET_MASK,
  TAO_SUBSET_MATRIXFREE)

  Output Parameters:
. Msub - the submatrix
@*/
PetscErrorCode TaoMatGetSubMat(Mat M, IS is, Vec v1, TaoSubsetType subset_type, Mat *Msub)
{
  PetscErrorCode ierr;
  IS             iscomp;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(M,MAT_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  ierr = MatDestroy(Msub);CHKERRQ(ierr);
  switch (subset_type) {
  case TAO_SUBSET_SUBVEC:
    ierr = MatCreateSubMatrix(M, is, is, MAT_INITIAL_MATRIX, Msub);CHKERRQ(ierr);
    break;

  case TAO_SUBSET_MASK:
    /* Get Reduced Hessian
     Msub[i,j] = M[i,j] if i,j in Free_Local or i==j
     Msub[i,j] = 0      if i!=j and i or j not in Free_Local
     */
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)M),NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-different_submatrix","use separate hessian matrix when computing submatrices","TaoSubsetType",flg,&flg,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (flg) {
      ierr = MatDuplicate(M, MAT_COPY_VALUES, Msub);CHKERRQ(ierr);
    } else {
      /* Act on hessian directly (default) */
      ierr = PetscObjectReference((PetscObject)M);CHKERRQ(ierr);
      *Msub = M;
    }
    /* Save the diagonal to temporary vector */
    ierr = MatGetDiagonal(*Msub,v1);CHKERRQ(ierr);

    /* Zero out rows and columns */
    ierr = ISComplementVec(is,v1,&iscomp);CHKERRQ(ierr);

    /* Use v1 instead of 0 here because of PETSc bug */
    ierr = MatZeroRowsColumnsIS(*Msub,iscomp,1.0,v1,v1);CHKERRQ(ierr);

    ierr = ISDestroy(&iscomp);CHKERRQ(ierr);
    break;
  case TAO_SUBSET_MATRIXFREE:
    ierr = ISComplementVec(is,v1,&iscomp);CHKERRQ(ierr);
    ierr = MatCreateSubMatrixFree(M,iscomp,iscomp,Msub);CHKERRQ(ierr);
    ierr = ISDestroy(&iscomp);CHKERRQ(ierr);
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
- S - step direction with which the active bounds will be estimated

  Output Parameters:
. bound_tol - tolerance for for the bound estimation
. active_lower - index set for active variables at the lower bound
. active_upper - index set for active variables at the upper bound
. active_fixed - index set for fixed variables
. active - index set for all active variables
. inactive - complementary index set for inactive variables
@*/
PetscErrorCode TaoEstimateActiveBounds(Vec X, Vec XL, Vec XU, Vec G, Vec S, PetscReal *bound_tol, 
                                       IS *active_lower, IS *active_upper, IS *active_fixed, IS *active, IS *inactive)
{
  PetscErrorCode               ierr;
  
  Vec                          W;
  PetscReal                    wnorm;
  PetscInt                     i, n_isl=0, n_isu=0, n_isf=0;
  PetscInt                     n, low, high;
  PetscInt                     *isl=NULL, *isu=NULL, *isf=NULL;
  const PetscScalar            *xl, *xu, *x, *g;

  PetscFunctionBegin;  
  /* Update the tolerance for bound detection (this is based on Bertsekas' method) */
  ierr = VecDuplicate(S, &W);CHKERRQ(ierr);
  ierr = VecCopy(S, W);CHKERRQ(ierr);
  ierr = VecAXPBY(W, 1.0, 0.001, X);CHKERRQ(ierr);
  ierr = VecMedian(XL, W, XU, W);CHKERRQ(ierr);
  ierr = VecAXPBY(W, 1.0, -1.0, X);CHKERRQ(ierr);
  ierr = VecNorm(W, NORM_2, &wnorm);CHKERRQ(ierr);
  *bound_tol = PetscMin(*bound_tol, wnorm);
  
  ierr = VecGetOwnershipRange(X, &low, &high);CHKERRQ(ierr);
  ierr = VecGetLocalSize(X, &n);CHKERRQ(ierr);
  if (n>0){
    ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
    ierr = VecGetArrayRead(XL, &xl);CHKERRQ(ierr);
    ierr = VecGetArrayRead(XU, &xu);CHKERRQ(ierr);
    ierr = VecGetArrayRead(G, &g);CHKERRQ(ierr);
    
    /* Loop over variables and categorize the indexes */
    ierr = PetscMalloc1(n, &isl);CHKERRQ(ierr);
    ierr = PetscMalloc1(n, &isu);CHKERRQ(ierr);
    ierr = PetscMalloc1(n, &isf);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      if (xl[i] == xu[i]) {
        /* Fixed variables here */
        isf[n_isf]=low+i; ++n_isf;
      } else if ((x[i] <= xl[i] + *bound_tol) && (g[i] > 0.0)) {
        /* Lower bounded variables here */
        isl[n_isl]=low+i; ++n_isl;
      } else if ((x[i] >= xu[i] - *bound_tol) && (g[i] < 0.0)) {
        /* Upper bounded variables here */
        isu[n_isu]=low+i; ++n_isu;
      }
    }
    
    ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(XL, &xl);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(XU, &xu);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(G, &g);CHKERRQ(ierr);
  }
  
  /* Create index set for lower bounded variables */
  ierr = ISDestroy(active_lower);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)X), n_isl, isl, PETSC_OWN_POINTER, active_lower);CHKERRQ(ierr);
  /* Create index set for upper bounded variables */
  ierr = ISDestroy(active_upper);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)X), n_isu, isu, PETSC_OWN_POINTER, active_upper);CHKERRQ(ierr);
  /* Create index set for fixed variables */
  ierr = ISDestroy(active_fixed);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)X), n_isf, isf, PETSC_OWN_POINTER, active_fixed);CHKERRQ(ierr);
  
  /* Create the combined active set */
  ierr = ISDestroy(active);CHKERRQ(ierr);
  if (*active_lower && *active_upper && *active_fixed) {
    /* All three types of active variables exist */
    const IS islist[3] = {*active_lower, *active_upper, *active_fixed};
    ierr = ISConcatenate(PetscObjectComm((PetscObject)X), 3, islist, active);CHKERRQ(ierr);
    ierr = ISSort(*active);CHKERRQ(ierr);
  } else if (*active_lower && *active_upper) {
    /* Only lower and upper bounded active variables exist */
    ierr = ISSum(*active_lower, *active_upper, active);CHKERRQ(ierr);
  } else if (*active_lower && *active_fixed) {
    /* Only lower bounded and fixed active variables exist */
    ierr = ISSum(*active_lower, *active_fixed, active);CHKERRQ(ierr);
  } else if (*active_upper && *active_fixed) {
    /* Only upper bounded and fixed active variables exist */
    ierr = ISSum(*active_upper, *active_fixed, active);CHKERRQ(ierr);
  } else if (*active_lower) {
    /* Only lower bounded active variables exist */
    *active = *active_lower;
  } else if (*active_upper) {
    /* Only upper bounded active variables exist */
    *active = *active_upper;
  } else if (*active_fixed) {
    /* Only fixed active variables exist */
    *active = *active_fixed;
  }
  /* Create the inactive set */
  ierr = ISDestroy(inactive);CHKERRQ(ierr);
  if (*active) { ierr = ISComplementVec(*active, X, inactive);CHKERRQ(ierr); }
  
  /* Clean up and exit */
  ierr = VecDestroy(&W);CHKERRQ(ierr);
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
- active_fixed - index set for fixed active variables

  Output Parameters:
. S - step direction to be modified
@*/
PetscErrorCode TaoBoundStep(Vec X, Vec XL, Vec XU, IS active_lower, IS active_upper, IS active_fixed, Vec S) 
{
  PetscErrorCode               ierr;
  
  Vec                          step_lower, step_upper, step_fixed;
  Vec                          x_lower, x_upper;
  Vec                          bound_lower, bound_upper;
  
  PetscFunctionBegin;
  /* Adjust step for variables at the estimated lower bound */
  if (active_lower) {
    ierr = VecGetSubVector(S, active_lower, &step_lower);CHKERRQ(ierr);
    ierr = VecGetSubVector(X, active_lower, &x_lower);CHKERRQ(ierr);
    ierr = VecGetSubVector(XL, active_lower, &bound_lower);CHKERRQ(ierr);
    ierr = VecCopy(bound_lower, step_lower);CHKERRQ(ierr);
    ierr = VecAXPY(step_lower, -1.0, x_lower);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(S, active_lower, &step_lower);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(X, active_lower, &x_lower);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(XL, active_lower, &bound_lower);CHKERRQ(ierr);
  }
  
  /* Adjust step for the variables at the estimated upper bound */
  if (active_upper) {
    ierr = VecGetSubVector(S, active_upper, &step_upper);CHKERRQ(ierr);
    ierr = VecGetSubVector(X, active_upper, &x_upper);CHKERRQ(ierr);
    ierr = VecGetSubVector(XU, active_upper, &bound_upper);CHKERRQ(ierr);
    ierr = VecCopy(bound_upper, step_upper);CHKERRQ(ierr);
    ierr = VecAXPY(step_upper, -1.0, x_upper);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(S, active_upper, &step_upper);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(X, active_upper, &x_upper);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(XU, active_upper, &bound_upper);CHKERRQ(ierr);
  }
  
  /* Zero out step for fixed variables */
  if (active_fixed) {
    ierr = VecGetSubVector(S, active_fixed, &step_fixed);CHKERRQ(ierr);
    ierr = VecSet(step_fixed, 0.0);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(S, active_fixed, &step_fixed);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
