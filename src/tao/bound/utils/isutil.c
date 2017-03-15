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
