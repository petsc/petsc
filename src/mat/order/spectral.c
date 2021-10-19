#include <petscmat.h> /*I <petscmat.h> I*/
#include <petscblaslapack.h>

/*@
  MatCreateLaplacian - Create the matrix Laplacian, with all values in the matrix less than the tolerance set to zero

  Input Parameters:
+ A   - The matrix
. tol - The zero tolerance
- weighted - Flag for using edge weights

  Output Parameters:
. L - The graph Laplacian matrix

  Level: intermediate

.seealso: MatChop()
 @*/
PetscErrorCode MatCreateLaplacian(Mat A, PetscReal tol, PetscBool weighted, Mat *L)
{
  PetscScalar   *newVals;
  PetscInt      *newCols;
  PetscInt       rStart, rEnd, r, colMax = 0;
  PetscInt      *dnnz, *onnz;
  PetscInt       m, n, M, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (weighted) SETERRQ(PetscObjectComm((PetscObject) A), PETSC_ERR_SUP, "Will get to this soon");
  ierr = MatCreate(PetscObjectComm((PetscObject) A), L);CHKERRQ(ierr);
  ierr = MatGetSize(A, &M, &N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A, &m, &n);CHKERRQ(ierr);
  ierr = MatSetSizes(*L, m, n, M, N);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A, &rStart, &rEnd);CHKERRQ(ierr);
  ierr = PetscMalloc2(m,&dnnz,m,&onnz);CHKERRQ(ierr);
  for (r = rStart; r < rEnd; ++r) {
    const PetscScalar *vals;
    const PetscInt    *cols;
    PetscInt           ncols, newcols, c;
    PetscBool          hasdiag = PETSC_FALSE;

    dnnz[r-rStart] = onnz[r-rStart] = 0;
    ierr = MatGetRow(A, r, &ncols, &cols, &vals);CHKERRQ(ierr);
    for (c = 0, newcols = 0; c < ncols; ++c) {
      if (cols[c] == r) {
        ++newcols;
        hasdiag = PETSC_TRUE;
        ++dnnz[r-rStart];
      } else if (PetscAbsScalar(vals[c]) >= tol) {
        if ((cols[c] >= rStart) && (cols[c] < rEnd)) ++dnnz[r-rStart];
        else                                         ++onnz[r-rStart];
        ++newcols;
      }
    }
    if (!hasdiag) {++newcols; ++dnnz[r-rStart];}
    colMax = PetscMax(colMax, newcols);
    ierr = MatRestoreRow(A, r, &ncols, &cols, &vals);CHKERRQ(ierr);
  }
  ierr = MatSetFromOptions(*L);CHKERRQ(ierr);
  ierr = MatXAIJSetPreallocation(*L, 1, dnnz, onnz, NULL, NULL);CHKERRQ(ierr);
  ierr = MatSetUp(*L);CHKERRQ(ierr);
  ierr = PetscMalloc2(colMax,&newCols,colMax,&newVals);CHKERRQ(ierr);
  for (r = rStart; r < rEnd; ++r) {
    const PetscScalar *vals;
    const PetscInt    *cols;
    PetscInt           ncols, newcols, c;
    PetscBool          hasdiag = PETSC_FALSE;

    ierr = MatGetRow(A, r, &ncols, &cols, &vals);CHKERRQ(ierr);
    for (c = 0, newcols = 0; c < ncols; ++c) {
      if (cols[c] == r) {
        newCols[newcols] = cols[c];
        newVals[newcols] = dnnz[r-rStart]+onnz[r-rStart]-1;
        ++newcols;
        hasdiag = PETSC_TRUE;
      } else if (PetscAbsScalar(vals[c]) >= tol) {
        newCols[newcols] = cols[c];
        newVals[newcols] = -1.0;
        ++newcols;
      }
      if (newcols > colMax) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Overran work space");
    }
    if (!hasdiag) {
      newCols[newcols] = r;
      newVals[newcols] = dnnz[r-rStart]+onnz[r-rStart]-1;
      ++newcols;
    }
    ierr = MatRestoreRow(A, r, &ncols, &cols, &vals);CHKERRQ(ierr);
    ierr = MatSetValues(*L, 1, &r, newcols, newCols, newVals, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree2(dnnz,onnz);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*L, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*L, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree2(newCols,newVals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  MatGetOrdering_Spectral - Find the symmetric reordering of the graph by .
*/
PETSC_INTERN PetscErrorCode MatGetOrdering_Spectral(Mat A, MatOrderingType type, IS *row, IS *col)
{
  Mat             L;
  const PetscReal eps = 1.0e-12;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatCreateLaplacian(A, eps, PETSC_FALSE, &L);CHKERRQ(ierr);
  {
    /* Check Laplacian */
    PetscReal norm;
    Vec       x, y;

    ierr = MatCreateVecs(L, &x, NULL);CHKERRQ(ierr);
    ierr = VecDuplicate(x, &y);CHKERRQ(ierr);
    ierr = VecSet(x, 1.0);CHKERRQ(ierr);
    ierr = MatMult(L, x, y);CHKERRQ(ierr);
    ierr = VecNorm(y, NORM_INFINITY, &norm);CHKERRQ(ierr);
    if (norm > 1.0e-10) SETERRQ(PetscObjectComm((PetscObject) y), PETSC_ERR_PLIB, "Invalid graph Laplacian");
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
  }
  /* Compute Fiedler vector (right now, all eigenvectors) */
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PetscObjectComm((PetscObject) A), PETSC_ERR_SUP, "Spectral partitioning does not support complex numbers");
#else
  {
    Mat          LD;
    PetscScalar *a;
    PetscReal   *realpart, *imagpart, *eigvec, *work;
    PetscReal    sdummy;
    PetscBLASInt bn, bN, lwork = 0, lierr, idummy;
    PetscInt     n, i, evInd, *perm, tmp;

    ierr = MatConvert(L, MATDENSE, MAT_INITIAL_MATRIX, &LD);CHKERRQ(ierr);
    ierr = MatGetLocalSize(LD, &n, NULL);CHKERRQ(ierr);
    ierr = MatDenseGetArray(LD, &a);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(n, &bn);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(n, &bN);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(5*n,&lwork);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(1,&idummy);CHKERRQ(ierr);
    ierr = PetscMalloc4(n,&realpart,n,&imagpart,n*n,&eigvec,lwork,&work);CHKERRQ(ierr);
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgeev", LAPACKgeev_("N","V",&bn,a,&bN,realpart,imagpart,&sdummy,&idummy,eigvec,&bN,work,&lwork,&lierr));
    if (lierr) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine %d", (int) lierr);
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(LD,&a);CHKERRQ(ierr);
    ierr = MatDestroy(&LD);CHKERRQ(ierr);
    /* Check lowest eigenvalue and eigenvector */
    ierr = PetscMalloc1(n, &perm);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) perm[i] = i;
    ierr = PetscSortRealWithPermutation(n,realpart,perm);CHKERRQ(ierr);
    evInd = perm[0];
    if ((realpart[evInd] > 1.0e-12) || (imagpart[evInd] > 1.0e-12)) SETERRQ(PetscObjectComm((PetscObject) L), PETSC_ERR_PLIB, "Graph Laplacian must have lowest eigenvalue 0");
    evInd = perm[1];
    if ((realpart[evInd] < 1.0e-12) && (imagpart[evInd] < 1.0e-12)) SETERRQ(PetscObjectComm((PetscObject) L), PETSC_ERR_PLIB, "Graph Laplacian must have only one zero eigenvalue");
    evInd = perm[0];
    for (i = 0; i < n; ++i) {
      if (PetscAbsReal(eigvec[evInd*n+i] - eigvec[evInd*n+0]) > 1.0e-10) SETERRQ3(PetscObjectComm((PetscObject) L), PETSC_ERR_PLIB, "Graph Laplacian must have constant lowest eigenvector ev_%d %g != ev_0 %g", i, eigvec[evInd*n+i], eigvec[evInd*n+0]);
    }
    /* Construct Fiedler partition */
    evInd = perm[1];
    for (i = 0; i < n; ++i) perm[i] = i;
    ierr = PetscSortRealWithPermutation(n, &eigvec[evInd*n], perm);CHKERRQ(ierr);
    for (i = 0; i < n/2; ++i) {
      tmp          = perm[n-1-i];
      perm[n-1-i] = perm[i];
      perm[i]     = tmp;
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF, n, perm, PETSC_OWN_POINTER, row);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject) *row);CHKERRQ(ierr);
    *col = *row;

    ierr = PetscFree4(realpart,imagpart,eigvec,work);CHKERRQ(ierr);
    ierr = MatDestroy(&L);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif
}
