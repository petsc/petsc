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

  PetscFunctionBegin;
  PetscCheck(!weighted,PetscObjectComm((PetscObject) A), PETSC_ERR_SUP, "Will get to this soon");
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject) A), L));
  CHKERRQ(MatGetSize(A, &M, &N));
  CHKERRQ(MatGetLocalSize(A, &m, &n));
  CHKERRQ(MatSetSizes(*L, m, n, M, N));
  CHKERRQ(MatGetOwnershipRange(A, &rStart, &rEnd));
  CHKERRQ(PetscMalloc2(m,&dnnz,m,&onnz));
  for (r = rStart; r < rEnd; ++r) {
    const PetscScalar *vals;
    const PetscInt    *cols;
    PetscInt           ncols, newcols, c;
    PetscBool          hasdiag = PETSC_FALSE;

    dnnz[r-rStart] = onnz[r-rStart] = 0;
    CHKERRQ(MatGetRow(A, r, &ncols, &cols, &vals));
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
    CHKERRQ(MatRestoreRow(A, r, &ncols, &cols, &vals));
  }
  CHKERRQ(MatSetFromOptions(*L));
  CHKERRQ(MatXAIJSetPreallocation(*L, 1, dnnz, onnz, NULL, NULL));
  CHKERRQ(MatSetUp(*L));
  CHKERRQ(PetscMalloc2(colMax,&newCols,colMax,&newVals));
  for (r = rStart; r < rEnd; ++r) {
    const PetscScalar *vals;
    const PetscInt    *cols;
    PetscInt           ncols, newcols, c;
    PetscBool          hasdiag = PETSC_FALSE;

    CHKERRQ(MatGetRow(A, r, &ncols, &cols, &vals));
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
      PetscCheck(newcols <= colMax,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Overran work space");
    }
    if (!hasdiag) {
      newCols[newcols] = r;
      newVals[newcols] = dnnz[r-rStart]+onnz[r-rStart]-1;
      ++newcols;
    }
    CHKERRQ(MatRestoreRow(A, r, &ncols, &cols, &vals));
    CHKERRQ(MatSetValues(*L, 1, &r, newcols, newCols, newVals, INSERT_VALUES));
  }
  CHKERRQ(PetscFree2(dnnz,onnz));
  CHKERRQ(MatAssemblyBegin(*L, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*L, MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscFree2(newCols,newVals));
  PetscFunctionReturn(0);
}

/*
  MatGetOrdering_Spectral - Find the symmetric reordering of the graph by .
*/
PETSC_INTERN PetscErrorCode MatGetOrdering_Spectral(Mat A, MatOrderingType type, IS *row, IS *col)
{
  Mat             L;
  const PetscReal eps = 1.0e-12;

  PetscFunctionBegin;
  CHKERRQ(MatCreateLaplacian(A, eps, PETSC_FALSE, &L));
  {
    /* Check Laplacian */
    PetscReal norm;
    Vec       x, y;

    CHKERRQ(MatCreateVecs(L, &x, NULL));
    CHKERRQ(VecDuplicate(x, &y));
    CHKERRQ(VecSet(x, 1.0));
    CHKERRQ(MatMult(L, x, y));
    CHKERRQ(VecNorm(y, NORM_INFINITY, &norm));
    PetscCheck(norm <= 1.0e-10,PetscObjectComm((PetscObject) y), PETSC_ERR_PLIB, "Invalid graph Laplacian");
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&y));
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

    CHKERRQ(MatConvert(L, MATDENSE, MAT_INITIAL_MATRIX, &LD));
    CHKERRQ(MatGetLocalSize(LD, &n, NULL));
    CHKERRQ(MatDenseGetArray(LD, &a));
    CHKERRQ(PetscBLASIntCast(n, &bn));
    CHKERRQ(PetscBLASIntCast(n, &bN));
    CHKERRQ(PetscBLASIntCast(5*n,&lwork));
    CHKERRQ(PetscBLASIntCast(1,&idummy));
    CHKERRQ(PetscMalloc4(n,&realpart,n,&imagpart,n*n,&eigvec,lwork,&work));
    CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKgeev", LAPACKgeev_("N","V",&bn,a,&bN,realpart,imagpart,&sdummy,&idummy,eigvec,&bN,work,&lwork,&lierr));
    PetscCheck(!lierr,PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine %d", (int) lierr);
    CHKERRQ(PetscFPTrapPop());
    CHKERRQ(MatDenseRestoreArray(LD,&a));
    CHKERRQ(MatDestroy(&LD));
    /* Check lowest eigenvalue and eigenvector */
    CHKERRQ(PetscMalloc1(n, &perm));
    for (i = 0; i < n; ++i) perm[i] = i;
    CHKERRQ(PetscSortRealWithPermutation(n,realpart,perm));
    evInd = perm[0];
    PetscCheck(realpart[evInd] <= 1.0e-12 && imagpart[evInd] <= 1.0e-12,PetscObjectComm((PetscObject) L), PETSC_ERR_PLIB, "Graph Laplacian must have lowest eigenvalue 0");
    evInd = perm[1];
    PetscCheck(realpart[evInd] >= 1.0e-12 || imagpart[evInd] >= 1.0e-12,PetscObjectComm((PetscObject) L), PETSC_ERR_PLIB, "Graph Laplacian must have only one zero eigenvalue");
    evInd = perm[0];
    for (i = 0; i < n; ++i) {
      PetscCheck(PetscAbsReal(eigvec[evInd*n+i] - eigvec[evInd*n+0]) <= 1.0e-10,PetscObjectComm((PetscObject) L), PETSC_ERR_PLIB, "Graph Laplacian must have constant lowest eigenvector ev_%" PetscInt_FMT " %g != ev_0 %g", i, (double)(eigvec[evInd*n+i]), (double)(eigvec[evInd*n+0]));
    }
    /* Construct Fiedler partition */
    evInd = perm[1];
    for (i = 0; i < n; ++i) perm[i] = i;
    CHKERRQ(PetscSortRealWithPermutation(n, &eigvec[evInd*n], perm));
    for (i = 0; i < n/2; ++i) {
      tmp          = perm[n-1-i];
      perm[n-1-i] = perm[i];
      perm[i]     = tmp;
    }
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, n, perm, PETSC_OWN_POINTER, row));
    CHKERRQ(PetscObjectReference((PetscObject) *row));
    *col = *row;

    CHKERRQ(PetscFree4(realpart,imagpart,eigvec,work));
    CHKERRQ(MatDestroy(&L));
    PetscFunctionReturn(0);
  }
#endif
}
