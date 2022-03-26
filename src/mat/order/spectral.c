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
  PetscCall(MatCreate(PetscObjectComm((PetscObject) A), L));
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatSetSizes(*L, m, n, M, N));
  PetscCall(MatGetOwnershipRange(A, &rStart, &rEnd));
  PetscCall(PetscMalloc2(m,&dnnz,m,&onnz));
  for (r = rStart; r < rEnd; ++r) {
    const PetscScalar *vals;
    const PetscInt    *cols;
    PetscInt           ncols, newcols, c;
    PetscBool          hasdiag = PETSC_FALSE;

    dnnz[r-rStart] = onnz[r-rStart] = 0;
    PetscCall(MatGetRow(A, r, &ncols, &cols, &vals));
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
    PetscCall(MatRestoreRow(A, r, &ncols, &cols, &vals));
  }
  PetscCall(MatSetFromOptions(*L));
  PetscCall(MatXAIJSetPreallocation(*L, 1, dnnz, onnz, NULL, NULL));
  PetscCall(MatSetUp(*L));
  PetscCall(PetscMalloc2(colMax,&newCols,colMax,&newVals));
  for (r = rStart; r < rEnd; ++r) {
    const PetscScalar *vals;
    const PetscInt    *cols;
    PetscInt           ncols, newcols, c;
    PetscBool          hasdiag = PETSC_FALSE;

    PetscCall(MatGetRow(A, r, &ncols, &cols, &vals));
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
    PetscCall(MatRestoreRow(A, r, &ncols, &cols, &vals));
    PetscCall(MatSetValues(*L, 1, &r, newcols, newCols, newVals, INSERT_VALUES));
  }
  PetscCall(PetscFree2(dnnz,onnz));
  PetscCall(MatAssemblyBegin(*L, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*L, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscFree2(newCols,newVals));
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
  PetscCall(MatCreateLaplacian(A, eps, PETSC_FALSE, &L));
  {
    /* Check Laplacian */
    PetscReal norm;
    Vec       x, y;

    PetscCall(MatCreateVecs(L, &x, NULL));
    PetscCall(VecDuplicate(x, &y));
    PetscCall(VecSet(x, 1.0));
    PetscCall(MatMult(L, x, y));
    PetscCall(VecNorm(y, NORM_INFINITY, &norm));
    PetscCheck(norm <= 1.0e-10,PetscObjectComm((PetscObject) y), PETSC_ERR_PLIB, "Invalid graph Laplacian");
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
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

    PetscCall(MatConvert(L, MATDENSE, MAT_INITIAL_MATRIX, &LD));
    PetscCall(MatGetLocalSize(LD, &n, NULL));
    PetscCall(MatDenseGetArray(LD, &a));
    PetscCall(PetscBLASIntCast(n, &bn));
    PetscCall(PetscBLASIntCast(n, &bN));
    PetscCall(PetscBLASIntCast(5*n,&lwork));
    PetscCall(PetscBLASIntCast(1,&idummy));
    PetscCall(PetscMalloc4(n,&realpart,n,&imagpart,n*n,&eigvec,lwork,&work));
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKgeev", LAPACKgeev_("N","V",&bn,a,&bN,realpart,imagpart,&sdummy,&idummy,eigvec,&bN,work,&lwork,&lierr));
    PetscCheck(!lierr,PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine %d", (int) lierr);
    PetscCall(PetscFPTrapPop());
    PetscCall(MatDenseRestoreArray(LD,&a));
    PetscCall(MatDestroy(&LD));
    /* Check lowest eigenvalue and eigenvector */
    PetscCall(PetscMalloc1(n, &perm));
    for (i = 0; i < n; ++i) perm[i] = i;
    PetscCall(PetscSortRealWithPermutation(n,realpart,perm));
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
    PetscCall(PetscSortRealWithPermutation(n, &eigvec[evInd*n], perm));
    for (i = 0; i < n/2; ++i) {
      tmp          = perm[n-1-i];
      perm[n-1-i] = perm[i];
      perm[i]     = tmp;
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, n, perm, PETSC_OWN_POINTER, row));
    PetscCall(PetscObjectReference((PetscObject) *row));
    *col = *row;

    PetscCall(PetscFree4(realpart,imagpart,eigvec,work));
    PetscCall(MatDestroy(&L));
    PetscFunctionReturn(0);
  }
#endif
}
