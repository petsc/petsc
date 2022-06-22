
static char help[] = "Extract submatrices using unsorted indices. For SEQSBAIJ either sort both rows and columns, or sort none.\n\n";
/*
   Take a 4x4 grid and form a 5-point stencil graph Laplacian over it.
   Partition the grid into two subdomains by splitting into two in the j-direction (slowest varying).
   Impose an overlap of 1 and order the subdomains with the j-direction varying fastest.
   Extract the subdomain submatrices, one per rank.
*/
/* Results:
    Sequential:
    - seqaij:   will error out, if rows or columns are unsorted
    - seqbaij:  will extract submatrices correctly even for unsorted row or column indices
    - seqsbaij: will extract submatrices correctly even for unsorted row and column indices (both must be sorted or not);
                CANNOT automatically report inversions, because MatGetRow is not available.
    MPI:
    - mpiaij:   will error out, if columns are unsorted
    - mpibaij:  will error out, if columns are unsorted.
    - mpisbaij: will error out, if columns are unsorted; even with unsorted rows will produce correct submatrices;
                CANNOT automatically report inversions, because MatGetRow is not available.
*/

#include <petscmat.h>
#include <petscis.h>

int main(int argc,char **args)
{
  Mat            A, *S;
  IS             rowis[2], colis[2];
  PetscInt       n,N,i,j,k,l,nsub,Jlow[2] = {0,1}, *jlow, Jhigh[2] = {3,4}, *jhigh, row, col, *subindices, ncols;
  const PetscInt *cols;
  PetscScalar    v;
  PetscMPIInt    rank, size, p, inversions, total_inversions;
  PetscBool      sort_rows, sort_cols, show_inversions;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size < 3,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE, "A uniprocessor or two-processor example only.");

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  if (size > 1) {
    n = 8; N = 16;
  } else {
    n = 16; N = 16;
  }
  PetscCall(MatSetSizes(A,n,n,N,N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /* Don't care if the entries are set multiple times by different procs. */
  for (i=0; i<4; ++i) {
    for (j = 0; j<4; ++j) {
      row = j*4+i;
      v   = -1.0;
      if (i>0) {
        col =  row-1; PetscCall(MatSetValues(A,1,&row,1,&col,&v,INSERT_VALUES));
      }
      if (i<3) {
        col = row+1; PetscCall(MatSetValues(A,1,&row,1,&col,&v,INSERT_VALUES));
      }
      if (j>0) {
        col = row-4; PetscCall(MatSetValues(A,1,&row,1,&col,&v,INSERT_VALUES));
      }
      if (j<3) {
        col = row+4; PetscCall(MatSetValues(A,1,&row,1,&col,&v,INSERT_VALUES));
      }
      v    = 4.0;
      PetscCall(MatSetValues(A,1,&row,1,&row,&v,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Original matrix\n"));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  if (size > 1) {
    nsub = 1; /* one subdomain per rank */
  }
  else {
    nsub = 2; /* both subdomains on rank 0 */
  }
  if (rank) {
    jlow = Jlow+1; jhigh = Jhigh+1;
  }
  else {
    jlow = Jlow; jhigh = Jhigh;
  }
  sort_rows = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL, "-sort_rows", &sort_rows, NULL));
  sort_cols = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL, "-sort_cols", &sort_cols, NULL));
  for (l = 0; l < nsub; ++l) {
    PetscCall(PetscMalloc1(12, &subindices));
    k    = 0;
    for (i = 0; i < 4; ++i) {
      for (j = jlow[l]; j < jhigh[l]; ++j) {
        subindices[k] = j*4+i;
        k++;
      }
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 12, subindices, PETSC_OWN_POINTER, rowis+l));
    if ((sort_rows && !sort_cols) || (!sort_rows && sort_cols)) {
      PetscCall(ISDuplicate(rowis[l],colis+l));
    } else {
      PetscCall(PetscObjectReference((PetscObject)rowis[l]));
      colis[l] = rowis[l];
    }
    if (sort_rows) {
      PetscCall(ISSort(rowis[l]));
    }
    if (sort_cols) {
      PetscCall(ISSort(colis[l]));
    }
  }

  PetscCall(MatCreateSubMatrices(A,nsub,rowis,colis,MAT_INITIAL_MATRIX, &S));

  show_inversions = PETSC_FALSE;

  PetscCall(PetscOptionsGetBool(NULL,NULL, "-show_inversions", &show_inversions, NULL));

  inversions = 0;
  for (p = 0; p < size; ++p) {
    if (p == rank) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%" PetscInt_FMT ":%" PetscInt_FMT "]: Number of subdomains: %" PetscInt_FMT ":\n", rank, size, nsub));
      for (l = 0; l < nsub; ++l) {
        PetscInt i0, i1;
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%" PetscInt_FMT ":%" PetscInt_FMT "]: Subdomain row IS %" PetscInt_FMT ":\n", rank, size, l));
        PetscCall(ISView(rowis[l],PETSC_VIEWER_STDOUT_SELF));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%" PetscInt_FMT ":%" PetscInt_FMT "]: Subdomain col IS %" PetscInt_FMT ":\n", rank, size, l));
        PetscCall(ISView(colis[l],PETSC_VIEWER_STDOUT_SELF));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%" PetscInt_FMT ":%" PetscInt_FMT "]: Submatrix %" PetscInt_FMT ":\n", rank, size, l));
        PetscCall(MatView(S[l],PETSC_VIEWER_STDOUT_SELF));
        if (show_inversions) {
          PetscCall(MatGetOwnershipRange(S[l], &i0,&i1));
          for (i = i0; i < i1; ++i) {
            PetscCall(MatGetRow(S[l], i, &ncols, &cols, NULL));
            for (j = 1; j < ncols; ++j) {
              if (cols[j] < cols[j-1]) {
                PetscCall(PetscPrintf(PETSC_COMM_SELF, "***Inversion in row %" PetscInt_FMT ": col[%" PetscInt_FMT "] = %" PetscInt_FMT " < %" PetscInt_FMT " = col[%" PetscInt_FMT "]\n", i, j, cols[j], cols[j-1], j-1));
                inversions++;
              }
            }
            PetscCall(MatRestoreRow(S[l], i, &ncols, &cols, NULL));
          }
        }
      }
    }
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  }
  if (show_inversions) {
    PetscCallMPI(MPI_Reduce(&inversions,&total_inversions,1,MPIU_INT, MPI_SUM,0,PETSC_COMM_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "*Total inversions: %" PetscInt_FMT "\n", total_inversions));
  }
  PetscCall(MatDestroy(&A));

  for (l = 0; l < nsub; ++l) {
    PetscCall(ISDestroy(&(rowis[l])));
    PetscCall(ISDestroy(&(colis[l])));
  }
  PetscCall(MatDestroySubMatrices(nsub,&S));
  PetscCall(PetscFinalize());
  return 0;
}
