#include <petscdm.h>


// TODO: Most of the arguments here can be stored in AdolcCtx

/*
  Simple matrix printing

  Input parameters:
  comm - MPI communicator
  name - name of matrix to print
  m,n  - number of rows and columns, respectively
  M    - matrix to print
*/
PetscErrorCode PrintMat(MPI_Comm comm,const char* name,PetscInt m,PetscInt n,PetscScalar **M)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr = PetscPrintf(comm,"%s \n",name);CHKERRQ(ierr);
  for(i=0; i<m ;i++) {
    ierr = PetscPrintf(comm,"\n %d: ",i);CHKERRQ(ierr);
    for(j=0; j<n ;j++)
      ierr = PetscPrintf(comm," %10.4f ", M[i][j]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm,"\n\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Print sparsity pattern

  Input parameters:
  comm     - MPI communicator
  m        - number of rows

  Output parameter:
  sparsity - matrix sparsity pattern, typically computed using an ADOL-C function such as jac_pat or
  hess_pat
*/
PetscErrorCode PrintSparsity(MPI_Comm comm,PetscInt m,unsigned int **sparsity)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr = PetscPrintf(comm,"Sparsity pattern:\n");CHKERRQ(ierr);
  for(i=0; i<m ;i++) {
    ierr = PetscPrintf(comm,"\n %2d: ",i);CHKERRQ(ierr);
    for(j=1; j<= (PetscInt) sparsity[i][0] ;j++)
      ierr = PetscPrintf(comm," %2d ",sparsity[i][j]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm,"\n\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Extract an index set coloring from a sparsity pattern

  Input parameters:
  da         - distributed array

  Output parameter:
  iscoloring - index set coloring corresponding to the sparsity pattern under the given coloring type

  Notes:
  Implementation works fine for DM_BOUNDARY_NONE or DM_BOUNDARY_GHOSTED. If DM_BOUNDARY_PERIODIC is
  used then implementation only currently works in parallel, where processors should not own two
  opposite boundaries which have been identified by the periodicity.
*/
PetscErrorCode GetColoring(DM da,ISColoring *iscoloring)
{
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = DMCreateColoring(da,IS_COLORING_LOCAL,iscoloring);CHKERRQ(ierr);
  //ierr = DMCreateColoring(da,IS_COLORING_GLOBAL,iscoloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Simple function to count the number of colors used in an index set coloring

  Input parameter:
  iscoloring - the index set coloring to count the number of colors of

  Output parameter:
  p          - number of colors used in iscoloring
*/
PetscErrorCode CountColors(ISColoring iscoloring,PetscInt *p)
{
  PetscErrorCode ierr;
  IS             *is;

  PetscFunctionBegin;
  ierr = ISColoringGetIS(iscoloring,p,&is);CHKERRQ(ierr);
  ierr = ISColoringRestoreIS(iscoloring,&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// FIXME: Generate sparsity pattern using PETSc alone
PetscErrorCode DMGetSparsity(DM da,unsigned int **sparsity)
{
  PetscErrorCode ierr;
  PetscInt       i,j,nnz;
  //PetscInt       nrows,ncols;
  PetscInt       rowstart,rowend;
  const PetscInt *cols;
  Mat            A;

  PetscFunctionBegin;
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  //ierr = MatGetSize(A,&nrows,&ncols);CHKERRQ(ierr);
  //for (i=0; i<nrows; i++) {
  ierr = MatGetOwnershipRange(A,&rowstart,&rowend);CHKERRQ(ierr);
  for (i=rowstart; i<rowend; i++) {
    ierr = MatGetRow(A,i,&nnz,&cols,NULL);CHKERRQ(ierr);

    ierr = PetscPrintf(MPI_COMM_WORLD,"%d: ",i);CHKERRQ(ierr);
    for (j=0; j<nnz; j++) {
      ierr = PetscPrintf(MPI_COMM_WORLD,"%d, ",cols[j]);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(MPI_COMM_WORLD,"\n");CHKERRQ(ierr);

    ierr = MatRestoreRow(A,i,&nnz,&cols,NULL);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/*
  Generate a seed matrix defining the partition of columns of a matrix by a particular coloring,
  used for matrix compression

  Input parameter:
  iscoloring - the index set coloring to be used

  Output parameter:
  S          - the resulting seed matrix

  Notes:
  Before calling GenerateSeedMatrix, Seed should be allocated as a logically 2d array with number of
  rows equal to the matrix to be compressed and number of columns equal to the number of colors used
  in iscoloring.
*/
PetscErrorCode GenerateSeedMatrix(ISColoring iscoloring,PetscScalar **S)
{
  PetscErrorCode ierr;
  IS             *is;
  PetscInt       p,size,colour,j;
  const PetscInt *indices;

  PetscFunctionBegin;
  ierr = ISColoringGetIS(iscoloring,&p,&is);CHKERRQ(ierr);
  for (colour=0; colour<p; colour++) {
    ierr = ISGetLocalSize(is[colour],&size);CHKERRQ(ierr);
    ierr = ISGetIndices(is[colour],&indices);CHKERRQ(ierr);
    for (j=0; j<size; j++)
      S[indices[j]][colour] = 1.;
    ierr = ISRestoreIndices(is[colour],&indices);CHKERRQ(ierr);
  }
  ierr = ISColoringRestoreIS(iscoloring,&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Establish a look-up vector whose entries contain the colour used for that diagonal entry. Clearly
  we require the number of dependent and independent variables to be equal in this case.
  Input parameters:
  S        - the seed matrix defining the coloring
  sparsity - the sparsity pattern of the matrix to be recovered, typically computed using an ADOL-C
             function, such as jac_pat or hess_pat
  m        - the number of rows of Seed (and the matrix to be recovered)
  p        - the number of colors used (also the number of columns in Seed)
  Output parameter:
  R        - the recovery vector to be used for de-compression
*/
PetscErrorCode GenerateSeedMatrixPlusRecovery(ISColoring iscoloring,PetscScalar **S,PetscScalar *R)
{
  PetscErrorCode ierr;
  IS             *is;
  PetscInt       p,size,colour,j;
  const PetscInt *indices;

  PetscFunctionBegin;
  ierr = ISColoringGetIS(iscoloring,&p,&is);CHKERRQ(ierr);
  for (colour=0; colour<p; colour++) {
    ierr = ISGetLocalSize(is[colour],&size);CHKERRQ(ierr);
    ierr = ISGetIndices(is[colour],&indices);CHKERRQ(ierr);
    for (j=0; j<size; j++) {
      S[indices[j]][colour] = 1.;
      R[indices[j]] = colour;
    }
    ierr = ISRestoreIndices(is[colour],&indices);CHKERRQ(ierr);
  }
  ierr = ISColoringRestoreIS(iscoloring,&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Establish a look-up matrix whose entries contain the column coordinates of the corresponding entry
  in a matrix which has been compressed using the coloring defined by some seed matrix

  Input parameters:
  S        - the seed matrix defining the coloring
  sparsity - the sparsity pattern of the matrix to be recovered, typically computed using an ADOL-C
             function, such as jac_pat or hess_pat
  m        - the number of rows of Seed (and the matrix to be recovered)
  p        - the number of colors used (also the number of columns in Seed)

  Output parameter:
  R        - the recovery matrix to be used for de-compression
*/
PetscErrorCode GetRecoveryMatrix(PetscScalar **S,unsigned int **sparsity,PetscInt m,PetscInt p,PetscScalar **R)
{
  PetscInt i,j,k,colour;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    for (colour=0; colour<p; colour++) {
      R[i][colour] = -1.;
      for (k=1; k<=(PetscInt) sparsity[i][0]; k++) {
        j = (PetscInt) sparsity[i][k];
        if (S[j][colour] == 1.) {
          R[i][colour] = j;
          break;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  Recover the values of a sparse matrix from a compressed format and insert these into a matrix

  Input parameters:
  mode - use INSERT_VALUES or ADD_VALUES, as required
  m    - number of rows of matrix.
  p    - number of colors used in the compression of J (also the number of columns of R)
  R    - recovery matrix to use in the decompression procedure
  C    - compressed matrix to recover values from
  a    - shift value for implicit problems (select NULL or unity for explicit problems)

  Output parameter:
  A    - Mat to be populated with values from compressed matrix
*/
PetscErrorCode RecoverJacobian(Mat A,InsertMode mode,PetscInt m,PetscInt p,PetscScalar **R,PetscScalar **C,PetscReal *a)
{
  PetscErrorCode ierr;
  PetscInt       i,j,colour;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    for (colour=0; colour<p; colour++) {
      j = (PetscInt) R[i][colour];
      if (j != -1) {
        if (a)
          C[i][colour] *= *a;
        ierr = MatSetValues(A,1,&i,1,&j,&C[i][colour],mode);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode RecoverJacobianLocal(Mat A,InsertMode mode,PetscInt m,PetscInt p,PetscScalar **R,PetscScalar **C,PetscReal *a)
{
  PetscErrorCode ierr;
  PetscInt       i,j,colour;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    for (colour=0; colour<p; colour++) {
      j = (PetscInt) R[i][colour];
      if (j != -1) {
        if (a)
          C[i][colour] *= *a;
        ierr = MatSetValuesLocal(A,1,&i,1,&j,&C[i][colour],mode);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  Recover the diagonal of the Jacobian from its compressed matrix format

  Input parameters:
  mode - use INSERT_VALUES or ADD_VALUES, as required
  m    - number of rows of matrix.
  R    - recovery vector to use in the decompression procedure
  C    - compressed matrix to recover values from
  a    - shift value for implicit problems (select NULL or unity for explicit problems)

  Output parameter:
  diag - Vec to be populated with values from compressed matrix
*/
PetscErrorCode RecoverDiagonalLocal(Vec diag,InsertMode mode,PetscInt m,PetscScalar *R,PetscScalar **C,PetscReal *a)
{
  PetscErrorCode ierr;
  PetscInt       i,colour;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    colour = (PetscInt)R[i];
    if (a)
      C[i][colour] *= *a;
    ierr = VecSetValuesLocal(diag,1,&i,&C[i][colour],mode);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
