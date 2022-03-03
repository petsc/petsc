
static char help[] = "Tests MatTransposeMatMult() on MatLoad() matrix \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,C,Bdense,Cdense;
  PetscErrorCode ierr;
  PetscViewer    fd;              /* viewer */
  char           file[PETSC_MAX_PATH_LEN]; /* input file name */
  PetscBool      flg,viewmats=PETSC_FALSE;
  PetscMPIInt    rank,size;
  PetscReal      fill=1.0;
  PetscInt       m,n,i,j,BN=10,rstart,rend,*rows,*cols;
  PetscScalar    *Barray,*Carray,rval,*array;
  Vec            x,y;
  PetscRandom    rand;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* Determine file from which we read the matrix A */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");

  /* Load matrix A */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  /* Print (for testing only) */
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-view_mats", &viewmats));
  if (viewmats) {
    if (rank == 0) printf("A_aij:\n");
    CHKERRQ(MatView(A,0));
  }

  /* Test MatTransposeMatMult_aij_aij() */
  CHKERRQ(MatTransposeMatMult(A,A,MAT_INITIAL_MATRIX,fill,&C));
  if (viewmats) {
    if (rank == 0) printf("\nC = A_aij^T * A_aij:\n");
    CHKERRQ(MatView(C,0));
  }
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatGetLocalSize(A,&m,&n));

  /* create a dense matrix Bdense */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&Bdense));
  CHKERRQ(MatSetSizes(Bdense,m,PETSC_DECIDE,PETSC_DECIDE,BN));
  CHKERRQ(MatSetType(Bdense,MATDENSE));
  CHKERRQ(MatSetFromOptions(Bdense));
  CHKERRQ(MatSetUp(Bdense));
  CHKERRQ(MatGetOwnershipRange(Bdense,&rstart,&rend));

  CHKERRQ(PetscMalloc3(m,&rows,BN,&cols,m*BN,&array));
  for (i=0; i<m; i++) rows[i] = rstart + i;
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));
  for (j=0; j<BN; j++) {
    cols[j] = j;
    for (i=0; i<m; i++) {
      CHKERRQ(PetscRandomGetValue(rand,&rval));
      array[m*j+i] = rval;
    }
  }
  CHKERRQ(MatSetValues(Bdense,m,rows,BN,cols,array,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(Bdense,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Bdense,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscRandomDestroy(&rand));
  CHKERRQ(PetscFree3(rows,cols,array));
  if (viewmats) {
    if (rank == 0) printf("\nBdense:\n");
    CHKERRQ(MatView(Bdense,0));
  }

  /* Test MatTransposeMatMult_aij_dense() */
  CHKERRQ(MatTransposeMatMult(A,Bdense,MAT_INITIAL_MATRIX,fill,&C));
  CHKERRQ(MatTransposeMatMult(A,Bdense,MAT_REUSE_MATRIX,fill,&C));
  if (viewmats) {
    if (rank == 0) printf("\nC=A^T*Bdense:\n");
    CHKERRQ(MatView(C,0));
  }

  /* Check accuracy */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&Cdense));
  CHKERRQ(MatSetSizes(Cdense,n,PETSC_DECIDE,PETSC_DECIDE,BN));
  CHKERRQ(MatSetType(Cdense,MATDENSE));
  CHKERRQ(MatSetFromOptions(Cdense));
  CHKERRQ(MatSetUp(Cdense));
  CHKERRQ(MatAssemblyBegin(Cdense,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Cdense,MAT_FINAL_ASSEMBLY));

  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  if (size == 1) {
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,m,NULL,&x));
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,n,NULL,&y));
  } else {
    CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,m,PETSC_DECIDE,NULL,&x));
    CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,NULL,&y));
  }

  /* Cdense[:,j] = A^T * Bdense[:,j] */
  CHKERRQ(MatDenseGetArray(Bdense,&Barray));
  CHKERRQ(MatDenseGetArray(Cdense,&Carray));
  for (j=0; j<BN; j++) {
    CHKERRQ(VecPlaceArray(x,Barray));
    CHKERRQ(VecPlaceArray(y,Carray));

    CHKERRQ(MatMultTranspose(A,x,y));

    CHKERRQ(VecResetArray(x));
    CHKERRQ(VecResetArray(y));
    Barray += m;
    Carray += n;
  }
  CHKERRQ(MatDenseRestoreArray(Bdense,&Barray));
  CHKERRQ(MatDenseRestoreArray(Cdense,&Carray));
  if (viewmats) {
    if (rank == 0) printf("\nCdense:\n");
    CHKERRQ(MatView(Cdense,0));
  }

  CHKERRQ(MatEqual(C,Cdense,&flg));
  if (!flg) {
    if (rank == 0) printf(" C != Cdense\n");
  }

  /* Free data structures */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&Bdense));
  CHKERRQ(MatDestroy(&Cdense));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex163.out

   test:
      suffix: 2
      nsize: 3
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex163.out

TEST*/
