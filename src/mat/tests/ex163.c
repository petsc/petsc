
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
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /* Determine file from which we read the matrix A */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");

  /* Load matrix A */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Print (for testing only) */
  ierr = PetscOptionsHasName(NULL,NULL, "-view_mats", &viewmats);CHKERRQ(ierr);
  if (viewmats) {
    if (!rank) printf("A_aij:\n");
    ierr = MatView(A,0);CHKERRQ(ierr);
  }

  /* Test MatTransposeMatMult_aij_aij() */
  ierr = MatTransposeMatMult(A,A,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
  if (viewmats) {
    if (!rank) printf("\nC = A_aij^T * A_aij:\n");
    ierr = MatView(C,0);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);

  /* create a dense matrix Bdense */
  ierr = MatCreate(PETSC_COMM_WORLD,&Bdense);CHKERRQ(ierr);
  ierr = MatSetSizes(Bdense,m,PETSC_DECIDE,PETSC_DECIDE,BN);CHKERRQ(ierr);
  ierr = MatSetType(Bdense,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Bdense);CHKERRQ(ierr);
  ierr = MatSetUp(Bdense);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Bdense,&rstart,&rend);CHKERRQ(ierr);

  ierr = PetscMalloc3(m,&rows,BN,&cols,m*BN,&array);CHKERRQ(ierr);
  for (i=0; i<m; i++) rows[i] = rstart + i;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  for (j=0; j<BN; j++) {
    cols[j] = j;
    for (i=0; i<m; i++) {
      ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
      array[m*j+i] = rval;
    }
  }
  ierr = MatSetValues(Bdense,m,rows,BN,cols,array,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Bdense,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Bdense,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFree3(rows,cols,array);CHKERRQ(ierr);
  if (viewmats) {
    if (!rank) printf("\nBdense:\n");
    ierr = MatView(Bdense,0);CHKERRQ(ierr);
  }

  /* Test MatTransposeMatMult_aij_dense() */
  ierr = MatTransposeMatMult(A,Bdense,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
  ierr = MatTransposeMatMult(A,Bdense,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
  if (viewmats) {
    if (!rank) printf("\nC=A^T*Bdense:\n");
    ierr = MatView(C,0);CHKERRQ(ierr);
  }

  /* Check accuracy */
  ierr = MatCreate(PETSC_COMM_WORLD,&Cdense);CHKERRQ(ierr);
  ierr = MatSetSizes(Cdense,n,PETSC_DECIDE,PETSC_DECIDE,BN);CHKERRQ(ierr);
  ierr = MatSetType(Cdense,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Cdense);CHKERRQ(ierr);
  ierr = MatSetUp(Cdense);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Cdense,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Cdense,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size == 1) {
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,m,NULL,&x);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,n,NULL,&y);CHKERRQ(ierr);
  } else {
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,m,PETSC_DECIDE,NULL,&x);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,NULL,&y);CHKERRQ(ierr);
  }

  /* Cdense[:,j] = A^T * Bdense[:,j] */
  ierr = MatDenseGetArray(Bdense,&Barray);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Cdense,&Carray);CHKERRQ(ierr);
  for (j=0; j<BN; j++) {
    ierr = VecPlaceArray(x,Barray);CHKERRQ(ierr);
    ierr = VecPlaceArray(y,Carray);CHKERRQ(ierr);

    ierr = MatMultTranspose(A,x,y);CHKERRQ(ierr);

    ierr = VecResetArray(x);CHKERRQ(ierr);
    ierr = VecResetArray(y);CHKERRQ(ierr);
    Barray += m;
    Carray += n;
  }
  ierr = MatDenseRestoreArray(Bdense,&Barray);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(Cdense,&Carray);CHKERRQ(ierr);
  if (viewmats) {
    if (!rank) printf("\nCdense:\n");
    ierr = MatView(Cdense,0);CHKERRQ(ierr);
  }

  ierr = MatEqual(C,Cdense,&flg);CHKERRQ(ierr);
  if (!flg) {
    if (!rank) printf(" C != Cdense\n");
  }

  /* Free data structures */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&Bdense);CHKERRQ(ierr);
  ierr = MatDestroy(&Cdense);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex163.out

   test:
      suffix: 2
      nsize: 3
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex163.out

TEST*/
