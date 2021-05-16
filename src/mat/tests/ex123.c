static char help[] = "Test MatSetPreallocationCOO and MatSetValuesCOO\n\n";

#include <petscmat.h>
#define MyMatView(a,b) PetscPrintf(PetscObjectComm((PetscObject)(a)),"LINE %d\n",__LINE__),MatView(a,b);
#define MyVecView(a,b) PetscPrintf(PetscObjectComm((PetscObject)(a)),"LINE %d\n",__LINE__),VecView(a,b);
int main(int argc,char **args)
{
  Mat            A,At,AAt;
  Vec            x,y,z;
  PetscLayout    rmap,cmap;
  PetscInt       n1 = 11, n2 = 9;
  PetscInt       i1[] = {   7,  6,  2,  0,  4,  1,  1,  0,  2,  2,  1 };
  PetscInt       j1[] = {   1,  4,  3,  5,  3,  3,  4,  5,  0,  3,  1 };
  PetscInt       i2[] = {   7,  6,  2,  0,  4,  1,  1,  2, 1 };
  PetscInt       j2[] = {   1,  4,  3,  5,  3,  3,  4,  0, 1 };
  PetscScalar    v1[] = { -1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  PetscScalar    v2[] = {  1.,-1.,-2.,-3.,-4.,-5.,-6.,-7.,-8.,-9.,-10.};
  PetscInt       N = 6, m = 8, rstart, cstart, i;
  PetscMPIInt    size;
  PetscBool      loc = PETSC_FALSE;
  PetscBool      locdiag = PETSC_TRUE, ismpiaij;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetBool(NULL,NULL,"-loc",&loc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-locdiag",&locdiag,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  if (loc) {
    if (locdiag) {
      ierr = MatSetSizes(A,m,N,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(A,m,m+N,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    }
  } else {
    ierr = MatSetSizes(A,m,PETSC_DECIDE,PETSC_DECIDE,N);CHKERRQ(ierr);
  }
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatGetLayouts(A,&rmap,&cmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(cmap);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,NULL,&z);CHKERRQ(ierr);
  ierr = VecSet(x,1.);CHKERRQ(ierr);
  ierr = VecSet(z,2.);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(rmap,&rstart,NULL);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(cmap,&cstart,NULL);CHKERRQ(ierr);
  for (i = 0; i < n1; i++) i1[i] += rstart;
  for (i = 0; i < n2; i++) i2[i] += rstart;
  if (loc) {
    if (locdiag) {
      for (i = 0; i < n1; i++) j1[i] += cstart;
      for (i = 0; i < n2; i++) j2[i] += cstart;
    } else {
      for (i = 0; i < n1; i++) j1[i] += cstart + m;
      for (i = 0; i < n2; i++) j2[i] += cstart + m;
    }
  }

  /* test with repeated entries */
  ierr = MatSetPreallocationCOO(A,n1,i1,j1);CHKERRQ(ierr);
  ierr = MatSetValuesCOO(A,v1,ADD_VALUES);CHKERRQ(ierr);
  ierr = MyMatView(A,NULL);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  ierr = MyVecView(y,NULL);CHKERRQ(ierr);
  ierr = MatSetValuesCOO(A,v2,ADD_VALUES);CHKERRQ(ierr);
  ierr = MyMatView(A,NULL);CHKERRQ(ierr);
  ierr = MatMultAdd(A,x,y,y);CHKERRQ(ierr);
  ierr = MyVecView(y,NULL);CHKERRQ(ierr);
  ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&At);CHKERRQ(ierr);
  ierr = MatMatMult(A,At,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AAt);CHKERRQ(ierr);
  ierr = MyMatView(AAt,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&AAt);CHKERRQ(ierr);
  ierr = MatMatMult(At,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AAt);CHKERRQ(ierr);
  ierr = MyMatView(AAt,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&AAt);CHKERRQ(ierr);
  ierr = MatDestroy(&At);CHKERRQ(ierr);
  /* INSERT_VALUES will overwrite matrix entries but
     still perform the sum of the repeated entries */
  ierr = MatSetValuesCOO(A,v2,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MyMatView(A,NULL);CHKERRQ(ierr);

  /* test with unique entries */
  ierr = MatSetPreallocationCOO(A,n2,i2,j2);CHKERRQ(ierr);
  ierr = MatSetValuesCOO(A,v1,ADD_VALUES);CHKERRQ(ierr);
  ierr = MyMatView(A,NULL);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  ierr = MyVecView(y,NULL);CHKERRQ(ierr);
  ierr = MatSetValuesCOO(A,v2,ADD_VALUES);CHKERRQ(ierr);
  ierr = MyMatView(A,NULL);CHKERRQ(ierr);
  ierr = MatMultAdd(A,x,y,z);CHKERRQ(ierr);
  ierr = MyVecView(z,NULL);CHKERRQ(ierr);
  ierr = MatSetPreallocationCOO(A,n2,i2,j2);CHKERRQ(ierr);
  ierr = MatSetValuesCOO(A,v1,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MyMatView(A,NULL);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  ierr = MyVecView(y,NULL);CHKERRQ(ierr);
  ierr = MatSetValuesCOO(A,v2,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MyMatView(A,NULL);CHKERRQ(ierr);
  ierr = MatMultAdd(A,x,y,z);CHKERRQ(ierr);
  ierr = MyVecView(z,NULL);CHKERRQ(ierr);
  ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&At);CHKERRQ(ierr);
  ierr = MatMatMult(A,At,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AAt);CHKERRQ(ierr);
  ierr = MyMatView(AAt,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&AAt);CHKERRQ(ierr);
  ierr = MatMatMult(At,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AAt);CHKERRQ(ierr);
  ierr = MyMatView(AAt,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&AAt);CHKERRQ(ierr);
  ierr = MatDestroy(&At);CHKERRQ(ierr);

  /* test providing diagonal first, the offdiagonal */
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
  if (ismpiaij && size > 1) {
    Mat               lA,lB;
    const PetscInt    *garray,*iA,*jA,*iB,*jB;
    const PetscScalar *vA,*vB;
    PetscScalar       *coo_v;
    PetscInt          *coo_i,*coo_j;
    PetscInt          i,j,nA,nB,nnz;
    PetscBool         flg;

    ierr = MatMPIAIJGetSeqAIJ(A,&lA,&lB,&garray);CHKERRQ(ierr);
    ierr = MatSeqAIJGetArrayRead(lA,&vA);CHKERRQ(ierr);
    ierr = MatSeqAIJGetArrayRead(lB,&vB);CHKERRQ(ierr);
    ierr = MatGetRowIJ(lA,0,PETSC_FALSE,PETSC_FALSE,&nA,&iA,&jA,&flg);CHKERRQ(ierr);
    ierr = MatGetRowIJ(lB,0,PETSC_FALSE,PETSC_FALSE,&nB,&iB,&jB,&flg);CHKERRQ(ierr);
    nnz  = iA[nA] + iB[nB];
    ierr = PetscMalloc3(nnz,&coo_i,nnz,&coo_j,nnz,&coo_v);CHKERRQ(ierr);
    nnz  = 0;
    for (i=0;i<nA;i++) {
      for (j=iA[i];j<iA[i+1];j++,nnz++) {
        coo_i[nnz] = i+rstart;
        coo_j[nnz] = jA[j]+cstart;
        coo_v[nnz] = vA[j];
      }
    }
    for (i=0;i<nB;i++) {
      for (j=iB[i];j<iB[i+1];j++,nnz++) {
        coo_i[nnz] = i+rstart;
        coo_j[nnz] = garray[jB[j]];
        coo_v[nnz] = vB[j];
      }
    }
    ierr = MatRestoreRowIJ(lA,0,PETSC_FALSE,PETSC_FALSE,&nA,&iA,&jA,&flg);CHKERRQ(ierr);
    ierr = MatRestoreRowIJ(lB,0,PETSC_FALSE,PETSC_FALSE,&nB,&iB,&jB,&flg);CHKERRQ(ierr);
    ierr = MatSeqAIJRestoreArrayRead(lA,&vA);CHKERRQ(ierr);
    ierr = MatSeqAIJRestoreArrayRead(lB,&vB);CHKERRQ(ierr);

    ierr = MatSetPreallocationCOO(A,nnz,coo_i,coo_j);CHKERRQ(ierr);
    ierr = MatSetValuesCOO(A,coo_v,ADD_VALUES);CHKERRQ(ierr);
    ierr = MyMatView(A,NULL);CHKERRQ(ierr);
    ierr = MatMult(A,x,y);CHKERRQ(ierr);
    ierr = MyVecView(y,NULL);CHKERRQ(ierr);
    ierr = MatSetValuesCOO(A,coo_v,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MyMatView(A,NULL);CHKERRQ(ierr);
    ierr = MatMult(A,x,y);CHKERRQ(ierr);
    ierr = MyVecView(y,NULL);CHKERRQ(ierr);
    ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&At);CHKERRQ(ierr);
    ierr = MatMatMult(A,At,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AAt);CHKERRQ(ierr);
    ierr = MyMatView(AAt,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&AAt);CHKERRQ(ierr);
    ierr = MatMatMult(At,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AAt);CHKERRQ(ierr);
    ierr = MyMatView(AAt,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&AAt);CHKERRQ(ierr);
    ierr = MatDestroy(&At);CHKERRQ(ierr);

    ierr = PetscFree3(coo_i,coo_j,coo_v);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     suffix: 1
     filter: grep -v type
     diff_args: -j
     args: -mat_type {{seqaij mpiaij}}

   test:
     requires: cuda
     suffix: 1_cuda
     filter: grep -v type
     diff_args: -j
     args: -mat_type {{seqaijcusparse mpiaijcusparse}}
     output_file: output/ex123_1.out

   test:
     suffix: 2
     nsize: 7
     filter: grep -v type
     diff_args: -j
     args: -mat_type mpiaij

   test:
     requires: cuda
     suffix: 2_cuda
     nsize: 7
     filter: grep -v type
     diff_args: -j
     args: -mat_type mpiaijcusparse
     output_file: output/ex123_2.out

   test:
     suffix: 3
     nsize: 3
     filter: grep -v type
     diff_args: -j
     args: -mat_type mpiaij -loc

   test:
     requires: cuda
     suffix: 3_cuda
     nsize: 3
     filter: grep -v type
     diff_args: -j
     args: -mat_type mpiaijcusparse -loc
     output_file: output/ex123_3.out

   test:
     suffix: 4
     nsize: 4
     filter: grep -v type
     diff_args: -j
     args: -mat_type mpiaij -loc -locdiag 0

   test:
     requires: cuda
     suffix: 4_cuda
     nsize: 4
     filter: grep -v type
     diff_args: -j
     args: -mat_type mpiaijcusparse -loc -locdiag 0
     output_file: output/ex123_4.out

TEST*/
