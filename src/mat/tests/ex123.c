static char help[] = "Test MatSetPreallocationCOO and MatSetValuesCOO\n\n";

#include <petscmat.h>
#define MyMatView(a,b) (PetscPrintf(PetscObjectComm((PetscObject)(a)),"LINE %d\n",__LINE__),MatView(a,b));
#define MyVecView(a,b) (PetscPrintf(PetscObjectComm((PetscObject)(a)),"LINE %d\n",__LINE__),VecView(a,b));
int main(int argc,char **args)
{
  Mat                    A,At,AAt;
  Vec                    x,y,z;
  ISLocalToGlobalMapping rl2g,cl2g;
  IS                     is;
  PetscLayout            rmap,cmap;
  PetscInt               *it,*jt;
  PetscInt               n1 = 11, n2 = 9;
  PetscInt               i1[] = {   7,  6,  2,  0,  4,  1,  1,  0,  2,  2,  1 , -1, -1};
  PetscInt               j1[] = {   1,  4,  3,  5,  3,  3,  4,  5,  0,  3,  1 , -1, -1};
  PetscInt               i2[] = {   7,  6,  2,  0,  4,  1,  1,  2,  1, -1, -1};
  PetscInt               j2[] = {   1,  4,  3,  5,  3,  3,  4,  0,  1, -1, -1};
  PetscScalar            v1[] = { -1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscScalar            v2[] = {  1.,-1.,-2.,-3.,-4.,-5.,-6.,-7.,-8.,-9.,-10., PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscInt               N = 6, m = 8, M, rstart, cstart, i;
  PetscMPIInt            size;
  PetscBool              loc = PETSC_FALSE;
  PetscBool              locdiag = PETSC_TRUE;
  PetscBool              localapi = PETSC_FALSE;
  PetscBool              neg = PETSC_FALSE;
  PetscBool              ismatis, ismpiaij;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-neg",&neg,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-loc",&loc,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-locdiag",&locdiag,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-localapi",&localapi,NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  if (loc) {
    if (locdiag) {
      PetscCall(MatSetSizes(A,m,N,PETSC_DECIDE,PETSC_DECIDE));
    } else {
      PetscCall(MatSetSizes(A,m,m+N,PETSC_DECIDE,PETSC_DECIDE));
    }
  } else {
    PetscCall(MatSetSizes(A,m,PETSC_DECIDE,PETSC_DECIDE,N));
  }
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatGetLayouts(A,&rmap,&cmap));
  PetscCall(PetscLayoutSetUp(rmap));
  PetscCall(PetscLayoutSetUp(cmap));
  PetscCall(PetscLayoutGetRange(rmap,&rstart,NULL));
  PetscCall(PetscLayoutGetRange(cmap,&cstart,NULL));
  PetscCall(PetscLayoutGetSize(rmap,&M));
  PetscCall(PetscLayoutGetSize(cmap,&N));

  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATIS,&ismatis));

  /* create fake l2g maps to test the local API */
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,M-rstart,rstart,1,&is));
  PetscCall(ISLocalToGlobalMappingCreateIS(is,&rl2g));
  PetscCall(ISDestroy(&is));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,N,0,1,&is));
  PetscCall(ISLocalToGlobalMappingCreateIS(is,&cl2g));
  PetscCall(ISDestroy(&is));
  PetscCall(MatSetLocalToGlobalMapping(A,rl2g,cl2g));
  PetscCall(ISLocalToGlobalMappingDestroy(&rl2g));
  PetscCall(ISLocalToGlobalMappingDestroy(&cl2g));

  PetscCall(MatCreateVecs(A,&x,&y));
  PetscCall(MatCreateVecs(A,NULL,&z));
  PetscCall(VecSet(x,1.));
  PetscCall(VecSet(z,2.));
  if (!localapi) for (i = 0; i < n1; i++) i1[i] += rstart;
  if (!localapi) for (i = 0; i < n2; i++) i2[i] += rstart;
  if (loc) {
    if (locdiag) {
      for (i = 0; i < n1; i++) j1[i] += cstart;
      for (i = 0; i < n2; i++) j2[i] += cstart;
    } else {
      for (i = 0; i < n1; i++) j1[i] += cstart + m;
      for (i = 0; i < n2; i++) j2[i] += cstart + m;
    }
  }
  if (neg) { n1 += 2; n2 += 2; }
  /* MatSetPreallocationCOOLocal maps the indices! */
  PetscCall(PetscMalloc2(PetscMax(n1,n2),&it,PetscMax(n1,n2),&jt));
  /* test with repeated entries */
  if (!localapi) {
    PetscCall(MatSetPreallocationCOO(A,n1,i1,j1));
  } else {
    PetscCall(PetscArraycpy(it,i1,n1));
    PetscCall(PetscArraycpy(jt,j1,n1));
    PetscCall(MatSetPreallocationCOOLocal(A,n1,it,jt));
  }
  PetscCall(MatSetValuesCOO(A,v1,ADD_VALUES));
  PetscCall(MatMult(A,x,y));
  PetscCall(MyMatView(A,NULL));
  PetscCall(MyVecView(y,NULL));
  PetscCall(MatSetValuesCOO(A,v2,ADD_VALUES));
  PetscCall(MatMultAdd(A,x,y,y));
  PetscCall(MyMatView(A,NULL));
  PetscCall(MyVecView(y,NULL));
  PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX,&At));
  if (!ismatis) {
    PetscCall(MatMatMult(A,At,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AAt));
    PetscCall(MyMatView(AAt,NULL));
    PetscCall(MatDestroy(&AAt));
    PetscCall(MatMatMult(At,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AAt));
    PetscCall(MyMatView(AAt,NULL));
    PetscCall(MatDestroy(&AAt));
  }
  PetscCall(MatDestroy(&At));

  /* INSERT_VALUES will overwrite matrix entries but
     still perform the sum of the repeated entries */
  PetscCall(MatSetValuesCOO(A,v2,INSERT_VALUES));
  PetscCall(MyMatView(A,NULL));

  /* test with unique entries */
  if (!localapi) {
    PetscCall(MatSetPreallocationCOO(A,n2,i2,j2));
  } else {
    PetscCall(PetscArraycpy(it,i2,n2));
    PetscCall(PetscArraycpy(jt,j2,n2));
    PetscCall(MatSetPreallocationCOOLocal(A,n2,it,jt));
  }
  PetscCall(MatSetValuesCOO(A,v1,ADD_VALUES));
  PetscCall(MatMult(A,x,y));
  PetscCall(MyMatView(A,NULL));
  PetscCall(MyVecView(y,NULL));
  PetscCall(MatSetValuesCOO(A,v2,ADD_VALUES));
  PetscCall(MatMultAdd(A,x,y,z));
  PetscCall(MyMatView(A,NULL));
  PetscCall(MyVecView(z,NULL));
  if (!localapi) {
    PetscCall(MatSetPreallocationCOO(A,n2,i2,j2));
  } else {
    PetscCall(PetscArraycpy(it,i2,n2));
    PetscCall(PetscArraycpy(jt,j2,n2));
    PetscCall(MatSetPreallocationCOOLocal(A,n2,it,jt));
  }
  PetscCall(MatSetValuesCOO(A,v1,INSERT_VALUES));
  PetscCall(MatMult(A,x,y));
  PetscCall(MyMatView(A,NULL));
  PetscCall(MyVecView(y,NULL));
  PetscCall(MatSetValuesCOO(A,v2,INSERT_VALUES));
  PetscCall(MatMultAdd(A,x,y,z));
  PetscCall(MyMatView(A,NULL));
  PetscCall(MyVecView(z,NULL));
  PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX,&At));
  if (!ismatis) {
    PetscCall(MatMatMult(A,At,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AAt));
    PetscCall(MyMatView(AAt,NULL));
    PetscCall(MatDestroy(&AAt));
    PetscCall(MatMatMult(At,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AAt));
    PetscCall(MyMatView(AAt,NULL));
    PetscCall(MatDestroy(&AAt));
  }
  PetscCall(MatDestroy(&At));

  /* test providing diagonal first, then offdiagonal */
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATMPIAIJ,&ismpiaij));
  if (ismpiaij && size > 1) {
    Mat               lA,lB;
    const PetscInt    *garray,*iA,*jA,*iB,*jB;
    const PetscScalar *vA,*vB;
    PetscScalar       *coo_v;
    PetscInt          *coo_i,*coo_j;
    PetscInt          i,j,nA,nB,nnz;
    PetscBool         flg;

    PetscCall(MatMPIAIJGetSeqAIJ(A,&lA,&lB,&garray));
    PetscCall(MatSeqAIJGetArrayRead(lA,&vA));
    PetscCall(MatSeqAIJGetArrayRead(lB,&vB));
    PetscCall(MatGetRowIJ(lA,0,PETSC_FALSE,PETSC_FALSE,&nA,&iA,&jA,&flg));
    PetscCall(MatGetRowIJ(lB,0,PETSC_FALSE,PETSC_FALSE,&nB,&iB,&jB,&flg));
    nnz  = iA[nA] + iB[nB];
    PetscCall(PetscMalloc3(nnz,&coo_i,nnz,&coo_j,nnz,&coo_v));
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
    PetscCall(MatRestoreRowIJ(lA,0,PETSC_FALSE,PETSC_FALSE,&nA,&iA,&jA,&flg));
    PetscCall(MatRestoreRowIJ(lB,0,PETSC_FALSE,PETSC_FALSE,&nB,&iB,&jB,&flg));
    PetscCall(MatSeqAIJRestoreArrayRead(lA,&vA));
    PetscCall(MatSeqAIJRestoreArrayRead(lB,&vB));

    PetscCall(MatSetPreallocationCOO(A,nnz,coo_i,coo_j));
    PetscCall(MatSetValuesCOO(A,coo_v,ADD_VALUES));
    PetscCall(MatMult(A,x,y));
    PetscCall(MyMatView(A,NULL));
    PetscCall(MyVecView(y,NULL));
    PetscCall(MatSetValuesCOO(A,coo_v,INSERT_VALUES));
    PetscCall(MatMult(A,x,y));
    PetscCall(MyMatView(A,NULL));
    PetscCall(MyVecView(y,NULL));
    PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX,&At));
    PetscCall(MatMatMult(A,At,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AAt));
    PetscCall(MyMatView(AAt,NULL));
    PetscCall(MatDestroy(&AAt));
    PetscCall(MatMatMult(At,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AAt));
    PetscCall(MyMatView(AAt,NULL));
    PetscCall(MatDestroy(&AAt));
    PetscCall(MatDestroy(&At));

    PetscCall(PetscFree3(coo_i,coo_j,coo_v));
  }
  PetscCall(PetscFree2(it,jt));
  PetscCall(VecDestroy(&z));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     suffix: 1
     filter: grep -v type
     diff_args: -j
     args: -mat_type {{seqaij mpiaij}} -localapi {{0 1}} -neg {{0 1}}

   test:
     requires: cuda
     suffix: 1_cuda
     filter: grep -v type
     diff_args: -j
     args: -mat_type {{seqaijcusparse mpiaijcusparse}} -localapi {{0 1}} -neg {{0 1}}
     output_file: output/ex123_1.out

   test:
     requires: kokkos_kernels !sycl
     suffix: 1_kokkos
     filter: grep -v type
     diff_args: -j
     args: -mat_type {{seqaijkokkos mpiaijkokkos}} -localapi {{0 1}} -neg {{0 1}}
     output_file: output/ex123_1.out

   test:
     suffix: 2
     nsize: 7
     filter: grep -v type
     diff_args: -j
     args: -mat_type mpiaij -localapi {{0 1}} -neg {{0 1}}

   test:
     requires: cuda
     suffix: 2_cuda
     nsize: 7
     filter: grep -v type
     diff_args: -j
     args: -mat_type mpiaijcusparse -localapi {{0 1}} -neg {{0 1}}
     output_file: output/ex123_2.out

   test:
     requires: kokkos_kernels !sycl
     suffix: 2_kokkos
     nsize: 7
     filter: grep -v type
     diff_args: -j
     args: -mat_type mpiaijkokkos -localapi {{0 1}} -neg {{0 1}}
     output_file: output/ex123_2.out

   test:
     suffix: 3
     nsize: 3
     filter: grep -v type
     diff_args: -j
     args: -mat_type mpiaij -loc -localapi {{0 1}} -neg {{0 1}}

   test:
     requires: cuda
     suffix: 3_cuda
     nsize: 3
     filter: grep -v type
     diff_args: -j
     args: -mat_type mpiaijcusparse -loc -localapi {{0 1}} -neg {{0 1}}
     output_file: output/ex123_3.out

   test:
     requires: !sycl kokkos_kernels
     suffix: 3_kokkos
     nsize: 3
     filter: grep -v type
     diff_args: -j
     args: -mat_type aijkokkos -loc -localapi {{0 1}} -neg {{0 1}}
     output_file: output/ex123_3.out

   test:
     suffix: 4
     nsize: 4
     filter: grep -v type
     diff_args: -j
     args: -mat_type mpiaij -loc -locdiag 0 -localapi {{0 1}} -neg {{0 1}}

   test:
     requires: cuda
     suffix: 4_cuda
     nsize: 4
     filter: grep -v type
     diff_args: -j
     args: -mat_type mpiaijcusparse -loc -locdiag 0 -localapi {{0 1}} -neg {{0 1}}
     output_file: output/ex123_4.out

   test:
     requires: !sycl kokkos_kernels
     suffix: 4_kokkos
     nsize: 4
     filter: grep -v type
     diff_args: -j
     args: -mat_type aijkokkos -loc -locdiag 0 -localapi {{0 1}} -neg {{0 1}}
     output_file: output/ex123_4.out

   test:
     suffix: matis
     nsize: 3
     filter: grep -v type
     diff_args: -j
     args: -mat_type is -localapi {{0 1}} -neg {{0 1}}

TEST*/
