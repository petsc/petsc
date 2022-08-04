static char help[] = "Tests MATHTOOL\n\n";

#include <petscmat.h>

static PetscErrorCode GenEntries(PetscInt sdim,PetscInt M,PetscInt N,const PetscInt *J,const PetscInt *K,PetscScalar *ptr,void *ctx)
{
  PetscInt  d,j,k;
  PetscReal diff = 0.0,*coords = (PetscReal*)(ctx);

  PetscFunctionBeginUser;
  for (j = 0; j < M; j++) {
    for (k = 0; k < N; k++) {
      diff = 0.0;
      for (d = 0; d < sdim; d++) diff += (coords[J[j]*sdim+d] - coords[K[k]*sdim+d]) * (coords[J[j]*sdim+d] - coords[K[k]*sdim+d]);
      ptr[j+M*k] = 1.0/(1.0e-2 + PetscSqrtReal(diff));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GenEntriesRectangular(PetscInt sdim,PetscInt M,PetscInt N,const PetscInt *J,const PetscInt *K,PetscScalar *ptr,void *ctx)
{
  PetscInt  d,j,k;
  PetscReal diff = 0.0,**coords = (PetscReal**)(ctx);

  PetscFunctionBeginUser;
  for (j = 0; j < M; j++) {
    for (k = 0; k < N; k++) {
      diff = 0.0;
      for (d = 0; d < sdim; d++) diff += (coords[0][J[j]*sdim+d] - coords[1][K[k]*sdim+d]) * (coords[0][J[j]*sdim+d] - coords[1][K[k]*sdim+d]);
      ptr[j+M*k] = 1.0/(1.0e-2 + PetscSqrtReal(diff));
    }
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            A,AT,D,B,P,R,RT;
  PetscInt       m = 100,dim = 3,M,K = 10,begin,n = 0,N,bs;
  PetscMPIInt    size;
  PetscScalar    *ptr;
  PetscReal      *coords,*gcoords,*scoords,*gscoords,*(ctx[2]),norm,epsilon;
  MatHtoolKernel kernel = GenEntries;
  PetscBool      flg,sym = PETSC_FALSE;
  PetscRandom    rdm;
  IS             iss,ist,is[2];
  Vec            right,left,perm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)NULL,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m_local",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n_local",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-K",&K,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-symmetric",&sym,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-mat_htool_epsilon",&epsilon,NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  M = size*m;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscMalloc1(m*dim,&coords));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  PetscCall(PetscRandomGetValuesReal(rdm,m*dim,coords));
  PetscCall(PetscCalloc1(M*dim,&gcoords));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,M,K,NULL,&B));
  PetscCall(MatSetRandom(B,rdm));
  PetscCall(MatGetOwnershipRange(B,&begin,NULL));
  PetscCall(PetscArraycpy(gcoords+begin*dim,coords,m*dim));
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE,gcoords,M*dim,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,m,M,M,dim,coords,coords,kernel,gcoords,&A));
  PetscCall(MatSetOption(A,MAT_SYMMETRIC,sym));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(A,NULL,"-A_view"));
  PetscCall(MatGetOwnershipIS(A,is,NULL));
  PetscCall(ISDuplicate(is[0],is+1));
  PetscCall(MatIncreaseOverlap(A,1,is,2));
  PetscCall(MatSetBlockSize(A,2));
  PetscCall(MatIncreaseOverlap(A,1,is+1,1));
  PetscCall(ISGetBlockSize(is[1],&bs));
  PetscCheck(bs == 2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect block size %" PetscInt_FMT " != 2",bs);
  PetscCall(MatSetBlockSize(A,1));
  PetscCall(ISEqual(is[0],is[1],&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unequal index sets");
  PetscCall(ISDestroy(is));
  PetscCall(ISDestroy(is+1));
  PetscCall(MatCreateVecs(A,&right,&left));
  PetscCall(VecSetRandom(right,rdm));
  PetscCall(MatMult(A,right,left));
  PetscCall(MatHtoolGetPermutationSource(A,&iss));
  PetscCall(MatHtoolGetPermutationTarget(A,&ist));
  PetscCall(VecDuplicate(left,&perm));
  PetscCall(VecCopy(left,perm));
  PetscCall(VecPermute(perm,ist,PETSC_FALSE));
  PetscCall(VecPermute(right,iss,PETSC_FALSE));
  PetscCall(MatHtoolUsePermutation(A,PETSC_FALSE));
  PetscCall(MatMult(A,right,left));
  PetscCall(VecAXPY(left,-1.0,perm));
  PetscCall(VecNorm(left,NORM_INFINITY,&norm));
  PetscCheck(PetscAbsReal(norm) <= PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"||y(with permutation)-y(without permutation)|| = %g (> %g)",(double)PetscAbsReal(norm),(double)PETSC_SMALL);
  PetscCall(MatHtoolUsePermutation(A,PETSC_TRUE));
  PetscCall(VecDestroy(&perm));
  PetscCall(VecDestroy(&left));
  PetscCall(VecDestroy(&right));
  PetscCall(ISDestroy(&ist));
  PetscCall(ISDestroy(&iss));
  if (PetscAbsReal(epsilon) >= PETSC_SMALL) { /* when there is compression, it is more difficult to check against MATDENSE, so just compare symmetric and nonsymmetric assemblies */
    PetscReal relative;
    PetscCall(MatDestroy(&B));
    PetscCall(MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,m,M,M,dim,coords,coords,kernel,gcoords,&B));
    PetscCall(MatSetOption(B,MAT_SYMMETRIC,(PetscBool)!sym));
    PetscCall(MatSetFromOptions(B));
    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatViewFromOptions(B,NULL,"-B_view"));
    PetscCall(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&P));
    PetscCall(MatNorm(P,NORM_FROBENIUS,&relative));
    PetscCall(MatConvert(B,MATDENSE,MAT_INITIAL_MATRIX,&R));
    PetscCall(MatAXPY(R,-1.0,P,SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(R,NORM_INFINITY,&norm));
    PetscCheck(PetscAbsReal(norm/relative) <= epsilon,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"||A(!symmetric)-A(symmetric)|| = %g (> %g)",(double)PetscAbsReal(norm/relative),(double)epsilon);
    PetscCall(MatDestroy(&B));
    PetscCall(MatDestroy(&R));
    PetscCall(MatDestroy(&P));
  } else {
    PetscCall(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&D));
    PetscCall(MatViewFromOptions(D,NULL,"-D_view"));
    PetscCall(MatMultEqual(A,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Ax != Dx");
    PetscCall(MatMultTransposeEqual(A,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"A^Tx != D^Tx");
    PetscCall(MatMultAddEqual(A,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"y+Ax != y+Dx");
    PetscCall(MatGetOwnershipRange(B,&begin,NULL));
    PetscCall(MatDenseGetArrayWrite(D,&ptr));
    for (PetscInt i = begin; i < m+begin; ++i)
      for (PetscInt j = 0; j < M; ++j) GenEntries(dim,1,1,&i,&j,ptr+i-begin+j*m,gcoords);
    PetscCall(MatDenseRestoreArrayWrite(D,&ptr));
    PetscCall(MatMultEqual(A,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Ax != Dx");
    PetscCall(MatTranspose(D,MAT_INPLACE_MATRIX,&D));
    PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX,&AT));
    PetscCall(MatMultEqual(AT,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"A^Tx != D^Tx");
    PetscCall(MatTranspose(A,MAT_REUSE_MATRIX,&AT));
    PetscCall(MatMultEqual(AT,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"A^Tx != D^Tx");
    PetscCall(MatAXPY(D,-1.0,AT,SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(D,NORM_INFINITY,&norm));
    PetscCheck(PetscAbsReal(norm) <= PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"||A-D|| = %g (> %g)",(double)norm,(double)PETSC_SMALL);
    PetscCall(MatDestroy(&AT));
    PetscCall(MatDestroy(&D));
    PetscCall(MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&P));
    PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatMatMultEqual(A,B,P,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"ABx != Px");
    PetscCall(MatTransposeMatMultEqual(A,B,P,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"A^TBx != P^Tx");
    PetscCall(MatDestroy(&B));
    PetscCall(MatDestroy(&P));
    if (n) {
      PetscCall(PetscMalloc1(n*dim,&scoords));
      PetscCall(PetscRandomGetValuesReal(rdm,n*dim,scoords));
      N = n;
      PetscCall(MPIU_Allreduce(MPI_IN_PLACE,&N,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
      PetscCall(PetscCalloc1(N*dim,&gscoords));
      PetscCallMPI(MPI_Exscan(&n,&begin,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
      PetscCall(PetscArraycpy(gscoords+begin*dim,scoords,n*dim));
      PetscCall(MPIU_Allreduce(MPI_IN_PLACE,gscoords,N*dim,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD));
      kernel = GenEntriesRectangular;
      ctx[0] = gcoords;
      ctx[1] = gscoords;
      PetscCall(MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,n,M,N,dim,coords,scoords,kernel,ctx,&R));
      PetscCall(MatSetFromOptions(R));
      PetscCall(MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY));
      PetscCall(MatViewFromOptions(R,NULL,"-R_view"));
      PetscCall(MatConvert(R,MATDENSE,MAT_INITIAL_MATRIX,&D));
      PetscCall(MatViewFromOptions(D,NULL,"-D_view"));
      PetscCall(MatMultEqual(R,D,10,&flg));
      PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Rx != Dx");
      PetscCall(MatTranspose(D,MAT_INPLACE_MATRIX,&D));
      PetscCall(MatTranspose(R,MAT_INITIAL_MATRIX,&RT));
      PetscCall(MatMultEqual(RT,D,10,&flg));
      PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"R^Tx != D^Tx");
      PetscCall(MatTranspose(R,MAT_REUSE_MATRIX,&RT));
      PetscCall(MatMultEqual(RT,D,10,&flg));
      PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"R^Tx != D^Tx");
      PetscCall(MatDestroy(&RT));
      PetscCall(MatDestroy(&D));
      PetscCall(MatCreateDense(PETSC_COMM_WORLD,n,PETSC_DECIDE,PETSC_DETERMINE,K,NULL,&B));
      PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
      PetscCall(MatSetRandom(B,rdm));
      PetscCall(MatMatMult(R,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&P));
      PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
      PetscCall(MatMatMultEqual(R,B,P,10,&flg));
      PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"RBx != Px");
      PetscCall(MatDestroy(&B));
      PetscCall(MatDestroy(&P));
      PetscCall(MatCreateVecs(R,&right,&left));
      PetscCall(VecSetRandom(right,rdm));
      PetscCall(MatMult(R,right,left));
      PetscCall(MatHtoolGetPermutationSource(R,&iss));
      PetscCall(MatHtoolGetPermutationTarget(R,&ist));
      PetscCall(VecDuplicate(left,&perm));
      PetscCall(VecCopy(left,perm));
      PetscCall(VecPermute(perm,ist,PETSC_FALSE));
      PetscCall(VecPermute(right,iss,PETSC_FALSE));
      PetscCall(MatHtoolUsePermutation(R,PETSC_FALSE));
      PetscCall(MatMult(R,right,left));
      PetscCall(VecAXPY(left,-1.0,perm));
      PetscCall(VecNorm(left,NORM_INFINITY,&norm));
      PetscCheck(PetscAbsReal(norm) <= PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"||y(with permutation)-y(without permutation)|| = %g (> %g)",(double)PetscAbsReal(norm),(double)PETSC_SMALL);
      PetscCall(MatHtoolUsePermutation(R,PETSC_TRUE));
      PetscCall(VecDestroy(&perm));
      PetscCall(VecDestroy(&left));
      PetscCall(VecDestroy(&right));
      PetscCall(ISDestroy(&ist));
      PetscCall(ISDestroy(&iss));
      PetscCall(MatDestroy(&R));
      PetscCall(PetscFree(gscoords));
      PetscCall(PetscFree(scoords));
    }
  }
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFree(gcoords));
  PetscCall(PetscFree(coords));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: htool

   test:
      requires: htool
      suffix: 1
      nsize: 4
      args: -m_local 80 -n_local 25 -mat_htool_epsilon 1.0e-11 -symmetric {{false true}shared output}
      output_file: output/ex101.out

   test:
      requires: htool
      suffix: 2
      nsize: 4
      args: -m_local 120 -mat_htool_epsilon 1.0e-2 -mat_htool_compressor {{sympartialACA fullACA SVD}shared output} -mat_htool_clustering {{PCARegular PCAGeometric BoundingBox1Regular BoundingBox1Geometric}shared output}
      output_file: output/ex101.out

TEST*/
