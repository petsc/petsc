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

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)NULL,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m_local",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n_local",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-K",&K,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-symmetric",&sym,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-mat_htool_epsilon",&epsilon,NULL));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  M = size*m;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscMalloc1(m*dim,&coords));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  CHKERRQ(PetscRandomGetValuesReal(rdm,m*dim,coords));
  CHKERRQ(PetscCalloc1(M*dim,&gcoords));
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,M,K,NULL,&B));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatSetRandom(B,rdm));
  CHKERRQ(MatGetOwnershipRange(B,&begin,NULL));
  CHKERRQ(PetscArraycpy(gcoords+begin*dim,coords,m*dim));
  CHKERRMPI(MPIU_Allreduce(MPI_IN_PLACE,gcoords,M*dim,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD));
  CHKERRQ(MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,m,M,M,dim,coords,coords,kernel,gcoords,&A));
  CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,sym));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatViewFromOptions(A,NULL,"-A_view"));
  CHKERRQ(MatGetOwnershipIS(A,is,NULL));
  CHKERRQ(ISDuplicate(is[0],is+1));
  CHKERRQ(MatIncreaseOverlap(A,1,is,2));
  CHKERRQ(MatSetBlockSize(A,2));
  CHKERRQ(MatIncreaseOverlap(A,1,is+1,1));
  CHKERRQ(ISGetBlockSize(is[1],&bs));
  PetscCheckFalse(bs != 2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect block size %" PetscInt_FMT " != 2",bs);
  CHKERRQ(MatSetBlockSize(A,1));
  CHKERRQ(ISEqual(is[0],is[1],&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unequal index sets");
  CHKERRQ(ISDestroy(is));
  CHKERRQ(ISDestroy(is+1));
  CHKERRQ(MatCreateVecs(A,&right,&left));
  CHKERRQ(VecSetRandom(right,rdm));
  CHKERRQ(MatMult(A,right,left));
  CHKERRQ(MatHtoolGetPermutationSource(A,&iss));
  CHKERRQ(MatHtoolGetPermutationTarget(A,&ist));
  CHKERRQ(VecDuplicate(left,&perm));
  CHKERRQ(VecCopy(left,perm));
  CHKERRQ(VecPermute(perm,ist,PETSC_FALSE));
  CHKERRQ(VecPermute(right,iss,PETSC_FALSE));
  CHKERRQ(MatHtoolUsePermutation(A,PETSC_FALSE));
  CHKERRQ(MatMult(A,right,left));
  CHKERRQ(VecAXPY(left,-1.0,perm));
  CHKERRQ(VecNorm(left,NORM_INFINITY,&norm));
  PetscCheckFalse(PetscAbsReal(norm) > PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"||y(with permutation)-y(without permutation)|| = %g (> %g)",(double)PetscAbsReal(norm),(double)PETSC_SMALL);
  CHKERRQ(MatHtoolUsePermutation(A,PETSC_TRUE));
  CHKERRQ(VecDestroy(&perm));
  CHKERRQ(VecDestroy(&left));
  CHKERRQ(VecDestroy(&right));
  CHKERRQ(ISDestroy(&ist));
  CHKERRQ(ISDestroy(&iss));
  if (PetscAbsReal(epsilon) >= PETSC_SMALL) { /* when there is compression, it is more difficult to check against MATDENSE, so just compare symmetric and nonsymmetric assemblies */
    PetscReal relative;
    CHKERRQ(MatDestroy(&B));
    CHKERRQ(MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,m,M,M,dim,coords,coords,kernel,gcoords,&B));
    CHKERRQ(MatSetOption(B,MAT_SYMMETRIC,(PetscBool)!sym));
    CHKERRQ(MatSetFromOptions(B));
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatViewFromOptions(B,NULL,"-B_view"));
    CHKERRQ(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&P));
    CHKERRQ(MatNorm(P,NORM_FROBENIUS,&relative));
    CHKERRQ(MatConvert(B,MATDENSE,MAT_INITIAL_MATRIX,&R));
    CHKERRQ(MatAXPY(R,-1.0,P,SAME_NONZERO_PATTERN));
    CHKERRQ(MatNorm(R,NORM_INFINITY,&norm));
    PetscCheckFalse(PetscAbsReal(norm/relative) > epsilon,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"||A(!symmetric)-A(symmetric)|| = %g (> %g)",(double)PetscAbsReal(norm/relative),(double)epsilon);
    CHKERRQ(MatDestroy(&B));
    CHKERRQ(MatDestroy(&R));
    CHKERRQ(MatDestroy(&P));
  } else {
    CHKERRQ(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&D));
    CHKERRQ(MatViewFromOptions(D,NULL,"-D_view"));
    CHKERRQ(MatMultEqual(A,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Ax != Dx");
    CHKERRQ(MatMultTransposeEqual(A,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"A^Tx != D^Tx");
    CHKERRQ(MatMultAddEqual(A,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"y+Ax != y+Dx");
    CHKERRQ(MatGetOwnershipRange(B,&begin,NULL));
    CHKERRQ(MatDenseGetArrayWrite(D,&ptr));
    for (PetscInt i = begin; i < m+begin; ++i)
      for (PetscInt j = 0; j < M; ++j) GenEntries(dim,1,1,&i,&j,ptr+i-begin+j*m,gcoords);
    CHKERRQ(MatDenseRestoreArrayWrite(D,&ptr));
    CHKERRQ(MatMultEqual(A,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Ax != Dx");
    CHKERRQ(MatTranspose(D,MAT_INPLACE_MATRIX,&D));
    CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&AT));
    CHKERRQ(MatMultEqual(AT,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"A^Tx != D^Tx");
    CHKERRQ(MatTranspose(A,MAT_REUSE_MATRIX,&AT));
    CHKERRQ(MatMultEqual(AT,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"A^Tx != D^Tx");
    CHKERRQ(MatAXPY(D,-1.0,AT,SAME_NONZERO_PATTERN));
    CHKERRQ(MatNorm(D,NORM_INFINITY,&norm));
    PetscCheckFalse(PetscAbsReal(norm) > PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"||A-D|| = %g (> %g)",(double)norm,(double)PETSC_SMALL);
    CHKERRQ(MatDestroy(&AT));
    CHKERRQ(MatDestroy(&D));
    CHKERRQ(MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&P));
    CHKERRQ(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatMatMultEqual(A,B,P,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"ABx != Px");
    CHKERRQ(MatTransposeMatMultEqual(A,B,P,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"A^TBx != P^Tx");
    CHKERRQ(MatDestroy(&B));
    CHKERRQ(MatDestroy(&P));
    if (n) {
      CHKERRQ(PetscMalloc1(n*dim,&scoords));
      CHKERRQ(PetscRandomGetValuesReal(rdm,n*dim,scoords));
      N = n;
      CHKERRMPI(MPIU_Allreduce(MPI_IN_PLACE,&N,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
      CHKERRQ(PetscCalloc1(N*dim,&gscoords));
      CHKERRMPI(MPI_Exscan(&n,&begin,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
      CHKERRQ(PetscArraycpy(gscoords+begin*dim,scoords,n*dim));
      CHKERRMPI(MPIU_Allreduce(MPI_IN_PLACE,gscoords,N*dim,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD));
      kernel = GenEntriesRectangular;
      ctx[0] = gcoords;
      ctx[1] = gscoords;
      CHKERRQ(MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,n,M,N,dim,coords,scoords,kernel,ctx,&R));
      CHKERRQ(MatSetFromOptions(R));
      CHKERRQ(MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatViewFromOptions(R,NULL,"-R_view"));
      CHKERRQ(MatConvert(R,MATDENSE,MAT_INITIAL_MATRIX,&D));
      CHKERRQ(MatViewFromOptions(D,NULL,"-D_view"));
      CHKERRQ(MatMultEqual(R,D,10,&flg));
      PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Rx != Dx");
      CHKERRQ(MatTranspose(D,MAT_INPLACE_MATRIX,&D));
      CHKERRQ(MatTranspose(R,MAT_INITIAL_MATRIX,&RT));
      CHKERRQ(MatMultEqual(RT,D,10,&flg));
      PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"R^Tx != D^Tx");
      CHKERRQ(MatTranspose(R,MAT_REUSE_MATRIX,&RT));
      CHKERRQ(MatMultEqual(RT,D,10,&flg));
      PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"R^Tx != D^Tx");
      CHKERRQ(MatDestroy(&RT));
      CHKERRQ(MatDestroy(&D));
      CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,n,PETSC_DECIDE,PETSC_DETERMINE,K,NULL,&B));
      CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatSetRandom(B,rdm));
      CHKERRQ(MatMatMult(R,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&P));
      CHKERRQ(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatMatMultEqual(R,B,P,10,&flg));
      PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"RBx != Px");
      CHKERRQ(MatDestroy(&B));
      CHKERRQ(MatDestroy(&P));
      CHKERRQ(MatCreateVecs(R,&right,&left));
      CHKERRQ(VecSetRandom(right,rdm));
      CHKERRQ(MatMult(R,right,left));
      CHKERRQ(MatHtoolGetPermutationSource(R,&iss));
      CHKERRQ(MatHtoolGetPermutationTarget(R,&ist));
      CHKERRQ(VecDuplicate(left,&perm));
      CHKERRQ(VecCopy(left,perm));
      CHKERRQ(VecPermute(perm,ist,PETSC_FALSE));
      CHKERRQ(VecPermute(right,iss,PETSC_FALSE));
      CHKERRQ(MatHtoolUsePermutation(R,PETSC_FALSE));
      CHKERRQ(MatMult(R,right,left));
      CHKERRQ(VecAXPY(left,-1.0,perm));
      CHKERRQ(VecNorm(left,NORM_INFINITY,&norm));
      PetscCheckFalse(PetscAbsReal(norm) > PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"||y(with permutation)-y(without permutation)|| = %g (> %g)",(double)PetscAbsReal(norm),(double)PETSC_SMALL);
      CHKERRQ(MatHtoolUsePermutation(R,PETSC_TRUE));
      CHKERRQ(VecDestroy(&perm));
      CHKERRQ(VecDestroy(&left));
      CHKERRQ(VecDestroy(&right));
      CHKERRQ(ISDestroy(&ist));
      CHKERRQ(ISDestroy(&iss));
      CHKERRQ(MatDestroy(&R));
      CHKERRQ(PetscFree(gscoords));
      CHKERRQ(PetscFree(scoords));
    }
  }
  CHKERRQ(PetscRandomDestroy(&rdm));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFree(gcoords));
  CHKERRQ(PetscFree(coords));
  CHKERRQ(PetscFinalize());
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
