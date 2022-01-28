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
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m_local",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n_local",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-K",&K,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-symmetric",&sym,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-mat_htool_epsilon",&epsilon,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  M = size*m;
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(m*dim,&coords);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomGetValuesReal(rdm,m*dim,coords);CHKERRQ(ierr);
  ierr = PetscCalloc1(M*dim,&gcoords);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,M,K,NULL,&B);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetRandom(B,rdm);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(B,&begin,NULL);CHKERRQ(ierr);
  ierr = PetscArraycpy(gcoords+begin*dim,coords,m*dim);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(MPI_IN_PLACE,gcoords,M*dim,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);CHKERRMPI(ierr);
  ierr = MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,m,M,M,dim,coords,coords,kernel,gcoords,&A);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_SYMMETRIC,sym);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatViewFromOptions(A,NULL,"-A_view");CHKERRQ(ierr);
  ierr = MatGetOwnershipIS(A,is,NULL);CHKERRQ(ierr);
  ierr = ISDuplicate(is[0],is+1);CHKERRQ(ierr);
  ierr = MatIncreaseOverlap(A,1,is,2);CHKERRQ(ierr);
  ierr = MatSetBlockSize(A,2);CHKERRQ(ierr);
  ierr = MatIncreaseOverlap(A,1,is+1,1);CHKERRQ(ierr);
  ierr = ISGetBlockSize(is[1],&bs);CHKERRQ(ierr);
  PetscAssertFalse(bs != 2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect block size %" PetscInt_FMT " != 2",bs);
  ierr = MatSetBlockSize(A,1);CHKERRQ(ierr);
  ierr = ISEqual(is[0],is[1],&flg);CHKERRQ(ierr);
  PetscAssertFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unequal index sets");
  ierr = ISDestroy(is);CHKERRQ(ierr);
  ierr = ISDestroy(is+1);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&right,&left);CHKERRQ(ierr);
  ierr = VecSetRandom(right,rdm);CHKERRQ(ierr);
  ierr = MatMult(A,right,left);CHKERRQ(ierr);
  ierr = MatHtoolGetPermutationSource(A,&iss);CHKERRQ(ierr);
  ierr = MatHtoolGetPermutationTarget(A,&ist);CHKERRQ(ierr);
  ierr = VecDuplicate(left,&perm);CHKERRQ(ierr);
  ierr = VecCopy(left,perm);CHKERRQ(ierr);
  ierr = VecPermute(perm,ist,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecPermute(right,iss,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatHtoolUsePermutation(A,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatMult(A,right,left);CHKERRQ(ierr);
  ierr = VecAXPY(left,-1.0,perm);CHKERRQ(ierr);
  ierr = VecNorm(left,NORM_INFINITY,&norm);CHKERRQ(ierr);
  PetscAssertFalse(PetscAbsReal(norm) > PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"||y(with permutation)-y(without permutation)|| = %g (> %g)",(double)PetscAbsReal(norm),(double)PETSC_SMALL);
  ierr = MatHtoolUsePermutation(A,PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecDestroy(&perm);CHKERRQ(ierr);
  ierr = VecDestroy(&left);CHKERRQ(ierr);
  ierr = VecDestroy(&right);CHKERRQ(ierr);
  ierr = ISDestroy(&ist);CHKERRQ(ierr);
  ierr = ISDestroy(&iss);CHKERRQ(ierr);
  if (PetscAbsReal(epsilon) >= PETSC_SMALL) { /* when there is compression, it is more difficult to check against MATDENSE, so just compare symmetric and nonsymmetric assemblies */
    PetscReal relative;
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,m,M,M,dim,coords,coords,kernel,gcoords,&B);CHKERRQ(ierr);
    ierr = MatSetOption(B,MAT_SYMMETRIC,(PetscBool)!sym);CHKERRQ(ierr);
    ierr = MatSetFromOptions(B);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatViewFromOptions(B,NULL,"-B_view");CHKERRQ(ierr);
    ierr = MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&P);CHKERRQ(ierr);
    ierr = MatNorm(P,NORM_FROBENIUS,&relative);CHKERRQ(ierr);
    ierr = MatConvert(B,MATDENSE,MAT_INITIAL_MATRIX,&R);CHKERRQ(ierr);
    ierr = MatAXPY(R,-1.0,P,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(R,NORM_INFINITY,&norm);CHKERRQ(ierr);
    PetscAssertFalse(PetscAbsReal(norm/relative) > epsilon,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"||A(!symmetric)-A(symmetric)|| = %g (> %g)",(double)PetscAbsReal(norm/relative),(double)epsilon);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = MatDestroy(&R);CHKERRQ(ierr);
    ierr = MatDestroy(&P);CHKERRQ(ierr);
  } else {
    ierr = MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
    ierr = MatViewFromOptions(D,NULL,"-D_view");CHKERRQ(ierr);
    ierr = MatMultEqual(A,D,10,&flg);CHKERRQ(ierr);
    PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Ax != Dx");
    ierr = MatMultTransposeEqual(A,D,10,&flg);CHKERRQ(ierr);
    PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"A^Tx != D^Tx");
    ierr = MatMultAddEqual(A,D,10,&flg);CHKERRQ(ierr);
    PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"y+Ax != y+Dx");
    ierr = MatGetOwnershipRange(B,&begin,NULL);CHKERRQ(ierr);
    ierr = MatDenseGetArrayWrite(D,&ptr);CHKERRQ(ierr);
    for (PetscInt i = begin; i < m+begin; ++i)
      for (PetscInt j = 0; j < M; ++j) GenEntries(dim,1,1,&i,&j,ptr+i-begin+j*m,gcoords);
    ierr = MatDenseRestoreArrayWrite(D,&ptr);CHKERRQ(ierr);
    ierr = MatMultEqual(A,D,10,&flg);CHKERRQ(ierr);
    PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Ax != Dx");
    ierr = MatTranspose(D,MAT_INPLACE_MATRIX,&D);CHKERRQ(ierr);
    ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&AT);CHKERRQ(ierr);
    ierr = MatMultEqual(AT,D,10,&flg);CHKERRQ(ierr);
    PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"A^Tx != D^Tx");
    ierr = MatTranspose(A,MAT_REUSE_MATRIX,&AT);CHKERRQ(ierr);
    ierr = MatMultEqual(AT,D,10,&flg);CHKERRQ(ierr);
    PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"A^Tx != D^Tx");
    ierr = MatAXPY(D,-1.0,AT,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(D,NORM_INFINITY,&norm);CHKERRQ(ierr);
    PetscAssertFalse(PetscAbsReal(norm) > PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"||A-D|| = %g (> %g)",(double)norm,(double)PETSC_SMALL);
    ierr = MatDestroy(&AT);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);
    ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&P);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatMatMultEqual(A,B,P,10,&flg);CHKERRQ(ierr);
    PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"ABx != Px");
    ierr = MatTransposeMatMultEqual(A,B,P,10,&flg);CHKERRQ(ierr);
    PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"A^TBx != P^Tx");
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = MatDestroy(&P);CHKERRQ(ierr);
    if (n) {
      ierr = PetscMalloc1(n*dim,&scoords);CHKERRQ(ierr);
      ierr = PetscRandomGetValuesReal(rdm,n*dim,scoords);CHKERRQ(ierr);
      N = n;
      ierr = MPIU_Allreduce(MPI_IN_PLACE,&N,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRMPI(ierr);
      ierr = PetscCalloc1(N*dim,&gscoords);CHKERRQ(ierr);
      ierr = MPI_Exscan(&n,&begin,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRMPI(ierr);
      ierr = PetscArraycpy(gscoords+begin*dim,scoords,n*dim);CHKERRQ(ierr);
      ierr = MPIU_Allreduce(MPI_IN_PLACE,gscoords,N*dim,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);CHKERRMPI(ierr);
      kernel = GenEntriesRectangular;
      ctx[0] = gcoords;
      ctx[1] = gscoords;
      ierr = MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,n,M,N,dim,coords,scoords,kernel,ctx,&R);CHKERRQ(ierr);
      ierr = MatSetFromOptions(R);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatViewFromOptions(R,NULL,"-R_view");CHKERRQ(ierr);
      ierr = MatConvert(R,MATDENSE,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
      ierr = MatViewFromOptions(D,NULL,"-D_view");CHKERRQ(ierr);
      ierr = MatMultEqual(R,D,10,&flg);CHKERRQ(ierr);
      PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Rx != Dx");
      ierr = MatTranspose(D,MAT_INPLACE_MATRIX,&D);CHKERRQ(ierr);
      ierr = MatTranspose(R,MAT_INITIAL_MATRIX,&RT);CHKERRQ(ierr);
      ierr = MatMultEqual(RT,D,10,&flg);CHKERRQ(ierr);
      PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"R^Tx != D^Tx");
      ierr = MatTranspose(R,MAT_REUSE_MATRIX,&RT);CHKERRQ(ierr);
      ierr = MatMultEqual(RT,D,10,&flg);CHKERRQ(ierr);
      PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"R^Tx != D^Tx");
      ierr = MatDestroy(&RT);CHKERRQ(ierr);
      ierr = MatDestroy(&D);CHKERRQ(ierr);
      ierr = MatCreateDense(PETSC_COMM_WORLD,n,PETSC_DECIDE,PETSC_DETERMINE,K,NULL,&B);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatSetRandom(B,rdm);CHKERRQ(ierr);
      ierr = MatMatMult(R,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&P);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatMatMultEqual(R,B,P,10,&flg);CHKERRQ(ierr);
      PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"RBx != Px");
      ierr = MatDestroy(&B);CHKERRQ(ierr);
      ierr = MatDestroy(&P);CHKERRQ(ierr);
      ierr = MatCreateVecs(R,&right,&left);CHKERRQ(ierr);
      ierr = VecSetRandom(right,rdm);CHKERRQ(ierr);
      ierr = MatMult(R,right,left);CHKERRQ(ierr);
      ierr = MatHtoolGetPermutationSource(R,&iss);CHKERRQ(ierr);
      ierr = MatHtoolGetPermutationTarget(R,&ist);CHKERRQ(ierr);
      ierr = VecDuplicate(left,&perm);CHKERRQ(ierr);
      ierr = VecCopy(left,perm);CHKERRQ(ierr);
      ierr = VecPermute(perm,ist,PETSC_FALSE);CHKERRQ(ierr);
      ierr = VecPermute(right,iss,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatHtoolUsePermutation(R,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatMult(R,right,left);CHKERRQ(ierr);
      ierr = VecAXPY(left,-1.0,perm);CHKERRQ(ierr);
      ierr = VecNorm(left,NORM_INFINITY,&norm);CHKERRQ(ierr);
      PetscAssertFalse(PetscAbsReal(norm) > PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"||y(with permutation)-y(without permutation)|| = %g (> %g)",(double)PetscAbsReal(norm),(double)PETSC_SMALL);
      ierr = MatHtoolUsePermutation(R,PETSC_TRUE);CHKERRQ(ierr);
      ierr = VecDestroy(&perm);CHKERRQ(ierr);
      ierr = VecDestroy(&left);CHKERRQ(ierr);
      ierr = VecDestroy(&right);CHKERRQ(ierr);
      ierr = ISDestroy(&ist);CHKERRQ(ierr);
      ierr = ISDestroy(&iss);CHKERRQ(ierr);
      ierr = MatDestroy(&R);CHKERRQ(ierr);
      ierr = PetscFree(gscoords);CHKERRQ(ierr);
      ierr = PetscFree(scoords);CHKERRQ(ierr);
    }
  }
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFree(gcoords);CHKERRQ(ierr);
  ierr = PetscFree(coords);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
