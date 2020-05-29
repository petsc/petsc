static char help[] = "Tests MATHARA\n\n";

#include <petscmat.h>
#include <petscsf.h>

static PetscScalar GenEntry_Symm(PetscInt sdim, PetscReal x[], PetscReal y[], void *ctx)
{
    PetscInt  d;
    PetscReal clength = sdim == 3 ? 0.2 : 0.1;
    PetscReal dist, diff = 0.0;

    for (d = 0; d < sdim; d++) { diff += (x[d] - y[d]) * (x[d] - y[d]); }
    dist = PetscSqrtReal(diff);
    return PetscExpReal(-dist / clength);
}

static PetscScalar GenEntry_Unsymm(PetscInt sdim, PetscReal x[], PetscReal y[], void *ctx)
{
    PetscInt  d;
    PetscReal clength = sdim == 3 ? 0.2 : 0.1;
    PetscReal dist, diff = 0.0, nx = 0.0, ny = 0.0;

    for (d = 0; d < sdim; d++) { nx += x[d]*x[d]; }
    for (d = 0; d < sdim; d++) { ny += y[d]*y[d]; }
    for (d = 0; d < sdim; d++) { diff += (x[d] - y[d]) * (x[d] - y[d]); }
    dist = PetscSqrtReal(diff);
    return nx > ny ? PetscExpReal(-dist / clength) : PetscExpReal(-dist / clength) + 1.;
}

int main(int argc,char **argv)
{
  Mat            A,B,C,D;
  Vec            v,x,y,Ax,Ay,Bx,By;
  PetscRandom    r;
  PetscLayout    map;
  PetscScalar    *Adata = NULL, *Cdata = NULL, scale = 1.0;
  PetscReal      *coords,nA,nD,nB,err,nX,norms[3];
  PetscInt       N, n = 64, dim = 1, i, j, nrhs = 11, lda = 0, ldc = 0, nt, ntrials = 2;
  PetscMPIInt    size,rank;
  PetscBool      testlayout = PETSC_FALSE,flg,symm = PETSC_FALSE, Asymm = PETSC_TRUE, kernel = PETSC_TRUE;
  PetscBool      checkexpl = PETSC_FALSE, agpu = PETSC_FALSE, bgpu = PETSC_FALSE, cgpu = PETSC_FALSE;
  PetscBool      testtrans, testnorm, randommat = PETSC_TRUE;
  void           (*approxnormfunc)(void);
  void           (*Anormfunc)(void);
  PetscErrorCode ierr;

  PETSC_MPI_THREAD_REQUIRED = MPI_THREAD_MULTIPLE;
  ierr = PetscInitialize(&argc,&argv,(char*) 0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-lda",&lda,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ldc",&ldc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-matmattrials",&ntrials,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-randommat",&randommat,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testlayout",&testlayout,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-Asymm",&Asymm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-symm",&symm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-kernel",&kernel,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-checkexpl",&checkexpl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-agpu",&agpu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-bgpu",&bgpu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-cgpu",&cgpu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-scale",&scale,NULL);CHKERRQ(ierr);
  if (!Asymm) symm = PETSC_FALSE;

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  /* MatMultTranspose for nonsymmetric matrices not implemented */
  testtrans = (PetscBool)(size == 1 || symm);
  testnorm = (PetscBool)(size == 1 || symm);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(PETSC_COMM_WORLD,&map);CHKERRQ(ierr);
  if (testlayout) {
    if (rank%2) n = PetscMax(2*n-5*rank,0);
    else n = 2*n+rank;
  }
  ierr = PetscLayoutSetLocalSize(map,n);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(map,&N);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&map);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
  if (lda) {
    ierr = PetscMalloc1(N*(n+lda),&Adata);CHKERRQ(ierr);
  }
  ierr = MatCreateDense(PETSC_COMM_WORLD,n,n,N,N,Adata,&A);CHKERRQ(ierr);
  ierr = MatDenseSetLDA(A,n+lda);CHKERRQ(ierr);
  ierr = PetscMalloc1(n*dim,&coords);CHKERRQ(ierr);
  for (j = 0; j < n; j++) {
    for (i = 0; i < dim; i++) {
      PetscScalar a;

      ierr = PetscRandomGetValue(r,&a);CHKERRQ(ierr);
      coords[j*dim + i] = PetscRealPart(a);
    }
  }
  if (kernel || !randommat) {
    MatHaraKernel k = Asymm ? GenEntry_Symm : GenEntry_Unsymm;
    PetscInt      ist,ien;
    PetscReal     *gcoords;

    if (size > 1) { /* replicated coords so that we can populate the dense matrix */
      PetscSF      sf;
      MPI_Datatype dtype;

      ierr = MPI_Type_contiguous(dim,MPIU_REAL,&dtype);CHKERRQ(ierr);
      ierr = MPI_Type_commit(&dtype);CHKERRQ(ierr);

      ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
      ierr = MatGetLayouts(A,&map,NULL);CHKERRQ(ierr);
      ierr = PetscSFSetGraphWithPattern(sf,map,PETSCSF_PATTERN_ALLGATHER);CHKERRQ(ierr);
      ierr = PetscMalloc1(dim*N,&gcoords);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(sf,dtype,coords,gcoords);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf,dtype,coords,gcoords);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
      ierr = MPI_Type_free(&dtype);CHKERRQ(ierr);
    } else gcoords = (PetscReal*)coords;

    ierr = MatGetOwnershipRange(A,&ist,&ien);CHKERRQ(ierr);
    for (i = ist; i < ien; i++) {
      for (j = 0; j < N; j++) {
        ierr = MatSetValue(A,i,j,(*k)(dim,gcoords + i*dim,gcoords + j*dim,NULL),INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (kernel) {
      ierr = MatCreateHaraFromKernel(PETSC_COMM_WORLD,n,n,N,N,dim,coords,k,NULL,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&B);CHKERRQ(ierr);
    } else {
      ierr = MatCreateHaraFromMat(A,dim,coords,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&B);CHKERRQ(ierr);
    }
    if (gcoords != coords) { ierr = PetscFree(gcoords);CHKERRQ(ierr); }
  } else {
    ierr = MatSetRandom(A,r);CHKERRQ(ierr);
    if (Asymm) {
      ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
      ierr = MatAXPY(A,1.0,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatDestroy(&B);CHKERRQ(ierr);
      ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = MatCreateHaraFromMat(A,dim,coords,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&B);CHKERRQ(ierr);
  }
  if (agpu) {
    ierr = MatConvert(A,MATDENSECUDA,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  }
  ierr = MatViewFromOptions(A,NULL,"-A_view");CHKERRQ(ierr);

  ierr = MatSetOption(B,MAT_SYMMETRIC,symm);CHKERRQ(ierr);
  ierr = PetscFree(coords);CHKERRQ(ierr);

  /* assemble the H-matrix */
  ierr = MatBindToCPU(B,(PetscBool)!bgpu);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatViewFromOptions(B,NULL,"-B_view");CHKERRQ(ierr);

  /* Test MatScale */
  ierr = MatScale(A,scale);CHKERRQ(ierr);
  ierr = MatScale(B,scale);CHKERRQ(ierr);

  /* Test MatMult */
  ierr = MatCreateVecs(A,&Ax,&Ay);CHKERRQ(ierr);
  ierr = MatCreateVecs(B,&Bx,&By);CHKERRQ(ierr);
  ierr = VecSetRandom(Ax,r);CHKERRQ(ierr);
  ierr = VecCopy(Ax,Bx);CHKERRQ(ierr);
  ierr = MatMult(A,Ax,Ay);CHKERRQ(ierr);
  ierr = MatMult(B,Bx,By);CHKERRQ(ierr);
  ierr = VecViewFromOptions(Ay,NULL,"-mult_vec_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(By,NULL,"-mult_vec_view");CHKERRQ(ierr);
  ierr = VecNorm(Ay,NORM_INFINITY,&nX);CHKERRQ(ierr);
  ierr = VecAXPY(Ay,-1.0,By);CHKERRQ(ierr);
  ierr = VecViewFromOptions(Ay,NULL,"-mult_vec_view");CHKERRQ(ierr);
  ierr = VecNorm(Ay,NORM_INFINITY,&err);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMult err %g\n",err/nX);CHKERRQ(ierr);
  ierr = VecScale(By,-1.0);CHKERRQ(ierr);
  ierr = MatMultAdd(B,Bx,By,By);CHKERRQ(ierr);
  ierr = VecNorm(By,NORM_INFINITY,&err);CHKERRQ(ierr);
  ierr = VecViewFromOptions(By,NULL,"-mult_vec_view");CHKERRQ(ierr);
  if (err > PETSC_SMALL) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMultAdd err %g\n",err);CHKERRQ(ierr);
  }

  /* Test MatNorm */
  ierr = MatNorm(A,NORM_INFINITY,&norms[0]);CHKERRQ(ierr);
  ierr = MatNorm(A,NORM_1,&norms[1]);CHKERRQ(ierr);
  norms[2] = -1.; /* NORM_2 not supported */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"A Matrix norms:        infty=%g, norm_1=%g, norm_2=%g\n",(double)norms[0],(double)norms[1],(double)norms[2]);CHKERRQ(ierr);
  ierr = MatGetOperation(A,MATOP_NORM,&Anormfunc);CHKERRQ(ierr);
  ierr = MatGetOperation(B,MATOP_NORM,&approxnormfunc);CHKERRQ(ierr);
  ierr = MatSetOperation(A,MATOP_NORM,approxnormfunc);CHKERRQ(ierr);
  ierr = MatNorm(A,NORM_INFINITY,&norms[0]);CHKERRQ(ierr);
  ierr = MatNorm(A,NORM_1,&norms[1]);CHKERRQ(ierr);
  ierr = MatNorm(A,NORM_2,&norms[2]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"A Approx Matrix norms: infty=%g, norm_1=%g, norm_2=%g\n",(double)norms[0],(double)norms[1],(double)norms[2]);CHKERRQ(ierr);
  if (testnorm) {
    ierr = MatNorm(B,NORM_INFINITY,&norms[0]);CHKERRQ(ierr);
    ierr = MatNorm(B,NORM_1,&norms[1]);CHKERRQ(ierr);
    ierr = MatNorm(B,NORM_2,&norms[2]);CHKERRQ(ierr);
  } else {
    norms[0] = -1.;
    norms[1] = -1.;
    norms[2] = -1.;
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"B Approx Matrix norms: infty=%g, norm_1=%g, norm_2=%g\n",(double)norms[0],(double)norms[1],(double)norms[2]);CHKERRQ(ierr);
  ierr = MatSetOperation(A,MATOP_NORM,Anormfunc);CHKERRQ(ierr);

  /* Test MatDuplicate */
  ierr = MatDuplicate(B,MAT_COPY_VALUES,&D);CHKERRQ(ierr);
  ierr = MatMultEqual(B,D,10,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMult error after MatDuplicate\n");CHKERRQ(ierr);
  }
  if (testtrans) {
    ierr = MatMultTransposeEqual(B,D,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMultTranpose error after MatDuplicate\n");CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&D);CHKERRQ(ierr);

  if (testtrans) { /* MatMultTranspose for nonsymmetric matrices not implemented */
    ierr = VecSetRandom(Ay,r);CHKERRQ(ierr);
    ierr = VecCopy(Ay,By);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,Ay,Ax);CHKERRQ(ierr);
    ierr = MatMultTranspose(B,By,Bx);CHKERRQ(ierr);
    ierr = VecViewFromOptions(Ax,NULL,"-multtrans_vec_view");CHKERRQ(ierr);
    ierr = VecViewFromOptions(Bx,NULL,"-multtrans_vec_view");CHKERRQ(ierr);
    ierr = VecNorm(Ax,NORM_INFINITY,&nX);CHKERRQ(ierr);
    ierr = VecAXPY(Ax,-1.0,Bx);CHKERRQ(ierr);
    ierr = VecViewFromOptions(Ax,NULL,"-multtrans_vec_view");CHKERRQ(ierr);
    ierr = VecNorm(Ax,NORM_INFINITY,&err);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMultTranspose err %g\n",err/nX);CHKERRQ(ierr);
    ierr = VecScale(Bx,-1.0);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(B,By,Bx,Bx);CHKERRQ(ierr);
    ierr = VecNorm(Bx,NORM_INFINITY,&err);CHKERRQ(ierr);
    if (err > PETSC_SMALL) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMultTransposeAdd err %g\n",err);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&Ax);CHKERRQ(ierr);
  ierr = VecDestroy(&Ay);CHKERRQ(ierr);
  ierr = VecDestroy(&Bx);CHKERRQ(ierr);
  ierr = VecDestroy(&By);CHKERRQ(ierr);

  /* Test MatMatMult */
  if (ldc) {
    ierr = PetscMalloc1(nrhs*(n+ldc),&Cdata);CHKERRQ(ierr);
  }
  ierr = MatCreateDense(PETSC_COMM_WORLD,n,PETSC_DECIDE,N,nrhs,Cdata,&C);CHKERRQ(ierr);
  ierr = MatDenseSetLDA(C,n+ldc);CHKERRQ(ierr);
  ierr = MatSetRandom(C,r);CHKERRQ(ierr);
  if (cgpu) {
    ierr = MatConvert(C,MATDENSECUDA,MAT_INPLACE_MATRIX,&C);CHKERRQ(ierr);
  }
  for (nt = 0; nt < ntrials; nt++) {
    ierr = MatMatMult(B,C,nt ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D);CHKERRQ(ierr);
    ierr = MatViewFromOptions(D,NULL,"-bc_view");CHKERRQ(ierr);
    ierr = PetscObjectBaseTypeCompareAny((PetscObject)D,&flg,MATSEQDENSE,MATMPIDENSE,"");CHKERRQ(ierr);
    if (flg) {
      ierr = MatCreateVecs(B,&x,&y);CHKERRQ(ierr);
      ierr = MatCreateVecs(D,NULL,&v);CHKERRQ(ierr);
      for (i = 0; i < nrhs; i++) {
        ierr = MatGetColumnVector(D,v,i);CHKERRQ(ierr);
        ierr = MatGetColumnVector(C,x,i);CHKERRQ(ierr);
        ierr = MatMult(B,x,y);CHKERRQ(ierr);
        ierr = VecAXPY(y,-1.0,v);CHKERRQ(ierr);
        ierr = VecNorm(y,NORM_INFINITY,&err);CHKERRQ(ierr);
        if (err > PETSC_SMALL) { ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMat err %D %g\n",i,err);CHKERRQ(ierr); }
      }
      ierr = VecDestroy(&y);CHKERRQ(ierr);
      ierr = VecDestroy(&x);CHKERRQ(ierr);
      ierr = VecDestroy(&v);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&D);CHKERRQ(ierr);

  /* Test MatTransposeMatMult */
  if (testtrans) { /* MatMultTranspose for nonsymmetric matrices not implemented */
    for (nt = 0; nt < ntrials; nt++) {
      ierr = MatTransposeMatMult(B,C,nt ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D);CHKERRQ(ierr);
      ierr = MatViewFromOptions(D,NULL,"-btc_view");CHKERRQ(ierr);
      ierr = PetscObjectBaseTypeCompareAny((PetscObject)D,&flg,MATSEQDENSE,MATMPIDENSE,"");CHKERRQ(ierr);
      if (flg) {
        ierr = MatCreateVecs(B,&y,&x);CHKERRQ(ierr);
        ierr = MatCreateVecs(D,NULL,&v);CHKERRQ(ierr);
        for (i = 0; i < nrhs; i++) {
          ierr = MatGetColumnVector(D,v,i);CHKERRQ(ierr);
          ierr = MatGetColumnVector(C,x,i);CHKERRQ(ierr);
          ierr = MatMultTranspose(B,x,y);CHKERRQ(ierr);
          ierr = VecAXPY(y,-1.0,v);CHKERRQ(ierr);
          ierr = VecNorm(y,NORM_INFINITY,&err);CHKERRQ(ierr);
          if (err > PETSC_SMALL) { ierr = PetscPrintf(PETSC_COMM_WORLD,"MatTransMat err %D %g\n",i,err);CHKERRQ(ierr); }
        }
        ierr = VecDestroy(&y);CHKERRQ(ierr);
        ierr = VecDestroy(&x);CHKERRQ(ierr);
        ierr = VecDestroy(&v);CHKERRQ(ierr);
      }
    }
    ierr = MatDestroy(&D);CHKERRQ(ierr);
  }

  /* check explicit operator */
  if (checkexpl) {
    Mat Be, Bet;

    ierr = MatComputeOperator(B,MATDENSE,&D);CHKERRQ(ierr);
    ierr = MatDuplicate(D,MAT_COPY_VALUES,&Be);CHKERRQ(ierr);
    ierr = MatNorm(D,NORM_FROBENIUS,&nB);CHKERRQ(ierr);
    ierr = MatViewFromOptions(D,NULL,"-expl_view");CHKERRQ(ierr);
    ierr = MatAXPY(D,-1.0,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatViewFromOptions(D,NULL,"-diff_view");CHKERRQ(ierr);
    ierr = MatNorm(D,NORM_FROBENIUS,&nD);CHKERRQ(ierr);
    ierr = MatNorm(A,NORM_FROBENIUS,&nA);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Approximation error %g (%g / %g, %g)\n",nD/nA,nD,nA,nB);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);

    if (testtrans) { /* MatMultTranspose for nonsymmetric matrices not implemented */
      ierr = MatTranspose(A,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
      ierr = MatComputeOperatorTranspose(B,MATDENSE,&D);CHKERRQ(ierr);
      ierr = MatDuplicate(D,MAT_COPY_VALUES,&Bet);CHKERRQ(ierr);
      ierr = MatNorm(D,NORM_FROBENIUS,&nB);CHKERRQ(ierr);
      ierr = MatViewFromOptions(D,NULL,"-expl_trans_view");CHKERRQ(ierr);
      ierr = MatAXPY(D,-1.0,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatViewFromOptions(D,NULL,"-diff_trans_view");CHKERRQ(ierr);
      ierr = MatNorm(D,NORM_FROBENIUS,&nD);CHKERRQ(ierr);
      ierr = MatNorm(A,NORM_FROBENIUS,&nA);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Approximation error transpose %g (%g / %g, %g)\n",nD/nA,nD,nA,nB);CHKERRQ(ierr);
      ierr = MatDestroy(&D);CHKERRQ(ierr);

      ierr = MatTranspose(Bet,MAT_INPLACE_MATRIX,&Bet);CHKERRQ(ierr);
      ierr = MatAXPY(Be,-1.0,Bet,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatViewFromOptions(Be,NULL,"-diff_expl_view");CHKERRQ(ierr);
      ierr = MatNorm(Be,NORM_FROBENIUS,&nB);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Approximation error B - (B^T)^T %g\n",nB);CHKERRQ(ierr);
      ierr = MatDestroy(&Be);CHKERRQ(ierr);
      ierr = MatDestroy(&Bet);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  ierr = PetscFree(Cdata);CHKERRQ(ierr);
  ierr = PetscFree(Adata);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: hara

#tests from kernel
   test:
     requires: hara
     nsize: 1
     suffix: 1
     args: -n {{17 33}} -kernel 1 -dim {{1 2 3}} -symm {{0 1}} -checkexpl -bgpu 0

   test:
     requires: hara
     nsize: 1
     suffix: 1_ld
     output_file: output/ex66_1.out
     args: -n 33 -kernel 1 -dim 1 -lda 13 -ldc 11 -symm 0 -checkexpl -bgpu 0

   test:
     requires: hara cuda
     nsize: 1
     suffix: 1_cuda
     output_file: output/ex66_1.out
     args: -n {{17 33}} -kernel 1 -dim {{1 2 3}} -symm {{0 1}} -checkexpl -bgpu 1

   test:
     requires: hara cuda
     nsize: 1
     suffix: 1_cuda_ld
     output_file: output/ex66_1.out
     args: -n 33 -kernel 1 -dim 1 -lda 13 -ldc 11 -symm 0 -checkexpl -bgpu 1

   test:
     requires: hara define(PETSC_HAVE_MPI_INIT_THREAD)
     nsize: 2
     suffix: 1_par
     args: -n 32 -kernel 1 -dim 1 -ldc 12 -testlayout {{0 1}} -bgpu 0 -cgpu 0

   test:
     requires: hara cuda define(PETSC_HAVE_MPI_INIT_THREAD)
     nsize: 2
     suffix: 1_par_cuda
     args: -n 32 -kernel 1 -dim 1 -ldc 12 -testlayout {{0 1}} -bgpu {{0 1}} -cgpu {{0 1}}
     output_file: output/ex66_1_par.out

#tests from matrix sampling (parallel or unsymmetric not supported)
   test:
     requires: hara
     nsize: 1
     suffix: 2
     args: -n {{17 33}} -kernel 0 -dim 2 -symm 1 -checkexpl -bgpu 0

   test:
     requires: hara cuda
     nsize: 1
     suffix: 2_cuda
     output_file: output/ex66_2.out
     args: -n {{17 33}} -kernel 0 -dim 2 -symm 1 -checkexpl -bgpu {{0 1}} -agpu {{0 1}}

TEST*/
