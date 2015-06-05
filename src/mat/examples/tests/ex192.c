
static char help[] = "Tests MatSolve() and MatMatSolve() with mumps sequential solver in Schur complement mode.\n\
Example: mpiexec -n 1 ./ex192 -f <matrix binary file> -nrhs 4 -symmetric_solve -hermitian_solve -sratio 0.3\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,RHS,C,F,X,S;
  Vec            u,x,b;
  Vec            xschur,bschur,uschur;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       icntl19,size_schur,*idxs_schur,i,m,n,nfact,nsolve,nrhs;
  PetscReal      norm,tol=1.e-12;
  PetscRandom    rand;
  PetscBool      flg,herm,symm;
  PetscReal      sratio = 5.1/12.;
  PetscViewer    fd;              /* viewer */
  char           file[PETSC_MAX_PATH_LEN]; /* input file name */

  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,1,"This is a uniprocessor test");
  /* Determine which type of solver we want to test for */
  herm = PETSC_FALSE;
  symm = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,"-symmetric_solve",&symm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-hermitian_solve",&herm,NULL);CHKERRQ(ierr);
  if (herm) symm = PETSC_TRUE;

  /* Determine file from which we read the matrix A */
  ierr = PetscOptionsGetString(NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) { /* get matrices from PETSc distribution */
    sprintf(file,PETSC_DIR);
    ierr = PetscStrcat(file,"/share/petsc/datafiles/matrices/");CHKERRQ(ierr);
    if (symm) {
#if defined (PETSC_USE_COMPLEX)
      ierr = PetscStrcat(file,"hpd-complex-");CHKERRQ(ierr);
#else
      ierr = PetscStrcat(file,"spd-real-");CHKERRQ(ierr);
#endif
    } else {
#if defined (PETSC_USE_COMPLEX)
      ierr = PetscStrcat(file,"nh-complex-");CHKERRQ(ierr);
#else
      ierr = PetscStrcat(file,"ns-real-");CHKERRQ(ierr);
#endif
    }
#if defined(PETSC_USE_64BIT_INDICES)
    ierr = PetscStrcat(file,"int64-");CHKERRQ(ierr);
#else
    ierr = PetscStrcat(file,"int32-");CHKERRQ(ierr);
#endif
#if defined (PETSC_USE_REAL_SINGLE)
    ierr = PetscStrcat(file,"float32");CHKERRQ(ierr);
#else
    ierr = PetscStrcat(file,"float64");CHKERRQ(ierr);
#endif
  }
  /* Load matrix A */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);

#if defined (PETSC_USE_COMPLEX)
  if (symm & !flg) { /* MUMPS (5.0.0) does not have support for hermitian matrices, so make them symmetric */
    PetscScalar im = PetscSqrtScalar((PetscScalar)-1.);
    PetscScalar val = -1.0;
    val = val + im;
    ierr = MatSetValue(A,1,0,val,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
#endif
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  if (m != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%d, %d)", m, n);

  /* Create dense matrix C and X; C holds true solution with identical colums */
  nrhs = 2;
  ierr = PetscOptionsGetInt(NULL,"-nrhs",&nrhs,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,m,PETSC_DECIDE,PETSC_DECIDE,nrhs);CHKERRQ(ierr);
  ierr = MatSetType(C,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = MatSetRandom(C,rand);CHKERRQ(ierr);
  ierr = MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&X);CHKERRQ(ierr);

  /* Create vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr); /* save the true solution */

  /* Test LU/Cholesky Factorization */
  if (!symm) {
    ierr = MatGetFactor(A,MATSOLVERMUMPS,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
  } else {
    if (herm) {
      ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatSetOption(A,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatSetOption(A,MAT_SPD,PETSC_FALSE);CHKERRQ(ierr);
    }
    ierr = MatGetFactor(A,MATSOLVERMUMPS,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetReal(NULL,"-schur_ratio",&sratio,NULL);CHKERRQ(ierr);
  if (sratio < 0. || sratio > 1.) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Invalid ratio for schur degrees of freedom %f", sratio);
  }
  size_schur = (PetscInt)(sratio*m);
  ierr = PetscMalloc1(size_schur,&idxs_schur);CHKERRQ(ierr);
  for (i=0;i<size_schur;i++) {
    idxs_schur[i] = m-size_schur+i+1; /* fortran like */
  }
  ierr = MatMumpsSetSchurIndices(F,size_schur,idxs_schur);CHKERRQ(ierr);
  ierr = PetscFree(idxs_schur);CHKERRQ(ierr);
  if (!symm) {
    ierr = MatLUFactorSymbolic(F,A,NULL,NULL,NULL);CHKERRQ(ierr);
  } else {
    ierr = MatCholeskyFactorSymbolic(F,A,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = MatMumpsGetIcntl(F,19,&icntl19);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Solving: nrhs %d, sym %d, herm %d, size schur %d, size mat %d, icntl19 %d\n",nrhs,symm,herm,size_schur,m,icntl19);CHKERRQ(ierr);

  for (nfact = 0; nfact < 3; nfact++) {

    if (!nfact) {
      ierr = VecSetRandom(x,rand);CHKERRQ(ierr);
      if (symm && herm) {
        ierr = VecAbs(x);CHKERRQ(ierr);
      }
      ierr = MatDiagonalSet(A,x,ADD_VALUES);CHKERRQ(ierr);
    }
    if (!symm) {
      ierr = MatLUFactorNumeric(F,A,NULL);CHKERRQ(ierr);
    } else {
      ierr = MatCholeskyFactorNumeric(F,A,NULL);CHKERRQ(ierr);
    }
    if (icntl19) {
      ierr = MatMumpsCreateSchurComplement(F,&S);CHKERRQ(ierr);
      ierr = MatCreateVecs(S,&xschur,&bschur);CHKERRQ(ierr);
      ierr = VecDuplicate(xschur,&uschur);CHKERRQ(ierr);
      if (nfact == 1) {
        ierr = MatMumpsInvertSchurComplement(F);CHKERRQ(ierr);
      }
    }
    for (nsolve = 0; nsolve < 2; nsolve++) {
      ierr = VecSetRandom(x,rand);CHKERRQ(ierr);
      ierr = VecCopy(x,u);CHKERRQ(ierr);

      if (nsolve) {
        ierr = MatMult(A,x,b);CHKERRQ(ierr);
        ierr = MatSolve(F,b,x);CHKERRQ(ierr);
      } else {
        ierr = MatMultTranspose(A,x,b);CHKERRQ(ierr);
        ierr = MatSolveTranspose(F,b,x);CHKERRQ(ierr);
      }
      /* Check the error */
      ierr = VecAXPY(u,-1.0,x);CHKERRQ(ierr);  /* u <- (-1.0)x + u */
      ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
      if (norm > tol) {
        PetscReal resi;
        if (nsolve) {
          ierr = MatMult(A,x,u);CHKERRQ(ierr); /* u = A*x */
        } else {
          ierr = MatMultTranspose(A,x,u);CHKERRQ(ierr); /* u = A*x */
        }
        ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);  /* u <- (-1.0)b + u */
        ierr = VecNorm(u,NORM_2,&resi);CHKERRQ(ierr);
        if (nsolve) {
          ierr = PetscPrintf(PETSC_COMM_SELF,"(f %d, s %d) MatSolve: Norm of error %g, residual %f\n",nfact,nsolve,norm,resi);CHKERRQ(ierr);
        } else {
          ierr = PetscPrintf(PETSC_COMM_SELF,"(f %d, s %d) MatSolveTranspose: Norm of error %g, residual %f\n",nfact,nsolve,norm,resi);CHKERRQ(ierr);
        }
      }
      if (icntl19) {
        ierr = VecSetRandom(xschur,rand);CHKERRQ(ierr);
        ierr = VecCopy(xschur,uschur);CHKERRQ(ierr);
        if (nsolve) {
          ierr = MatMult(S,xschur,bschur);CHKERRQ(ierr);
          ierr = MatMumpsSolveSchurComplement(F,bschur,xschur);CHKERRQ(ierr);
        } else {
          ierr = MatMultTranspose(S,xschur,bschur);CHKERRQ(ierr);
          ierr = MatMumpsSolveSchurComplementTranspose(F,bschur,xschur);CHKERRQ(ierr);
        }
        /* Check the error */
        ierr = VecAXPY(uschur,-1.0,xschur);CHKERRQ(ierr);  /* u <- (-1.0)x + u */
        ierr = VecNorm(uschur,NORM_2,&norm);CHKERRQ(ierr);
        if (norm > tol) {
          PetscReal resi;
          if (nsolve) {
            ierr = MatMult(S,xschur,uschur);CHKERRQ(ierr); /* u = A*x */
          } else {
            ierr = MatMultTranspose(S,xschur,uschur);CHKERRQ(ierr); /* u = A*x */
          }
          ierr = VecAXPY(uschur,-1.0,bschur);CHKERRQ(ierr);  /* u <- (-1.0)b + u */
          ierr = VecNorm(uschur,NORM_2,&resi);CHKERRQ(ierr);
          if (nsolve) {
            ierr = PetscPrintf(PETSC_COMM_SELF,"(f %d, s %d) MatMumpsSolveSchurComplement: Norm of error %g, residual %f\n",nfact,nsolve,norm,resi);CHKERRQ(ierr);
          } else {
            ierr = PetscPrintf(PETSC_COMM_SELF,"(f %d, s %d) MatMumpsSolveSchurComplementTranspose: Norm of error %g, residual %f\n",nfact,nsolve,norm,resi);CHKERRQ(ierr);
          }
        }
      }
    }

    if (!nfact) {
      ierr = MatMatMult(A,C,MAT_INITIAL_MATRIX,2.0,&RHS);CHKERRQ(ierr);
    } else {
      ierr = MatMatMult(A,C,MAT_REUSE_MATRIX,2.0,&RHS);CHKERRQ(ierr);
    }
    for (nsolve = 0; nsolve < 2; nsolve++) {
      ierr = MatMatSolve(F,RHS,X);CHKERRQ(ierr);

      /* Check the error */
      ierr = MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(X,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
      if (norm > tol) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"(f %d, s %d) MatMatSolve: Norm of error %g\n",nfact,nsolve,norm);CHKERRQ(ierr);
      }
    }
    if (icntl19) {
      ierr = MatDestroy(&S);CHKERRQ(ierr);
      ierr = VecDestroy(&xschur);CHKERRQ(ierr);
      ierr = VecDestroy(&bschur);CHKERRQ(ierr);
      ierr = VecDestroy(&uschur);CHKERRQ(ierr);
    }
  }
  /* Free data structures */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&RHS);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
