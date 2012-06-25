static char help[] = "Tests PLAPACK interface.\n\n";

/* Usage:
     mpiexec -n 4 ./ex107 -M 50 -mat_plapack_nprows 2 -mat_plapack_npcols 2 -mat_plapack_nb 1 
 */

#include <petscmat.h>
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C,F,Cpetsc,Csymm; 
  Vec            u,x,b,bpla;
  PetscErrorCode ierr;
  PetscMPIInt    rank,nproc;
  PetscInt       i,j,k,M = 10,m,nfact,nsolve,Istart,Iend,*im,*in,start,end;
  PetscScalar    *array,rval;
  PetscReal      norm,tol=1.e-12;
  IS             perm,iperm;
  MatFactorInfo  info;
  PetscRandom    rand;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &nproc);CHKERRQ(ierr);

  /* Test non-symmetric operations */
  /*-------------------------------*/
  /* Create a Plapack dense matrix C */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,M,M);CHKERRQ(ierr);
  ierr = MatSetType(C,MATDENSE);CHKERRQ(ierr); 
  ierr = MatSetFromOptions(C);CHKERRQ(ierr); 
  ierr = MatSetUp(C);CHKERRQ(ierr);

  /* Create vectors */
  ierr = MatGetOwnershipRange(C,&start,&end);CHKERRQ(ierr);
  m    = end - start;
  /* printf("[%d] C - local size m: %d\n",rank,m); */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&bpla);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr); /* save the true solution */

  /* Create a petsc dense matrix Cpetsc */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&Cpetsc);CHKERRQ(ierr);
  ierr = MatSetSizes(Cpetsc,m,m,M,M);CHKERRQ(ierr);
  ierr = MatSetType(Cpetsc,MATDENSE);CHKERRQ(ierr); 
  ierr = MatMPIDenseSetPreallocation(Cpetsc,PETSC_NULL);CHKERRQ(ierr); 
  ierr = MatSetFromOptions(Cpetsc);CHKERRQ(ierr);
  ierr = MatSetUp(Cpetsc);CHKERRQ(ierr);

  ierr = MatSetOption(Cpetsc,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr); 
  ierr = MatSetOption(C,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr); 

  /* Assembly */
  /* PLAPACK doesn't support INSERT_VALUES mode, zero all entries before calling MatSetValues() */
  ierr = MatZeroEntries(C);CHKERRQ(ierr);
  ierr = MatZeroEntries(Cpetsc);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(C,&Istart,&Iend);CHKERRQ(ierr);
  /* printf(" [%d] C m: %d, Istart/end: %d %d\n",rank,m,Istart,Iend); */
  
  ierr = PetscMalloc((m*M+1)*sizeof(PetscScalar),&array);CHKERRQ(ierr);
  ierr = PetscMalloc2(m,PetscInt,&im,M,PetscInt,&in);CHKERRQ(ierr);
  k = 0;
  for (j=0; j<M; j++){ /* column oriented! */
    in[j] = j;
    for (i=0; i<m; i++){
      im[i] = i+Istart;
      ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
      array[k++] = rval; 
    }
  }
  ierr = MatSetValues(Cpetsc,m,im,M,in,array,ADD_VALUES);CHKERRQ(ierr); 
  ierr = MatSetValues(C,m,im,M,in,array,ADD_VALUES);CHKERRQ(ierr);
  ierr = PetscFree(array);CHKERRQ(ierr);
  ierr = PetscFree2(im,in);CHKERRQ(ierr); 

  ierr = MatAssemblyBegin(Cpetsc,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Cpetsc,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); 
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);   
  /*
  if (!rank) {printf("main, Cpetsc: \n");}
  ierr = MatView(Cpetsc,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
  */
  ierr = MatGetOrdering(C,MATORDERINGNATURAL,&perm,&iperm);CHKERRQ(ierr);

  /* Test nonsymmetric MatMult() */
  ierr = VecGetArray(x,&array);CHKERRQ(ierr);
  for (i=0; i<m; i++){
    ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
    array[i] = rval;                   
  }
  ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);
 
  ierr = MatMult(Cpetsc,x,b);CHKERRQ(ierr);
  ierr = MatMult(C,x,bpla);CHKERRQ(ierr);
  ierr = VecAXPY(bpla,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(bpla,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > 1.e-12 && !rank){
    ierr = PetscPrintf(PETSC_COMM_SELF,"Nonsymmetric MatMult_Plapack error: |b_pla - b|= %g\n",norm);CHKERRQ(ierr);
  }

  /* Test LU Factorization */
  if (nproc == 1){
    ierr = MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
  } else {
    ierr = MatGetFactor(C,MATSOLVERPLAPACK,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
  }
  ierr = MatLUFactorSymbolic(F,C,perm,iperm,&info);CHKERRQ(ierr); 
  for (nfact = 0; nfact < 2; nfact++){
    if (!rank) printf(" LU nfact %d\n",nfact);   
    if (nfact>0){ /* change matrix value for testing repeated MatLUFactorNumeric() */
      if (!rank){ 
        i = j = 0;
        rval = nfact;
        ierr = MatSetValues(Cpetsc,1,&i,1,&j,&rval,ADD_VALUES);CHKERRQ(ierr);   
        ierr = MatSetValues(C,1,&i,1,&j,&rval,ADD_VALUES);CHKERRQ(ierr); 
      } else { /* PLAPACK seems requiring all processors call MatSetValues(), so we add 0.0 on processesses with rank>0! */
        i = j = 0;
        rval = 0.0;
        ierr = MatSetValues(Cpetsc,1,&i,1,&j,&rval,ADD_VALUES);CHKERRQ(ierr);   
        ierr = MatSetValues(C,1,&i,1,&j,&rval,ADD_VALUES);CHKERRQ(ierr); 
      } 
      ierr = MatAssemblyBegin(Cpetsc,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(Cpetsc,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);       
      ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);   
    }    
    ierr = MatLUFactorNumeric(F,C,&info);CHKERRQ(ierr);

    /* Test MatSolve() */
    for (nsolve = 0; nsolve < 2; nsolve++){
      ierr = VecGetArray(x,&array);CHKERRQ(ierr);
      for (i=0; i<m; i++){
        ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
        array[i] = rval;                   
          /* array[i] = rank + 1; */
      }
      ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);
      ierr = VecCopy(x,u);CHKERRQ(ierr); 
      ierr = MatMult(C,x,b);CHKERRQ(ierr);
      ierr = MatSolve(F,b,x);CHKERRQ(ierr); 

      /* Check the error */
      ierr = VecAXPY(u,-1.0,x);CHKERRQ(ierr);  /* u <- (-1.0)x + u */
      ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
      if (norm > tol){
        if (!rank){
          ierr = PetscPrintf(PETSC_COMM_SELF,"Error: Norm of error %g, LU nfact %d\n",norm,nfact);CHKERRQ(ierr);
        }
      }
    }
  } 
  ierr = MatDestroy(&F);CHKERRQ(ierr); 
  
  /* Test non-symmetric operations */
  /*-------------------------------*/
  /* Create a symmetric Plapack dense matrix Csymm */
  ierr = MatCreate(PETSC_COMM_WORLD,&Csymm);CHKERRQ(ierr);
  ierr = MatSetSizes(Csymm,PETSC_DECIDE,PETSC_DECIDE,M,M);CHKERRQ(ierr);
  ierr = MatSetType(Csymm,MATDENSE);CHKERRQ(ierr); 
  ierr = MatSetFromOptions(Csymm);CHKERRQ(ierr);
  ierr = MatSetUp(Csymm);CHKERRQ(ierr);

  ierr = MatSetOption(Csymm,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSetOption(Csymm,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(Csymm,MAT_SYMMETRY_ETERNAL,PETSC_TRUE);CHKERRQ(ierr);

  ierr = MatZeroEntries(Csymm);CHKERRQ(ierr);
  ierr = MatZeroEntries(Cpetsc);CHKERRQ(ierr);
  for (i=Istart; i<Iend; i++){
    for (j=0; j<=i; j++){
      ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
      ierr = MatSetValues(Cpetsc,1,&i,1,&j,&rval,ADD_VALUES);CHKERRQ(ierr); 
      ierr = MatSetValues(Csymm,1,&i,1,&j,&rval,ADD_VALUES);CHKERRQ(ierr);
      if (j<i){ 
        /* Although PLAPACK only requires lower triangular entries, we must add all the entries.
           MatSetValues_Plapack() will ignore the upper triangular entries AFTER an index map! */
        ierr = MatSetValues(Cpetsc,1,&j,1,&i,&rval,ADD_VALUES);CHKERRQ(ierr); 
        ierr = MatSetValues(Csymm,1,&j,1,&i,&rval,ADD_VALUES);CHKERRQ(ierr); 
      }
    }
  }
  ierr = MatAssemblyBegin(Cpetsc,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Cpetsc,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); 
  ierr = MatAssemblyBegin(Csymm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Csymm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  

  /* Test symmetric MatMult() */
  ierr = VecGetArray(x,&array);CHKERRQ(ierr);
  for (i=0; i<m; i++){
    ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
    array[i] = rval;                   
  }
  ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);
 
  ierr = MatMult(Cpetsc,x,b);CHKERRQ(ierr);
  ierr = MatMult(Csymm,x,bpla);CHKERRQ(ierr);
  ierr = VecAXPY(bpla,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(bpla,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > 1.e-12 && !rank){
    ierr = PetscPrintf(PETSC_COMM_SELF,"Symmetric MatMult_Plapack error: |b_pla - b|= %g\n",norm);CHKERRQ(ierr);
  }

  /* Test Cholesky Factorization */
  ierr = MatShift(Csymm,M);CHKERRQ(ierr);  /* make Csymm positive definite */
  if (nproc == 1){
    ierr = MatGetFactor(Csymm,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);
  } else {
    ierr = MatGetFactor(Csymm,MATSOLVERPLAPACK,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);
  }
  ierr = MatCholeskyFactorSymbolic(F,Csymm,perm,&info);CHKERRQ(ierr);
  for (nfact = 0; nfact < 2; nfact++){
    if (!rank) printf(" Cholesky nfact %d\n",nfact);
    ierr = MatCholeskyFactorNumeric(F,Csymm,&info);CHKERRQ(ierr);

    /* Test MatSolve() */
    for (nsolve = 0; nsolve < 2; nsolve++){
      ierr = VecGetArray(x,&array);CHKERRQ(ierr);
      for (i=0; i<m; i++){
        ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
        array[i] = rval; 
      }
      ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);
      ierr = VecCopy(x,u);CHKERRQ(ierr); 
      ierr = MatMult(Csymm,x,b);CHKERRQ(ierr);
      ierr = MatSolve(F,b,x);CHKERRQ(ierr); 

      /* Check the error */
      ierr = VecAXPY(u,-1.0,x);CHKERRQ(ierr);  /* u <- (-1.0)x + u */
      ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
      if (norm > tol){ 
        if (!rank){
          ierr = PetscPrintf(PETSC_COMM_SELF,"Error: Norm of error %g, Cholesky nfact %d\n",norm,nfact);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatDestroy(&F);CHKERRQ(ierr); 

  /* Free data structures */
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = ISDestroy(&iperm);CHKERRQ(ierr);
  
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr); 
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&bpla);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr); 
  
  ierr = MatDestroy(&Cpetsc);CHKERRQ(ierr); 
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&Csymm);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
