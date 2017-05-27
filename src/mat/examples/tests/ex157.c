static char help[]="This program illustrates the use of PETSc-fftw interface for parallel real DFT\n";
#include <petscmat.h>
#include <fftw3-mpi.h>
PetscInt main(PetscInt argc,char **args)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       N0=2048,N1=2048,N2=3,N3=5,N4=5,N=N0*N1;
  PetscRandom    rdm;
  PetscReal      enorm;
  Vec            x,y,z,input,output;
  Mat            A;
  PetscInt       DIM, dim[5],vsize;
  PetscReal      fac;
  PetscScalar    one=1,two=2,three=3;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "This example requires real numbers");
#endif
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD, &rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&input);CHKERRQ(ierr);
  ierr = VecSetSizes(input,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(input);CHKERRQ(ierr);
/*  ierr = VecSet(input,one);CHKERRQ(ierr); */
/*  ierr = VecSetValue(input,1,two,INSERT_VALUES);CHKERRQ(ierr); */
/*  ierr = VecSetValue(input,2,three,INSERT_VALUES);CHKERRQ(ierr); */
/*  ierr = VecSetValue(input,3,three,INSERT_VALUES);CHKERRQ(ierr); */
  ierr = VecSetRandom(input,rdm);CHKERRQ(ierr);
/*  ierr = VecSetRandom(input,rdm);CHKERRQ(ierr); */
/*  ierr = VecSetRandom(input,rdm);CHKERRQ(ierr); */
  ierr = VecDuplicate(input,&output);

  DIM  = 2; dim[0] = N0; dim[1] = N1; dim[2] = N2; dim[3] = N3; dim[4] = N4;
  ierr = MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A);CHKERRQ(ierr);
  ierr = MatCreateVecsFFTW(A,&x,&y,&z);CHKERRQ(ierr);
/*  ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr); */
/*  ierr = MatCreateVecs(A,&z,NULL);CHKERRQ(ierr); */

  ierr = VecGetSize(x,&vsize);CHKERRQ(ierr);
  printf("The vector size  of input from the main routine is %d\n",vsize);

  ierr = VecGetSize(z,&vsize);CHKERRQ(ierr);
  printf("The vector size of output from the main routine is %d\n",vsize);

  ierr = InputTransformFFT(A,input,x);CHKERRQ(ierr);

  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatMultTranspose(A,y,z);CHKERRQ(ierr);

  ierr = OutputTransformFFT(A,z,output);CHKERRQ(ierr);
  fac  = 1.0/(PetscReal)N;
  ierr = VecScale(output,fac);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(input);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(input);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(output);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(output);CHKERRQ(ierr);

/*  ierr = VecView(input,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
/*  ierr = VecView(output,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  ierr = VecAXPY(output,-1.0,input);CHKERRQ(ierr);
  ierr = VecNorm(output,NORM_1,&enorm);CHKERRQ(ierr);
/*  if (enorm > 1.e-14) { */
  ierr = PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %e\n",enorm);CHKERRQ(ierr);
/*      } */

  ierr = VecDestroy(&output);CHKERRQ(ierr);
  ierr = VecDestroy(&input);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;

}
