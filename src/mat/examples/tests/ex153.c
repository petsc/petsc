static char help[]="This program illustrates the use of PETSc-fftw interface for sequential real DFT\n";
#include <petscmat.h>
#include <fftw3-mpi.h>
#undef __FUNCT__
#define __FUNCT__ "main"
PetscInt main(PetscInt argc,char **args)
{
  PetscErrorCode  ierr;
  PetscMPIInt     rank,size;
  PetscInt        N0=10,N1=10,N2=10,N3=10,N4=10,N=N0*N1*N2*N3*N4;
  PetscRandom     rdm;
  PetscReal       enorm;
  Vec             x,y,z,input,output;
  Mat             A;
  PetscInt        DIM, dim[5],vsize;
  PetscReal       fac;

  ierr = PetscInitialize(&argc,&args,(char *)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  
  if (size!=1)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "This is a uni-processor example only");
  ierr = PetscRandomCreate(PETSC_COMM_SELF, &rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF,&input);CHKERRQ(ierr);
  ierr = VecSetSizes(input,N,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(input);CHKERRQ(ierr);
  ierr = VecSetRandom(input,rdm);CHKERRQ(ierr);
  ierr = VecDuplicate(input,&output);
 
  DIM = 5; dim[0] = N0; dim[1] = N1; dim[2] = N2; dim[3] = N3; dim[4] = N4;
  ierr = MatCreateFFT(PETSC_COMM_SELF,DIM,dim,MATFFTW,&A);CHKERRQ(ierr);
  ierr = MatGetVecs(A,&x,&y);CHKERRQ(ierr);
  ierr = MatGetVecs(A,&z,PETSC_NULL);CHKERRQ(ierr);

  ierr = VecGetSize(x,&vsize);CHKERRQ(ierr);
  printf("The vector size  of input from the main routine is %d\n",vsize);

  ierr = VecGetSize(z,&vsize);CHKERRQ(ierr);
  printf("The vector size of output from the main routine is %d\n",vsize);

  ierr = InputTransformFFT(A,input,x);CHKERRQ(ierr);

  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  ierr = MatMultTranspose(A,y,z);CHKERRQ(ierr);

  ierr = OutputTransformFFT(A,z,output);CHKERRQ(ierr);
  fac = 1.0/(PetscReal)N;
  ierr = VecScale(output,fac);CHKERRQ(ierr);
/*  
  ierr = VecAssemblyBegin(input);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(input);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(output);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(output);CHKERRQ(ierr);

  ierr = VecView(input,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(output,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
*/
  ierr = VecAXPY(output,-1.0,input);CHKERRQ(ierr);
  ierr = VecNorm(output,NORM_1,&enorm);CHKERRQ(ierr);
//  if (enorm > 1.e-14){
      ierr = PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %e\n",enorm);CHKERRQ(ierr);
//      }

  ierr = VecDestroy(&output);CHKERRQ(ierr);
  ierr = VecDestroy(&input);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  PetscFinalize();
  return 0;

}
