static char help[] = "Tests VecMDot(),VecDot(),VecMTDot(), and VecTDot()\n";

#include <petscvec.h>

int main(int argc, char **argv)
{
  Vec            *V,t;
  PetscInt       i,j,reps,n=15,k=6;
  PetscRandom    rctx;
  PetscScalar    *val_dot,*val_mdot,*tval_dot,*tval_mdot;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test with %" PetscInt_FMT " random vectors of length %" PetscInt_FMT "\n",k,n));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  CHKERRQ(PetscRandomSetFromOptions(rctx));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscRandomSetInterval(rctx,-1.+4.*PETSC_i,1.+5.*PETSC_i));
#else
  CHKERRQ(PetscRandomSetInterval(rctx,-1.,1.));
#endif
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&t));
  CHKERRQ(VecSetSizes(t,n,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(t));
  CHKERRQ(VecDuplicateVecs(t,k,&V));
  CHKERRQ(VecSetRandom(t,rctx));
  CHKERRQ(VecViewFromOptions(t,NULL,"-t_view"));
  CHKERRQ(PetscMalloc1(k,&val_dot));
  CHKERRQ(PetscMalloc1(k,&val_mdot));
  CHKERRQ(PetscMalloc1(k,&tval_dot));
  CHKERRQ(PetscMalloc1(k,&tval_mdot));
  for (i=0; i<k; i++) CHKERRQ(VecSetRandom(V[i],rctx));
  for (reps=0; reps<20; reps++) {
    for (i=1; i<k; i++) {
      CHKERRQ(VecMDot(t,i,V,val_mdot));
      CHKERRQ(VecMTDot(t,i,V,tval_mdot));
      for (j=0;j<i;j++) {
        CHKERRQ(VecDot(t,V[j],&val_dot[j]));
        CHKERRQ(VecTDot(t,V[j],&tval_dot[j]));
      }
      /* Check result */
      for (j=0;j<i;j++) {
        if (PetscAbsScalar(val_mdot[j] - val_dot[j])/PetscAbsScalar(val_dot[j]) > 1e-5) {
          CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "[TEST FAILED] i=%" PetscInt_FMT ", j=%" PetscInt_FMT ", val_mdot[j]=%g, val_dot[j]=%g\n",i,j,(double)PetscAbsScalar(val_mdot[j]), (double)PetscAbsScalar(val_dot[j])));
          break;
        }
        if (PetscAbsScalar(tval_mdot[j] - tval_dot[j])/PetscAbsScalar(tval_dot[j]) > 1e-5) {
          CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "[TEST FAILED] i=%" PetscInt_FMT ", j=%" PetscInt_FMT ", tval_mdot[j]=%g, tval_dot[j]=%g\n",i,j,(double)PetscAbsScalar(tval_mdot[j]), (double)PetscAbsScalar(tval_dot[j])));
          break;
        }
      }
    }
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test completed successfully!\n"));
  CHKERRQ(PetscFree(val_dot));
  CHKERRQ(PetscFree(val_mdot));
  CHKERRQ(PetscFree(tval_dot));
  CHKERRQ(PetscFree(tval_mdot));
  CHKERRQ(VecDestroyVecs(k,&V));
  CHKERRQ(VecDestroy(&t));
  CHKERRQ(PetscRandomDestroy(&rctx));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

   testset:
      output_file: output/ex43_1.out

      test:
         suffix: cuda
         args: -vec_type cuda -random_type curand
         requires: cuda

      test:
         suffix: kokkos
         args: -vec_type kokkos
         requires: kokkos_kernels

      test:
         suffix: hip
         args: -vec_type hip
         requires: hip
TEST*/
