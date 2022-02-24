
static char help[] = "Bilinear elements on the unit square for the Laplacian. Input arguments are:\n\
  -m <size> : problem size\n\n";

#include <petscksp.h>

int FormElementStiffness(PetscReal H,PetscScalar *Ke)
{
  Ke[0]  = H/6.0;    Ke[1]  = -.125*H; Ke[2]  = H/12.0;   Ke[3]  = -.125*H;
  Ke[4]  = -.125*H;  Ke[5]  = H/6.0;   Ke[6]  = -.125*H;  Ke[7]  = H/12.0;
  Ke[8]  = H/12.0;   Ke[9]  = -.125*H; Ke[10] = H/6.0;    Ke[11] = -.125*H;
  Ke[12] = -.125*H;  Ke[13] = H/12.0;  Ke[14] = -.125*H;  Ke[15] = H/6.0;
  return 0;
}

int FormElementRhs(PetscReal x,PetscReal y,PetscReal H,PetscScalar *r)
{
  r[0] = 0.; r[1] = 0.; r[2] = 0.; r[3] = 0.0;
  return 0;
}

/* Note: this code is for testing purposes only. The assembly process is not scalable */
int main(int argc,char **args)
{
  Mat            C;
  PetscErrorCode ierr;
  PetscInt       i,m = 2,N,M,its,idx[4],count,*rows;
  PetscScalar    val,Ke[16],r[4];
  PetscReal      x,y,h,norm;
  Vec            u,ustar,b;
  KSP            ksp;
  PetscMPIInt    rank;
  PetscBool      usezerorows = PETSC_TRUE;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-usezerorows",&usezerorows,NULL));
  N    = (m+1)*(m+1); /* dimension of matrix */
  M    = m*m;         /* number of elements */
  h    = 1.0/m;       /* mesh width */

  /* create stiffness matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatMPIAIJSetPreallocation(C,9,NULL,9,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(C,9,NULL));
#if defined(PETSC_HAVE_HYPRE)
  CHKERRQ(MatHYPRESetPreallocation(C,9,NULL,9,NULL));
#endif

  /* forms the element stiffness for the Laplacian */
  CHKERRQ(FormElementStiffness(h*h,Ke));

  /* assemble the matrix: only process 0 adds the values, not scalable */
  if (!rank) {
    for (i=0; i<M; i++) {
      /* node numbers for the four corners of element */
      idx[0] = (m+1)*(i/m) + (i % m);
      idx[1] = idx[0] + 1;
      idx[2] = idx[1] + m + 1;
      idx[3] = idx[2] - 1;
      if (i == M-1 && !usezerorows) { /* If MatZeroRows not supported -> make it non-singular */
        for (PetscInt ii = 0; ii < 4; ii++) {
          Ke[ 2*4 + ii] = 0.0;
          Ke[ii*4 +  2] = 0.0;
        }
        Ke[10] = 1.0;
      }
      CHKERRQ(MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* create right hand side and solution */
  CHKERRQ(MatCreateVecs(C,&u,&b));
  CHKERRQ(VecDuplicate(u,&ustar));
  CHKERRQ(VecSet(u,0.0));
  CHKERRQ(VecSet(b,0.0));

  /* assemble the right hand side: only process 0 adds the values, not scalable */
  if (!rank) {
    for (i=0; i<M; i++) {
      /* location of lower left corner of element */
      x = h*(i%m);
      y = h*(i/m);
      /* node numbers for the four corners of element */
      idx[0] = (m+1)*(i/m) + (i%m);
      idx[1] = idx[0]+1;
      idx[2] = idx[1]+m+1;
      idx[3] = idx[2]-1;
      CHKERRQ(FormElementRhs(x,y,h*h,r));
      CHKERRQ(VecSetValues(b,4,idx,r,ADD_VALUES));
    }
  }
  CHKERRQ(VecAssemblyBegin(b));
  CHKERRQ(VecAssemblyEnd(b));

  /* modify matrix and rhs for Dirichlet boundary conditions */
  if (!rank) {
    CHKERRQ(PetscMalloc1(4*m+1,&rows));
    for (i=0; i<m+1; i++) {
      rows[i]       = i; /* bottom */
      rows[3*m-1+i] = m*(m+1)+i; /* top */
    }
    count = m+1; /* left side */
    for (i=m+1; i<m*(m+1); i+=m+1) rows[count++] = i;

    count = 2*m; /* right side */
    for (i=2*m+1; i<m*(m+1); i+=m+1) rows[count++] = i;

    for (i=0; i<4*m; i++) {
      val  = h*(rows[i]/(m+1));
      CHKERRQ(VecSetValues(u,1,&rows[i],&val,INSERT_VALUES));
      CHKERRQ(VecSetValues(b,1,&rows[i],&val,INSERT_VALUES));
    }
    if (usezerorows) CHKERRQ(MatZeroRows(C,4*m,rows,1.0,0,0));
    CHKERRQ(PetscFree(rows));
  } else {
    if (usezerorows) CHKERRQ(MatZeroRows(C,0,NULL,1.0,0,0));
  }
  CHKERRQ(VecAssemblyBegin(u));
  CHKERRQ(VecAssemblyEnd(u));
  CHKERRQ(VecAssemblyBegin(b));
  CHKERRQ(VecAssemblyEnd(b));

  if (!usezerorows) {
    CHKERRQ(VecSet(ustar,1.0));
    CHKERRQ(MatMult(C,ustar,b));
  }

  /* solve linear system */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,C,C));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSetInitialGuessNonzero(ksp,PETSC_TRUE));
  CHKERRQ(KSPSolve(ksp,b,u));

  /* check error */
  if (usezerorows) {
    if (!rank) {
      for (i=0; i<N; i++) {
        val  = h*(i/(m+1));
        CHKERRQ(VecSetValues(ustar,1,&i,&val,INSERT_VALUES));
      }
    }
    CHKERRQ(VecAssemblyBegin(ustar));
    CHKERRQ(VecAssemblyEnd(ustar));
  }

  CHKERRQ(VecAXPY(u,-1.0,ustar));
  CHKERRQ(VecNorm(u,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g Iterations %D\n",(double)(norm*h),its));

  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&ustar));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&C));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -ksp_monitor_short -m 5 -pc_type jacobi -ksp_gmres_cgs_refinement_type refine_always

    test:
      suffix: 3
      args: -pc_type sor -pc_sor_symmetric -ksp_monitor_short -m 5 -ksp_gmres_cgs_refinement_type refine_always

    test:
      suffix: 5
      args: -pc_type eisenstat -ksp_monitor_short -m 5  -ksp_gmres_cgs_refinement_type refine_always

    test:
      requires: hypre defined(PETSC_HAVE_HYPRE_DEVICE)
      suffix: hypre_device_none
      output_file: output/ex4_hypre_none.out
      nsize: {{1 2}}
      args: -usezerorows 0 -mat_type hypre -pc_type none -m 5

    test:
      requires: hypre defined(PETSC_HAVE_HYPRE_DEVICE)
      suffix: hypre_device_amg
      nsize: {{1 2}}
      args: -usezerorows 0 -mat_type hypre -pc_type hypre -m 25 -ksp_type cg -ksp_norm_type natural

TEST*/
