
static char help[] = "Bilinear elements on the unit square for Laplacian.  To test the parallel\n\
matrix assembly, the matrix is intentionally laid out across processors\n\
differently from the way it is assembled.  Input arguments are:\n\
  -m <size> : problem size\n\n";

/* Addendum: piggy-backing on this example to test KSPChebyshev methods */

#include <petscksp.h>

int FormElementStiffness(PetscReal H,PetscScalar *Ke)
{
  PetscFunctionBeginUser;
  Ke[0]  = H/6.0;    Ke[1]  = -.125*H; Ke[2]  = H/12.0;   Ke[3]  = -.125*H;
  Ke[4]  = -.125*H;  Ke[5]  = H/6.0;   Ke[6]  = -.125*H;  Ke[7]  = H/12.0;
  Ke[8]  = H/12.0;   Ke[9]  = -.125*H; Ke[10] = H/6.0;    Ke[11] = -.125*H;
  Ke[12] = -.125*H;  Ke[13] = H/12.0;  Ke[14] = -.125*H;  Ke[15] = H/6.0;
  PetscFunctionReturn(0);
}
int FormElementRhs(PetscReal x,PetscReal y,PetscReal H,PetscScalar *r)
{
  PetscFunctionBeginUser;
  r[0] = 0.; r[1] = 0.; r[2] = 0.; r[3] = 0.0;
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            C;
  PetscMPIInt    rank,size;
  PetscInt       i,m = 5,N,start,end,M,its;
  PetscScalar    val,Ke[16],r[4];
  PetscReal      x,y,h,norm;
  PetscInt       idx[4],count,*rows;
  Vec            u,ustar,b;
  KSP            ksp;
  PetscBool      viewkspest = PETSC_FALSE;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-ksp_est_view",&viewkspest,NULL));
  N    = (m+1)*(m+1); /* dimension of matrix */
  M    = m*m; /* number of elements */
  h    = 1.0/m;    /* mesh width */
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Create stiffness matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));
  start = rank*(M/size) + ((M%size) < rank ? (M%size) : rank);
  end   = start + M/size + ((M%size) > rank);

  /* Assemble matrix */
  CHKERRQ(FormElementStiffness(h*h,Ke));   /* element stiffness for Laplacian */
  for (i=start; i<end; i++) {
    /* node numbers for the four corners of element */
    idx[0] = (m+1)*(i/m) + (i % m);
    idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
    CHKERRQ(MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* Create right-hand-side and solution vectors */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&u));
  CHKERRQ(VecSetSizes(u,PETSC_DECIDE,N));
  CHKERRQ(VecSetFromOptions(u));
  CHKERRQ(PetscObjectSetName((PetscObject)u,"Approx. Solution"));
  CHKERRQ(VecDuplicate(u,&b));
  CHKERRQ(PetscObjectSetName((PetscObject)b,"Right hand side"));
  CHKERRQ(VecDuplicate(b,&ustar));
  CHKERRQ(VecSet(u,0.0));
  CHKERRQ(VecSet(b,0.0));

  /* Assemble right-hand-side vector */
  for (i=start; i<end; i++) {
    /* location of lower left corner of element */
    x = h*(i % m); y = h*(i/m);
    /* node numbers for the four corners of element */
    idx[0] = (m+1)*(i/m) + (i % m);
    idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
    CHKERRQ(FormElementRhs(x,y,h*h,r));
    CHKERRQ(VecSetValues(b,4,idx,r,ADD_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(b));
  CHKERRQ(VecAssemblyEnd(b));

  /* Modify matrix and right-hand-side for Dirichlet boundary conditions */
  CHKERRQ(PetscMalloc1(4*m,&rows));
  for (i=0; i<m+1; i++) {
    rows[i]          = i; /* bottom */
    rows[3*m - 1 +i] = m*(m+1) + i; /* top */
  }
  count = m+1; /* left side */
  for (i=m+1; i<m*(m+1); i+= m+1) rows[count++] = i;

  count = 2*m; /* left side */
  for (i=2*m+1; i<m*(m+1); i+= m+1) rows[count++] = i;
  for (i=0; i<4*m; i++) {
    val  = h*(rows[i]/(m+1));
    CHKERRQ(VecSetValues(u,1,&rows[i],&val,INSERT_VALUES));
    CHKERRQ(VecSetValues(b,1,&rows[i],&val,INSERT_VALUES));
  }
  CHKERRQ(MatZeroRows(C,4*m,rows,1.0,0,0));

  CHKERRQ(PetscFree(rows));
  CHKERRQ(VecAssemblyBegin(u));
  CHKERRQ(VecAssemblyEnd(u));
  CHKERRQ(VecAssemblyBegin(b));
  CHKERRQ(VecAssemblyEnd(b));

  { Mat A;
    CHKERRQ(MatConvert(C,MATSAME,MAT_INITIAL_MATRIX,&A));
    CHKERRQ(MatDestroy(&C));
    CHKERRQ(MatConvert(A,MATSAME,MAT_INITIAL_MATRIX,&C));
    CHKERRQ(MatDestroy(&A));
  }

  /* Solve linear system */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,C,C));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSetInitialGuessNonzero(ksp,PETSC_TRUE));
  CHKERRQ(KSPSolve(ksp,b,u));

  if (viewkspest) {
    KSP kspest;

    CHKERRQ(KSPChebyshevEstEigGetKSP(ksp,&kspest));
    if (kspest) CHKERRQ(KSPView(kspest,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Check error */
  CHKERRQ(VecGetOwnershipRange(ustar,&start,&end));
  for (i=start; i<end; i++) {
    val  = h*(i/(m+1));
    CHKERRQ(VecSetValues(ustar,1,&i,&val,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(ustar));
  CHKERRQ(VecAssemblyEnd(ustar));
  CHKERRQ(VecAXPY(u,-1.0,ustar));
  CHKERRQ(VecNorm(u,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g Iterations %D\n",(double)(norm*h),its));

  /* Free work space */
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&ustar));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -pc_type jacobi -ksp_monitor_short -m 5 -ksp_gmres_cgs_refinement_type refine_always

    test:
      suffix: 2
      nsize: 2
      args: -pc_type jacobi -ksp_monitor_short -m 5 -ksp_gmres_cgs_refinement_type refine_always

    test:
      suffix: 2_kokkos
      nsize: 2
      args: -pc_type jacobi -ksp_monitor_short -m 5 -ksp_gmres_cgs_refinement_type refine_always -mat_type aijkokkos -vec_type kokkos
      output_file: output/ex3_2.out
      requires: kokkos_kernels

    test:
      suffix: nocheby
      args: -ksp_est_view

    test:
      suffix: chebynoest
      args: -ksp_est_view -ksp_type chebyshev -ksp_chebyshev_eigenvalues 0.1,1.0

    test:
      suffix: chebyest
      args: -ksp_est_view -ksp_type chebyshev -ksp_chebyshev_esteig
      filter:  sed -e "s/Iterations 19/Iterations 20/g"

TEST*/
