
static char help[] = "Bilinear elements on the unit square for Laplacian.  To test the parallel\n\
matrix assembly,the matrix is intentionally laid out across processors\n\
differently from the way it is assembled.  Input arguments are:\n\
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

int main(int argc,char **args)
{
  Mat            C;
  PetscMPIInt    rank,size;
  PetscInt       i,m = 5,N,start,end,M;
  PetscInt       idx[4];
  PetscScalar    Ke[16];
  PetscReal      h;
  Vec            u,b;
  KSP            ksp;
  MatNullSpace   nullsp;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  N    = (m+1)*(m+1); /* dimension of matrix */
  M    = m*m; /* number of elements */
  h    = 1.0/m;    /* mesh width */
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Create stiffness matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  start = rank*(M/size) + ((M%size) < rank ? (M%size) : rank);
  end   = start + M/size + ((M%size) > rank);

  /* Assemble matrix */
  PetscCall(FormElementStiffness(h*h,Ke));   /* element stiffness for Laplacian */
  for (i=start; i<end; i++) {
    /* location of lower left corner of element */
    /* node numbers for the four corners of element */
    idx[0] = (m+1)*(i/m) + (i % m);
    idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
    PetscCall(MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* Create right-hand-side and solution vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
  PetscCall(VecSetSizes(u,PETSC_DECIDE,N));
  PetscCall(VecSetFromOptions(u));
  PetscCall(PetscObjectSetName((PetscObject)u,"Approx. Solution"));
  PetscCall(VecDuplicate(u,&b));
  PetscCall(PetscObjectSetName((PetscObject)b,"Right hand side"));

  PetscCall(VecSet(b,1.0));
  PetscCall(VecSetValue(b,0,1.2,ADD_VALUES));
  PetscCall(VecSet(u,0.0));

  /* Solve linear system */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,C,C));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetInitialGuessNonzero(ksp,PETSC_TRUE));

  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nullsp));
  /*
     The KSP solver will remove this nullspace from the solution at each iteration
  */
  PetscCall(MatSetNullSpace(C,nullsp));
  /*
     The KSP solver will remove from the right hand side any portion in this nullspace, thus making the linear system consistent.
  */
  PetscCall(MatSetTransposeNullSpace(C,nullsp));
  PetscCall(MatNullSpaceDestroy(&nullsp));

  PetscCall(KSPSolve(ksp,b,u));

  /* Free work space */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -ksp_monitor_short

TEST*/
