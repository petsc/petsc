/*$Id: ex20.c,v 1.19 2001/08/07 21:30:50 bsmith Exp $*/

static char help[] = "This example solves a linear system in parallel with SLES.  The matrix\n\
uses simple bilinear elements on the unit square.  To test the parallel\n\
matrix assembly,the matrix is intentionally laid out across processors\n\
differently from the way it is assembled.  Input arguments are:\n\
  -m <size> : problem size\n\n";

#include "petscsles.h"

#undef __FUNCT__
#define __FUNCT__ "FormElementStiffness"
int FormElementStiffness(PetscReal H,PetscScalar *Ke)
{
  Ke[0]  = H/6.0;    Ke[1]  = -.125*H; Ke[2]  = H/12.0;   Ke[3]  = -.125*H;
  Ke[4]  = -.125*H;  Ke[5]  = H/6.0;   Ke[6]  = -.125*H;  Ke[7]  = H/12.0;
  Ke[8]  = H/12.0;   Ke[9]  = -.125*H; Ke[10] = H/6.0;    Ke[11] = -.125*H;
  Ke[12] = -.125*H;  Ke[13] = H/12.0;  Ke[14] = -.125*H;  Ke[15] = H/6.0;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat          C; 
  int          i,m = 5,rank,size,N,start,end,M;
  int          ierr,idx[4];
  PetscTruth   flg;
  PetscScalar  zero = 0.0,Ke[16], one = 1.0;
  PetscReal    h;
  Vec          u,b;
  SLES         sles;
  KSP          ksp;
  MatNullSpace nullsp;
  PC           pc;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  N = (m+1)*(m+1); /* dimension of matrix */
  M = m*m; /* number of elements */
  h = 1.0/m;       /* mesh width */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* Create stiffness matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,&C);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  start = rank*(M/size) + ((M%size) < rank ? (M%size) : rank);
  end   = start + M/size + ((M%size) > rank); 

  /* Assemble matrix */
  ierr = FormElementStiffness(h*h,Ke);   /* element stiffness for Laplacian */
  for (i=start; i<end; i++) {
     /* location of lower left corner of element */
     /* node numbers for the four corners of element */
     idx[0] = (m+1)*(i/m) + (i % m);
     idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
     ierr = MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Create right-hand-side and solution vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr); 
  ierr = VecSetSizes(u,PETSC_DECIDE,N);CHKERRQ(ierr); 
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"Approx. Solution");CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)b,"Right hand side");CHKERRQ(ierr);

  ierr = VecSet(&one,u);CHKERRQ(ierr);
  ierr = MatMult(C,u,b);CHKERRQ(ierr);
  ierr = VecSet(&zero,u);CHKERRQ(ierr);

  /* Solve linear system */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRQ(ierr);
  ierr = SLESSetOperators(sles,C,C,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRQ(ierr);
  ierr = SLESGetKSP(sles,&ksp);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-fixnullspace",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,1,0,PETSC_NULL,&nullsp);CHKERRQ(ierr);
    ierr = PCNullSpaceAttach(pc,nullsp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(nullsp);CHKERRQ(ierr);
  }

  ierr = SLESSolve(sles,b,u);CHKERRQ(ierr);


  /* Free work space */
  ierr = SLESDestroy(sles);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}


