/*$Id: ex20.c,v 1.3 1999/06/14 20:49:02 balay Exp bsmith $*/

static char help[] = 
"This example solves a linear system in parallel with SLES.  The matrix\n\
uses simple bilinear elements on the unit square.  To test the parallel\n\
matrix assembly, the matrix is intentionally laid out across processors\n\
differently from the way it is assembled.  Input arguments are:\n\
  -m <size> : problem size\n\n";

#include "sles.h"

#undef __FUNC__
#define __FUNC__ "FormElementStiffness"
int FormElementStiffness(double H,Scalar *Ke)
{
  Ke[0]  = H/6.0;    Ke[1]  = -.125*H; Ke[2]  = H/12.0;   Ke[3]  = -.125*H;
  Ke[4]  = -.125*H;  Ke[5]  = H/6.0;   Ke[6]  = -.125*H;  Ke[7]  = H/12.0;
  Ke[8]  = H/12.0;   Ke[9]  = -.125*H; Ke[10] = H/6.0;    Ke[11] = -.125*H;
  Ke[12] = -.125*H;  Ke[13] = H/12.0;  Ke[14] = -.125*H;  Ke[15] = H/6.0;
  return 0;
}

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C; 
  int         i, m = 5, rank, size, N, start, end, M, its, flg;
  Scalar      zero = 0.0,Ke[16];
  double      h;
  int         ierr, idx[4];
  Vec         u, b;
  SLES        sles;
  KSP         ksp;
  PCNullSpace nullsp;
  PC          pc;
  PetscRandom rand;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg);CHKERRA(ierr);
  N = (m+1)*(m+1); /* dimension of matrix */
  M = m*m; /* number of elements */
  h = 1.0/m;       /* mesh width */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);

  /* Create stiffness matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,&C);CHKERRA(ierr);
  start = rank*(M/size) + ((M%size) < rank ? (M%size) : rank);
  end   = start + M/size + ((M%size) > rank); 

  /* Assemble matrix */
  ierr = FormElementStiffness(h*h,Ke);   /* element stiffness for Laplacian */
  for ( i=start; i<end; i++ ) {
     /* location of lower left corner of element */
     /* node numbers for the four corners of element */
     idx[0] = (m+1)*(i/m) + ( i % m);
     idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
     ierr = MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES);CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* Create right-hand-side and solution vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,N,&u);CHKERRA(ierr); 
  ierr = VecSetFromOptions(u);CHKERRA(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"Approx. Solution");CHKERRA(ierr);
  ierr = VecDuplicate(u,&b);CHKERRA(ierr);
  ierr = PetscObjectSetName((PetscObject)b,"Right hand side");CHKERRA(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&rand);CHKERRA(ierr);
  ierr = VecSetRandom(rand,u);CHKERRA(ierr);
  ierr = PetscRandomDestroy(rand);CHKERRA(ierr);
  ierr = MatMult(C,u,b);CHKERRA(ierr);
  ierr = VecSet(&zero,u);CHKERRA(ierr);

  /* Solve linear system */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);
  ierr = SLESSetOperators(sles,C,C,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);
  ierr = SLESGetKSP(sles,&ksp);CHKERRA(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp);CHKERRA(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-fixnullspace",&flg);CHKERRA(ierr);
  if (flg) {
    ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);
    ierr = PCNullSpaceCreate(PETSC_COMM_WORLD,1,0,PETSC_NULL,&nullsp);CHKERRA(ierr);
    ierr = PCNullSpaceAttach(pc,nullsp);CHKERRA(ierr);
    ierr = PCNullSpaceDestroy(nullsp);CHKERRA(ierr);
  }

  ierr = SLESSolve(sles,b,u,&its);CHKERRA(ierr);


  /* Free work space */
  ierr = SLESDestroy(sles);CHKERRA(ierr);
  ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = MatDestroy(C);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}


