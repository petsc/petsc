
#include <stdlib.h>

static char help[] = "Tries to solve u`` + u^{2} = f for an easy case and an impossible case\n\n";

/*
       This example was contributed by Peter Graf to show how SNES fails when given a nonlinear problem with no solution.

       Run with -n 14 to see fail to converge and -n 15 to see convergence

       The option -second_order uses a different discretization of the Neumann boundary condition and always converges
    
*/

#include <petscsnes.h>

int dirichlet_at1 = PETSC_TRUE;
int proper_scaling = PETSC_TRUE;
int second_order = PETSC_FALSE;
#define X0DOT -2 /* -2 */
#define X1 5 /* 5 */
#define X1DOT -3
#define KPOW 2 /* 2 */
const PetscScalar perturb = 1.0; //1.1;

/* 
   User-defined routines
*/
PetscErrorCode FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
PetscErrorCode FormFunction(SNES,Vec,Vec,void*);

int main(int argc,char **argv)
{
  SNES           snes;                /* SNES context */
  Vec            x,r,F;               /* vectors */
  Mat            J,JPrec;             /* Jacobian,preconditioner matrices */
  PetscErrorCode ierr;
  PetscInt       it,n = 11,i;
  PetscMPIInt    size;
  PetscReal      h,xp = 0.0;
  PetscScalar    v;
  

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  //  if (size != 1) SETERRQ(1,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  h = 1.0/(n-1);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecCreate(PETSC_COMM_SELF,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&F);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,r,FormFunction,(void*)F);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structures; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,PETSC_NULL,&J);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,1,PETSC_NULL,&JPrec);CHKERRQ(ierr);

  /*
     Note that in this case we create separate matrices for the Jacobian
     and preconditioner matrix.  Both of these are computed in the
     routine FormJacobian()
  */
  //  ierr = SNESSetJacobian(snes,NULL,JPrec,FormJacobian,0);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,0);CHKERRQ(ierr);
  ///  ierr = SNESSetJacobian(snes,J,JPrec,FormJacobian,0);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /*
     Save all the linear residuals for all the Newton steps; this enables us
     to retain complete convergence history for printing after the conclusion
     of SNESSolve().  Alternatively, one could use the monitoring options
           -snes_monitor -ksp_monitor
     to see this information during the solver's execution; however, such
     output during the run distorts performance evaluation data.  So, the
     following is a good option when monitoring code performance, for example
     when using -log_summary.
  */


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize application:
     Store right-hand-side of PDE and exact solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  const PetscScalar a = X0DOT;
  const PetscScalar b = X1;
  const PetscScalar k = KPOW;
  PetscScalar v2;
#define SQR(x) ((x)*(x))
  xp = 0.0;
  printf ("setting rhs for u'(0)=a=%e, u(1)=b=%e, exponent k=%e\n", a,b,k);
  for (i=0; i<n; i++) 
    {
      //    v = 6.0*xp + pow(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
      v = k*(k-1)*(b-a)*PetscPowScalar(xp,k-2) + SQR(a*xp) + SQR(b-a)*PetscPowScalar(xp,2*k) + 2*a*(b-a)*PetscPowScalar(xp,k+1);
      ierr = VecSetValues(F,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
      v2 = a*xp + (b-a)*PetscPowScalar(xp,k);
      printf ("%e %e %e\n", xp, v2, v);
      ierr = VecSetValues(x,1,&i,&v2,INSERT_VALUES);CHKERRQ(ierr);
      xp += h;
    }


  // perturb
   PetscScalar *xx;
  ierr = VecGetArray(x,&xx);
  for (i=0; i<n; i++) 
    {
      v2 = xx[i]*perturb;
      ierr = VecSetValues(x,1,&i,&v2,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr); 

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  
  ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&it);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Newton iterations = %D\n\n",it);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecDestroy(&x);CHKERRQ(ierr);     ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);     ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = MatDestroy(&JPrec);CHKERRQ(ierr); ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}
/* ------------------------------------------------------------------- */
/* 
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
PetscErrorCode FormFunction(SNES snes,Vec x,Vec f,void *dummy)
{
  PetscScalar    *xx,*ff,*FF,d,d2,g;
  PetscErrorCode ierr;
  PetscInt       i,n;

  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);
  ierr = VecGetArray((Vec)dummy,&FF);CHKERRQ(ierr);
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  d = (PetscReal)(n - 1); d2 = d*d;
  if (proper_scaling)
    g = d;
  else
    g = 1.0;
  
  //  ff[0]   = 1 + xx[0]*xx[0];   //////THIS SHOWS in what manner SNES FAILS WHEN GIVEN UNSOLVABLE PROBLEM: WILL TELL YOU "line search failure"
  //  ff[0]   = xx[0]*xx[0];
  if (second_order)
    {
      ff[0] = g*(0.5*d*(-xx[2] + 4*xx[1] - 3*xx[0]) - X0DOT);
    }
  else
    ff[0] = g*(d*(xx[1] - xx[0]) - X0DOT);  // x'(0) = X0DOT


  for (i=1; i<n-1; i++) {
    ff[i] = d2*(xx[i-1] - 2.0*xx[i] + xx[i+1]) + xx[i]*xx[i] - FF[i];
  }
  if (dirichlet_at1)
    ff[n-1] = g*g*(xx[n-1] - X1);
  else
    ff[n-1] = g*(d*(xx[n-1] - xx[n-2]) - X1DOT);  // x'(1)=X1DOT
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr);
  ierr = VecRestoreArray((Vec)dummy,&FF);CHKERRQ(ierr);
  return 0;
}


/* ------------------------------------------------------------------- */
/*
   FormJacobian - This routine demonstrates the use of different
   matrices for the Jacobian and preconditioner

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  ptr - optional user-defined context, as set by SNESSetJacobian()

   Output Parameters:
.  A - Jacobian matrix
.  B - different preconditioning matrix
.  flag - flag indicating matrix structure
*/
PetscErrorCode FormJacobian(SNES snes,Vec x,Mat *jac,Mat *prejac,MatStructure *flag,void *dummy)
{
  PetscScalar    *xx,A[3],d,d2,g;
  PetscInt       i,n,j[3];
  PetscErrorCode ierr;
  PetscBool     nouse_jac;

  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  d = (PetscReal)(n - 1); d2 = d*d;
  if (proper_scaling)
    g = d;
  else
    g = 1.0;

  ierr = PetscOptionsHasName(PETSC_NULL,"-snes_mf_operator", &nouse_jac);CHKERRQ(ierr);
  //  nouse_jac = (PetscBool)0;

  /* Form Jacobian.  Also form a different preconditioning matrix that 
     has only the diagonal elements. */
  i = 0; 
  //A[0] = 1.0; 
  A[0] = xx[0]; 
  if (! nouse_jac)
    {
      if (second_order)
	{
	  //	  ff[0] = g*(0.5*d*(-xx[2] + 4*xx[1] - 3*xx[0]));
	  j[0] = 0; j[1] = 1; j[2] = 2;
	  A[0] = -3*d*g*0.5; A[1] = 4*d*g*0.5;  A[2] = -1*d*g*0.5;   
	  ierr = MatSetValues(*jac,1,&i,3,j,A,INSERT_VALUES);CHKERRQ(ierr);
	}
      else
	{
	  j[0] = 0; j[1] = 1;
	  A[0] = -d*g; A[1] = d*g;        // from x'(0) ~ (x[1] - x[0])/h          
	  ierr = MatSetValues(*jac,1,&i,2,j,A,INSERT_VALUES);CHKERRQ(ierr);
	}
      //      ierr = MatSetValues(*jac,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);
    }
  ierr = MatSetValues(*prejac,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);
  for (i=1; i<n-1; i++) 
    {
      j[0] = i - 1; j[1] = i;                   j[2] = i + 1; 
      A[0] = d2;     A[1] = -2.0*d2 + 2.0*xx[i];  A[2] = d2; 
      //        A[0] = 0;     A[1] = -2.0*d + 2.0*xx[i];  A[2] = 0; 
      if (! nouse_jac)
	ierr = MatSetValues(*jac,1,&i,3,j,A,INSERT_VALUES);CHKERRQ(ierr);
      /// ierr = MatSetValues(*prejac,1,&i,1,&i,&A[1],INSERT_VALUES);CHKERRQ(ierr);
    }

  i = n-1; 
  if (dirichlet_at1)
    {
      A[0] = g*g; 
      if (! nouse_jac)
	ierr = MatSetValues(*jac,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);
      //ierr = MatSetValues(*prejac,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);
    }
  else
    {
      j[0] = n-2;
      j[1] = n-1;
      A[0] = -d*g; A[1] = d*g;        // from x'(1) ~ (x[n-1] - x[n-2])/h          
      if (! nouse_jac)
	ierr = MatSetValues(*jac,1,&i,2,j,A,INSERT_VALUES);CHKERRQ(ierr);
    }

  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*prejac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*prejac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;

  return 0;
}
