
static char help[] = "Tests MATNEST containing MATSHELL\n\n";


/* ------------------------------------------------------------------------

   This tests if MATNEST properly detects it cannot perform MatMultTranspose()
   when one of its shell matrices cannot perform MatMultTransposeAdd()

   The Jacobian is intentionally wrong to force a DIVERGED_LINE_SEARCH which
   forces a call to SNESNEWTONLSCheckLocalMin_Private() to confirm that
   MatHasOperation(mat,MATOP_MULT_TRANSPOSE,&flg) returns that the nest matrix
   cannot apply the transpose.

   The original bug that was fixed by providing a MatHasOperation_Nest()
   was reported by Manav Bhatia <bhatiamanav@gmail.com>.

  ------------------------------------------------------------------------- */


#include <petscsnes.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormJacobian() and
   FormFunction().
*/
typedef struct {
  PetscReal param;              /* test problem parameter */
  PetscInt  mx;                 /* Discretization in x-direction */
  PetscInt  my;                 /* Discretization in y-direction */
} AppCtx;

PetscErrorCode mymatmult(Mat shell,Vec x,Vec y)
{
  Mat            J;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(shell,&J);CHKERRQ(ierr);
  ierr = MatMult(J,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode mymatdestroy(Mat shell)
{
  Mat            J;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(shell,&J);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   User-defined routines
*/
extern PetscErrorCode FormJacobian(SNES,Vec,Mat,Mat,void*);
extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode FormInitialGuess(AppCtx*,Vec);

int main(int argc,char **argv)
{
  SNES           snes;                 /* nonlinear solver context */
  Vec            x,r;                 /* solution, residual vectors */
  Mat            J,nest,shell;        /* Jacobian matrix */
  AppCtx         user;                 /* user-defined application context */
  PetscErrorCode ierr;
  PetscInt       its,N;
  PetscMPIInt    size;
  PetscReal      bratu_lambda_max = 6.81,bratu_lambda_min = 0.;
  IS             is;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  /*
     Initialize problem parameters
  */
  user.mx = 4; user.my = 4; user.param = 6.0;
  ierr    = PetscOptionsGetInt(NULL,NULL,"-mx",&user.mx,NULL);CHKERRQ(ierr);
  ierr    = PetscOptionsGetInt(NULL,NULL,"-my",&user.my,NULL);CHKERRQ(ierr);
  ierr    = PetscOptionsGetReal(NULL,NULL,"-par",&user.param,NULL);CHKERRQ(ierr);
  if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) SETERRQ(PETSC_COMM_SELF,1,"Lambda is out of range");
  N = user.mx*user.my;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  /*
     Set function evaluation routine and vector.  Whenever the nonlinear
     solver needs to evaluate the nonlinear function, it will call this
     routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        function evaluation routine.
  */
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)&user);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create matrix. Here we only approximately preallocate storage space
     for the Jacobian.  See the users manual for a discussion of better
     techniques for preallocating matrix memory.
  */
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,N,N,5,NULL,&J);CHKERRQ(ierr);
  ierr = MatCreateShell(PETSC_COMM_WORLD,N,N,N,N,J,&shell);CHKERRQ(ierr);
  ierr = MatShellSetOperation(shell,MATOP_MULT,(void(*)(void))mymatmult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(shell,MATOP_DESTROY,(void(*)(void))mymatdestroy);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,user.mx*user.my,0,1,&is);CHKERRQ(ierr);
  ierr = MatCreateNest(PETSC_COMM_WORLD,1,&is,1,&is,&shell,&nest);CHKERRQ(ierr);
  ierr = MatDestroy(&shell);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  ierr = SNESSetJacobian(snes,nest,nest,FormJacobian,(void*)&user);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Set runtime options (e.g., -snes_monitor -snes_rtol <rtol> -ksp_type <type>)
  */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */
  ierr = FormInitialGuess(&user,x);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D\n",its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatDestroy(&nest);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
PetscErrorCode FormInitialGuess(AppCtx *user,Vec X)
{
  PetscInt       i,j,row,mx,my;
  PetscErrorCode ierr;
  PetscReal      lambda,temp1,temp,hx,hy;
  PetscScalar    *x;

  mx     = user->mx;
  my     = user->my;
  lambda = user->param;

  hx = 1.0 / (PetscReal)(mx-1);
  hy = 1.0 / (PetscReal)(my-1);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr  = VecGetArray(X,&x);CHKERRQ(ierr);
  temp1 = lambda/(lambda + 1.0);
  for (j=0; j<my; j++) {
    temp = (PetscReal)(PetscMin(j,my-j-1))*hy;
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        x[row] = 0.0;
        continue;
      }
      x[row] = temp1*PetscSqrtReal(PetscMin((PetscReal)(PetscMin(i,mx-i-1))*hx,temp));
    }
  }

  /*
     Restore vector
  */
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  X - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx            *user = (AppCtx*)ptr;
  PetscInt          i,j,row,mx,my;
  PetscErrorCode    ierr;
  PetscReal         two = 2.0,one = 1.0,lambda,hx,hy,hxdhy,hydhx;
  PetscScalar       ut,ub,ul,ur,u,uxx,uyy,sc,*f;
  const PetscScalar *x;

  mx     = user->mx;
  my     = user->my;
  lambda = user->param;
  hx     = one / (PetscReal)(mx-1);
  hy     = one / (PetscReal)(my-1);
  sc     = hx*hy;
  hxdhy  = hx/hy;
  hydhx  = hy/hx;

  /*
     Get pointers to vector data
  */
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  /*
     Compute function
  */
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        f[row] = x[row];
        continue;
      }
      u      = x[row];
      ub     = x[row - mx];
      ul     = x[row - 1];
      ut     = x[row + mx];
      ur     = x[row + 1];
      uxx    = (-ur + two*u - ul)*hydhx;
      uyy    = (-ut + two*u - ub)*hxdhy;
      f[row] = uxx + uyy - sc*lambda*PetscExpScalar(u);
    }
  }

  /*
     Restore vectors
  */
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   FormJacobian - Evaluates Jacobian matrix.

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  ptr - optional user-defined context, as set by SNESSetJacobian()

   Output Parameters:
.  A - Jacobian matrix
.  B - optionally different preconditioning matrix
.  flag - flag indicating matrix structure
*/
PetscErrorCode FormJacobian(SNES snes,Vec X,Mat nest,Mat jac,void *ptr)
{
  AppCtx            *user = (AppCtx*)ptr;   /* user-defined applicatin context */
  PetscInt          i,j,row,mx,my,col[5];
  PetscErrorCode    ierr;
  PetscScalar       two = 2.0,one = 1.0,lambda,v[5],sc;
  const PetscScalar *x;
  PetscReal         hx,hy,hxdhy,hydhx;
  Mat               J,shell;

  ierr = MatNestGetSubMat(nest,0,0,&shell);CHKERRQ(ierr);
  ierr = MatShellGetContext(shell,&J);CHKERRQ(ierr);

  mx     = user->mx;
  my     = user->my;
  lambda = user->param;
  hx     = 1.0 / (PetscReal)(mx-1);
  hy     = 1.0 / (PetscReal)(my-1);
  sc     = hx*hy;
  hxdhy  = hx/hy;
  hydhx  = hy/hx;

  /*
     Get pointer to vector data
  */
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);

  /*
     Compute entries of the Jacobian
  */
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        ierr = MatSetValues(J,1,&row,1,&row,&one,INSERT_VALUES);CHKERRQ(ierr);
        continue;
      }
      /* the 10 in the next line is intentionally an incorrect addition to the Jacobian */
      v[0] = 10. + -hxdhy; col[0] = row - mx;
      v[1] = -hydhx; col[1] = row - 1;
      v[2] = two*(hydhx + hxdhy) - sc*lambda*PetscExpScalar(x[row]); col[2] = row;
      v[3] = -hydhx; col[3] = row + 1;
      v[4] = -hxdhy; col[4] = row + mx;
      ierr = MatSetValues(J,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /*
     Restore vector
  */
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);

  /*
     Assemble matrix
  */
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(shell,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(shell,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(nest,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(nest,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return 0;
}

