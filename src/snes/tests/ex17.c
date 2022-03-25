static const char help[] = "Newton's method to solve a two-variable system, sequentially.\n"
                           "The same problem is solved twice - i) fully assembled system + ii) block system\n\n";

/*T
Concepts: SNES^basic uniprocessor example, block objects
Processors: 1
T*/

/*
Include "petscsnes.h" so that we can use SNES solvers.  Note that this
file automatically includes:
petscsys.h       - base PETSc routines   petscvec.h - vectors
petscsys.h    - system routines       petscmat.h - matrices
petscis.h     - index sets            petscksp.h - Krylov subspace methods
petscviewer.h - viewers               petscpc.h  - preconditioners
petscksp.h   - linear solvers
*/
#include <petscsnes.h>

/*
This example is block version of the test found at
  ${PETSC_DIR}/src/snes/tutorials/ex1.c
In this test we replace the Jacobian systems
  [J]{x} = {F}
where

[J] = (j_00, j_01),  {x} = (x_0, x_1)^T,   {F} = (f_0, f_1)^T
      (j_10, j_11)
where [J] \in \mathbb^{2 \times 2}, {x},{F} \in \mathbb^{2 \times 1},

with a block system in which each block is of length 1.
i.e. The block system is thus

[J] = ([j00], [j01]),  {x} = ({x0}, {x1})^T, {F} = ({f0}, {f1})^T
      ([j10], [j11])
where
[j00], [j01], [j10], [j11] \in \mathbb^{1 \times 1}
{x0}, {x1}, {f0}, {f1} \in \mathbb^{1 \times 1}

In practice we would not bother defing blocks of size one, and would instead assemble the
full system. This is just a simple test to illustrate how to manipulate the blocks and
to confirm the implementation is correct.
*/

/*
User-defined routines
*/
static PetscErrorCode FormJacobian1(SNES,Vec,Mat,Mat,void*);
static PetscErrorCode FormFunction1(SNES,Vec,Vec,void*);
static PetscErrorCode FormJacobian2(SNES,Vec,Mat,Mat,void*);
static PetscErrorCode FormFunction2(SNES,Vec,Vec,void*);
static PetscErrorCode FormJacobian1_block(SNES,Vec,Mat,Mat,void*);
static PetscErrorCode FormFunction1_block(SNES,Vec,Vec,void*);
static PetscErrorCode FormJacobian2_block(SNES,Vec,Mat,Mat,void*);
static PetscErrorCode FormFunction2_block(SNES,Vec,Vec,void*);

static PetscErrorCode assembled_system(void)
{
  SNES           snes;         /* nonlinear solver context */
  KSP            ksp;         /* linear solver context */
  PC             pc;           /* preconditioner context */
  Vec            x,r;         /* solution, residual vectors */
  Mat            J;            /* Jacobian matrix */
  PetscInt       its;
  PetscScalar    pfive = .5,*xx;
  PetscBool      flg;

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\n========================= Assembled system =========================\n\n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Create nonlinear solver context
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Create matrix and vector data structures; set corresponding routines
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
  Create vectors for solution and nonlinear function
  */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,2,&x));
  PetscCall(VecDuplicate(x,&r));

  /*
  Create Jacobian matrix data structure
  */
  PetscCall(MatCreate(PETSC_COMM_SELF,&J));
  PetscCall(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,2,2));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));

  PetscCall(PetscOptionsHasName(NULL,NULL,"-hard",&flg));
  if (!flg) {
    /*
    Set function evaluation routine and vector.
    */
    PetscCall(SNESSetFunction(snes,r,FormFunction1,NULL));

    /*
    Set Jacobian matrix data structure and Jacobian evaluation routine
    */
    PetscCall(SNESSetJacobian(snes,J,J,FormJacobian1,NULL));
  } else {
    PetscCall(SNESSetFunction(snes,r,FormFunction2,NULL));
    PetscCall(SNESSetJacobian(snes,J,J,FormJacobian2,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Customize nonlinear solver; set runtime options
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
  Set linear solver defaults for this problem. By extracting the
  KSP, KSP, and PC contexts from the SNES context, we can then
  directly call any KSP, KSP, and PC routines to set various options.
  */
  PetscCall(SNESGetKSP(snes,&ksp));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCNONE));
  PetscCall(KSPSetTolerances(ksp,1.e-4,PETSC_DEFAULT,PETSC_DEFAULT,20));

  /*
  Set SNES/KSP/KSP/PC runtime options, e.g.,
  -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
  These options will override those specified above as long as
  SNESSetFromOptions() is called _after_ any other customization
  routines.
  */
  PetscCall(SNESSetFromOptions(snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Evaluate initial guess; then solve nonlinear system
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!flg) {
    PetscCall(VecSet(x,pfive));
  } else {
    PetscCall(VecGetArray(x,&xx));
    xx[0] = 2.0; xx[1] = 3.0;
    PetscCall(VecRestoreArray(x,&xx));
  }
  /*
  Note: The user should initialize the vector, x, with the initial guess
  for the nonlinear solver prior to calling SNESSolve().  In particular,
  to employ an initial guess of zero, the user should explicitly set
  this vector to zero by calling VecSet().
  */

  PetscCall(SNESSolve(snes,NULL,x));
  PetscCall(SNESGetIterationNumber(snes,&its));
  if (flg) {
    Vec f;
    PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(SNESGetFunction(snes,&f,0,0));
    PetscCall(VecView(r,PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"number of SNES iterations = %D\n\n",its));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Free work space.  All PETSc objects should be destroyed when they
  are no longer needed.
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&r));
  PetscCall(MatDestroy(&J)); PetscCall(SNESDestroy(&snes));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
FormFunction1 - Evaluates nonlinear function, F(x).

Input Parameters:
.  snes - the SNES context
.  x - input vector
.  dummy - optional user-defined context (not used here)

Output Parameter:
.  f - function vector
*/
static PetscErrorCode FormFunction1(SNES snes,Vec x,Vec f,void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       *ff;

  PetscFunctionBeginUser;
  /*
  Get pointers to vector data.
  - For default PETSc vectors, VecGetArray() returns a pointer to
  the data array.  Otherwise, the routine is implementation dependent.
  - You MUST call VecRestoreArray() when you no longer need access to
  the array.
  */
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(f,&ff));

  /*
  Compute function
  */
  ff[0] = xx[0]*xx[0] + xx[0]*xx[1] - 3.0;
  ff[1] = xx[0]*xx[1] + xx[1]*xx[1] - 6.0;

  /*
  Restore vectors
  */
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(f,&ff));
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
/*
FormJacobian1 - Evaluates Jacobian matrix.

Input Parameters:
.  snes - the SNES context
.  x - input vector
.  dummy - optional user-defined context (not used here)

Output Parameters:
.  jac - Jacobian matrix
.  B - optionally different preconditioning matrix
.  flag - flag indicating matrix structure
*/
static PetscErrorCode FormJacobian1(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       A[4];
  PetscInt          idx[2] = {0,1};

  PetscFunctionBeginUser;
  /*
  Get pointer to vector data
  */
  PetscCall(VecGetArrayRead(x,&xx));

  /*
  Compute Jacobian entries and insert into matrix.
  - Since this is such a small problem, we set all entries for
  the matrix at once.
  */
  A[0]  = 2.0*xx[0] + xx[1]; A[1] = xx[0];
  A[2]  = xx[1]; A[3] = xx[0] + 2.0*xx[1];
  PetscCall(MatSetValues(jac,2,idx,2,idx,A,INSERT_VALUES));

  /*
  Restore vector
  */
  PetscCall(VecRestoreArrayRead(x,&xx));

  /*
  Assemble matrix
  */
  PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
static PetscErrorCode FormFunction2(SNES snes,Vec x,Vec f,void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       *ff;

  PetscFunctionBeginUser;
  /*
  Get pointers to vector data.
  - For default PETSc vectors, VecGetArray() returns a pointer to
  the data array.  Otherwise, the routine is implementation dependent.
  - You MUST call VecRestoreArray() when you no longer need access to
  the array.
  */
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(f,&ff));

  /*
  Compute function
  */
  ff[0] = PetscSinScalar(3.0*xx[0]) + xx[0];
  ff[1] = xx[1];

  /*
  Restore vectors
  */
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(f,&ff));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
static PetscErrorCode FormJacobian2(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       A[4];
  PetscInt          idx[2] = {0,1};

  PetscFunctionBeginUser;
  /*
  Get pointer to vector data
  */
  PetscCall(VecGetArrayRead(x,&xx));

  /*
  Compute Jacobian entries and insert into matrix.
  - Since this is such a small problem, we set all entries for
  the matrix at once.
  */
  A[0]  = 3.0*PetscCosScalar(3.0*xx[0]) + 1.0; A[1] = 0.0;
  A[2]  = 0.0;                     A[3] = 1.0;
  PetscCall(MatSetValues(jac,2,idx,2,idx,A,INSERT_VALUES));

  /*
  Restore vector
  */
  PetscCall(VecRestoreArrayRead(x,&xx));

  /*
  Assemble matrix
  */
  PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode block_system(void)
{
  SNES           snes;         /* nonlinear solver context */
  KSP            ksp;         /* linear solver context */
  PC             pc;           /* preconditioner context */
  Vec            x,r;         /* solution, residual vectors */
  Mat            J;            /* Jacobian matrix */
  PetscInt       its;
  PetscScalar    pfive = .5;
  PetscBool      flg;

  Mat            j11, j12, j21, j22;
  Vec            x1, x2, r1, r2;
  Vec            bv;
  Vec            bx[2];
  Mat            bA[2][2];

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\n========================= Block system =========================\n\n"));

  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Create matrix and vector data structures; set corresponding routines
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
  Create sub vectors for solution and nonlinear function
  */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,1,&x1));
  PetscCall(VecDuplicate(x1,&r1));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF,1,&x2));
  PetscCall(VecDuplicate(x2,&r2));

  /*
  Create the block vectors
  */
  bx[0] = x1;
  bx[1] = x2;
  PetscCall(VecCreateNest(PETSC_COMM_WORLD,2,NULL,bx,&x));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));
  PetscCall(VecDestroy(&x1));
  PetscCall(VecDestroy(&x2));

  bx[0] = r1;
  bx[1] = r2;
  PetscCall(VecCreateNest(PETSC_COMM_WORLD,2,NULL,bx,&r));
  PetscCall(VecDestroy(&r1));
  PetscCall(VecDestroy(&r2));
  PetscCall(VecAssemblyBegin(r));
  PetscCall(VecAssemblyEnd(r));

  /*
  Create sub Jacobian matrix data structure
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &j11));
  PetscCall(MatSetSizes(j11, 1, 1, 1, 1));
  PetscCall(MatSetType(j11, MATSEQAIJ));
  PetscCall(MatSetUp(j11));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &j12));
  PetscCall(MatSetSizes(j12, 1, 1, 1, 1));
  PetscCall(MatSetType(j12, MATSEQAIJ));
  PetscCall(MatSetUp(j12));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &j21));
  PetscCall(MatSetSizes(j21, 1, 1, 1, 1));
  PetscCall(MatSetType(j21, MATSEQAIJ));
  PetscCall(MatSetUp(j21));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &j22));
  PetscCall(MatSetSizes(j22, PETSC_DECIDE, PETSC_DECIDE, 1, 1));
  PetscCall(MatSetType(j22, MATSEQAIJ));
  PetscCall(MatSetUp(j22));
  /*
  Create block Jacobian matrix data structure
  */
  bA[0][0] = j11;
  bA[0][1] = j12;
  bA[1][0] = j21;
  bA[1][1] = j22;

  PetscCall(MatCreateNest(PETSC_COMM_WORLD,2,NULL,2,NULL,&bA[0][0],&J));
  PetscCall(MatSetUp(J));
  PetscCall(MatNestSetVecType(J,VECNEST));
  PetscCall(MatDestroy(&j11));
  PetscCall(MatDestroy(&j12));
  PetscCall(MatDestroy(&j21));
  PetscCall(MatDestroy(&j22));

  PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));

  PetscCall(PetscOptionsHasName(NULL,NULL,"-hard",&flg));
  if (!flg) {
    /*
    Set function evaluation routine and vector.
    */
    PetscCall(SNESSetFunction(snes,r,FormFunction1_block,NULL));

    /*
    Set Jacobian matrix data structure and Jacobian evaluation routine
    */
    PetscCall(SNESSetJacobian(snes,J,J,FormJacobian1_block,NULL));
  } else {
    PetscCall(SNESSetFunction(snes,r,FormFunction2_block,NULL));
    PetscCall(SNESSetJacobian(snes,J,J,FormJacobian2_block,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Customize nonlinear solver; set runtime options
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
  Set linear solver defaults for this problem. By extracting the
  KSP, KSP, and PC contexts from the SNES context, we can then
  directly call any KSP, KSP, and PC routines to set various options.
  */
  PetscCall(SNESGetKSP(snes,&ksp));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCNONE));
  PetscCall(KSPSetTolerances(ksp,1.e-4,PETSC_DEFAULT,PETSC_DEFAULT,20));

  /*
  Set SNES/KSP/KSP/PC runtime options, e.g.,
  -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
  These options will override those specified above as long as
  SNESSetFromOptions() is called _after_ any other customization
  routines.
  */
  PetscCall(SNESSetFromOptions(snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Evaluate initial guess; then solve nonlinear system
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!flg) {
    PetscCall(VecSet(x,pfive));
  } else {
    Vec *vecs;
    PetscCall(VecNestGetSubVecs(x, NULL, &vecs));
    bv   = vecs[0];
/*    PetscCall(VecBlockGetSubVec(x, 0, &bv)); */
    PetscCall(VecSetValue(bv, 0, 2.0, INSERT_VALUES));  /* xx[0] = 2.0; */
    PetscCall(VecAssemblyBegin(bv));
    PetscCall(VecAssemblyEnd(bv));

/*    PetscCall(VecBlockGetSubVec(x, 1, &bv)); */
    bv   = vecs[1];
    PetscCall(VecSetValue(bv, 0, 3.0, INSERT_VALUES));  /* xx[1] = 3.0; */
    PetscCall(VecAssemblyBegin(bv));
    PetscCall(VecAssemblyEnd(bv));
  }
  /*
  Note: The user should initialize the vector, x, with the initial guess
  for the nonlinear solver prior to calling SNESSolve().  In particular,
  to employ an initial guess of zero, the user should explicitly set
  this vector to zero by calling VecSet().
  */
  PetscCall(SNESSolve(snes,NULL,x));
  PetscCall(SNESGetIterationNumber(snes,&its));
  if (flg) {
    Vec f;
    PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(SNESGetFunction(snes,&f,0,0));
    PetscCall(VecView(r,PETSC_VIEWER_STDOUT_WORLD));
  }

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"number of SNES iterations = %D\n\n",its));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Free work space.  All PETSc objects should be destroyed when they
  are no longer needed.
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&r));
  PetscCall(MatDestroy(&J)); PetscCall(SNESDestroy(&snes));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
static PetscErrorCode FormFunction1_block(SNES snes,Vec x,Vec f,void *dummy)
{
  Vec            *xx, *ff, x1,x2, f1,f2;
  PetscScalar    ff_0, ff_1;
  PetscScalar    xx_0, xx_1;
  PetscInt       index,nb;

  PetscFunctionBeginUser;
  /* get blocks for function */
  PetscCall(VecNestGetSubVecs(f, &nb, &ff));
  f1   = ff[0];  f2 = ff[1];

  /* get blocks for solution */
  PetscCall(VecNestGetSubVecs(x, &nb, &xx));
  x1   = xx[0];  x2 = xx[1];

  /* get solution values */
  index = 0;
  PetscCall(VecGetValues(x1,1, &index, &xx_0));
  PetscCall(VecGetValues(x2,1, &index, &xx_1));

  /* Compute function */
  ff_0 = xx_0*xx_0 + xx_0*xx_1 - 3.0;
  ff_1 = xx_0*xx_1 + xx_1*xx_1 - 6.0;

  /* set function values */
  PetscCall(VecSetValue(f1, index, ff_0, INSERT_VALUES));

  PetscCall(VecSetValue(f2, index, ff_1, INSERT_VALUES));

  PetscCall(VecAssemblyBegin(f));
  PetscCall(VecAssemblyEnd(f));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
static PetscErrorCode FormJacobian1_block(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
  Vec            *xx, x1,x2;
  PetscScalar    xx_0, xx_1;
  PetscInt       index,nb;
  PetscScalar    A_00, A_01, A_10, A_11;
  Mat            j11, j12, j21, j22;
  Mat            **mats;

  PetscFunctionBeginUser;
  /* get blocks for solution */
  PetscCall(VecNestGetSubVecs(x, &nb, &xx));
  x1   = xx[0];  x2 = xx[1];

  /* get solution values */
  index = 0;
  PetscCall(VecGetValues(x1,1, &index, &xx_0));
  PetscCall(VecGetValues(x2,1, &index, &xx_1));

  /* get block matrices */
  PetscCall(MatNestGetSubMats(jac,NULL,NULL,&mats));
  j11  = mats[0][0];
  j12  = mats[0][1];
  j21  = mats[1][0];
  j22  = mats[1][1];

  /* compute jacobian entries */
  A_00 = 2.0*xx_0 + xx_1;
  A_01 = xx_0;
  A_10 = xx_1;
  A_11 = xx_0 + 2.0*xx_1;

  /* set jacobian values */
  PetscCall(MatSetValue(j11, 0,0, A_00, INSERT_VALUES));
  PetscCall(MatSetValue(j12, 0,0, A_01, INSERT_VALUES));
  PetscCall(MatSetValue(j21, 0,0, A_10, INSERT_VALUES));
  PetscCall(MatSetValue(j22, 0,0, A_11, INSERT_VALUES));

  /* Assemble sub matrix */
  PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
static PetscErrorCode FormFunction2_block(SNES snes,Vec x,Vec f,void *dummy)
{
  PetscScalar       *ff;
  const PetscScalar *xx;

  PetscFunctionBeginUser;
  /*
  Get pointers to vector data.
  - For default PETSc vectors, VecGetArray() returns a pointer to
  the data array.  Otherwise, the routine is implementation dependent.
  - You MUST call VecRestoreArray() when you no longer need access to
  the array.
  */
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(f,&ff));

  /*
  Compute function
  */
  ff[0] = PetscSinScalar(3.0*xx[0]) + xx[0];
  ff[1] = xx[1];

  /*
  Restore vectors
  */
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(f,&ff));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
static PetscErrorCode FormJacobian2_block(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       A[4];
  PetscInt          idx[2] = {0,1};

  PetscFunctionBeginUser;
  /*
  Get pointer to vector data
  */
  PetscCall(VecGetArrayRead(x,&xx));

  /*
  Compute Jacobian entries and insert into matrix.
  - Since this is such a small problem, we set all entries for
  the matrix at once.
  */
  A[0]  = 3.0*PetscCosScalar(3.0*xx[0]) + 1.0; A[1] = 0.0;
  A[2]  = 0.0;                     A[3] = 1.0;
  PetscCall(MatSetValues(jac,2,idx,2,idx,A,INSERT_VALUES));

  /*
  Restore vector
  */
  PetscCall(VecRestoreArrayRead(x,&xx));

  /*
  Assemble matrix
  */
  PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscMPIInt    size;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  PetscCall(assembled_system());
  PetscCall(block_system());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -snes_monitor_short
      requires: !single

TEST*/
