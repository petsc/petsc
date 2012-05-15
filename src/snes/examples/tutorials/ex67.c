static char help[] = "Simple test for using advanced discretizations with DMDA\n\n\n";

#include <petscdmda.h>
#include <petscsnes.h>

typedef struct {
  DM            dm;                /* REQUIRED in order to use SNES evaluation functions */
  /* Domain and mesh definition */
  PetscInt      dim;               /* The topological mesh dimension */
  PetscInt      debug;             /* The debugging level */
  PetscLogEvent residualEvent, jacobianEvent, integrateResCPUEvent, integrateJacCPUEvent, integrateJacActionCPUEvent;
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "VecChop"
PetscErrorCode VecChop(Vec v, PetscReal tol)
{
  PetscScalar   *a;
  PetscInt       n, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  ierr = VecGetArray(v, &a);CHKERRQ(ierr);
  for(i = 0; i < n; ++i) {
    if (PetscAbsScalar(a[i]) < tol) a[i] = 0.0;
  }
  ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->dim             = 2;

  ierr = PetscOptionsBegin(comm, "", "DMDA Test Problem Options", "DMDA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex62.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex62.c", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim = user->dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch(dim) {
  case 2:
    ierr = DMDACreate2d(comm, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_STAR, -4, -4, PETSC_DECIDE, PETSC_DECIDE, 2, 1, PETSC_NULL, PETSC_NULL, dm);CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMDACreate3d(comm, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_STAR, -4, -4, -4, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 2, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, dm);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Could not create mesh for dimension %d", dim);
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMDASetFieldName(*dm, 0, "velocity");CHKERRQ(ierr);
  ierr = DMDASetFieldName(*dm, 1, "pressure");CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupSection"
/* This is now Q_1-P_0 */
PetscErrorCode SetupSection(DM dm, AppCtx *user) {
  PetscInt       dim             = user->dim;
  PetscInt       numComp[2]      = {dim, 1};
  PetscInt       numVertexDof[2] = {dim, 0};
  PetscInt       numCellDof[2]   = {0, 1};
  PetscInt       numFaceDof[6];
  PetscInt       d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(d = 0; d < dim; ++d) {
    numFaceDof[0*dim + d] = 0;
    numFaceDof[1*dim + d] = 0;
  }
  ierr = DMDACreateSection(dm, numComp, numVertexDof, numFaceDof, numCellDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
/*
  FormFunctionLocal - Form the local residual F from the local input X

  Input Parameters:
+ dm - The mesh
. X  - Local input vector
- user - The user context

  Output Parameter:
. F  - Local output vector

  Note:
  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

.seealso: FormJacobianLocal()
*/
PetscErrorCode FormFunctionLocal(DM dm, Vec X, Vec F, AppCtx *user)
{
  const PetscInt   debug = user->debug;
  const PetscInt   dim   = user->dim;
  PetscReal       *coords, *v0, *J, *invJ, *detJ;
  PetscScalar     *elemVec, *u;
  const PetscInt   numCells = cEnd - cStart;
  PetscInt         cellDof  = 0;
  PetscInt         maxQuad  = 0;
  PetscInt         jacSize  = 1;
  PetscInt         cStart, cEnd, c, field;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->residualEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = DMDAGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for(field = 0; field < numFields; ++field) {
    PetscInt dof = 1;
    for(d = 0; d < dim; ++d) {dof *= user->q[field].numBasisFuncs*user->q[field].numComponents;}
    cellDof += dof;
    maxQuad  = PetscMax(maxQuad, user->q[field].numQuadPoints);
  }
  for(d = 0; d < dim; ++d) {jacSize *= maxQuad;}
  ierr = PetscMalloc3(dim,PetscReal,&coords,dim,PetscReal,&v0,jacSize,PetscReal,&J);CHKERRQ(ierr);
  ierr = PetscMalloc4(numCells*cellDof,PetscScalar,&u,numCells*jacSize,PetscReal,&invJ,numCells,PetscReal,&detJ,numCells*cellDof,PetscScalar,&elemVec);CHKERRQ(ierr);
  for(c = cStart; c < cEnd; ++c) {
    const PetscScalar *x;
    PetscInt           i;

    ierr = DMDAComputeCellGeometry(dm, c, v0, J, &invJ[c*jacSize], &detJ[c]);CHKERRQ(ierr);
    if (detJ[c] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[c], c);
    ierr = DMDAVecGetClosure(dm, PETSC_NULL, X, c, &x);CHKERRQ(ierr);

    for(i = 0; i < cellDof; ++i) {
      u[c*cellDof+i] = x[i];
    }
  }
  for(field = 0; field < numFields; ++field) {
    const PetscInt numQuadPoints = user->q[field].numQuadPoints;
    const PetscInt numBasisFuncs = user->q[field].numBasisFuncs;
    void (*f0)(PetscScalar u[], const PetscScalar gradU[], PetscScalar f0[]) = user->f0Funcs[field];
    void (*f1)(PetscScalar u[], const PetscScalar gradU[], PetscScalar f1[]) = user->f1Funcs[field];
    /* Conforming batches */
    PetscInt blockSize  = numBasisFuncs*numQuadPoints;
    PetscInt numBlocks  = 1;
    PetscInt batchSize  = numBlocks * blockSize;
    PetscInt numBatches = user->numBatches;
    PetscInt numChunks  = numCells / (numBatches*batchSize);
    ierr = IntegrateResidualBatchCPU(numChunks*numBatches*batchSize, numFields, field, u, invJ, detJ, user->q, f0, f1, elemVec, user);CHKERRQ(ierr);
    /* Remainder */
    PetscInt numRemainder = numCells % (numBatches * batchSize);
    PetscInt offset       = numCells - numRemainder;
    ierr = IntegrateResidualBatchCPU(numRemainder, numFields, field, &u[offset*cellDof], &invJ[offset*dim*dim], &detJ[offset],
                                     user->q, f0, f1, &elemVec[offset*cellDof], user);CHKERRQ(ierr);
  }
  for(c = cStart; c < cEnd; ++c) {
    if (debug) {ierr = DMPrintCellVector(c, "Residual", cellDof, &elemVec[c*cellDof]);CHKERRQ(ierr);}
    ierr = DMComplexVecSetClosure(dm, PETSC_NULL, F, c, &elemVec[c*cellDof], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree4(u,invJ,detJ,elemVec);CHKERRQ(ierr);
  ierr = PetscFree3(coords,v0,J);CHKERRQ(ierr);
  if (user->showResidual) {
    PetscInt p;

    ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual:\n");CHKERRQ(ierr);
    for(p = 0; p < user->numProcs; ++p) {
      if (p == user->rank) {
        Vec f;

        ierr = VecDuplicate(F, &f);CHKERRQ(ierr);
        ierr = VecCopy(F, f);CHKERRQ(ierr);
        ierr = VecChop(f, 1.0e-10);CHKERRQ(ierr);
        ierr = VecView(f, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
        ierr = VecDestroy(&f);CHKERRQ(ierr);
      }
      ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(user->residualEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal"
/*
  FormJacobianLocal - Form the local portion of the Jacobian matrix J from the local input X.

  Input Parameters:
+ dm - The mesh
. X  - Local input vector
- user - The user context

  Output Parameter:
. Jac  - Jacobian matrix

  Note:
  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

.seealso: FormFunctionLocal()
*/
PetscErrorCode FormJacobianLocal(DM dm, Vec X, Mat Jac, Mat JacP, AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->jacobianEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->jacobianEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  SNES           snes;                 /* nonlinear solver */
  Mat            A,J;                  /* Jacobian,preconditioner matrix */
  Vec            u,r;                  /* solution, residual vectors */
  AppCtx         user;                 /* user-defined work context */
  PetscReal      error = 0.0;          /* L_2 error in the solution */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &user.dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, user.dm);CHKERRQ(ierr);

  ierr = SetupSection(user.dm, &user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.dm, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = DMCreateMatrix(user.dm, MATAIJ, &J);CHKERRQ(ierr);
  A    = J;

  ierr = DMSetLocalFunction(user.dm, (DMLocalFunction1) FormFunctionLocal);CHKERRQ(ierr);
  ierr = DMSetLocalJacobian(user.dm, (DMLocalJacobian1) FormJacobianLocal);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, r, SNESDMComputeFunction, &user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, A, J, SNESDMComputeJacobian, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  {
    PetscReal res;

    /* Check discretization error */
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
    ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    /* ierr = ComputeError(u, &error, &user);CHKERRQ(ierr); */
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", error);CHKERRQ(ierr);
    /* Check residual */
    ierr = SNESDMComputeFunction(snes, u, r, &user);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial Residual\n");CHKERRQ(ierr);
    ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
    ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", res);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
