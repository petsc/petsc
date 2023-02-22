#include "wash.h"

/* Subroutines for Pipe                                  */
/* -------------------------------------------------------*/

/*
   PipeCreate - Create Pipe object.

   Input Parameters:
   comm - MPI communicator

   Output Parameter:
.  pipe - location to put the PIPE context
*/
PetscErrorCode PipeCreate(MPI_Comm comm, Pipe *pipe)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(pipe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PipeDestroy - Destroy Pipe object.

   Input Parameters:
   pipe - Reference to pipe intended to be destroyed.
*/
PetscErrorCode PipeDestroy(Pipe *pipe)
{
  PetscFunctionBegin;
  if (!*pipe) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PipeDestroyJacobian(*pipe));
  PetscCall(VecDestroy(&(*pipe)->x));
  PetscCall(DMDestroy(&(*pipe)->da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PipeSetParameters - Set parameters for Pipe context

   Input Parameter:
+  pipe - PIPE object
.  length -
.  nnodes -
.  D -
.  a -
-  fric -
*/
PetscErrorCode PipeSetParameters(Pipe pipe, PetscReal length, PetscReal D, PetscReal a, PetscReal fric)
{
  PetscFunctionBegin;
  pipe->length = length;
  pipe->D      = D;
  pipe->a      = a;
  pipe->fric   = fric;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    PipeSetUp - Set up pipe based on set parameters.
*/
PetscErrorCode PipeSetUp(Pipe pipe)
{
  DMDALocalInfo info;

  PetscFunctionBegin;
  PetscCall(DMDACreate1d(PETSC_COMM_SELF, DM_BOUNDARY_GHOSTED, pipe->nnodes, 2, 1, NULL, &pipe->da));
  PetscCall(DMSetFromOptions(pipe->da));
  PetscCall(DMSetUp(pipe->da));
  PetscCall(DMDASetFieldName(pipe->da, 0, "Q"));
  PetscCall(DMDASetFieldName(pipe->da, 1, "H"));
  PetscCall(DMDASetUniformCoordinates(pipe->da, 0, pipe->length, 0, 0, 0, 0));
  PetscCall(DMCreateGlobalVector(pipe->da, &(pipe->x)));

  PetscCall(DMDAGetLocalInfo(pipe->da, &info));

  pipe->rad = pipe->D / 2;
  pipe->A   = PETSC_PI * pipe->rad * pipe->rad;
  pipe->R   = pipe->fric / (2 * pipe->D * pipe->A);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    PipeCreateJacobian - Create Jacobian matrix structures for a Pipe.

    Collective

    Input Parameter:
+   pipe - the Pipe object
-   Jin - array of three constructed Jacobian matrices to be reused. Set NULL if it is not available

    Output Parameter:
.   J  - array of three empty Jacobian matrices

    Level: beginner
*/
PetscErrorCode PipeCreateJacobian(Pipe pipe, Mat *Jin, Mat *J[])
{
  Mat         *Jpipe;
  PetscInt     M, rows[2], cols[2], *nz;
  PetscScalar *aa;

  PetscFunctionBegin;
  if (Jin) {
    *J             = Jin;
    pipe->jacobian = Jin;
    PetscCall(PetscObjectReference((PetscObject)(Jin[0])));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscMalloc1(3, &Jpipe));

  /* Jacobian for this pipe */
  PetscCall(DMSetMatrixStructureOnly(pipe->da, PETSC_TRUE));
  PetscCall(DMCreateMatrix(pipe->da, &Jpipe[0]));
  PetscCall(DMSetMatrixStructureOnly(pipe->da, PETSC_FALSE));

  /* Jacobian for upstream vertex */
  PetscCall(MatGetSize(Jpipe[0], &M, NULL));
  PetscCall(PetscCalloc2(M, &nz, 4, &aa));

  PetscCall(MatCreate(PETSC_COMM_SELF, &Jpipe[1]));
  PetscCall(MatSetSizes(Jpipe[1], PETSC_DECIDE, PETSC_DECIDE, M, 2));
  PetscCall(MatSetFromOptions(Jpipe[1]));
  PetscCall(MatSetOption(Jpipe[1], MAT_STRUCTURE_ONLY, PETSC_TRUE));
  nz[0]   = 2;
  nz[1]   = 2;
  rows[0] = 0;
  rows[1] = 1;
  cols[0] = 0;
  cols[1] = 1;
  PetscCall(MatSeqAIJSetPreallocation(Jpipe[1], 0, nz));
  PetscCall(MatSetValues(Jpipe[1], 2, rows, 2, cols, aa, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(Jpipe[1], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpipe[1], MAT_FINAL_ASSEMBLY));

  /* Jacobian for downstream vertex */
  PetscCall(MatCreate(PETSC_COMM_SELF, &Jpipe[2]));
  PetscCall(MatSetSizes(Jpipe[2], PETSC_DECIDE, PETSC_DECIDE, M, 2));
  PetscCall(MatSetFromOptions(Jpipe[2]));
  PetscCall(MatSetOption(Jpipe[2], MAT_STRUCTURE_ONLY, PETSC_TRUE));
  nz[0]     = 0;
  nz[1]     = 0;
  nz[M - 2] = 2;
  nz[M - 1] = 2;
  rows[0]   = M - 2;
  rows[1]   = M - 1;
  PetscCall(MatSeqAIJSetPreallocation(Jpipe[2], 0, nz));
  PetscCall(MatSetValues(Jpipe[2], 2, rows, 2, cols, aa, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(Jpipe[2], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpipe[2], MAT_FINAL_ASSEMBLY));

  PetscCall(PetscFree2(nz, aa));

  *J             = Jpipe;
  pipe->jacobian = Jpipe;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PipeDestroyJacobian(Pipe pipe)
{
  Mat     *Jpipe = pipe->jacobian;
  PetscInt i;

  PetscFunctionBegin;
  if (Jpipe) {
    for (i = 0; i < 3; i++) PetscCall(MatDestroy(&Jpipe[i]));
  }
  PetscCall(PetscFree(Jpipe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    JunctionCreateJacobian - Create Jacobian matrices for a vertex.

    Collective

    Input Parameter:
+   dm - the DMNetwork object
.   v - vertex point
-   Jin - Jacobian patterns created by JunctionCreateJacobianSample() for reuse

    Output Parameter:
.   J  - array of Jacobian matrices (see dmnetworkimpl.h)

    Level: beginner
*/
PetscErrorCode JunctionCreateJacobian(DM dm, PetscInt v, Mat *Jin, Mat *J[])
{
  Mat            *Jv;
  PetscInt        nedges, e, i, M, N, *rows, *cols;
  PetscBool       isSelf;
  const PetscInt *edges, *cone;
  PetscScalar    *zeros;

  PetscFunctionBegin;
  /* Get array size of Jv */
  PetscCall(DMNetworkGetSupportingEdges(dm, v, &nedges, &edges));
  PetscCheck(nedges > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "%" PetscInt_FMT " vertex, nedges %" PetscInt_FMT, v, nedges);

  /* two Jacobians for each connected edge: J(v,e) and J(v,vc); adding J(v,v), total 2*nedges+1 Jacobians */
  PetscCall(PetscCalloc1(2 * nedges + 1, &Jv));

  /* Create dense zero block for this vertex: J[0] = Jacobian(v,v) */
  PetscCall(DMNetworkGetComponent(dm, v, -1, NULL, NULL, &M));
  PetscCheck(M == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "%" PetscInt_FMT " != 2", M);
  PetscCall(PetscMalloc3(M, &rows, M, &cols, M * M, &zeros));
  PetscCall(PetscArrayzero(zeros, M * M));
  for (i = 0; i < M; i++) rows[i] = i;

  for (e = 0; e < nedges; e++) {
    /* create Jv[2*e+1] = Jacobian(v,e), e: supporting edge */
    PetscCall(DMNetworkGetConnectedVertices(dm, edges[e], &cone));
    isSelf = (v == cone[0]) ? PETSC_TRUE : PETSC_FALSE;

    if (Jin) {
      if (isSelf) {
        Jv[2 * e + 1] = Jin[0];
      } else {
        Jv[2 * e + 1] = Jin[1];
      }
      Jv[2 * e + 2] = Jin[2];
      PetscCall(PetscObjectReference((PetscObject)(Jv[2 * e + 1])));
      PetscCall(PetscObjectReference((PetscObject)(Jv[2 * e + 2])));
    } else {
      /* create J(v,e) */
      PetscCall(MatCreate(PETSC_COMM_SELF, &Jv[2 * e + 1]));
      PetscCall(DMNetworkGetComponent(dm, edges[e], -1, NULL, NULL, &N));
      PetscCall(MatSetSizes(Jv[2 * e + 1], PETSC_DECIDE, PETSC_DECIDE, M, N));
      PetscCall(MatSetFromOptions(Jv[2 * e + 1]));
      PetscCall(MatSetOption(Jv[2 * e + 1], MAT_STRUCTURE_ONLY, PETSC_TRUE));
      PetscCall(MatSeqAIJSetPreallocation(Jv[2 * e + 1], 2, NULL));
      if (N) {
        if (isSelf) { /* coupling at upstream */
          for (i = 0; i < 2; i++) cols[i] = i;
        } else { /* coupling at downstream */
          cols[0] = N - 2;
          cols[1] = N - 1;
        }
        PetscCall(MatSetValues(Jv[2 * e + 1], 2, rows, 2, cols, zeros, INSERT_VALUES));
      }
      PetscCall(MatAssemblyBegin(Jv[2 * e + 1], MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(Jv[2 * e + 1], MAT_FINAL_ASSEMBLY));

      /* create Jv[2*e+2] = Jacobian(v,vc), vc: connected vertex.
       In WashNetwork, v and vc are not connected, thus Jacobian(v,vc) is empty */
      PetscCall(MatCreate(PETSC_COMM_SELF, &Jv[2 * e + 2]));
      PetscCall(MatSetSizes(Jv[2 * e + 2], PETSC_DECIDE, PETSC_DECIDE, M, M)); /* empty matrix, sizes can be arbitrary */
      PetscCall(MatSetFromOptions(Jv[2 * e + 2]));
      PetscCall(MatSetOption(Jv[2 * e + 2], MAT_STRUCTURE_ONLY, PETSC_TRUE));
      PetscCall(MatSeqAIJSetPreallocation(Jv[2 * e + 2], 1, NULL));
      PetscCall(MatAssemblyBegin(Jv[2 * e + 2], MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(Jv[2 * e + 2], MAT_FINAL_ASSEMBLY));
    }
  }
  PetscCall(PetscFree3(rows, cols, zeros));

  *J = Jv;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode JunctionDestroyJacobian(DM dm, PetscInt v, Junction junc)
{
  Mat            *Jv = junc->jacobian;
  const PetscInt *edges;
  PetscInt        nedges, e;

  PetscFunctionBegin;
  if (!Jv) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(DMNetworkGetSupportingEdges(dm, v, &nedges, &edges));
  for (e = 0; e < nedges; e++) {
    PetscCall(MatDestroy(&Jv[2 * e + 1]));
    PetscCall(MatDestroy(&Jv[2 * e + 2]));
  }
  PetscCall(PetscFree(Jv));
  PetscFunctionReturn(PETSC_SUCCESS);
}
