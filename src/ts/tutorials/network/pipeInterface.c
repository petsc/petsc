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
PetscErrorCode PipeCreate(MPI_Comm comm,Pipe *pipe)
{
  PetscFunctionBegin;
  CHKERRQ(PetscNew(pipe));
  PetscFunctionReturn(0);
}

/*
   PipeDestroy - Destroy Pipe object.

   Input Parameters:
   pipe - Reference to pipe intended to be destroyed.
*/
PetscErrorCode PipeDestroy(Pipe *pipe)
{
  PetscFunctionBegin;
  if (!*pipe) PetscFunctionReturn(0);

  CHKERRQ(PipeDestroyJacobian(*pipe));
  CHKERRQ(VecDestroy(&(*pipe)->x));
  CHKERRQ(DMDestroy(&(*pipe)->da));
  PetscFunctionReturn(0);
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
PetscErrorCode PipeSetParameters(Pipe pipe,PetscReal length,PetscReal D,PetscReal a,PetscReal fric)
{
  PetscFunctionBegin;
  pipe->length = length;
  pipe->D      = D;
  pipe->a      = a;
  pipe->fric   = fric;
  PetscFunctionReturn(0);
}

/*
    PipeSetUp - Set up pipe based on set parameters.
*/
PetscErrorCode PipeSetUp(Pipe pipe)
{
  DMDALocalInfo  info;

  PetscFunctionBegin;
  CHKERRQ(DMDACreate1d(PETSC_COMM_SELF, DM_BOUNDARY_GHOSTED, pipe->nnodes, 2, 1, NULL, &pipe->da));
  CHKERRQ(DMSetFromOptions(pipe->da));
  CHKERRQ(DMSetUp(pipe->da));
  CHKERRQ(DMDASetFieldName(pipe->da, 0, "Q"));
  CHKERRQ(DMDASetFieldName(pipe->da, 1, "H"));
  CHKERRQ(DMDASetUniformCoordinates(pipe->da, 0, pipe->length, 0, 0, 0, 0));
  CHKERRQ(DMCreateGlobalVector(pipe->da, &(pipe->x)));

  CHKERRQ(DMDAGetLocalInfo(pipe->da, &info));

  pipe->rad = pipe->D / 2;
  pipe->A   = PETSC_PI*pipe->rad*pipe->rad;
  pipe->R   = pipe->fric / (2*pipe->D*pipe->A);
  PetscFunctionReturn(0);
}

/*
    PipeCreateJacobian - Create Jacobian matrix structures for a Pipe.

    Collective on Pipe

    Input Parameter:
+   pipe - the Pipe object
-   Jin - array of three constructed Jacobian matrices to be reused. Set NULL if it is not available

    Output Parameter:
.   J  - array of three empty Jacobian matrices

    Level: beginner
*/
PetscErrorCode PipeCreateJacobian(Pipe pipe,Mat *Jin,Mat *J[])
{
  Mat            *Jpipe;
  PetscInt       M,rows[2],cols[2],*nz;
  PetscScalar    *aa;

  PetscFunctionBegin;
  if (Jin) {
    *J = Jin;
    pipe->jacobian = Jin;
    CHKERRQ(PetscObjectReference((PetscObject)(Jin[0])));
    PetscFunctionReturn(0);
  }
  CHKERRQ(PetscMalloc1(3,&Jpipe));

  /* Jacobian for this pipe */
  CHKERRQ(DMSetMatrixStructureOnly(pipe->da,PETSC_TRUE));
  CHKERRQ(DMCreateMatrix(pipe->da,&Jpipe[0]));
  CHKERRQ(DMSetMatrixStructureOnly(pipe->da,PETSC_FALSE));

  /* Jacobian for upstream vertex */
  CHKERRQ(MatGetSize(Jpipe[0],&M,NULL));
  CHKERRQ(PetscCalloc2(M,&nz,4,&aa));

  CHKERRQ(MatCreate(PETSC_COMM_SELF,&Jpipe[1]));
  CHKERRQ(MatSetSizes(Jpipe[1],PETSC_DECIDE,PETSC_DECIDE,M,2));
  CHKERRQ(MatSetFromOptions(Jpipe[1]));
  CHKERRQ(MatSetOption(Jpipe[1],MAT_STRUCTURE_ONLY,PETSC_TRUE));
  nz[0] = 2; nz[1] = 2;
  rows[0] = 0; rows[1] = 1;
  cols[0] = 0; cols[1] = 1;
  CHKERRQ(MatSeqAIJSetPreallocation(Jpipe[1],0,nz));
  CHKERRQ(MatSetValues(Jpipe[1],2,rows,2,cols,aa,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(Jpipe[1],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Jpipe[1],MAT_FINAL_ASSEMBLY));

  /* Jacobian for downstream vertex */
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&Jpipe[2]));
  CHKERRQ(MatSetSizes(Jpipe[2],PETSC_DECIDE,PETSC_DECIDE,M,2));
  CHKERRQ(MatSetFromOptions(Jpipe[2]));
  CHKERRQ(MatSetOption(Jpipe[2],MAT_STRUCTURE_ONLY,PETSC_TRUE));
  nz[0] = 0; nz[1] = 0; nz[M-2] = 2; nz[M-1] = 2;
  rows[0] = M - 2; rows[1] = M - 1;
  CHKERRQ(MatSeqAIJSetPreallocation(Jpipe[2],0,nz));
  CHKERRQ(MatSetValues(Jpipe[2],2,rows,2,cols,aa,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(Jpipe[2],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Jpipe[2],MAT_FINAL_ASSEMBLY));

  CHKERRQ(PetscFree2(nz,aa));

  *J = Jpipe;
  pipe->jacobian = Jpipe;
  PetscFunctionReturn(0);
}

PetscErrorCode PipeDestroyJacobian(Pipe pipe)
{
  Mat            *Jpipe = pipe->jacobian;
  PetscInt       i;

  PetscFunctionBegin;
  if (Jpipe) {
    for (i=0; i<3; i++) {
      CHKERRQ(MatDestroy(&Jpipe[i]));
    }
  }
  CHKERRQ(PetscFree(Jpipe));
  PetscFunctionReturn(0);
}

/*
    JunctionCreateJacobian - Create Jacobian matrices for a vertex.

    Collective on Pipe

    Input Parameter:
+   dm - the DMNetwork object
.   v - vertex point
-   Jin - Jacobian patterns created by JunctionCreateJacobianSample() for reuse

    Output Parameter:
.   J  - array of Jacobian matrices (see dmnetworkimpl.h)

    Level: beginner
*/
PetscErrorCode JunctionCreateJacobian(DM dm,PetscInt v,Mat *Jin,Mat *J[])
{
  Mat            *Jv;
  PetscInt       nedges,e,i,M,N,*rows,*cols;
  PetscBool      isSelf;
  const PetscInt *edges,*cone;
  PetscScalar    *zeros;

  PetscFunctionBegin;
  /* Get array size of Jv */
  CHKERRQ(DMNetworkGetSupportingEdges(dm,v,&nedges,&edges));
  PetscCheckFalse(nedges <= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"%d vertex, nedges %d",v,nedges);

  /* two Jacobians for each connected edge: J(v,e) and J(v,vc); adding J(v,v), total 2*nedges+1 Jacobians */
  CHKERRQ(PetscCalloc1(2*nedges+1,&Jv));

  /* Create dense zero block for this vertex: J[0] = Jacobian(v,v) */
  CHKERRQ(DMNetworkGetComponent(dm,v,-1,NULL,NULL,&M));
  PetscCheckFalse(M !=2,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"M != 2",M);
  CHKERRQ(PetscMalloc3(M,&rows,M,&cols,M*M,&zeros));
  CHKERRQ(PetscArrayzero(zeros,M*M));
  for (i=0; i<M; i++) rows[i] = i;

  for (e=0; e<nedges; e++) {
    /* create Jv[2*e+1] = Jacobian(v,e), e: supporting edge */
    CHKERRQ(DMNetworkGetConnectedVertices(dm,edges[e],&cone));
    isSelf = (v == cone[0]) ? PETSC_TRUE:PETSC_FALSE;

    if (Jin) {
      if (isSelf) {
        Jv[2*e+1] = Jin[0];
      } else {
        Jv[2*e+1] = Jin[1];
      }
      Jv[2*e+2] = Jin[2];
      CHKERRQ(PetscObjectReference((PetscObject)(Jv[2*e+1])));
      CHKERRQ(PetscObjectReference((PetscObject)(Jv[2*e+2])));
    } else {
      /* create J(v,e) */
      CHKERRQ(MatCreate(PETSC_COMM_SELF,&Jv[2*e+1]));
      CHKERRQ(DMNetworkGetComponent(dm,edges[e],-1,NULL,NULL,&N));
      CHKERRQ(MatSetSizes(Jv[2*e+1],PETSC_DECIDE,PETSC_DECIDE,M,N));
      CHKERRQ(MatSetFromOptions(Jv[2*e+1]));
      CHKERRQ(MatSetOption(Jv[2*e+1],MAT_STRUCTURE_ONLY,PETSC_TRUE));
      CHKERRQ(MatSeqAIJSetPreallocation(Jv[2*e+1],2,NULL));
      if (N) {
        if (isSelf) { /* coupling at upstream */
          for (i=0; i<2; i++) cols[i] = i;
        } else { /* coupling at downstream */
          cols[0] = N-2; cols[1] = N-1;
        }
        CHKERRQ(MatSetValues(Jv[2*e+1],2,rows,2,cols,zeros,INSERT_VALUES));
      }
      CHKERRQ(MatAssemblyBegin(Jv[2*e+1],MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(Jv[2*e+1],MAT_FINAL_ASSEMBLY));

      /* create Jv[2*e+2] = Jacobian(v,vc), vc: connected vertex.
       In WashNetwork, v and vc are not connected, thus Jacobian(v,vc) is empty */
      CHKERRQ(MatCreate(PETSC_COMM_SELF,&Jv[2*e+2]));
      CHKERRQ(MatSetSizes(Jv[2*e+2],PETSC_DECIDE,PETSC_DECIDE,M,M)); /* empty matrix, sizes can be arbitrary */
      CHKERRQ(MatSetFromOptions(Jv[2*e+2]));
      CHKERRQ(MatSetOption(Jv[2*e+2],MAT_STRUCTURE_ONLY,PETSC_TRUE));
      CHKERRQ(MatSeqAIJSetPreallocation(Jv[2*e+2],1,NULL));
      CHKERRQ(MatAssemblyBegin(Jv[2*e+2],MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(Jv[2*e+2],MAT_FINAL_ASSEMBLY));
    }
  }
  CHKERRQ(PetscFree3(rows,cols,zeros));

  *J = Jv;
  PetscFunctionReturn(0);
}

PetscErrorCode JunctionDestroyJacobian(DM dm,PetscInt v,Junction junc)
{
  Mat            *Jv=junc->jacobian;
  const PetscInt *edges;
  PetscInt       nedges,e;

  PetscFunctionBegin;
  if (!Jv) PetscFunctionReturn(0);

  CHKERRQ(DMNetworkGetSupportingEdges(dm,v,&nedges,&edges));
  for (e=0; e<nedges; e++) {
    CHKERRQ(MatDestroy(&Jv[2*e+1]));
    CHKERRQ(MatDestroy(&Jv[2*e+2]));
  }
  CHKERRQ(PetscFree(Jv));
  PetscFunctionReturn(0);
}
