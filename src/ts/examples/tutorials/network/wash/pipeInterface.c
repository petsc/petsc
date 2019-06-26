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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(pipe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   PipeDestroy - Destroy Pipe object.

   Input Parameters:
   pipe - Reference to pipe intended to be destroyed.
*/
PetscErrorCode PipeDestroy(Pipe *pipe)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*pipe) PetscFunctionReturn(0);

  ierr = PipeDestroyJacobian(*pipe);CHKERRQ(ierr);
  ierr = VecDestroy(&(*pipe)->x);CHKERRQ(ierr);
  ierr = DMDestroy(&(*pipe)->da);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDACreate1d(PETSC_COMM_SELF, DM_BOUNDARY_GHOSTED, pipe->nnodes, 2, 1, NULL, &pipe->da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(pipe->da);CHKERRQ(ierr);
  ierr = DMSetUp(pipe->da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(pipe->da, 0, "Q");CHKERRQ(ierr);
  ierr = DMDASetFieldName(pipe->da, 1, "H");CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(pipe->da, 0, pipe->length, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(pipe->da, &(pipe->x));CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(pipe->da, &info);CHKERRQ(ierr);

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
  PetscErrorCode ierr;
  Mat            *Jpipe;
  PetscInt       M,rows[2],cols[2],*nz;
  PetscScalar    *aa;

  PetscFunctionBegin;
  if (Jin) {
    *J = Jin;
    pipe->jacobian = Jin;
    ierr = PetscObjectReference((PetscObject)(Jin[0]));CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscMalloc1(3,&Jpipe);CHKERRQ(ierr);

  /* Jacobian for this pipe */
  ierr = DMSetMatrixStructureOnly(pipe->da,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMCreateMatrix(pipe->da,&Jpipe[0]);CHKERRQ(ierr);
  ierr = DMSetMatrixStructureOnly(pipe->da,PETSC_FALSE);CHKERRQ(ierr);

  /* Jacobian for upstream vertex */
  ierr = MatGetSize(Jpipe[0],&M,NULL);CHKERRQ(ierr);
  ierr = PetscCalloc2(M,&nz,4,&aa);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF,&Jpipe[1]);CHKERRQ(ierr);
  ierr = MatSetSizes(Jpipe[1],PETSC_DECIDE,PETSC_DECIDE,M,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Jpipe[1]);CHKERRQ(ierr);
  ierr = MatSetOption(Jpipe[1],MAT_STRUCTURE_ONLY,PETSC_TRUE);CHKERRQ(ierr);
  nz[0] = 2; nz[1] = 2;
  rows[0] = 0; rows[1] = 1;
  cols[0] = 0; cols[1] = 1;
  ierr = MatSeqAIJSetPreallocation(Jpipe[1],0,nz);CHKERRQ(ierr);
  ierr = MatSetValues(Jpipe[1],2,rows,2,cols,aa,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Jpipe[1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jpipe[1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Jacobian for downstream vertex */
  ierr = MatCreate(PETSC_COMM_SELF,&Jpipe[2]);CHKERRQ(ierr);
  ierr = MatSetSizes(Jpipe[2],PETSC_DECIDE,PETSC_DECIDE,M,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Jpipe[2]);CHKERRQ(ierr);
  ierr = MatSetOption(Jpipe[2],MAT_STRUCTURE_ONLY,PETSC_TRUE);CHKERRQ(ierr);
  nz[0] = 0; nz[1] = 0; nz[M-2] = 2; nz[M-1] = 2;
  rows[0] = M - 2; rows[1] = M - 1;
  ierr = MatSeqAIJSetPreallocation(Jpipe[2],0,nz);CHKERRQ(ierr);
  ierr = MatSetValues(Jpipe[2],2,rows,2,cols,aa,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Jpipe[2],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jpipe[2],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree2(nz,aa);CHKERRQ(ierr);

  *J = Jpipe;
  pipe->jacobian = Jpipe;
  PetscFunctionReturn(0);
}

PetscErrorCode PipeDestroyJacobian(Pipe pipe)
{
  PetscErrorCode ierr;
  Mat            *Jpipe = pipe->jacobian;
  PetscInt       i;

  PetscFunctionBegin;
  if (Jpipe) {
    for (i=0; i<3; i++) {
      ierr = MatDestroy(&Jpipe[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(Jpipe);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  Mat            *Jv;
  PetscInt       nedges,e,i,M,N,*rows,*cols;
  PetscBool      isSelf;
  const PetscInt *edges,*cone;
  PetscScalar    *zeros;

  PetscFunctionBegin;
  /* Get arrary size of Jv */
  ierr = DMNetworkGetSupportingEdges(dm,v,&nedges,&edges);CHKERRQ(ierr);
  if (nedges <= 0) SETERRQ2(PETSC_COMM_SELF,1,"%d vertex, nedges %d\n",v,nedges);

  /* two Jacobians for each connected edge: J(v,e) and J(v,vc); adding J(v,v), total 2*nedges+1 Jacobians */
  ierr = PetscCalloc1(2*nedges+1,&Jv);CHKERRQ(ierr);

  /* Create dense zero block for this vertex: J[0] = Jacobian(v,v) */
  ierr = DMNetworkGetNumVariables(dm,v,&M);CHKERRQ(ierr);
  if (M !=2) SETERRQ1(PETSC_COMM_SELF,1,"M != 2",M);
  ierr = PetscMalloc3(M,&rows,M,&cols,M*M,&zeros);CHKERRQ(ierr);
  ierr = PetscArrayzero(zeros,M*M);CHKERRQ(ierr);
  for (i=0; i<M; i++) rows[i] = i;

  for (e=0; e<nedges; e++) {
    /* create Jv[2*e+1] = Jacobian(v,e), e: supporting edge */
    ierr = DMNetworkGetConnectedVertices(dm,edges[e],&cone);CHKERRQ(ierr);
    isSelf = (v == cone[0]) ? PETSC_TRUE:PETSC_FALSE;

    if (Jin) {
      if (isSelf) {
        Jv[2*e+1] = Jin[0];
      } else {
        Jv[2*e+1] = Jin[1];
      }
      Jv[2*e+2] = Jin[2];
      ierr = PetscObjectReference((PetscObject)(Jv[2*e+1]));CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)(Jv[2*e+2]));CHKERRQ(ierr);
    } else {
      /* create J(v,e) */
      ierr = MatCreate(PETSC_COMM_SELF,&Jv[2*e+1]);CHKERRQ(ierr);
      ierr = DMNetworkGetNumVariables(dm,edges[e],&N);CHKERRQ(ierr);
      ierr = MatSetSizes(Jv[2*e+1],PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
      ierr = MatSetFromOptions(Jv[2*e+1]);CHKERRQ(ierr);
      ierr = MatSetOption(Jv[2*e+1],MAT_STRUCTURE_ONLY,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatSeqAIJSetPreallocation(Jv[2*e+1],2,NULL);CHKERRQ(ierr);
      if (N) {
        if (isSelf) { /* coupling at upstream */
          for (i=0; i<2; i++) cols[i] = i;
        } else { /* coupling at downstream */
          cols[0] = N-2; cols[1] = N-1;
        }
        ierr = MatSetValues(Jv[2*e+1],2,rows,2,cols,zeros,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(Jv[2*e+1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(Jv[2*e+1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

      /* create Jv[2*e+2] = Jacobian(v,vc), vc: connected vertex.
       In WashNetwork, v and vc are not connected, thus Jacobian(v,vc) is empty */
      ierr = MatCreate(PETSC_COMM_SELF,&Jv[2*e+2]);CHKERRQ(ierr);
      ierr = MatSetSizes(Jv[2*e+2],PETSC_DECIDE,PETSC_DECIDE,M,M);CHKERRQ(ierr); /* empty matrix, sizes can be arbitrary */
      ierr = MatSetFromOptions(Jv[2*e+2]);CHKERRQ(ierr);
      ierr = MatSetOption(Jv[2*e+2],MAT_STRUCTURE_ONLY,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatSeqAIJSetPreallocation(Jv[2*e+2],1,NULL);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(Jv[2*e+2],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(Jv[2*e+2],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree3(rows,cols,zeros);CHKERRQ(ierr);

  *J = Jv;
  PetscFunctionReturn(0);
}

PetscErrorCode JunctionDestroyJacobian(DM dm,PetscInt v,Junction junc)
{
  PetscErrorCode ierr;
  Mat            *Jv=junc->jacobian;
  const PetscInt *edges;
  PetscInt       nedges,e;

  PetscFunctionBegin;
  if (!Jv) PetscFunctionReturn(0);

  ierr = DMNetworkGetSupportingEdges(dm,v,&nedges,&edges);CHKERRQ(ierr);
  for (e=0; e<nedges; e++) {
    ierr = MatDestroy(&Jv[2*e+1]);CHKERRQ(ierr);
    ierr = MatDestroy(&Jv[2*e+2]);CHKERRQ(ierr);
  }
  ierr = PetscFree(Jv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
