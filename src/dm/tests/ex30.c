static char help[] = "Tests DMSLICED operations\n\n";

#include <petscdmsliced.h>

int main(int argc,char *argv[])
{
  char           mat_type[256] = MATAIJ; /* default matrix type */
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscMPIInt    rank,size;
  DM             slice;
  PetscInt       i,bs=1,N=5,n,m,rstart,ghosts[2],*d_nnz,*o_nnz,dfill[4]={1,0,0,1},ofill[4]={1,1,1,1};
  PetscReal      alpha   =1,K=1,rho0=1,u0=0,sigma=0.2;
  PetscBool      useblock=PETSC_TRUE;
  PetscScalar    *xx;
  Mat            A;
  Vec            x,b,lf;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  ierr = PetscOptionsBegin(comm,0,"Options for DMSliced test",0);CHKERRQ(ierr);
  {
    ierr = PetscOptionsInt("-n","Global number of nodes","",N,&N,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-bs","Block size (1 or 2)","",bs,&bs,NULL);CHKERRQ(ierr);
    if (bs != 1) {
      if (bs != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Block size must be 1 or 2");
      ierr = PetscOptionsReal("-alpha","Inverse time step for wave operator","",alpha,&alpha,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-K","Bulk modulus of compressibility","",K,&K,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-rho0","Reference density","",rho0,&rho0,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-u0","Reference velocity","",u0,&u0,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-sigma","Width of Gaussian density perturbation","",sigma,&sigma,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-block","Use block matrix assembly","",useblock,&useblock,NULL);CHKERRQ(ierr);
    }
    ierr = PetscOptionsString("-sliced_mat_type","Matrix type to use (aij or baij)","",mat_type,mat_type,sizeof(mat_type),NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Split ownership, set up periodic grid in 1D */
  n         = PETSC_DECIDE;
  ierr      = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  rstart    = 0;
  ierr      = MPI_Scan(&n,&rstart,1,MPIU_INT,MPI_SUM,comm);CHKERRMPI(ierr);
  rstart   -= n;
  ghosts[0] = (N+rstart-1)%N;
  ghosts[1] = (rstart+n)%N;

  ierr = PetscMalloc2(n,&d_nnz,n,&o_nnz);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (size > 1 && (i==0 || i==n-1)) {
      d_nnz[i] = 2;
      o_nnz[i] = 1;
    } else {
      d_nnz[i] = 3;
      o_nnz[i] = 0;
    }
  }
  ierr = DMSlicedCreate(comm,bs,n,2,ghosts,d_nnz,o_nnz,&slice);CHKERRQ(ierr); /* Currently does not copy X_nnz so we can't free them until after DMSlicedGetMatrix */

  if (!useblock) {ierr = DMSlicedSetBlockFills(slice,dfill,ofill);CHKERRQ(ierr);} /* Irrelevant for baij formats */
  ierr = DMSetMatType(slice,mat_type);CHKERRQ(ierr);
  ierr = DMCreateMatrix(slice,&A);CHKERRQ(ierr);
  ierr = PetscFree2(d_nnz,o_nnz);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(slice,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);

  ierr = VecGhostGetLocalForm(x,&lf);CHKERRQ(ierr);
  ierr = VecGetSize(lf,&m);CHKERRQ(ierr);
  if (m != (n+2)*bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"size of local form %D, expected %D",m,(n+2)*bs);
  ierr = VecGetArray(lf,&xx);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    PetscInt        row[2],col[9],im,ip;
    PetscScalar     v[12];
    const PetscReal xref = 2.0*(rstart+i)/N - 1; /* [-1,1] */
    const PetscReal h    = 1.0/N;                /* grid spacing */
    im = (i==0) ? n : i-1;
    ip = (i==n-1) ? n+1 : i+1;
    switch (bs) {
    case 1:                     /* Laplacian with periodic boundaries */
      col[0] = im;         col[1] = i;        col[2] = ip;
      v[0]   = -h;           v[1] = 2*h;        v[2] = -h;
      ierr   = MatSetValuesLocal(A,1,&i,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
      xx[i]  = PetscSinReal(xref*PETSC_PI);
      break;
    case 2:                     /* Linear acoustic wave operator in variables [rho, u], central differences, periodic, timestep 1/alpha */
      v[0] = -0.5*u0;   v[1] = -0.5*K;      v[2] = alpha; v[3] = 0;       v[4] = 0.5*u0;    v[5] = 0.5*K;
      v[6] = -0.5/rho0; v[7] = -0.5*u0;     v[8] = 0;     v[9] = alpha;   v[10] = 0.5/rho0; v[11] = 0.5*u0;
      if (useblock) {
        row[0] = i; col[0] = im; col[1] = i; col[2] = ip;
        ierr   = MatSetValuesBlockedLocal(A,1,row,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        row[0] = 2*i; row[1] = 2*i+1;
        col[0] = 2*im; col[1] = 2*im+1; col[2] = 2*i; col[3] = 2*ip; col[4] = 2*ip+1;
        v[3]   = v[4]; v[4] = v[5];                                                     /* pack values in first row */
        ierr   = MatSetValuesLocal(A,1,row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
        col[2] = 2*i+1;
        v[8]   = v[9]; v[9] = v[10]; v[10] = v[11];                                     /* pack values in second row */
        ierr   = MatSetValuesLocal(A,1,row+1,5,col,v+6,INSERT_VALUES);CHKERRQ(ierr);
      }
      /* Set current state (gaussian density perturbation) */
      xx[2*i]   = 0.2*PetscExpReal(-PetscSqr(xref)/(2*PetscSqr(sigma)));
      xx[2*i+1] = 0;
      break;
    default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"not implemented for block size %D",bs);
    }
  }
  ierr = VecRestoreArray(lf,&xx);CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(x,&lf);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatMult(A,x,b);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Update the ghosted values, view the result on rank 0. */
  ierr = VecGhostUpdateBegin(b,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(b,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (!rank) {
    ierr = VecGhostGetLocalForm(b,&lf);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"Local form of b on rank 0, last two nodes are ghost nodes\n");CHKERRQ(ierr);
    ierr = VecView(lf,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = VecGhostRestoreLocalForm(b,&lf);CHKERRQ(ierr);
  }

  ierr = DMDestroy(&slice);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      nsize: 2
      args: -bs 2 -block 0 -sliced_mat_type baij -alpha 10 -u0 0.1

   test:
      suffix: 2
      nsize: 2
      args: -bs 2 -block 1 -sliced_mat_type aij -alpha 10 -u0 0.1

   test:
      suffix: 3
      nsize: 2
      args: -bs 2 -block 0 -sliced_mat_type aij -alpha 10 -u0 0.1

TEST*/
