static char help[] = "Tests DMSLICED operations\n\n";

#include <petscdmsliced.h>

int main(int argc,char *argv[])
{
  char           mat_type[256] = MATAIJ; /* default matrix type */
  MPI_Comm       comm;
  PetscMPIInt    rank,size;
  DM             slice;
  PetscInt       i,bs=1,N=5,n,m,rstart,ghosts[2],*d_nnz,*o_nnz,dfill[4]={1,0,0,1},ofill[4]={1,1,1,1};
  PetscReal      alpha   =1,K=1,rho0=1,u0=0,sigma=0.2;
  PetscBool      useblock=PETSC_TRUE;
  PetscScalar    *xx;
  Mat            A;
  Vec            x,b,lf;

  PetscCall(PetscInitialize(&argc,&argv,0,help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  PetscOptionsBegin(comm,0,"Options for DMSliced test",0);
  {
    PetscCall(PetscOptionsInt("-n","Global number of nodes","",N,&N,NULL));
    PetscCall(PetscOptionsInt("-bs","Block size (1 or 2)","",bs,&bs,NULL));
    if (bs != 1) {
      PetscCheck(bs == 2,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Block size must be 1 or 2");
      PetscCall(PetscOptionsReal("-alpha","Inverse time step for wave operator","",alpha,&alpha,NULL));
      PetscCall(PetscOptionsReal("-K","Bulk modulus of compressibility","",K,&K,NULL));
      PetscCall(PetscOptionsReal("-rho0","Reference density","",rho0,&rho0,NULL));
      PetscCall(PetscOptionsReal("-u0","Reference velocity","",u0,&u0,NULL));
      PetscCall(PetscOptionsReal("-sigma","Width of Gaussian density perturbation","",sigma,&sigma,NULL));
      PetscCall(PetscOptionsBool("-block","Use block matrix assembly","",useblock,&useblock,NULL));
    }
    PetscCall(PetscOptionsString("-sliced_mat_type","Matrix type to use (aij or baij)","",mat_type,mat_type,sizeof(mat_type),NULL));
  }
  PetscOptionsEnd();

  /* Split ownership, set up periodic grid in 1D */
  n         = PETSC_DECIDE;
  PetscCall(PetscSplitOwnership(comm,&n,&N));
  rstart    = 0;
  PetscCallMPI(MPI_Scan(&n,&rstart,1,MPIU_INT,MPI_SUM,comm));
  rstart   -= n;
  ghosts[0] = (N+rstart-1)%N;
  ghosts[1] = (rstart+n)%N;

  PetscCall(PetscMalloc2(n,&d_nnz,n,&o_nnz));
  for (i=0; i<n; i++) {
    if (size > 1 && (i==0 || i==n-1)) {
      d_nnz[i] = 2;
      o_nnz[i] = 1;
    } else {
      d_nnz[i] = 3;
      o_nnz[i] = 0;
    }
  }
  PetscCall(DMSlicedCreate(comm,bs,n,2,ghosts,d_nnz,o_nnz,&slice)); /* Currently does not copy X_nnz so we can't free them until after DMSlicedGetMatrix */

  if (!useblock) PetscCall(DMSlicedSetBlockFills(slice,dfill,ofill)); /* Irrelevant for baij formats */
  PetscCall(DMSetMatType(slice,mat_type));
  PetscCall(DMCreateMatrix(slice,&A));
  PetscCall(PetscFree2(d_nnz,o_nnz));
  PetscCall(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));

  PetscCall(DMCreateGlobalVector(slice,&x));
  PetscCall(VecDuplicate(x,&b));

  PetscCall(VecGhostGetLocalForm(x,&lf));
  PetscCall(VecGetSize(lf,&m));
  PetscCheck(m == (n+2)*bs,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"size of local form %" PetscInt_FMT ", expected %" PetscInt_FMT,m,(n+2)*bs);
  PetscCall(VecGetArray(lf,&xx));
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
      PetscCall(MatSetValuesLocal(A,1,&i,3,col,v,INSERT_VALUES));
      xx[i]  = PetscSinReal(xref*PETSC_PI);
      break;
    case 2:                     /* Linear acoustic wave operator in variables [rho, u], central differences, periodic, timestep 1/alpha */
      v[0] = -0.5*u0;   v[1] = -0.5*K;      v[2] = alpha; v[3] = 0;       v[4] = 0.5*u0;    v[5] = 0.5*K;
      v[6] = -0.5/rho0; v[7] = -0.5*u0;     v[8] = 0;     v[9] = alpha;   v[10] = 0.5/rho0; v[11] = 0.5*u0;
      if (useblock) {
        row[0] = i; col[0] = im; col[1] = i; col[2] = ip;
        PetscCall(MatSetValuesBlockedLocal(A,1,row,3,col,v,INSERT_VALUES));
      } else {
        row[0] = 2*i; row[1] = 2*i+1;
        col[0] = 2*im; col[1] = 2*im+1; col[2] = 2*i; col[3] = 2*ip; col[4] = 2*ip+1;
        v[3]   = v[4]; v[4] = v[5];                                                     /* pack values in first row */
        PetscCall(MatSetValuesLocal(A,1,row,5,col,v,INSERT_VALUES));
        col[2] = 2*i+1;
        v[8]   = v[9]; v[9] = v[10]; v[10] = v[11];                                     /* pack values in second row */
        PetscCall(MatSetValuesLocal(A,1,row+1,5,col,v+6,INSERT_VALUES));
      }
      /* Set current state (gaussian density perturbation) */
      xx[2*i]   = 0.2*PetscExpReal(-PetscSqr(xref)/(2*PetscSqr(sigma)));
      xx[2*i+1] = 0;
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"not implemented for block size %" PetscInt_FMT,bs);
    }
  }
  PetscCall(VecRestoreArray(lf,&xx));
  PetscCall(VecGhostRestoreLocalForm(x,&lf));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(MatMult(A,x,b));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(b,PETSC_VIEWER_STDOUT_WORLD));

  /* Update the ghosted values, view the result on rank 0. */
  PetscCall(VecGhostUpdateBegin(b,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecGhostUpdateEnd(b,INSERT_VALUES,SCATTER_FORWARD));
  if (rank == 0) {
    PetscCall(VecGhostGetLocalForm(b,&lf));
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"Local form of b on rank 0, last two nodes are ghost nodes\n"));
    PetscCall(VecView(lf,PETSC_VIEWER_STDOUT_SELF));
    PetscCall(VecGhostRestoreLocalForm(b,&lf));
  }

  PetscCall(DMDestroy(&slice));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
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
