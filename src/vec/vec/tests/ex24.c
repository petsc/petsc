
static char help[] = "Scatters from a parallel vector to a sequential vector.\n\
Tests where the local part of the scatter is a copy.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscMPIInt    size,rank;
  PetscInt       n = 5,i,*blks,bs = 1,m = 2;
  PetscScalar    value;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;
  PetscViewer    sviewer;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL));

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* create two vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,size*bs*n));
  PetscCall(VecSetFromOptions(x));

  /* create two index sets */
  if (rank < size-1) m = n + 2;
  else m = n;

  PetscCall(PetscMalloc1(m,&blks));
  blks[0] = n*rank;
  for (i=1; i<m; i++) blks[i] = blks[i-1] + 1;
  PetscCall(ISCreateBlock(PETSC_COMM_SELF,bs,m,blks,PETSC_COPY_VALUES,&is1));
  PetscCall(PetscFree(blks));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF,bs*m,&y));
  PetscCall(ISCreateStride(PETSC_COMM_SELF,bs*m,0,1,&is2));

  /* each processor inserts the entire vector */
  /* this is redundant but tests assembly */
  for (i=0; i<bs*n*size; i++) {
    value = (PetscScalar) i;
    PetscCall(VecSetValues(x,1,&i,&value,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecScatterCreate(x,is1,y,is2,&ctx));
  PetscCall(VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));

  PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"----\n"));
  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  PetscCall(VecView(y,sviewer)); fflush(stdout);
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecScatterDestroy(&ctx));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      nsize: 3
      output_file: output/ex24_1.out
      filter: grep -v "  type:"
      test:
        suffix: standard
        args: -vec_type standard
      test:
        requires: cuda
        suffix: cuda
        args: -vec_type cuda
      test:
        requires: viennacl
        suffix:  viennacl
        args: -vec_type viennacl

TEST*/
