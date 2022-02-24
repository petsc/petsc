
static char help[] = "Scatters from a parallel vector to a sequential vector.\n\
Tests where the local part of the scatter is a copy.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscInt       n = 5,i,*blks,bs = 1,m = 2;
  PetscScalar    value;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;
  PetscViewer    sviewer;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL));

  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* create two vectors */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,size*bs*n));
  CHKERRQ(VecSetFromOptions(x));

  /* create two index sets */
  if (rank < size-1) m = n + 2;
  else m = n;

  CHKERRQ(PetscMalloc1(m,&blks));
  blks[0] = n*rank;
  for (i=1; i<m; i++) blks[i] = blks[i-1] + 1;
  CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,m,blks,PETSC_COPY_VALUES,&is1));
  CHKERRQ(PetscFree(blks));

  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,bs*m,&y));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,bs*m,0,1,&is2));

  /* each processor inserts the entire vector */
  /* this is redundant but tests assembly */
  for (i=0; i<bs*n*size; i++) {
    value = (PetscScalar) i;
    CHKERRQ(VecSetValues(x,1,&i,&value,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecScatterCreate(x,is1,y,is2,&ctx));
  CHKERRQ(VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));

  CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"----\n"));
  CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(VecView(y,sviewer)); fflush(stdout);
  CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecScatterDestroy(&ctx));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));

  ierr = PetscFinalize();
  return ierr;
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
