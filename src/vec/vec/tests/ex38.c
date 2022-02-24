static const char help[] = "Test VecGetSubVector()\n\n";

#include <petscvec.h>

int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Vec            X,Y,Z,W;
  PetscMPIInt    rank,size;
  PetscInt       i,rstart,rend,idxs[3];
  PetscScalar    *x,*y,*w,*z;
  PetscViewer    viewer;
  IS             is0,is1,is2;
  PetscBool      iscuda;
  PetscErrorCode ierr;

  ierr   = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  comm   = PETSC_COMM_WORLD;
  viewer = PETSC_VIEWER_STDOUT_WORLD;
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));

  CHKERRQ(VecCreate(comm,&X));
  CHKERRQ(VecSetSizes(X,10,PETSC_DETERMINE));
  CHKERRQ(VecSetFromOptions(X));
  CHKERRQ(VecGetOwnershipRange(X,&rstart,&rend));

  CHKERRQ(VecGetArray(X,&x));
  for (i=0; i<rend-rstart; i++) x[i] = rstart+i;
  CHKERRQ(VecRestoreArray(X,&x));
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)X,&iscuda,VECSEQCUDA,VECMPICUDA,""));
  if (iscuda) { /* trigger a copy of the data on the GPU */
    const PetscScalar *xx;

    CHKERRQ(VecCUDAGetArrayRead(X,&xx));
    CHKERRQ(VecCUDARestoreArrayRead(X,&xx));
  }

  CHKERRQ(VecView(X,viewer));

  idxs[0] = (size - rank - 1)*10 + 5;
  idxs[1] = (size - rank - 1)*10 + 2;
  idxs[2] = (size - rank - 1)*10 + 3;

  CHKERRQ(ISCreateStride(comm,(rend-rstart)/3+3*(rank>size/2),rstart,1,&is0));
  CHKERRQ(ISComplement(is0,rstart,rend,&is1));
  CHKERRQ(ISCreateGeneral(comm,3,idxs,PETSC_USE_POINTER,&is2));

  CHKERRQ(ISView(is0,viewer));
  CHKERRQ(ISView(is1,viewer));
  CHKERRQ(ISView(is2,viewer));

  CHKERRQ(VecGetSubVector(X,is0,&Y));
  CHKERRQ(VecGetSubVector(X,is1,&Z));
  CHKERRQ(VecGetSubVector(X,is2,&W));
  CHKERRQ(VecView(Y,viewer));
  CHKERRQ(VecView(Z,viewer));
  CHKERRQ(VecView(W,viewer));
  CHKERRQ(VecGetArray(Y,&y));
  y[0] = 1000*(rank+1);
  CHKERRQ(VecRestoreArray(Y,&y));
  CHKERRQ(VecGetArray(Z,&z));
  z[0] = -1000*(rank+1);
  CHKERRQ(VecRestoreArray(Z,&z));
  CHKERRQ(VecGetArray(W,&w));
  w[0] = -10*(rank+1);
  CHKERRQ(VecRestoreArray(W,&w));
  CHKERRQ(VecRestoreSubVector(X,is0,&Y));
  CHKERRQ(VecRestoreSubVector(X,is1,&Z));
  CHKERRQ(VecRestoreSubVector(X,is2,&W));
  CHKERRQ(VecView(X,viewer));

  CHKERRQ(ISDestroy(&is0));
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));
  CHKERRQ(VecDestroy(&X));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
      nsize: 3
      output_file: output/ex38_1.out
      filter: grep -v "  type:"
      diff_args: -j
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
      test:
        requires: kokkos_kernels
        suffix: kokkos
        args: -vec_type kokkos
      test:
        requires: hip
        suffix: hip
        args: -vec_type hip

TEST*/
