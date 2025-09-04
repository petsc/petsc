static char help[] = "Tests ISLocalToGlobalMappingView() and ISLocalToGlobalMappingLoad()\n\n";

#include <petscis.h>
#include <petscviewer.h>

static PetscErrorCode TestEqual(MPI_Comm comm, ISLocalToGlobalMapping m1, ISLocalToGlobalMapping m2, const char *tname)
{
  PetscInt        n1, n2, b1, b2;
  const PetscInt *idx1, *idx2;
  PetscBool       flg;

  PetscFunctionBegin;
  PetscCall(ISLocalToGlobalMappingGetSize(m1, &n1));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(m1, &b1));
  PetscCall(ISLocalToGlobalMappingGetIndices(m1, &idx1));
  PetscCall(ISLocalToGlobalMappingGetSize(m2, &n2));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(m2, &b2));
  PetscCall(ISLocalToGlobalMappingGetIndices(m2, &idx2));
  flg = (PetscBool)(b1 == b2);
  if (!flg) PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s: different block sizes %" PetscInt_FMT " %" PetscInt_FMT "\n", tname, b1, b2));
  flg = (PetscBool)(n1 == n2);
  if (!flg) PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s: different sizes %" PetscInt_FMT " %" PetscInt_FMT "\n", tname, n1, n2));
  if (flg) {
    PetscCall(PetscArraycmp(idx1, idx2, n1, &flg));
    if (!flg) PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s: different indices\n", tname));
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &flg, 1, MPI_C_BOOL, MPI_LAND, comm));
  if (!flg) {
    PetscCall(ISLocalToGlobalMappingView(m1, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(ISLocalToGlobalMappingView(m2, PETSC_VIEWER_STDOUT_(comm)));
  }
  PetscCall(ISLocalToGlobalMappingRestoreIndices(m1, &idx1));
  PetscCall(ISLocalToGlobalMappingRestoreIndices(m2, &idx2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  ISLocalToGlobalMapping lg1l, lg1v, lg1lh, lg2l, lg2v, lg2lh;
  IS                     is1, is2;
  PetscInt               n, n1, n2, b1, b2;
  PetscInt              *idx;
  PetscMPIInt            size, rank;
  PetscViewer            vx;
  MPI_Comm               comm;
  char                   fname[PETSC_MAX_PATH_LEN], fnameh[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD, size - rank, -size - 1, rank + 1, &is1));
  PetscCall(ISGetLocalSize(is1, &n));
  PetscCall(ISGetIndices(is1, (const PetscInt **)&idx));
  PetscCall(ISCreateBlock(PETSC_COMM_WORLD, 3, n, idx, PETSC_COPY_VALUES, &is2));
  PetscCall(ISRestoreIndices(is1, (const PetscInt **)&idx));
  PetscCall(ISLocalToGlobalMappingCreateIS(is1, &lg1v));
  PetscCall(ISLocalToGlobalMappingCreateIS(is2, &lg2v));
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));

  /* Test MATLAB ASCII viewer */
  PetscCall(PetscObjectSetName((PetscObject)lg1v, "map1"));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_MATLAB));
  PetscCall(ISLocalToGlobalMappingView(lg1v, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISLocalToGlobalMappingView(lg2v, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscObjectSetName((PetscObject)lg2v, "map2"));

  /* Now test view/load of type binary */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "testfile", FILE_MODE_WRITE, &vx));
  PetscCall(ISLocalToGlobalMappingView(lg1v, vx));
  PetscCall(ISLocalToGlobalMappingView(lg2v, vx));
  PetscCall(PetscViewerDestroy(&vx));

  PetscCall(PetscSNPrintf(fname, PETSC_STATIC_ARRAY_LENGTH(fname), "testfile_seq_%d", rank));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, fname, FILE_MODE_WRITE, &vx));
  PetscCall(ISLocalToGlobalMappingView(lg1v, vx));
  PetscCall(ISLocalToGlobalMappingView(lg2v, vx));
  PetscCall(PetscViewerDestroy(&vx));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "testfile_noheader", FILE_MODE_WRITE, &vx));
  PetscCall(PetscViewerBinarySetSkipHeader(vx, PETSC_TRUE));
  PetscCall(ISLocalToGlobalMappingView(lg1v, vx));
  PetscCall(ISLocalToGlobalMappingView(lg2v, vx));
  PetscCall(PetscViewerDestroy(&vx));

  PetscCall(PetscSNPrintf(fnameh, PETSC_STATIC_ARRAY_LENGTH(fname), "testfile_noheader_seq_%d", rank));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, fnameh, FILE_MODE_WRITE, &vx));
  PetscCall(PetscViewerBinarySetSkipHeader(vx, PETSC_TRUE));
  PetscCall(ISLocalToGlobalMappingView(lg1v, vx));
  PetscCall(ISLocalToGlobalMappingView(lg2v, vx));
  PetscCall(PetscViewerDestroy(&vx));

  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 77, 0, NULL, PETSC_USE_POINTER, &lg1l));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 99, 0, NULL, PETSC_OWN_POINTER, &lg2l));
  PetscCall(ISLocalToGlobalMappingGetSize(lg1v, &n1));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(lg1v, &b1));
  n1 /= b1;
  PetscCall(ISLocalToGlobalMappingGetSize(lg2v, &n2));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(lg2v, &b2));
  n2 /= b2;
  PetscCall(PetscMalloc1(n1, &idx));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, b1, n1, idx, PETSC_OWN_POINTER, &lg1lh));
  PetscCall(PetscMalloc1(n2, &idx));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, b2, n2, idx, PETSC_OWN_POINTER, &lg2lh));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "testfile", FILE_MODE_READ, &vx));
  PetscCall(ISLocalToGlobalMappingLoad(lg1l, vx));
  PetscCall(ISLocalToGlobalMappingLoad(lg2l, vx));
  PetscCall(PetscViewerDestroy(&vx));
  PetscCall(TestEqual(PETSC_COMM_WORLD, lg1v, lg1l, "load_world_map_world 1"));
  PetscCall(TestEqual(PETSC_COMM_WORLD, lg2v, lg2l, "load_world_map_world 2"));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "testfile_noheader", FILE_MODE_READ, &vx));
  PetscCall(PetscViewerBinarySetSkipHeader(vx, PETSC_TRUE));
  PetscCall(ISLocalToGlobalMappingLoad(lg1l, vx));
  PetscCall(ISLocalToGlobalMappingLoad(lg2l, vx));
  PetscCall(PetscViewerDestroy(&vx));
  PetscCall(TestEqual(PETSC_COMM_WORLD, lg1v, lg1l, "load_world_map_world_noheader 1"));
  PetscCall(TestEqual(PETSC_COMM_WORLD, lg2v, lg2l, "load_world_map_world_noheader 2"));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "testfile_noheader", FILE_MODE_READ, &vx));
  PetscCall(PetscViewerBinarySetSkipHeader(vx, PETSC_TRUE));
  PetscCall(ISLocalToGlobalMappingLoad(lg1lh, vx));
  PetscCall(ISLocalToGlobalMappingLoad(lg2lh, vx));
  PetscCall(PetscViewerDestroy(&vx));
  PetscCall(TestEqual(PETSC_COMM_WORLD, lg1v, lg1lh, "load_world_map_self_noheader 1"));
  PetscCall(TestEqual(PETSC_COMM_WORLD, lg2v, lg2lh, "load_world_map_self_noheader 2"));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, fname, FILE_MODE_READ, &vx));
  PetscCall(ISLocalToGlobalMappingLoad(lg1l, vx));
  PetscCall(ISLocalToGlobalMappingLoad(lg2l, vx));
  PetscCall(PetscViewerDestroy(&vx));
  PetscCall(TestEqual(PETSC_COMM_WORLD, lg1v, lg1l, "load_self_map_world 1"));
  PetscCall(TestEqual(PETSC_COMM_WORLD, lg2v, lg2l, "load_self_map_world 2"));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, fnameh, FILE_MODE_READ, &vx));
  PetscCall(PetscViewerBinarySetSkipHeader(vx, PETSC_TRUE));
  PetscCall(ISLocalToGlobalMappingLoad(lg1l, vx));
  PetscCall(ISLocalToGlobalMappingLoad(lg2l, vx));
  PetscCall(PetscViewerDestroy(&vx));
  PetscCall(TestEqual(PETSC_COMM_WORLD, lg1v, lg1l, "load_self_map_world_noheader 1"));
  PetscCall(TestEqual(PETSC_COMM_WORLD, lg2v, lg2l, "load_self_map_world_noheader 2"));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, fnameh, FILE_MODE_READ, &vx));
  PetscCall(PetscViewerBinarySetSkipHeader(vx, PETSC_TRUE));
  PetscCall(ISLocalToGlobalMappingLoad(lg1lh, vx));
  PetscCall(ISLocalToGlobalMappingLoad(lg2lh, vx));
  PetscCall(PetscViewerDestroy(&vx));
  PetscCall(TestEqual(PETSC_COMM_WORLD, lg1v, lg1lh, "load_self_map_self_noheader 1"));
  PetscCall(TestEqual(PETSC_COMM_WORLD, lg2v, lg2lh, "load_self_map_self_noheader 2"));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "View world maps\n"));
  PetscCall(ISLocalToGlobalMappingView(lg1v, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISLocalToGlobalMappingView(lg2v, PETSC_VIEWER_STDOUT_WORLD));

  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, rank < 2, rank, &comm));
  if (rank < 2) {
    PetscCall(ISLocalToGlobalMappingDestroy(&lg1l));
    PetscCall(ISLocalToGlobalMappingDestroy(&lg2l));
    PetscCall(ISLocalToGlobalMappingCreate(comm, 77, 1, &n, PETSC_USE_POINTER, &lg1l));
    PetscCall(ISLocalToGlobalMappingCreate(comm, 99, 0, NULL, PETSC_OWN_POINTER, &lg2l));

    PetscCall(PetscViewerBinaryOpen(comm, "testfile", FILE_MODE_READ, &vx));
    PetscCall(PetscPrintf(comm, "View world maps loaded from subcomm\n"));
    PetscCall(ISLocalToGlobalMappingLoad(lg1l, vx));
    PetscCall(ISLocalToGlobalMappingLoad(lg2l, vx));
    PetscCall(ISLocalToGlobalMappingView(lg1l, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(ISLocalToGlobalMappingView(lg2l, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(PetscViewerDestroy(&vx));

    PetscCall(ISLocalToGlobalMappingDestroy(&lg1l));
    PetscCall(ISLocalToGlobalMappingDestroy(&lg2l));
    PetscCall(ISLocalToGlobalMappingCreate(comm, 77, 1, &n, PETSC_USE_POINTER, &lg1l));
    PetscCall(ISLocalToGlobalMappingCreate(comm, 99, 0, NULL, PETSC_OWN_POINTER, &lg2l));
    PetscCall(PetscViewerBinaryOpen(comm, "testfile_seq_0", FILE_MODE_READ, &vx));
    PetscCall(PetscPrintf(comm, "View sequential maps from rank 0 loaded from subcomm\n"));
    PetscCall(ISLocalToGlobalMappingLoad(lg1l, vx));
    PetscCall(ISLocalToGlobalMappingLoad(lg2l, vx));
    PetscCall(ISLocalToGlobalMappingView(lg1l, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(ISLocalToGlobalMappingView(lg2l, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(PetscViewerDestroy(&vx));
  }
  PetscCallMPI(MPI_Comm_free(&comm));

  PetscCall(ISLocalToGlobalMappingDestroy(&lg1lh));
  PetscCall(ISLocalToGlobalMappingDestroy(&lg2lh));
  PetscCall(ISLocalToGlobalMappingDestroy(&lg1l));
  PetscCall(ISLocalToGlobalMappingDestroy(&lg2l));
  PetscCall(ISLocalToGlobalMappingDestroy(&lg1v));
  PetscCall(ISLocalToGlobalMappingDestroy(&lg2v));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      args: -viewer_binary_mpiio 0
      test:
        output_file: output/ex8_1_1.out
        suffix: 1_stdio_1
        nsize: 1
      test:
        output_file: output/ex8_1_2.out
        suffix: 1_stdio_2
        nsize: 2
      test:
        output_file: output/ex8_1_3.out
        suffix: 1_stdio_3
        nsize: 3

   testset:
      requires: mpiio
      args: -viewer_binary_mpiio 1
      test:
        output_file: output/ex8_1_1.out
        suffix: 1_mpiio_1
        nsize: 1
      test:
        output_file: output/ex8_1_2.out
        suffix: 1_mpiio_2
        nsize: 2
      test:
        output_file: output/ex8_1_3.out
        suffix: 1_mpiio_3
        nsize: 3

TEST*/
