
static char help[] = "Tests PetscViewerBinary VecView()/VecLoad() function correctly when binary header is skipped.\n\n";

#include <petscviewer.h>
#include <petscvec.h>

#define VEC_LEN 10
const PetscReal test_values[] = {0.311256, 88.068, 11.077444, 9953.62, 7.345, 64.8943, 3.1458, 6699.95, 0.00084, 0.0647};

PetscErrorCode MyVecDump(const char fname[], PetscBool skippheader, PetscBool usempiio, Vec x)
{
  MPI_Comm    comm;
  PetscViewer viewer;
  PetscBool   ismpiio, isskip;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)x, &comm));

  PetscCall(PetscViewerCreate(comm, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERBINARY));
  if (skippheader) PetscCall(PetscViewerBinarySetSkipHeader(viewer, PETSC_TRUE));
  PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));
  if (usempiio) PetscCall(PetscViewerBinarySetUseMPIIO(viewer, PETSC_TRUE));
  PetscCall(PetscViewerFileSetName(viewer, fname));

  PetscCall(VecView(x, viewer));

  PetscCall(PetscViewerBinaryGetUseMPIIO(viewer, &ismpiio));
  if (ismpiio) PetscCall(PetscPrintf(comm, "*** PetscViewer[write] using MPI-IO ***\n"));
  PetscCall(PetscViewerBinaryGetSkipHeader(viewer, &isskip));
  if (isskip) PetscCall(PetscPrintf(comm, "*** PetscViewer[write] skipping header ***\n"));

  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode MyVecLoad(const char fname[], PetscBool skippheader, PetscBool usempiio, Vec x)
{
  MPI_Comm    comm;
  PetscViewer viewer;
  PetscBool   ismpiio, isskip;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)x, &comm));

  PetscCall(PetscViewerCreate(comm, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERBINARY));
  if (skippheader) PetscCall(PetscViewerBinarySetSkipHeader(viewer, PETSC_TRUE));
  PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_READ));
  if (usempiio) PetscCall(PetscViewerBinarySetUseMPIIO(viewer, PETSC_TRUE));
  PetscCall(PetscViewerFileSetName(viewer, fname));

  PetscCall(VecLoad(x, viewer));

  PetscCall(PetscViewerBinaryGetSkipHeader(viewer, &isskip));
  if (isskip) PetscCall(PetscPrintf(comm, "*** PetscViewer[load] skipping header ***\n"));
  PetscCall(PetscViewerBinaryGetUseMPIIO(viewer, &ismpiio));
  if (ismpiio) PetscCall(PetscPrintf(comm, "*** PetscViewer[load] using MPI-IO ***\n"));

  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode VecFill(Vec x)
{
  PetscInt i, s, e;

  PetscFunctionBeginUser;
  PetscCall(VecGetOwnershipRange(x, &s, &e));
  for (i = s; i < e; i++) PetscCall(VecSetValue(x, i, (PetscScalar)test_values[i], INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));
  PetscFunctionReturn(0);
}

PetscErrorCode VecCompare(Vec a, Vec b)
{
  PetscInt  locmin[2], locmax[2];
  PetscReal min[2], max[2];
  Vec       ref;

  PetscFunctionBeginUser;
  PetscCall(VecMin(a, &locmin[0], &min[0]));
  PetscCall(VecMax(a, &locmax[0], &max[0]));

  PetscCall(VecMin(b, &locmin[1], &min[1]));
  PetscCall(VecMax(b, &locmax[1], &max[1]));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "VecCompare\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  min(a)   = %+1.2e [loc %" PetscInt_FMT "]\n", (double)min[0], locmin[0]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  max(a)   = %+1.2e [loc %" PetscInt_FMT "]\n", (double)max[0], locmax[0]));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  min(b)   = %+1.2e [loc %" PetscInt_FMT "]\n", (double)min[1], locmin[1]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  max(b)   = %+1.2e [loc %" PetscInt_FMT "]\n", (double)max[1], locmax[1]));

  PetscCall(VecDuplicate(a, &ref));
  PetscCall(VecCopy(a, ref));
  PetscCall(VecAXPY(ref, -1.0, b));
  PetscCall(VecMin(ref, &locmin[0], &min[0]));
  if (PetscAbsReal(min[0]) > 1.0e-10) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  ERROR: min(a-b) > 1.0e-10\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  min(a-b) = %+1.10e\n", (double)PetscAbsReal(min[0])));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  min(a-b) < 1.0e-10\n"));
  }
  PetscCall(VecDestroy(&ref));
  PetscFunctionReturn(0);
}

PetscErrorCode HeaderlessBinaryRead(const char name[])
{
  int         fdes;
  PetscScalar buffer[VEC_LEN];
  PetscInt    i;
  PetscMPIInt rank;
  PetscBool   dataverified = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank == 0) {
    PetscCall(PetscBinaryOpen(name, FILE_MODE_READ, &fdes));
    PetscCall(PetscBinaryRead(fdes, buffer, VEC_LEN, NULL, PETSC_SCALAR));
    PetscCall(PetscBinaryClose(fdes));

    for (i = 0; i < VEC_LEN; i++) {
      PetscScalar v;
      v = PetscAbsScalar(test_values[i] - buffer[i]);
#if defined(PETSC_USE_COMPLEX)
      if ((PetscRealPart(v) > 1.0e-10) || (PetscImaginaryPart(v) > 1.0e-10)) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "ERROR: Difference > 1.0e-10 occurred (delta = (%+1.12e,%+1.12e) [loc %" PetscInt_FMT "])\n", (double)PetscRealPart(buffer[i]), (double)PetscImaginaryPart(buffer[i]), i));
        dataverified = PETSC_FALSE;
      }
#else
      if (PetscRealPart(v) > 1.0e-10) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "ERROR: Difference > 1.0e-10 occurred (delta = %+1.12e [loc %" PetscInt_FMT "])\n", (double)PetscRealPart(buffer[i]), i));
        dataverified = PETSC_FALSE;
      }
#endif
    }
    if (dataverified) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Headerless read of data verified\n"));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TestBinary(void)
{
  Vec       x, y;
  PetscBool skipheader = PETSC_TRUE;
  PetscBool usempiio   = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, VEC_LEN));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecFill(x));
  PetscCall(MyVecDump("xH.pbvec", skipheader, usempiio, x));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
  PetscCall(VecSetSizes(y, PETSC_DECIDE, VEC_LEN));
  PetscCall(VecSetFromOptions(y));

  PetscCall(MyVecLoad("xH.pbvec", skipheader, usempiio, y));
  PetscCall(VecCompare(x, y));

  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));

  PetscCall(HeaderlessBinaryRead("xH.pbvec"));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
PetscErrorCode TestBinaryMPIIO(void)
{
  Vec       x, y;
  PetscBool skipheader = PETSC_TRUE;
  PetscBool usempiio   = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, VEC_LEN));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecFill(x));
  PetscCall(MyVecDump("xHmpi.pbvec", skipheader, usempiio, x));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
  PetscCall(VecSetSizes(y, PETSC_DECIDE, VEC_LEN));
  PetscCall(VecSetFromOptions(y));

  PetscCall(MyVecLoad("xHmpi.pbvec", skipheader, usempiio, y));
  PetscCall(VecCompare(x, y));

  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));

  PetscCall(HeaderlessBinaryRead("xHmpi.pbvec"));
  PetscFunctionReturn(0);
}
#endif

int main(int argc, char **args)
{
  PetscBool usempiio = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-usempiio", &usempiio, NULL));
  if (!usempiio) {
    PetscCall(TestBinary());
  } else {
#if defined(PETSC_HAVE_MPIIO)
    PetscCall(TestBinaryMPIIO());
#else
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Executing TestBinaryMPIIO() requires a working MPI-2 implementation\n"));
#endif
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      output_file: output/ex46_1_p1.out

   test:
      suffix: 2
      nsize: 6
      output_file: output/ex46_1_p6.out

   test:
      suffix: 3
      nsize: 12
      output_file: output/ex46_1_p12.out

   testset:
      requires: mpiio
      args: -usempiio
      test:
         suffix: mpiio_1
         output_file: output/ex46_2_p1.out
      test:
         suffix: mpiio_2
         nsize: 6
         output_file: output/ex46_2_p6.out
      test:
         suffix: mpiio_3
         nsize: 12
         output_file: output/ex46_2_p12.out

TEST*/
