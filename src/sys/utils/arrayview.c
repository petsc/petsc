#include <petsc/private/petscimpl.h> /*I "petscsys.h" I*/
#include <petscviewer.h>

/*@
  PetscIntViewNumColumns - Prints an array of integers; useful for debugging.

  Collective

  Input Parameters:
+ N      - number of integers in array
. Ncol   - number of integers to print per row
. idx    - array of integers
- viewer - an optional `PetscViewer` visualization context

  Level: intermediate

  Note:
  This may be called from within the debugger, passing 0 as the viewer

  This API may be removed in the future.

  Developer Note:
  `idx` cannot be const because may be passed to binary viewer where temporary byte swapping may be done

.seealso: `PetscViewer`, `PetscIntView()`, `PetscRealView()`
@*/
PetscErrorCode PetscIntViewNumColumns(PetscInt N, PetscInt Ncol, const PetscInt idx[], PetscViewer viewer)
{
  PetscMPIInt rank, size;
  PetscInt    j, i, n = N / Ncol, p = N % Ncol;
  PetscBool   isascii, isbinary;
  MPI_Comm    comm;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  if (N) PetscAssertPointer(idx, 3);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    for (i = 0; i < n; i++) {
      if (size > 1) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] %" PetscInt_FMT ":", rank, Ncol * i));
      } else {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%" PetscInt_FMT ":", Ncol * i));
      }
      for (j = 0; j < Ncol; j++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %" PetscInt_FMT, idx[i * Ncol + j]));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
    }
    if (p) {
      if (size > 1) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] %" PetscInt_FMT ":", rank, Ncol * n));
      } else {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%" PetscInt_FMT ":", Ncol * n));
      }
      for (i = 0; i < p; i++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %" PetscInt_FMT, idx[Ncol * n + i]));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
    }
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  } else if (isbinary) {
    PetscMPIInt *sizes, Ntotal, *displs, NN;
    PetscInt    *array;

    PetscCall(PetscMPIIntCast(N, &NN));

    if (size > 1) {
      if (rank) {
        PetscCallMPI(MPI_Gather(&NN, 1, MPI_INT, NULL, 0, MPI_INT, 0, comm));
        PetscCallMPI(MPI_Gatherv(idx, NN, MPIU_INT, NULL, NULL, NULL, MPIU_INT, 0, comm));
      } else {
        PetscCall(PetscMalloc1(size, &sizes));
        PetscCallMPI(MPI_Gather(&NN, 1, MPI_INT, sizes, 1, MPI_INT, 0, comm));
        Ntotal = sizes[0];
        PetscCall(PetscMalloc1(size, &displs));
        displs[0] = 0;
        for (i = 1; i < size; i++) {
          Ntotal += sizes[i];
          displs[i] = displs[i - 1] + sizes[i - 1];
        }
        PetscCall(PetscMalloc1(Ntotal, &array));
        PetscCallMPI(MPI_Gatherv(idx, NN, MPIU_INT, array, sizes, displs, MPIU_INT, 0, comm));
        PetscCall(PetscViewerBinaryWrite(viewer, array, Ntotal, PETSC_INT));
        PetscCall(PetscFree(sizes));
        PetscCall(PetscFree(displs));
        PetscCall(PetscFree(array));
      }
    } else {
      PetscCall(PetscViewerBinaryWrite(viewer, idx, N, PETSC_INT));
    }
  } else {
    const char *tname;
    PetscCall(PetscObjectGetName((PetscObject)viewer, &tname));
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle that PetscViewer of type %s", tname);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRealViewNumColumns - Prints an array of doubles; useful for debugging.

  Collective

  Input Parameters:
+ N      - number of `PetscReal` in array
. Ncol   - number of `PetscReal` to print per row
. idx    - array of `PetscReal`
- viewer - an optional `PetscViewer` visualization context

  Level: intermediate

  Note:
  This may be called from within the debugger, passing 0 as the viewer

  This API may be removed in the future.

  Developer Note:
  `idx` cannot be const because may be passed to binary viewer where temporary byte swapping may be done

.seealso: `PetscViewer`, `PetscRealView()`, `PetscIntView()`
@*/
PetscErrorCode PetscRealViewNumColumns(PetscInt N, PetscInt Ncol, const PetscReal idx[], PetscViewer viewer)
{
  PetscMPIInt rank, size;
  PetscInt    j, i, n = N / Ncol, p = N % Ncol;
  PetscBool   isascii, isbinary;
  MPI_Comm    comm;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  if (N) PetscAssertPointer(idx, 3);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  if (isascii) {
    PetscInt tab;

    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    PetscCall(PetscViewerASCIIGetTab(viewer, &tab));
    for (i = 0; i < n; i++) {
      PetscCall(PetscViewerASCIISetTab(viewer, tab));
      if (size > 1) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] %2" PetscInt_FMT ":", rank, Ncol * i));
      } else {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%2" PetscInt_FMT ":", Ncol * i));
      }
      PetscCall(PetscViewerASCIISetTab(viewer, 0));
      for (j = 0; j < Ncol; j++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %12.4e", (double)idx[i * Ncol + j]));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
    }
    if (p) {
      PetscCall(PetscViewerASCIISetTab(viewer, tab));
      if (size > 1) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] %2" PetscInt_FMT ":", rank, Ncol * n));
      } else {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%2" PetscInt_FMT ":", Ncol * n));
      }
      PetscCall(PetscViewerASCIISetTab(viewer, 0));
      for (i = 0; i < p; i++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %12.4e", (double)idx[Ncol * n + i]));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
    }
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerASCIISetTab(viewer, tab));
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  } else if (isbinary) {
    PetscMPIInt *sizes, *displs, Ntotal, NN;
    PetscReal   *array;

    PetscCall(PetscMPIIntCast(N, &NN));

    if (size > 1) {
      if (rank) {
        PetscCallMPI(MPI_Gather(&NN, 1, MPI_INT, NULL, 0, MPI_INT, 0, comm));
        PetscCallMPI(MPI_Gatherv(idx, NN, MPIU_REAL, NULL, NULL, NULL, MPIU_REAL, 0, comm));
      } else {
        PetscCall(PetscMalloc1(size, &sizes));
        PetscCallMPI(MPI_Gather(&NN, 1, MPI_INT, sizes, 1, MPI_INT, 0, comm));
        Ntotal = sizes[0];
        PetscCall(PetscMalloc1(size, &displs));
        displs[0] = 0;
        for (i = 1; i < size; i++) {
          Ntotal += sizes[i];
          displs[i] = displs[i - 1] + sizes[i - 1];
        }
        PetscCall(PetscMalloc1(Ntotal, &array));
        PetscCallMPI(MPI_Gatherv(idx, NN, MPIU_REAL, array, sizes, displs, MPIU_REAL, 0, comm));
        PetscCall(PetscViewerBinaryWrite(viewer, array, Ntotal, PETSC_REAL));
        PetscCall(PetscFree(sizes));
        PetscCall(PetscFree(displs));
        PetscCall(PetscFree(array));
      }
    } else {
      PetscCall(PetscViewerBinaryWrite(viewer, (void *)idx, N, PETSC_REAL));
    }
  } else {
    const char *tname;
    PetscCall(PetscObjectGetName((PetscObject)viewer, &tname));
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle that PetscViewer of type %s", tname);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscScalarViewNumColumns - Prints an array of doubles; useful for debugging.

  Collective

  Input Parameters:
+ N      - number of `PetscScalar` in array
. Ncol   - number of `PetscScalar` to print per row
. idx    - array of `PetscScalar`
- viewer - an optional `PetscViewer` visualization context

  Level: intermediate

  Note:
  This may be called from within the debugger, passing 0 as the viewer

  This API may be removed in the future.

  Developer Note:
  `idx` cannot be const because may be passed to binary viewer where temporary byte swapping may be done

.seealso: `PetscViewer`, `PetscRealView()`, `PetscScalarView()`, `PetscIntView()`
@*/
PetscErrorCode PetscScalarViewNumColumns(PetscInt N, PetscInt Ncol, const PetscScalar idx[], PetscViewer viewer)
{
  PetscMPIInt rank, size;
  PetscInt    j, i, n = N / Ncol, p = N % Ncol;
  PetscBool   isascii, isbinary;
  MPI_Comm    comm;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  if (N) PetscAssertPointer(idx, 3);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    for (i = 0; i < n; i++) {
      if (size > 1) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] %2" PetscInt_FMT ":", rank, Ncol * i));
      } else {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%2" PetscInt_FMT ":", Ncol * i));
      }
      for (j = 0; j < Ncol; j++) {
#if PetscDefined(USE_COMPLEX)
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " (%12.4e,%12.4e)", (double)PetscRealPart(idx[i * Ncol + j]), (double)PetscImaginaryPart(idx[i * Ncol + j])));
#else
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %12.4e", (double)idx[i * Ncol + j]));
#endif
      }
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
    }
    if (p) {
      if (size > 1) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] %2" PetscInt_FMT ":", rank, Ncol * n));
      } else {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%2" PetscInt_FMT ":", Ncol * n));
      }
      for (i = 0; i < p; i++) {
#if PetscDefined(USE_COMPLEX)
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " (%12.4e,%12.4e)", (double)PetscRealPart(idx[n * Ncol + i]), (double)PetscImaginaryPart(idx[n * Ncol + i])));
#else
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %12.4e", (double)idx[Ncol * n + i]));
#endif
      }
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
    }
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  } else if (isbinary) {
    PetscMPIInt *sizes, Ntotal, *displs, NN;
    PetscScalar *array;

    PetscCall(PetscMPIIntCast(N, &NN));

    if (size > 1) {
      if (rank) {
        PetscCallMPI(MPI_Gather(&NN, 1, MPI_INT, NULL, 0, MPI_INT, 0, comm));
        PetscCallMPI(MPI_Gatherv((void *)idx, NN, MPIU_SCALAR, NULL, NULL, NULL, MPIU_SCALAR, 0, comm));
      } else {
        PetscCall(PetscMalloc1(size, &sizes));
        PetscCallMPI(MPI_Gather(&NN, 1, MPI_INT, sizes, 1, MPI_INT, 0, comm));
        Ntotal = sizes[0];
        PetscCall(PetscMalloc1(size, &displs));
        displs[0] = 0;
        for (i = 1; i < size; i++) {
          Ntotal += sizes[i];
          displs[i] = displs[i - 1] + sizes[i - 1];
        }
        PetscCall(PetscMalloc1(Ntotal, &array));
        PetscCallMPI(MPI_Gatherv((void *)idx, NN, MPIU_SCALAR, array, sizes, displs, MPIU_SCALAR, 0, comm));
        PetscCall(PetscViewerBinaryWrite(viewer, array, Ntotal, PETSC_SCALAR));
        PetscCall(PetscFree(sizes));
        PetscCall(PetscFree(displs));
        PetscCall(PetscFree(array));
      }
    } else {
      PetscCall(PetscViewerBinaryWrite(viewer, (void *)idx, N, PETSC_SCALAR));
    }
  } else {
    const char *tname;
    PetscCall(PetscObjectGetName((PetscObject)viewer, &tname));
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle that PetscViewer of type %s", tname);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscIntView - Prints an array of integers; useful for debugging.

  Collective

  Input Parameters:
+ N      - number of integers in array
. idx    - array of integers
- viewer - an optional `PetscViewer` visualization context

  Level: intermediate

  Note:
  This may be called from within the debugger, passing 0 as the viewer

  This API may be removed in the future.

  Same as `PetscIntViewNumColumns()` with 20 values per row

  Developer Note:
  `idx` cannot be const because may be passed to binary viewer where temporary byte swapping may be done

.seealso: `PetscViewer`, `PetscIntViewNumColumns()`, `PetscIntCSRView()`, `PetscRealView()`
@*/
PetscErrorCode PetscIntView(PetscInt N, const PetscInt idx[], PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscIntViewNumColumns(N, 20, idx, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscIntCSRView - Prints a graph represented in compressed sparse row (CSR) format; useful for debugging.

  Collective

  Input Parameters:
+ N      - number of local vertices
. ia     - row pointers of length `N + 1`
. ja     - adjacency list of length `ia[N]`
- viewer - an optional `PetscViewer` visualization context

  Level: intermediate

  Notes:
  This may be called from within the debugger, passing `NULL` as the viewer.

  `ia[0]` must be zero and the row pointers must be nondecreasing.

  Only ASCII viewers are supported.

.seealso: `PetscViewer`, `PetscIntView()`, `PetscPartitionerPartition()`
@*/
PetscErrorCode PetscIntCSRView(PetscInt N, const PetscInt ia[], const PetscInt ja[], PetscViewer viewer)
{
  PetscMPIInt rank;
  PetscBool   isascii;

  PetscFunctionBegin;
  PetscCheck(N >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of rows must be non-negative");
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  if (N) {
    PetscAssertPointer(ia, 2);
    PetscCheck(!ia[0], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "First row offset must be zero, got %" PetscInt_FMT, ia[0]);
    for (PetscInt i = 0; i < N; i++)
      PetscCheck(ia[i] <= ia[i + 1], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Row offsets must be nondecreasing, got ia[%" PetscInt_FMT "] = %" PetscInt_FMT " > ia[%" PetscInt_FMT "] = %" PetscInt_FMT, i, ia[i], i + 1, ia[i + 1]);
    if (ia[N]) PetscAssertPointer(ja, 3);
  }
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCheck(isascii, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Only ASCII viewers are supported");
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]Nv: %" PetscInt_FMT "\n", rank, N));
  for (PetscInt i = 0; i < N; i++) {
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]  ", rank));
    for (PetscInt j = ia[i]; j < ia[i + 1]; j++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%" PetscInt_FMT " ", ja[j]));
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%" PetscInt_FMT "-%" PetscInt_FMT ")\n", ia[i], ia[i + 1]));
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRealView - Prints an array of doubles; useful for debugging.

  Collective

  Input Parameters:
+ N      - number of `PetscReal` in array
. idx    - array of `PetscReal`
- viewer - an optional `PetscViewer` visualization context

  Level: intermediate

  Note:
  This may be called from within the debugger, passing 0 as the viewer

  This API may be removed in the future.

  Same as `PetscRealViewNumColumns()` with 5 values per row

  Developer Note:
  `idx` cannot be const because may be passed to binary viewer where temporary byte swapping may be done

.seealso: `PetscViewer`, `PetscIntView()`
@*/
PetscErrorCode PetscRealView(PetscInt N, const PetscReal idx[], PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscRealViewNumColumns(N, 5, idx, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscScalarView - Prints an array of `PetscScalar`; useful for debugging.

  Collective

  Input Parameters:
+ N      - number of scalars in array
. idx    - array of scalars
- viewer - an optional `PetscViewer` visualization context

  Level: intermediate

  Note:
  This may be called from within the debugger, passing 0 as the viewer

  This API may be removed in the future.

  Same as `PetscScalarViewNumColumns()` with 3 values per row

  Developer Note:
  `idx` cannot be const because may be passed to binary viewer where byte swapping may be done

.seealso: `PetscViewer`, `PetscIntView()`, `PetscRealView()`
@*/
PetscErrorCode PetscScalarView(PetscInt N, const PetscScalar idx[], PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscScalarViewNumColumns(N, 3, idx, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
