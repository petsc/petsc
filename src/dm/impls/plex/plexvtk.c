#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/
#include <../src/sys/classes/viewer/impls/vtk/vtkvimpl.h>

PetscErrorCode DMPlexVTKGetCellType_Internal(DM dm, PetscInt dim, PetscInt corners, PetscInt *cellType)
{
  PetscFunctionBegin;
  *cellType = -1;
  switch (dim) {
  case 0:
    switch (corners) {
    case 1:
      *cellType = 1; /* VTK_VERTEX */
      break;
    default:
      break;
    }
    break;
  case 1:
    switch (corners) {
    case 2:
      *cellType = 3; /* VTK_LINE */
      break;
    case 3:
      *cellType = 21; /* VTK_QUADRATIC_EDGE */
      break;
    default:
      break;
    }
    break;
  case 2:
    switch (corners) {
    case 3:
      *cellType = 5; /* VTK_TRIANGLE */
      break;
    case 4:
      *cellType = 9; /* VTK_QUAD */
      break;
    case 6:
      *cellType = 22; /* VTK_QUADRATIC_TRIANGLE */
      break;
    case 9:
      *cellType = 23; /* VTK_QUADRATIC_QUAD */
      break;
    default:
      break;
    }
    break;
  case 3:
    switch (corners) {
    case 4:
      *cellType = 10; /* VTK_TETRA */
      break;
    case 6:
      *cellType = 13; /* VTK_WEDGE */
      break;
    case 8:
      *cellType = 12; /* VTK_HEXAHEDRON */
      break;
    case 10:
      *cellType = 24; /* VTK_QUADRATIC_TETRA */
      break;
    case 27:
      *cellType = 29; /* VTK_QUADRATIC_HEXAHEDRON */
      break;
    default:
      break;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexVTKWriteCells_ASCII(DM dm, FILE *fp, PetscInt *totalCells)
{
  MPI_Comm       comm;
  DMLabel        label;
  IS             globalVertexNumbers = NULL;
  const PetscInt *gvertex;
  PetscInt       dim;
  PetscInt       numCorners = 0, totCorners = 0, maxCorners, *corners;
  PetscInt       numCells   = 0, totCells   = 0, maxCells, cellHeight;
  PetscInt       numLabelCells, maxLabelCells, cStart, cEnd, c, vStart, vEnd, v;
  PetscMPIInt    size, rank, proc, tag;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCall(PetscCommGetNewTag(comm, &tag));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
  PetscCall(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMGetLabel(dm, "vtk", &label));
  PetscCall(DMGetStratumSize(dm, "vtk", 1, &numLabelCells));
  PetscCallMPI(MPIU_Allreduce(&numLabelCells, &maxLabelCells, 1, MPIU_INT, MPI_MAX, comm));
  if (!maxLabelCells) label = NULL;
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL;
    PetscInt closureSize, value;

    if (label) {
      PetscCall(DMLabelGetValue(label, c, &value));
      if (value != 1) continue;
    }
    PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    for (v = 0; v < closureSize*2; v += 2) {
      if ((closure[v] >= vStart) && (closure[v] < vEnd)) ++numCorners;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    ++numCells;
  }
  maxCells = numCells;
  PetscCallMPI(MPI_Reduce(&numCells, &totCells, 1, MPIU_INT, MPI_SUM, 0, comm));
  PetscCallMPI(MPI_Reduce(&numCells, &maxCells, 1, MPIU_INT, MPI_MAX, 0, comm));
  PetscCallMPI(MPI_Reduce(&numCorners, &totCorners, 1, MPIU_INT, MPI_SUM, 0, comm));
  PetscCallMPI(MPI_Reduce(&numCorners, &maxCorners, 1, MPIU_INT, MPI_MAX, 0, comm));
  PetscCall(DMPlexGetVertexNumbering(dm, &globalVertexNumbers));
  PetscCall(ISGetIndices(globalVertexNumbers, &gvertex));
  PetscCall(PetscMalloc1(maxCells, &corners));
  PetscCall(PetscFPrintf(comm, fp, "CELLS %D %D\n", totCells, totCorners+totCells));
  if (rank == 0) {
    PetscInt *remoteVertices, *vertices;

    PetscCall(PetscMalloc1(maxCorners, &vertices));
    for (c = cStart, numCells = 0; c < cEnd; ++c) {
      PetscInt *closure = NULL;
      PetscInt closureSize, value, nC = 0;

      if (label) {
        PetscCall(DMLabelGetValue(label, c, &value));
        if (value != 1) continue;
      }
      PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
      for (v = 0; v < closureSize*2; v += 2) {
        if ((closure[v] >= vStart) && (closure[v] < vEnd)) {
          const PetscInt gv = gvertex[closure[v] - vStart];
          vertices[nC++] = gv < 0 ? -(gv+1) : gv;
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
      PetscCall(DMPlexReorderCell(dm, c, vertices));
      corners[numCells++] = nC;
      PetscCall(PetscFPrintf(comm, fp, "%D ", nC));
      for (v = 0; v < nC; ++v) {
        PetscCall(PetscFPrintf(comm, fp, " %D", vertices[v]));
      }
      PetscCall(PetscFPrintf(comm, fp, "\n"));
    }
    if (size > 1) PetscCall(PetscMalloc1(maxCorners+maxCells, &remoteVertices));
    for (proc = 1; proc < size; ++proc) {
      MPI_Status status;

      PetscCallMPI(MPI_Recv(&numCorners, 1, MPIU_INT, proc, tag, comm, &status));
      PetscCallMPI(MPI_Recv(remoteVertices, numCorners, MPIU_INT, proc, tag, comm, &status));
      for (c = 0; c < numCorners;) {
        PetscInt nC = remoteVertices[c++];

        for (v = 0; v < nC; ++v, ++c) {
          vertices[v] = remoteVertices[c];
        }
        PetscCall(PetscFPrintf(comm, fp, "%D ", nC));
        for (v = 0; v < nC; ++v) {
          PetscCall(PetscFPrintf(comm, fp, " %D", vertices[v]));
        }
        PetscCall(PetscFPrintf(comm, fp, "\n"));
      }
    }
    if (size > 1) PetscCall(PetscFree(remoteVertices));
    PetscCall(PetscFree(vertices));
  } else {
    PetscInt *localVertices, numSend = numCells+numCorners, k = 0;

    PetscCall(PetscMalloc1(numSend, &localVertices));
    for (c = cStart, numCells = 0; c < cEnd; ++c) {
      PetscInt *closure = NULL;
      PetscInt closureSize, value, nC = 0;

      if (label) {
        PetscCall(DMLabelGetValue(label, c, &value));
        if (value != 1) continue;
      }
      PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
      for (v = 0; v < closureSize*2; v += 2) {
        if ((closure[v] >= vStart) && (closure[v] < vEnd)) {
          const PetscInt gv = gvertex[closure[v] - vStart];
          closure[nC++] = gv < 0 ? -(gv+1) : gv;
        }
      }
      corners[numCells++] = nC;
      localVertices[k++]  = nC;
      for (v = 0; v < nC; ++v, ++k) {
        localVertices[k] = closure[v];
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
      PetscCall(DMPlexReorderCell(dm, c, localVertices+k-nC));
    }
    PetscCheckFalse(k != numSend,PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of vertices to send %D should be %D", k, numSend);
    PetscCallMPI(MPI_Send(&numSend, 1, MPIU_INT, 0, tag, comm));
    PetscCallMPI(MPI_Send(localVertices, numSend, MPIU_INT, 0, tag, comm));
    PetscCall(PetscFree(localVertices));
  }
  PetscCall(ISRestoreIndices(globalVertexNumbers, &gvertex));
  PetscCall(PetscFPrintf(comm, fp, "CELL_TYPES %D\n", totCells));
  if (rank == 0) {
    PetscInt cellType;

    for (c = 0; c < numCells; ++c) {
      PetscCall(DMPlexVTKGetCellType_Internal(dm, dim, corners[c], &cellType));
      PetscCall(PetscFPrintf(comm, fp, "%D\n", cellType));
    }
    for (proc = 1; proc < size; ++proc) {
      MPI_Status status;

      PetscCallMPI(MPI_Recv(&numCells, 1, MPIU_INT, proc, tag, comm, &status));
      PetscCallMPI(MPI_Recv(corners, numCells, MPIU_INT, proc, tag, comm, &status));
      for (c = 0; c < numCells; ++c) {
        PetscCall(DMPlexVTKGetCellType_Internal(dm, dim, corners[c], &cellType));
        PetscCall(PetscFPrintf(comm, fp, "%D\n", cellType));
      }
    }
  } else {
    PetscCallMPI(MPI_Send(&numCells, 1, MPIU_INT, 0, tag, comm));
    PetscCallMPI(MPI_Send(corners, numCells, MPIU_INT, 0, tag, comm));
  }
  PetscCall(PetscFree(corners));
  *totalCells = totCells;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexVTKWritePartition_ASCII(DM dm, FILE *fp)
{
  MPI_Comm       comm;
  PetscInt       numCells = 0, cellHeight;
  PetscInt       numLabelCells, cStart, cEnd, c;
  PetscMPIInt    size, rank, proc, tag;
  PetscBool      hasLabel;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCall(PetscCommGetNewTag(comm, &tag));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
  PetscCall(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  PetscCall(DMGetStratumSize(dm, "vtk", 1, &numLabelCells));
  hasLabel = numLabelCells > 0 ? PETSC_TRUE : PETSC_FALSE;
  for (c = cStart; c < cEnd; ++c) {
    if (hasLabel) {
      PetscInt value;

      PetscCall(DMGetLabelValue(dm, "vtk", c, &value));
      if (value != 1) continue;
    }
    ++numCells;
  }
  if (rank == 0) {
    for (c = 0; c < numCells; ++c) PetscCall(PetscFPrintf(comm, fp, "%d\n", rank));
    for (proc = 1; proc < size; ++proc) {
      MPI_Status status;

      PetscCallMPI(MPI_Recv(&numCells, 1, MPIU_INT, proc, tag, comm, &status));
      for (c = 0; c < numCells; ++c) PetscCall(PetscFPrintf(comm, fp, "%d\n", proc));
    }
  } else {
    PetscCallMPI(MPI_Send(&numCells, 1, MPIU_INT, 0, tag, comm));
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL___FLOAT128)
typedef double PetscVTKReal;
#elif defined(PETSC_USE_REAL_SINGLE) || defined(PETSC_USE_REAL___FP16)
typedef float PetscVTKReal;
#else
typedef PetscReal PetscVTKReal;
#endif

static PetscErrorCode DMPlexVTKWriteSection_ASCII(DM dm, PetscSection section, PetscSection globalSection, Vec v, FILE *fp, PetscInt enforceDof, PetscInt precision, PetscReal scale, PetscInt imag)
{
  MPI_Comm           comm;
  const MPI_Datatype mpiType = MPIU_SCALAR;
  PetscScalar        *array;
  PetscInt           numDof = 0, maxDof;
  PetscInt           numLabelCells, cellHeight, cStart, cEnd, numLabelVertices, vStart, vEnd, pStart, pEnd, p;
  PetscMPIInt        size, rank, proc, tag;
  PetscBool          hasLabel;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,4);
  if (precision < 0) precision = 6;
  PetscCall(PetscCommGetNewTag(comm, &tag));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
  /* VTK only wants the values at cells or vertices */
  PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
  PetscCall(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  pStart   = PetscMax(PetscMin(cStart, vStart), pStart);
  pEnd     = PetscMin(PetscMax(cEnd,   vEnd),   pEnd);
  PetscCall(DMGetStratumSize(dm, "vtk", 1, &numLabelCells));
  PetscCall(DMGetStratumSize(dm, "vtk", 2, &numLabelVertices));
  hasLabel = numLabelCells > 0 || numLabelVertices > 0 ? PETSC_TRUE : PETSC_FALSE;
  for (p = pStart; p < pEnd; ++p) {
    /* Reject points not either cells or vertices */
    if (((p < cStart) || (p >= cEnd)) && ((p < vStart) || (p >= vEnd))) continue;
    if (hasLabel) {
      PetscInt value;

      if (((p >= cStart) && (p < cEnd) && numLabelCells) ||
          ((p >= vStart) && (p < vEnd) && numLabelVertices)) {
        PetscCall(DMGetLabelValue(dm, "vtk", p, &value));
        if (value != 1) continue;
      }
    }
    PetscCall(PetscSectionGetDof(section, p, &numDof));
    if (numDof) break;
  }
  PetscCallMPI(MPIU_Allreduce(&numDof, &maxDof, 1, MPIU_INT, MPI_MAX, comm));
  enforceDof = PetscMax(enforceDof, maxDof);
  PetscCall(VecGetArray(v, &array));
  if (rank == 0) {
    PetscVTKReal dval;
    PetscScalar  val;
    char formatString[8];

    PetscCall(PetscSNPrintf(formatString, 8, "%%.%de", precision));
    for (p = pStart; p < pEnd; ++p) {
      /* Here we lose a way to filter points by keeping them out of the Numbering */
      PetscInt dof, off, goff, d;

      /* Reject points not either cells or vertices */
      if (((p < cStart) || (p >= cEnd)) && ((p < vStart) || (p >= vEnd))) continue;
      if (hasLabel) {
        PetscInt value;

        if (((p >= cStart) && (p < cEnd) && numLabelCells) ||
            ((p >= vStart) && (p < vEnd) && numLabelVertices)) {
          PetscCall(DMGetLabelValue(dm, "vtk", p, &value));
          if (value != 1) continue;
        }
      }
      PetscCall(PetscSectionGetDof(section, p, &dof));
      PetscCall(PetscSectionGetOffset(section, p, &off));
      PetscCall(PetscSectionGetOffset(globalSection, p, &goff));
      if (dof && goff >= 0) {
        for (d = 0; d < dof; d++) {
          if (d > 0) {
            PetscCall(PetscFPrintf(comm, fp, " "));
          }
          val = array[off+d];
          dval = (PetscVTKReal) ((imag ? PetscImaginaryPart(val) : PetscRealPart(val)) * scale);
          PetscCall(PetscFPrintf(comm, fp, formatString, dval));
        }
        for (d = dof; d < enforceDof; d++) {
          PetscCall(PetscFPrintf(comm, fp, " 0.0"));
        }
        PetscCall(PetscFPrintf(comm, fp, "\n"));
      }
    }
    for (proc = 1; proc < size; ++proc) {
      PetscScalar *remoteValues;
      PetscInt    size = 0, d;
      MPI_Status  status;

      PetscCallMPI(MPI_Recv(&size, 1, MPIU_INT, proc, tag, comm, &status));
      PetscCall(PetscMalloc1(size, &remoteValues));
      PetscCallMPI(MPI_Recv(remoteValues, size, mpiType, proc, tag, comm, &status));
      for (p = 0; p < size/maxDof; ++p) {
        for (d = 0; d < maxDof; ++d) {
          if (d > 0) {
            PetscCall(PetscFPrintf(comm, fp, " "));
          }
          val = remoteValues[p*maxDof+d];
          dval = (PetscVTKReal) ((imag ? PetscImaginaryPart(val) : PetscRealPart(val)) * scale);
          PetscCall(PetscFPrintf(comm, fp, formatString, dval));
        }
        for (d = maxDof; d < enforceDof; ++d) {
          PetscCall(PetscFPrintf(comm, fp, " 0.0"));
        }
        PetscCall(PetscFPrintf(comm, fp, "\n"));
      }
      PetscCall(PetscFree(remoteValues));
    }
  } else {
    PetscScalar *localValues;
    PetscInt    size, k = 0;

    PetscCall(PetscSectionGetStorageSize(section, &size));
    PetscCall(PetscMalloc1(size, &localValues));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, goff, d;

      /* Reject points not either cells or vertices */
      if (((p < cStart) || (p >= cEnd)) && ((p < vStart) || (p >= vEnd))) continue;
      if (hasLabel) {
        PetscInt value;

        if (((p >= cStart) && (p < cEnd) && numLabelCells) ||
            ((p >= vStart) && (p < vEnd) && numLabelVertices)) {
          PetscCall(DMGetLabelValue(dm, "vtk", p, &value));
          if (value != 1) continue;
        }
      }
      PetscCall(PetscSectionGetDof(section, p, &dof));
      PetscCall(PetscSectionGetOffset(section, p, &off));
      PetscCall(PetscSectionGetOffset(globalSection, p, &goff));
      if (goff >= 0) {
        for (d = 0; d < dof; ++d) {
          localValues[k++] = array[off+d];
        }
      }
    }
    PetscCallMPI(MPI_Send(&k, 1, MPIU_INT, 0, tag, comm));
    PetscCallMPI(MPI_Send(localValues, k, mpiType, 0, tag, comm));
    PetscCall(PetscFree(localValues));
  }
  PetscCall(VecRestoreArray(v, &array));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexVTKWriteField_ASCII(DM dm, PetscSection section, PetscSection globalSection, Vec field, const char name[], FILE *fp, PetscInt enforceDof, PetscInt precision, PetscReal scale, PetscBool nameComplex, PetscInt imag)
{
  MPI_Comm       comm;
  PetscInt       numDof = 0, maxDof;
  PetscInt       pStart, pEnd, p;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscCall(PetscSectionGetDof(section, p, &numDof));
    if (numDof) break;
  }
  numDof = PetscMax(numDof, enforceDof);
  PetscCallMPI(MPIU_Allreduce(&numDof, &maxDof, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)));
  if (!name) name = "Unknown";
  if (maxDof == 3) {
    if (nameComplex) {
      PetscCall(PetscFPrintf(comm, fp, "VECTORS %s.%s double\n", name, imag ? "Im" : "Re"));
    } else {
      PetscCall(PetscFPrintf(comm, fp, "VECTORS %s double\n", name));
    }
  } else {
    if (nameComplex) {
      PetscCall(PetscFPrintf(comm, fp, "SCALARS %s.%s double %D\n", name, imag ? "Im" : "Re", maxDof));
    } else {
      PetscCall(PetscFPrintf(comm, fp, "SCALARS %s double %D\n", name, maxDof));
    }
    PetscCall(PetscFPrintf(comm, fp, "LOOKUP_TABLE default\n"));
  }
  PetscCall(DMPlexVTKWriteSection_ASCII(dm, section, globalSection, field, fp, enforceDof, precision, scale, imag));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexVTKWriteAll_ASCII(DM dm, PetscViewer viewer)
{
  MPI_Comm                 comm;
  PetscViewer_VTK          *vtk = (PetscViewer_VTK*) viewer->data;
  FILE                     *fp;
  PetscViewerVTKObjectLink link;
  PetscInt                 totVertices, totCells = 0, loops_per_scalar, l;
  PetscBool                hasPoint = PETSC_FALSE, hasCell = PETSC_FALSE, writePartition = PETSC_FALSE, localized, writeComplex;
  const char               *dmname;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  loops_per_scalar = 2;
  writeComplex = PETSC_TRUE;
#else
  loops_per_scalar = 1;
  writeComplex = PETSC_FALSE;
#endif
  PetscCall(DMGetCoordinatesLocalized(dm,&localized));
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCheckFalse(localized,comm,PETSC_ERR_SUP,"VTK output with localized coordinates not yet supported");
  PetscCall(PetscFOpen(comm, vtk->filename, "wb", &fp));
  PetscCall(PetscObjectGetName((PetscObject)dm, &dmname));
  PetscCall(PetscFPrintf(comm, fp, "# vtk DataFile Version 2.0\n"));
  PetscCall(PetscFPrintf(comm, fp, "%s\n", dmname));
  PetscCall(PetscFPrintf(comm, fp, "ASCII\n"));
  PetscCall(PetscFPrintf(comm, fp, "DATASET UNSTRUCTURED_GRID\n"));
  /* Vertices */
  {
    PetscSection  section, coordSection, globalCoordSection;
    Vec           coordinates;
    PetscReal     lengthScale;
    DMLabel       label;
    IS            vStratumIS;
    PetscLayout   vLayout;

    PetscCall(DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale));
    PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
    PetscCall(DMPlexGetDepthLabel(dm, &label));
    PetscCall(DMLabelGetStratumIS(label, 0, &vStratumIS));
    PetscCall(DMGetCoordinateSection(dm, &section));                                 /* This section includes all points */
    PetscCall(PetscSectionCreateSubmeshSection(section, vStratumIS, &coordSection)); /* This one includes just vertices */
    PetscCall(PetscSectionCreateGlobalSection(coordSection, dm->sf, PETSC_FALSE, PETSC_FALSE, &globalCoordSection));
    PetscCall(PetscSectionGetPointLayout(comm, globalCoordSection, &vLayout));
    PetscCall(PetscLayoutGetSize(vLayout, &totVertices));
    PetscCall(PetscFPrintf(comm, fp, "POINTS %D double\n", totVertices));
    PetscCall(DMPlexVTKWriteSection_ASCII(dm, coordSection, globalCoordSection, coordinates, fp, 3, PETSC_DETERMINE, lengthScale, 0));
    PetscCall(ISDestroy(&vStratumIS));
    PetscCall(PetscLayoutDestroy(&vLayout));
    PetscCall(PetscSectionDestroy(&coordSection));
    PetscCall(PetscSectionDestroy(&globalCoordSection));
  }
  /* Cells */
  PetscCall(DMPlexVTKWriteCells_ASCII(dm, fp, &totCells));
  /* Vertex fields */
  for (link = vtk->link; link; link = link->next) {
    if ((link->ft == PETSC_VTK_POINT_FIELD) || (link->ft == PETSC_VTK_POINT_VECTOR_FIELD)) hasPoint = PETSC_TRUE;
    if ((link->ft == PETSC_VTK_CELL_FIELD)  || (link->ft == PETSC_VTK_CELL_VECTOR_FIELD))  hasCell  = PETSC_TRUE;
  }
  if (hasPoint) {
    PetscCall(PetscFPrintf(comm, fp, "POINT_DATA %D\n", totVertices));
    for (link = vtk->link; link; link = link->next) {
      Vec          X = (Vec) link->vec;
      PetscSection section = NULL, globalSection, newSection = NULL;
      char         namebuf[256];
      const char   *name;
      PetscInt     enforceDof = PETSC_DETERMINE;

      if ((link->ft != PETSC_VTK_POINT_FIELD) && (link->ft != PETSC_VTK_POINT_VECTOR_FIELD)) continue;
      if (link->ft == PETSC_VTK_POINT_VECTOR_FIELD) enforceDof = 3;
      PetscCall(PetscObjectGetName(link->vec, &name));
      PetscCall(PetscObjectQuery(link->vec, "section", (PetscObject*) &section));
      if (!section) {
        DM           dmX;

        PetscCall(VecGetDM(X, &dmX));
        if (dmX) {
          DMLabel  subpointMap, subpointMapX;
          PetscInt dim, dimX, pStart, pEnd, qStart, qEnd;

          PetscCall(DMGetLocalSection(dmX, &section));
          /* Here is where we check whether dmX is a submesh of dm */
          PetscCall(DMGetDimension(dm,  &dim));
          PetscCall(DMGetDimension(dmX, &dimX));
          PetscCall(DMPlexGetChart(dm,  &pStart, &pEnd));
          PetscCall(DMPlexGetChart(dmX, &qStart, &qEnd));
          PetscCall(DMPlexGetSubpointMap(dm,  &subpointMap));
          PetscCall(DMPlexGetSubpointMap(dmX, &subpointMapX));
          if (((dim != dimX) || ((pEnd-pStart) < (qEnd-qStart))) && subpointMap && !subpointMapX) {
            const PetscInt *ind = NULL;
            IS              subpointIS;
            PetscInt        n = 0, q;

            PetscCall(PetscSectionGetChart(section, &qStart, &qEnd));
            PetscCall(DMPlexGetSubpointIS(dm, &subpointIS));
            if (subpointIS) {
              PetscCall(ISGetLocalSize(subpointIS, &n));
              PetscCall(ISGetIndices(subpointIS, &ind));
            }
            PetscCall(PetscSectionCreate(comm, &newSection));
            PetscCall(PetscSectionSetChart(newSection, pStart, pEnd));
            for (q = qStart; q < qEnd; ++q) {
              PetscInt dof, off, p;

              PetscCall(PetscSectionGetDof(section, q, &dof));
              if (dof) {
                PetscCall(PetscFindInt(q, n, ind, &p));
                if (p >= pStart) {
                  PetscCall(PetscSectionSetDof(newSection, p, dof));
                  PetscCall(PetscSectionGetOffset(section, q, &off));
                  PetscCall(PetscSectionSetOffset(newSection, p, off));
                }
              }
            }
            if (subpointIS) {
              PetscCall(ISRestoreIndices(subpointIS, &ind));
            }
            /* No need to setup section */
            section = newSection;
          }
        }
      }
      PetscCheck(section,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Vector %s had no PetscSection composed with it and could not create one from VecGetDM()", name);
      if (link->field >= 0) {
        const char *fieldname;

        PetscCall(PetscSectionGetFieldName(section, link->field, &fieldname));
        PetscCall(PetscSectionGetField(section, link->field, &section));
        if (fieldname) {
          PetscCall(PetscSNPrintf(namebuf, sizeof(namebuf), "%s%s", name, fieldname));
        } else {
          PetscCall(PetscSNPrintf(namebuf, sizeof(namebuf), "%s%D", name, link->field));
        }
      } else {
        PetscCall(PetscSNPrintf(namebuf, sizeof(namebuf), "%s", name));
      }
      PetscCall(PetscViewerVTKSanitizeName_Internal(namebuf, sizeof(namebuf)));
      PetscCall(PetscSectionCreateGlobalSection(section, dm->sf, PETSC_FALSE, PETSC_FALSE, &globalSection));
      for (l = 0; l < loops_per_scalar; l++) {
        PetscCall(DMPlexVTKWriteField_ASCII(dm, section, globalSection, X, namebuf, fp, enforceDof, PETSC_DETERMINE, 1.0, writeComplex, l));
      }
      PetscCall(PetscSectionDestroy(&globalSection));
      if (newSection) PetscCall(PetscSectionDestroy(&newSection));
    }
  }
  /* Cell Fields */
  PetscCall(PetscOptionsGetBool(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_view_partition", &writePartition, NULL));
  if (hasCell || writePartition) {
    PetscCall(PetscFPrintf(comm, fp, "CELL_DATA %D\n", totCells));
    for (link = vtk->link; link; link = link->next) {
      Vec          X = (Vec) link->vec;
      PetscSection section = NULL, globalSection;
      const char   *name = "";
      char         namebuf[256];
      PetscInt     enforceDof = PETSC_DETERMINE;

      if ((link->ft != PETSC_VTK_CELL_FIELD) && (link->ft != PETSC_VTK_CELL_VECTOR_FIELD)) continue;
      if (link->ft == PETSC_VTK_CELL_VECTOR_FIELD) enforceDof = 3;
      PetscCall(PetscObjectGetName(link->vec, &name));
      PetscCall(PetscObjectQuery(link->vec, "section", (PetscObject*) &section));
      if (!section) {
        DM           dmX;

        PetscCall(VecGetDM(X, &dmX));
        if (dmX) {
          PetscCall(DMGetLocalSection(dmX, &section));
        }
      }
      PetscCheck(section,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Vector %s had no PetscSection composed with it and could not create one from VecGetDM()", name);
      if (link->field >= 0) {
        const char *fieldname;

        PetscCall(PetscSectionGetFieldName(section, link->field, &fieldname));
        PetscCall(PetscSectionGetField(section, link->field, &section));
        if (fieldname) {
          PetscCall(PetscSNPrintf(namebuf, sizeof(namebuf), "%s%s", name, fieldname));
        } else {
          PetscCall(PetscSNPrintf(namebuf, sizeof(namebuf), "%s%D", name, link->field));
        }
      } else {
        PetscCall(PetscSNPrintf(namebuf, sizeof(namebuf), "%s", name));
      }
      PetscCall(PetscViewerVTKSanitizeName_Internal(namebuf, sizeof(namebuf)));
      PetscCall(PetscSectionCreateGlobalSection(section, dm->sf, PETSC_FALSE, PETSC_FALSE, &globalSection));
      for (l = 0; l < loops_per_scalar; l++) {
        PetscCall(DMPlexVTKWriteField_ASCII(dm, section, globalSection, X, namebuf, fp, enforceDof, PETSC_DETERMINE, 1.0, writeComplex, l));
      }
      PetscCall(PetscSectionDestroy(&globalSection));
    }
    if (writePartition) {
      PetscCall(PetscFPrintf(comm, fp, "SCALARS partition int 1\n"));
      PetscCall(PetscFPrintf(comm, fp, "LOOKUP_TABLE default\n"));
      PetscCall(DMPlexVTKWritePartition_ASCII(dm, fp));
    }
  }
  /* Cleanup */
  PetscCall(PetscFClose(comm, fp));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexVTKWriteAll - Write a file containing all the fields that have been provided to the viewer

  Collective

  Input Parameters:
+ odm - The DMPlex specifying the mesh, passed as a PetscObject
- viewer - viewer of type VTK

  Level: developer

  Note:
  This function is a callback used by the VTK viewer to actually write the file.
  The reason for this odd model is that the VTK file format does not provide any way to write one field at a time.
  Instead, metadata for the entire file needs to be available up-front before you can start writing the file.

.seealso: PETSCVIEWERVTK
@*/
PetscErrorCode DMPlexVTKWriteAll(PetscObject odm, PetscViewer viewer)
{
  DM             dm = (DM) odm;
  PetscBool      isvtk;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK, &isvtk));
  PetscCheck(isvtk,PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_INCOMP, "Cannot use viewer type %s", ((PetscObject)viewer)->type_name);
  switch (viewer->format) {
  case PETSC_VIEWER_ASCII_VTK_DEPRECATED:
    PetscCall(DMPlexVTKWriteAll_ASCII(dm, viewer));
    break;
  case PETSC_VIEWER_VTK_VTU:
    PetscCall(DMPlexVTKWriteAll_VTU(dm, viewer));
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "No support for format '%s'", PetscViewerFormats[viewer->format]);
  }
  PetscFunctionReturn(0);
}
