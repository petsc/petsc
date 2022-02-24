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
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  CHKERRQ(PetscCommGetNewTag(comm, &tag));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
  CHKERRQ(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMGetLabel(dm, "vtk", &label));
  CHKERRQ(DMGetStratumSize(dm, "vtk", 1, &numLabelCells));
  CHKERRMPI(MPIU_Allreduce(&numLabelCells, &maxLabelCells, 1, MPIU_INT, MPI_MAX, comm));
  if (!maxLabelCells) label = NULL;
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL;
    PetscInt closureSize, value;

    if (label) {
      CHKERRQ(DMLabelGetValue(label, c, &value));
      if (value != 1) continue;
    }
    CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    for (v = 0; v < closureSize*2; v += 2) {
      if ((closure[v] >= vStart) && (closure[v] < vEnd)) ++numCorners;
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    ++numCells;
  }
  maxCells = numCells;
  CHKERRMPI(MPI_Reduce(&numCells, &totCells, 1, MPIU_INT, MPI_SUM, 0, comm));
  CHKERRMPI(MPI_Reduce(&numCells, &maxCells, 1, MPIU_INT, MPI_MAX, 0, comm));
  CHKERRMPI(MPI_Reduce(&numCorners, &totCorners, 1, MPIU_INT, MPI_SUM, 0, comm));
  CHKERRMPI(MPI_Reduce(&numCorners, &maxCorners, 1, MPIU_INT, MPI_MAX, 0, comm));
  CHKERRQ(DMPlexGetVertexNumbering(dm, &globalVertexNumbers));
  CHKERRQ(ISGetIndices(globalVertexNumbers, &gvertex));
  CHKERRQ(PetscMalloc1(maxCells, &corners));
  CHKERRQ(PetscFPrintf(comm, fp, "CELLS %D %D\n", totCells, totCorners+totCells));
  if (rank == 0) {
    PetscInt *remoteVertices, *vertices;

    CHKERRQ(PetscMalloc1(maxCorners, &vertices));
    for (c = cStart, numCells = 0; c < cEnd; ++c) {
      PetscInt *closure = NULL;
      PetscInt closureSize, value, nC = 0;

      if (label) {
        CHKERRQ(DMLabelGetValue(label, c, &value));
        if (value != 1) continue;
      }
      CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
      for (v = 0; v < closureSize*2; v += 2) {
        if ((closure[v] >= vStart) && (closure[v] < vEnd)) {
          const PetscInt gv = gvertex[closure[v] - vStart];
          vertices[nC++] = gv < 0 ? -(gv+1) : gv;
        }
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
      CHKERRQ(DMPlexReorderCell(dm, c, vertices));
      corners[numCells++] = nC;
      CHKERRQ(PetscFPrintf(comm, fp, "%D ", nC));
      for (v = 0; v < nC; ++v) {
        CHKERRQ(PetscFPrintf(comm, fp, " %D", vertices[v]));
      }
      CHKERRQ(PetscFPrintf(comm, fp, "\n"));
    }
    if (size > 1) CHKERRQ(PetscMalloc1(maxCorners+maxCells, &remoteVertices));
    for (proc = 1; proc < size; ++proc) {
      MPI_Status status;

      CHKERRMPI(MPI_Recv(&numCorners, 1, MPIU_INT, proc, tag, comm, &status));
      CHKERRMPI(MPI_Recv(remoteVertices, numCorners, MPIU_INT, proc, tag, comm, &status));
      for (c = 0; c < numCorners;) {
        PetscInt nC = remoteVertices[c++];

        for (v = 0; v < nC; ++v, ++c) {
          vertices[v] = remoteVertices[c];
        }
        CHKERRQ(PetscFPrintf(comm, fp, "%D ", nC));
        for (v = 0; v < nC; ++v) {
          CHKERRQ(PetscFPrintf(comm, fp, " %D", vertices[v]));
        }
        CHKERRQ(PetscFPrintf(comm, fp, "\n"));
      }
    }
    if (size > 1) CHKERRQ(PetscFree(remoteVertices));
    CHKERRQ(PetscFree(vertices));
  } else {
    PetscInt *localVertices, numSend = numCells+numCorners, k = 0;

    CHKERRQ(PetscMalloc1(numSend, &localVertices));
    for (c = cStart, numCells = 0; c < cEnd; ++c) {
      PetscInt *closure = NULL;
      PetscInt closureSize, value, nC = 0;

      if (label) {
        CHKERRQ(DMLabelGetValue(label, c, &value));
        if (value != 1) continue;
      }
      CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
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
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
      CHKERRQ(DMPlexReorderCell(dm, c, localVertices+k-nC));
    }
    PetscCheckFalse(k != numSend,PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of vertices to send %D should be %D", k, numSend);
    CHKERRMPI(MPI_Send(&numSend, 1, MPIU_INT, 0, tag, comm));
    CHKERRMPI(MPI_Send(localVertices, numSend, MPIU_INT, 0, tag, comm));
    CHKERRQ(PetscFree(localVertices));
  }
  CHKERRQ(ISRestoreIndices(globalVertexNumbers, &gvertex));
  CHKERRQ(PetscFPrintf(comm, fp, "CELL_TYPES %D\n", totCells));
  if (rank == 0) {
    PetscInt cellType;

    for (c = 0; c < numCells; ++c) {
      CHKERRQ(DMPlexVTKGetCellType_Internal(dm, dim, corners[c], &cellType));
      CHKERRQ(PetscFPrintf(comm, fp, "%D\n", cellType));
    }
    for (proc = 1; proc < size; ++proc) {
      MPI_Status status;

      CHKERRMPI(MPI_Recv(&numCells, 1, MPIU_INT, proc, tag, comm, &status));
      CHKERRMPI(MPI_Recv(corners, numCells, MPIU_INT, proc, tag, comm, &status));
      for (c = 0; c < numCells; ++c) {
        CHKERRQ(DMPlexVTKGetCellType_Internal(dm, dim, corners[c], &cellType));
        CHKERRQ(PetscFPrintf(comm, fp, "%D\n", cellType));
      }
    }
  } else {
    CHKERRMPI(MPI_Send(&numCells, 1, MPIU_INT, 0, tag, comm));
    CHKERRMPI(MPI_Send(corners, numCells, MPIU_INT, 0, tag, comm));
  }
  CHKERRQ(PetscFree(corners));
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
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  CHKERRQ(PetscCommGetNewTag(comm, &tag));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
  CHKERRQ(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  CHKERRQ(DMGetStratumSize(dm, "vtk", 1, &numLabelCells));
  hasLabel = numLabelCells > 0 ? PETSC_TRUE : PETSC_FALSE;
  for (c = cStart; c < cEnd; ++c) {
    if (hasLabel) {
      PetscInt value;

      CHKERRQ(DMGetLabelValue(dm, "vtk", c, &value));
      if (value != 1) continue;
    }
    ++numCells;
  }
  if (rank == 0) {
    for (c = 0; c < numCells; ++c) CHKERRQ(PetscFPrintf(comm, fp, "%d\n", rank));
    for (proc = 1; proc < size; ++proc) {
      MPI_Status status;

      CHKERRMPI(MPI_Recv(&numCells, 1, MPIU_INT, proc, tag, comm, &status));
      for (c = 0; c < numCells; ++c) CHKERRQ(PetscFPrintf(comm, fp, "%d\n", proc));
    }
  } else {
    CHKERRMPI(MPI_Send(&numCells, 1, MPIU_INT, 0, tag, comm));
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
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,4);
  if (precision < 0) precision = 6;
  CHKERRQ(PetscCommGetNewTag(comm, &tag));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(PetscSectionGetChart(section, &pStart, &pEnd));
  /* VTK only wants the values at cells or vertices */
  CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
  CHKERRQ(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  pStart   = PetscMax(PetscMin(cStart, vStart), pStart);
  pEnd     = PetscMin(PetscMax(cEnd,   vEnd),   pEnd);
  CHKERRQ(DMGetStratumSize(dm, "vtk", 1, &numLabelCells));
  CHKERRQ(DMGetStratumSize(dm, "vtk", 2, &numLabelVertices));
  hasLabel = numLabelCells > 0 || numLabelVertices > 0 ? PETSC_TRUE : PETSC_FALSE;
  for (p = pStart; p < pEnd; ++p) {
    /* Reject points not either cells or vertices */
    if (((p < cStart) || (p >= cEnd)) && ((p < vStart) || (p >= vEnd))) continue;
    if (hasLabel) {
      PetscInt value;

      if (((p >= cStart) && (p < cEnd) && numLabelCells) ||
          ((p >= vStart) && (p < vEnd) && numLabelVertices)) {
        CHKERRQ(DMGetLabelValue(dm, "vtk", p, &value));
        if (value != 1) continue;
      }
    }
    CHKERRQ(PetscSectionGetDof(section, p, &numDof));
    if (numDof) break;
  }
  CHKERRMPI(MPIU_Allreduce(&numDof, &maxDof, 1, MPIU_INT, MPI_MAX, comm));
  enforceDof = PetscMax(enforceDof, maxDof);
  CHKERRQ(VecGetArray(v, &array));
  if (rank == 0) {
    PetscVTKReal dval;
    PetscScalar  val;
    char formatString[8];

    CHKERRQ(PetscSNPrintf(formatString, 8, "%%.%de", precision));
    for (p = pStart; p < pEnd; ++p) {
      /* Here we lose a way to filter points by keeping them out of the Numbering */
      PetscInt dof, off, goff, d;

      /* Reject points not either cells or vertices */
      if (((p < cStart) || (p >= cEnd)) && ((p < vStart) || (p >= vEnd))) continue;
      if (hasLabel) {
        PetscInt value;

        if (((p >= cStart) && (p < cEnd) && numLabelCells) ||
            ((p >= vStart) && (p < vEnd) && numLabelVertices)) {
          CHKERRQ(DMGetLabelValue(dm, "vtk", p, &value));
          if (value != 1) continue;
        }
      }
      CHKERRQ(PetscSectionGetDof(section, p, &dof));
      CHKERRQ(PetscSectionGetOffset(section, p, &off));
      CHKERRQ(PetscSectionGetOffset(globalSection, p, &goff));
      if (dof && goff >= 0) {
        for (d = 0; d < dof; d++) {
          if (d > 0) {
            CHKERRQ(PetscFPrintf(comm, fp, " "));
          }
          val = array[off+d];
          dval = (PetscVTKReal) ((imag ? PetscImaginaryPart(val) : PetscRealPart(val)) * scale);
          CHKERRQ(PetscFPrintf(comm, fp, formatString, dval));
        }
        for (d = dof; d < enforceDof; d++) {
          CHKERRQ(PetscFPrintf(comm, fp, " 0.0"));
        }
        CHKERRQ(PetscFPrintf(comm, fp, "\n"));
      }
    }
    for (proc = 1; proc < size; ++proc) {
      PetscScalar *remoteValues;
      PetscInt    size = 0, d;
      MPI_Status  status;

      CHKERRMPI(MPI_Recv(&size, 1, MPIU_INT, proc, tag, comm, &status));
      CHKERRQ(PetscMalloc1(size, &remoteValues));
      CHKERRMPI(MPI_Recv(remoteValues, size, mpiType, proc, tag, comm, &status));
      for (p = 0; p < size/maxDof; ++p) {
        for (d = 0; d < maxDof; ++d) {
          if (d > 0) {
            CHKERRQ(PetscFPrintf(comm, fp, " "));
          }
          val = remoteValues[p*maxDof+d];
          dval = (PetscVTKReal) ((imag ? PetscImaginaryPart(val) : PetscRealPart(val)) * scale);
          CHKERRQ(PetscFPrintf(comm, fp, formatString, dval));
        }
        for (d = maxDof; d < enforceDof; ++d) {
          CHKERRQ(PetscFPrintf(comm, fp, " 0.0"));
        }
        CHKERRQ(PetscFPrintf(comm, fp, "\n"));
      }
      CHKERRQ(PetscFree(remoteValues));
    }
  } else {
    PetscScalar *localValues;
    PetscInt    size, k = 0;

    CHKERRQ(PetscSectionGetStorageSize(section, &size));
    CHKERRQ(PetscMalloc1(size, &localValues));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, goff, d;

      /* Reject points not either cells or vertices */
      if (((p < cStart) || (p >= cEnd)) && ((p < vStart) || (p >= vEnd))) continue;
      if (hasLabel) {
        PetscInt value;

        if (((p >= cStart) && (p < cEnd) && numLabelCells) ||
            ((p >= vStart) && (p < vEnd) && numLabelVertices)) {
          CHKERRQ(DMGetLabelValue(dm, "vtk", p, &value));
          if (value != 1) continue;
        }
      }
      CHKERRQ(PetscSectionGetDof(section, p, &dof));
      CHKERRQ(PetscSectionGetOffset(section, p, &off));
      CHKERRQ(PetscSectionGetOffset(globalSection, p, &goff));
      if (goff >= 0) {
        for (d = 0; d < dof; ++d) {
          localValues[k++] = array[off+d];
        }
      }
    }
    CHKERRMPI(MPI_Send(&k, 1, MPIU_INT, 0, tag, comm));
    CHKERRMPI(MPI_Send(localValues, k, mpiType, 0, tag, comm));
    CHKERRQ(PetscFree(localValues));
  }
  CHKERRQ(VecRestoreArray(v, &array));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexVTKWriteField_ASCII(DM dm, PetscSection section, PetscSection globalSection, Vec field, const char name[], FILE *fp, PetscInt enforceDof, PetscInt precision, PetscReal scale, PetscBool nameComplex, PetscInt imag)
{
  MPI_Comm       comm;
  PetscInt       numDof = 0, maxDof;
  PetscInt       pStart, pEnd, p;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  CHKERRQ(PetscSectionGetChart(section, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    CHKERRQ(PetscSectionGetDof(section, p, &numDof));
    if (numDof) break;
  }
  numDof = PetscMax(numDof, enforceDof);
  CHKERRMPI(MPIU_Allreduce(&numDof, &maxDof, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)));
  if (!name) name = "Unknown";
  if (maxDof == 3) {
    if (nameComplex) {
      CHKERRQ(PetscFPrintf(comm, fp, "VECTORS %s.%s double\n", name, imag ? "Im" : "Re"));
    } else {
      CHKERRQ(PetscFPrintf(comm, fp, "VECTORS %s double\n", name));
    }
  } else {
    if (nameComplex) {
      CHKERRQ(PetscFPrintf(comm, fp, "SCALARS %s.%s double %D\n", name, imag ? "Im" : "Re", maxDof));
    } else {
      CHKERRQ(PetscFPrintf(comm, fp, "SCALARS %s double %D\n", name, maxDof));
    }
    CHKERRQ(PetscFPrintf(comm, fp, "LOOKUP_TABLE default\n"));
  }
  CHKERRQ(DMPlexVTKWriteSection_ASCII(dm, section, globalSection, field, fp, enforceDof, precision, scale, imag));
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
  CHKERRQ(DMGetCoordinatesLocalized(dm,&localized));
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCheckFalse(localized,comm,PETSC_ERR_SUP,"VTK output with localized coordinates not yet supported");
  CHKERRQ(PetscFOpen(comm, vtk->filename, "wb", &fp));
  CHKERRQ(PetscObjectGetName((PetscObject)dm, &dmname));
  CHKERRQ(PetscFPrintf(comm, fp, "# vtk DataFile Version 2.0\n"));
  CHKERRQ(PetscFPrintf(comm, fp, "%s\n", dmname));
  CHKERRQ(PetscFPrintf(comm, fp, "ASCII\n"));
  CHKERRQ(PetscFPrintf(comm, fp, "DATASET UNSTRUCTURED_GRID\n"));
  /* Vertices */
  {
    PetscSection  section, coordSection, globalCoordSection;
    Vec           coordinates;
    PetscReal     lengthScale;
    DMLabel       label;
    IS            vStratumIS;
    PetscLayout   vLayout;

    CHKERRQ(DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale));
    CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
    CHKERRQ(DMPlexGetDepthLabel(dm, &label));
    CHKERRQ(DMLabelGetStratumIS(label, 0, &vStratumIS));
    CHKERRQ(DMGetCoordinateSection(dm, &section));                                 /* This section includes all points */
    CHKERRQ(PetscSectionCreateSubmeshSection(section, vStratumIS, &coordSection)); /* This one includes just vertices */
    CHKERRQ(PetscSectionCreateGlobalSection(coordSection, dm->sf, PETSC_FALSE, PETSC_FALSE, &globalCoordSection));
    CHKERRQ(PetscSectionGetPointLayout(comm, globalCoordSection, &vLayout));
    CHKERRQ(PetscLayoutGetSize(vLayout, &totVertices));
    CHKERRQ(PetscFPrintf(comm, fp, "POINTS %D double\n", totVertices));
    CHKERRQ(DMPlexVTKWriteSection_ASCII(dm, coordSection, globalCoordSection, coordinates, fp, 3, PETSC_DETERMINE, lengthScale, 0));
    CHKERRQ(ISDestroy(&vStratumIS));
    CHKERRQ(PetscLayoutDestroy(&vLayout));
    CHKERRQ(PetscSectionDestroy(&coordSection));
    CHKERRQ(PetscSectionDestroy(&globalCoordSection));
  }
  /* Cells */
  CHKERRQ(DMPlexVTKWriteCells_ASCII(dm, fp, &totCells));
  /* Vertex fields */
  for (link = vtk->link; link; link = link->next) {
    if ((link->ft == PETSC_VTK_POINT_FIELD) || (link->ft == PETSC_VTK_POINT_VECTOR_FIELD)) hasPoint = PETSC_TRUE;
    if ((link->ft == PETSC_VTK_CELL_FIELD)  || (link->ft == PETSC_VTK_CELL_VECTOR_FIELD))  hasCell  = PETSC_TRUE;
  }
  if (hasPoint) {
    CHKERRQ(PetscFPrintf(comm, fp, "POINT_DATA %D\n", totVertices));
    for (link = vtk->link; link; link = link->next) {
      Vec          X = (Vec) link->vec;
      PetscSection section = NULL, globalSection, newSection = NULL;
      char         namebuf[256];
      const char   *name;
      PetscInt     enforceDof = PETSC_DETERMINE;

      if ((link->ft != PETSC_VTK_POINT_FIELD) && (link->ft != PETSC_VTK_POINT_VECTOR_FIELD)) continue;
      if (link->ft == PETSC_VTK_POINT_VECTOR_FIELD) enforceDof = 3;
      CHKERRQ(PetscObjectGetName(link->vec, &name));
      CHKERRQ(PetscObjectQuery(link->vec, "section", (PetscObject*) &section));
      if (!section) {
        DM           dmX;

        CHKERRQ(VecGetDM(X, &dmX));
        if (dmX) {
          DMLabel  subpointMap, subpointMapX;
          PetscInt dim, dimX, pStart, pEnd, qStart, qEnd;

          CHKERRQ(DMGetLocalSection(dmX, &section));
          /* Here is where we check whether dmX is a submesh of dm */
          CHKERRQ(DMGetDimension(dm,  &dim));
          CHKERRQ(DMGetDimension(dmX, &dimX));
          CHKERRQ(DMPlexGetChart(dm,  &pStart, &pEnd));
          CHKERRQ(DMPlexGetChart(dmX, &qStart, &qEnd));
          CHKERRQ(DMPlexGetSubpointMap(dm,  &subpointMap));
          CHKERRQ(DMPlexGetSubpointMap(dmX, &subpointMapX));
          if (((dim != dimX) || ((pEnd-pStart) < (qEnd-qStart))) && subpointMap && !subpointMapX) {
            const PetscInt *ind = NULL;
            IS              subpointIS;
            PetscInt        n = 0, q;

            CHKERRQ(PetscSectionGetChart(section, &qStart, &qEnd));
            CHKERRQ(DMPlexGetSubpointIS(dm, &subpointIS));
            if (subpointIS) {
              CHKERRQ(ISGetLocalSize(subpointIS, &n));
              CHKERRQ(ISGetIndices(subpointIS, &ind));
            }
            CHKERRQ(PetscSectionCreate(comm, &newSection));
            CHKERRQ(PetscSectionSetChart(newSection, pStart, pEnd));
            for (q = qStart; q < qEnd; ++q) {
              PetscInt dof, off, p;

              CHKERRQ(PetscSectionGetDof(section, q, &dof));
              if (dof) {
                CHKERRQ(PetscFindInt(q, n, ind, &p));
                if (p >= pStart) {
                  CHKERRQ(PetscSectionSetDof(newSection, p, dof));
                  CHKERRQ(PetscSectionGetOffset(section, q, &off));
                  CHKERRQ(PetscSectionSetOffset(newSection, p, off));
                }
              }
            }
            if (subpointIS) {
              CHKERRQ(ISRestoreIndices(subpointIS, &ind));
            }
            /* No need to setup section */
            section = newSection;
          }
        }
      }
      PetscCheckFalse(!section,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Vector %s had no PetscSection composed with it and could not create one from VecGetDM()", name);
      if (link->field >= 0) {
        const char *fieldname;

        CHKERRQ(PetscSectionGetFieldName(section, link->field, &fieldname));
        CHKERRQ(PetscSectionGetField(section, link->field, &section));
        if (fieldname) {
          CHKERRQ(PetscSNPrintf(namebuf, sizeof(namebuf), "%s%s", name, fieldname));
        } else {
          CHKERRQ(PetscSNPrintf(namebuf, sizeof(namebuf), "%s%D", name, link->field));
        }
      } else {
        CHKERRQ(PetscSNPrintf(namebuf, sizeof(namebuf), "%s", name));
      }
      CHKERRQ(PetscViewerVTKSanitizeName_Internal(namebuf, sizeof(namebuf)));
      CHKERRQ(PetscSectionCreateGlobalSection(section, dm->sf, PETSC_FALSE, PETSC_FALSE, &globalSection));
      for (l = 0; l < loops_per_scalar; l++) {
        CHKERRQ(DMPlexVTKWriteField_ASCII(dm, section, globalSection, X, namebuf, fp, enforceDof, PETSC_DETERMINE, 1.0, writeComplex, l));
      }
      CHKERRQ(PetscSectionDestroy(&globalSection));
      if (newSection) CHKERRQ(PetscSectionDestroy(&newSection));
    }
  }
  /* Cell Fields */
  CHKERRQ(PetscOptionsGetBool(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_view_partition", &writePartition, NULL));
  if (hasCell || writePartition) {
    CHKERRQ(PetscFPrintf(comm, fp, "CELL_DATA %D\n", totCells));
    for (link = vtk->link; link; link = link->next) {
      Vec          X = (Vec) link->vec;
      PetscSection section = NULL, globalSection;
      const char   *name = "";
      char         namebuf[256];
      PetscInt     enforceDof = PETSC_DETERMINE;

      if ((link->ft != PETSC_VTK_CELL_FIELD) && (link->ft != PETSC_VTK_CELL_VECTOR_FIELD)) continue;
      if (link->ft == PETSC_VTK_CELL_VECTOR_FIELD) enforceDof = 3;
      CHKERRQ(PetscObjectGetName(link->vec, &name));
      CHKERRQ(PetscObjectQuery(link->vec, "section", (PetscObject*) &section));
      if (!section) {
        DM           dmX;

        CHKERRQ(VecGetDM(X, &dmX));
        if (dmX) {
          CHKERRQ(DMGetLocalSection(dmX, &section));
        }
      }
      PetscCheckFalse(!section,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Vector %s had no PetscSection composed with it and could not create one from VecGetDM()", name);
      if (link->field >= 0) {
        const char *fieldname;

        CHKERRQ(PetscSectionGetFieldName(section, link->field, &fieldname));
        CHKERRQ(PetscSectionGetField(section, link->field, &section));
        if (fieldname) {
          CHKERRQ(PetscSNPrintf(namebuf, sizeof(namebuf), "%s%s", name, fieldname));
        } else {
          CHKERRQ(PetscSNPrintf(namebuf, sizeof(namebuf), "%s%D", name, link->field));
        }
      } else {
        CHKERRQ(PetscSNPrintf(namebuf, sizeof(namebuf), "%s", name));
      }
      CHKERRQ(PetscViewerVTKSanitizeName_Internal(namebuf, sizeof(namebuf)));
      CHKERRQ(PetscSectionCreateGlobalSection(section, dm->sf, PETSC_FALSE, PETSC_FALSE, &globalSection));
      for (l = 0; l < loops_per_scalar; l++) {
        CHKERRQ(DMPlexVTKWriteField_ASCII(dm, section, globalSection, X, namebuf, fp, enforceDof, PETSC_DETERMINE, 1.0, writeComplex, l));
      }
      CHKERRQ(PetscSectionDestroy(&globalSection));
    }
    if (writePartition) {
      CHKERRQ(PetscFPrintf(comm, fp, "SCALARS partition int 1\n"));
      CHKERRQ(PetscFPrintf(comm, fp, "LOOKUP_TABLE default\n"));
      CHKERRQ(DMPlexVTKWritePartition_ASCII(dm, fp));
    }
  }
  /* Cleanup */
  CHKERRQ(PetscFClose(comm, fp));
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
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK, &isvtk));
  PetscCheckFalse(!isvtk,PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_INCOMP, "Cannot use viewer type %s", ((PetscObject)viewer)->type_name);
  switch (viewer->format) {
  case PETSC_VIEWER_ASCII_VTK_DEPRECATED:
    CHKERRQ(DMPlexVTKWriteAll_ASCII(dm, viewer));
    break;
  case PETSC_VIEWER_VTK_VTU:
    CHKERRQ(DMPlexVTKWriteAll_VTU(dm, viewer));
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "No support for format '%s'", PetscViewerFormats[viewer->format]);
  }
  PetscFunctionReturn(0);
}
