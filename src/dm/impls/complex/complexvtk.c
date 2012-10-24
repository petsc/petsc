#define PETSCDM_DLL
#include <petsc-private/compleximpl.h>    /*I   "petscdmcomplex.h"   I*/
#include <../src/sys/viewer/impls/vtk/vtkvimpl.h>

#undef __FUNCT__
#define __FUNCT__ "DMComplexVTKCellTypeValid"
PetscErrorCode DMComplexVTKCellTypeValid(DM dm, PetscInt cellType, PetscBool *valid)
{
  PetscFunctionBegin;
  *valid = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexVTKGetCellType"
static PetscErrorCode DMComplexVTKGetCellType(DM dm, PetscInt dim, PetscInt corners, PetscInt *cellType) {
  PetscFunctionBegin;
  *cellType = -1;
  switch(dim) {
  case 0:
    switch(corners) {
    case 1:
      *cellType = 1; /* VTK_VERTEX */
      break;
    default:
      break;
    }
    break;
  case 1:
    switch(corners) {
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
    switch(corners) {
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
    switch(corners) {
    case 4:
      *cellType = 10; /* VTK_TETRA */
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

#undef __FUNCT__
#define __FUNCT__ "DMComplexVTKWriteCells"
PetscErrorCode DMComplexVTKWriteCells(DM dm, PetscSection globalConeSection, FILE *fp, PetscInt *totalCells)
{
  MPI_Comm       comm = ((PetscObject) dm)->comm;
  PetscInt       dim;
  PetscInt       numCorners = 0, maxCorners;
  PetscInt       numCells   = 0, totCells = 0, cellType, cellHeight;
  PetscInt       numLabelCells, cMax, cStart, cEnd, c, vStart, vEnd, v;
  PetscMPIInt    numProcs, rank, proc, tag = 1;
  PetscBool      hasLabel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMComplexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMComplexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMComplexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMComplexGetVTKBounds(dm, &cMax, PETSC_NULL);CHKERRQ(ierr);
  if (cMax >= 0) {cEnd = PetscMin(cEnd, cMax);}
  ierr = DMComplexGetStratumSize(dm, "vtk", 1, &numLabelCells);CHKERRQ(ierr);
  hasLabel = numLabelCells > 0 ? PETSC_TRUE : PETSC_FALSE;
  for (c = cStart; c < cEnd; ++c) {
    PetscInt  nC;
    PetscBool valid;

    if (hasLabel) {
      PetscInt value;

      ierr = DMComplexGetLabelValue(dm, "vtk", c, &value);CHKERRQ(ierr);
      if (value != 1) continue;
    }
    /* TODO: Does not work for interpolated meshes */
    ierr = PetscSectionGetDof(globalConeSection, c, &nC);CHKERRQ(ierr);
    ierr = DMComplexVTKGetCellType(dm, dim, nC, &cellType);CHKERRQ(ierr);
    ierr = DMComplexVTKCellTypeValid(dm, cellType, &valid);CHKERRQ(ierr);
    if (!valid) continue;
    if (!numCorners) {
      numCorners = nC;
    } else if (numCorners != nC) {
      SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %d has %d corners, which should be %d", c, nC, numCorners);
    }
    ++numCells;
  }
  ierr = MPI_Reduce(&numCells, &totCells, 1, MPIU_INT, MPI_SUM, 0, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&numCorners, &maxCorners, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
  if (numCorners && numCorners != maxCorners) SETERRQ2(comm, PETSC_ERR_ARG_WRONG, "All elements must have %d corners, not %d", maxCorners, numCorners);
  ierr = PetscFPrintf(comm, fp, "CELLS %d %d\n", totCells, totCells*(maxCorners+1));CHKERRQ(ierr);
  if (!rank) {
    for (c = cStart; c < cEnd; ++c) {
      PetscInt *closure = PETSC_NULL;
      PetscInt  coneSize, closureSize;
      PetscBool valid;

      if (hasLabel) {
        PetscInt value;

        ierr = DMComplexGetLabelValue(dm, "vtk", c, &value);CHKERRQ(ierr);
        if (value != 1) continue;
      }
      ierr = PetscSectionGetDof(globalConeSection, c, &coneSize);CHKERRQ(ierr);
      ierr = DMComplexVTKGetCellType(dm, dim, coneSize, &cellType);CHKERRQ(ierr);
      ierr = DMComplexVTKCellTypeValid(dm, cellType, &valid);CHKERRQ(ierr);
      if (!valid) continue;
      ierr = PetscFPrintf(comm, fp, "%d ", maxCorners);CHKERRQ(ierr);
      ierr = DMComplexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      /* TODO Need global vertex numbering */
      for (v = 0; v < closureSize*2; v += 2) {
        const PetscInt vertex = closure[v];

        if ((vertex < vStart) || (vertex >= vEnd)) continue;
        ierr = PetscFPrintf(comm, fp, " %d", vertex - vStart);CHKERRQ(ierr);
      }
      ierr = PetscFPrintf(comm, fp, "\n");CHKERRQ(ierr);
      ierr = DMComplexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    for (proc = 1; proc < numProcs; ++proc) {
      PetscInt  *remoteVertices;
      MPI_Status status;

      ierr = MPI_Recv(&numCells, 1, MPIU_INT, proc, tag, comm, &status);CHKERRQ(ierr);
      ierr = PetscMalloc(numCells*maxCorners * sizeof(PetscInt), &remoteVertices);CHKERRQ(ierr);
      ierr = MPI_Recv(remoteVertices, numCells*maxCorners, MPIU_INT, proc, tag, comm, &status);CHKERRQ(ierr);
      for (c = 0; c < numCells; ++c) {
        ierr = PetscFPrintf(comm, fp, "%d ", maxCorners);CHKERRQ(ierr);
        for (v = 0; v < maxCorners; ++v) {
          ierr = PetscFPrintf(comm, fp, " %d", remoteVertices[c*maxCorners+v]);CHKERRQ(ierr);
        }
        ierr = PetscFPrintf(comm, fp, "\n");CHKERRQ(ierr);
      }
      ierr = PetscFree(remoteVertices);CHKERRQ(ierr);
    }
  } else {
    PetscInt *localVertices, k = 0;

    ierr = PetscMalloc(numCells*maxCorners * sizeof(PetscInt), &localVertices);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      PetscInt *closure = PETSC_NULL;
      PetscInt  coneSize, closureSize;
      PetscBool valid;

      if (hasLabel) {
        PetscInt value;

        ierr = DMComplexGetLabelValue(dm, "vtk", c, &value);CHKERRQ(ierr);
        if (value != 1) continue;
      }
      /* TODO Use closure for interpolated meshes */
      ierr = PetscSectionGetDof(globalConeSection, c, &coneSize);CHKERRQ(ierr);
      ierr = DMComplexVTKGetCellType(dm, dim, coneSize, &cellType);CHKERRQ(ierr);
      ierr = DMComplexVTKCellTypeValid(dm, cellType, &valid);CHKERRQ(ierr);
      if (!valid) continue;
      ierr = DMComplexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      /* TODO Need global vertex numbering */
      for (v = 0; v < closureSize*2; v += 2) {
        const PetscInt vertex = closure[v];

        if ((vertex < vStart) || (vertex >= vEnd)) continue;
        localVertices[k++] = vertex - vStart;
      }
      ierr = DMComplexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    if (k != numCells*maxCorners) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of vertices to send %d should be %d", k, numCells*maxCorners);
    ierr = MPI_Send(&numCells, 1, MPIU_INT, 0, tag, comm);CHKERRQ(ierr);
    ierr = MPI_Send(localVertices, numCells*maxCorners, MPI_INT, 0, tag, comm);CHKERRQ(ierr);
    ierr = PetscFree(localVertices);CHKERRQ(ierr);
  }
  ierr = PetscFPrintf(comm, fp, "CELL_TYPES %d\n", totCells);CHKERRQ(ierr);
  ierr = DMComplexVTKGetCellType(dm, dim, maxCorners, &cellType);CHKERRQ(ierr);
  for (c = 0; c < totCells; ++c) {
    ierr = PetscFPrintf(comm, fp, "%d\n", cellType);CHKERRQ(ierr);
  }
  *totalCells = totCells;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexVTKWriteSection"
PetscErrorCode DMComplexVTKWriteSection(DM dm, PetscSection section, PetscSection globalSection, Vec v, FILE *fp, PetscInt enforceDof, PetscInt precision, PetscReal scale) {
  MPI_Comm           comm    = ((PetscObject) dm)->comm;
  const MPI_Datatype mpiType = MPIU_SCALAR;
  PetscScalar       *array;
  PetscInt           numDof = 0, maxDof;
  PetscInt           numLabelCells, cellHeight, cMax, cStart, cEnd, numLabelVertices, vMax, vStart, vEnd, pStart, pEnd, p;
  PetscMPIInt        numProcs, rank, proc, tag = 1;
  PetscBool          hasLabel;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (precision < 0) precision = 6;
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  /* VTK only wants the values at cells or vertices */
  ierr = DMComplexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMComplexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMComplexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMComplexGetVTKBounds(dm, &cMax, &vMax);CHKERRQ(ierr);
  if (cMax >= 0) {cEnd = PetscMin(cEnd, cMax);}
  if (vMax >= 0) {vEnd = PetscMin(vEnd, vMax);}
  pStart = PetscMax(PetscMin(cStart, vStart), pStart);
  pEnd   = PetscMin(PetscMax(cEnd,   vEnd),   pEnd);
  ierr = DMComplexGetStratumSize(dm, "vtk", 1, &numLabelCells);CHKERRQ(ierr);
  ierr = DMComplexGetStratumSize(dm, "vtk", 2, &numLabelVertices);CHKERRQ(ierr);
  hasLabel = numLabelCells > 0 || numLabelVertices > 0 ? PETSC_TRUE : PETSC_FALSE;
  for(p = pStart; p < pEnd; ++p) {
    /* Reject points not either cells or vertices */
    if (((p < cStart) || (p >= cEnd)) && ((p < vStart) || (p >= vEnd))) continue;
    if (hasLabel) {
      PetscInt value;

      if (((p >= cStart) && (p < cEnd) && numLabelCells) ||
          ((p >= vStart) && (p < vEnd) && numLabelVertices)) {
        ierr = DMComplexGetLabelValue(dm, "vtk", p, &value);CHKERRQ(ierr);
        if (value != 1) continue;
      }
    }
    ierr = PetscSectionGetDof(section, p, &numDof);CHKERRQ(ierr);
    if (numDof) {break;}
  }
  ierr = MPI_Allreduce(&numDof, &maxDof, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
  enforceDof = PetscMax(enforceDof, maxDof);
  ierr = VecGetArray(v, &array);CHKERRQ(ierr);
  if (!rank) {
    char formatString[8];

    ierr = PetscSNPrintf(formatString, 8, "%%.%de", precision);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      /* Here we lose a way to filter points by keeping them out of the Numbering */
      PetscInt dof, off, goff, d;

      /* Reject points not either cells or vertices */
      if (((p < cStart) || (p >= cEnd)) && ((p < vStart) || (p >= vEnd))) continue;
      if (hasLabel) {
        PetscInt value;

        if (((p >= cStart) && (p < cEnd) && numLabelCells) ||
            ((p >= vStart) && (p < vEnd) && numLabelVertices)) {
          ierr = DMComplexGetLabelValue(dm, "vtk", p, &value);CHKERRQ(ierr);
          if (value != 1) continue;
        }
      }
      ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(globalSection, p, &goff);CHKERRQ(ierr);
      if (dof && goff >= 0) {
        for (d = 0; d < dof; d++) {
          if (d > 0) {
            ierr = PetscFPrintf(comm, fp, " ");CHKERRQ(ierr);
          }
          ierr = PetscFPrintf(comm, fp, formatString, PetscRealPart(array[off+d])*scale);CHKERRQ(ierr);
        }
        for (d = dof; d < enforceDof; d++) {
          ierr = PetscFPrintf(comm, fp, " 0.0");CHKERRQ(ierr);
        }
        ierr = PetscFPrintf(comm, fp, "\n");CHKERRQ(ierr);
      }
    }
    for (proc = 1; proc < numProcs; ++proc) {
      PetscScalar *remoteValues;
      PetscInt     size, d;
      MPI_Status   status;

      ierr = MPI_Recv(&size, 1, MPIU_INT, proc, tag, comm, &status);CHKERRQ(ierr);
      ierr = PetscMalloc(size * sizeof(PetscScalar), &remoteValues);CHKERRQ(ierr);
      ierr = MPI_Recv(remoteValues, size, mpiType, proc, tag, comm, &status);CHKERRQ(ierr);
      for (p = 0; p < size/maxDof; ++p) {
        for (d = 0; d < maxDof; ++d) {
          if (d > 0) {
            ierr = PetscFPrintf(comm, fp, " ");CHKERRQ(ierr);
          }
          ierr = PetscFPrintf(comm, fp, formatString, PetscRealPart(remoteValues[p*maxDof+d])*scale);CHKERRQ(ierr);
        }
        for (d = maxDof; d < enforceDof; ++d) {
          ierr = PetscFPrintf(comm, fp, " 0.0");CHKERRQ(ierr);
        }
        ierr = PetscFPrintf(comm, fp, "\n");CHKERRQ(ierr);
      }
      ierr = PetscFree(remoteValues);CHKERRQ(ierr);
    }
  } else {
    PetscScalar *localValues;
    PetscInt     size, k = 0;

    ierr = PetscSectionGetStorageSize(section, &size);CHKERRQ(ierr);
    ierr = PetscMalloc(size * sizeof(PetscScalar), &localValues);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, goff, d;

      /* Reject points not either cells or vertices */
      if (((p < cStart) || (p >= cEnd)) && ((p < vStart) || (p >= vEnd))) continue;
      if (hasLabel) {
        PetscInt value;

        if (((p >= cStart) && (p < cEnd) && numLabelCells) ||
            ((p >= vStart) && (p < vEnd) && numLabelVertices)) {
          ierr = DMComplexGetLabelValue(dm, "vtk", p, &value);CHKERRQ(ierr);
          if (value != 1) continue;
        }
      }
      ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(globalSection, p, &goff);CHKERRQ(ierr);
      if (goff >= 0) {
        for (d = 0; d < dof; ++d) {
          localValues[k++] = array[off+d];
        }
      }
    }
    ierr = MPI_Send(&k, 1, MPIU_INT, 0, tag, comm);CHKERRQ(ierr);
    ierr = MPI_Send(localValues, k, mpiType, 0, tag, comm);CHKERRQ(ierr);
    ierr = PetscFree(localValues);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(v, &array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexVTKWriteField"
PetscErrorCode DMComplexVTKWriteField(DM dm, PetscSection section, PetscSection globalSection, Vec field, const char name[], FILE *fp, PetscInt enforceDof, PetscInt precision, PetscReal scale)
{
  MPI_Comm       comm = ((PetscObject) dm)->comm;
  PetscInt       numDof = 0, maxDof;
  PetscInt       pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    ierr = PetscSectionGetDof(section, p, &numDof);CHKERRQ(ierr);
    if (numDof) {break;}
  }
  numDof = PetscMax(numDof, enforceDof);
  ierr = MPI_Allreduce(&numDof, &maxDof, 1, MPIU_INT, MPI_MAX, ((PetscObject) dm)->comm);CHKERRQ(ierr);
  if (!name) name = "Unknown";
  if (maxDof == 3) {
    ierr = PetscFPrintf(comm, fp, "VECTORS %s double\n", name);CHKERRQ(ierr);
  } else {
    ierr = PetscFPrintf(comm, fp, "SCALARS %s double %d\n", name, maxDof);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm, fp, "LOOKUP_TABLE default\n");CHKERRQ(ierr);
  }
  ierr = DMComplexVTKWriteSection(dm, section, globalSection, field, fp, enforceDof, precision, scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexVTKWriteAll_ASCII"
static PetscErrorCode DMComplexVTKWriteAll_ASCII(DM dm, PetscViewer viewer)
{
  MPI_Comm                 comm = ((PetscObject) dm)->comm;
  PetscViewer_VTK         *vtk  = (PetscViewer_VTK *) viewer->data;
  FILE                    *fp;
  PetscViewerVTKObjectLink link;
  PetscSection             coordSection, globalCoordSection;
  PetscSection             coneSection, globalConeSection;
  PetscLayout              vLayout;
  Vec                      coordinates;
  PetscReal                lengthScale;
  PetscInt                 vMax, totVertices, totCells;
  PetscBool                hasPoint = PETSC_FALSE, hasCell = PETSC_FALSE;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = PetscFOpen(comm, vtk->filename, "wb", &fp);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fp, "# vtk DataFile Version 2.0\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fp, "Simplicial Mesh Example\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fp, "ASCII\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fp, "DATASET UNSTRUCTURED_GRID\n");CHKERRQ(ierr);
  /* Vertices */
  ierr = DMComplexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale);CHKERRQ(ierr);
  ierr = DMComplexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionCreateGlobalSection(coordSection, dm->sf, PETSC_FALSE, &globalCoordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMComplexGetVTKBounds(dm, PETSC_NULL, &vMax);CHKERRQ(ierr);
  if (vMax >= 0) {
    PetscInt pStart, pEnd, p, localSize = 0;

    ierr = PetscSectionGetChart(globalCoordSection, &pStart, &pEnd);CHKERRQ(ierr);
    pEnd = PetscMin(pEnd, vMax);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof;

      ierr = PetscSectionGetDof(globalCoordSection, p, &dof);CHKERRQ(ierr);
      if (dof > 0) {++localSize;}
    }
    ierr = PetscLayoutCreate(((PetscObject) dm)->comm, &vLayout);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(vLayout, localSize);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(vLayout, 1);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(vLayout);CHKERRQ(ierr);
  } else {
    ierr = PetscSectionGetPointLayout(((PetscObject) dm)->comm, globalCoordSection, &vLayout);CHKERRQ(ierr);
  }
  ierr = PetscLayoutGetSize(vLayout, &totVertices);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fp, "POINTS %d double\n", totVertices);CHKERRQ(ierr);
  ierr = DMComplexVTKWriteSection(dm, coordSection, globalCoordSection, coordinates, fp, 3, PETSC_DETERMINE, lengthScale);CHKERRQ(ierr);
  /* Cells */
  ierr = DMComplexGetConeSection(dm, &coneSection);CHKERRQ(ierr);
  ierr = PetscSectionCreateGlobalSection(coneSection, dm->sf, PETSC_FALSE, &globalConeSection);CHKERRQ(ierr);
  ierr = DMComplexVTKWriteCells(dm, globalConeSection, fp, &totCells);CHKERRQ(ierr);
  /* Vertex fields */
  for (link = vtk->link; link; link = link->next) {
    if ((link->ft == PETSC_VTK_POINT_FIELD) || (link->ft == PETSC_VTK_POINT_VECTOR_FIELD)) hasPoint = PETSC_TRUE;
    if ((link->ft == PETSC_VTK_CELL_FIELD)  || (link->ft == PETSC_VTK_CELL_VECTOR_FIELD))  hasCell  = PETSC_TRUE;
  }
  if (hasPoint) {
    ierr = PetscFPrintf(comm, fp, "POINT_DATA %d\n", totVertices);
    for (link = vtk->link; link; link = link->next) {
      Vec            X = (Vec) link->vec;
      PetscContainer c;
      PetscSection   section, globalSection;
      const char    *name;
      PetscInt       enforceDof = PETSC_DETERMINE;

      if ((link->ft != PETSC_VTK_POINT_FIELD) && (link->ft != PETSC_VTK_POINT_VECTOR_FIELD)) continue;
      if (link->ft == PETSC_VTK_POINT_VECTOR_FIELD) enforceDof = 3;
      ierr = PetscObjectGetName(link->vec, &name);CHKERRQ(ierr);
      ierr = PetscObjectQuery(link->vec, "section", (PetscObject *) &c);CHKERRQ(ierr);
      if (!c) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "Vector %s had no PetscSection composed with it", name);
      ierr = PetscContainerGetPointer(c, (void **) &section);CHKERRQ(ierr);
      if (!section) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "Vector %s had no PetscSection composed with it", name);
      ierr = PetscSectionCreateGlobalSection(section, dm->sf, PETSC_FALSE, &globalSection);CHKERRQ(ierr);
      ierr = DMComplexVTKWriteField(dm, section, globalSection, X, name, fp, enforceDof, PETSC_DETERMINE, 1.0);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&globalSection);CHKERRQ(ierr);
    }
  }
  /* Cell Fields */
  if (hasCell) {
    ierr = PetscFPrintf(comm, fp, "CELL_DATA %d\n", totCells);
    for (link = vtk->link; link; link = link->next) {
      Vec            X = (Vec) link->vec;
      PetscContainer c;
      PetscSection   section, globalSection;
      const char    *name;
      PetscInt       enforceDof = PETSC_DETERMINE;

      if ((link->ft != PETSC_VTK_CELL_FIELD) && (link->ft != PETSC_VTK_CELL_VECTOR_FIELD)) continue;
      if (link->ft == PETSC_VTK_CELL_VECTOR_FIELD) enforceDof = 3;
      ierr = PetscObjectGetName(link->vec, &name);CHKERRQ(ierr);
      ierr = PetscObjectQuery(link->vec, "section", (PetscObject *) &c);CHKERRQ(ierr);
      if (!c) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "Vector %s had no PetscSection composed with it", name);
      ierr = PetscContainerGetPointer(c, (void **) &section);CHKERRQ(ierr);
      if (!section) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "Vector %s had no PetscSection composed with it", name);
      ierr = PetscSectionCreateGlobalSection(section, dm->sf, PETSC_FALSE, &globalSection);CHKERRQ(ierr);
      ierr = DMComplexVTKWriteField(dm, section, globalSection, X, name, fp, enforceDof, PETSC_DETERMINE, 1.0);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&globalSection);CHKERRQ(ierr);
    }
  }
  /* Cleanup */
  ierr = PetscSectionDestroy(&globalCoordSection);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&vLayout);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&globalConeSection);CHKERRQ(ierr);
  ierr = PetscFClose(comm, fp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexVTKWriteAll"
/*@C
  DMComplexVTKWriteAll - Write a file containing all the fields that have been provided to the viewer

  Collective

  Input Arguments:
+ odm - The DMComplex specifying the mesh, passed as a PetscObject
- viewer - viewer of type VTK

  Level: developer

  Note:
  This function is a callback used by the VTK viewer to actually write the file.
  The reason for this odd model is that the VTK file format does not provide any way to write one field at a time.
  Instead, metadata for the entire file needs to be available up-front before you can start writing the file.

.seealso: PETSCVIEWERVTK
@*/
PetscErrorCode DMComplexVTKWriteAll(PetscObject odm, PetscViewer viewer)
{
  DM              dm   = (DM) odm;
  PetscBool       isvtk;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK, &isvtk);CHKERRQ(ierr);
  if (!isvtk) SETERRQ1(((PetscObject) viewer)->comm, PETSC_ERR_ARG_INCOMP, "Cannot use viewer type %s", ((PetscObject)viewer)->type_name);
  switch (viewer->format) {
  case PETSC_VIEWER_ASCII_VTK:
    ierr = DMComplexVTKWriteAll_ASCII(dm, viewer);CHKERRQ(ierr);
    break;
  default: SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_SUP, "No support for format '%s'", PetscViewerFormats[viewer->format]);
  }
  PetscFunctionReturn(0);
}
