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
  PetscInt       numCells   = 0, totCells, cellType;
  PetscInt       cStart, cEnd, c, vStart, vEnd, v;
  PetscMPIInt    numProcs, rank, proc, tag = 1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMComplexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMComplexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  for(c = cStart; c < cEnd; ++c) {
    PetscInt  nC;
    PetscBool valid;

    /* TODO: Does not work for interpolated meshes */
    ierr = PetscSectionGetDof(globalConeSection, c, &nC);CHKERRQ(ierr);
    ierr = DMComplexVTKGetCellType(dm, dim, nC, &cellType);CHKERRQ(ierr);
    ierr = DMComplexVTKCellTypeValid(dm, cellType, &valid);CHKERRQ(ierr);
    if (!valid) continue;
    if (!numCorners) numCorners = nC;
    ++numCells;
  }
  ierr = MPI_Reduce(&numCells, &totCells, 1, MPIU_INT, MPI_SUM, 0, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&numCorners, &maxCorners, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
  if (numCorners && numCorners != maxCorners) SETERRQ2(comm, PETSC_ERR_ARG_WRONG, "All elements must have %d corners, not %d", maxCorners, numCorners);
  ierr = PetscFPrintf(comm, fp, "CELLS %d %d\n", totCells, totCells*(maxCorners+1));CHKERRQ(ierr);
  if (!rank) {
    for(c = cStart; c < cEnd; ++c) {
      const PetscInt *cone;
      PetscInt        coneSize;
      PetscBool       valid;

      /* TODO Use closure for interpolated meshes */
      ierr = PetscSectionGetDof(globalConeSection, c, &coneSize);CHKERRQ(ierr);
      ierr = DMComplexVTKGetCellType(dm, dim, coneSize, &cellType);CHKERRQ(ierr);
      ierr = DMComplexVTKCellTypeValid(dm, cellType, &valid);CHKERRQ(ierr);
      if (!valid) continue;
      ierr = PetscFPrintf(comm, fp, "%d ", maxCorners);CHKERRQ(ierr);
      ierr = DMComplexGetCone(dm, c, &cone);CHKERRQ(ierr);
      /* TODO Need global vertex numbering */
      for(v = 0; v < coneSize; ++v) {
        ierr = PetscFPrintf(comm, fp, " %d", cone[v] - vStart);CHKERRQ(ierr);
      }
      ierr = PetscFPrintf(comm, fp, "\n");CHKERRQ(ierr);
    }
    for(proc = 1; proc < numProcs; ++proc) {
      PetscInt  *remoteVertices;
      MPI_Status status;

      ierr = MPI_Recv(&numCells, 1, MPIU_INT, proc, tag, comm, &status);CHKERRQ(ierr);
      ierr = PetscMalloc(numCells*maxCorners * sizeof(PetscInt), &remoteVertices);CHKERRQ(ierr);
      ierr = MPI_Recv(remoteVertices, numCells*maxCorners, MPIU_INT, proc, tag, comm, &status);CHKERRQ(ierr);
      for(c = 0; c < numCells; ++c) {
        ierr = PetscFPrintf(comm, fp, "%d ", maxCorners);CHKERRQ(ierr);
        for(v = 0; v < maxCorners; ++v) {
          ierr = PetscFPrintf(comm, fp, " %d", remoteVertices[c*maxCorners+v]);CHKERRQ(ierr);
        }
        ierr = PetscFPrintf(comm, fp, "\n");CHKERRQ(ierr);
      }
      ierr = PetscFree(remoteVertices);CHKERRQ(ierr);
    }
  } else {
    PetscInt *localVertices, k = 0;

    ierr = PetscMalloc(numCells*maxCorners * sizeof(PetscInt), &localVertices);CHKERRQ(ierr);
    for(c = cStart; c < cEnd; ++c) {
      const PetscInt *cone;
      PetscInt        coneSize;
      PetscBool       valid;

      /* TODO Use closure for interpolated meshes */
      ierr = PetscSectionGetDof(globalConeSection, c, &coneSize);CHKERRQ(ierr);
      ierr = DMComplexVTKGetCellType(dm, dim, coneSize, &cellType);CHKERRQ(ierr);
      ierr = DMComplexVTKCellTypeValid(dm, cellType, &valid);CHKERRQ(ierr);
      if (!valid) continue;
      ierr = DMComplexGetCone(dm, c, &cone);CHKERRQ(ierr);
      /* TODO Need global vertex numbering */
      for(v = 0; v < coneSize; ++v) {
        localVertices[k++] = cone[v];
      }
    }
    if (k != numCells*maxCorners) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of vertices to send %d should be %d", k, numCells*maxCorners);
    ierr = MPI_Send(&numCells, 1, MPIU_INT, 0, tag, comm);CHKERRQ(ierr);
    ierr = MPI_Send(localVertices, numCells*maxCorners, MPI_INT, 0, tag, comm);CHKERRQ(ierr);
    ierr = PetscFree(localVertices);CHKERRQ(ierr);
  }
  ierr = PetscFPrintf(comm, fp, "CELL_TYPES %d\n", totCells);CHKERRQ(ierr);
  ierr = DMComplexVTKGetCellType(dm, dim, maxCorners, &cellType);CHKERRQ(ierr);
  for(c = 0; c < totCells; ++c) {
    ierr = PetscFPrintf(comm, fp, "%d\n", cellType);CHKERRQ(ierr);
  }
  *totalCells = totCells;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexVTKWriteSection"
PetscErrorCode DMComplexVTKWriteSection(DM dm, PetscSection section, Vec v, FILE *fp, PetscInt enforceDof, PetscInt precision) {
  MPI_Comm           comm    = ((PetscObject) v)->comm;
  const MPI_Datatype mpiType = MPIU_SCALAR;
  PetscScalar       *array;
  PetscInt           numDof = 0, maxDof;
  PetscInt           pStart, pEnd, p;
  PetscMPIInt        numProcs, rank, proc, tag = 1;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (precision < 0) precision = 6;
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
    ierr = PetscSectionGetDof(section, p, &numDof);CHKERRQ(ierr);
    if (numDof) {break;}
  }
  ierr = MPI_Allreduce(&numDof, &maxDof, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
  enforceDof = PetscMax(enforceDof, maxDof);
  ierr = VecGetArray(v, &array);CHKERRQ(ierr);
  if (!rank) {
    char formatString[8];

    ierr = PetscSNPrintf(formatString, 8, "%%.%de", precision);CHKERRQ(ierr);
    for(p = pStart; p < pEnd; ++p) {
      /* Here we lose a way to filter points by keeping them out of the Numbering */
      PetscInt dof, off, d;

      ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
      if (dof && off >= 0) {
        for(d = 0; d < dof; d++) {
          if (d > 0) {
            ierr = PetscFPrintf(comm, fp, " ");CHKERRQ(ierr);
          }
          ierr = PetscFPrintf(comm, fp, formatString, PetscRealPart(array[off+d]));CHKERRQ(ierr);
        }
        for(d = dof; d < enforceDof; d++) {
          ierr = PetscFPrintf(comm, fp, " 0.0");CHKERRQ(ierr);
        }
        ierr = PetscFPrintf(comm, fp, "\n");CHKERRQ(ierr);
      }
    }
    for(proc = 1; proc < numProcs; ++proc) {
      PetscScalar *remoteValues;
      PetscInt     size, d;
      MPI_Status   status;

      ierr = MPI_Recv(&size, 1, MPIU_INT, proc, tag, comm, &status);CHKERRQ(ierr);
      ierr = PetscMalloc(size * sizeof(PetscScalar), &remoteValues);CHKERRQ(ierr);
      ierr = MPI_Recv(remoteValues, size, mpiType, proc, tag, comm, &status);CHKERRQ(ierr);
      for(p = 0; p < size/maxDof; ++p) {
        for(d = 0; d < maxDof; ++d) {
          if (d > 0) {
            ierr = PetscFPrintf(comm, fp, " ");CHKERRQ(ierr);
          }
          ierr = PetscFPrintf(comm, fp, formatString, PetscRealPart(remoteValues[p*maxDof+d]));CHKERRQ(ierr);
        }
        for(d = maxDof; d < enforceDof; ++d) {
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
    for(p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, d;

      ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
      if (off >= 0) {
        for(d = 0; d < dof; ++d) {
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
PetscErrorCode DMComplexVTKWriteField(DM dm, PetscSection section, Vec field, const char name[], FILE *fp, PetscInt enforceDof, PetscInt precision)
{
  MPI_Comm       comm = ((PetscObject) dm)->comm;
  PetscInt       numDof = 0, maxDof;
  PetscInt       pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
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
  ierr = DMComplexVTKWriteSection(dm, section, field, fp, enforceDof, precision);CHKERRQ(ierr);
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
  PetscInt                 totVertices, totCells;
  PetscBool                hasPoint = PETSC_FALSE, hasCell = PETSC_FALSE;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = PetscFOpen(comm, vtk->filename, "wb", &fp);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fp, "# vtk DataFile Version 2.0\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fp, "Simplicial Mesh Example\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fp, "ASCII\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fp, "DATASET UNSTRUCTURED_GRID\n");CHKERRQ(ierr);
  /* Vertices */
  /* TODO: Need to check for "coordinates_dimensioned" */
  ierr = DMComplexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionCreateGlobalSection(coordSection, dm->sf, &globalCoordSection);CHKERRQ(ierr);
  ierr = DMComplexGetCoordinateVec(dm, &coordinates);CHKERRQ(ierr);
  ierr = PetscSectionGetPointLayout(((PetscObject) dm)->comm, globalCoordSection, &vLayout);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(vLayout, &totVertices);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fp, "POINTS %d double\n", totVertices);CHKERRQ(ierr);
  ierr = DMComplexVTKWriteSection(dm, globalCoordSection, coordinates, fp, 3, PETSC_DETERMINE);CHKERRQ(ierr);
  /* Cells */
  ierr = DMComplexGetConeSection(dm, &coneSection);CHKERRQ(ierr);
  ierr = PetscSectionCreateGlobalSection(coneSection, dm->sf, &globalConeSection);CHKERRQ(ierr);
  ierr = DMComplexVTKWriteCells(dm, globalConeSection, fp, &totCells);CHKERRQ(ierr);
  /* Vertex fields */
  for(link = vtk->link; link; link = link->next) {
    if ((link->ft == PETSC_VTK_POINT_FIELD) || (link->ft == PETSC_VTK_POINT_VECTOR_FIELD)) hasPoint = PETSC_TRUE;
    if ((link->ft == PETSC_VTK_CELL_FIELD)  || (link->ft == PETSC_VTK_CELL_VECTOR_FIELD))  hasCell  = PETSC_TRUE;
  }
  if (hasPoint) {
    ierr = PetscFPrintf(comm, fp, "POINT_DATA %d\n", totVertices);
    for(link = vtk->link; link; link = link->next) {
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
      ierr = PetscSectionCreateGlobalSection(section, dm->sf, &globalSection);CHKERRQ(ierr);
      ierr = DMComplexVTKWriteField(dm, globalSection, X, name, fp, enforceDof, PETSC_DETERMINE);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&globalSection);CHKERRQ(ierr);
    }
  }
  /* Cell Fields */
  if (hasCell) {
    ierr = PetscFPrintf(comm, fp, "CELL_DATA %d\n", totCells);
    for(link = vtk->link; link; link = link->next) {
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
      ierr = PetscSectionCreateGlobalSection(section, dm->sf, &globalSection);CHKERRQ(ierr);
      ierr = DMComplexVTKWriteField(dm, globalSection, X, name, fp, enforceDof, PETSC_DETERMINE);CHKERRQ(ierr);
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
