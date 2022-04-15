#define PETSC_DESIRE_FEATURE_TEST_MACROS /* for fileno() */
#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

/*@C
  DMPlexCreateFluentFromFile - Create a DMPlex mesh from a Fluent mesh file

+ comm        - The MPI communicator
. filename    - Name of the Fluent mesh file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Level: beginner

.seealso: DMPlexCreateFromFile(), DMPlexCreateFluent(), DMPlexCreate()
@*/
PetscErrorCode DMPlexCreateFluentFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscViewer     viewer;

  PetscFunctionBegin;
  /* Create file viewer and build plex */
  PetscCall(PetscViewerCreate(comm, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERASCII));
  PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_READ));
  PetscCall(PetscViewerFileSetName(viewer, filename));
  PetscCall(DMPlexCreateFluent(comm, viewer, interpolate, dm));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateFluent_ReadString(PetscViewer viewer, char *buffer, char delim)
{
  PetscInt ret, i = 0;

  PetscFunctionBegin;
  do PetscCall(PetscViewerRead(viewer, &(buffer[i++]), 1, &ret, PETSC_CHAR));
  while (ret > 0 && buffer[i-1] != '\0' && buffer[i-1] != delim);
  if (!ret) buffer[i-1] = '\0'; else buffer[i] = '\0';
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateFluent_ReadValues(PetscViewer viewer, void *data, PetscInt count, PetscDataType dtype, PetscBool binary)
{
  int            fdes=0;
  FILE          *file;
  PetscInt       i;

  PetscFunctionBegin;
  if (binary) {
    /* Extract raw file descriptor to read binary block */
    PetscCall(PetscViewerASCIIGetPointer(viewer, &file));
    fflush(file); fdes = fileno(file);
  }

  if (!binary && dtype == PETSC_INT) {
    char         cbuf[256];
    unsigned int ibuf;
    int          snum;
    /* Parse hexadecimal ascii integers */
    for (i = 0; i < count; i++) {
      PetscCall(PetscViewerRead(viewer, cbuf, 1, NULL, PETSC_STRING));
      snum = sscanf(cbuf, "%x", &ibuf);
      PetscCheck(snum == 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
      ((PetscInt*)data)[i] = (PetscInt)ibuf;
    }
  } else if (binary && dtype == PETSC_INT) {
    /* Always read 32-bit ints and cast to PetscInt */
    int *ibuf;
    PetscCall(PetscMalloc1(count, &ibuf));
    PetscCall(PetscBinaryRead(fdes, ibuf, count, NULL, PETSC_ENUM));
    PetscCall(PetscByteSwap(ibuf, PETSC_ENUM, count));
    for (i = 0; i < count; i++) ((PetscInt*)data)[i] = (PetscInt)(ibuf[i]);
    PetscCall(PetscFree(ibuf));

 } else if (binary && dtype == PETSC_SCALAR) {
    float *fbuf;
    /* Always read 32-bit floats and cast to PetscScalar */
    PetscCall(PetscMalloc1(count, &fbuf));
    PetscCall(PetscBinaryRead(fdes, fbuf, count, NULL, PETSC_FLOAT));
    PetscCall(PetscByteSwap(fbuf, PETSC_FLOAT, count));
    for (i = 0; i < count; i++) ((PetscScalar*)data)[i] = (PetscScalar)(fbuf[i]);
    PetscCall(PetscFree(fbuf));
  } else {
    PetscCall(PetscViewerASCIIRead(viewer, data, count, NULL, dtype));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateFluent_ReadSection(PetscViewer viewer, FluentSection *s)
{
  char           buffer[PETSC_MAX_PATH_LEN];
  int            snum;

  PetscFunctionBegin;
  /* Fast-forward to next section and derive its index */
  PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, '('));
  PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ' '));
  snum = sscanf(buffer, "%d", &(s->index));
  /* If we can't match an index return -1 to signal end-of-file */
  if (snum < 1) {s->index = -1;   PetscFunctionReturn(0);}

  if (s->index == 0) {           /* Comment */
    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));

  } else if (s->index == 2) {    /* Dimension */
    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
    snum = sscanf(buffer, "%d", &(s->nd));
    PetscCheck(snum == 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");

  } else if (s->index == 10 || s->index == 2010) {   /* Vertices */
    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
    snum = sscanf(buffer, "(%x %x %x %d %d)", &(s->zoneID), &(s->first), &(s->last), &(s->type), &(s->nd));
    PetscCheck(snum == 5,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    if (s->zoneID > 0) {
      PetscInt numCoords = s->last - s->first + 1;
      PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, '('));
      PetscCall(PetscMalloc1(s->nd*numCoords, (PetscScalar**)&s->data));
      PetscCall(DMPlexCreateFluent_ReadValues(viewer, s->data, s->nd*numCoords, PETSC_SCALAR, s->index==2010 ? PETSC_TRUE : PETSC_FALSE));
      PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
    }
    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));

  } else if (s->index == 12 || s->index == 2012) {   /* Cells */
    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
    snum = sscanf(buffer, "(%x", &(s->zoneID));
    PetscCheck(snum == 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    if (s->zoneID == 0) {  /* Header section */
      snum = sscanf(buffer, "(%x %x %x %d)", &(s->zoneID), &(s->first), &(s->last), &(s->nd));
      PetscCheck(snum == 4,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    } else {               /* Data section */
      snum = sscanf(buffer, "(%x %x %x %d %d)", &(s->zoneID), &(s->first), &(s->last), &(s->type), &(s->nd));
      PetscCheck(snum == 5,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
      if (s->nd == 0) {
        /* Read cell type definitions for mixed cells */
        PetscInt numCells = s->last - s->first + 1;
        PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, '('));
        PetscCall(PetscMalloc1(numCells, (PetscInt**)&s->data));
        PetscCall(DMPlexCreateFluent_ReadValues(viewer, s->data, numCells, PETSC_INT, s->index==2012 ? PETSC_TRUE : PETSC_FALSE));
        PetscCall(PetscFree(s->data));
        PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
      }
    }
    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));

  } else if (s->index == 13 || s->index == 2013) {   /* Faces */
    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
    snum = sscanf(buffer, "(%x", &(s->zoneID));
    PetscCheck(snum == 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    if (s->zoneID == 0) {  /* Header section */
      snum = sscanf(buffer, "(%x %x %x %d)", &(s->zoneID), &(s->first), &(s->last), &(s->nd));
      PetscCheck(snum == 4,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    } else {               /* Data section */
      PetscInt f, numEntries, numFaces;
      snum = sscanf(buffer, "(%x %x %x %d %d)", &(s->zoneID), &(s->first), &(s->last), &(s->type), &(s->nd));
      PetscCheck(snum == 5,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
      PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, '('));
      switch (s->nd) {
      case 0: numEntries = PETSC_DETERMINE; break;
      case 2: numEntries = 2 + 2; break;  /* linear */
      case 3: numEntries = 2 + 3; break;  /* triangular */
      case 4: numEntries = 2 + 4; break;  /* quadrilateral */
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown face type in Fluent file");
      }
      numFaces = s->last-s->first + 1;
      if (numEntries != PETSC_DETERMINE) {
        /* Allocate space only if we already know the size of the block */
        PetscCall(PetscMalloc1(numEntries*numFaces, (PetscInt**)&s->data));
      }
      for (f = 0; f < numFaces; f++) {
        if (s->nd == 0) {
          /* Determine the size of the block for "mixed" facets */
          PetscInt numFaceVert = 0;
          PetscCall(DMPlexCreateFluent_ReadValues(viewer, &numFaceVert, 1, PETSC_INT, s->index==2013 ? PETSC_TRUE : PETSC_FALSE));
          if (numEntries == PETSC_DETERMINE) {
            numEntries = numFaceVert + 2;
            PetscCall(PetscMalloc1(numEntries*numFaces, (PetscInt**)&s->data));
          } else {
            PetscCheckFalse(numEntries != numFaceVert + 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No support for mixed faces in Fluent files");
          }
        }
        PetscCall(DMPlexCreateFluent_ReadValues(viewer, &(((PetscInt*)s->data)[f*numEntries]), numEntries, PETSC_INT, s->index==2013 ? PETSC_TRUE : PETSC_FALSE));
      }
      s->nd = numEntries - 2;
      PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
    }
    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));

  } else {                       /* Unknown section type */
    PetscInt depth = 1;
    do {
      /* Match parentheses when parsing unknown sections */
      do PetscCall(PetscViewerRead(viewer, &(buffer[0]), 1, NULL, PETSC_CHAR));
      while (buffer[0] != '(' && buffer[0] != ')');
      if (buffer[0] == '(') depth++;
      if (buffer[0] == ')') depth--;
    } while (depth > 0);
    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, '\n'));
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateFluent - Create a DMPlex mesh from a Fluent mesh file.

  Collective

  Input Parameters:
+ comm  - The MPI communicator
. viewer - The Viewer associated with a Fluent mesh file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: http://aerojet.engr.ucdavis.edu/fluenthelp/html/ug/node1490.htm

  Level: beginner

.seealso: DMPLEX, DMCreate()
@*/
PetscErrorCode DMPlexCreateFluent(MPI_Comm comm, PetscViewer viewer, PetscBool interpolate, DM *dm)
{
  PetscMPIInt    rank;
  PetscInt       c, v, dim = PETSC_DETERMINE, numCells = 0, numVertices = 0, numCellVertices = PETSC_DETERMINE;
  PetscInt       numFaces = PETSC_DETERMINE, f, numFaceEntries = PETSC_DETERMINE, numFaceVertices = PETSC_DETERMINE;
  PetscInt      *faces = NULL, *cellVertices = NULL, *faceZoneIDs = NULL;
  DMLabel        faceSets = NULL;
  PetscInt       d, coordSize;
  PetscScalar   *coords, *coordsIn = NULL;
  PetscSection   coordSection;
  Vec            coordinates;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (rank == 0) {
    FluentSection s;
    do {
      PetscCall(DMPlexCreateFluent_ReadSection(viewer, &s));
      if (s.index == 2) {                 /* Dimension */
        dim = s.nd;

      } else if (s.index == 10 || s.index == 2010) { /* Vertices */
        if (s.zoneID == 0) numVertices = s.last;
        else {
          PetscCheck(!coordsIn,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Currently no support for multiple coordinate sets in Fluent files");
          coordsIn = (PetscScalar *) s.data;
        }

      } else if (s.index == 12 || s.index == 2012) { /* Cells */
        if (s.zoneID == 0) numCells = s.last;
        else {
          switch (s.nd) {
          case 0: numCellVertices = PETSC_DETERMINE; break;
          case 1: numCellVertices = 3; break;  /* triangular */
          case 2: numCellVertices = 4; break;  /* tetrahedral */
          case 3: numCellVertices = 4; break;  /* quadrilateral */
          case 4: numCellVertices = 8; break;  /* hexahedral */
          case 5: numCellVertices = 5; break;  /* pyramid */
          case 6: numCellVertices = 6; break;  /* wedge */
          default: numCellVertices = PETSC_DETERMINE;
          }
        }

      } else if (s.index == 13 || s.index == 2013) { /* Facets */
        if (s.zoneID == 0) {  /* Header section */
          numFaces = (PetscInt) (s.last - s.first + 1);
          if (s.nd == 0 || s.nd == 5) numFaceVertices = PETSC_DETERMINE;
          else numFaceVertices = s.nd;
        } else {              /* Data section */
          unsigned int z;

          PetscCheckFalse(numFaceVertices != PETSC_DETERMINE && s.nd != numFaceVertices,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mixed facets in Fluent files are not supported");
          PetscCheck(numFaces >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No header section for facets in Fluent file");
          if (numFaceVertices == PETSC_DETERMINE) numFaceVertices = s.nd;
          numFaceEntries = numFaceVertices + 2;
          if (!faces) PetscCall(PetscMalloc1(numFaces*numFaceEntries, &faces));
          if (!faceZoneIDs) PetscCall(PetscMalloc1(numFaces, &faceZoneIDs));
          PetscCall(PetscMemcpy(&faces[(s.first-1)*numFaceEntries], s.data, (s.last-s.first+1)*numFaceEntries*sizeof(PetscInt)));
          /* Record the zoneID for each face set */
          for (z = s.first -1; z < s.last; z++) faceZoneIDs[z] = s.zoneID;
          PetscCall(PetscFree(s.data));
        }
      }
    } while (s.index >= 0);
  }
  PetscCallMPI(MPI_Bcast(&dim, 1, MPIU_INT, 0, comm));
  PetscCheck(dim >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Fluent file does not include dimension");

  /* Allocate cell-vertex mesh */
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetDimension(*dm, dim));
  PetscCall(DMPlexSetChart(*dm, 0, numCells + numVertices));
  if (rank == 0) {
    PetscCheck(numCells >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown number of cells in Fluent file");
    /* If no cell type was given we assume simplices */
    if (numCellVertices == PETSC_DETERMINE) numCellVertices = numFaceVertices + 1;
    for (c = 0; c < numCells; ++c) PetscCall(DMPlexSetConeSize(*dm, c, numCellVertices));
  }
  PetscCall(DMSetUp(*dm));

  if (rank == 0 && faces) {
    /* Derive cell-vertex list from face-vertex and face-cell maps */
    PetscCall(PetscMalloc1(numCells*numCellVertices, &cellVertices));
    for (c = 0; c < numCells*numCellVertices; c++) cellVertices[c] = -1;
    for (f = 0; f < numFaces; f++) {
      PetscInt *cell;
      const PetscInt cl = faces[f*numFaceEntries + numFaceVertices];
      const PetscInt cr = faces[f*numFaceEntries + numFaceVertices + 1];
      const PetscInt *face = &(faces[f*numFaceEntries]);

      if (cl > 0) {
        cell = &(cellVertices[(cl-1) * numCellVertices]);
        for (v = 0; v < numFaceVertices; v++) {
          PetscBool found = PETSC_FALSE;
          for (c = 0; c < numCellVertices; c++) {
            if (cell[c] < 0) break;
            if (cell[c] == face[v]-1 + numCells) {found = PETSC_TRUE; break;}
          }
          if (!found) cell[c] = face[v]-1 + numCells;
        }
      }
      if (cr > 0) {
        cell = &(cellVertices[(cr-1) * numCellVertices]);
        for (v = 0; v < numFaceVertices; v++) {
          PetscBool found = PETSC_FALSE;
          for (c = 0; c < numCellVertices; c++) {
            if (cell[c] < 0) break;
            if (cell[c] == face[v]-1 + numCells) {found = PETSC_TRUE; break;}
          }
          if (!found) cell[c] = face[v]-1 + numCells;
        }
      }
    }
    for (c = 0; c < numCells; c++) {
      PetscCall(DMPlexSetCone(*dm, c, &(cellVertices[c*numCellVertices])));
    }
  }
  PetscCall(DMPlexSymmetrize(*dm));
  PetscCall(DMPlexStratify(*dm));
  if (interpolate) {
    DM idm;

    PetscCall(DMPlexInterpolate(*dm, &idm));
    PetscCall(DMDestroy(dm));
    *dm  = idm;
  }

  if (rank == 0 && faces) {
    PetscInt fi, joinSize, meetSize, *fverts, cells[2];
    const PetscInt *join, *meet;
    PetscCall(PetscMalloc1(numFaceVertices, &fverts));
    /* Mark facets by finding the full join of all adjacent vertices */
    for (f = 0; f < numFaces; f++) {
      const PetscInt cl = faces[f*numFaceEntries + numFaceVertices] - 1;
      const PetscInt cr = faces[f*numFaceEntries + numFaceVertices + 1] - 1;
      if (cl > 0 && cr > 0) {
        /* If we know both adjoining cells we can use a single-level meet */
        cells[0] = cl; cells[1] = cr;
        PetscCall(DMPlexGetMeet(*dm, 2, cells, &meetSize, &meet));
        PetscCheck(meetSize == 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not determine Plex facet for Fluent face %" PetscInt_FMT, f);
        PetscCall(DMSetLabelValue_Fast(*dm, &faceSets, "Face Sets", meet[0], faceZoneIDs[f]));
        PetscCall(DMPlexRestoreMeet(*dm, numFaceVertices, fverts, &meetSize, &meet));
      } else {
        for (fi = 0; fi < numFaceVertices; fi++) fverts[fi] = faces[f*numFaceEntries + fi] + numCells - 1;
        PetscCall(DMPlexGetFullJoin(*dm, numFaceVertices, fverts, &joinSize, &join));
        PetscCheck(joinSize == 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not determine Plex facet for Fluent face %" PetscInt_FMT, f);
        PetscCall(DMSetLabelValue_Fast(*dm, &faceSets, "Face Sets", join[0], faceZoneIDs[f]));
        PetscCall(DMPlexRestoreJoin(*dm, numFaceVertices, fverts, &joinSize, &join));
      }
    }
    PetscCall(PetscFree(fverts));
  }

  { /* Create Face Sets label at all processes */
    enum {n = 1};
    PetscBool flag[n];

    flag[0] = faceSets ? PETSC_TRUE : PETSC_FALSE;
    PetscCallMPI(MPI_Bcast(flag, n, MPIU_BOOL, 0, comm));
    if (flag[0]) PetscCall(DMCreateLabel(*dm, "Face Sets"));
  }

  /* Read coordinates */
  PetscCall(DMGetCoordinateSection(*dm, &coordSection));
  PetscCall(PetscSectionSetNumFields(coordSection, 1));
  PetscCall(PetscSectionSetFieldComponents(coordSection, 0, dim));
  PetscCall(PetscSectionSetChart(coordSection, numCells, numCells + numVertices));
  for (v = numCells; v < numCells+numVertices; ++v) {
    PetscCall(PetscSectionSetDof(coordSection, v, dim));
    PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, dim));
  }
  PetscCall(PetscSectionSetUp(coordSection));
  PetscCall(PetscSectionGetStorageSize(coordSection, &coordSize));
  PetscCall(VecCreate(PETSC_COMM_SELF, &coordinates));
  PetscCall(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
  PetscCall(VecSetSizes(coordinates, coordSize, PETSC_DETERMINE));
  PetscCall(VecSetType(coordinates, VECSTANDARD));
  PetscCall(VecGetArray(coordinates, &coords));
  if (rank == 0 && coordsIn) {
    for (v = 0; v < numVertices; ++v) {
      for (d = 0; d < dim; ++d) {
        coords[v*dim+d] = coordsIn[v*dim+d];
      }
    }
  }
  PetscCall(VecRestoreArray(coordinates, &coords));
  PetscCall(DMSetCoordinatesLocal(*dm, coordinates));
  PetscCall(VecDestroy(&coordinates));

  if (rank == 0) {
    PetscCall(PetscFree(cellVertices));
    PetscCall(PetscFree(faces));
    PetscCall(PetscFree(faceZoneIDs));
    PetscCall(PetscFree(coordsIn));
  }
  PetscFunctionReturn(0);
}
