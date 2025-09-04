#define PETSC_DESIRE_FEATURE_TEST_MACROS /* for fileno() */
#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/

/* Utility struct to store the contents of a Fluent file in memory */
typedef struct {
  int          index; /* Type of section */
  unsigned int zoneID;
  unsigned int first;
  unsigned int last;
  int          type;
  int          nd; /* Either ND or element-type */
  void        *data;
} FluentSection;

/*@
  DMPlexCreateFluentFromFile - Create a `DMPLEX` mesh from a Fluent mesh file

  Collective

  Input Parameters:
+ comm        - The MPI communicator
. filename    - Name of the Fluent mesh file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm - The `DM` object representing the mesh

  Level: beginner

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMPlexCreateFromFile()`, `DMPlexCreateFluent()`, `DMPlexCreate()`
@*/
PetscErrorCode DMPlexCreateFluentFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscViewer viewer;

  PetscFunctionBegin;
  /* Create file viewer and build plex */
  PetscCall(PetscViewerCreate(comm, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERASCII));
  PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_READ));
  PetscCall(PetscViewerFileSetName(viewer, filename));
  PetscCall(DMPlexCreateFluent(comm, viewer, interpolate, dm));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexCreateFluent_ReadString(PetscViewer viewer, char *buffer, char delim)
{
  PetscInt ret, i = 0;

  PetscFunctionBegin;
  do PetscCall(PetscViewerRead(viewer, &buffer[i++], 1, &ret, PETSC_CHAR));
  while (ret > 0 && buffer[i - 1] != '\0' && buffer[i - 1] != delim && i < PETSC_MAX_PATH_LEN - 1);
  if (!ret) buffer[i - 1] = '\0';
  else buffer[i] = '\0';
  PetscCheck(i < PETSC_MAX_PATH_LEN - 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Buffer overflow! This is not a valid Fluent file.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexCreateFluent_ReadValues(PetscViewer viewer, void *data, PetscInt count, PetscDataType dtype, PetscBool binary, PetscInt *numClosingParens)
{
  int      fdes = 0;
  FILE    *file;
  PetscInt i;

  PetscFunctionBegin;
  *numClosingParens = 0;
  if (binary) {
    /* Extract raw file descriptor to read binary block */
    PetscCall(PetscViewerASCIIGetPointer(viewer, &file));
    PetscCall(PetscFFlush(file));
    fdes = fileno(file);
  }

  if (!binary && dtype == PETSC_INT) {
    char         cbuf[256];
    unsigned int ibuf;
    int          snum;
    /* Parse hexadecimal ascii integers */
    for (i = 0; i < count; i++) {
      size_t len;

      PetscCall(PetscViewerRead(viewer, cbuf, 1, NULL, PETSC_STRING));
      snum = sscanf(cbuf, "%x", &ibuf);
      PetscCheck(snum == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
      ((PetscInt *)data)[i] = (PetscInt)ibuf;
      // Check for trailing parentheses
      PetscCall(PetscStrlen(cbuf, &len));
      while (cbuf[len - 1] == ')' && len > 0) {
        ++(*numClosingParens);
        --len;
      }
    }
  } else if (binary && dtype == PETSC_INT) {
    /* Always read 32-bit ints and cast to PetscInt */
    int *ibuf;
    PetscCall(PetscMalloc1(count, &ibuf));
    PetscCall(PetscBinaryRead(fdes, ibuf, count, NULL, PETSC_ENUM));
    PetscCall(PetscByteSwap(ibuf, PETSC_ENUM, count));
    for (i = 0; i < count; i++) ((PetscInt *)data)[i] = ibuf[i];
    PetscCall(PetscFree(ibuf));

  } else if (binary && dtype == PETSC_SCALAR) {
    float *fbuf;
    /* Always read 32-bit floats and cast to PetscScalar */
    PetscCall(PetscMalloc1(count, &fbuf));
    PetscCall(PetscBinaryRead(fdes, fbuf, count, NULL, PETSC_FLOAT));
    PetscCall(PetscByteSwap(fbuf, PETSC_FLOAT, count));
    for (i = 0; i < count; i++) ((PetscScalar *)data)[i] = fbuf[i];
    PetscCall(PetscFree(fbuf));
  } else {
    PetscCall(PetscViewerASCIIRead(viewer, data, count, NULL, dtype));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexCreateFluent_ReadSection(PetscViewer viewer, FluentSection *s)
{
  char buffer[PETSC_MAX_PATH_LEN];
  int  snum;

  PetscFunctionBegin;
  /* Fast-forward to next section and derive its index */
  PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, '('));
  PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ' '));
  snum = sscanf(buffer, "%d", &s->index);
  /* If we can't match an index return -1 to signal end-of-file */
  if (snum < 1) {
    s->index = -1;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (s->index == 0) { /* Comment */
    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));

  } else if (s->index == 2) { /* Dimension */
    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
    snum = sscanf(buffer, "%d", &s->nd);
    PetscCheck(snum == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");

  } else if (s->index == 10 || s->index == 2010) { /* Vertices */
    PetscInt numClosingParens = 0;

    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
    snum = sscanf(buffer, "(%x %x %x %d %d)", &s->zoneID, &s->first, &s->last, &s->type, &s->nd);
    PetscCheck(snum == 5, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    if (s->zoneID > 0) {
      PetscInt numCoords = s->last - s->first + 1;
      PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, '('));
      PetscCall(PetscMalloc1(s->nd * numCoords, (PetscScalar **)&s->data));
      PetscCall(DMPlexCreateFluent_ReadValues(viewer, s->data, s->nd * numCoords, PETSC_SCALAR, s->index == 2010 ? PETSC_TRUE : PETSC_FALSE, &numClosingParens));
      if (!numClosingParens) PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
      else --numClosingParens;
    }
    if (!numClosingParens) PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
    else --numClosingParens;
    PetscCheck(!numClosingParens, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
  } else if (s->index == 12 || s->index == 2012) { /* Cells */
    PetscInt numClosingParens = 0;

    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
    snum = sscanf(buffer, "(%x", &s->zoneID);
    PetscCheck(snum == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    if (s->zoneID == 0) { /* Header section */
      snum = sscanf(buffer, "(%x %x %x %d)", &s->zoneID, &s->first, &s->last, &s->nd);
      PetscCheck(snum == 4, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    } else { /* Data section */
      snum = sscanf(buffer, "(%x %x %x %d %d)", &s->zoneID, &s->first, &s->last, &s->type, &s->nd);
      PetscCheck(snum == 5, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
      if (s->nd == 0) {
        /* Read cell type definitions for mixed cells */
        PetscInt numCells = s->last - s->first + 1;
        PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, '('));
        PetscCall(PetscMalloc1(numCells, (PetscInt **)&s->data));
        PetscCall(DMPlexCreateFluent_ReadValues(viewer, s->data, numCells, PETSC_INT, s->index == 2012 ? PETSC_TRUE : PETSC_FALSE, &numClosingParens));
        if (!numClosingParens) PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
        else --numClosingParens;
      }
    }
    if (!numClosingParens) PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
    else --numClosingParens;
    PetscCheck(!numClosingParens, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
  } else if (s->index == 13 || s->index == 2013) { /* Faces */
    PetscInt numClosingParens = 0;

    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
    snum = sscanf(buffer, "(%x", &s->zoneID);
    PetscCheck(snum == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    if (s->zoneID == 0) { /* Header section */
      snum = sscanf(buffer, "(%x %x %x %d)", &s->zoneID, &s->first, &s->last, &s->nd);
      PetscCheck(snum == 4, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    } else { /* Data section */
      PetscInt numEntries, numFaces, maxsize = 0, offset = 0;

      snum = sscanf(buffer, "(%x %x %x %d %d)", &s->zoneID, &s->first, &s->last, &s->type, &s->nd);
      PetscCheck(snum == 5, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
      PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, '('));
      switch (s->nd) {
      case 0:
        numEntries = PETSC_DETERMINE;
        break;
      case 2:
        numEntries = 2 + 2;
        break; /* linear */
      case 3:
        numEntries = 2 + 3;
        break; /* triangular */
      case 4:
        numEntries = 2 + 4;
        break; /* quadrilateral */
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown face type in Fluent file");
      }
      numFaces = s->last - s->first + 1;
      if (numEntries != PETSC_DETERMINE) {
        /* Allocate space only if we already know the size of the block */
        PetscCall(PetscMalloc1(numEntries * numFaces, (PetscInt **)&s->data));
      }
      for (PetscInt f = 0; f < numFaces; f++) {
        if (s->nd == 0) {
          /* Determine the size of the block for "mixed" facets */
          PetscInt numFaceVert = 0;
          PetscCall(DMPlexCreateFluent_ReadValues(viewer, &numFaceVert, 1, PETSC_INT, s->index == 2013 ? PETSC_TRUE : PETSC_FALSE, &numClosingParens));
          if (!f) {
            maxsize = (numFaceVert + 3) * numFaces;
            PetscCall(PetscMalloc1(maxsize, (PetscInt **)&s->data));
          } else {
            if (offset + numFaceVert + 3 >= maxsize) {
              PetscInt *tmp;

              PetscCall(PetscMalloc1(maxsize * 2, &tmp));
              PetscCall(PetscArraycpy(tmp, (PetscInt *)s->data, maxsize));
              PetscCall(PetscFree(s->data));
              maxsize *= 2;
              s->data = tmp;
            }
          }
          ((PetscInt *)s->data)[offset] = numFaceVert;
          ++offset;
          numEntries = numFaceVert + 2;
        }
        PetscCall(DMPlexCreateFluent_ReadValues(viewer, &(((PetscInt *)s->data)[offset]), numEntries, PETSC_INT, s->index == 2013 ? PETSC_TRUE : PETSC_FALSE, &numClosingParens));
        offset += numEntries;
      }
      if (s->nd != 0) PetscCall(PetscMPIIntCast(numEntries - 2, &s->nd));
      if (!numClosingParens) PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
      else --numClosingParens;
    }
    if (!numClosingParens) PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
    else --numClosingParens;
    PetscCheck(!numClosingParens, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
  } else if (s->index == 39) { /* Label information */
    char labelName[PETSC_MAX_PATH_LEN];
    char caseName[PETSC_MAX_PATH_LEN];

    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, ')'));
    snum = sscanf(buffer, "(%u %s %s %d)", &s->zoneID, caseName, labelName, &s->nd);
    PetscCheck(snum == 4, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file: %d", snum);
    PetscInt depth = 1;
    do {
      /* Match parentheses when parsing unknown sections */
      do PetscCall(PetscViewerRead(viewer, &buffer[0], 1, NULL, PETSC_CHAR));
      while (buffer[0] != '(' && buffer[0] != ')');
      if (buffer[0] == '(') depth++;
      if (buffer[0] == ')') depth--;
    } while (depth > 0);
    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, '\n'));
    PetscCall(PetscStrallocpy(labelName, (char **)&s->data));
    PetscCall(PetscInfo((PetscObject)viewer, "CASE: Zone ID %u is label %s\n", s->zoneID, labelName));
  } else { /* Unknown section type */
    PetscInt depth = 1;
    do {
      /* Match parentheses when parsing unknown sections */
      do PetscCall(PetscViewerRead(viewer, &buffer[0], 1, NULL, PETSC_CHAR));
      while (buffer[0] != '(' && buffer[0] != ')');
      if (buffer[0] == '(') depth++;
      if (buffer[0] == ')') depth--;
    } while (depth > 0);
    PetscCall(DMPlexCreateFluent_ReadString(viewer, buffer, '\n'));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Inserts point `face` with orientation `ornt` into the cone of point `cell` at position `c`, which is the first empty slot
static PetscErrorCode InsertFace(DM dm, PetscInt cell, PetscInt face, PetscInt ornt)
{
  const PetscInt *cone;
  PetscInt        coneSize, c;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCone(dm, cell, &cone));
  PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
  for (c = 0; c < coneSize; ++c)
    if (cone[c] < 0) break;
  PetscCheck(c < coneSize, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " could not be inserted in cone of cell %" PetscInt_FMT " with size %" PetscInt_FMT, face, cell, coneSize);
  PetscCall(DMPlexInsertCone(dm, cell, c, face));
  PetscCall(DMPlexInsertConeOrientation(dm, cell, c, ornt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReorderPolygon(DM dm, PetscInt cell)
{
  const PetscInt *cone, *ornt;
  PetscInt        coneSize, newCone[16], newOrnt[16];

  PetscFunctionBegin;
  PetscCall(DMPlexGetOrientedCone(dm, cell, &cone, &ornt));
  PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
  newCone[0] = cone[0];
  newOrnt[0] = ornt[0];
  for (PetscInt c = 1; c < coneSize; ++c) {
    const PetscInt *fcone;
    PetscInt        firstVertex, lastVertex, c2;

    PetscCall(DMPlexGetCone(dm, newCone[c - 1], &fcone));
    lastVertex = newOrnt[c - 1] ? fcone[0] : fcone[1];
    for (c2 = 0; c2 < coneSize; ++c2) {
      const PetscInt *fcone2;

      PetscCall(DMPlexGetCone(dm, cone[c2], &fcone2));
      firstVertex = ornt[c2] ? fcone2[1] : fcone2[0];
      if (lastVertex == firstVertex) {
        // Point `cell` matched point `lastVertex` on face `cone[c2]` with orientation `ornt[c2]`
        break;
      }
    }
    PetscCheck(c2 < coneSize, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %" PetscInt_FMT " could not find a face match as position %" PetscInt_FMT, cell, c);
    newCone[c] = cone[c2];
    newOrnt[c] = ornt[c2];
  }
  {
    const PetscInt *fcone, *fcone2;
    PetscInt        vertex, vertex2;

    PetscCall(DMPlexGetCone(dm, newCone[coneSize - 1], &fcone));
    PetscCall(DMPlexGetCone(dm, newCone[0], &fcone2));
    vertex  = newOrnt[coneSize - 1] ? fcone[0] : fcone[1];
    vertex2 = newOrnt[0] ? fcone2[1] : fcone2[0];
    PetscCheck(vertex == vertex2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %" PetscInt_FMT " did not match at the endpoint", cell);
  }
  PetscCall(DMPlexSetCone(dm, cell, newCone));
  PetscCall(DMPlexSetConeOrientation(dm, cell, newOrnt));
  PetscCall(DMPlexRestoreOrientedCone(dm, cell, &cone, &ornt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReorderTetrahedron(PetscViewer viewer, DM dm, PetscInt cell)
{
  const PetscInt *cone, *ornt, *fcone, *fornt, *farr, faces[4] = {0, 1, 3, 2};
  PetscInt        newCone[16], newOrnt[16];

  PetscFunctionBegin;
  PetscCall(DMPlexGetOrientedCone(dm, cell, &cone, &ornt));
  newCone[0] = cone[0];
  newOrnt[0] = ornt[0];
  PetscCall(DMPlexGetOrientedCone(dm, newCone[0], &fcone, &fornt));
  farr = DMPolytopeTypeGetArrangement(DM_POLYTOPE_TRIANGLE, newOrnt[0]);
  // Loop over each edge in the initial triangle
  for (PetscInt e = 0; e < 3; ++e) {
    const PetscInt edge = fcone[farr[e * 2 + 0]], eornt = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, farr[e * 2 + 1], fornt[farr[e * 2 + 0]]);
    PetscInt       c;

    // Loop over each remaining face in the tetrahedron
    //   On face `newCone[0]`, trying to match edge `edge` with final orientation `eornt` to an edge on another face
    for (c = 1; c < 4; ++c) {
      const PetscInt *fcone2, *fornt2, *farr2;
      PetscInt        c2;
      PetscBool       flip = PETSC_FALSE;

      // Checking face `cone[c]` with orientation `ornt[c]`
      PetscCall(DMPlexGetOrientedCone(dm, cone[c], &fcone2, &fornt2));
      farr2 = DMPolytopeTypeGetArrangement(DM_POLYTOPE_TRIANGLE, ornt[c]);
      // Check for edge
      for (c2 = 0; c2 < 3; ++c2) {
        const PetscInt edge2 = fcone2[farr2[c2 * 2 + 0]], eornt2 = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, farr2[c2 * 2 + 1], fornt2[farr2[c2 * 2 + 0]]);
        // Trying to match edge `edge2` with final orientation `eornt2`
        if (edge == edge2) {
          //PetscCheck(eornt == -(eornt2 + 1), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell % " PetscInt_FMT " edge %" PetscInt_FMT " (%" PetscInt_FMT ") found twice with the same orientation in face %" PetscInt_FMT " edge %" PetscInt_FMT, cell, edge, e, c, c2);
          // Matched face `newCone[0]` with orientation `newOrnt[0]` to face `cone[c]` with orientation `ornt[c]` along edge `edge`
          PetscCall(PetscInfo((PetscObject)viewer, "CASE: Matched cell %" PetscInt_FMT " edge %" PetscInt_FMT "/%" PetscInt_FMT " (%" PetscInt_FMT ") to face %" PetscInt_FMT "/%" PetscInt_FMT " edge %" PetscInt_FMT " (%" PetscInt_FMT ")\n", cell, edge, e, eornt, cone[c], c, c2, eornt2));
          flip = eornt != -(eornt2 + 1) ? PETSC_TRUE : PETSC_FALSE;
          break;
        }
      }
      if (c2 < 3) {
        newCone[faces[e + 1]] = cone[c];
        // Compute new orientation of face based on which edge was matched (only the first edge matches a side different from 0)
        //   Face 1 should match its edge 2
        //   Face 2 should match its edge 0
        //   Face 3 should match its edge 0
        if (flip) {
          newOrnt[faces[e + 1]] = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_TRIANGLE, -((c2 + (!e ? 1 : 2)) % 3 + 1), ornt[c]);
        } else {
          newOrnt[faces[e + 1]] = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_TRIANGLE, !e ? (c2 + 1) % 3 : c2, ornt[c]);
        }
        break;
      }
    }
    PetscCheck(c < 4, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %" PetscInt_FMT " could not find a face match for edge %" PetscInt_FMT, cell, e);
  }
  PetscCall(DMPlexRestoreOrientedCone(dm, newCone[0], &fcone, &fornt));
  PetscCall(DMPlexSetCone(dm, cell, newCone));
  PetscCall(DMPlexSetConeOrientation(dm, cell, newOrnt));
  PetscCall(DMPlexRestoreOrientedCone(dm, cell, &cone, &ornt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReorderHexahedron(DM dm, PetscInt cell)
{
  const PetscInt *cone, *ornt, *fcone, *fornt, *farr;
  const PetscInt  faces[6] = {0, 5, 3, 4, 2, 1};
  PetscInt        used[6]  = {1, 0, 0, 0, 0, 0};
  PetscInt        newCone[16], newOrnt[16];

  PetscFunctionBegin;
  PetscCall(DMPlexGetOrientedCone(dm, cell, &cone, &ornt));
  newCone[0] = cone[0];
  newOrnt[0] = ornt[0];
  PetscCall(DMPlexGetOrientedCone(dm, newCone[0], &fcone, &fornt));
  farr = DMPolytopeTypeGetArrangement(DM_POLYTOPE_QUADRILATERAL, newOrnt[0]);
  // Loop over each edge in the initial quadrilateral
  for (PetscInt e = 0; e < 4; ++e) {
    const PetscInt edge = fcone[farr[e * 2 + 0]], eornt = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, farr[e * 2 + 1], fornt[farr[e * 2 + 0]]);
    PetscInt       c;

    // Loop over each remaining face in the hexahedron
    //   On face `newCone[0]`, trying to match edge `edge` with final orientation `eornt` to an edge on another face
    for (c = 1; c < 6; ++c) {
      const PetscInt *fcone2, *fornt2, *farr2;
      PetscInt        c2;

      // Checking face `cone[c]` with orientation `ornt[c]`
      PetscCall(DMPlexGetOrientedCone(dm, cone[c], &fcone2, &fornt2));
      farr2 = DMPolytopeTypeGetArrangement(DM_POLYTOPE_QUADRILATERAL, ornt[c]);
      // Check for edge
      for (c2 = 0; c2 < 4; ++c2) {
        const PetscInt edge2 = fcone2[farr2[c2 * 2 + 0]], eornt2 = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, farr2[c2 * 2 + 1], fornt2[farr2[c2 * 2 + 0]]);
        // Trying to match edge `edge2` with final orientation `eornt2`
        if (edge == edge2) {
          PetscCheck(eornt == -(eornt2 + 1), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Edge %" PetscInt_FMT " found twice with the same orientation", edge);
          // Matched face `newCone[0]` with orientation `newOrnt[0]` to face `cone[c]` with orientation `ornt[c]` along edge `edge`
          break;
        }
      }
      if (c2 < 4) {
        used[c]               = 1;
        newCone[faces[e + 1]] = cone[c];
        // Compute new orientation of face based on which edge was matched (only the first edge matches a side different from 0)
        newOrnt[faces[e + 1]] = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_QUADRILATERAL, !e ? (c2 + 1) % 4 : c2, ornt[c]);
        break;
      }
    }
    PetscCheck(c < 6, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %" PetscInt_FMT " could not find a face match for edge %" PetscInt_FMT, cell, e);
  }
  PetscCall(DMPlexRestoreOrientedCone(dm, newCone[0], &fcone, &fornt));
  // Add last face
  {
    PetscInt c, c2;

    for (c = 1; c < 6; ++c)
      if (!used[c]) break;
    PetscCheck(c < 6, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %" PetscInt_FMT " could not find an available face", cell);
    // Match first edge to 3rd edge in newCone[2]
    {
      const PetscInt *fcone2, *fornt2, *farr2;

      PetscCall(DMPlexGetOrientedCone(dm, newCone[2], &fcone, &fornt));
      farr = DMPolytopeTypeGetArrangement(DM_POLYTOPE_QUADRILATERAL, newOrnt[2]);
      PetscCall(DMPlexGetOrientedCone(dm, cone[c], &fcone2, &fornt2));
      farr2 = DMPolytopeTypeGetArrangement(DM_POLYTOPE_QUADRILATERAL, ornt[c]);

      const PetscInt e    = 2;
      const PetscInt edge = fcone[farr[e * 2 + 0]], eornt = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, farr[e * 2 + 1], fornt[farr[e * 2 + 0]]);
      // Trying to match edge `edge` with final orientation `eornt` of face `newCone[2]` to some edge of face `cone[c]` with orientation `ornt[c]`
      for (c2 = 0; c2 < 4; ++c2) {
        const PetscInt edge2 = fcone2[farr2[c2 * 2 + 0]], eornt2 = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, farr2[c2 * 2 + 1], fornt2[farr2[c2 * 2 + 0]]);
        // Trying to match edge `edge2` with final orientation `eornt2`
        if (edge == edge2) {
          PetscCheck(eornt == -(eornt2 + 1), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Edge %" PetscInt_FMT " found twice with the same orientation", edge);
          // Matched face `newCone[2]` with orientation `newOrnt[2]` to face `cone[c]` with orientation `ornt[c]` along edge `edge`
          break;
        }
      }
      PetscCheck(c2 < 4, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not fit last face in");
    }
    newCone[faces[5]] = cone[c];
    // Compute new orientation of face based on which edge was matched
    newOrnt[faces[5]] = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_QUADRILATERAL, c2, ornt[c]);
    PetscCall(DMPlexRestoreOrientedCone(dm, newCone[0], &fcone, &fornt));
  }
  PetscCall(DMPlexSetCone(dm, cell, newCone));
  PetscCall(DMPlexSetConeOrientation(dm, cell, newOrnt));
  PetscCall(DMPlexRestoreOrientedCone(dm, cell, &cone, &ornt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// {0, 1, 2}, {3, 4, 5}, {0, 2, 4, 3}, {2, 1, 5, 4}, {1, 0, 3, 5}
static PetscErrorCode ReorderWedge(DM dm, PetscInt cell)
{
  const PetscInt *cone, *ornt, *fcone, *fornt, *farr;
  const PetscInt  faces[5] = {0, 4, 3, 2, 1};
  PetscInt        used[5]  = {0, 0, 0, 0, 0};
  PetscInt        newCone[16], newOrnt[16], cS, bottom = 0;

  PetscFunctionBegin;
  PetscCall(DMPlexGetConeSize(dm, cell, &cS));
  PetscCall(DMPlexGetOrientedCone(dm, cell, &cone, &ornt));
  for (PetscInt c = 0; c < cS; ++c) {
    DMPolytopeType ct;

    PetscCall(DMPlexGetCellType(dm, cone[c], &ct));
    if (ct == DM_POLYTOPE_TRIANGLE) {
      bottom = c;
      break;
    }
  }
  used[bottom] = 1;
  newCone[0]   = cone[bottom];
  newOrnt[0]   = ornt[bottom];
  PetscCall(DMPlexGetOrientedCone(dm, newCone[0], &fcone, &fornt));
  farr = DMPolytopeTypeGetArrangement(DM_POLYTOPE_TRIANGLE, newOrnt[0]);
  // Loop over each edge in the initial triangle
  for (PetscInt e = 0; e < 3; ++e) {
    const PetscInt edge = fcone[farr[e * 2 + 0]], eornt = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, farr[e * 2 + 1], fornt[farr[e * 2 + 0]]);
    PetscInt       c;

    // Loop over each remaining face in the prism
    //   On face `newCone[0]`, trying to match edge `edge` with final orientation `eornt` to an edge on another face
    for (c = 0; c < 5; ++c) {
      const PetscInt *fcone2, *fornt2, *farr2;
      DMPolytopeType  ct;
      PetscInt        c2;

      if (c == bottom) continue;
      PetscCall(DMPlexGetCellType(dm, cone[c], &ct));
      if (ct != DM_POLYTOPE_QUADRILATERAL) continue;
      // Checking face `cone[c]` with orientation `ornt[c]`
      PetscCall(DMPlexGetOrientedCone(dm, cone[c], &fcone2, &fornt2));
      farr2 = DMPolytopeTypeGetArrangement(DM_POLYTOPE_QUADRILATERAL, ornt[c]);
      // Check for edge
      for (c2 = 0; c2 < 4; ++c2) {
        const PetscInt edge2 = fcone2[farr2[c2 * 2 + 0]], eornt2 = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, farr2[c2 * 2 + 1], fornt2[farr2[c2 * 2 + 0]]);
        // Trying to match edge `edge2` with final orientation `eornt2`
        if (edge == edge2) {
          PetscCheck(eornt == -(eornt2 + 1), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Edge %" PetscInt_FMT " found twice with the same orientation", edge);
          // Matched face `newCone[0]` with orientation `newOrnt[0]` to face `cone[c]` with orientation `ornt[c]` along edge `edge`
          break;
        }
      }
      if (c2 < 4) {
        used[c]               = 1;
        newCone[faces[e + 1]] = cone[c];
        // Compute new orientation of face based on which edge was matched, edge 0 should always match the bottom
        newOrnt[faces[e + 1]] = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_QUADRILATERAL, c2, ornt[c]);
        break;
      }
    }
    PetscCheck(c < 5, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %" PetscInt_FMT " could not find a face match for edge %" PetscInt_FMT, cell, e);
  }
  PetscCall(DMPlexRestoreOrientedCone(dm, newCone[0], &fcone, &fornt));
  // Add last face
  {
    PetscInt c, c2;

    for (c = 0; c < 5; ++c)
      if (!used[c]) break;
    PetscCheck(c < 5, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %" PetscInt_FMT " could not find an available face", cell);
    // Match first edge to 3rd edge in newCone[2]
    {
      const PetscInt *fcone2, *fornt2, *farr2;

      PetscCall(DMPlexGetOrientedCone(dm, newCone[2], &fcone, &fornt));
      farr = DMPolytopeTypeGetArrangement(DM_POLYTOPE_QUADRILATERAL, newOrnt[2]);
      PetscCall(DMPlexGetOrientedCone(dm, cone[c], &fcone2, &fornt2));
      farr2 = DMPolytopeTypeGetArrangement(DM_POLYTOPE_TRIANGLE, ornt[c]);

      const PetscInt e    = 2;
      const PetscInt edge = fcone[farr[e * 2 + 0]], eornt = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, farr[e * 2 + 1], fornt[farr[e * 2 + 0]]);
      // Trying to match edge `edge` with final orientation `eornt` of face `newCone[2]` to some edge of face `cone[c]` with orientation `ornt[c]`
      for (c2 = 0; c2 < 3; ++c2) {
        const PetscInt edge2 = fcone2[farr2[c2 * 2 + 0]], eornt2 = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, farr2[c2 * 2 + 1], fornt2[farr2[c2 * 2 + 0]]);
        // Trying to match edge `edge2` with final orientation `eornt2`
        if (edge == edge2) {
          PetscCheck(eornt == -(eornt2 + 1), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Edge %" PetscInt_FMT " found twice with the same orientation", edge);
          // Matched face `newCone[2]` with orientation `newOrnt[2]` to face `cone[c]` with orientation `ornt[c]` along edge `edge`
          break;
        }
      }
      PetscCheck(c2 < 3, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not fit last face in");
    }
    newCone[faces[4]] = cone[c];
    // Compute new orientation of face based on which edge was matched
    newOrnt[faces[4]] = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_TRIANGLE, c2, ornt[c]);
    PetscCall(DMPlexRestoreOrientedCone(dm, newCone[0], &fcone, &fornt));
  }
  PetscCall(DMPlexSetCone(dm, cell, newCone));
  PetscCall(DMPlexSetConeOrientation(dm, cell, newOrnt));
  PetscCall(DMPlexRestoreOrientedCone(dm, cell, &cone, &ornt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReorderCell(PetscViewer viewer, DM dm, PetscInt cell, DMPolytopeType ct)
{
  PetscFunctionBegin;
  switch (ct) {
  case DM_POLYTOPE_TRIANGLE:
  case DM_POLYTOPE_QUADRILATERAL:
    PetscCall(ReorderPolygon(dm, cell));
    break;
  case DM_POLYTOPE_TETRAHEDRON:
    PetscCall(ReorderTetrahedron(viewer, dm, cell));
    break;
  case DM_POLYTOPE_HEXAHEDRON:
    PetscCall(ReorderHexahedron(dm, cell));
    break;
  case DM_POLYTOPE_TRI_PRISM:
    PetscCall(ReorderWedge(dm, cell));
    break;
  default:
    PetscCheck(0, PETSC_COMM_SELF, PETSC_ERR_SUP, "Celltype %s is unsupported", DMPolytopeTypes[ct]);
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetNumCellFaces(int nd, PetscInt *numCellFaces, DMPolytopeType *ct)
{
  PetscFunctionBegin;
  *ct = DM_POLYTOPE_POINT;
  switch (nd) {
  case 0:
    *numCellFaces = PETSC_DETERMINE;
    break;
  case 1:
    *numCellFaces = 3;
    *ct           = DM_POLYTOPE_TRIANGLE;
    break;
  case 2:
    *numCellFaces = 4;
    *ct           = DM_POLYTOPE_TETRAHEDRON;
    break;
  case 3:
    *numCellFaces = 4;
    *ct           = DM_POLYTOPE_QUADRILATERAL;
    break;
  case 4:
    *numCellFaces = 6;
    *ct           = DM_POLYTOPE_HEXAHEDRON;
    break;
  case 5:
    *numCellFaces = 5;
    *ct           = DM_POLYTOPE_PYRAMID;
    break;
  case 6:
    *numCellFaces = 5;
    *ct           = DM_POLYTOPE_TRI_PRISM;
    break;
  default:
    *numCellFaces = PETSC_DETERMINE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexCreateFluent - Create a `DMPLEX` mesh from a Fluent mesh file <http://aerojet.engr.ucdavis.edu/fluenthelp/html/ug/node1490.htm>.

  Collective

  Input Parameters:
+ comm        - The MPI communicator
. viewer      - The `PetscViewer` associated with a Fluent mesh file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm - The `DM` object representing the mesh

  Level: beginner

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMCreate()`
@*/
PetscErrorCode DMPlexCreateFluent(MPI_Comm comm, PetscViewer viewer, PetscBool interpolate, DM *dm)
{
  PetscInt        dim          = PETSC_DETERMINE;
  PetscInt        numCells     = 0;
  PetscInt        numVertices  = 0;
  PetscInt       *cellSizes    = NULL;
  DMPolytopeType *cellTypes    = NULL;
  PetscInt        numFaces     = 0;
  PetscInt       *faces        = NULL;
  PetscInt       *faceSizes    = NULL;
  PetscInt       *faceAdjCell  = NULL;
  PetscInt       *cellVertices = NULL;
  unsigned int   *faceZoneIDs  = NULL;
  DMLabel         faceSets     = NULL;
  DMLabel        *zoneLabels   = NULL;
  const char    **zoneNames    = NULL;
  unsigned int    maxZoneID    = 0;
  PetscScalar    *coordsIn     = NULL;
  PetscScalar    *coords;
  PetscSection    coordSection;
  Vec             coordinates;
  PetscInt        coordSize, maxFaceSize = 0, totFaceVert = 0, f;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (rank == 0) {
    FluentSection s;

    s.data   = NULL;
    numFaces = PETSC_DETERMINE;
    do {
      PetscCall(DMPlexCreateFluent_ReadSection(viewer, &s));
      if (s.index == 2) { /* Dimension */
        dim = s.nd;
        PetscCall(PetscInfo((PetscObject)viewer, "CASE: Found dimension: %" PetscInt_FMT "\n", dim));
      } else if (s.index == 10 || s.index == 2010) { /* Vertices */
        if (s.zoneID == 0) {
          numVertices = s.last;
          PetscCall(PetscInfo((PetscObject)viewer, "CASE: Found number of vertices: %" PetscInt_FMT "\n", numVertices));
        } else {
          PetscCall(PetscInfo((PetscObject)viewer, "CASE: Found vertex coordinates\n"));
          PetscCheck(!coordsIn, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Currently no support for multiple coordinate sets in Fluent files");
          coordsIn = (PetscScalar *)s.data;
        }

      } else if (s.index == 12 || s.index == 2012) { /* Cells */
        if (s.zoneID == 0) {
          numCells = s.last;
          PetscCall(PetscInfo((PetscObject)viewer, "CASE: Found number of cells %" PetscInt_FMT "\n", numCells));
        } else {
          PetscCall(PetscMalloc2(numCells, &cellSizes, numCells, &cellTypes));
          for (PetscInt c = 0; c < numCells; ++c) PetscCall(GetNumCellFaces(s.nd ? s.nd : (int)((PetscInt *)s.data)[c], &cellSizes[c], &cellTypes[c]));
          PetscCall(PetscFree(s.data));
          PetscCall(PetscInfo((PetscObject)viewer, "CASE: Found number of cell faces %" PetscInt_FMT "\n", numCells && s.nd ? cellSizes[0] : 0));
        }
      } else if (s.index == 13 || s.index == 2013) { /* Facets */
        if (s.zoneID == 0) {                         /* Header section */
          numFaces = (PetscInt)(s.last - s.first + 1);
          PetscCall(PetscInfo((PetscObject)viewer, "CASE: Found number of faces %" PetscInt_FMT " face vertices: %d\n", numFaces, s.nd));
        } else { /* Data section */
          PetscInt *tmp;
          PetscInt  totSize = 0, offset = 0, doffset;

          PetscCheck(numFaces >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No header section for facets in Fluent file");
          if (!faceZoneIDs) PetscCall(PetscMalloc3(numFaces, &faceSizes, numFaces * 2, &faceAdjCell, numFaces, &faceZoneIDs));
          // Record the zoneID and face size for each face set
          for (unsigned int z = s.first - 1; z < s.last; z++) {
            faceZoneIDs[z] = s.zoneID;
            if (s.nd) {
              faceSizes[z] = s.nd;
            } else {
              faceSizes[z] = ((PetscInt *)s.data)[offset];
              offset += faceSizes[z] + 3;
            }
            totSize += faceSizes[z];
            maxFaceSize = PetscMax(maxFaceSize, faceSizes[z]);
          }

          offset  = totFaceVert;
          doffset = s.nd ? 0 : 1;
          PetscCall(PetscMalloc1(totFaceVert + totSize, &tmp));
          if (faces) PetscCall(PetscArraycpy(tmp, faces, totFaceVert));
          PetscCall(PetscFree(faces));
          totFaceVert += totSize;
          faces = tmp;

          // Record face vertices and adjacent faces
          const PetscInt Nfz = s.last - s.first + 1;
          for (PetscInt f = 0; f < Nfz; ++f) {
            const PetscInt face     = f + s.first - 1;
            const PetscInt faceSize = faceSizes[face];

            for (PetscInt v = 0; v < faceSize; ++v) faces[offset + v] = ((PetscInt *)s.data)[doffset + v];
            faceAdjCell[face * 2 + 0] = ((PetscInt *)s.data)[doffset + faceSize + 0];
            faceAdjCell[face * 2 + 1] = ((PetscInt *)s.data)[doffset + faceSize + 1];
            offset += faceSize;
            doffset += faceSize + (s.nd ? 2 : 3);
          }
          PetscCall(PetscFree(s.data));
        }
      } else if (s.index == 39) { /* Label information */
        if (s.zoneID >= maxZoneID) {
          DMLabel     *tmpL;
          const char **tmp;
          unsigned int newmax = maxZoneID + 1;

          while (newmax < s.zoneID + 1) newmax *= 2;
          PetscCall(PetscCalloc2(newmax, &tmp, newmax, &tmpL));
          for (PetscInt i = 0; i < (PetscInt)maxZoneID; ++i) {
            tmp[i]  = zoneNames[i];
            tmpL[i] = zoneLabels[i];
          }
          maxZoneID = newmax;
          PetscCall(PetscFree2(zoneNames, zoneLabels));
          zoneNames  = tmp;
          zoneLabels = tmpL;
        }
        zoneNames[s.zoneID] = (const char *)s.data;
      }
    } while (s.index >= 0);
  }
  PetscCallMPI(MPI_Bcast(&dim, 1, MPIU_INT, 0, comm));
  PetscCheck(dim >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Fluent file does not include dimension");

  /* Allocate cell-vertex mesh */
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetDimension(*dm, dim));
  // We do not want this label automatically computed, instead we fill it here
  PetscCall(DMCreateLabel(*dm, "celltype"));
  PetscCall(DMPlexSetChart(*dm, 0, numCells + numFaces + numVertices));
  if (rank == 0) {
    PetscCheck(numCells >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown number of cells in Fluent file");
    for (PetscInt c = 0; c < numCells; ++c) {
      PetscCall(DMPlexSetConeSize(*dm, c, cellSizes[c]));
      PetscCall(DMPlexSetCellType(*dm, c, cellTypes[c]));
    }
    for (PetscInt v = numCells; v < numCells + numVertices; ++v) PetscCall(DMPlexSetCellType(*dm, v, DM_POLYTOPE_POINT));
    for (PetscInt f = 0; f < numFaces; ++f) {
      DMPolytopeType ct;

      switch (faceSizes[f]) {
      case 2:
        ct = DM_POLYTOPE_SEGMENT;
        break;
      case 3:
        ct = DM_POLYTOPE_TRIANGLE;
        break;
      case 4:
        ct = DM_POLYTOPE_QUADRILATERAL;
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown face type in Fluent file with cone size %" PetscInt_FMT, faceSizes[f]);
      }
      PetscCall(DMPlexSetConeSize(*dm, f + numCells + numVertices, faceSizes[f]));
      PetscCall(DMPlexSetCellType(*dm, f + numCells + numVertices, ct));
    }
  }
  PetscCall(DMSetUp(*dm));

  if (rank == 0 && faces) {
    PetscSection s;
    PetscInt    *cones, csize, foffset = 0;

    PetscCall(DMPlexGetCones(*dm, &cones));
    PetscCall(DMPlexGetConeSection(*dm, &s));
    PetscCall(PetscSectionGetConstrainedStorageSize(s, &csize));
    for (PetscInt c = 0; c < csize; ++c) cones[c] = -1;
    for (PetscInt f = 0; f < numFaces; f++) {
      const PetscInt cl   = faceAdjCell[f * 2 + 0] - 1;
      const PetscInt cr   = faceAdjCell[f * 2 + 1] - 1;
      const PetscInt face = f + numCells + numVertices;
      PetscInt       fcone[16];

      // How could Fluent define the outward normal differently? Is there no end to the pain?
      if (dim == 3) {
        if (cl >= 0) PetscCall(InsertFace(*dm, cl, face, -1));
        if (cr >= 0) PetscCall(InsertFace(*dm, cr, face, 0));
      } else {
        if (cl >= 0) PetscCall(InsertFace(*dm, cl, face, 0));
        if (cr >= 0) PetscCall(InsertFace(*dm, cr, face, -1));
      }
      PetscCheck(faceSizes[f] < 16, PETSC_COMM_SELF, PETSC_ERR_SUP, "Number of face vertices %" PetscInt_FMT " exceeds temporary storage", faceSizes[f]);
      for (PetscInt v = 0; v < faceSizes[f]; ++v) fcone[v] = faces[foffset + v] + numCells - 1;
      foffset += faceSizes[f];
      PetscCall(DMPlexSetCone(*dm, face, fcone));
    }
  }
  PetscCall(DMPlexSymmetrize(*dm));
  PetscCall(DMPlexStratify(*dm));
  if (dim == 3) {
    DM idm;

    PetscCall(DMCreate(PetscObjectComm((PetscObject)*dm), &idm));
    PetscCall(DMSetType(idm, DMPLEX));
    PetscCall(DMSetDimension(idm, dim));
    PetscCall(DMPlexInterpolateFaces_Internal(*dm, 1, idm));
    PetscCall(DMDestroy(dm));
    *dm = idm;
  }
  PetscCall(DMViewFromOptions(*dm, NULL, "-cas_dm_view"));
  if (rank == 0 && faces) {
    for (PetscInt c = 0; c < numCells; ++c) PetscCall(ReorderCell(viewer, *dm, c, cellTypes[c]));
  }

  if (rank == 0 && faces) {
    PetscInt        joinSize, meetSize, *fverts, cells[2];
    const PetscInt *join, *meet;
    PetscInt        foffset = 0;

    PetscCall(PetscMalloc1(maxFaceSize, &fverts));
    /* Mark facets by finding the full join of all adjacent vertices */
    for (f = 0; f < numFaces; f++) {
      const PetscInt cl = faceAdjCell[f * 2 + 0] - 1;
      const PetscInt cr = faceAdjCell[f * 2 + 1] - 1;
      const PetscInt id = (PetscInt)faceZoneIDs[f];

      if (cl > 0 && cr > 0) {
        /* If we know both adjoining cells we can use a single-level meet */
        cells[0] = cl;
        cells[1] = cr;
        PetscCall(DMPlexGetMeet(*dm, 2, cells, &meetSize, &meet));
        PetscCheck(meetSize == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not determine Plex facet for Fluent face %" PetscInt_FMT " cells: %" PetscInt_FMT ", %" PetscInt_FMT, f, cl, cr);
        PetscCall(DMSetLabelValue_Fast(*dm, &faceSets, "Face Sets", meet[0], id));
        if (zoneNames && zoneNames[id]) PetscCall(DMSetLabelValue_Fast(*dm, &zoneLabels[id], zoneNames[id], meet[0], 1));
        PetscCall(DMPlexRestoreMeet(*dm, meetSize, fverts, &meetSize, &meet));
      } else {
        for (PetscInt fi = 0; fi < faceSizes[f]; fi++) fverts[fi] = faces[foffset + fi] + numCells - 1;
        PetscCall(DMPlexGetFullJoin(*dm, faceSizes[f], fverts, &joinSize, &join));
        PetscCheck(joinSize == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not determine Plex facet for Fluent face %" PetscInt_FMT, f);
        PetscCall(DMSetLabelValue_Fast(*dm, &faceSets, "Face Sets", join[0], id));
        if (zoneNames && zoneNames[id]) PetscCall(DMSetLabelValue_Fast(*dm, &zoneLabels[id], zoneNames[id], join[0], 1));
        PetscCall(DMPlexRestoreJoin(*dm, joinSize, fverts, &joinSize, &join));
      }
      foffset += faceSizes[f];
    }
    PetscCall(PetscFree(fverts));
  }

  { /* Create Face Sets label at all processes */
    enum {
      n = 1
    };
    PetscBool flag[n];

    flag[0] = faceSets ? PETSC_TRUE : PETSC_FALSE;
    PetscCallMPI(MPI_Bcast(flag, n, MPI_C_BOOL, 0, comm));
    if (flag[0]) PetscCall(DMCreateLabel(*dm, "Face Sets"));
    // TODO Code to create all the zone labels on each process
  }

  if (!interpolate) {
    DM udm;

    PetscCall(DMPlexUninterpolate(*dm, &udm));
    PetscCall(DMDestroy(dm));
    *dm = udm;
  }

  /* Read coordinates */
  PetscCall(DMGetCoordinateSection(*dm, &coordSection));
  PetscCall(PetscSectionSetNumFields(coordSection, 1));
  PetscCall(PetscSectionSetFieldComponents(coordSection, 0, dim));
  PetscCall(PetscSectionSetChart(coordSection, numCells, numCells + numVertices));
  for (PetscInt v = numCells; v < numCells + numVertices; ++v) {
    PetscCall(PetscSectionSetDof(coordSection, v, dim));
    PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, dim));
  }
  PetscCall(PetscSectionSetUp(coordSection));
  PetscCall(PetscSectionGetStorageSize(coordSection, &coordSize));
  PetscCall(VecCreate(PETSC_COMM_SELF, &coordinates));
  PetscCall(PetscObjectSetName((PetscObject)coordinates, "coordinates"));
  PetscCall(VecSetSizes(coordinates, coordSize, PETSC_DETERMINE));
  PetscCall(VecSetType(coordinates, VECSTANDARD));
  PetscCall(VecGetArray(coordinates, &coords));
  if (rank == 0 && coordsIn) {
    for (PetscInt v = 0; v < numVertices; ++v) {
      for (PetscInt d = 0; d < dim; ++d) coords[v * dim + d] = coordsIn[v * dim + d];
    }
  }
  PetscCall(VecRestoreArray(coordinates, &coords));
  PetscCall(DMSetCoordinatesLocal(*dm, coordinates));
  PetscCall(VecDestroy(&coordinates));

  if (rank == 0) {
    PetscCall(PetscFree(cellVertices));
    PetscCall(PetscFree2(cellSizes, cellTypes));
    PetscCall(PetscFree(faces));
    PetscCall(PetscFree3(faceSizes, faceAdjCell, faceZoneIDs));
    PetscCall(PetscFree(coordsIn));
    if (zoneNames)
      for (PetscInt i = 0; i < (PetscInt)maxZoneID; ++i) PetscCall(PetscFree(zoneNames[i]));
    PetscCall(PetscFree2(zoneNames, zoneLabels));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
