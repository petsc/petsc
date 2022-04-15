#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

/*@C
  DMPlexCreatePLYFromFile - Create a DMPlex mesh from a PLY file.

+ comm        - The MPI communicator
. filename    - Name of the .med file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: https://en.wikipedia.org/wiki/PLY_(file_format)

  Level: beginner

.seealso: DMPlexCreateFromFile(), DMPlexCreateMedFromFile(), DMPlexCreateGmsh(), DMPlexCreate()
@*/
PetscErrorCode DMPlexCreatePLYFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscViewer     viewer;
  Vec             coordinates;
  PetscSection    coordSection;
  PetscScalar    *coords;
  char            line[PETSC_MAX_PATH_LEN], ntype[16], itype[16], name[1024], vtype[16];
  PetscBool       match, byteSwap = PETSC_FALSE;
  PetscInt        dim = 2, cdim = 3, Nvp = 0, coordSize, xi = -1, yi = -1, zi = -1, v, c, p;
  PetscMPIInt     rank;
  int             snum, Nv, Nc;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscViewerCreate(comm, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_READ));
  PetscCall(PetscViewerFileSetName(viewer, filename));
  if (rank == 0) {
    PetscBool isAscii, isBinaryBig, isBinaryLittle;

    /* Check for PLY file */
    PetscCall(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    PetscCall(PetscStrncmp(line, "ply", PETSC_MAX_PATH_LEN, &match));
    PetscCheck(match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid PLY file");
    /* Check PLY format */
    PetscCall(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    PetscCall(PetscStrncmp(line, "format", PETSC_MAX_PATH_LEN, &match));
    PetscCheck(match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid PLY file");
    PetscCall(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    PetscCall(PetscStrncmp(line, "ascii", PETSC_MAX_PATH_LEN, &isAscii));
    PetscCall(PetscStrncmp(line, "binary_big_endian", PETSC_MAX_PATH_LEN, &isBinaryBig));
    PetscCall(PetscStrncmp(line, "binary_little_endian", PETSC_MAX_PATH_LEN, &isBinaryLittle));
    PetscCheck(!isAscii,PETSC_COMM_SELF, PETSC_ERR_SUP, "PLY ascii format not yet supported");
    else if (isBinaryLittle) byteSwap = PETSC_TRUE;
    PetscCall(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    PetscCall(PetscStrncmp(line, "1.0", PETSC_MAX_PATH_LEN, &match));
    PetscCheck(match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid version of PLY file, %s", line);
    /* Ignore comments */
    match = PETSC_TRUE;
    while (match) {
      char buf = '\0';
      PetscCall(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
      PetscCall(PetscStrncmp(line, "comment", PETSC_MAX_PATH_LEN, &match));
      if (match) while (buf != '\n') PetscCall(PetscViewerRead(viewer, &buf, 1, NULL, PETSC_CHAR));
    }
    /* Read vertex information */
    PetscCall(PetscStrncmp(line, "element", PETSC_MAX_PATH_LEN, &match));
    PetscCheck(match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    PetscCall(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    PetscCall(PetscStrncmp(line, "vertex", PETSC_MAX_PATH_LEN, &match));
    PetscCheck(match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    PetscCall(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    snum = sscanf(line, "%d", &Nv);
    PetscCheck(snum == 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    match = PETSC_TRUE;
    while (match) {
      PetscCall(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
      PetscCall(PetscStrncmp(line, "property", PETSC_MAX_PATH_LEN, &match));
      if (match) {
        PetscBool matchB;

        PetscCall(PetscViewerRead(viewer, line, 2, NULL, PETSC_STRING));
        PetscCheck(Nvp < 16,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle more than 16 property statements in PLY file header: %s", line);
        snum = sscanf(line, "%s %s", ntype, name);
        PetscCheck(snum == 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
        PetscCall(PetscStrncmp(ntype, "float32", 16, &matchB));
        if (matchB) {
          vtype[Nvp] = 'f';
        } else {
          PetscCall(PetscStrncmp(ntype, "int32", 16, &matchB));
          if (matchB) {
            vtype[Nvp] = 'd';
          } else {
            PetscCall(PetscStrncmp(ntype, "uint8", 16, &matchB));
            if (matchB) {
              vtype[Nvp] = 'c';
            } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse type in PLY file header: %s", line);
          }
        }
        PetscCall(PetscStrncmp(name, "x", 16, &matchB));
        if (matchB) {xi = Nvp;}
        PetscCall(PetscStrncmp(name, "y", 16, &matchB));
        if (matchB) {yi = Nvp;}
        PetscCall(PetscStrncmp(name, "z", 16, &matchB));
        if (matchB) {zi = Nvp;}
        ++Nvp;
      }
    }
    /* Read cell information */
    PetscCall(PetscStrncmp(line, "element", PETSC_MAX_PATH_LEN, &match));
    PetscCheck(match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    PetscCall(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    PetscCall(PetscStrncmp(line, "face", PETSC_MAX_PATH_LEN, &match));
    PetscCheck(match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    PetscCall(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    snum = sscanf(line, "%d", &Nc);
    PetscCheck(snum == 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    PetscCall(PetscViewerRead(viewer, line, 5, NULL, PETSC_STRING));
    snum = sscanf(line, "property list %s %s %s", ntype, itype, name);
    PetscCheck(snum == 3,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    PetscCall(PetscStrncmp(ntype, "uint8", 1024, &match));
    PetscCheck(match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid size type in PLY file header: %s", line);
    PetscCall(PetscStrncmp(name, "vertex_indices", 1024, &match));
    PetscCheck(match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid property in PLY file header: %s", line);
    /* Header should terminate */
    PetscCall(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    PetscCall(PetscStrncmp(line, "end_header", PETSC_MAX_PATH_LEN, &match));
    PetscCheck(match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid PLY file: %s", line);
  } else {
    Nc = Nv = 0;
  }
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMPlexSetChart(*dm, 0, Nc+Nv));
  PetscCall(DMSetDimension(*dm, dim));
  PetscCall(DMSetCoordinateDim(*dm, cdim));
  /* Read coordinates */
  PetscCall(DMGetCoordinateSection(*dm, &coordSection));
  PetscCall(PetscSectionSetNumFields(coordSection, 1));
  PetscCall(PetscSectionSetFieldComponents(coordSection, 0, cdim));
  PetscCall(PetscSectionSetChart(coordSection, Nc, Nc + Nv));
  for (v = Nc; v < Nc+Nv; ++v) {
    PetscCall(PetscSectionSetDof(coordSection, v, cdim));
    PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, cdim));
  }
  PetscCall(PetscSectionSetUp(coordSection));
  PetscCall(PetscSectionGetStorageSize(coordSection, &coordSize));
  PetscCall(VecCreate(PETSC_COMM_SELF, &coordinates));
  PetscCall(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
  PetscCall(VecSetSizes(coordinates, coordSize, PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(coordinates, cdim));
  PetscCall(VecSetType(coordinates, VECSTANDARD));
  PetscCall(VecGetArray(coordinates, &coords));
  if (rank == 0) {
    float rbuf[1];
    int   ibuf[1];

    for (v = 0; v < Nv; ++v) {
      for (p = 0; p < Nvp; ++p) {
        if (vtype[p] == 'f') {
          PetscCall(PetscViewerRead(viewer, &rbuf, 1, NULL, PETSC_FLOAT));
          if (byteSwap) PetscCall(PetscByteSwap(&rbuf, PETSC_FLOAT, 1));
          if      (p == xi) coords[v*cdim+0] = rbuf[0];
          else if (p == yi) coords[v*cdim+1] = rbuf[0];
          else if (p == zi) coords[v*cdim+2] = rbuf[0];
        } else if (vtype[p] == 'd') {
          PetscCall(PetscViewerRead(viewer, &ibuf, 1, NULL, PETSC_INT));
          if (byteSwap) PetscCall(PetscByteSwap(&ibuf, PETSC_INT, 1));
        } else if (vtype[p] == 'c') {
          PetscCall(PetscViewerRead(viewer, &ibuf, 1, NULL, PETSC_CHAR));
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid vertex property type in PLY file");
      }
    }
  }
  PetscCall(VecRestoreArray(coordinates, &coords));
  PetscCall(DMSetCoordinatesLocal(*dm, coordinates));
  PetscCall(VecDestroy(&coordinates));
  /* Read topology */
  if (rank == 0) {
    char     ibuf[1];
    PetscInt vbuf[16], corners;

    /* Assume same cells */
    PetscCall(PetscViewerRead(viewer, &ibuf, 1, NULL, PETSC_CHAR));
    corners = ibuf[0];
    for (c = 0; c < Nc; ++c) PetscCall(DMPlexSetConeSize(*dm, c, corners));
    PetscCall(DMSetUp(*dm));
    for (c = 0; c < Nc; ++c) {
      if (c > 0) {
        PetscCall(PetscViewerRead(viewer, &ibuf, 1, NULL, PETSC_CHAR));
      }
      PetscCheck(ibuf[0] == corners,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "All cells must have the same number of vertices in PLY file: %" PetscInt_FMT " != %" PetscInt_FMT, (PetscInt)ibuf[0], corners);
      PetscCall(PetscViewerRead(viewer, &vbuf, ibuf[0], NULL, PETSC_INT));
      if (byteSwap) PetscCall(PetscByteSwap(&vbuf, PETSC_INT, ibuf[0]));
      for (v = 0; v < ibuf[0]; ++v) vbuf[v] += Nc;
      PetscCall(DMPlexSetCone(*dm, c, vbuf));
    }
  }
  PetscCall(DMPlexSymmetrize(*dm));
  PetscCall(DMPlexStratify(*dm));
  PetscCall(PetscViewerDestroy(&viewer));
  if (interpolate) {
    DM idm;

    PetscCall(DMPlexInterpolate(*dm, &idm));
    PetscCall(DMDestroy(dm));
    *dm  = idm;
  }
  PetscFunctionReturn(0);
}
