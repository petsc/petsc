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
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(PetscViewerCreate(comm, &viewer));
  CHKERRQ(PetscViewerSetType(viewer, PETSCVIEWERBINARY));
  CHKERRQ(PetscViewerFileSetMode(viewer, FILE_MODE_READ));
  CHKERRQ(PetscViewerFileSetName(viewer, filename));
  if (rank == 0) {
    PetscBool isAscii, isBinaryBig, isBinaryLittle;

    /* Check for PLY file */
    CHKERRQ(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    CHKERRQ(PetscStrncmp(line, "ply", PETSC_MAX_PATH_LEN, &match));
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid PLY file");
    /* Check PLY format */
    CHKERRQ(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    CHKERRQ(PetscStrncmp(line, "format", PETSC_MAX_PATH_LEN, &match));
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid PLY file");
    CHKERRQ(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    CHKERRQ(PetscStrncmp(line, "ascii", PETSC_MAX_PATH_LEN, &isAscii));
    CHKERRQ(PetscStrncmp(line, "binary_big_endian", PETSC_MAX_PATH_LEN, &isBinaryBig));
    CHKERRQ(PetscStrncmp(line, "binary_little_endian", PETSC_MAX_PATH_LEN, &isBinaryLittle));
    PetscCheckFalse(isAscii,PETSC_COMM_SELF, PETSC_ERR_SUP, "PLY ascii format not yet supported");
    else if (isBinaryLittle) byteSwap = PETSC_TRUE;
    CHKERRQ(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    CHKERRQ(PetscStrncmp(line, "1.0", PETSC_MAX_PATH_LEN, &match));
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid version of PLY file, %s", line);
    /* Ignore comments */
    match = PETSC_TRUE;
    while (match) {
      char buf = '\0';
      CHKERRQ(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
      CHKERRQ(PetscStrncmp(line, "comment", PETSC_MAX_PATH_LEN, &match));
      if (match) while (buf != '\n') CHKERRQ(PetscViewerRead(viewer, &buf, 1, NULL, PETSC_CHAR));
    }
    /* Read vertex information */
    CHKERRQ(PetscStrncmp(line, "element", PETSC_MAX_PATH_LEN, &match));
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    CHKERRQ(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    CHKERRQ(PetscStrncmp(line, "vertex", PETSC_MAX_PATH_LEN, &match));
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    CHKERRQ(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    snum = sscanf(line, "%d", &Nv);
    PetscCheckFalse(snum != 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    match = PETSC_TRUE;
    while (match) {
      CHKERRQ(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
      CHKERRQ(PetscStrncmp(line, "property", PETSC_MAX_PATH_LEN, &match));
      if (match) {
        PetscBool matchB;

        CHKERRQ(PetscViewerRead(viewer, line, 2, NULL, PETSC_STRING));
        PetscCheckFalse(Nvp >= 16,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle more than 16 property statements in PLY file header: %s", line);
        snum = sscanf(line, "%s %s", ntype, name);
        PetscCheckFalse(snum != 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
        CHKERRQ(PetscStrncmp(ntype, "float32", 16, &matchB));
        if (matchB) {
          vtype[Nvp] = 'f';
        } else {
          CHKERRQ(PetscStrncmp(ntype, "int32", 16, &matchB));
          if (matchB) {
            vtype[Nvp] = 'd';
          } else {
            CHKERRQ(PetscStrncmp(ntype, "uint8", 16, &matchB));
            if (matchB) {
              vtype[Nvp] = 'c';
            } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse type in PLY file header: %s", line);
          }
        }
        CHKERRQ(PetscStrncmp(name, "x", 16, &matchB));
        if (matchB) {xi = Nvp;}
        CHKERRQ(PetscStrncmp(name, "y", 16, &matchB));
        if (matchB) {yi = Nvp;}
        CHKERRQ(PetscStrncmp(name, "z", 16, &matchB));
        if (matchB) {zi = Nvp;}
        ++Nvp;
      }
    }
    /* Read cell information */
    CHKERRQ(PetscStrncmp(line, "element", PETSC_MAX_PATH_LEN, &match));
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    CHKERRQ(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    CHKERRQ(PetscStrncmp(line, "face", PETSC_MAX_PATH_LEN, &match));
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    CHKERRQ(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    snum = sscanf(line, "%d", &Nc);
    PetscCheckFalse(snum != 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    CHKERRQ(PetscViewerRead(viewer, line, 5, NULL, PETSC_STRING));
    snum = sscanf(line, "property list %s %s %s", ntype, itype, name);
    PetscCheckFalse(snum != 3,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    CHKERRQ(PetscStrncmp(ntype, "uint8", 1024, &match));
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid size type in PLY file header: %s", line);
    CHKERRQ(PetscStrncmp(name, "vertex_indices", 1024, &match));
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid property in PLY file header: %s", line);
    /* Header should terminate */
    CHKERRQ(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    CHKERRQ(PetscStrncmp(line, "end_header", PETSC_MAX_PATH_LEN, &match));
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid PLY file: %s", line);
  } else {
    Nc = Nv = 0;
  }
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMPlexSetChart(*dm, 0, Nc+Nv));
  CHKERRQ(DMSetDimension(*dm, dim));
  CHKERRQ(DMSetCoordinateDim(*dm, cdim));
  /* Read coordinates */
  CHKERRQ(DMGetCoordinateSection(*dm, &coordSection));
  CHKERRQ(PetscSectionSetNumFields(coordSection, 1));
  CHKERRQ(PetscSectionSetFieldComponents(coordSection, 0, cdim));
  CHKERRQ(PetscSectionSetChart(coordSection, Nc, Nc + Nv));
  for (v = Nc; v < Nc+Nv; ++v) {
    CHKERRQ(PetscSectionSetDof(coordSection, v, cdim));
    CHKERRQ(PetscSectionSetFieldDof(coordSection, v, 0, cdim));
  }
  CHKERRQ(PetscSectionSetUp(coordSection));
  CHKERRQ(PetscSectionGetStorageSize(coordSection, &coordSize));
  CHKERRQ(VecCreate(PETSC_COMM_SELF, &coordinates));
  CHKERRQ(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
  CHKERRQ(VecSetSizes(coordinates, coordSize, PETSC_DETERMINE));
  CHKERRQ(VecSetBlockSize(coordinates, cdim));
  CHKERRQ(VecSetType(coordinates, VECSTANDARD));
  CHKERRQ(VecGetArray(coordinates, &coords));
  if (rank == 0) {
    float rbuf[1];
    int   ibuf[1];

    for (v = 0; v < Nv; ++v) {
      for (p = 0; p < Nvp; ++p) {
        if (vtype[p] == 'f') {
          CHKERRQ(PetscViewerRead(viewer, &rbuf, 1, NULL, PETSC_FLOAT));
          if (byteSwap) CHKERRQ(PetscByteSwap(&rbuf, PETSC_FLOAT, 1));
          if      (p == xi) coords[v*cdim+0] = rbuf[0];
          else if (p == yi) coords[v*cdim+1] = rbuf[0];
          else if (p == zi) coords[v*cdim+2] = rbuf[0];
        } else if (vtype[p] == 'd') {
          CHKERRQ(PetscViewerRead(viewer, &ibuf, 1, NULL, PETSC_INT));
          if (byteSwap) CHKERRQ(PetscByteSwap(&ibuf, PETSC_INT, 1));
        } else if (vtype[p] == 'c') {
          CHKERRQ(PetscViewerRead(viewer, &ibuf, 1, NULL, PETSC_CHAR));
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid vertex property type in PLY file");
      }
    }
  }
  CHKERRQ(VecRestoreArray(coordinates, &coords));
  CHKERRQ(DMSetCoordinatesLocal(*dm, coordinates));
  CHKERRQ(VecDestroy(&coordinates));
  /* Read topology */
  if (rank == 0) {
    char     ibuf[1];
    PetscInt vbuf[16], corners;

    /* Assume same cells */
    CHKERRQ(PetscViewerRead(viewer, &ibuf, 1, NULL, PETSC_CHAR));
    corners = ibuf[0];
    for (c = 0; c < Nc; ++c) CHKERRQ(DMPlexSetConeSize(*dm, c, corners));
    CHKERRQ(DMSetUp(*dm));
    for (c = 0; c < Nc; ++c) {
      if (c > 0) {
        CHKERRQ(PetscViewerRead(viewer, &ibuf, 1, NULL, PETSC_CHAR));
      }
      PetscCheckFalse(ibuf[0] != corners,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "All cells must have the same number of vertices in PLY file: %D != %D", ibuf[0], corners);
      CHKERRQ(PetscViewerRead(viewer, &vbuf, ibuf[0], NULL, PETSC_INT));
      if (byteSwap) CHKERRQ(PetscByteSwap(&vbuf, PETSC_INT, ibuf[0]));
      for (v = 0; v < ibuf[0]; ++v) vbuf[v] += Nc;
      CHKERRQ(DMPlexSetCone(*dm, c, vbuf));
    }
  }
  CHKERRQ(DMPlexSymmetrize(*dm));
  CHKERRQ(DMPlexStratify(*dm));
  CHKERRQ(PetscViewerDestroy(&viewer));
  if (interpolate) {
    DM idm;

    CHKERRQ(DMPlexInterpolate(*dm, &idm));
    CHKERRQ(DMDestroy(dm));
    *dm  = idm;
  }
  PetscFunctionReturn(0);
}
