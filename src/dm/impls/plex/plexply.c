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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
  if (rank == 0) {
    PetscBool isAscii, isBinaryBig, isBinaryLittle;

    /* Check for PLY file */
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "ply", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid PLY file");
    /* Check PLY format */
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "format", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid PLY file");
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "ascii", PETSC_MAX_PATH_LEN, &isAscii);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "binary_big_endian", PETSC_MAX_PATH_LEN, &isBinaryBig);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "binary_little_endian", PETSC_MAX_PATH_LEN, &isBinaryLittle);CHKERRQ(ierr);
    PetscCheckFalse(isAscii,PETSC_COMM_SELF, PETSC_ERR_SUP, "PLY ascii format not yet supported");
    else if (isBinaryLittle) byteSwap = PETSC_TRUE;
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "1.0", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid version of PLY file, %s", line);
    /* Ignore comments */
    match = PETSC_TRUE;
    while (match) {
      char buf = '\0';
      ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
      ierr = PetscStrncmp(line, "comment", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
      if (match) while (buf != '\n') {ierr = PetscViewerRead(viewer, &buf, 1, NULL, PETSC_CHAR);CHKERRQ(ierr);}
    }
    /* Read vertex information */
    ierr = PetscStrncmp(line, "element", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "vertex", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    snum = sscanf(line, "%d", &Nv);
    PetscCheckFalse(snum != 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    match = PETSC_TRUE;
    while (match) {
      ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
      ierr = PetscStrncmp(line, "property", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
      if (match) {
        PetscBool matchB;

        ierr = PetscViewerRead(viewer, line, 2, NULL, PETSC_STRING);CHKERRQ(ierr);
        PetscCheckFalse(Nvp >= 16,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle more than 16 property statements in PLY file header: %s", line);
        snum = sscanf(line, "%s %s", ntype, name);
        PetscCheckFalse(snum != 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
        ierr = PetscStrncmp(ntype, "float32", 16, &matchB);CHKERRQ(ierr);
        if (matchB) {
          vtype[Nvp] = 'f';
        } else {
          ierr = PetscStrncmp(ntype, "int32", 16, &matchB);CHKERRQ(ierr);
          if (matchB) {
            vtype[Nvp] = 'd';
          } else {
            ierr = PetscStrncmp(ntype, "uint8", 16, &matchB);CHKERRQ(ierr);
            if (matchB) {
              vtype[Nvp] = 'c';
            } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse type in PLY file header: %s", line);
          }
        }
        ierr = PetscStrncmp(name, "x", 16, &matchB);CHKERRQ(ierr);
        if (matchB) {xi = Nvp;}
        ierr = PetscStrncmp(name, "y", 16, &matchB);CHKERRQ(ierr);
        if (matchB) {yi = Nvp;}
        ierr = PetscStrncmp(name, "z", 16, &matchB);CHKERRQ(ierr);
        if (matchB) {zi = Nvp;}
        ++Nvp;
      }
    }
    /* Read cell information */
    ierr = PetscStrncmp(line, "element", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "face", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    snum = sscanf(line, "%d", &Nc);
    PetscCheckFalse(snum != 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    ierr = PetscViewerRead(viewer, line, 5, NULL, PETSC_STRING);CHKERRQ(ierr);
    snum = sscanf(line, "property list %s %s %s", ntype, itype, name);
    PetscCheckFalse(snum != 3,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse PLY file header: %s", line);
    ierr = PetscStrncmp(ntype, "uint8", 1024, &match);CHKERRQ(ierr);
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid size type in PLY file header: %s", line);
    ierr = PetscStrncmp(name, "vertex_indices", 1024, &match);CHKERRQ(ierr);
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid property in PLY file header: %s", line);
    /* Header should terminate */
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "end_header", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    PetscCheckFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid PLY file: %s", line);
  } else {
    Nc = Nv = 0;
  }
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetChart(*dm, 0, Nc+Nv);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(*dm, cdim);CHKERRQ(ierr);
  /* Read coordinates */
  ierr = DMGetCoordinateSection(*dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, cdim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, Nc, Nc + Nv);CHKERRQ(ierr);
  for (v = Nc; v < Nc+Nv; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, cdim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, cdim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, cdim);CHKERRQ(ierr);
  ierr = VecSetType(coordinates, VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  if (rank == 0) {
    float rbuf[1];
    int   ibuf[1];

    for (v = 0; v < Nv; ++v) {
      for (p = 0; p < Nvp; ++p) {
        if (vtype[p] == 'f') {
          ierr = PetscViewerRead(viewer, &rbuf, 1, NULL, PETSC_FLOAT);CHKERRQ(ierr);
          if (byteSwap) {ierr = PetscByteSwap(&rbuf, PETSC_FLOAT, 1);CHKERRQ(ierr);}
          if      (p == xi) coords[v*cdim+0] = rbuf[0];
          else if (p == yi) coords[v*cdim+1] = rbuf[0];
          else if (p == zi) coords[v*cdim+2] = rbuf[0];
        } else if (vtype[p] == 'd') {
          ierr = PetscViewerRead(viewer, &ibuf, 1, NULL, PETSC_INT);CHKERRQ(ierr);
          if (byteSwap) {ierr = PetscByteSwap(&ibuf, PETSC_INT, 1);CHKERRQ(ierr);}
        } else if (vtype[p] == 'c') {
          ierr = PetscViewerRead(viewer, &ibuf, 1, NULL, PETSC_CHAR);CHKERRQ(ierr);
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid vertex property type in PLY file");
      }
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  /* Read topology */
  if (rank == 0) {
    char     ibuf[1];
    PetscInt vbuf[16], corners;

    /* Assume same cells */
    ierr = PetscViewerRead(viewer, &ibuf, 1, NULL, PETSC_CHAR);CHKERRQ(ierr);
    corners = ibuf[0];
    for (c = 0; c < Nc; ++c) {ierr = DMPlexSetConeSize(*dm, c, corners);CHKERRQ(ierr);}
    ierr = DMSetUp(*dm);CHKERRQ(ierr);
    for (c = 0; c < Nc; ++c) {
      if (c > 0) {
        ierr = PetscViewerRead(viewer, &ibuf, 1, NULL, PETSC_CHAR);CHKERRQ(ierr);
      }
      PetscCheckFalse(ibuf[0] != corners,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "All cells must have the same number of vertices in PLY file: %D != %D", ibuf[0], corners);
      ierr = PetscViewerRead(viewer, &vbuf, ibuf[0], NULL, PETSC_INT);CHKERRQ(ierr);
      if (byteSwap) {ierr = PetscByteSwap(&vbuf, PETSC_INT, ibuf[0]);CHKERRQ(ierr);}
      for (v = 0; v < ibuf[0]; ++v) vbuf[v] += Nc;
      ierr = DMPlexSetCone(*dm, c, vbuf);CHKERRQ(ierr);
    }
  }
  ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(*dm);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  if (interpolate) {
    DM idm;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }
  PetscFunctionReturn(0);
}
