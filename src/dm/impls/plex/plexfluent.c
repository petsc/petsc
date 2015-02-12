#define PETSCDM_DLL
#include <petsc-private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateFluentFromFile"
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Create file viewer and build plex */
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
  ierr = DMPlexCreateFluent(comm, viewer, interpolate, dm);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateFluent_ReadString"
PetscErrorCode DMPlexCreateFluent_ReadString(PetscViewer viewer, char *buffer, char delim)
{
  PetscInt i = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  do {ierr = PetscViewerRead(viewer, &(buffer[i++]), 1, PETSC_CHAR);CHKERRQ(ierr);}
  while (buffer[i-1] != '\0' && buffer[i-1] != delim);
  buffer[i] = '\0';
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateFluent_ReadSection"
PetscErrorCode DMPlexCreateFluent_ReadSection(PetscViewer viewer, FluentSection *s)
{
  char           buffer[PETSC_MAX_PATH_LEN];
  int            snum;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Fast-forward to next section and derive its index */
  ierr = DMPlexCreateFluent_ReadString(viewer, buffer, '(');CHKERRQ(ierr);
  ierr = DMPlexCreateFluent_ReadString(viewer, buffer, ' ');CHKERRQ(ierr);
  snum = sscanf(buffer, "%d", &(s->index));
  /* If we can't match an index return -1 to signal end-of-file */
  if (snum < 1) {s->index = -1;   PetscFunctionReturn(0);}

  if (s->index == 0) {           /* Comment */
    ierr = DMPlexCreateFluent_ReadString(viewer, buffer, ')');CHKERRQ(ierr);

  } else if (s->index == 2) {    /* Dimension */
    ierr = DMPlexCreateFluent_ReadString(viewer, buffer, ')');CHKERRQ(ierr);
    snum = sscanf(buffer, "%d", &(s->nd));
    if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");

  } else if (s->index == 10) {   /* Vertices */
    ierr = DMPlexCreateFluent_ReadString(viewer, buffer, ')');CHKERRQ(ierr);
    snum = sscanf(buffer, "(%x %x %x %d %d)", &(s->zoneID), &(s->first), &(s->last), &(s->type), &(s->nd));
    if (snum != 5) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    ierr = DMPlexCreateFluent_ReadString(viewer, buffer, ')');CHKERRQ(ierr);

  } else if (s->index == 12) {   /* Cells */
    ierr = DMPlexCreateFluent_ReadString(viewer, buffer, ')');CHKERRQ(ierr);
    snum = sscanf(buffer, "(%x", &(s->zoneID));
    if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    if (s->zoneID == 0) {  /* Header section */
      snum = sscanf(buffer, "(%x %x %x %d)", &(s->zoneID), &(s->first), &(s->last), &(s->nd));
      if (snum != 4) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    } else {
      snum = sscanf(buffer, "(%x %x %x %d %d)", &(s->zoneID), &(s->first), &(s->last), &(s->type), &(s->nd));
      if (snum != 5) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    }
    ierr = DMPlexCreateFluent_ReadString(viewer, buffer, ')');CHKERRQ(ierr);

  } else if (s->index == 13) {   /* Faces */
    ierr = DMPlexCreateFluent_ReadString(viewer, buffer, ')');CHKERRQ(ierr);
    snum = sscanf(buffer, "(%x", &(s->zoneID));
    if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    if (s->zoneID == 0) {  /* Header section */
      snum = sscanf(buffer, "(%x %x %x %d)", &(s->zoneID), &(s->first), &(s->last), &(s->nd));
      if (snum != 4) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    } else {
      snum = sscanf(buffer, "(%x %x %x %d %d)", &(s->zoneID), &(s->first), &(s->last), &(s->type), &(s->nd));
      if (snum != 5) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Fluent file");
    }
    ierr = DMPlexCreateFluent_ReadString(viewer, buffer, ')');CHKERRQ(ierr);

  } else {                       /* Unknown section type */
    PetscInt depth = 1;
    do {
      /* Match parentheses when parsing unknown sections */
      do {ierr = PetscViewerRead(viewer, &(buffer[0]), 1, PETSC_CHAR);CHKERRQ(ierr);}
      while (buffer[0] != '(' && buffer[0] != ')');
      if (buffer[0] == '(') depth++;
      if (buffer[0] == ')') depth--;
    } while (depth > 0);
    ierr = DMPlexCreateFluent_ReadString(viewer, buffer, '\n');CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateFluent"
/*@C
  DMPlexCreateFluent - Create a DMPlex mesh from a Fluent mesh file.

  Collective on comm

  Input Parameters:
+ comm  - The MPI communicator
. viewer - The Viewer associated with a Fluent mesh file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: http://aerojet.engr.ucdavis.edu/fluenthelp/html/ug/node1490.htm

  Level: beginner

.keywords: mesh, fluent, case
.seealso: DMPLEX, DMCreate()
@*/
PetscErrorCode DMPlexCreateFluent(MPI_Comm comm, PetscViewer viewer, PetscBool interpolate, DM *dm)
{
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);

  if (!rank) {
    FluentSection s;
    do {
      ierr = DMPlexCreateFluent_ReadSection(viewer, &s);CHKERRQ(ierr);
      if (s.index == 2) {                 /* Dimension */

      } else if (s.index == 10) {         /* Vertices */

      } else if (s.index == 12) {         /* Cells */

      } else if (s.index == 13) {         /* Facets */

      }
    } while (s.index >= 0);
  }
  PetscFunctionReturn(0);
}
