#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

#if defined(PETSC_HAVE_MED)
#include <med.h>
#endif

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateMedFromFile"
/*@C
  DMPlexCreateMedFromFile - Create a DMPlex mesh from a (Salome-)Med file.

+ comm        - The MPI communicator
. filename    - Name of the .med file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: http://www.code-aster.org/outils/med/html/index.html

  Level: beginner

.seealso: DMPlexCreateFromFile(), DMPlexCreateGmsh(), DMPlexCreate()
@*/
PetscErrorCode DMPlexCreateMedFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscMPIInt     rank;
  PetscInt        fid;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);

#if defined(PETSC_HAVE_MED)
  fid = MEDfileOpen(filename, MED_ACC_RDONLY);
  if (fid < 0) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Unable to open .med mesh file: %s", filename);

  ierr = MEDfileClose(fid);CHKERRQ(ierr);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires Med mesh reader support. Reconfigure using --download-med");
#endif
  PetscFunctionReturn(0);
}
