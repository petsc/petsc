#include <petsc-private/fortranimpl.h>
#include <petscdmplex.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmplexdistribute_          DMPLEXDISTRIBUTE
#define dmplexhaslabel_            DMPLEXHASLABEL
#define dmplexgetlabelvalue_       DMPLEXGETLABELVALUE
#define dmplexsetlabelvalue_       DMPLEXSETLABELVALUE
#define dmplexgetlabelsize_        DMPLEXGETLABELSIZE
#define dmplexgetlabelidis_        DMPLEXGETLABELIDIS
#define dmplexgetstratumsize_      DMPLEXGETSTRATUMSIZE
#define dmplexgetstratumis_        DMPLEXGETSTRATUMIS
#define dmplexcreatefromcelllist_  DMPLEXCREATEFROMCELLLIST
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmplexdistribute_          dmplexdistribute
#define dmplexhaslabel_            dmplexhaslabel
#define dmplexgetlabelvalue_       dmplexgetlabelvalue
#define dmplexsetlabelvalue_       dmplexsetlabelvalue
#define dmplexgetlabelsize_        dmplexlabelsize
#define dmplexgetlabelidis_        dmplexlabelidis
#define dmplexgetstratumsize_      dmplexgetstratumsize
#define dmplexgetstratumis_        dmplexgetstratumis
#define dmplexcreatefromcelllist_  dmplexcreatefromcelllist
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void PETSC_STDCALL dmplexdistribute_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *overlap, PetscSF *sf, DM *dmParallel, int *ierr PETSC_END_LEN(lenN))
{
  char *partitioner;

  FIXCHAR(name, lenN, partitioner);
  *ierr = DMPlexDistribute(*dm, partitioner, *overlap, sf, dmParallel);
  FREECHAR(name, partitioner);
}

PETSC_EXTERN void PETSC_STDCALL dmplexhaslabel_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscBool *hasLabel, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMPlexHasLabel(*dm, lname, hasLabel);
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetlabelvalue_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *point, PetscInt *value, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMPlexGetLabelValue(*dm, lname, *point, value);
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmplexsetlabelvalue_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *point, PetscInt *value, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMPlexSetLabelValue(*dm, lname, *point, *value);
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetlabelsize_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *size, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMPlexGetLabelSize(*dm, lname, size);
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetlabelidis_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), IS *ids, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMPlexGetLabelIdIS(*dm, lname, ids);
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetstratumsize_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *value, PetscInt *size, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMPlexGetStratumSize(*dm, lname, *value, size);
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetstratumis_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *value, IS *is, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMPlexGetStratumIS(*dm, lname, *value, is);
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmplexcreatefromcelllist_(MPI_Comm *comm, PetscInt *dim, PetscInt *numCells, PetscInt *numVertices, PetscInt *numCorners, PetscBool *interpolate, const int cells[], PetscInt *spaceDim, const double vertexCoords[], DM *dm, int *ierr)
{
  *ierr = DMPlexCreateFromCellList(*comm, *dim, *numCells, *numVertices, *numCorners, *interpolate, cells, *spaceDim, vertexCoords, dm);
}
