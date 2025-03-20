#pragma once

#include <petscvec.h> /*I      "petscvec.h"    I*/
#include <petscmat.h> /*I      "petscmat.h"    I*/
#include <petscdm.h>  /*I      "petscdm.h"     I*/
#include <petscdt.h>  /*I      "petscdt.h"     I*/

/* MANSEC = DM */
/* SUBMANSEC = DMMOAB */

#include <string>
#include <moab/Core.hpp> /*I      "moab/Core.hpp"    I*/
#ifdef MOAB_HAVE_MPI
  #include <moab/ParallelComm.hpp> /*I      "moab/ParallelComm.hpp"    I*/
#endif

/* The MBERR macro is used to save typing. It checks a MOAB error code
 * (rval) and calls SETERRQ if not MB_SUCCESS. A message (msg) can
 * also be passed in. */
#define MBERR(msg, rval) PetscCheck(rval == moab::MB_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "MOAB ERROR (%i): %s", (PetscErrorCode)rval, msg)
#define MBERRNM(rval)    PetscCheck(rval == moab::MB_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "MOAB ERROR (%i)", rval)
#define MBERRV(mbif, rval) \
  do { \
    if (rval != moab::MB_SUCCESS) { \
      std::string emsg; \
      mbif->get_last_error(emsg); \
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "MOAB ERROR (%i): %s", (PetscErrorCode)rval, emsg.c_str()); \
    } \
  } while (0)
#define MBERRVM(mbif, msg, rval) \
  do { \
    if (rval != moab::MB_SUCCESS) { \
      std::string emsg; \
      mbif->get_last_error(emsg); \
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "MOAB ERROR (%i): %s :: %s", (PetscErrorCode)rval, msg, emsg.c_str()); \
    } \
  } while (0)

/* define enums for options to read and write MOAB files in parallel */
typedef enum {
  READ_PART,
  READ_DELETE,
  BCAST_DELETE
} MoabReadMode;
static const char *const MoabReadModes[] = {"READ_PART", "READ_DELETE", "BCAST_DELETE", "MoabReadMode", "", 0};
typedef enum {
  WRITE_PART,
  FORMAT
} MoabWriteMode;
static const char *const MoabWriteModes[] = {"WRITE_PART", "FORMAT", "MoabWriteMode", "", 0};

PETSC_EXTERN PetscErrorCode DMMoabCreate(MPI_Comm, DM *);
PETSC_EXTERN PetscErrorCode DMMoabCreateMoab(MPI_Comm, moab::Interface *, moab::Tag *, moab::Range *, DM *);
PETSC_EXTERN PetscErrorCode DMMoabOutput(DM, const char *, const char *);

PETSC_EXTERN PetscErrorCode DMMoabSetInterface(DM, moab::Interface *);
PETSC_EXTERN PetscErrorCode DMMoabGetInterface(DM, moab::Interface **);
#ifdef MOAB_HAVE_MPI
PETSC_EXTERN PetscErrorCode DMMoabGetParallelComm(DM, moab::ParallelComm **);
#endif

PETSC_EXTERN PetscErrorCode DMMoabSetLocalVertices(DM, moab::Range *);
PETSC_EXTERN PetscErrorCode DMMoabGetAllVertices(DM, moab::Range *);
PETSC_EXTERN PetscErrorCode DMMoabGetLocalVertices(DM, const moab::Range **, const moab::Range **);
PETSC_EXTERN PetscErrorCode DMMoabSetLocalElements(DM, moab::Range *);
PETSC_EXTERN PetscErrorCode DMMoabGetLocalElements(DM, const moab::Range **);
PETSC_EXTERN PetscErrorCode DMMoabSetLocalToGlobalTag(DM, moab::Tag);
PETSC_EXTERN PetscErrorCode DMMoabGetLocalToGlobalTag(DM, moab::Tag *);
PETSC_EXTERN PetscErrorCode DMMoabSetBlockSize(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMMoabGetBlockSize(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMoabSetBlockFills(DM, const PetscInt *, const PetscInt *);
PETSC_EXTERN PetscErrorCode DMMoabGetHierarchyLevel(DM, PetscInt *);

PETSC_EXTERN PetscErrorCode DMMoabGetDimension(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMoabGetBoundaryEntities(DM, moab::Range *, moab::Range *, moab::Range *);
PETSC_EXTERN PetscErrorCode DMMoabGetMaterialBlock(DM, const moab::EntityHandle, PetscInt *);

PETSC_EXTERN PetscErrorCode DMMoabGetSize(DM, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMoabGetLocalSize(DM, PetscInt *, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMoabGetOffset(DM, PetscInt *);

PETSC_EXTERN PetscErrorCode DMMoabVecGetArrayRead(DM, Vec, void *);
PETSC_EXTERN PetscErrorCode DMMoabVecRestoreArrayRead(DM, Vec, void *);
PETSC_EXTERN PetscErrorCode DMMoabVecGetArray(DM, Vec, void *);
PETSC_EXTERN PetscErrorCode DMMoabVecRestoreArray(DM, Vec, void *);

PETSC_EXTERN PetscErrorCode DMMoabCreateVector(DM, moab::Tag, const moab::Range *, PetscBool, PetscBool, Vec *);
PETSC_EXTERN PetscErrorCode DMMoabGetVecTag(Vec, moab::Tag *);
PETSC_EXTERN PetscErrorCode DMMoabGetVecRange(Vec, moab::Range *);

PETSC_EXTERN PetscErrorCode DMMoabSetFieldVector(DM, PetscInt, Vec);
PETSC_EXTERN PetscErrorCode DMMoabSetGlobalFieldVector(DM, Vec);

PETSC_EXTERN PetscErrorCode DMMoabCreateVertices(DM, const PetscReal *, PetscInt, moab::Range *);
PETSC_EXTERN PetscErrorCode DMMoabCreateElement(DM, const moab::EntityType, const moab::EntityHandle *, PetscInt, moab::EntityHandle *);
PETSC_EXTERN PetscErrorCode DMMoabCreateSubmesh(DM, DM *);
PETSC_EXTERN PetscErrorCode DMMoabRenumberMeshEntities(DM);

PETSC_EXTERN PetscErrorCode DMMoabGetFieldName(DM, PetscInt, const char **);
PETSC_EXTERN PetscErrorCode DMMoabSetFieldName(DM, PetscInt, const char *);
PETSC_EXTERN PetscErrorCode DMMoabSetFieldNames(DM, PetscInt, const char *[]);
PETSC_EXTERN PetscErrorCode DMMoabGetFieldDof(DM, moab::EntityHandle, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMoabGetFieldDofs(DM, PetscInt, const moab::EntityHandle *, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMoabGetFieldDofsLocal(DM, PetscInt, const moab::EntityHandle *, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMoabGetDofs(DM, PetscInt, const moab::EntityHandle *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMoabGetDofsLocal(DM, PetscInt, const moab::EntityHandle *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMoabGetDofsBlocked(DM, PetscInt, const moab::EntityHandle *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMoabGetDofsBlockedLocal(DM, PetscInt, const moab::EntityHandle *, PetscInt *);

PETSC_EXTERN PetscErrorCode DMMoabGetVertexDofsBlocked(DM, PetscInt **);
PETSC_EXTERN PetscErrorCode DMMoabGetVertexDofsBlockedLocal(DM, PetscInt **);

/* discretization and assembly specific DMMoab interface functions */
PETSC_EXTERN PetscErrorCode DMMoabGetElementConnectivity(DM, moab::EntityHandle, PetscInt *, const moab::EntityHandle **);
PETSC_EXTERN PetscErrorCode DMMoabGetVertexConnectivity(DM, moab::EntityHandle, PetscInt *, moab::EntityHandle **);
PETSC_EXTERN PetscErrorCode DMMoabRestoreVertexConnectivity(DM, moab::EntityHandle, PetscInt *, moab::EntityHandle **);
PETSC_EXTERN PetscErrorCode DMMoabGetVertexCoordinates(DM, PetscInt, const moab::EntityHandle *, PetscReal *);
PETSC_EXTERN PetscErrorCode DMMoabIsEntityOnBoundary(DM, const moab::EntityHandle, PetscBool *);
PETSC_EXTERN PetscErrorCode DMMoabCheckBoundaryVertices(DM, PetscInt, const moab::EntityHandle *, PetscBool *);
PETSC_EXTERN PetscErrorCode DMMoabGetBoundaryMarkers(DM, const moab::Range **, const moab::Range **, const moab::Range **);

/* TODO: Replace nverts/coords with just moab::EntityHandle -- can also eliminate dim */
/* TODO: Replace quad/npts with PetscDT */
PETSC_EXTERN PetscErrorCode DMMoabFEMCreateQuadratureDefault(const PetscInt, const PetscInt, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode DMMoabFEMComputeBasis(const PetscInt, const PetscInt, const PetscReal *, const PetscQuadrature, PetscReal *, PetscReal *, PetscReal *, PetscReal **);
PETSC_EXTERN PetscErrorCode DMMoabPToRMapping(const PetscInt, const PetscInt, const PetscReal *, const PetscReal *, PetscReal *, PetscReal *);

/* DM utility creation interface */
PETSC_EXTERN PetscErrorCode DMMoabCreateBoxMesh(MPI_Comm, PetscInt, PetscBool, const PetscReal *, PetscInt, PetscInt, DM *);
PETSC_EXTERN PetscErrorCode DMMoabLoadFromFile(MPI_Comm, PetscInt, PetscInt, const char *, const char *, DM *);

/* Uniform refinement hierarchy interface */
PETSC_EXTERN PetscErrorCode DMMoabGenerateHierarchy(DM, PetscInt, PetscInt *);
