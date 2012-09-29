/*
  DMComplex, for parallel unstructured distributed mesh problems.
*/
#if !defined(__PETSCDMCOMPLEX_H)
#define __PETSCDMCOMPLEX_H

#include <petscsf.h>
#include <petscdm.h>

/*S
  DMCOMPLEX - DM object that encapsulates an unstructured mesh, or CW Complex, which can be expressed using a Hasse Diagram.

  Level: intermediate

  Concepts: grids, grid refinement

.seealso:  DM, DMComplexCreate()
S*/
PETSC_EXTERN PetscErrorCode DMComplexCreate(MPI_Comm, DM*);
PETSC_EXTERN PetscErrorCode DMComplexCreateSubmesh(DM, const char[], DM*);
PETSC_EXTERN PetscErrorCode DMComplexCreateFromCellList(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscBool, const int[], const double[], DM*);
PETSC_EXTERN PetscErrorCode DMComplexClone(DM, DM*);
PETSC_EXTERN PetscErrorCode DMComplexGetDimension(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMComplexSetDimension(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMComplexGetChart(DM, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMComplexSetChart(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMComplexGetConeSize(DM, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMComplexSetConeSize(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMComplexGetCone(DM, PetscInt, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMComplexSetCone(DM, PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode DMComplexGetConeOrientation(DM, PetscInt, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMComplexSetConeOrientation(DM, PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode DMComplexGetSupportSize(DM, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMComplexSetSupportSize(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMComplexGetSupport(DM, PetscInt, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMComplexSetSupport(DM, PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode DMComplexGetConeSection(DM, PetscSection *);
PETSC_EXTERN PetscErrorCode DMComplexGetCones(DM, PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMComplexGetConeOrientations(DM, PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMComplexGetMaxSizes(DM, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMComplexSymmetrize(DM);
PETSC_EXTERN PetscErrorCode DMComplexStratify(DM);
PETSC_EXTERN PetscErrorCode DMComplexGetCoordinateSection(DM, PetscSection *);
PETSC_EXTERN PetscErrorCode DMComplexSetCoordinateSection(DM, PetscSection);

PETSC_EXTERN PetscErrorCode DMComplexHasLabel(DM, const char [], PetscBool *);
PETSC_EXTERN PetscErrorCode DMComplexGetLabelValue(DM, const char[], PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMComplexSetLabelValue(DM, const char[], PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMComplexClearLabelValue(DM, const char[], PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMComplexGetLabelSize(DM, const char[], PetscInt *);
PETSC_EXTERN PetscErrorCode DMComplexGetLabelIdIS(DM, const char[], IS *);
PETSC_EXTERN PetscErrorCode DMComplexGetStratumSize(DM, const char [], PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMComplexGetStratumIS(DM, const char [], PetscInt, IS *);
PETSC_EXTERN PetscErrorCode DMComplexClearLabelStratum(DM, const char[], PetscInt);

PETSC_EXTERN PetscErrorCode DMComplexGetMeet(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMComplexGetFullMeet(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMComplexRestoreMeet(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMComplexGetJoin(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMComplexGetFullJoin(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMComplexRestoreJoin(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMComplexGetTransitiveClosure(DM, PetscInt, PetscBool, PetscInt *, PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMComplexRestoreTransitiveClosure(DM, PetscInt, PetscBool, PetscInt *, PetscInt *[]);

PETSC_EXTERN PetscErrorCode DMComplexCreatePartition(DM, PetscSection *, IS *, PetscInt);
PETSC_EXTERN PetscErrorCode DMComplexCreatePartitionClosure(DM, PetscSection, IS, PetscSection *, IS *);

PETSC_EXTERN PetscErrorCode DMComplexGenerate(DM, const char [], PetscBool , DM *);
PETSC_EXTERN PetscErrorCode DMComplexSetRefinementLimit(DM, PetscReal);
PETSC_EXTERN PetscErrorCode DMComplexDistribute(DM, const char[], DM*);
PETSC_EXTERN PetscErrorCode DMComplexLoad(PetscViewer, DM);

PETSC_EXTERN PetscErrorCode DMComplexCreateCubeBoundary(DM, const PetscReal [], const PetscReal [], const PetscInt []);
PETSC_EXTERN PetscErrorCode DMComplexCreateBoxMesh(MPI_Comm, PetscInt, PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMComplexCreateHexBoxMesh(MPI_Comm,PetscInt,const PetscInt[],DM *);
PETSC_EXTERN PetscErrorCode DMComplexGetDepth(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMComplexGetDepthStratum(DM, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMComplexGetHeightStratum(DM, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMComplexCreateSection(DM, PetscInt, PetscInt, PetscInt [], PetscInt [], PetscInt, PetscInt [], IS [], PetscSection *);
PETSC_EXTERN PetscErrorCode DMComplexCreateConeSection(DM, PetscSection *);

/* FEM Support */
PETSC_EXTERN PetscErrorCode DMComplexComputeCellGeometry(DM, PetscInt, PetscReal *, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode DMComplexVecGetClosure(DM, PetscSection, Vec, PetscInt, PetscInt *, const PetscScalar *[]);
PETSC_EXTERN PetscErrorCode DMComplexVecRestoreClosure(DM, PetscSection, Vec, PetscInt, PetscInt *, const PetscScalar *[]);
PETSC_EXTERN PetscErrorCode DMComplexVecSetClosure(DM, PetscSection, Vec, PetscInt, const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode DMComplexMatSetClosure(DM, PetscSection, PetscSection, Mat, PetscInt, PetscScalar[], InsertMode);

PETSC_EXTERN PetscErrorCode DMComplexCreateExodus(MPI_Comm , PetscInt , DM *);

PETSC_EXTERN PetscErrorCode DMComplexGetVTKBounds(DM, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMComplexSetVTKBounds(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMComplexVTKWriteAll(PetscObject, PetscViewer);

#endif
