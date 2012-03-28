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
extern PetscErrorCode DMComplexCreate(MPI_Comm, DM*);
extern PetscErrorCode DMComplexGetDimension(DM, PetscInt *);
extern PetscErrorCode DMComplexSetDimension(DM, PetscInt);
extern PetscErrorCode DMComplexGetChart(DM, PetscInt *, PetscInt *);
extern PetscErrorCode DMComplexSetChart(DM, PetscInt, PetscInt);
extern PetscErrorCode DMComplexGetConeSize(DM, PetscInt, PetscInt *);
extern PetscErrorCode DMComplexSetConeSize(DM, PetscInt, PetscInt);
extern PetscErrorCode DMComplexGetCone(DM, PetscInt, const PetscInt *[]);
extern PetscErrorCode DMComplexSetCone(DM, PetscInt, const PetscInt[]);
extern PetscErrorCode DMComplexGetConeOrientation(DM, PetscInt, const PetscInt *[]);
extern PetscErrorCode DMComplexSetConeOrientation(DM, PetscInt, const PetscInt[]);
extern PetscErrorCode DMComplexGetSupportSize(DM, PetscInt, PetscInt *);
extern PetscErrorCode DMComplexGetSupport(DM, PetscInt, const PetscInt *[]);
extern PetscErrorCode DMComplexGetConeSection(DM, PetscSection *);
extern PetscErrorCode DMComplexGetCones(DM, PetscInt *[]);
extern PetscErrorCode DMComplexGetConeOrientations(DM, PetscInt *[]);
extern PetscErrorCode DMComplexGetMaxSizes(DM, PetscInt *, PetscInt *);
extern PetscErrorCode DMComplexSetUp(DM);
extern PetscErrorCode DMComplexSymmetrize(DM);
extern PetscErrorCode DMComplexStratify(DM);

extern PetscErrorCode DMComplexHasLabel(DM, const char [], PetscBool *);
extern PetscErrorCode DMComplexGetLabelValue(DM, const char[], PetscInt, PetscInt *);
extern PetscErrorCode DMComplexSetLabelValue(DM, const char[], PetscInt, PetscInt);
extern PetscErrorCode DMComplexGetLabelSize(DM, const char[], PetscInt *);
extern PetscErrorCode DMComplexGetLabelIdIS(DM, const char[], IS *);
extern PetscErrorCode DMComplexGetStratumSize(DM, const char [], PetscInt, PetscInt *);
extern PetscErrorCode DMComplexGetStratumIS(DM, const char [], PetscInt, IS *);

extern PetscErrorCode DMComplexMeetPoints(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
extern PetscErrorCode DMComplexJoinPoints(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
extern PetscErrorCode DMComplexGetTransitiveClosure(DM, PetscInt, PetscBool, PetscInt *, PetscInt *[]);

extern PetscErrorCode DMComplexCreatePartition(DM, PetscSection *, IS *, PetscInt);
extern PetscErrorCode DMComplexCreatePartitionClosure(DM, PetscSection, IS, PetscSection *, IS *);

extern PetscErrorCode DMComplexGenerate(DM, const char [], PetscBool , DM *);
extern PetscErrorCode DMComplexSetRefinementLimit(DM, PetscReal);
extern PetscErrorCode DMComplexDistribute(DM, const char[], DM*);
extern PetscErrorCode DMComplexLoad(PetscViewer, DM);

extern PetscErrorCode DMComplexCreateCubeBoundary(DM, const PetscReal [], const PetscReal [], const PetscInt []);
extern PetscErrorCode DMComplexCreateBoxMesh(MPI_Comm, PetscInt, PetscBool, DM *);
extern PetscErrorCode DMComplexGetDepth(DM, PetscInt *);
extern PetscErrorCode DMComplexGetDepthStratum(DM, PetscInt, PetscInt *, PetscInt *);
extern PetscErrorCode DMComplexGetHeightStratum(DM, PetscInt, PetscInt *, PetscInt *);
extern PetscErrorCode DMComplexCreateSection(DM, PetscInt, PetscInt, PetscInt [], PetscInt [], PetscInt, PetscInt [], IS [], PetscSection *);
extern PetscErrorCode DMComplexGetCoordinateSection(DM, PetscSection *);
extern PetscErrorCode DMComplexSetCoordinateSection(DM, PetscSection);
extern PetscErrorCode DMComplexGetCoordinateVec(DM, Vec *);
extern PetscErrorCode DMComplexCreateConeSection(DM, PetscSection *);

/* FEM Support */
extern PetscErrorCode DMComplexGetDefaultSection(DM, PetscSection *);
extern PetscErrorCode DMComplexSetDefaultSection(DM, PetscSection);
extern PetscErrorCode DMComplexGetDefaultGlobalSection(DM, PetscSection *);
extern PetscErrorCode DMComplexCreateDefaultSF(DM);
extern PetscErrorCode DMComplexGetLocalFunction(DM, PetscErrorCode (**)(DM, Vec, Vec, void *));
extern PetscErrorCode DMComplexSetLocalFunction(DM, PetscErrorCode (*)(DM, Vec, Vec, void *));
extern PetscErrorCode DMComplexGetLocalJacobian(DM, PetscErrorCode (**)(DM, Vec, Mat, Mat, void *));
extern PetscErrorCode DMComplexSetLocalJacobian(DM, PetscErrorCode (*)(DM, Vec, Mat, Mat, void *));
typedef PetscErrorCode (*DMComplexLocalFunction1)(DM, Vec, Vec, void*);
typedef PetscErrorCode (*DMComplexLocalJacobian1)(DM, Vec, Mat, Mat, void*);

extern PetscErrorCode DMComplexComputeCellGeometry(DM, PetscInt, PetscReal *, PetscReal *, PetscReal *, PetscReal *);
extern PetscErrorCode DMComplexVecGetClosure(DM, PetscSection, Vec, PetscInt, const PetscScalar *[]);
extern PetscErrorCode DMComplexVecSetClosure(DM, PetscSection, Vec, PetscInt, const PetscScalar[], InsertMode);
extern PetscErrorCode DMComplexMatSetClosure(DM, PetscSection, PetscSection, Mat, PetscInt, PetscScalar[], InsertMode);

extern PetscErrorCode DMComplexCreateExodus(MPI_Comm , PetscInt , DM *);


#endif
