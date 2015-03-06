/*
      Objects which encapsulate discretizations+continuum residuals
*/
#if !defined(__PETSCDS_H)
#define __PETSCDS_H
#include <petscfe.h>
#include <petscfv.h>
#include <petscdstypes.h>

PETSC_EXTERN PetscErrorCode PetscDSInitializePackage(void);

PETSC_EXTERN PetscClassId PETSCDS_CLASSID;

/*J
  PetscDSType - String with the name of a PETSc discrete system

  Level: beginner

.seealso: PetscDSSetType(), PetscDS
J*/
typedef const char *PetscDSType;
#define PETSCDSBASIC "basic"

PETSC_EXTERN PetscFunctionList PetscDSList;
PETSC_EXTERN PetscErrorCode PetscDSCreate(MPI_Comm, PetscDS *);
PETSC_EXTERN PetscErrorCode PetscDSDestroy(PetscDS *);
PETSC_EXTERN PetscErrorCode PetscDSSetType(PetscDS, PetscDSType);
PETSC_EXTERN PetscErrorCode PetscDSGetType(PetscDS, PetscDSType *);
PETSC_EXTERN PetscErrorCode PetscDSSetUp(PetscDS);
PETSC_EXTERN PetscErrorCode PetscDSSetFromOptions(PetscDS);
PETSC_EXTERN PetscErrorCode PetscDSViewFromOptions(PetscDS,const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscDSView(PetscDS,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDSRegister(const char [], PetscErrorCode (*)(PetscDS));
PETSC_EXTERN PetscErrorCode PetscDSRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode PetscDSGetSpatialDimension(PetscDS, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDSGetNumFields(PetscDS, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDSGetTotalDimension(PetscDS, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDSGetTotalBdDimension(PetscDS, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDSGetTotalComponents(PetscDS, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDSGetFieldOffset(PetscDS, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDSGetBdFieldOffset(PetscDS, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDSGetComponentOffset(PetscDS, PetscInt, PetscInt *);

PETSC_EXTERN PetscErrorCode PetscDSGetDiscretization(PetscDS, PetscInt, PetscObject *);
PETSC_EXTERN PetscErrorCode PetscDSSetDiscretization(PetscDS, PetscInt, PetscObject);
PETSC_EXTERN PetscErrorCode PetscDSAddDiscretization(PetscDS, PetscObject);
PETSC_EXTERN PetscErrorCode PetscDSGetBdDiscretization(PetscDS, PetscInt, PetscObject *);
PETSC_EXTERN PetscErrorCode PetscDSSetBdDiscretization(PetscDS, PetscInt, PetscObject);
PETSC_EXTERN PetscErrorCode PetscDSAddBdDiscretization(PetscDS, PetscObject);
PETSC_EXTERN PetscErrorCode PetscDSGetObjective(PetscDS, PetscInt, void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscDSSetObjective(PetscDS, PetscInt, void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscDSGetResidual(PetscDS, PetscInt,
                                                    void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                    void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscDSSetResidual(PetscDS, PetscInt,
                                                    void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                    void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscDSGetJacobian(PetscDS, PetscInt, PetscInt,
                                                    void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                    void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                    void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                    void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscDSSetJacobian(PetscDS, PetscInt, PetscInt,
                                                    void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                    void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                    void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                    void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscDSGetRiemannSolver(PetscDS, PetscInt,
                                                    void (**)(const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscScalar[], void *));
PETSC_EXTERN PetscErrorCode PetscDSSetRiemannSolver(PetscDS, PetscInt,
                                                    void (*)(const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscScalar[], void *));
PETSC_EXTERN PetscErrorCode PetscDSGetContext(PetscDS, PetscInt, void **);
PETSC_EXTERN PetscErrorCode PetscDSSetContext(PetscDS, PetscInt, void *);
PETSC_EXTERN PetscErrorCode PetscDSGetBdResidual(PetscDS, PetscInt,
                                                      void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                      void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscDSSetBdResidual(PetscDS, PetscInt,
                                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscDSGetBdJacobian(PetscDS, PetscInt, PetscInt,
                                                      void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                      void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                      void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                      void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscDSSetBdJacobian(PetscDS, PetscInt, PetscInt,
                                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscDSGetTabulation(PetscDS, PetscReal ***, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscDSGetBdTabulation(PetscDS, PetscReal ***, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscDSGetEvaluationArrays(PetscDS, PetscScalar **, PetscScalar **, PetscScalar **);
PETSC_EXTERN PetscErrorCode PetscDSGetWeakFormArrays(PetscDS, PetscScalar **, PetscScalar **, PetscScalar **, PetscScalar **, PetscScalar **, PetscScalar **);
PETSC_EXTERN PetscErrorCode PetscDSGetRefCoordArrays(PetscDS, PetscReal **, PetscScalar **);

#endif
