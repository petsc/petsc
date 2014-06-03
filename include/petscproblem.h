/*
      Objects which encapsulate discretizations+continuum residuals
*/
#if !defined(__PETSCPROBLEM_H)
#define __PETSCPROBLEM_H
#include <petscfe.h>
#include <petscfv.h>
#include <petscproblemtypes.h>

PETSC_EXTERN PetscErrorCode PetscProblemInitializePackage(void);

PETSC_EXTERN PetscClassId PETSCPROBLEM_CLASSID;

/*J
  PetscProblemType - String with the name of a PETSc problem

  Level: beginner

.seealso: PetscProblemSetType(), PetscProblem
J*/
typedef const char *PetscProblemType;
#define PETSCPROBLEMBASIC "basic"

PETSC_EXTERN PetscFunctionList PetscProblemList;
PETSC_EXTERN PetscBool         PetscProblemRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscProblemCreate(MPI_Comm, PetscProblem *);
PETSC_EXTERN PetscErrorCode PetscProblemDestroy(PetscProblem *);
PETSC_EXTERN PetscErrorCode PetscProblemSetType(PetscProblem, PetscProblemType);
PETSC_EXTERN PetscErrorCode PetscProblemGetType(PetscProblem, PetscProblemType *);
PETSC_EXTERN PetscErrorCode PetscProblemSetUp(PetscProblem);
PETSC_EXTERN PetscErrorCode PetscProblemSetFromOptions(PetscProblem);
PETSC_EXTERN PetscErrorCode PetscProblemViewFromOptions(PetscProblem,const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscProblemView(PetscProblem,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscProblemRegister(const char [], PetscErrorCode (*)(PetscProblem));
PETSC_EXTERN PetscErrorCode PetscProblemRegisterAll(void);
PETSC_EXTERN PetscErrorCode PetscProblemRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode PetscProblemGetSpatialDimension(PetscProblem, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscProblemGetNumFields(PetscProblem, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscProblemGetTotalDimension(PetscProblem, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscProblemGetTotalBdDimension(PetscProblem, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscProblemGetTotalComponents(PetscProblem, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscProblemGetFieldOffset(PetscProblem, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscProblemGetBdFieldOffset(PetscProblem, PetscInt, PetscInt *);

PETSC_EXTERN PetscErrorCode PetscProblemGetDiscretization(PetscProblem, PetscInt, PetscObject *);
PETSC_EXTERN PetscErrorCode PetscProblemSetDiscretization(PetscProblem, PetscInt, PetscObject);
PETSC_EXTERN PetscErrorCode PetscProblemAddDiscretization(PetscProblem, PetscObject);
PETSC_EXTERN PetscErrorCode PetscProblemGetBdDiscretization(PetscProblem, PetscInt, PetscObject *);
PETSC_EXTERN PetscErrorCode PetscProblemSetBdDiscretization(PetscProblem, PetscInt, PetscObject);
PETSC_EXTERN PetscErrorCode PetscProblemAddBdDiscretization(PetscProblem, PetscObject);
PETSC_EXTERN PetscErrorCode PetscProblemGetObjective(PetscProblem, PetscInt, void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscProblemSetObjective(PetscProblem, PetscInt, void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscProblemGetResidual(PetscProblem, PetscInt,
                                                    void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                    void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscProblemSetResidual(PetscProblem, PetscInt,
                                                    void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                    void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscProblemGetJacobian(PetscProblem, PetscInt, PetscInt,
                                                    void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                    void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                    void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                    void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscProblemSetJacobian(PetscProblem, PetscInt, PetscInt,
                                                    void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                    void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                    void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                    void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscProblemGetBdResidual(PetscProblem, PetscInt,
                                                      void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                      void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscProblemSetBdResidual(PetscProblem, PetscInt,
                                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscProblemGetBdJacobian(PetscProblem, PetscInt, PetscInt,
                                                      void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                      void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                      void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                      void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscProblemSetBdJacobian(PetscProblem, PetscInt, PetscInt,
                                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode PetscProblemGetTabulation(PetscProblem, PetscReal ***, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscProblemGetBdTabulation(PetscProblem, PetscReal ***, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscProblemGetEvaluationArrays(PetscProblem, PetscScalar **, PetscScalar **, PetscScalar **);
PETSC_EXTERN PetscErrorCode PetscProblemGetWeakFormArrays(PetscProblem, PetscScalar **, PetscScalar **, PetscScalar **, PetscScalar **, PetscScalar **, PetscScalar **);
PETSC_EXTERN PetscErrorCode PetscProblemGetRefCoordArrays(PetscProblem, PetscReal **, PetscScalar **);

#endif
