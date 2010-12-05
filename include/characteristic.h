/*
   Defines the interface functions for the method of characteristics solvers
*/
#ifndef __PETSCCHARACTERISTICS_H
#define __PETSCCHARACTERISTICS_H

#include <petscvec.h>
#include <petscdm.h>

extern PetscErrorCode CharacteristicInitializePackage(const char[]);

/*S
     Characteristic - Abstract PETSc object that manages method of characteristics solves

   Level: beginner

  Concepts: Method of characteristics

.seealso:  CharacteristicCreate(), CharacteristicSetType(), CharacteristicType, SNES, TS, PC, KSP
S*/
typedef struct _p_Characteristic *Characteristic;

/*E
    CharacteristicType - String with the name of a characteristics method or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mymoccreate()

   Level: beginner

.seealso: CharacteristicSetType(), Characteristic
E*/
#define CHARACTERISTICDA "da"
#define CharacteristicType char*

/* Logging support */
extern PetscClassId CHARACTERISTIC_CLASSID;
extern PetscLogEvent  CHARACTERISTIC_SetUp, CHARACTERISTIC_Solve, CHARACTERISTIC_QueueSetup, CHARACTERISTIC_DAUpdate;
extern PetscLogEvent  CHARACTERISTIC_HalfTimeLocal, CHARACTERISTIC_HalfTimeRemote, CHARACTERISTIC_HalfTimeExchange;
extern PetscLogEvent  CHARACTERISTIC_FullTimeLocal, CHARACTERISTIC_FullTimeRemote, CHARACTERISTIC_FullTimeExchange;

/* function to wrap coordinates around boundary */
extern PetscErrorCode DMDAMapCoordsToPeriodicDomain(DM, PetscScalar*, PetscScalar*);

extern PetscErrorCode CharacteristicCreate(MPI_Comm, Characteristic *);
extern PetscErrorCode CharacteristicSetType(Characteristic, const CharacteristicType);
extern PetscErrorCode CharacteristicSetUp(Characteristic);
extern PetscErrorCode CharacteristicSetVelocityInterpolation(Characteristic, DM, Vec, Vec, PetscInt, PetscInt[], PetscErrorCode (*)(Vec, PetscReal[], PetscInt, PetscInt[], PetscScalar[], void *), void *);
extern PetscErrorCode CharacteristicSetVelocityInterpolationLocal(Characteristic, DM, Vec, Vec, PetscInt, PetscInt[], PetscErrorCode (*)(void *, PetscReal[], PetscInt, PetscInt[], PetscScalar[], void *), void *);
extern PetscErrorCode CharacteristicSetFieldInterpolation(Characteristic, DM, Vec, PetscInt, PetscInt[], PetscErrorCode (*)(Vec, PetscReal[], PetscInt, PetscInt[], PetscScalar[], void *), void *);
extern PetscErrorCode CharacteristicSetFieldInterpolationLocal(Characteristic, DM, Vec, PetscInt, PetscInt[], PetscErrorCode (*)(void *, PetscReal[], PetscInt, PetscInt[], PetscScalar[], void *), void *);
extern PetscErrorCode CharacteristicSolve(Characteristic, PetscReal, Vec);
extern PetscErrorCode CharacteristicDestroy(Characteristic);

extern PetscFList CharacteristicList;
extern PetscErrorCode CharacteristicRegisterAll(const char[]);
extern PetscErrorCode CharacteristicRegisterDestroy(void);

extern PetscErrorCode CharacteristicRegister(const char[],const char[],const char[],PetscErrorCode (*)(Characteristic));

/*MC
   CharacteristicRegisterDynamic - Adds a solver to the method of characteristics package.

   Synopsis:
   PetscErrorCode CharacteristicRegisterDynamic(const char *name_solver,const char *path,const char *name_create,PetscErrorCode (*routine_create)(Characteristic))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   CharacteristicRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   CharacteristicRegisterDynamic("my_solver",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     CharacteristicSetType(ksp,"my_solver")
   or at runtime via the option
$     -characteristic_type my_solver

   Level: advanced

   Notes: Environmental variables such as ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR},
          and others of the form ${any_environmental_variable} occuring in pathname will be 
          replaced with appropriate values.
         If your function is not being put into a shared library then use CharacteristicRegister() instead

.keywords: Characteristic, register

.seealso: CharacteristicRegisterAll(), CharacteristicRegisterDestroy()

M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define CharacteristicRegisterDynamic(a,b,c,d) CharacteristicRegister(a,b,c,0)
#else
#define CharacteristicRegisterDynamic(a,b,c,d) CharacteristicRegister(a,b,c,d)
#endif

#endif /*__PETSCCHARACTERISTICS_H*/
