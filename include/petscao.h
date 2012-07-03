/* 
  An application ordering is mapping between an application-centric
  ordering (the ordering that is "natural" for the application) and 
  the parallel ordering that PETSc uses.
*/
#if !defined(__PETSCAO_H)
#define __PETSCAO_H
#include "petscis.h"
#include "petscmat.h"

/*S
     AO - Abstract PETSc object that manages mapping between different global numbering

   Level: intermediate

  Concepts: global numbering

.seealso:  AOCreateBasic(), AOCreateBasicIS(), AOPetscToApplication(), AOView(), AOApplicationToPetsc()
S*/
typedef struct _p_AO* AO;

/*J
    AOType - String with the name of a PETSc application ordering or the creation function
       with an optional dynamic library name.

   Level: beginner

.seealso: AOSetType(), AO
J*/
#define AOType char*
#define AOBASIC               "basic"
#define AOADVANCED            "advanced"
#define AOMAPPING             "mapping"
#define AOMEMORYSCALABLE      "memoryscalable"

/* Logging support */
PETSC_EXTERN PetscClassId AO_CLASSID;

PETSC_EXTERN PetscErrorCode AOInitializePackage(const char[]);

PETSC_EXTERN PetscErrorCode AOCreate(MPI_Comm,AO*);
PETSC_EXTERN PetscErrorCode AOSetIS(AO,IS,IS);
PETSC_EXTERN PetscErrorCode AOSetFromOptions(AO);

PETSC_EXTERN PetscErrorCode AOCreateBasic(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],AO*);
PETSC_EXTERN PetscErrorCode AOCreateBasicIS(IS,IS,AO*);
PETSC_EXTERN PetscErrorCode AOCreateMemoryScalable(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],AO*);
PETSC_EXTERN PetscErrorCode AOCreateMemoryScalableIS(IS,IS,AO*);
PETSC_EXTERN PetscErrorCode AOCreateMapping(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],AO*);
PETSC_EXTERN PetscErrorCode AOCreateMappingIS(IS,IS,AO*);

PETSC_EXTERN PetscErrorCode AOView(AO,PetscViewer);
PETSC_EXTERN PetscErrorCode AODestroy(AO*);

/* Dynamic creation and loading functions */
PETSC_EXTERN PetscFList AOList;
PETSC_EXTERN PetscBool AORegisterAllCalled;
PETSC_EXTERN PetscErrorCode AOSetType(AO, const AOType);
PETSC_EXTERN PetscErrorCode AOGetType(AO, const AOType *);

PETSC_EXTERN PetscErrorCode AORegister(const char [], const char [], const char [], PetscErrorCode (*)(AO));
PETSC_EXTERN PetscErrorCode AORegisterAll(const char []);
PETSC_EXTERN PetscErrorCode AORegisterDestroy(void);

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define AORegisterDynamic(a,b,c,d) AORegister(a,b,c,0)
#else
#define AORegisterDynamic(a,b,c,d) AORegister(a,b,c,d)
#endif

PETSC_EXTERN PetscErrorCode AOPetscToApplication(AO,PetscInt,PetscInt[]);
PETSC_EXTERN PetscErrorCode AOApplicationToPetsc(AO,PetscInt,PetscInt[]);
PETSC_EXTERN PetscErrorCode AOPetscToApplicationIS(AO,IS);
PETSC_EXTERN PetscErrorCode AOApplicationToPetscIS(AO,IS);

PETSC_EXTERN PetscErrorCode AOPetscToApplicationPermuteInt(AO, PetscInt, PetscInt[]);
PETSC_EXTERN PetscErrorCode AOApplicationToPetscPermuteInt(AO, PetscInt, PetscInt[]);
PETSC_EXTERN PetscErrorCode AOPetscToApplicationPermuteReal(AO, PetscInt, PetscReal[]);
PETSC_EXTERN PetscErrorCode AOApplicationToPetscPermuteReal(AO, PetscInt, PetscReal[]);

PETSC_EXTERN PetscErrorCode AOMappingHasApplicationIndex(AO, PetscInt, PetscBool  *);
PETSC_EXTERN PetscErrorCode AOMappingHasPetscIndex(AO, PetscInt, PetscBool  *);

/* ----------------------------------------------------*/
#endif
