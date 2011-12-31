/* 
  An application ordering is mapping between an application-centric
  ordering (the ordering that is "natural" for the application) and 
  the parallel ordering that PETSc uses.
*/
#if !defined(__PETSCAO_H)
#define __PETSCAO_H
#include "petscis.h"
#include "petscmat.h"
PETSC_EXTERN_CXX_BEGIN

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
extern PetscClassId  AO_CLASSID;

extern PetscErrorCode  AOInitializePackage(const char[]);

extern PetscErrorCode  AOCreate(MPI_Comm,AO*);
extern PetscErrorCode  AOSetIS(AO,IS,IS);
extern PetscErrorCode  AOSetFromOptions(AO);

extern PetscErrorCode  AOCreateBasic(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],AO*);
extern PetscErrorCode  AOCreateBasicIS(IS,IS,AO*);
extern PetscErrorCode  AOCreateMemoryScalable(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],AO*);
extern PetscErrorCode  AOCreateMemoryScalableIS(IS,IS,AO*);
extern PetscErrorCode  AOCreateMapping(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],AO*);
extern PetscErrorCode  AOCreateMappingIS(IS,IS,AO*);

extern PetscErrorCode  AOView(AO,PetscViewer);
extern PetscErrorCode  AODestroy(AO*);

/* Dynamic creation and loading functions */
extern PetscFList AOList;
extern PetscBool  AORegisterAllCalled;
extern PetscErrorCode  AOSetType(AO, const AOType);
extern PetscErrorCode  AOGetType(AO, const AOType *);

extern PetscErrorCode  AORegister(const char [], const char [], const char [], PetscErrorCode (*)(AO));
extern PetscErrorCode  AORegisterAll(const char []);
extern PetscErrorCode  AORegisterDestroy(void);

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define AORegisterDynamic(a,b,c,d) AORegister(a,b,c,0)
#else
#define AORegisterDynamic(a,b,c,d) AORegister(a,b,c,d)
#endif

extern PetscErrorCode  AOPetscToApplication(AO,PetscInt,PetscInt[]);
extern PetscErrorCode  AOApplicationToPetsc(AO,PetscInt,PetscInt[]);
extern PetscErrorCode  AOPetscToApplicationIS(AO,IS);
extern PetscErrorCode  AOApplicationToPetscIS(AO,IS);

extern PetscErrorCode  AOPetscToApplicationPermuteInt(AO, PetscInt, PetscInt[]);
extern PetscErrorCode  AOApplicationToPetscPermuteInt(AO, PetscInt, PetscInt[]);
extern PetscErrorCode  AOPetscToApplicationPermuteReal(AO, PetscInt, PetscReal[]);
extern PetscErrorCode  AOApplicationToPetscPermuteReal(AO, PetscInt, PetscReal[]);

extern PetscErrorCode  AOMappingHasApplicationIndex(AO, PetscInt, PetscBool  *);
extern PetscErrorCode  AOMappingHasPetscIndex(AO, PetscInt, PetscBool  *);

/* ----------------------------------------------------*/
PETSC_EXTERN_CXX_END
#endif
