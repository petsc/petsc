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

typedef enum {AO_BASIC=0, AO_ADVANCED=1, AO_MAPPING=2} AOType;

/*S
     AO - Abstract PETSc object that manages mapping between different global numbering

   Level: intermediate

  Concepts: global numbering

.seealso:  AOCreateBasic(), AOCreateBasicIS(), AOPetscToApplication(), AOView(), AOApplicationToPetsc()
S*/
typedef struct _p_AO* AO;

/* Logging support */
extern PetscClassId  AO_CLASSID;

extern PetscErrorCode  AOInitializePackage(const char[]);

extern PetscErrorCode  AOCreateBasic(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],AO*);
extern PetscErrorCode  AOCreateBasicIS(IS,IS,AO*);

extern PetscErrorCode  AOCreateBasicMemoryScalable(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],AO*);
extern PetscErrorCode  AOCreateBasicMemoryScalableIS(IS,IS,AO*);

extern PetscErrorCode  AOCreateMapping(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],AO*);
extern PetscErrorCode  AOCreateMappingIS(IS,IS,AO*);

extern PetscErrorCode  AOView(AO,PetscViewer);
extern PetscErrorCode  AODestroy(AO);

extern PetscErrorCode  AORegister(const char [], const char [], const char [], PetscErrorCode (*)(AO));
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
