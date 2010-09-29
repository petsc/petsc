
#if !defined(__PETSCBAG_H)
#define __PETSCBAG_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN

/*S
     PetscBag - PETSc object that manages a collection of user data including parameters.
           A bag is essentially a C struct with serialization (you can save it and load it from files).

   Level: beginner

    Sample Usage:
$      typedef struct {
$         PetscInt     height;
$         PetscScalar  root;
$         PetscReal    byebye;
$      } MyParameters;
$
$      PetscBag     bag;
$      MyParameters *params;
$      
$      ierr = PetscBagCreate(PETSC_COMM_WORLD,sizeof(MyParameters),&bag);
$      ierr = PetscBagGetData(bag,(void **)&params);
$      ierr = PetscBagSetName(bag,"MyParameters");
$      ierr = PetscBagRegisterInt(bag,&params.height,22,"height","Height of the water tower");
$       

.seealso:  PetscBagSetName(), PetscBagGetName(), PetscBagView(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterTruth(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagRegisterVec(), PetscBagCreate(), PetscBagDestroy(), PetscBagRegisterEnum()
S*/
typedef struct _n_PetscBag*     PetscBag;
typedef struct _n_PetscBagItem* PetscBagItem;

EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagCreate(MPI_Comm,size_t,PetscBag*);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagDestroy(PetscBag);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagGetData(PetscBag,void **);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagRegisterReal(PetscBag,void*,PetscReal, const char*, const char*);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagRegisterString(PetscBag,void*,PetscInt,const char*, const char*, const char*);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagRegisterScalar(PetscBag,void*,PetscScalar,const  char*,const  char*);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagRegisterInt(PetscBag,void*,PetscInt,const  char*,const  char*);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagRegisterEnum(PetscBag,void*,const  char*[],PetscEnum,const char*,const  char*);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagRegisterTruth(PetscBag,void*,PetscBool ,const  char*,const  char*);

EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagSetFromOptions(PetscBag);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagGetName(PetscBag, char **);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagSetName(PetscBag, const char *, const char *);

EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagView(PetscBag,PetscViewer);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagLoad(PetscViewer,PetscBag*);

EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagSetViewer(PetscBag,PetscErrorCode (*)(PetscBag,PetscViewer));
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagSetLoader(PetscBag,PetscErrorCode (*)(PetscBag,PetscViewer));
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscBagSetDestroy(PetscBag,PetscErrorCode (*)(PetscBag));

PETSC_EXTERN_CXX_END
#endif
