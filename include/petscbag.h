
#if !defined(__PETSCBAG_H)
#define __PETSCBAG_H
#include "petsc.h"
PETSC_EXTERN_CXX_BEGIN


/*S
     PetscBag - PETSc object that manages a collection of user data including parameters.

   Level: beginner

    Sample Usage:
$      typedef struct {
$         PetscBag     bag;
$         PetscInt     height;
$         PetscScalar  root;
$         PetscReal    byebye;
$      } MyParameters;
$
$      MyParameters *params;
$      
$      ierr = PetscBagCreate(MyParameters,&params);
$      ierr = PetscBagSetName(params,"MyParameters");
$      ierr = PetscBagRegisterInt(params,22,"height","Height of the water tower");
$       
$       
$       
$       
$       

.seealso:  PetscBagSetName(), PetscBagGetName(), PetscBagSetSize(), PetscBagView(), PetscBagLoad()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterTruth(), PetscBagRegisterScalar()
           PetscBagSetFromOptions()
S*/
typedef struct {
  size_t bagsize;
  char*  bagname;
} PetscBag;

#define PetscBagCreate(A,B)  PetscNew(A,B) || (*(B)->bagsize = sizeof(A),0)
extern PetscErrorCode PetscBagDestroy(PetscBag*);

extern PetscErrorCode PetscBagSetName(PetscBag*,char*);
extern PetscErrorCode PetscBagGetName(PetscBag*,char**);
extern PetscErrorCode PetscBagSetSize(PetscBag*,size_t);

extern PetscErrorCode PetscBagRegisterReal(PetscBag*,PetscReal, char*, char*);
extern PetscErrorCode PetscBagRegisterScalar(PetscBag*,PetscScalar, char*, char*);
extern PetscErrorCode PetscBagRegisterInt(PetscBag*,PetscInt, char*, char*);
extern PetscErrorCode PetscBagRegisterTruth(PetscBag*,PetscTruth, char*, char*);

extern PetscErrorCode PetscBagSetFromOptions(PetscBag*);

extern PetscErrorCode PetscBagView(PetscBag*,PetscViewer);
extern PetscErrorCode PetscBagLoad(PetscViewer,PetscBag**);

extern PetscErrorCode PetscBagSetViewer(PetscBag*,PetscErrorCode (*)(PetscBag*,PetscViewer));
extern PetscErrorCode PetscBagSetLoader(PetscBag*,PetscErrorCode (*)(PetscBag*,PetscViewer));
extern PetscErrorCode PetscBagSetDestroy(PetscBag*,PetscErrorCode (*)(PetscBag*));

#endif
