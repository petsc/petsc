
#if !defined(__PETSCBAG_H)
#define __PETSCBAG_H
#include "petsc.h"
PETSC_EXTERN_CXX_BEGIN

#define PETSC_BAG_NAME_LENGTH 64
#define PETSC_BAG_HELP_LENGTH 128
#define PETSC_BAG_FILE_COOKIE 1211219

typedef struct _p_PetscBagItem *PetscBagItem;
struct _p_PetscBagItem {PetscDataType dtype;PetscInt offset;size_t msize;char name[PETSC_BAG_NAME_LENGTH],help[PETSC_BAG_HELP_LENGTH];PetscBagItem next;};
/*S
     PetscBag - PETSc object that manages a collection of user data including parameters.
           A bag is essentially a C struct with serialization (you can save it and load it from files).

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
$      ierr = PetscBagRegisterInt(params,&params.height,22,"height","Height of the water tower");
$       
$       
$       
$       
$       

.seealso:  PetscBagSetName(), PetscBagGetName(), PetscBagView(), PetscBagLoad()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterTruth(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagRegisterVec(), PetscBagCreate(), PetscBagDestroy()
S*/
typedef struct {
  MPI_Comm     bagcomm;
  size_t       bagsize;
  PetscInt     count;
  char         bagname[PETSC_BAG_NAME_LENGTH];
  char         baghelp[PETSC_BAG_HELP_LENGTH];
  PetscBagItem bagitems;
} PetscBag;

/*MC
    PetscBagCreate - Create a bag of values

  Collective on MPI_Comm

  Level: Intermediate

  Synopsis:
     PetscErrorCode PetscBagCreate(MPI_Comm comm,C struct name,PetscBag **bag);

  Input Parameters:
+  comm - communicator to share bag
-  C struct name - name of the C structure holding the values

  Output Parameter:
.   bag - the bag of values


.seealso: PetscBag, PetscBagGetName(), PetscBagView(), PetscBagLoad()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterTruth(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagRegisterVec(), PetscBagCreate(), PetscBagDestroy()
M*/ 
#define PetscBagCreate(C,A,B)  PetscNew(A,B) || ((*(B))->bagsize = sizeof(A),(*(B))->bagcomm = C,0)

extern PetscErrorCode PetscBagDestroy(PetscBag*);

/*MC
    PetscBagSetName - Sets the name of a bag of values

  Not Collective

  Level: Intermediate

  Synopsis:
     PetscErrorCode PetscBagSetName(PetscBag *bag,const char *name, const char *help);

  Input Parameters:
+   bag - the bag of values
.   name - the name assigned to the bag
-   help - help message for bag

.seealso: PetscBag, PetscBagGetName(), PetscBagView(), PetscBagLoad()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterTruth(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagRegisterVec(), PetscBagCreate(), PetscBagDestroy()
M*/ 
#define PetscBagSetName(A,B,C) (PetscStrncpy((A)->bagname,B,PETSC_BAG_NAME_LENGTH-1) || PetscStrncpy((A)->baghelp,C,PETSC_BAG_HELP_LENGTH-1))

/*MC
    PetscBagGetName - Gets the name of a bag of values

  Not Collective

  Level: Intermediate

  Synopsis:
     PetscErrorCode PetscBagGetName(PetscBag *bag,char **name);

  Input Parameter:
.   bag - the bag of values

  Output Parameter:
.   name - the name assigned to the bag

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterTruth(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagRegisterVec(), PetscBagCreate(), PetscBagDestroy()
M*/ 
#define PetscBagGetName(A,B) (*(B) = A->bagname,0)

extern PetscErrorCode PetscBagRegisterReal(PetscBag*,void*,PetscReal, const char*, const char*);
extern PetscErrorCode PetscBagRegisterString(PetscBag*,void*,size_t,const char*, const char*, const char*);
extern PetscErrorCode PetscBagRegisterScalar(PetscBag*,void*,PetscScalar,const  char*,const  char*);
extern PetscErrorCode PetscBagRegisterInt(PetscBag*,void*,PetscInt,const  char*,const  char*);
extern PetscErrorCode PetscBagRegisterTruth(PetscBag*,void*,PetscTruth,const  char*,const  char*);
extern PetscErrorCode PetscBagRegisterVec(PetscBag*,void*,const char*,const  char*);

extern PetscErrorCode PetscBagSetFromOptions(PetscBag*);

extern PetscErrorCode PetscBagView(PetscBag*,PetscViewer);
extern PetscErrorCode PetscBagLoad(PetscViewer,PetscBag**);

extern PetscErrorCode PetscBagSetViewer(PetscBag*,PetscErrorCode (*)(PetscBag*,PetscViewer));
extern PetscErrorCode PetscBagSetLoader(PetscBag*,PetscErrorCode (*)(PetscBag*,PetscViewer));
extern PetscErrorCode PetscBagSetDestroy(PetscBag*,PetscErrorCode (*)(PetscBag*));

#endif
