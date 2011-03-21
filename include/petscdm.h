/*
      Objects to manage the interactions between the mesh data structures and the algebraic objects
*/
#if !defined(__PETSCDM_H)
#define __PETSCDM_H
#include "petscmat.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscErrorCode  DMInitializePackage(const char[]);
/*S
     DM - Abstract PETSc object that manages an abstract grid object

   Level: intermediate

  Concepts: grids, grid refinement

   Notes: The DMDACreate() based object and the DMCompositeCreate() based object are examples of DMs

          Though the DM objects require the petscsnes.h include files the DM library is
    NOT dependent on the SNES or KSP library. In fact, the KSP and SNES libraries depend on
    DM. (This is not great design, but not trivial to fix).

.seealso:  DMCompositeCreate(), DMDACreate()
S*/
typedef struct _p_DM* DM;

extern PetscClassId  DM_CLASSID;

/*E
    DMType - String with the name of a PETSc DM or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:myveccreate()

   Level: beginner

.seealso: DMSetType(), DM
E*/

#define DMType char*
#define DMDA        "da"
#define DMADDA      "adda"
#define DMCOMPOSITE "composite"
#define DMSLICED    "sliced"
#define DMMESH      "mesh"
#define DMCARTESIAN "cartesian"

extern PetscFList DMList;
extern PetscBool  DMRegisterAllCalled;
extern PetscErrorCode  DMCreate(MPI_Comm,DM*);
extern PetscErrorCode  DMSetType(DM, const DMType);
extern PetscErrorCode  DMGetType(DM, const DMType *);
extern PetscErrorCode  DMRegister(const char[],const char[],const char[],PetscErrorCode (*)(DM));
extern PetscErrorCode  DMRegisterAll(const char []);
extern PetscErrorCode  DMRegisterDestroy(void);

/*MC
  DMRegisterDynamic - Adds a new DM component implementation

  Synopsis:
  PetscErrorCode DMRegisterDynamic(const char *name,const char *path,const char *func_name, PetscErrorCode (*create_func)(DM))

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
. path        - The path (either absolute or relative) of the library containing this routine
. func_name   - The name of routine to create method context
- create_func - The creation routine itself

  Notes:
  DMRegisterDynamic() may be called multiple times to add several user-defined DMs

  If dynamic libraries are used, then the fourth input argument (routine_create) is ignored.

  Sample usage:
.vb
    DMRegisterDynamic("my_da","/home/username/my_lib/lib/libO/solaris/libmy.a", "MyDMCreate", MyDMCreate);
.ve

  Then, your DM type can be chosen with the procedural interface via
.vb
    DMCreate(MPI_Comm, DM *);
    DMSetType(DM,"my_da_name");
.ve
   or at runtime via the option
.vb
    -da_type my_da_name
.ve

  Notes: $PETSC_ARCH occuring in pathname will be replaced with appropriate values.
         If your function is not being put into a shared library then use DMRegister() instead

  Level: advanced

.keywords: DM, register
.seealso: DMRegisterAll(), DMRegisterDestroy(), DMRegister()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define DMRegisterDynamic(a,b,c,d) DMRegister(a,b,c,0)
#else
#define DMRegisterDynamic(a,b,c,d) DMRegister(a,b,c,d)
#endif

extern PetscErrorCode   DMView(DM,PetscViewer);
extern PetscErrorCode   DMDestroy(DM);
extern PetscErrorCode   DMCreateGlobalVector(DM,Vec*);
extern PetscErrorCode   DMCreateLocalVector(DM,Vec*);
extern PetscErrorCode   DMGetLocalVector(DM,Vec *);
extern PetscErrorCode   DMRestoreLocalVector(DM,Vec *);
extern PetscErrorCode   DMGetGlobalVector(DM,Vec *);
extern PetscErrorCode   DMRestoreGlobalVector(DM,Vec *);
extern PetscErrorCode   DMGetLocalToGlobalMapping(DM,ISLocalToGlobalMapping*);
extern PetscErrorCode   DMGetLocalToGlobalMappingBlock(DM,ISLocalToGlobalMapping*);
extern PetscErrorCode   DMGetBlockSize(DM,PetscInt*);
extern PetscErrorCode   DMGetColoring(DM,ISColoringType,const MatType,ISColoring*);
extern PetscErrorCode   DMGetMatrix(DM, const MatType,Mat*);
extern PetscErrorCode   DMGetInterpolation(DM,DM,Mat*,Vec*);
extern PetscErrorCode   DMGetInjection(DM,DM,VecScatter*);
extern PetscErrorCode   DMRefine(DM,MPI_Comm,DM*);
extern PetscErrorCode   DMCoarsen(DM,MPI_Comm,DM*);
extern PetscErrorCode   DMRefineHierarchy(DM,PetscInt,DM[]);
extern PetscErrorCode   DMCoarsenHierarchy(DM,PetscInt,DM[]);
extern PetscErrorCode   DMSetFromOptions(DM);
extern PetscErrorCode   DMSetUp(DM);
extern PetscErrorCode   DMGetInterpolationScale(DM,DM,Mat,Vec*);
extern PetscErrorCode   DMGetAggregates(DM,DM,Mat*);
extern PetscErrorCode   DMGlobalToLocalBegin(DM,Vec,InsertMode,Vec);
extern PetscErrorCode   DMGlobalToLocalEnd(DM,Vec,InsertMode,Vec);
extern PetscErrorCode   DMLocalToGlobalBegin(DM,Vec,InsertMode,Vec);
extern PetscErrorCode   DMLocalToGlobalEnd(DM,Vec,InsertMode,Vec);
extern PetscErrorCode   DMGetElements(DM,PetscInt *,PetscInt *,const PetscInt*[]);
extern PetscErrorCode   DMRestoreElements(DM,PetscInt *,PetscInt *,const PetscInt*[]);
extern PetscErrorCode   DMConvert(DM,const DMType,DM*);

extern PetscErrorCode   DMSetOptionsPrefix(DM,const char []);
extern PetscErrorCode   DMSetVecType(DM,const VecType);
extern PetscErrorCode   DMSetContext(DM,void*);
extern PetscErrorCode   DMGetContext(DM,void**);
extern PetscErrorCode   DMSetInitialGuess(DM,PetscErrorCode (*)(DM,Vec));
extern PetscErrorCode   DMSetFunction(DM,PetscErrorCode (*)(DM,Vec,Vec));
extern PetscErrorCode   DMSetJacobian(DM,PetscErrorCode (*)(DM,Vec,Mat,Mat,MatStructure *));
extern PetscErrorCode   DMHasInitialGuess(DM,PetscBool *);
extern PetscErrorCode   DMHasFunction(DM,PetscBool *);
extern PetscErrorCode   DMHasJacobian(DM,PetscBool *);
extern PetscErrorCode   DMComputeInitialGuess(DM,Vec);
extern PetscErrorCode   DMComputeFunction(DM,Vec,Vec);
extern PetscErrorCode   DMComputeJacobian(DM,Vec,Mat,Mat,MatStructure *);
extern PetscErrorCode   DMComputeJacobianDefault(DM,Vec,Mat,Mat,MatStructure *);
extern PetscErrorCode   DMFinalizePackage(void);

typedef struct NLF_DAAD* NLF;

#include "petscbag.h"

extern PetscErrorCode  PetscViewerBinaryMatlabOpen(MPI_Comm, const char [], PetscViewer*);
extern PetscErrorCode  PetscViewerBinaryMatlabDestroy(PetscViewer);
extern PetscErrorCode  PetscViewerBinaryMatlabOutputBag(PetscViewer, const char [], PetscBag);
extern PetscErrorCode  PetscViewerBinaryMatlabOutputVec(PetscViewer, const char [], Vec);
extern PetscErrorCode  PetscViewerBinaryMatlabOutputVecDA(PetscViewer, const char [], Vec, DM);

PETSC_EXTERN_CXX_END
#endif
