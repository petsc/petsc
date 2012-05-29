/*
      Objects to manage the interactions between the mesh data structures and the algebraic objects
*/
#if !defined(__PETSCDM_H)
#define __PETSCDM_H
#include "petscmat.h"

PETSC_EXTERN PetscErrorCode DMInitializePackage(const char[]);
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

PETSC_EXTERN PetscClassId DM_CLASSID;

/*J
    DMType - String with the name of a PETSc DM or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:myveccreate()

   Level: beginner

.seealso: DMSetType(), DM
J*/
#define DMType char*
#define DMDA        "da"
#define DMADDA      "adda"
#define DMCOMPOSITE "composite"
#define DMSLICED    "sliced"
#define DMSHELL     "shell"
#define DMMESH      "mesh"
#define DMCOMPLEX   "complex"
#define DMCARTESIAN "cartesian"
#define DMIGA       "iga"
#define DMREDUNDANT "redundant"
#define DMAKKT      "akkt"

PETSC_EXTERN PetscFList DMList;
PETSC_EXTERN PetscBool DMRegisterAllCalled;
PETSC_EXTERN PetscErrorCode DMCreate(MPI_Comm,DM*);
PETSC_EXTERN PetscErrorCode DMSetType(DM, const DMType);
PETSC_EXTERN PetscErrorCode DMGetType(DM, const DMType *);
PETSC_EXTERN PetscErrorCode DMRegister(const char[],const char[],const char[],PetscErrorCode (*)(DM));
PETSC_EXTERN PetscErrorCode DMRegisterAll(const char []);
PETSC_EXTERN PetscErrorCode DMRegisterDestroy(void);


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

PETSC_EXTERN PetscErrorCode DMView(DM,PetscViewer);
PETSC_EXTERN PetscErrorCode DMLoad(DM,PetscViewer);
PETSC_EXTERN PetscErrorCode DMDestroy(DM*);
PETSC_EXTERN PetscErrorCode DMCreateGlobalVector(DM,Vec*);
PETSC_EXTERN PetscErrorCode DMCreateLocalVector(DM,Vec*);
PETSC_EXTERN PetscErrorCode DMGetLocalVector(DM,Vec *);
PETSC_EXTERN PetscErrorCode DMRestoreLocalVector(DM,Vec *);
PETSC_EXTERN PetscErrorCode DMGetGlobalVector(DM,Vec *);
PETSC_EXTERN PetscErrorCode DMRestoreGlobalVector(DM,Vec *);
PETSC_EXTERN PetscErrorCode DMClearGlobalVectors(DM);
PETSC_EXTERN PetscErrorCode DMGetNamedGlobalVector(DM,const char*,Vec*);
PETSC_EXTERN PetscErrorCode DMRestoreNamedGlobalVector(DM,const char*,Vec*);
PETSC_EXTERN PetscErrorCode DMGetLocalToGlobalMapping(DM,ISLocalToGlobalMapping*);
PETSC_EXTERN PetscErrorCode DMGetLocalToGlobalMappingBlock(DM,ISLocalToGlobalMapping*);
PETSC_EXTERN PetscErrorCode DMCreateFieldIS(DM,PetscInt*,char***,IS**);
PETSC_EXTERN PetscErrorCode DMGetBlockSize(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMCreateColoring(DM,ISColoringType,const MatType,ISColoring*);
PETSC_EXTERN PetscErrorCode DMCreateMatrix(DM,const MatType,Mat*);
PETSC_EXTERN PetscErrorCode DMSetMatrixPreallocateOnly(DM,PetscBool);
PETSC_EXTERN PetscErrorCode DMCreateInterpolation(DM,DM,Mat*,Vec*);
PETSC_EXTERN PetscErrorCode DMCreateInjection(DM,DM,VecScatter*);
PETSC_EXTERN PetscErrorCode DMGetWorkArray(DM,PetscInt,PetscScalar**);
PETSC_EXTERN PetscErrorCode DMRefine(DM,MPI_Comm,DM*);
PETSC_EXTERN PetscErrorCode DMCoarsen(DM,MPI_Comm,DM*);
PETSC_EXTERN PetscErrorCode DMRefineHierarchy(DM,PetscInt,DM[]);
PETSC_EXTERN PetscErrorCode DMCoarsenHierarchy(DM,PetscInt,DM[]);
PETSC_EXTERN PetscErrorCode DMCoarsenHookAdd(DM,PetscErrorCode (*)(DM,DM,void*),PetscErrorCode (*)(DM,Mat,Vec,Mat,DM,void*),void*);
PETSC_EXTERN PetscErrorCode DMRefineHookAdd(DM,PetscErrorCode (*)(DM,DM,void*),PetscErrorCode (*)(DM,Mat,DM,void*),void*);
PETSC_EXTERN PetscErrorCode DMRestrict(DM,Mat,Vec,Mat,DM);
PETSC_EXTERN PetscErrorCode DMInterpolate(DM,Mat,DM);
PETSC_EXTERN PetscErrorCode DMSetFromOptions(DM);
PETSC_EXTERN PetscErrorCode DMSetUp(DM);
PETSC_EXTERN PetscErrorCode DMCreateInterpolationScale(DM,DM,Mat,Vec*);
PETSC_EXTERN PetscErrorCode DMCreateAggregates(DM,DM,Mat*);
PETSC_EXTERN PetscErrorCode DMGlobalToLocalBegin(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMGlobalToLocalEnd(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMLocalToGlobalBegin(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMLocalToGlobalEnd(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMConvert(DM,const DMType,DM*);

PETSC_EXTERN PetscErrorCode DMSetOptionsPrefix(DM,const char []);
PETSC_EXTERN PetscErrorCode DMSetVecType(DM,const VecType);
PETSC_EXTERN PetscErrorCode DMSetMatType(DM,const MatType);
PETSC_EXTERN PetscErrorCode DMSetApplicationContext(DM,void*);
PETSC_EXTERN PetscErrorCode DMSetApplicationContextDestroy(DM,PetscErrorCode (*)(void**));
PETSC_EXTERN PetscErrorCode DMGetApplicationContext(DM,void*);
PETSC_EXTERN PetscErrorCode DMSetInitialGuess(DM,PetscErrorCode (*)(DM,Vec));
PETSC_EXTERN PetscErrorCode DMSetFunction(DM,PetscErrorCode (*)(DM,Vec,Vec));
PETSC_EXTERN PetscErrorCode DMSetJacobian(DM,PetscErrorCode (*)(DM,Vec,Mat,Mat,MatStructure *));
PETSC_EXTERN PetscErrorCode DMSetVariableBounds(DM,PetscErrorCode (*)(DM,Vec,Vec));
PETSC_EXTERN PetscErrorCode DMHasInitialGuess(DM,PetscBool *);
PETSC_EXTERN PetscErrorCode DMHasFunction(DM,PetscBool *);
PETSC_EXTERN PetscErrorCode DMHasJacobian(DM,PetscBool *);
PETSC_EXTERN PetscErrorCode DMHasVariableBounds(DM,PetscBool *);
PETSC_EXTERN PetscErrorCode DMComputeInitialGuess(DM,Vec);
PETSC_EXTERN PetscErrorCode DMComputeFunction(DM,Vec,Vec);
PETSC_EXTERN PetscErrorCode DMComputeJacobian(DM,Vec,Mat,Mat,MatStructure *);
PETSC_EXTERN PetscErrorCode DMComputeJacobianDefault(DM,Vec,Mat,Mat,MatStructure *);
PETSC_EXTERN PetscErrorCode DMComputeVariableBounds(DM,Vec,Vec);

PETSC_EXTERN PetscErrorCode DMCreateFieldDecompositionDM(DM,const char*,DM*);
PETSC_EXTERN PetscErrorCode DMCreateFieldDecomposition(DM,PetscInt*,char***,IS**,DM**);
PETSC_EXTERN PetscErrorCode DMCreateDomainDecompositionDM(DM,const char*,DM*);
PETSC_EXTERN PetscErrorCode DMCreateDomainDecomposition(DM,PetscInt*,char***,IS**,IS**,DM**);

PETSC_EXTERN PetscErrorCode DMGetRefineLevel(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMGetCoarsenLevel(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMFinalizePackage(void);

typedef struct NLF_DAAD* NLF;

#include "petscbag.h"

PETSC_EXTERN PetscErrorCode PetscViewerBinaryMatlabOpen(MPI_Comm, const char [], PetscViewer*);
PETSC_EXTERN PetscErrorCode PetscViewerBinaryMatlabDestroy(PetscViewer*);
PETSC_EXTERN PetscErrorCode PetscViewerBinaryMatlabOutputBag(PetscViewer, const char [], PetscBag);
PETSC_EXTERN PetscErrorCode PetscViewerBinaryMatlabOutputVec(PetscViewer, const char [], Vec);
PETSC_EXTERN PetscErrorCode PetscViewerBinaryMatlabOutputVecDA(PetscViewer, const char [], Vec, DM);

#define DM_FILE_CLASSID 1211221

/* FEM support */
PETSC_EXTERN PetscErrorCode DMGetLocalFunction(DM, PetscErrorCode (**)(DM, Vec, Vec, void *));
PETSC_EXTERN PetscErrorCode DMSetLocalFunction(DM, PetscErrorCode (*)(DM, Vec, Vec, void *));
PETSC_EXTERN PetscErrorCode DMGetLocalJacobian(DM, PetscErrorCode (**)(DM, Vec, Mat, Mat, void *));
PETSC_EXTERN PetscErrorCode DMSetLocalJacobian(DM, PetscErrorCode (*)(DM, Vec, Mat, Mat, void *));
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*DMLocalFunction1)(DM, Vec, Vec, void*);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*DMLocalJacobian1)(DM, Vec, Mat, Mat, void*);
PETSC_EXTERN PetscErrorCode DMPrintCellVector(PetscInt, const char [], PetscInt, const PetscScalar []);
PETSC_EXTERN PetscErrorCode DMPrintCellMatrix(PetscInt, const char [], PetscInt, PetscInt, const PetscScalar []);

PETSC_EXTERN PetscErrorCode DMGetDefaultSection(DM, PetscSection *);
PETSC_EXTERN PetscErrorCode DMSetDefaultSection(DM, PetscSection);
PETSC_EXTERN PetscErrorCode DMGetDefaultGlobalSection(DM, PetscSection *);
PETSC_EXTERN PetscErrorCode DMGetDefaultSF(DM, PetscSF *);
PETSC_EXTERN PetscErrorCode DMSetDefaultSF(DM, PetscSF);
PETSC_EXTERN PetscErrorCode DMCreateDefaultSF(DM, PetscSection, PetscSection);

typedef struct {
  PetscInt         numQuadPoints; /* The number of quadrature points on an element */
  const PetscReal *quadPoints;    /* The quadrature point coordinates */
  const PetscReal *quadWeights;   /* The quadrature weights */
  PetscInt         numBasisFuncs; /* The number of finite element basis functions on an element */
  PetscInt         numComponents; /* The number of components for each basis function */
  const PetscReal *basis;         /* The basis functions tabulated at the quadrature points */
  const PetscReal *basisDer;      /* The basis function derivatives tabulated at the quadrature points */
} PetscQuadrature;
#endif
