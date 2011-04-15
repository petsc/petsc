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
#define DMIGA       "iga"

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
extern PetscErrorCode   DMDestroy(DM*);
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
extern PetscErrorCode   DMSetMatrixPreallocateOnly(DM,PetscBool);
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

/*-------------------------------------------------------------------------*/
/* ISMapping */
/*-------------------------------------------------------------------------*/
extern  PetscClassId IS_MAPPING_CLASSID;
extern PetscErrorCode  ISMappingInitializePackage(const char[]);
/*S
   ISMapping -   a generalization of ISLocalToGlobalMapping
               maps from a domain [0,M) of indices to a range [0,N) of indices.  
               The mapping can be multivalued and can be thought of as a directed 
               graph with the start and end vertices drawn from the domain and range, 
               respectively. In the simplest case, an ISMapping is specified by pairs of ISs 
               of equal length prescribing the endpoints of as set of graph edges. 

                 The domain is partitioned in parallel into local ownership ranges, the same way 
               as a Vec's indices. Since this is equivalent to specifying a PetscLayout, the domain
               is said to be "laid out". Once the edges have been specified, the ISMapping is 
               assembled each rank has all of the edges with the source points in its ownership range.

                 After assembly, the mapping can be used to map the indices in the local ownership
               range [m_p, m_{p+1}) to the global range indices on the other end of the edges.  
               Similarly, local indices from [0,m_{p+1}-m_p) can be mapped to the corresponding
               global range indices. 
                 Unlike with ISLocalToGlobalMapping, an ISMapping can be multivalued and some local 
               indices might have empty images.  Because of that the output array resulting from the 
               application of the mapping to an input array of length m is supplemented with an offset
               array of size m+1 to delineate the images of the consecuitive input indices. 
                 In addition to mapping just indices, indices together with  scalar arrays (of equal 
               sizes) can be mapped, with the scalar values simply "following" the input indices to 
               their images.  Since ISMappings are multivalued in general, the scalar values will be 
               replicated.  This is useful for employing ISMappings in VecSetValuesLocal or 
               MatSetValuesLocal. 


   Level: intermediate

.seealso:  ISMappingCreate(), ISMappingSetDomainSizes(), ISMappingApplyLocal(), ISMappingApplyWithValuesLocal()
S*/
typedef struct _p_ISMapping *ISMapping;

extern PetscErrorCode  ISMappingRegister(const char[],const char[],const char[],PetscErrorCode (*)(ISMapping));

/*MC
   ISMappingRegisterDynamic - Adds a method to the ISMapping registry.

   Synopsis:
   PetscErrorCode ISMappingRegisterDynamic(const char *name_mapping,const char *path,const char *name_create,PetscErrorCode (*routine_create)(ISMapping))

   Not Collective

   Input Parameters:
+  name_mapping - name of a new user-defined mapping module
.  path - path (either absolute or relative) the library containing this mapping
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Level: developer

   Notes:
   ISMappingRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   ISMappingRegisterDynamic("my_mapping",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyMappingCreate",MyMappingCreate);
.ve

   Then, your mapping can be chosen with the procedural interface via
$     ISMappingSetType(mfctx,"my_mapping")
   or at runtime via the option
$     -is_mapping_type my_mapping

.keywords: ISMapping, register

.seealso: ISMappingRegisterAll(), ISMappingRegisterDestroy()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define ISMappingRegisterDynamic(a,b,c,d) ISMappingRegister(a,b,c,0)
#else
#define ISMappingRegisterDynamic(a,b,c,d) ISMappingRegister(a,b,c,d)
#endif

extern PetscErrorCode  ISMappingRegisterAll(const char[]);
extern PetscErrorCode  ISMappingRegisterDestroy(void);

/* 
 Only one real type for now. 
 Will wrap sparse Mat and VecScatter objects as ISMappings in the future. 
 */
#define ISMappingType char*
#define IS_MAPPING_IS      "ISMappingIS"

extern  PetscErrorCode ISMappingCreate(MPI_Comm comm, ISMapping *mapping);
extern  PetscErrorCode ISMappingView(ISMapping mapping, PetscViewer viewer);
extern  PetscErrorCode ISMappingDestroy(ISMapping *mapping);
extern  PetscErrorCode ISMappingSetType(ISMapping mapping, const ISMappingType maptype); 
extern  PetscErrorCode ISMappingSetSizes(ISMapping mapping, PetscInt m, PetscInt n, PetscInt M, PetscInt N);
extern  PetscErrorCode ISMappingGetSizes(ISMapping mapping, PetscInt *m, PetscInt *n, PetscInt *M, PetscInt *N);

extern  PetscErrorCode ISMappingSetUp(ISMapping mapping);
extern  PetscErrorCode ISMappingAssemblyBegin(ISMapping mapping);
extern  PetscErrorCode ISMappingAssemblyEnd(ISMapping mapping);

extern  PetscErrorCode ISMappingGetSupportIS(ISMapping mapping, IS *supp);
extern  PetscErrorCode ISMappingGetImageIS(ISMapping mapping, IS *image);
extern  PetscErrorCode ISMappingGetSupportSizeLocal(ISMapping mapping, PetscInt *supp_size);
extern  PetscErrorCode ISMappingGetImageSizeLocal(ISMapping mapping, PetscInt *image_size);
extern  PetscErrorCode ISMappingGetMaxImageSizeLocal(ISMapping mapping, PetscInt *max_image_size);
extern  PetscErrorCode ISMappingISGetEdges(ISMapping mapping, IS *ix, IS *iy);

extern  PetscErrorCode ISMappingMapIndicesLocal(ISMapping mapping, PetscInt insize, const PetscInt inidx[], PetscInt *outsize, PetscInt outidx[], PetscInt offsets[]);
extern PetscErrorCode ISMappingMapValuesLocal(ISMapping map, PetscInt insize, const PetscInt inidx[], const PetscScalar invals[], PetscInt *outsize, PetscInt outidx[], PetscScalar outvals[], PetscInt offsets[]);
extern  PetscErrorCode ISMappingBinIndicesLocal(ISMapping mapping, PetscInt insize, const PetscInt inidx[], PetscInt *outsize, PetscInt outidx[], PetscInt offsets[]);
extern PetscErrorCode ISMappingBinValuesLocal(ISMapping map, PetscInt insize, const PetscInt inidx[], const PetscScalar invals[], PetscInt *outsize, PetscInt outidx[], PetscScalar outvals[], PetscInt offsets[]);

extern  PetscErrorCode ISMappingMapIndices(ISMapping mapping, PetscInt insize, const PetscInt inidx[], PetscInt *outsize, PetscInt outidx[], PetscInt offsets[], PetscBool drop);
extern PetscErrorCode ISMappingMapValues(ISMapping map, PetscInt insize, const PetscInt inidx[], const PetscScalar invals[], PetscInt *outsize, PetscInt outidx[], PetscScalar outvals[], PetscInt offsets[], PetscBool drop);
extern  PetscErrorCode ISMappingBinIndices(ISMapping mapping, PetscInt insize, const PetscInt inidx[], PetscInt *outsize, PetscInt outidx[], PetscInt offsets[], PetscBool drop);
extern PetscErrorCode ISMappingBinValues(ISMapping map, PetscInt insize, const PetscInt inidx[], const PetscScalar invals[], PetscInt *outsize, PetscInt outidx[], PetscScalar outvals[], PetscInt offsets[], PetscBool drop);

extern PetscErrorCode ISMappingInvert(ISMapping mapping, ISMapping *imapping);
extern PetscErrorCode ISMappingPullback(ISMapping mapping1, ISMapping mapping2, ISMapping *mapping);
extern PetscErrorCode ISMappingPushforward(ISMapping mapping1, ISMapping mapping2, ISMapping *mapping);

extern PetscErrorCode ISMappingGetOperator(ISMapping mapping, Mat *op);

/* IS_MAPPING_IS */
extern  PetscErrorCode ISMappingISSetEdges(ISMapping mapping, IS ix, IS iy);


PETSC_EXTERN_CXX_END
#endif
