/*
      ISMapping and friends: Int-Scalar Mapping is conceptually a graph that maps integers (and attached scalars) to other integers (and scalars).
      This is more basic than Mat, and can be thought of loosely as a (generally nonlinear) mapping of sparse vectors.
*/
#if !defined(__PETSCMAP_H)
#define __PETSCMAP_H
#include "petscmat.h"
PETSC_EXTERN_CXX_BEGIN
extern PetscErrorCode  ISMappingInitializePackage(const char[]);

/*-------------------------------------------------------------------------*/
/* ISArray: Int-Scalar sparse arrays. */
/*-------------------------------------------------------------------------*/
/*S
   ISArray -   a sparse array of indices with optional scalar weights.

   Level: advanced

.seealso:  ISMapping(), ISArrayCreate(), ISArrayDestroy(), ISArrayISWeighted(), ISArrayExpand(), ISArrayGetLength(), ISArrayGetIndices(), ISArrayGetValues(), ISArrayGetSegments()
S*/
typedef struct _n_ISArray *ISArray;

typedef enum{ISARRAY_I = 1, ISARRAY_J = 2} ISArrayIndex;
typedef PetscInt ISArrayComponents;
#define ISARRAY_W 4

extern PetscErrorCode ISArrayCreate(ISArrayComponents mask, ISArray *_chain);
extern PetscErrorCode ISArrayCreateArrays(PetscInt mask, PetscInt count, const PetscInt *lengths, ISArray **arrays);
extern PetscErrorCode ISArrayClear(ISArray chain);
extern PetscErrorCode ISArrayDuplicate(ISArray arr, ISArray *darr);
extern PetscErrorCode ISArrayDestroy(ISArray chain);
extern PetscErrorCode ISArrayAddData(ISArray chain, const PetscInt len, const PetscInt *ia, const PetscScalar *wa, const PetscInt *ja);
extern PetscErrorCode ISArrayAddI(ISArray chain, const PetscInt len, PetscInt i, const PetscScalar* wa, const PetscInt *ja);
extern PetscErrorCode ISArrayAddJ(ISArray chain, const PetscInt len, const PetscInt *ia, const PetscScalar* wa, PetscInt j);
extern PetscErrorCode ISArrayGetLength(ISArray chain, PetscInt *_length);
extern PetscErrorCode ISArrayGetData(ISArray chain, const PetscInt *ia[], const PetscScalar *wa[], const PetscInt *ja[]);




/*-------------------------------------------------------------------------*/
/* ISMapping: Int-Scalar sparse array maps. */
/*-------------------------------------------------------------------------*/
extern  PetscClassId IS_MAPPING_CLASSID;
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
 Only one impl type for now. 
 Will wrap sparse Mat and VecScatter objects as ISMappings in the future. 
 */
#define ISMappingType char*
#define IS_MAPPING_GRAPH     "ISMappingGraph"

extern  PetscErrorCode ISMappingCreate(MPI_Comm comm, ISMapping *mapping);
extern  PetscErrorCode ISMappingView(ISMapping mapping, PetscViewer viewer);
extern  PetscErrorCode ISMappingDestroy(ISMapping mapping);
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

extern  PetscErrorCode ISMappingMapLocal(ISMapping mapping, ISArray inarr, ISArrayIndex index, ISArray outarr);
extern  PetscErrorCode ISMappingBinLocal(ISMapping mapping, ISArray array, ISArrayIndex index, ISArray outarr);
extern  PetscErrorCode ISMappingMap(ISMapping mapping, ISArray inarr, ISArrayIndex index, ISArray outarr);
extern  PetscErrorCode ISMappingBin(ISMapping mapping, ISArray array, ISArrayIndex index, ISArray outarr);

extern  PetscErrorCode ISMappingMapSplitLocal(ISMapping mapping, ISArray inarr, ISArrayIndex index, ISArray *arrs);
extern  PetscErrorCode ISMappingBinSplitLocal(ISMapping mapping, ISArray array, ISArrayIndex index, ISArray *bins);
extern  PetscErrorCode ISMappingMapSplit(ISMapping mapping, ISArray inarr, ISArrayIndex index, ISArray *arrs);
extern  PetscErrorCode ISMappingBinSplit(ISMapping mapping, ISArray array, ISArrayIndex index, ISArray *bins);



extern  PetscErrorCode ISMappingInvert(ISMapping mapping, ISMapping *imapping);
extern  PetscErrorCode ISMappingPullback(ISMapping mapping1, ISMapping mapping2, ISMapping *mapping);
extern  PetscErrorCode ISMappingPushforward(ISMapping mapping1, ISMapping mapping2, ISMapping *mapping);

extern  PetscErrorCode ISMappingGetOperator(ISMapping mapping, Mat *op);

/* IS_MAPPING_GRAPH */
extern  PetscErrorCode ISMappingGraphAddEdgeArray(ISMapping mapping, ISArray edges);
extern  PetscErrorCode ISMappingGraphAddEdges(ISMapping mapping, PetscInt len, const PetscInt x[], const PetscInt y[]);
extern  PetscErrorCode ISMappingGraphGetEdgeArray(ISMapping mapping, ISArray edges);


PETSC_EXTERN_CXX_END
#endif
