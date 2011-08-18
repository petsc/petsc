/*
      SA is a sparse array indexed by i, optionally by j, with an optional weight. SAMapping is a mapping of SAs.
*/
#if !defined(__PETSCSA_H)
#define __PETSCSA_H
#include "petscmat.h"
PETSC_EXTERN_CXX_BEGIN
extern PetscErrorCode  SAMappingInitializePackage(const char[]);

/*-------------------------------------------------------------------------*/
/* SA: Int-Scalar-Int sparse arrays. */
/*-------------------------------------------------------------------------*/
/*S
   SA -   a sparse array of indices with optional scalar weights.

   Level: advanced

.seealso:  SACreate(), SADestroy(), SAAddData(), SAGetLength(), SAGetData(), SAMapping
S*/
typedef struct _n_SA *SA;

typedef enum{SA_I = 1, SA_J = 2} SAIndex;
typedef PetscInt SAComponents;
#define SA_W 4

extern PetscErrorCode SACreate(SAComponents, SA*);
extern PetscErrorCode SACreateArrays(PetscInt, PetscInt, SA **);
extern PetscErrorCode SAClear(SA);
extern PetscErrorCode SADuplicate(SA, SA*);
extern PetscErrorCode SADestroy(SA*);
extern PetscErrorCode SAAddArray(SA, SA);
extern PetscErrorCode SAAddData(SA, PetscInt, const PetscInt*, const PetscScalar*, const PetscInt*);
extern PetscErrorCode SAAddI(SA, PetscInt, PetscInt, const PetscScalar*, const PetscInt*);
extern PetscErrorCode SAAddJ(SA, PetscInt, const PetscInt*, const PetscScalar*, PetscInt);
extern PetscErrorCode SAGetLength(SA, PetscInt*);
extern PetscErrorCode SAGetData(SA, PetscInt*, PetscScalar*, PetscInt*);

/*-------------------------------------------------------------------------*/
/* SAMapping: Int-Scalar sparse array maps. */
/*-------------------------------------------------------------------------*/
extern  PetscClassId SA_MAPPING_CLASSID;
/*S
   SAMapping -   a generalization of ISLocalToGlobalMapping
               maps from a domain [0,M) of indices to a range [0,N) of indices.  
               The mapping can be multivalued and can be thought of as a directed 
               graph with the start and end vertices drawn from the domain and range, 
               respectively. In the simplest case, an SAMapping is specified by pairs of ISs 
               of equal length prescribing the endpoints of as set of graph edges. 

                 The domain is partitioned in parallel into local ownership ranges, the same way 
               as a Vec's indices. Since this is equivalent to specifying a PetscLayout, the domain
               is said to be "laid out". Once the edges have been specified, the SAMapping is 
               assembled each rank has all of the edges with the source points in its ownership range.

                 After assembly, the mapping can be used to map the indices in the local ownership
               range [m_p, m_{p+1}) to the global range indices on the other end of the edges.  
               Similarly, local indices from [0,m_{p+1}-m_p) can be mapped to the corresponding
               global range indices. 
                 Unlike with ISLocalToGlobalMapping, an SAMapping can be multivalued and some local 
               indices might have empty images.  Because of that the output array resulting from the 
               application of the mapping to an input array of length m is supplemented with an offset
               array of size m+1 to delineate the images of the consecuitive input indices. 
                 In addition to mapping just indices, indices together with  scalar arrays (of equal 
               sizes) can be mapped, with the scalar values simply "following" the input indices to 
               their images.  Since SAMappings are multivalued in general, the scalar values will be 
               replicated.  This is useful for employing SAMappings in VecSetValuesLocal or 
               MatSetValuesLocal. 


   Level: intermediate

.seealso:  SAMappingCreate(), SAMappingSetDomainSizes(), SAMappingApplyLocal(), SAMappingApplyWithValuesLocal()
S*/
typedef struct _p_SAMapping *SAMapping;

extern PetscErrorCode  SAMappingRegister(const char[],const char[],const char[],PetscErrorCode (*)(SAMapping));

/*MC
   SAMappingRegisterDynamic - Adds a method to the SAMapping registry.

   Synopsis:
   PetscErrorCode SAMappingRegisterDynamic(const char *name_mapping,const char *path,const char *name_create,PetscErrorCode (*routine_create)(SAMapping))

   Not Collective

   Input Parameters:
+  name_mapping - name of a new user-defined mapping module
.  path - path (either absolute or relative) the library containing this mapping
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Level: developer

   Notes:
   SAMappingRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   SAMappingRegisterDynamic("my_mapping",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyMappingCreate",MyMappingCreate);
.ve

   Then, your mapping can be chosen with the procedural interface via
$     SAMappingSetType(mfctx,"my_mapping")
   or at runtime via the option
$     -is_mapping_type my_mapping

.keywords: SAMapping, register

.seealso: SAMappingRegisterAll(), SAMappingRegisterDestroy()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define SAMappingRegisterDynamic(a,b,c,d) SAMappingRegister(a,b,c,0)
#else
#define SAMappingRegisterDynamic(a,b,c,d) SAMappingRegister(a,b,c,d)
#endif

extern PetscErrorCode  SAMappingRegisterAll(const char[]);
extern PetscErrorCode  SAMappingRegisterDestroy(void);

/* 
 Only one impl type for now. 
 Will wrap sparse Mat and VecScatter objects as SAMappings in the future. 
 */
#define SAMappingType char*
#define SA_MAPPING_GRAPH     "SAMappingGraph"

extern  PetscErrorCode SAMappingCreate(MPI_Comm, SAMapping *);
extern  PetscErrorCode SAMappingView(SAMapping , PetscViewer viewer);
extern  PetscErrorCode SAMappingDestroy(SAMapping *);
extern  PetscErrorCode SAMappingSetType(SAMapping , const SAMappingType maptype); 
extern  PetscErrorCode SAMappingSetSizes(SAMapping , PetscInt m, PetscInt n, PetscInt M, PetscInt N);
extern  PetscErrorCode SAMappingGetSizes(SAMapping , PetscInt *m, PetscInt *n, PetscInt *M, PetscInt *N);

extern  PetscErrorCode SAMappingSetUp(SAMapping );
extern  PetscErrorCode SAMappingAssemblyBegin(SAMapping );
extern  PetscErrorCode SAMappingAssemblyEnd(SAMapping );

extern  PetscErrorCode SAMappingGetSupport(SAMapping ,  PetscInt *, PetscInt *[]);
extern  PetscErrorCode SAMappingGetSupportIS(SAMapping , IS *);
extern  PetscErrorCode SAMappingGetSupportSA(SAMapping , SA *);
extern  PetscErrorCode SAMappingGetImage(SAMapping , PetscInt *, PetscInt *[]);
extern  PetscErrorCode SAMappingGetImageIS(SAMapping , IS *);
extern  PetscErrorCode SAMappingGetImageSA(SAMapping , SA *);
extern  PetscErrorCode SAMappingGetMaxImageSize(SAMapping , PetscInt*);

extern  PetscErrorCode SAMappingMapLocal(SAMapping , SA, SAIndex, SA);
extern  PetscErrorCode SAMappingBinLocal(SAMapping , SA, SAIndex, SA);
extern  PetscErrorCode SAMappingMap(SAMapping , SA, SAIndex, SA);
extern  PetscErrorCode SAMappingBin(SAMapping , SA, SAIndex, SA);

extern  PetscErrorCode SAMappingMapSplitLocal(SAMapping, SA , SAIndex , SA *);
extern  PetscErrorCode SAMappingBinSplitLocal(SAMapping, SA , SAIndex , SA *);
extern  PetscErrorCode SAMappingMapSplit(SAMapping, SA, SAIndex, SA *);
extern  PetscErrorCode SAMappingBinSplit(SAMapping, SA, SAIndex, SA *);

extern  PetscErrorCode SAMappingInvert(SAMapping, SAMapping *);
extern  PetscErrorCode SAMappingPullback(SAMapping, SAMapping, SAMapping *);
extern  PetscErrorCode SAMappingPushforward(SAMapping, SAMapping, SAMapping *);

extern  PetscErrorCode SAMappingGetOperator(SAMapping , Mat*);

/* SA_MAPPING_GRAPH */
extern  PetscErrorCode SAMappingGraphAddEdgesSA(SAMapping , SA);
extern  PetscErrorCode SAMappingGraphAddEdgesIS(SAMapping , IS, IS);
extern  PetscErrorCode SAMappingGraphAddEdges(SAMapping , PetscInt, const PetscInt[], const PetscInt[]);
extern  PetscErrorCode SAMappingGraphGetEdgesSA(SAMapping , SA *);
extern  PetscErrorCode SAMappingGraphGetEdgesIS(SAMapping , IS *, IS *);
extern  PetscErrorCode SAMappingGraphGetEdges(SAMapping, PetscInt *, PetscInt *[], PetscInt *[]);


PETSC_EXTERN_CXX_END
#endif
