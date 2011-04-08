#define PETSCDM_DLL

#include "private/ismapimpl.h"  /*I "petscdm.h"  I*/


PetscClassId  IS_MAPPING_CLASSID;
PetscLogEvent IS_MAPPING_Map, IS_MAPPING_Bin, IS_MAPPING_AssemblyBegin, IS_MAPPING_AssemblyEnd, IS_MAPPING_Invert, IS_MAPPING_Pushforward, IS_MAPPING_Pullback;

PetscFList ISMappingList               = PETSC_NULL;
PetscBool  ISMappingRegisterAllCalled  = PETSC_FALSE;
PetscBool  ISMappingPackageInitialized = PETSC_FALSE;

extern PetscErrorCode  ISMappingRegisterAll(const char *path);


#undef __FUNCT__  
#define __FUNCT__ "ISMappingFinalizePackage"
/*@C
  ISMappingFinalizePackage - This function destroys everything in the ISMapping package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode  ISMappingFinalizePackage(void)
{
  PetscFunctionBegin;
  ISMappingPackageInitialized = PETSC_FALSE;
  ISMappingRegisterAllCalled  = PETSC_FALSE;
  ISMappingList               = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingInitializePackage"
/*@C
  ISMappingInitializePackage - This function initializes everything in the ISMapping package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to ISMappingCreate()
  when using static libraries.

  Input Parameter:
. path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: ISMapping, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode  ISMappingInitializePackage(const char path[])
{
  char              logList[256];
  char              *className;
  PetscBool         opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (ISMappingPackageInitialized) PetscFunctionReturn(0);

  ISMappingPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("ISMapping",&IS_MAPPING_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = ISMappingRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("ISMappingMap", IS_MAPPING_CLASSID,&IS_MAPPING_Map);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ISMappingBin", IS_MAPPING_CLASSID,&IS_MAPPING_Bin);CHKERRQ(ierr);

  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "is_mapping", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(IS_MAPPING_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "is_mapping", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(IS_MAPPING_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(ISMappingFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingRegister"
/*@C
  ISMappingRegister - See ISMappingRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode  ISMappingRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(ISMapping))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&ISMappingList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "ISMappingRegisterDestroy"
/*@C
   ISMappingRegisterDestroy - Frees the list of ISMapping methods that were
   registered by ISMappingRegisterDynamic).

   Not Collective

   Level: developer

.keywords: ISMapping, register, destroy

.seealso: ISMappingRegisterDynamic), ISMappingRegisterAll()
@*/
PetscErrorCode  ISMappingRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&ISMappingList);CHKERRQ(ierr);
  ISMappingRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscErrorCode ISMappingCreate_IS(ISMapping);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "ISMappingRegisterAll"
/*@C
  ISMappingRegisterAll - Registers all of the mapping constructors in the ISMapping package.

  Not Collective

  Level: developer

.keywords: ISMapping, register, all

.seealso:  ISMappingRegisterDestroy(), ISMappingRegisterDynamic), ISMappingCreate(), 
           ISMappingSetType()
@*/
PetscErrorCode  ISMappingRegisterAll(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ISMappingRegisterAllCalled = PETSC_TRUE;
  ierr = ISMappingRegisterDynamic(IS_MAPPING_IS,path,"ISMappingCreate_IS",ISMappingCreate_IS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingMapIndicesLocal"
/*@
    ISMappingMapIndicesLocal - maps local indices from the rank's support to global indices from the rank's range.

    Not collective

    Input Parameters:
+   map    - mapping of indices
.   insize - size of the input array of indices
-   inidx  - input array of local indices from the rank's support


    Output Parameters:
+   outsize - number of mapped indices (PETSC_NULL, if not needed)
.   outidx  - array of global indices of size *outsize from the rank's support (PETSC_NULL, if not needed)
-   offsets - array of offsets of size inidx+1 delineating the images of individual input indices (PETSC_NULL, if not needed)

    Level: advanced

    Concepts: mapping^indices

.seealso: ISMappingGetSupport(), ISMappingGetImage(), ISMappingGetSupportSize(), ISMappingGetImageSize(),
          ISMappingMapIndices(), ISMappingMapValuesLocal(), ISMappingBinIndicesLocal(), ISMappingBinValuesLocal()

@*/
PetscErrorCode ISMappingMapIndicesLocal(ISMapping map, PetscInt insize, const PetscInt inidx[], PetscInt *outsize, PetscInt outidx[], PetscInt offsets[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_IS, 1);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  ISMappingCheckMethod(map,map->ops->maplocal,"ISMappingMapLocal");
  ierr = PetscLogEventBegin(IS_MAPPING_Map,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->maplocal)(map,insize,inidx,PETSC_NULL,outsize,outidx,PETSC_NULL,offsets,PETSC_FALSE); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_Map,map,0,0,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingMapIndices"
/*@
    ISMappingMapIndices - maps indices from the rank's support to global indices from the rank's range.

    Not collective

    Input Parameters:
+   map    - mapping of indices
.   insize - size of the input array of indices
.   inidx  - input array of indices from the rank's support
-   drop   - ignore indices outside of local support; generate an error otherwise


    Output Parameters:
+   outsize - number of mapped indices (PETSC_NULL, if not needed)
.   outidx  - array of global indices of size *outsize from the rank's support (PETSC_NULL, if not needed)
-   offsets - array of offsets of size inidx+1 delineating the images of individual input indices (PETSC_NULL, if not needed)

    Level: advanced

    Concepts: mapping^indices

.seealso: ISMappingGetSupport(), ISMappingGetImage(), ISMappingGetSupportSize(), ISMappingGetImageSize(),
          ISMappingMapIndicesLocal(), ISMappingMapValues(), ISMappingBinIndices(), ISMappingBinValues()

@*/
PetscErrorCode ISMappingMapIndices(ISMapping map, PetscInt insize, const PetscInt inidx[], PetscInt *outsize, PetscInt outidx[], PetscInt offsets[], PetscBool drop)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_IS,1);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  ISMappingCheckMethod(map,map->ops->map,"ISMappingMap");
  ierr = PetscLogEventBegin(IS_MAPPING_Map,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->map)(map,insize,inidx,PETSC_NULL,outsize,outidx,PETSC_NULL,offsets,drop); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_Map,map,0,0,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingMapValuesLocal"
/*@
    ISMappingMapValuesLocal - maps local indices from the rank's support together with associated values
                              to an array of indices and values.

    Not collective

    Input Parameters:
+   map    - mapping of indices
.   insize - size of the input array of global indices from the rank's support
.   inidx  - input array of local indices from the rank's support
-   invals - input array of scalar values corresponding to the indices in inidx (PETSC_NULL, if not needed)



    Output Parameters:
+   outsize - number of mapped indices (PETSC_NULL, if not needed)
.   outidx  - array of indices of size *outsize from the rank's range (PETSC_NULL, if not needed)
.   outvals - array of output values of size  *outsize (PETSC_NULL, if not needed or if invals is PETSC_NULL)
-   offsets - array of offsets of size inidx+1 delineating the images of individual input indices (PETSC_NULL, if not needed)

    Note: values are merely copied to the new locations prescribed by the location of the mapped indices.
          If both invals and outvals are PETSC_NULL, this is equivalent to ISMappingMapIndicesLocal().
    Level: advanced

    Concepts: mapping^indices and values

.seealso: ISMappingGetSupport(), ISMappingGetImage(), ISMappingGetSupportSize(), ISMappingGetImageSize(),
          ISMappingMapValues(), ISMappingMapIndicesLocal(), ISMappingBinIndicesLocal(), ISMappingBinValuesLocal()
@*/
PetscErrorCode ISMappingMapValuesLocal(ISMapping map, PetscInt insize, const PetscInt inidx[], const PetscScalar invals[], PetscInt *outsize, PetscInt outidx[], PetscScalar outvals[], PetscInt offsets[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_IS,1);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  ISMappingCheckMethod(map,map->ops->maplocal,"ISMappingMapLocal");
  ierr = PetscLogEventBegin(IS_MAPPING_Map,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->maplocal)(map,insize,inidx,invals,outsize,outidx,outvals,offsets,PETSC_FALSE); CHKERRQ(ierr);
  ierr = PetscLogEventBegin(IS_MAPPING_Map,map,0,0,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingMapValues"
/*@
    ISMappingMapValues - maps indices from the rank's support together with associated values
                         to an array of indices and values.

    Not collective

    Input Parameters:
+   map    - mapping of indices
.   insize - size of the input array of global indices from the rank's support
.   inidx  - input array of indices from the rank's support
.   invals - input array of scalar values corresponding to the indices in inidx (PETSC_NULL, if not needed)
-   drop   - ignore indices outside of local support; generate an error otherwise




    Output Parameters:
+   outsize - number of mapped indices (PETSC_NULL, if not needed)
.   outidx  - array of indices of size *outsize from the rank's range (PETSC_NULL, if not needed)
.   outvals - array of output values of size  *outsize (PETSC_NULL, if not needed or if invals is PETSC_NULL)
-   offsets - array of offsets of size inidx+1 delineating the images of individual input indices (PETSC_NULL, if not needed)

    Note: values are merely copied to the new locations prescribed by the location of the mapped indices.
          If both invals and outvals are PETSC_NULL, this is equivalent to ISMappingMapIndicesLocal().
    Level: advanced

    Concepts: mapping^indices and values

.seealso: ISMappingGetSupport(), ISMappingGetImage(), ISMappingGetSupportSize(), ISMappingGetImageSize(),
          ISMappingMapValuesLocal(), ISMappingMapIndices(), ISMappingBinIndices(), ISMappingBinValues()
@*/
PetscErrorCode ISMappingMapValues(ISMapping map, PetscInt insize, const PetscInt inidx[], const PetscScalar invals[], PetscInt *outsize, PetscInt outidx[], PetscScalar outvals[], PetscInt offsets[], PetscBool drop)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_IS,1);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  ISMappingCheckMethod(map,map->ops->map,"ISMappingMap");
  ierr = PetscLogEventBegin(IS_MAPPING_Map,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->map)(map,insize,inidx,invals,outsize,outidx,outvals,offsets,drop); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_Map,map,0,0,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingBinIndicesLocal"
/*@
    ISMappingBinIndicesLocal - group local indices from the rank's support into n groups or "bins" (some possibly empty)
                               according to which of the n image indices they are mapped to on this rank.
                               Since the mapping is potentially multivalued, the same index can appear in multiple bins.


    Not collective

    Input Parameters:
+   map    - mapping of indices
.   insize - size of the input array of local indices from the rank's support
-   inidx  - input array of local indices from the rank's support


    Output Parameters:
+   outsize - number of indices in all bins (PETSC_NULL, if not needed) 
.   outidx  - concatenated bins containing the indices from inidx arranged by bin (PETSC_NULL, if not needed)
-   offsets - array of offsets of size n+1 delineating the bins; n is the number of distinct image indices on this rank (PETSC_NULL, if not needed)

    Level: advanced

    Concepts: binning^local indices

.seealso: ISMappingGetSupport(), ISMappingGetImage(), ISMappingGetSupportSize(), ISMappingGetImageSize(),
          ISMappingBinIndices(), ISMappingMapIndicesLocal(), ISMappingMapValuesLocal(), ISMappingBinValuesLocal()

@*/
PetscErrorCode ISMappingBinIndicesLocal(ISMapping map, PetscInt insize, const PetscInt inidx[], PetscInt *outsize, PetscInt outidx[], PetscInt offsets[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_IS,1);
  ISMappingCheckAssembled(map,PETSC_TRUE, 1);
  ISMappingCheckMethod(map,map->ops->binlocal,"ISMappingBinLocal");
  ierr = PetscLogEventBegin(IS_MAPPING_Bin,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->binlocal)(map,insize,inidx,PETSC_NULL,outsize,outidx,PETSC_NULL,offsets,PETSC_FALSE); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_Bin,map,0,0,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingBinIndices"
/*@
    ISMappingBinIndices      - group indices from the rank's support into n groups or "bins" (some possibly empty)
                               according to which of the n image indices they are mapped to on this rank.
                               Since the mapping is potentially multivalued, the same index can appear in multiple bins.


    Not collective

    Input Parameters:
+   map    - mapping of indices
.   insize - size of the input array of indices from the rank's support
.   inidx  - input array of global indices from the rank's support
-   drop   - ignore indices outside of local support; generate an error otherwise


    Output Parameters:
+   outsize - number of indices in all bins (PETSC_NULL, if not needed) 
.   outidx  - concatenated bins containing the indices from inidx arranged by bin (PETSC_NULL, if not needed)
-   offsets - array of offsets of size n+1 delineating the bins; n is the number of distinct image indices on this rank (PETSC_NULL, if not needed)

    Level: advanced

    Concepts: binning^local indices

.seealso: ISMappingGetSupport(), ISMappingGetImage(), ISMappingGetSupportSize(), ISMappingGetImageSize(),
          ISMappingBinIndicesLocal(), ISMappingMapIndices(), ISMappingMapValues(), ISMappingBinValues()

@*/
PetscErrorCode ISMappingBinIndices(ISMapping map, PetscInt insize, const PetscInt inidx[], PetscInt *outsize, PetscInt outidx[], PetscInt offsets[], PetscBool drop)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_IS,1);
  ISMappingCheckAssembled(map,PETSC_TRUE, 1);
  ISMappingCheckMethod(map,map->ops->bin,"ISMappingBin");
  ierr = PetscLogEventBegin(IS_MAPPING_Bin, map, 0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->bin)(map,insize,inidx,PETSC_NULL,outsize,outidx,PETSC_NULL,offsets,drop); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_Bin, map, 0,0,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingBinValuesLocal"
/*@
    ISMappingBinValuesLocal - group local indices from the rank's support into n groups or "bins" (some possibly empty)
                              together with the corresponding scalar values.  The indices and values are arranged according to
                              to which of the n image indices they are mapped to on this rank.  Since the mapping is potentially 
                              multivalued, the same index and value can appear in multiple bins.

    Not collective

    Input Parameters:
+   map    - mapping of indices
.   insize - size of the input array of local indices from the rank's support
.   inidx  - input array of local indices from the rank's support
-   invals - input array of scalar values corresponding to the indices in inidx (PETSC_NULL, if not needed)

    Output Parameters:
+   outsize - number of indices and values in all bins (PETSC_NULL, if not needed) 
.   outidx  - concatenated bins containing the indices from inidx arranged by bin (PETSC_NULL, if not needed)
.   outvals - concatednated bins containing the values from invals arranged by the bin (PETSC_NULL, if not needed or if invals is PETSC_NULL)
-   offsets - array of offsets of size n+1 delineating the bins; n is the number of distinct image indices on this rank (PETSC_NULL, if not needed)


    Note: values are merely copied to the new locations prescribed by the location of the binned indices.
          If both invals and outvals are PETSC_NULL, this is equivalent to ISMappingBinIndicesLocal().
    Level: advanced

    Concepts: binning^indices and values

.seealso: ISMappingGetSupport(), ISMappingGetImage(), ISMappingGetSupportSize(), ISMappingGetImageSize(),
          ISMappingBinValues(), ISMappingMapIndicesLocal(), ISMappingMapValuesLocal(), ISMappingBinIndicesLocal()
@*/
PetscErrorCode ISMappingBinValuesLocal(ISMapping map, PetscInt insize, const PetscInt inidx[], const PetscScalar invals[], PetscInt *outsize, PetscInt outidx[], PetscScalar outvals[], PetscInt offsets[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_IS,1);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  ISMappingCheckMethod(map,map->ops->binlocal,"ISMappingBinLocal");
  ierr = PetscLogEventBegin(IS_MAPPING_Bin, map, 0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->binlocal)(map,insize,inidx,invals,outsize,outidx,outvals,offsets,PETSC_FALSE); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_Bin, map, 0,0,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingBinValues"
/*@
    ISMappingBinValues      - group indices from the rank's support into n groups or "bins" (some possibly empty)
                              together with the corresponding scalar values.  The indices and values are arranged according to
                              to which of the n image indices they are mapped to on this rank.  Since the mapping is potentially 
                              multivalued, the same index and value can appear in multiple bins.

    Not collective

    Input Parameters:
+   map    - mapping of indices
.   insize - size of the input array of indices from the rank's support
.   inidx  - input array of indices from the rank's support
.   invals - input array of scalar values corresponding to the indices in inidx (PETSC_NULL, if not needed)
-   drop   - ignore indices outside of local support; generate an error otherwise




    Output Parameters:
+   outsize - number of indices and values in all bins (PETSC_NULL, if not needed) 
.   outidx  - concatenated bins containing the indices from inidx arranged by bin (PETSC_NULL, if not needed)
.   outvals - concatednated bins containing the values from invals arranged by the bin (PETSC_NULL, if not needed or if invals is PETSC_NULL)
-   offsets - array of offsets of size n+1 delineating the bins; n is the number of distinct image indices on this rank (PETSC_NULL, if not needed)


    Note: values are merely copied to the new locations prescribed by the location of the binned indices.
          If both invals and outvals are PETSC_NULL, this is equivalent to ISMappingBinIndicesLocal().
    Level: advanced

    Concepts: binning^indices and values

.seealso: ISMappingGetSupport(), ISMappingGetImage(), ISMappingGetSupportSize(), ISMappingGetImageSize(),
          ISMappingBinValuesLocal(), ISMappingMapIndices(), ISMappingMapValues(), ISMappingBinIndices()
@*/
PetscErrorCode ISMappingBinValues(ISMapping map, PetscInt insize, const PetscInt inidx[], const PetscScalar invals[], PetscInt *outsize, PetscInt outidx[], PetscScalar outvals[], PetscInt offsets[], PetscBool drop)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ISMappingCheckType(map,IS_MAPPING_IS,1);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  ISMappingCheckMethod(map,map->ops->bin,"ISMappingBin");
  ierr = PetscLogEventBegin(IS_MAPPING_Bin, map, 0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->bin)(map,insize,inidx,invals,outsize,outidx,outvals,offsets,drop); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_Bin, map, 0,0,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingSetSizes"
/*@
  ISMappingSetSizes - Sets the local and global sizes for the domain and range, and checks to determine compatibility

  Collective on ISMapping

  Input Parameters:
+  map     - the mapping
.  m       - number of local  domain indices (or PETSC_DETERMINE)
.  n       - number of local  range columns  (or PETSC_DETERMINE)
.  M       - number of global domain indices (or PETSC_DETERMINE or PETSC_IGNORE)
-  N       - number of global range  indices (or PETSC_DETERMINE or PETSC_IGNORE)

   Notes:
   The sizes specify (i) what the domain and range for the graph underlying ISMapping is, 
   and (b) what the parallel layout of the domain and range are.  The local and global sizes
   m and M (and n and N) are not independent.  Two situations are possible the domain
   (and,analogously, for the range):
  
   (P) Parallel layout for the domain demands that the local values of m on all ranks of 
   the communicator add up to M (see more at MatSetSizes, for example).  Thus, m and M must 
   be specified in this compatible way.  One way to ensure this is to specify m and leave
   M as PETSC_DETERMINE -- then M is computed by summing the local m across the ranks. 
   The other option is to specify M (the same on all ranks, which will be checked) and 
   leave m as PETSC_DETERMINE.  In this case the local m is determined by dividing M as 
   equally as possible among the ranks (m might end up being 0).  If both m and M are specified,
   their compatibility is verified by summing the m across the ranks.  If m or M are PETSC_DETERMINE
   on one rank, they must be PETSC_DETERMINE on all of the ranks, or the program might hang.
   Finally, both m and M cannot be PETSC_DETERMINE at once.  

   In any case, domain indices can have any value 0 <= i < M on every rank (with the same M).
   However, domain indices are split up among the ranks: each rank will "own" m indices with 
   the indices owned by rank 0 numbered [0,m), followed by the indices on rank 1, and so on.

   (S) Sequential layout for the domain makes it essentially into a disjoint union of local 
   domains of local size m.  This is signalled by specifying M as PETSC_IGNORE.  

   In this case, domain indices can have any value 0 <= i < m on every rank (with its own m).

   Assembly/Mapping:
   Whether the domain is laid out in parallel (P) or not (S), determines the behavior of ISMapping
   during assembly.  In case (P), the edges of the underlying graph are migrated to the rank that
   owns the corresponding domain indices.  ISMapping can map indices lying in its local range, 
   which is a subset of its local domain.  This means that due to parallel assembly edges inserted
   by different ranks might be used during the mapping.  This is completely analogous to matrix 
   assembly.

   When the domain is not laid out in parallel, no migration takes place and the mapping of indices
   is done purely locally.
   
   

   Support/Image:
   Observe that the support and image of the graph may be strictly smaller than its domain and range,
   if no edges from some domain points (or to some range points) are added to ISMapping.
   
   Operator:
   Observe also that the linear operator defined by ISMapping will behave essentially as a VecScatter
   (i)   between MPI vectors with sizes (m,M) and (n,N), if both the domain and the range are (P),
   (ii)  between an MPI Vec with size (m,M) and a collection of SEQ Vecs (one per rank) of local size (n), 
         if the domain is (P) and the range is (S),
   (iii) between a collection of SEQ Vecs (one per rank) of local size (m) and an MPI Vec of size (n,N),
         if the domain is (S) and the range is (P),
   (iv)  between collections of SEQ Vecs (one per rank) or local sizes (m) and (n), if both the domain 
         and the range are (S).

  Level: beginner

.seealso: ISMappingGetSizes(), ISMappingGetSupportSize(), ISMappingGetImageSize(), ISMappingMapIndicesLocal()
@*/
PetscErrorCode  ISMappingSetSizes(ISMapping map, PetscInt m, PetscInt n, PetscInt M, PetscInt N)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,IS_MAPPING_CLASSID,1); 
  if (M > 0 && m > M) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local column size %D cannot be larger than global column size %D",m,M);
  if (N > 0 && n > N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local row size %D cannot be larger than global row size %D",n,N);
  if(!map->xlayout) {
    if(M == PETSC_IGNORE) {
      ierr = PetscLayoutCreate(PETSC_COMM_SELF, &(map->xlayout));       CHKERRQ(ierr);
    }
    else {
      ierr = PetscLayoutCreate(((PetscObject)map)->comm, &(map->xlayout)); CHKERRQ(ierr);
    }
  }
  if(!map->ylayout) {
    if(N == PETSC_IGNORE) {
      ierr = PetscLayoutCreate(PETSC_COMM_SELF, &(map->ylayout));       CHKERRQ(ierr);
    }
    else {
      ierr = PetscLayoutCreate(((PetscObject)map)->comm, &(map->ylayout)); CHKERRQ(ierr);
    }
  }
  if ((map->xlayout->n >= 0 || map->xlayout->N >= 0) && (map->xlayout->n != m || map->xlayout->N != M)) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change/reset domain sizes to %D local %D global after previously setting them to %D local %D global",m,M,map->xlayout->n,map->xlayout->N);
  if ((map->ylayout->n >= 0 || map->ylayout->N >= 0) && (map->ylayout->n != n || map->ylayout->N != N)) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change/reset range sizes to %D local %D global after previously setting them to %D local %D global",n,N,map->ylayout->n,map->ylayout->N);
  
  map->xlayout->n = m;
  map->ylayout->n = n;
  map->xlayout->N = M;
  map->ylayout->N = N;

  map->setup = PETSC_FALSE;
  map->assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingGetSizes"
PetscErrorCode  ISMappingGetSizes(ISMapping map, PetscInt *m, PetscInt *n, PetscInt *M, PetscInt *N)
{

  PetscFunctionBegin;
  *m = map->xlayout->n;
  *n = map->ylayout->n;
  *M = map->xlayout->N;
  *N = map->ylayout->N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingSetUp"
/*@
   ISMappingSetUp - Sets up the internal mapping data structures for the later use.

   Collective on ISMapping

   Input Parameters:
.  map - the ISMapping context

   Notes:
   For basic use of the ISMapping classes the user need not explicitly call
   ISMappingSetUp(), since these actions will happen automatically.

   Level: advanced

.keywords: ISMapping, setup

.seealso: ISMappingCreate(), ISMappingDestroy(), ISMappingSetSizes()
@*/
PetscErrorCode ISMappingSetUp(ISMapping map)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, IS_MAPPING_CLASSID,1);
  ISMappingCheckMethod(map,map->ops->setup,"ISMappingSetUp");
  ierr = (*(map->ops->setup))(map); CHKERRQ(ierr);
  map->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}/* ISMappingGetSizes() */

#undef  __FUNCT__
#define __FUNCT__ "ISMappingAssemblyBegin"
PetscErrorCode ISMappingAssemblyBegin(ISMapping map)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, IS_MAPPING_CLASSID,1);

  if(map->assembled) PetscFunctionReturn(0);
  ierr = ISMappingSetUp(map); CHKERRQ(ierr);
  
  ISMappingCheckMethod(map,map->ops->assemblybegin, "ISMappingAsemblyBegin");
  ierr = PetscLogEventBegin(IS_MAPPING_AssemblyBegin, map,0,0,0); CHKERRQ(ierr);
  ierr = (*(map->ops->assemblybegin))(map); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_AssemblyBegin, map,0,0,0); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}/* ISMappingAssemblyBegin() */

#undef __FUNCT__
#define __FUNCT__ "ISMappingAssemblyEnd"
PetscErrorCode ISMappingAssemblyEnd(ISMapping map)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ISMappingCheckMethod(map,map->ops->assemblyend, "ISMappingAsemblyEnd");
  ierr = PetscLogEventBegin(IS_MAPPING_AssemblyEnd, map,0,0,0); CHKERRQ(ierr);
  ierr = (*(map->ops->assemblyend))(map); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_AssemblyBegin, map,0,0,0); CHKERRQ(ierr);
  map->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}/* ISMappingAssemblyEnd() */

#undef  __FUNCT__
#define __FUNCT__ "ISMappingGetSupportIS"
PetscErrorCode ISMappingGetSupportIS(ISMapping map, IS *supp) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(map, IS_MAPPING_CLASSID,1);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  if(!supp) PetscFunctionReturn(0);
  ISMappingCheckMethod(map,map->ops->getsupportis,"ISMappingGetSupportIS");
  ierr = (*(map->ops->getsupportis))(map,supp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingGetSupportIS() */

#undef  __FUNCT__
#define __FUNCT__ "ISMappingGetImageIS"
PetscErrorCode ISMappingGetImageIS(ISMapping map, IS *image) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(map, IS_MAPPING_CLASSID,1);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  if(!image) PetscFunctionReturn(0);
  ISMappingCheckMethod(map,map->ops->getimageis,"ISMappingGetImageIS");
  ierr = (*(map->ops->getimageis))(map,image); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingGetImageIS() */

#undef __FUNCT__  
#define __FUNCT__ "ISMappingGetMaxImageSizeLocal"
PetscErrorCode ISMappingGetMaxImageSizeLocal(ISMapping map, PetscInt *maxsize)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,IS_MAPPING_CLASSID,1);
  PetscValidIntPointer(maxsize,2);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  ISMappingCheckMethod(map, map->ops->getmaximagesizelocal,"ISMappingGetMaxImageSizeLocal");
  ierr = (*map->ops->getmaximagesizelocal)(map,maxsize); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingGetImageSizeLocal"
PetscErrorCode ISMappingGetImageSizeLocal(ISMapping map, PetscInt *size)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,IS_MAPPING_CLASSID,1);
  PetscValidIntPointer(size,2);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  ISMappingCheckMethod(map, map->ops->getimagesizelocal,"ISMappingGetImageSizeLocal");
  ierr = (*map->ops->getimagesizelocal)(map,size); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingGetSupportSizeLocal"
PetscErrorCode ISMappingGetSupportSizeLocal(ISMapping map, PetscInt *size)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,IS_MAPPING_CLASSID,1);
  PetscValidIntPointer(size,2);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  ISMappingCheckMethod(map, map->ops->getsupportsizelocal,"ISMappingGetSupportSizeLocal");
  ierr = (*map->ops->getsupportsizelocal)(map,size); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingGetOperator"
PetscErrorCode ISMappingGetOperator(ISMapping map, Mat *mat)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,IS_MAPPING_CLASSID,1);
  PetscValidPointer(mat,2);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  ISMappingCheckMethod(map, map->ops->getoperator,"ISMappingGetOperator");
  ierr = (*map->ops->getoperator)(map,mat); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISMappingView"
PetscErrorCode ISMappingView(ISMapping map, PetscViewer v) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,IS_MAPPING_CLASSID,1);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  ISMappingCheckMethod(map,map->ops->view, "ISMappingView");
  ierr = (*(map->ops->view))(map,v); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingView() */

#undef  __FUNCT__
#define __FUNCT__ "ISMappingInvert"
PetscErrorCode ISMappingInvert(ISMapping map, ISMapping *imap) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,IS_MAPPING_CLASSID, 1);
  PetscValidPointer(imap,2);
  ISMappingCheckMethod(map,map->ops->invert, "ISMappingInvert");
  ierr = PetscLogEventBegin(IS_MAPPING_Invert, map, 0,0,0); CHKERRQ(ierr);
  ierr = (*(map->ops->invert))(map,imap); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_Invert, map, 0,0,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISMappingPullback"
/*@
   ISMappingPullback - compose mappings C = A*B

   Collective on ISMapping

   Input Parameters:
+  A - the left  mapping
-  B - the right mapping


   Output Parameters:
.  C - the product mapping: domain as in A, range as in B


   Level: intermediate

.seealso: ISMappingPushforward(), ISMappingInvert()
@*/
PetscErrorCode  ISMappingPullback(ISMapping A,ISMapping B, ISMapping *C) 
{
  PetscErrorCode ierr;
  PetscErrorCode (*pullback)(ISMapping,ISMapping,ISMapping*);
  char  pullbackname[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,IS_MAPPING_CLASSID,1);
  PetscValidType(A,1);
  if (!A->assembled) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled mapping");
  PetscValidHeaderSpecific(B,IS_MAPPING_CLASSID,2);
  PetscValidType(B,2);
  if (!B->assembled) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled mapping");
  PetscValidPointer(C,3);

  /* dispatch based on the type of A and B */
  ierr = PetscStrcpy(pullbackname,"ISMappingPullback_");CHKERRQ(ierr);
  ierr = PetscStrcat(pullbackname,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = PetscStrcat(pullbackname,"_");    CHKERRQ(ierr);
  ierr = PetscStrcat(pullbackname,((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = PetscStrcat(pullbackname,"_C");   CHKERRQ(ierr); /* e.g., pullbackname = "ISPullback_ismappingis_ismappingis_C" */
  ierr = PetscObjectQueryFunction((PetscObject)B,pullbackname,(void (**)(void))&pullback);CHKERRQ(ierr);
  if (!pullback) SETERRQ2(((PetscObject)A)->comm,PETSC_ERR_ARG_INCOMP,"ISMappingPullback requires A, %s, to be compatible with B, %s",((PetscObject)A)->type_name,((PetscObject)B)->type_name);    
  ierr = PetscLogEventBegin(IS_MAPPING_Pullback, A,B,0,0); CHKERRQ(ierr);
  ierr = (*pullback)(A,B,C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_Pullback, A,B,0,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "ISMappingPushforward"
/*@
   ISMappingPushforward - mapping from the range of A to the range of B, pointwise on the common domain

   Collective on ISMapping

   Input Parameters:
+  A - the left  mapping
-  B - the right mapping


   Output Parameters:
.  C - the product mapping: domain as the range of  A, range as the range of B


   Level: intermediate
.seealso: ISMappingPullback(), ISMappingInvert()
@*/
PetscErrorCode  ISMappingPushforward(ISMapping A,ISMapping B, ISMapping *C) 
{
  PetscErrorCode ierr;
  PetscErrorCode (*pushforward)(ISMapping,ISMapping,ISMapping*);
  char  pushforwardname[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,IS_MAPPING_CLASSID,1);
  PetscValidType(A,1);
  if (!A->assembled) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled mapping");
  PetscValidHeaderSpecific(B,IS_MAPPING_CLASSID,2);
  PetscValidType(B,2);
  if (!B->assembled) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled mapping");
  PetscValidPointer(C,3);

  /* dispatch based on the type of A and B */
  ierr = PetscStrcpy(pushforwardname,"ISMappingPushforward_");CHKERRQ(ierr);
  ierr = PetscStrcat(pushforwardname,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = PetscStrcat(pushforwardname,"_");    CHKERRQ(ierr);
  ierr = PetscStrcat(pushforwardname,((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = PetscStrcat(pushforwardname,"_C");   CHKERRQ(ierr); /* e.g., pushforwardname = "ISPushforward_ismappingis_ismappingis_C" */
  ierr = PetscObjectQueryFunction((PetscObject)B,pushforwardname,(void (**)(void))&pushforward);CHKERRQ(ierr);
  if (!pushforward) SETERRQ2(((PetscObject)A)->comm,PETSC_ERR_ARG_INCOMP,"ISMappingPushforward requires A, %s, to be compatible with B, %s",((PetscObject)A)->type_name,((PetscObject)B)->type_name);    
  ierr = PetscLogEventBegin(IS_MAPPING_Pushforward, A,B,0,0); CHKERRQ(ierr);
  ierr = (*pushforward)(A,B,C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_Pushforward, A,B,0,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef  __FUNCT__
#define __FUNCT__ "ISMappingSetType"
PetscErrorCode ISMappingSetType(ISMapping map, const ISMappingType maptype) {
  PetscErrorCode ierr;
  PetscErrorCode (*ctor)(ISMapping);
  PetscBool sametype;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,IS_MAPPING_CLASSID, 1);
  ierr = PetscTypeCompare((PetscObject)map,maptype, &sametype); CHKERRQ(ierr);
  if(sametype) PetscFunctionReturn(0);

  if(!ISMappingRegisterAllCalled) {
    ierr = ISMappingRegisterAll(PETSC_NULL); CHKERRQ(ierr);
  }
  ierr =  PetscFListFind(ISList,((PetscObject)map)->comm,maptype,(void(**)(void))&ctor);CHKERRQ(ierr);
  if(!ctor) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unrecognized ISMapping type: %s", maptype); 

  /* destroy the old implementation, if it existed */
  if(map->ops->destroy) {
    ierr = (*(map->ops->destroy))(map); CHKERRQ(ierr);
    map->ops->destroy = PETSC_NULL;
  }
  
  /* create the new implementation */
  ierr = (*ctor)(map); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingSetType() */

#undef  __FUNCT__
#define __FUNCT__ "ISMappingDestroy"
PetscErrorCode ISMappingDestroy(ISMapping map) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,IS_MAPPING_CLASSID,1);
  if(--((PetscObject)map)->refct > 0) PetscFunctionReturn(0);
  if(map->ops->destroy) {
    ierr = (*map->ops->destroy)(map); CHKERRQ(ierr);
  }
  if(map->xlayout) {
    ierr = PetscLayoutDestroy(map->xlayout); CHKERRQ(ierr);
  }
  if(map->ylayout) {
    ierr = PetscLayoutDestroy(map->ylayout); CHKERRQ(ierr);
  }
  ierr = PetscHeaderDestroy(map); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* ISMappingDestroy() */

#undef  __FUNCT__
#define __FUNCT__ "ISMappingCreate"
PetscErrorCode ISMappingCreate(MPI_Comm comm, ISMapping *_map) 
{
  ISMapping map;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_map,2);
  *_map = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = ISMappingInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(map,_p_ISMapping,struct _ISMappingOps,IS_MAPPING_CLASSID,0,"ISMapping",comm,ISMappingDestroy,ISMappingView); CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm,&(map->xlayout)); CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm,&(map->ylayout)); CHKERRQ(ierr);
  *_map = map;
  PetscFunctionReturn(0);
}/* ISMappingCreate() */

