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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,IS_MAPPING_CLASSID,1);
  PetscValidType(A,1);
  if (!A->assembled) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled mapping");
  PetscValidHeaderSpecific(B,IS_MAPPING_CLASSID,2);
  PetscValidType(B,2);
  if (!B->assembled) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled mapping");
  PetscValidPointer(C,3);

  /* dispatch based on the type of A and B */
  char  pullbackname[256];
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,IS_MAPPING_CLASSID,1);
  PetscValidType(A,1);
  if (!A->assembled) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled mapping");
  PetscValidHeaderSpecific(B,IS_MAPPING_CLASSID,2);
  PetscValidType(B,2);
  if (!B->assembled) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled mapping");
  PetscValidPointer(C,3);

  /* dispatch based on the type of A and B */
  char  pushforwardname[256];
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
  PetscValidPointer(map,2);
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



#undef  __FUNCT__
#define __FUNCT__ "ISArrayHunkCreate"
PetscErrorCode ISArrayHunkCreate(PetscInt length, PetscInt mask, ISArrayHunk *_hunk) {
  PetscErrorCode ierr;
  ISArrayHunk hunk;
  PetscFunctionBegin;
  PetscValidPointer(_hunk,3);

  if(!((mask & ISARRAY_I) | (mask & ISARRAY_J) | (mask & ISARRAY_W))) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "ISHunk mask %D must include at least one field: I, J or W", mask);
  if(length <= 0) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Nonpositive ISArrayHunk length: %D", length);

  ierr = PetscNew(struct _n_ISArrayHunk, &hunk); CHKERRQ(ierr);
  hunk->mask = mask;
  hunk->length = length;
  if(mask & ISARRAY_I) {
    PetscInt *ia;
    ierr = PetscMalloc(sizeof(PetscInt)*length, &ia);  CHKERRQ(ierr);
    ierr = PetscMemzero(ia, sizeof(PetscInt)*length);  CHKERRQ(ierr);
    hunk->i = ia;
    hunk->imode = PETSC_OWN_POINTER;
  }
  if(mask & ISARRAY_J) {
    PetscInt *ja;
    ierr = PetscMalloc(sizeof(PetscInt)*length, &ja);  CHKERRQ(ierr);
    ierr = PetscMemzero(ja, sizeof(PetscInt)*length); CHKERRQ(ierr);
    hunk->j = ja;
    hunk->jmode = PETSC_OWN_POINTER;
  }
  if(mask & ISARRAY_W) {
    PetscScalar *wa;
    ierr = PetscMalloc(sizeof(PetscScalar)*length, &wa);  CHKERRQ(ierr);
    ierr = PetscMemzero(wa, sizeof(PetscScalar)*length); CHKERRQ(ierr);
    hunk->w = wa;
    hunk->wmode = PETSC_OWN_POINTER;
  }
  *_hunk = hunk;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "ISArrayHunkCreateWithArrays"
PetscErrorCode ISArrayHunkCreateWithArrays(PetscInt length, const PetscInt *i, PetscCopyMode imode, const PetscScalar *w, PetscCopyMode wmode, const PetscInt *j, PetscCopyMode jmode, ISArrayHunk *_hunk) {
  PetscErrorCode ierr;
  PetscInt mask;
  ISArrayHunk hunk;
  PetscFunctionBegin;
  PetscValidPointer(_hunk,8);

  mask = (i != PETSC_NULL) | ((j != PETSC_NULL)<<1) | ((w != PETSC_NULL)<<2);
  if(!((mask & ISARRAY_I) | (mask & ISARRAY_J) | (mask & ISARRAY_W))) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "ISHunk mask %D must include at least one field: I, J or W", mask);
  if(length <= 0) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Nonpositive ISArrayHunk length: %D", length);

  ierr = PetscNew(struct _n_ISArrayHunk, &hunk); CHKERRQ(ierr);
  hunk->mask = mask;


  if(i) {
    if(imode == PETSC_COPY_VALUES) {
      PetscInt *newi;
      ierr = PetscMalloc(sizeof(PetscInt)*length, &(newi)); CHKERRQ(ierr);
      ierr = PetscMemcpy(newi, i, sizeof(PetscInt)*length); CHKERRQ(ierr);
      hunk->i = newi;
    }
    else {
      hunk->i = i;
    }
    hunk->imode = imode;
  }
  if(j) {
    if(jmode == PETSC_COPY_VALUES) {
      PetscInt *newj;
      ierr = PetscMalloc(sizeof(PetscInt)*length, &(newj)); CHKERRQ(ierr);
      ierr = PetscMemcpy(newj, j, sizeof(PetscInt)*length); CHKERRQ(ierr);
      hunk->j = newj;
    }
    else {
      hunk->j = j;
    }
    hunk->jmode = jmode;
  }
  if(w) {
    if(wmode == PETSC_COPY_VALUES) {
      PetscScalar *neww;
      ierr = PetscMalloc(sizeof(PetscScalar)*length, &(neww)); CHKERRQ(ierr);
      ierr = PetscMemcpy(neww, w, sizeof(PetscScalar)*length); CHKERRQ(ierr);
      hunk->w = neww;
    }
    else {
      hunk->w = w;
    }
    hunk->wmode = wmode;
  }
  *_hunk = hunk;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayHunkDestroy"
PetscErrorCode ISArrayHunkDestroy(ISArrayHunk hunk) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  hunk->next = PETSC_NULL;
  if(--(hunk->refcnt) > 0) PetscFunctionReturn(0);
  if(hunk->length) {
    if(hunk->imode == PETSC_COPY_VALUES || hunk->imode == PETSC_OWN_POINTER) {
      ierr = PetscFree(hunk->i); CHKERRQ(ierr);
    }
    if(hunk->wmode == PETSC_COPY_VALUES || hunk->wmode == PETSC_OWN_POINTER) {
      ierr = PetscFree(hunk->w); CHKERRQ(ierr);
    }
    if(hunk->jmode == PETSC_COPY_VALUES || hunk->jmode == PETSC_OWN_POINTER) {
      ierr = PetscFree(hunk->j); CHKERRQ(ierr);
    }
    hunk->length = 0;
  }
  ierr = ISArrayHunkDestroy(hunk->parent); CHKERRQ(ierr);
  ierr = PetscFree(hunk); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "ISArrayHunkDestroyHunks"
PetscErrorCode ISArrayHunkDestroyHunks(ISArrayHunk hunk) {
  PetscErrorCode ierr;
  ISArrayHunk next;
  PetscFunctionBegin;
  PetscValidPointer(hunk,1);
  while(hunk) {
    next = hunk->next;
    ierr = ISArrayHunkDestroy(hunk); CHKERRQ(ierr);
    hunk = next;
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayHunkGetSubHunk"
PetscErrorCode ISArrayHunkGetSubHunk(ISArrayHunk hunk, PetscInt submask, PetscInt offset, PetscInt length, ISArrayHunk *_subhunk) {
  PetscInt mask;
  PetscErrorCode ierr;
  const PetscInt *i = PETSC_NULL, *j = PETSC_NULL;
  const PetscScalar *w;
  PetscFunctionBegin;
  PetscValidPointer(hunk,1);
  PetscValidPointer(_subhunk,5);

  if((length <= 0) || (length > hunk->length)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid subhunk length %D for a hunk of length %D", length, hunk->length);
  if((offset <= 0) || (offset+length) > hunk->length) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid subhunk offset %D for a subhunk length %D and hunk length %D", length, length, hunk->length);
  ierr = PetscNew(struct _n_ISArrayHunk, _subhunk); CHKERRQ(ierr);
  mask = submask & hunk->mask;
  if(mask & ISARRAY_I) i = hunk->i + offset;
  if(mask & ISARRAY_J) j = hunk->j + offset;
  if(mask & ISARRAY_W) w = hunk->w + offset;
  ierr = ISArrayHunkCreateWithArrays(length, i, PETSC_USE_POINTER, w, PETSC_USE_POINTER, j, PETSC_USE_POINTER, _subhunk); CHKERRQ(ierr);
  ++(hunk->refcnt);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayHunkMergeHunks"
PetscErrorCode ISArrayHunkMergeHunks(ISArrayHunk hunks, ISArrayHunk *_merged) {
  PetscErrorCode ierr;
  ISArrayHunk hunk;
  PetscInt    length, *i = PETSC_NULL, *j = PETSC_NULL;
  PetscInt    count;
  PetscScalar *w = PETSC_NULL;
  PetscBool foundmask = PETSC_FALSE;
  PetscInt mask;
  PetscFunctionBegin;
  PetscValidPointer(hunks,1);
  PetscValidPointer(_merged,2);

  *_merged = PETSC_NULL;
  /* Calculate the number of links in the chain and the length of the merged chain. */
  hunk = hunks;
  length = 0;
  count = 0;
  while(hunk) {
    /* Determine the mask and perform a consistency check. */
    if(!foundmask) 
      mask = hunk->mask;
    else if(mask != hunk->mask) 
      SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Inconsistent ISArrayHunk components  encountered on hunk %D: have %D but previously  %D", count, hunk->mask, mask);
    length   += hunk->length;
     hunk     = hunk->next;
     ++count;
  }
  if(count == 1) PetscFunctionReturn(0);

  if(length) {
  /* Allocate the space for the merged indices and weights. */
    if(mask & ISARRAY_I) {
      ierr = PetscMalloc(sizeof(PetscInt)*length, &i); CHKERRQ(ierr);
    }
    if(mask & ISARRAY_J) {
      ierr = PetscMalloc(sizeof(PetscInt)*length, &j); CHKERRQ(ierr);
    }
    if(mask & ISARRAY_W) {
      ierr = PetscMalloc(sizeof(PetscScalar)*length, &w); CHKERRQ(ierr);
    }
    /* Copy the indices and weights into the merged arrays. */
    length = 0;
    hunk = hunks;
    while(hunk) {
      if(mask & ISARRAY_I) {
        ierr = PetscMemcpy(i+length, hunk->i, sizeof(PetscInt)*hunk->length); CHKERRQ(ierr);
      }
      if(mask & ISARRAY_J) {
        ierr = PetscMemcpy(j+length, hunk->j, sizeof(PetscInt)*hunk->length); CHKERRQ(ierr);
      }
      if(mask & ISARRAY_W) {
        ierr = PetscMemcpy(w+length, hunk->w, sizeof(PetscScalar)*hunk->length); CHKERRQ(ierr);
      }
      length += hunk->length;
    }
  }/* if(length) */
  ierr = ISArrayHunkCreateWithArrays(length, i, PETSC_OWN_POINTER, w, PETSC_OWN_POINTER, j, PETSC_OWN_POINTER, _merged); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayCreate"
PetscErrorCode ISArrayCreate(ISArray *_chain) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_chain,1);
  ierr = PetscNew(struct _n_ISArray, _chain); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayClear"
PetscErrorCode ISArrayClear(ISArray chain) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  ierr = ISArrayHunkDestroyHunks(chain->first); CHKERRQ(ierr);
  chain->first = PETSC_NULL;
  chain->last  = PETSC_NULL;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "ISArrayDestroy"
PetscErrorCode ISArrayDestroy(ISArray chain) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  ierr = ISArrayHunkDestroyHunks(chain->first); CHKERRQ(ierr);
  ierr = PetscFree(chain);                      CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayAdd"
PetscErrorCode ISArrayAdd(ISArray chain, PetscInt length, const PetscInt *i, PetscCopyMode imode, const PetscScalar *w, PetscCopyMode wmode, const PetscInt *j, PetscCopyMode jmode) {
  PetscErrorCode ierr;
  ISArrayHunk hunk;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  if(!i && !j) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Both index arrays I and J are null");
  ierr = ISArrayHunkCreateWithArrays(length, i, imode, w, wmode, j, jmode, &hunk); CHKERRQ(ierr);
  ierr = ISArrayAddHunk(chain, hunk);                                              CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayAddHunk"
PetscErrorCode ISArrayAddHunk(ISArray chain, ISArrayHunk hunk){
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  PetscValidPointer(hunk,2);
  if(!hunk->length) PetscFunctionReturn(0);
  if(!(hunk->mask & (ISARRAY_I|ISARRAY_J))) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot add a hunk with no indices: hunk component mask %D", hunk->mask);
  if(!(chain->first) && chain->first->mask != hunk->mask) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Components of hunk to be added %D incompatible with the existing ISArray components %D", hunk->mask, chain->first->mask);

  /* Invariant: chain->first and chain->last are either PETSC_NULL or non PETSC_NULL together. */
  if(chain->last) {
    chain->last->next = hunk;
    chain->last = hunk;
  }
  else {
    chain->first = chain->last = hunk;
  }
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "ISArrayAddI"
PetscErrorCode ISArrayAddI(ISArray chain, PetscInt length, const PetscInt i, const PetscScalar *wa, PetscCopyMode wmode, const PetscInt *ja, PetscCopyMode jmode) {
  PetscErrorCode ierr;
  PetscInt k,*ia;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  if(!length) PetscFunctionReturn(0);
  if(length < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Negative array length: %D", length);
  ierr = PetscMalloc(sizeof(PetscInt)*length, &ia); CHKERRQ(ierr);
  for(k = 0; k < length; ++k) ia[k] = i;
  ierr = ISArrayAdd(chain, length, ia, PETSC_OWN_POINTER, wa, wmode, ja, jmode); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayAddJ"
PetscErrorCode ISArrayAddJ(ISArray chain, PetscInt length, const PetscInt ia[], PetscCopyMode imode, const PetscScalar wa[], PetscCopyMode wmode, PetscInt j) {
  PetscErrorCode ierr;
  PetscInt k,*ja;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  if(!length) PetscFunctionReturn(0);
  if(length < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Nengative array length: %D", length);
  ierr = PetscMalloc(sizeof(PetscInt)*length, &ja); CHKERRQ(ierr);
  for(k = 0; k < length; ++k) ja[k] = j;
  ierr = ISArrayAdd(chain, length, ia, imode, wa, wmode, ja, PETSC_OWN_POINTER); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayMerge_Private"
static PetscErrorCode ISArrayMerge_Private(ISArray chain) {
  PetscErrorCode ierr;
  ISArrayHunk merged;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  ierr = ISArrayHunkMergeHunks(chain->first, &merged); CHKERRQ(ierr);
  if(merged) {
    ierr = ISArrayHunkDestroyHunks(chain->first); CHKERRQ(ierr);
    chain->first = chain->last = PETSC_NULL;
    ierr = ISArrayAddHunk(chain, merged);         CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



#undef  __FUNCT__
#define __FUNCT__ "ISArrayGetComponents"
PetscErrorCode ISArrayGetComponents(ISArray chain, PetscInt *_mask) {
  ISArrayHunk hunk;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  PetscValidPointer(_mask,2);
  
  *_mask = 0;
  hunk = chain->first;
  while(hunk) 
    if(!hunk->mask) {
      *_mask = hunk->mask;
      break;
    }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayGetLength"
PetscErrorCode ISArrayGetLength(ISArray chain, PetscInt *_length) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  PetscValidPointer(_length,2);
  if(!chain->first) {
    *_length = 0;
  }
  else if(chain->first->next) {
    ierr = ISArrayMerge_Private(chain); CHKERRQ(ierr);
  }
  *_length = chain->first->length;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayGetI"
PetscErrorCode ISArrayGetI(ISArray chain, const PetscInt *_i[]) {
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  PetscValidPointer(_i,2);
  *_i = PETSC_NULL;
  if(chain->first) *_i = chain->first->i;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayGetJ"
PetscErrorCode ISArrayGetJ(ISArray chain, const PetscInt *_j[]) {
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  PetscValidPointer(_j,2);
  *_j = PETSC_NULL;
  if(chain->first) *_j = chain->first->j;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "ISArrayGetW"
PetscErrorCode ISArrayGetWeights(ISArray chain, const PetscScalar *_w[]) {
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  PetscValidPointer(_w,2);
  *_w = PETSC_NULL;
  if(chain->first) *_w = chain->first->w;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "ISArrayJoinArrays"
PetscErrorCode ISArrayJoinArrays(PetscInt len, ISArray chains[], ISArray *_joined) {
  PetscErrorCode ierr;
  PetscInt i;
  ISArrayHunk first,last,merged;
  PetscFunctionBegin;
  PetscValidPointer(chains,2);
  PetscValidPointer(_joined,3);
  *_joined = PETSC_NULL;
  if(!len) PetscFunctionReturn(0);
  if(len < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Negative array length: %D", len);

  /* Temporarily link the chains. */
  first = PETSC_NULL;
  last  = PETSC_NULL;
  for(i = 1; i < len; ++i) {
    if(!chains[i] || !chains[i]->first) continue;
    if(!last) {
      first = chains[i]->first;
      last = chains[i]->last;
    }
    else {
      last->next = chains[i]->first;
      last = chains[i]->last;
    }
  }
  /* Merge the unified chain. */
  ierr = ISArrayHunkMergeHunks(first, &merged); CHKERRQ(ierr);
  ierr = ISArrayCreate(_joined);                CHKERRQ(ierr);
  ierr = ISArrayAddHunk(*_joined, merged);      CHKERRQ(ierr);

  /* Unlink the chains. */
  for(i = 1; i < len; ++i) {
    if(!chains[i] || !chains[i]->first) continue;
    chains[i]->last->next = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISArrayGetSubArray"
PetscErrorCode ISArrayGetSubArray(ISArray array, PetscInt submask, PetscInt sublength, PetscInt offset, ISArray *_subarray) {
  PetscErrorCode ierr;
  ISArrayHunk subhunk;
  PetscFunctionBegin;
  ierr = ISArrayCreate(_subarray);    CHKERRQ(ierr);
  ierr = ISArrayMerge_Private(array); CHKERRQ(ierr);
  if(!array->first) PetscFunctionReturn(0);
  ierr = ISArrayHunkGetSubHunk(array->first, submask, sublength, offset, &subhunk); CHKERRQ(ierr);
  ierr = ISArrayAddHunk(*_subarray, subhunk); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Checks whether all indices are within [imin,imax) and generate an error, if they are not and 
     if outOfBoundsError == PETSC_TRUE.  Return the result in flag.
 */
#undef __FUNCT__  
#define __FUNCT__ "ISArray_CheckIntArrayRange"
static PetscErrorCode ISArray_CheckIntArrayRange(PetscInt len, const PetscInt idx[],  PetscInt imin, PetscInt imax, PetscBool outOfBoundsError, PetscBool *flag)
{
  PetscInt i;
  PetscBool inBounds = PETSC_TRUE;
  PetscFunctionBegin;
  for (i=0; i<len; ++i) {
    if (idx[i] <  imin) {
      if(outOfBoundsError) {
        SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %D at location %D is less than min %D",idx[i],i,imin);
      }
      inBounds = PETSC_FALSE;
      break;
    }
    if (idx[i] >= imax) {
      if(outOfBoundsError) {
        SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %D at location %D is greater than max %D",idx[i],i,imax);
      }
      inBounds = PETSC_FALSE;
      break;
    }
  }
  if(flag) *flag = inBounds;
  PetscFunctionReturn(0);
}

/* 
 FIX: If we wanted to split this into AssemblyBegin/End, some data must be passed between the stages (e.g., tags,
      waits), this could be made into an ISMapping method.  However, then an ISMapping of some sort must be instantiated
      for ISArray assembly.  Maybe it always happens in the common use cases, anyhow. 
 */
#undef __FUNCT__  
#define __FUNCT__ "ISArrayAssemble"
PetscErrorCode ISArrayAssemble(ISArray chain, PetscInt mask, PetscLayout layout, ISArray *_achain) {
  PetscErrorCode ierr;
  ISArray achain;
  PetscInt chainmask;
  MPI_Comm comm = layout->comm;
  PetscMPIInt size, rank, tag_i, tag_v;
  PetscMPIInt chainmaskmpi, allchainmask;
  const PetscInt *ixidx, *iyidx;
  PetscInt idx, lastidx;
  PetscInt i, j, p;
  PetscInt    *owner = PETSC_NULL;
  PetscInt    nsends, nrecvs;
  PetscMPIInt    *plengths, *sstarts = PETSC_NULL;
  PetscMPIInt *rnodes, *rlengths, *rstarts = PETSC_NULL, rlengthtotal;
  PetscInt    **rindices = PETSC_NULL, *sindices= PETSC_NULL;
  PetscScalar *svalues = PETSC_NULL, **rvalues = PETSC_NULL;
  MPI_Request *recv_reqs_i, *recv_reqs_v, *send_reqs;
  MPI_Status  recv_status, *send_statuses;
  const PetscInt *ia = PETSC_NULL, *ja = PETSC_NULL;
#if defined(PETSC_USE_DEBUG)
  PetscBool found;
#endif
  PetscInt len, ni, nv, count, alen;
  PetscInt *aixidx, *aiyidx = PETSC_NULL;
  PetscScalar *aval = PETSC_NULL;
  const PetscScalar *val;

  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  PetscValidPointer(_achain, 4);

  /* Make sure that at least one of the indices -- I or J -- is being assembled on, and that the index being assembled on is present in the ISArray. */
  ierr = ISArrayGetComponents(chain, &chainmask); CHKERRQ(ierr);
  if((mask != ISARRAY_I && mask != ISARRAY_J)|| !(mask & chainmask))
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot assemble ISArray with components %D on component %D", chainmask, mask);

  /* Comm parameters */
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* If layout isn't parallel, then this is a noop. */
  if(size == 1) PetscFunctionReturn(0);

  /* Make sure that chain type is the same across the comm. */
  chainmaskmpi = PetscMPIIntCast(chainmask);
  ierr = MPI_Allreduce(&(chainmaskmpi), &allchainmask, 1, MPI_INT, MPI_BOR, comm); CHKERRQ(ierr);
  if(allchainmask^chainmaskmpi) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Chain mask must be the same across the communicator. Got %D globally and %D locally", allchainmask, chainmaskmpi);

  /* How many index arrays are being sent? One or two? */
  ni = 0;
  ni += ((chainmask & ISARRAY_I) > 0);
  ni += ((chainmask & ISARRAY_J) > 0);

  /* How many value arrays are being sent?  One or none? */
  nv =  ((chainmask & ISARRAY_W) > 0);

  /* Merge the chain. */
  ierr = ISArrayMerge_Private(chain); CHKERRQ(ierr);

  /* 
   Split into the "base" and "fiber" indices: base indices (ixidx) are being assembled on and the fiber indices (iyidx), 
   if any, are along for the ride as are the scalar weights. iyidx and val will be null, if they are not present in chain 
   (ni == 1 and nv == 0, respectively).
   */
  if(mask == ISARRAY_I) {
    ixidx = chain->first->i;
    iyidx = chain->first->j;
  }
  else {
    ixidx = chain->first->j;
    iyidx = chain->first->i;
  }
  val = chain->first->w;
  len = chain->first->length;

  /* Verify the indices against the layout. */
  ierr = ISArray_CheckIntArrayRange(len, ixidx, 0,layout->N, PETSC_TRUE, PETSC_NULL); CHKERRQ(ierr);
  
/*
   Each processor ships off its ixidx[j] and, possibly, the appropriate combination of iyidx[j] 
   and val[j] to the appropriate processor.
   */
  /*  first count number of contributors to each processor */
  ierr  = PetscMalloc2(size,PetscMPIInt,&plengths,len,PetscInt,&owner);CHKERRQ(ierr);
  ierr  = PetscMemzero(plengths,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  lastidx = -1;
  p       = 0;
  for (i=0; i<len; ++i) {
    /* if indices are NOT locally sorted, need to start search for the proc owning inidx[i] at the beginning */
    if (lastidx > (idx = ixidx[i])) p = 0;
    lastidx = idx;
    for (; p<size; ++p) {
      if (idx >= layout->range[p] && idx < layout->range[p+1]) {
        plengths[p]++; 
        owner[i] = p; 
#if defined(PETSC_USE_DEBUG)
        found = PETSC_TRUE; 
#endif
        break;
      }
    }
#if defined(PETSC_USE_DEBUG)
    if (!found) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %D out of range",idx);
    found = PETSC_FALSE;
#endif
  }
  nsends = 0;  for (p=0; p<size; ++p) { nsends += (plengths[p] > 0);} 
    
  /* inform other processors of number of messages and max length*/
  ierr = PetscGatherNumberOfMessages(comm,PETSC_NULL,plengths,&nrecvs);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nsends,nrecvs,plengths,&rnodes,&rlengths);CHKERRQ(ierr);
  /* no values are sent if nv == 0 */
  if(nv) {
    /* values in message i are rlengths[i] in number */
    ierr = PetscCommGetNewTag(layout->comm, &tag_v);CHKERRQ(ierr);
    ierr = PetscPostIrecvScalar(comm,tag_v,nrecvs,rnodes,rlengths,&rvalues,&recv_reqs_v);CHKERRQ(ierr);
  }
  /* we are sending ni*rlengths[i] indices in each message (ni == 1 or 2, is the number of index arrays being sent) */
  if(ni == 2) {
    for (i=0; i<nrecvs; ++i) rlengths[i] *=2;
  }
  ierr = PetscCommGetNewTag(layout->comm, &tag_i);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tag_i,nrecvs,rnodes,rlengths,&rindices,&recv_reqs_i);CHKERRQ(ierr);
  ierr = PetscFree(rnodes);CHKERRQ(ierr);

  if(ni == 2) {
    for (i=0; i<nrecvs; ++i) rlengths[i] /=2;
  }

  /* prepare send buffers and offsets.
      sindices is the index send buffer; svalues is the value send buffer, allocated only if nv > 0.
      sstarts[p] gives the starting offset for values going to the pth processor, if any;
      because the indices (I & J, if both are being sent) are sent from the same buffer, 
      the index offset is ni*sstarts[p].
  */
  ierr     = PetscMalloc((size+1)*sizeof(PetscMPIInt),&sstarts);  CHKERRQ(ierr);
  ierr     = PetscMalloc(ni*len*sizeof(PetscInt),&sindices);      CHKERRQ(ierr);
  if(nv) {
    ierr     = PetscMalloc(len*sizeof(PetscScalar),&svalues);     CHKERRQ(ierr);
  }

  /* Compute buffer offsets for the segments of data going to different processors,
     and zero out plengths: they will be used below as running counts when packing data
     into send buffers; as a result of that, plengths are recomputed by the end of the loop.
   */
  sstarts[0] = 0;
  for (p=0; p<size; ++p) { 
    sstarts[p+1] = sstarts[p] + plengths[p];
    plengths[p] = 0;
  }

  /* Now pack the indices and, possibly, values into the appropriate buffer segments. */
  for(i = 0; i < len; ++i) {
    p = owner[i];
    sindices[ni*sstarts[p]+plengths[p]]                             = ixidx[i];
    if(ni==2) 
      sindices[ni*sstarts[p]+(sstarts[p+1]-sstarts[p])+plengths[p]] = iyidx[i];
    if(nv) 
      svalues[sstarts[p]+plengths[p]]                               = val[i];
    ++plengths[p];
  }
  /* Allocate send requests: for the indices, and possibly one more for the scalar values, hence +nv */
  ierr     = PetscMalloc((1+nv)*nsends*sizeof(MPI_Request),&send_reqs);  CHKERRQ(ierr);

  /* Post sends */
  for (p=0,count=0; p<size; ++p) {
    if (plengths[p]) {
      ierr = MPI_Isend(sindices+ni*sstarts[p],ni*plengths[p],MPIU_INT,p,tag_i,comm,send_reqs+count++);CHKERRQ(ierr);
      ierr = MPI_Isend(svalues+sstarts[p],plengths[p],MPIU_SCALAR,p,tag_v,comm,send_reqs+count++);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(plengths,owner);CHKERRQ(ierr);
  ierr = PetscFree(sstarts);CHKERRQ(ierr);

  /* Prepare to receive indices and values. */
  /* Compute the offsets of the individual received segments in the unified index/value arrays. */
  ierr = PetscMalloc(sizeof(PetscMPIInt)*(nrecvs+1), &rstarts); CHKERRQ(ierr);
  rstarts[0] = 0;
  for(j = 0; j < nrecvs; ++j) rstarts[j+1] = rstarts[j] + rlengths[j];
  /* Allocate the unified index/value arrays to pack the received segments from each proc into. */
  alen = rstarts[nrecvs];
  ierr  = PetscMalloc(alen*sizeof(PetscInt), &aixidx);   CHKERRQ(ierr);
  if(ni == 2) {
    ierr  = PetscMalloc(alen*sizeof(PetscInt), &aiyidx); CHKERRQ(ierr);
  }
  if(nv) {
    ierr = PetscMalloc(alen*sizeof(PetscScalar), &aval); CHKERRQ(ierr);
  }

  /* Receive indices and values, and pack them into unified arrays. */
  if(nv) {
    /*  wait on scalar values receives and pack the received values into aval */
    count = nrecvs; 
    rlengthtotal = 0;
    while (count) {
      PetscMPIInt n,k;
      ierr = MPI_Waitany(nrecvs,recv_reqs_v,&k,&recv_status);CHKERRQ(ierr);
      ierr = MPI_Get_count(&recv_status,MPIU_SCALAR,&n);CHKERRQ(ierr);
      rlengthtotal += n;
      count--;
      ierr = PetscMemcpy(aval+rstarts[k],rvalues[k],sizeof(PetscScalar)*rlengths[k]); CHKERRQ(ierr);
    }
    if (rstarts[nrecvs] != rlengthtotal) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Total message lengths %D not as expected %D",rlengthtotal,rstarts[nrecvs]);
  }
    
  /*  wait on index receives and pack the received indices into aixidx and aiyidx, as necessary. */
  count = nrecvs; 
  rlengthtotal = 0;
  while (count) {
    PetscMPIInt n,k;
    ierr = MPI_Waitany(nrecvs,recv_reqs_i,&k,&recv_status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&n);CHKERRQ(ierr);
    rlengthtotal += n/ni;
    count--;
    ierr = PetscMemcpy(aixidx+rstarts[k],rindices[k],sizeof(PetscInt)*rlengths[k]); CHKERRQ(ierr);    
    if(ni == 2) {
      ierr = PetscMemcpy(aiyidx+rstarts[k],rindices[k]+rlengths[k],sizeof(PetscInt)*rlengths[k]); CHKERRQ(ierr);    
    }
  }
  if (rstarts[nrecvs] != rlengthtotal) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Total message lengths %D not as expected %D",rlengthtotal,rstarts[nrecvs]);
  

  ierr = PetscFree(rlengths);    CHKERRQ(ierr);
  ierr = PetscFree(rstarts);     CHKERRQ(ierr); 
  ierr = PetscFree(rindices[0]); CHKERRQ(ierr);
  ierr = PetscFree(rindices);    CHKERRQ(ierr);
  if(nv) {
    ierr = PetscFree(rvalues[0]); CHKERRQ(ierr);
    ierr = PetscFree(rvalues);    CHKERRQ(ierr);
  }
  /* wait on sends */
  if (nsends) {
    ierr = PetscMalloc(sizeof(MPI_Status)*nsends,&send_statuses);     CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,send_reqs,send_statuses);              CHKERRQ(ierr);
    ierr = PetscFree(send_statuses);                                  CHKERRQ(ierr);
  }

  ierr = PetscFree(sindices);    CHKERRQ(ierr);
  if(nv) {
    ierr = PetscFree(svalues);  CHKERRQ(ierr);
  }

  /* Assemble ISArray out of aixidx, aiyidx and aval, as appropriate. */
  if(mask == ISARRAY_I) {
    ia = aixidx;
    ja = aiyidx;
  }
  else {
    ia = aiyidx;
    ja = aixidx;
  }
  ierr = ISArrayCreate(&achain); CHKERRQ(ierr);
  ierr = ISArrayAdd(achain, alen, ia, PETSC_OWN_POINTER, aval, PETSC_OWN_POINTER, ja, PETSC_OWN_POINTER); CHKERRQ(ierr);
  *_achain = achain;


  PetscFunctionReturn(0);
}/* ISArrayAssemble() */
