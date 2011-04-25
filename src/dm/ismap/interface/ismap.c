#define PETSCDM_DLL

#include "private/ismapimpl.h"  /*I "petscmap.h"  I*/


PetscClassId  IS_MAPPING_CLASSID;
PetscLogEvent IS_MAPPING_Map, IS_MAPPING_MapLocal, IS_MAPPING_MapSplit, IS_MAPPING_MapSplitLocal;
PetscLogEvent IS_MAPPING_Bin, IS_MAPPING_BinLocal, IS_MAPPING_BinSplit, IS_MAPPING_BinSplitLocal;
PetscLogEvent IS_MAPPING_AssemblyBegin, IS_MAPPING_AssemblyEnd, IS_MAPPING_Invert, IS_MAPPING_Pushforward, IS_MAPPING_Pullback;

PetscFList ISMappingList               = PETSC_NULL;
PetscBool  ISMappingRegisterAllCalled  = PETSC_FALSE;
PetscBool  ISMappingPackageInitialized = PETSC_FALSE;

EXTERN_C_BEGIN
extern PetscErrorCode ISMappingCreate_Graph(ISMapping);
EXTERN_C_END


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
  ierr = PetscClassIdRegister("ISMapping",&IS_MAPPING_CLASSID);                                         CHKERRQ(ierr);
  /* Register Constructors */
  ierr = ISMappingRegisterAll(path);                                                                    CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("ISMappingMap",           IS_MAPPING_CLASSID,&IS_MAPPING_Map);           CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ISMappingMapLocal",      IS_MAPPING_CLASSID,&IS_MAPPING_MapLocal);      CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ISMappingBin",           IS_MAPPING_CLASSID,&IS_MAPPING_Bin);           CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ISMappingBinLocal",      IS_MAPPING_CLASSID,&IS_MAPPING_BinLocal);      CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ISMappingMap",           IS_MAPPING_CLASSID,&IS_MAPPING_MapSplit);      CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ISMappingMapLocal",      IS_MAPPING_CLASSID,&IS_MAPPING_MapSplitLocal); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ISMappingBin",           IS_MAPPING_CLASSID,&IS_MAPPING_BinSplit);      CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ISMappingBinLocal",      IS_MAPPING_CLASSID,&IS_MAPPING_BinSplitLocal); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ISMappingAssemblyBegin", IS_MAPPING_CLASSID,&IS_MAPPING_AssemblyBegin); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ISMappingAssemblyEnd",   IS_MAPPING_CLASSID,&IS_MAPPING_AssemblyEnd);   CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ISMappingPushforward",   IS_MAPPING_CLASSID,&IS_MAPPING_Pushforward);   CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ISMappingPullback",      IS_MAPPING_CLASSID,&IS_MAPPING_Pullback);      CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ISMappingInvert",        IS_MAPPING_CLASSID,&IS_MAPPING_Invert);        CHKERRQ(ierr);

  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);                        CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "is_mapping", &className);                                              CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(IS_MAPPING_CLASSID);                                              CHKERRQ(ierr);
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
  ierr = ISMappingRegisterDynamic(IS_MAPPING_GRAPH,path,"ISMappingCreate_Graph",ISMappingCreate_Graph);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingMapLocal"
/*@
    ISMappingMapLocal - maps an ISArray with local indices from the rank's support to global indices from the rank's range.
                        Since ISMapping is in general multivalued, some local indices are mapped to multiple global indices.
                        Only selected indices (I or J) are mapped; the other indices and weights, if any, are preserved on 
                        the images.


    Not collective

    Input Parameters:
+   map    - mapping of indices
.   inarr  - input ISArray
-   index   - selection of the index to map (ISARRAY_I or ISARRAY_J; PETSC_NULL is equivalent to ISARRAY_I)


    Output Parameters:
.   outarr - ISArray with the selected indices mapped


    Level: advanced

    Concepts: mapping^indices

.seealso: ISMappingGetSupport(), ISMappingGetImage(), ISMappingGetSupportSizeLocal(), ISMappingGetImageSizeLocal(),
          ISMappingMap(),        ISMappingBin(),      ISMappingBinLocal(),            ISMappingMapSplitLocal()

@*/
PetscErrorCode ISMappingMapLocal(ISMapping map, ISArray inarr, ISArrayIndex index, ISArray outarr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, IS_MAPPING_CLASSID, 1);
  PetscValidPointer(outarr, 4);
  if(!index) index = ISARRAY_I;
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  ISMappingCheckMethod(map,map->ops->maplocal,"ISMappingMapLocal");
  ierr = PetscLogEventBegin(IS_MAPPING_MapLocal,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->maplocal)(map,inarr,index,outarr);      CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_MapLocal,map,0,0,0);   CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingMap"
/*@
    ISMappingMap      - maps an ISArray with global indices from the rank's support to global indices from the rank's range.
                        Since ISMapping is in general multivalued, some indices are mapped to multiple global indices.
                        Only indices of the selected type (I or J) are mapped; the other indices and weights, if any, are 
                        preserved on the images.

    Not collective

    Input Parameters:
+   map    - mapping of indices
.   inarr  - input ISArray of indices and weights to map
-   index  - selection of the index to map (ISARRAY_I or ISARRAY_J; PETSC_NULL is equivalent to ISARRAY_I)


    Output Parameters:
.   outarr - ISArray with the selected indices mapped


    Level: advanced

    Concepts: mapping^indices global

.seealso: ISMappingGetSupport(), ISMappingGetImage(), ISMappingGetSupportSizeLocal(), ISMappingGetImageSizeLocal(),
          ISMappingMapLocal(),   ISMappingBin(),      ISMappingBinLocal(),            ISMappingMapSlit()

@*/
PetscErrorCode ISMappingMap(ISMapping map, ISArray inarr, ISArrayIndex index, ISArray outarr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, IS_MAPPING_CLASSID, 1);
  PetscValidPointer(outarr, 4);
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  if(!index) index = ISARRAY_I;
  ISMappingCheckMethod(map,map->ops->map,"ISMappingMap");
  ierr = PetscLogEventBegin(IS_MAPPING_Map,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->map)(map,inarr,index,outarr);     CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_Map,map,0,0,0);   CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "ISMappingBinLocal"
/*@
    ISMappingBinLocal        - order local indices from the rank's support into n consecutive groups or "bins" (some possibly empty)
                               according to which of the n image indices they are mapped to on this rank. The groups are concatenated
                               and returned as a single array. See ISMappingBinSplitLocal() if separate bin output is desired. 
                               Since ISMapping is potentially multivalued, the same index can appear in multiple bins.
                               The binning is done on the indices of the selected type(I or J); the other indices and weights, if any, 
                               are moved to the appropriate bin together with the selected indices.


    Not collective

    Input Parameters:
+   map    - mapping of indices
.   array  - ISArray with indices to bin
-   index  - selection of the index to bin on (ISARRAY_I or ISARRAY_J; PETSC_NULL is equivalent to ISARRAY_I)


    Output Parameters:
.   bins    - ISArray containing concatenated binned indices; the number of bins is the same as the result of ISGetImageSizeLocal().

    Level: advanced

    Concepts: binning^local indices

.seealso: ISMappingGetSupport(), ISMappingGetImage(), ISMappingGetSupportSizeLocal(), ISMappingGetImageSizeLocal(),
          ISMappingBin(),        ISMappingMapLocal(), ISMappingMapLocal(),            ISMappingBinSplitLocal()

@*/
PetscErrorCode ISMappingBinLocal(ISMapping map, ISArray array, ISArrayIndex index, ISArray bins)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, IS_MAPPING_CLASSID, 1);
  PetscValidPointer(bins,4);
  ISMappingCheckAssembled(map,PETSC_TRUE, 1);
  if(!index) index = ISARRAY_I;
  ISMappingCheckMethod(map,map->ops->binlocal,"ISMappingBinLocal");
  ierr = PetscLogEventBegin(IS_MAPPING_BinLocal,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->binlocal)(map,array,index,bins);       CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_BinLocal,map,0,0,0);   CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "ISMappingBin"
/*@
    ISMappingBin             - group local indices from the rank's support into n groups or "bins" (some possibly empty)
                               according to which of the n image indices they are mapped to on this rank. The groups are 
                               concatenated and returned as a single array. See ISMappingBinSplit() if separate bin output 
                               is desired.
                               Since ISMapping is potentially multivalued, the same index can appear in multiple bins.
                               The binning is done only on the indices of the selected type (I or J); the other indices and weights, 
                               if any, are moved to the appropriate bin together with the selected indices.


    Not collective

    Input Parameters:
+   map    - mapping of indices
.   array  - ISArray with indices to bin
-   index  - selection of the index to bin on (ISARRAY_I or ISARRAY_J; PETSC_NULL is equivalent to ISARRAY_I)


    Output Parameters:
.   bins    - ISArray containing the concatenated binned indices; the number of bins is the same as the result of ISGetImageSizeLocal().

    Level: advanced

    Concepts: binning^global indices

.seealso: ISMappingGetSupport(), ISMappingGetImage(), ISMappingGetSupportSizeLocal(), ISMappingGetImageSizeLocal(),
          ISMappingBinLocal(),   ISMappingMapLocal(), ISMappingMapLocal(),            ISMappingBinSplit()

@*/
PetscErrorCode ISMappingBin(ISMapping map, ISArray array, ISArrayIndex index, ISArray bins)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, IS_MAPPING_CLASSID, 1);
  PetscValidPointer(bins,4);
  ISMappingCheckAssembled(map,PETSC_TRUE, 1);
  if(!index) index = ISARRAY_I;
  ISMappingCheckMethod(map,map->ops->bin,"ISMappingBin");
  ierr = PetscLogEventBegin(IS_MAPPING_Bin,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->bin)(map,array,index,bins);       CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_Bin,map,0,0,0);   CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingMapSplit"
/*@
    ISMappingMapSplit      - maps an ISArray with global indices from the rank's support to global indices from the rank's range.
                             The image of each index is a separate ISArray. See ISMappingMap, if concatenated output is desired. 
                             Since ISMapping is in general multivalued, some global indices are mapped to multiple global indices.  
                             Only the indices of the selected type (I or J) are mapped; the other indices and weights, if any, 
                             are preserved on the images.


    Not collective

    Input Parameters:
+   map    - mapping of indices
.   inarr  - input ISArray
-   index  - selection of the index to map (ISARRAY_I or ISARRAY_J; PETSC_NULL is equivalent to ISARRAY_I)


    Output Parameters:
.   outarrs - ISArray list; the list length is the same as inarr's ISArray length.


    Level: advanced

    Concepts: mapping^indices global split

.seealso: ISMappingGetSupport(), ISMappingGetImage(), ISMappingGetSupportSizeLocal(), ISMappingGetImageSizeLocal(),
          ISMappingMap(),        ISMappingMapLocalSplit(), ISMappingBinSplit(),       ISMappingBinSplitLocal()

@*/
PetscErrorCode ISMappingMapSplit(ISMapping map, ISArray inarr, ISArrayIndex index, ISArray *outarr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, IS_MAPPING_CLASSID, 1);
  PetscValidPointer(outarr, 4);
  if(!index) index = ISARRAY_I;
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  ISMappingCheckMethod(map,map->ops->mapsplit,"ISMappingMapSplit");
  ierr = PetscLogEventBegin(IS_MAPPING_MapSplit,map,0,0,0);  CHKERRQ(ierr);
  ierr = (*map->ops->mapsplit)(map,inarr,index,outarr); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_MapSplit,map,0,0,0);    CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingMapSplitLocal"
/*@
    ISMappingMapSplitLocal - maps an ISArray with local indices from the rank's support to global indices from the rank's range.
                             The image of each index is a separate ISArray. Since ISMapping is in general multivalued, some local 
                             indices are mapped to multiple global indices.  Only the indices of the selected type (I or J) are mapped; 
                             the other indices and weights, if any, are preserved on the images.


    Not collective

    Input Parameters:
+   map    - mapping of indices
.   inarr  - input ISArray
-   index   - selection of the index to map (ISARRAY_I or ISARRAY_J; PETSC_NULL is equivalent to ISARRAY_I)


    Output Parameters:
.   outarrs - ISArray list; the list length is the same as inarr's ISArray length.


    Level: advanced

    Concepts: mapping^indices local split

.seealso: ISMappingGetSupport(), ISMappingGetImage(), ISMappingGetSupportSizeLocal(), ISMappingGetImageSizeLocal(),
          ISMappingMapLocal(),   ISMappingMapSplit(), ISMappingBinSplit(),            ISMappingBinSplitLocal()

@*/
PetscErrorCode ISMappingMapSplitLocal(ISMapping map, ISArray inarr, ISArrayIndex index, ISArray *outarr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, IS_MAPPING_CLASSID, 1);
  PetscValidPointer(outarr, 4);
  if(!index) index = ISARRAY_I;
  ISMappingCheckAssembled(map,PETSC_TRUE,1);
  ISMappingCheckMethod(map,map->ops->mapsplitlocal,"ISMappingMapSplitLocal");
  ierr = PetscLogEventBegin(IS_MAPPING_MapSplitLocal,map,0,0,0);  CHKERRQ(ierr);
  ierr = (*map->ops->mapsplitlocal)(map,inarr,index,outarr);      CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_MapSplitLocal,map,0,0,0);    CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISMappingBinSplitLocal"
/*@
    ISMappingBinSplitLocal   - order local indices from the rank's support into n consecutive groups or "bins" (some possibly empty)
                               according to which of the n image indices they are mapped to on this rank. The bins are returned 
                               as individual ISArrays. See ISMappingBinLocal() if concatenated bin output is desired.
                               Since ISMapping is potentially multivalued, the same index can appear in multiple bins.
                               The binning is done on the indices of the selected type (I or J); the other indices and weights, if any, 
                               are moved to the appropriate bin together with the selected indices.


    Not collective

    Input Parameters:
+   map    - mapping of indices
.   array  - ISArray with indices to bin
-   index  - selection of the index to bin on (ISARRAY_I or ISARRAY_J; PETSC_NULL is equivalent to ISARRAY_I)


    Output Parameters:
.   bins    - ISArray list of bins; the number of bins is the same as the result of ISGetImageSizeLocal().

    Level: advanced

    Concepts: binning^local indices split

.seealso: ISMappingGetSupport(), ISMappingGetImage(), ISMappingGetSupportSizeLocal(), ISMappingGetImageSizeLocal(),
          ISMappingBinLocal(),   ISMappingMapSplit(), ISMappingMapSplitLocal(),       ISMappingBinSplit()

@*/
PetscErrorCode ISMappingBinSplitLocal(ISMapping map, ISArray array, ISArrayIndex index, ISArray *bins)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, IS_MAPPING_CLASSID, 1);
  PetscValidPointer(bins,4);
  ISMappingCheckAssembled(map,PETSC_TRUE, 1);
  if(!index) index = ISARRAY_I;
  ISMappingCheckMethod(map,map->ops->binsplitlocal,"ISMappingBinSplitLocal");
  ierr = PetscLogEventBegin(IS_MAPPING_BinSplitLocal,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->binsplitlocal)(map,array,index,bins);       CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_BinSplitLocal,map,0,0,0);   CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "ISMappingBinSplit"
/*@
    ISMappingBinSplit        - group global indices from the rank's support into n groups or "bins" (some possibly empty)
                               according to which of the n image indices they are mapped to on this rank. The bins and 
                               returned as individual ISArrays. See ISMappingBin() if concatenated bin output is desired.
                               Since ISMapping is potentially multivalued, the same index can appear in multiple bins.
                               The binning is done on the indices of selected type (I or J); the other indices and weights, 
                               if any, are moved to the appropriate bin together with the selected indices.


    Not collective

    Input Parameters:
+   map    - mapping of indices
.   array  - ISArray with indices to bin
-   index  - selection of the index to bin on (ISARRAY_I or ISARRAY_J; PETSC_NULL is equivalent to ISARRAY_I)


    Output Parameters:
.   bins    - ISArray list of bins; the number of bins is the same as the result of ISGetImageSizeLocal().

    Level: advanced

    Concepts: binning^global indices split

.seealso: ISMappingGetSupport(), ISMappingGetImage(), ISMappingGetSupportSizeLocal(), ISMappingGetImageSizeLocal(),
          ISMappingBin(),        ISMappingMapSplit(), ISMappingMapSplitLocal(),       ISMappingBinSplitLocal()

@*/
PetscErrorCode ISMappingBinSplit(ISMapping map, ISArray array, ISArrayIndex index, ISArray *bins)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, IS_MAPPING_CLASSID, 1);
  PetscValidPointer(bins,4);
  ISMappingCheckAssembled(map,PETSC_TRUE, 1);
  if(!index) index = ISARRAY_I;
  ISMappingCheckMethod(map,map->ops->binsplit,"ISMappingBinSplit");
  ierr = PetscLogEventBegin(IS_MAPPING_BinSplit,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->binsplit)(map,array,index,bins);       CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IS_MAPPING_BinSplit,map,0,0,0);   CHKERRQ(ierr);
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
#define __FUNCT__ "ISMappingSetUp_ISMapping"
PetscErrorCode ISMappingSetUp_ISMapping(ISMapping map)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscLayoutSetBlockSize(map->xlayout,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(map->ylayout,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(map->xlayout);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(map->ylayout);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}/* ISMappingSetUp_ISMapping() */

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
  PetscValidHeaderSpecific(map, IS_MAPPING_CLASSID, 1);
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
PetscErrorCode ISArrayHunkCreate(PetscInt maxlength, ISArrayComponents mask, ISArrayHunk *_hunk) 
{
  PetscErrorCode ierr;
  ISArrayHunk hunk;
  PetscFunctionBegin;
  PetscValidPointer(_hunk,3);

  if(maxlength <= 0) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Nonpositive ISArrayHunk maxlength: %D", maxlength);

  ierr = PetscNew(struct _n_ISArrayHunk, &hunk); CHKERRQ(ierr);
  hunk->mask = mask;
  hunk->maxlength = maxlength;
  if(mask & ISARRAY_I) {
    PetscInt *ia;
    ierr = PetscMalloc(sizeof(PetscInt)*maxlength, &ia);  CHKERRQ(ierr);
    ierr = PetscMemzero(ia, sizeof(PetscInt)*maxlength);  CHKERRQ(ierr);
    hunk->i = ia;
  }
  if(mask & ISARRAY_J) {
    PetscInt *ja;
    ierr = PetscMalloc(sizeof(PetscInt)*maxlength, &ja);  CHKERRQ(ierr);
    ierr = PetscMemzero(ja, sizeof(PetscInt)*maxlength); CHKERRQ(ierr);
    hunk->j = ja;
  }
  if(mask & ISARRAY_W) {
    PetscScalar *wa;
    ierr = PetscMalloc(sizeof(PetscScalar)*maxlength, &wa);  CHKERRQ(ierr);
    ierr = PetscMemzero(wa, sizeof(PetscScalar)*maxlength); CHKERRQ(ierr);
    hunk->w = wa;
  }
  hunk->mode = PETSC_OWN_POINTER;
  hunk->length = 0;
  *_hunk = hunk;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayHunkAddData"
PetscErrorCode ISArrayHunkAddData(ISArrayHunk hunk, PetscInt length, const PetscInt *i, const PetscScalar *w, const PetscInt *j) 
{
  PetscErrorCode ierr;
  PetscInt mask;
  PetscFunctionBegin;
  PetscValidPointer(hunk,1);
  mask = (i != PETSC_NULL) | ((j != PETSC_NULL)<<1) | ((w != PETSC_NULL)<<2);
  if(mask != hunk->mask) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Data components %D incompatible with the ISArrayHunk mask", mask,hunk->mask);
  if(mask & ISARRAY_I) {
    ierr = PetscMemcpy(hunk->i, i, sizeof(PetscInt)*length);
  }
  if(mask & ISARRAY_J) {
    ierr = PetscMemcpy(hunk->j, j, sizeof(PetscInt)*length);
  }
  if(mask & ISARRAY_W) {
    ierr = PetscMemcpy(hunk->w, w, sizeof(PetscScalar)*length);
  }
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "ISArrayHunkGetSubHunk"
PetscErrorCode ISArrayHunkGetSubHunk(ISArrayHunk hunk, PetscInt maxlength, ISArrayComponents mask, ISArrayHunk *_subhunk) 
{
  PetscErrorCode ierr;
  ISArrayHunk subhunk;
  PetscFunctionBegin;
  PetscValidPointer(hunk,1);
  PetscValidPointer(_subhunk,3);
  *_subhunk = PETSC_NULL;
  if(maxlength <= 0) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Nonpositive subhunk maxlength: %D", maxlength);
  if(mask & (~(hunk->mask))) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Subhunk mask %D is not a submask of the hunk mask %D", mask, hunk->mask);

  if(hunk->length+maxlength > hunk->maxlength) PetscFunctionReturn(0);
  ierr = PetscNew(struct _n_ISArrayHunk, &subhunk); CHKERRQ(ierr);
  subhunk->mask = mask;
  subhunk->maxlength = maxlength;
  subhunk->length   = 0;
  if(mask & ISARRAY_I) {
    subhunk->i = hunk->i+hunk->length;
  }
  if(mask & ISARRAY_J) {
    subhunk->j = hunk->j+hunk->length;
  }
  if(mask & ISARRAY_W) {
    subhunk->w = hunk->w+hunk->length;
  }
  subhunk->mode = PETSC_USE_POINTER;
  hunk->length += maxlength;
  ++(hunk->refcnt);
  *_subhunk = subhunk;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "ISArrayHunkDestroy"
PetscErrorCode ISArrayHunkDestroy(ISArrayHunk hunk) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if((hunk->refcnt)--) PetscFunctionReturn(0);
  if(hunk->length && hunk->mode != PETSC_USE_POINTER) {
      ierr = PetscFree(hunk->i); CHKERRQ(ierr);
      ierr = PetscFree(hunk->w); CHKERRQ(ierr);
      ierr = PetscFree(hunk->j); CHKERRQ(ierr);
  }
  hunk->length = 0;
  if(hunk->parent) {
    ierr = ISArrayHunkDestroy(hunk->parent); CHKERRQ(ierr);
  }
  hunk->parent = PETSC_NULL;
  ierr = PetscFree(hunk); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "ISArrayCreate"
PetscErrorCode ISArrayCreate(ISArrayComponents mask, ISArray *_arr) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_arr,2);
  if(!(mask & ISARRAY_I) && !(mask & ISARRAY_J)) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "ISArrayComponents %D must contain at least one of the indices: I or J", mask);
  }
  ierr = PetscNew(struct _n_ISArray, _arr); CHKERRQ(ierr);
  (*_arr)->mask = mask;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayDuplicate"
PetscErrorCode ISArrayDuplicate(ISArray arr, ISArray *_darr) 
{
  PetscErrorCode ierr;
  ISArray darr;
  ISArrayLink link;
  PetscFunctionBegin;
  PetscValidPointer(arr,1);
  PetscValidPointer(_darr,2);
  ierr = ISArrayCreate(arr->mask, &darr); CHKERRQ(ierr);
  link  = arr->first;
  while(link) {
    ierr = ISArrayAddHunk(darr,link->hunk);  CHKERRQ(ierr);
  }
  *_darr = arr;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayCreateArrays"
PetscErrorCode ISArrayCreateArrays(ISArrayComponents mask, PetscInt count, ISArray **_arrays) 
{
  PetscErrorCode ierr;
  PetscInt i;
  PetscFunctionBegin;
  PetscValidPointer(_arrays,3);
  if(!(mask & ISARRAY_I) && !(mask & ISARRAY_J)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "ISArrayComponents %D must contain at least one of the indices: I or J", mask);
  if(count < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Negative array count: %D", count);
  ierr = PetscMalloc(sizeof(ISArray), _arrays); CHKERRQ(ierr);
  for(i = 0; i < count; ++i) {
    ierr = ISArrayCreate(mask, *_arrays+i); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayClear"
PetscErrorCode ISArrayClear(ISArray arr) 
{
  PetscErrorCode ierr;
  ISArrayLink    link;
  PetscFunctionBegin;
  PetscValidPointer(arr,1);
  link = arr->first;
  while(link) {
    ierr = ISArrayHunkDestroy(link->hunk); CHKERRQ(ierr);
    link = link->next;
  }
  arr->first = PETSC_NULL;
  arr->last  = PETSC_NULL;
  arr->length = 0;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "ISArrayDestroy"
PetscErrorCode ISArrayDestroy(ISArray chain) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  ierr = ISArrayClear(chain); CHKERRQ(ierr);
  chain->mask   = 0;
  ierr = PetscFree(chain);    CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* 
 Right now we merely allocate a new hunk.
 In the future, this can manage a pool of hunks, use a buffer to draw subhunks from, etc.
 */
#undef  __FUNCT__
#define __FUNCT__ "ISArrayGetHunk"
PetscErrorCode ISArrayGetHunk(ISArray chain, PetscInt length, ISArrayHunk *_hunk) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = ISArrayHunkCreate(length, chain->mask, _hunk); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayAddHunk"
PetscErrorCode ISArrayAddHunk(ISArray chain, ISArrayHunk hunk) 
{
  PetscErrorCode ierr;
  ISArrayLink    link;
  PetscFunctionBegin;
  if(chain->mask & (~(hunk->mask))) {
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Hunk mask %D incompatible with the array mask %D", hunk->mask, chain->mask);
  }
  ierr = PetscMalloc(sizeof(struct _n_ISArrayLink), &link);
  link->hunk = hunk;
  ++(hunk->refcnt);
  if(chain->last) {
    chain->last->next = link;
    chain->last =       link;
  }
  else {
    chain->first = chain->last = link;
  }
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "ISArrayAddData"
PetscErrorCode ISArrayAddData(ISArray chain, PetscInt length, const PetscInt *i, const PetscScalar *w, const PetscInt *j) 
{
  PetscErrorCode ierr;
  ISArrayHunk hunk;
  ISArrayComponents mask;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  mask = (i != PETSC_NULL) | ((j != PETSC_NULL)<<1) | ((w != PETSC_NULL)<<2);
  if(mask != chain->mask) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Data components %D incompatible with the ISArrayComponents", mask,chain->mask);
  ierr = ISArrayGetHunk(chain, length, &hunk);  CHKERRQ(ierr);
  ierr = ISArrayHunkAddData(hunk, length, i,w,j);       CHKERRQ(ierr);
  ierr = ISArrayAddHunk(chain, hunk); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "ISArrayAddI"
PetscErrorCode ISArrayAddI(ISArray chain, PetscInt length, PetscInt i, const PetscScalar wa[], const PetscInt ja[]) 
{
  PetscErrorCode ierr;
  ISArrayHunk hunk;
  PetscInt mask;
  PetscInt k;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  mask = (ISARRAY_I | (ja != PETSC_NULL)<<1 | (wa != PETSC_NULL)<<2);
  if(mask & (~(chain->mask))) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Array components provided %D incompatible with array mask %D", mask, chain->mask);
  if(!length) PetscFunctionReturn(0);
  ierr = ISArrayGetHunk(chain, length, &hunk);           CHKERRQ(ierr);
  for(k = 0; k < length; ++k) hunk->i[k] = i;
  if(ja) {
    ierr = PetscMemcpy(hunk->j, ja, sizeof(PetscInt)*length);    CHKERRQ(ierr);
  }
  if(wa) {
    ierr = PetscMemcpy(hunk->w, wa, sizeof(PetscScalar)*length); CHKERRQ(ierr);
  }
  ierr = ISArrayAddHunk(chain, hunk);                            CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayAddJ"
PetscErrorCode ISArrayAddJ(ISArray chain, PetscInt length, const PetscInt ia[], const PetscScalar wa[], PetscInt j) 
{
  PetscErrorCode ierr;
  ISArrayHunk hunk;
  PetscInt mask;
  PetscInt k;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  mask = ((ia != PETSC_NULL) | ISARRAY_J | (wa != PETSC_NULL)<<2);
  if(mask & (~(chain->mask))) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Array components provided %D incompatible with array mask %D", mask, chain->mask);
  if(!length) PetscFunctionReturn(0);
  ierr = ISArrayGetHunk(chain, length, &hunk);           CHKERRQ(ierr);
  for(k = 0; k < length; ++k) hunk->j[k] = j;
  if(ia) {
    ierr = PetscMemcpy(hunk->i, ia, sizeof(PetscInt)*length);    CHKERRQ(ierr);
  }
  if(wa) {
    ierr = PetscMemcpy(hunk->w, wa, sizeof(PetscScalar)*length); CHKERRQ(ierr);
  }
  ierr = ISArrayAddHunk(chain, hunk);                            CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayMerge_Private"
static PetscErrorCode ISArrayMerge_Private(ISArray arr) 
{
  PetscErrorCode ierr;
  ISArrayLink link;
  ISArrayHunk merged;
  PetscInt count, offset;
  PetscFunctionBegin;
  PetscValidPointer(arr,1);

  /* Determine the number of links in the chain and perform the mask consistency check. */
  link = arr->first;
  count = 0;
  while(link) {
    /* Mask consistency check. */
    if(arr->mask & (~(link->hunk->mask))) {
      SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Hunk components %D in link %D incompatible with the merge mask %D", link->hunk->mask, count, arr->mask);
    }
    link     = link->next;
    ++count;
  }
  if(count == 1) PetscFunctionReturn(0);

  if(arr->length) {
    ierr = ISArrayHunkCreate(arr->length, arr->mask, &merged);                           CHKERRQ(ierr);
    /* Copy the indices and weights into the merged arrays. */
    offset = 0;
    link = arr->first;
    while(link) {
      ISArrayHunk hunk = link->hunk;
      if(arr->mask & ISARRAY_I) {
        ierr = PetscMemcpy(merged->i+offset, hunk->i, sizeof(PetscInt)*hunk->length);    CHKERRQ(ierr);
      }
      if(arr->mask & ISARRAY_J) {
        ierr = PetscMemcpy(merged->j+offset, hunk->j, sizeof(PetscInt)*hunk->length);    CHKERRQ(ierr);
      }
      if(arr->mask & ISARRAY_W) {
        ierr = PetscMemcpy(merged->w+offset, hunk->w, sizeof(PetscScalar)*hunk->length); CHKERRQ(ierr);
      }
      offset += hunk->length;
    }
  }/* if(arr->length) */
  merged->length = offset;
  ierr = ISArrayClear(arr);            CHKERRQ(ierr);
  ierr = ISArrayAddHunk(arr, merged);  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayGetLength"
PetscErrorCode ISArrayGetLength(ISArray chain, PetscInt *_length) 
{
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  PetscValidPointer(_length,2);
  *_length = chain->length;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayGetData"
PetscErrorCode ISArrayGetData(ISArray chain, const PetscInt *_i[], const PetscScalar *_w[], const PetscInt *_j[]) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  ierr = ISArrayMerge_Private(chain); CHKERRQ(ierr);
  if(_i && (chain->mask&ISARRAY_I)) {
    if(chain->first) *_i = chain->first->hunk->i;
    else             *_i = PETSC_NULL;
  }
  if(_w && (chain->mask&ISARRAY_W)) {
    if(chain->first) *_w = chain->first->hunk->w;
    else             *_w = PETSC_NULL;
  }
  if(_j && (chain->mask&ISARRAY_J)) {
    if(chain->first) *_j = chain->first->hunk->j;
    else             *_j = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ISArrayAddArray"
PetscErrorCode ISArrayAddArray(ISArray chain, ISArray chain2) 
{
  PetscErrorCode ierr;
  ISArrayLink link;
  PetscFunctionBegin;
  PetscValidPointer(chain, 1);
  PetscValidPointer(chain2,2);
  if(chain->mask & (~(chain2->mask))) {
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "New mask %D ncompatible with original %D", chain2->mask, chain->mask);
  }
  link = chain2->first;
  while(link) {
    ierr = ISArrayAddHunk(chain, link->hunk); CHKERRQ(ierr);
    link = link->next;
  }
  PetscFunctionReturn(0);
}





/*
     Checks whether all indices are within [imin,imax) and generate an error, if they are not and 
     if outOfBoundsError == PETSC_TRUE.  Return the result in flag.
 */
#undef __FUNCT__  
#define __FUNCT__ "PetscCheckIntArrayRange"
PetscErrorCode PetscCheckIntArrayRange(PetscInt len, const PetscInt idx[],  PetscInt imin, PetscInt imax, PetscBool outOfBoundsError, PetscBool *flag)
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
     Checks if any indices are within [imin,imax) and generate an error, if they are not and 
     if outOfBoundsError == PETSC_TRUE.  Return the result in flag.
 */
#undef __FUNCT__  
#define __FUNCT__ "PetscCheckISRange"
PetscErrorCode PetscCheckISRange(IS is, PetscInt imin, PetscInt imax, PetscBool outOfBoundsError, PetscBool *flag)
{
  PetscInt n;
  PetscBool inBounds = PETSC_TRUE, isstride;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  ierr = ISGetLocalSize(is, &n); CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)is,ISSTRIDE,&isstride); CHKERRQ(ierr);
  if(isstride) {
    PetscInt first, step, last;
    
    ierr = ISStrideGetInfo(is, &first, &step); CHKERRQ(ierr);
    last = first + step*n;
    if (first < imin || last < imin) {
      if(outOfBoundsError) 
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index smaller than min %D", imin);
      inBounds = PETSC_FALSE;
      goto functionend;
    }
    if (first >= imax || last >= imax) {
      if(outOfBoundsError) 
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index greater than max %D", imax);
      inBounds = PETSC_FALSE;
      goto functionend;
    }
  } else { /* not stride */
    const PetscInt *idx;
    ierr = ISGetIndices(is, &idx); CHKERRQ(ierr);
    ierr = PetscCheckIntArrayRange(n,idx,imin,imax,outOfBoundsError,flag); CHKERRQ(ierr);
    ierr = ISRestoreIndices(is, &idx); CHKERRQ(ierr);
  }
  functionend:
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
PetscErrorCode ISArrayAssemble(ISArray chain, PetscInt mask, PetscLayout layout, ISArray *_achain) 
{
  PetscErrorCode ierr;
  ISArray achain;
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
#if defined(PETSC_USE_DEBUG)
  PetscBool found;
#endif
  PetscInt ni, nv, count, alength;
  PetscInt *aixidx, *aiyidx = PETSC_NULL;
  PetscScalar *aval = PETSC_NULL;
  const PetscScalar *val;
  ISArrayLink link;
  ISArrayHunk hunk, ahunk;

  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  PetscValidPointer(_achain, 4);

  /* Make sure that at least one of the indices -- I or J -- is being assembled on, and that the index being assembled on is present in the ISArray. */
  if((mask != ISARRAY_I && mask != ISARRAY_J)|| !(mask & chain->mask))
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot assemble ISArray with components %D on component %D", chain->mask, mask);

  /* Comm parameters */
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* If layout isn't parallel, then this is a noop. */
  if(size == 1) PetscFunctionReturn(0);

  /* Make sure that chain type is the same across the comm. */
  chainmaskmpi = PetscMPIIntCast(chain->mask);
  ierr = MPI_Allreduce(&(chainmaskmpi), &allchainmask, 1, MPI_INT, MPI_BOR, comm); CHKERRQ(ierr);
  if(allchainmask^chainmaskmpi) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Chain mask must be the same across the communicator. Got %D globally and %D locally", allchainmask, chainmaskmpi);

  /* How many index arrays are being sent? One or two? */
  ni = 0;
  ni += ((chain->mask & ISARRAY_I) > 0);
  ni += ((chain->mask & ISARRAY_J) > 0);

  /* How many value arrays are being sent?  One or none? */
  nv =  ((chain->mask & ISARRAY_W) > 0);


  
  /*
   Each processor ships off its ixidx[j] and, possibly, the appropriate combination of iyidx[j] 
   and val[j] to the appropriate processor.
   */
  /*  first count number of contributors to each processor */
  ierr  = PetscMalloc2(size,PetscMPIInt,&plengths,chain->length,PetscInt,&owner);CHKERRQ(ierr);
  ierr  = PetscMemzero(plengths,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  lastidx = -1;
  count   = 0;
  p       = 0;
  link = chain->first;
  while(link) {
    hunk = link->hunk;
    for (i=0; i<hunk->length; ++i) {
      if(mask == ISARRAY_I) {
        ixidx = hunk->i;
        iyidx = hunk->j;
      }
      else {
        ixidx = hunk->j;
        iyidx = hunk->i;
      }
      /* if indices are NOT locally sorted, need to start search for the proc owning inidx[i] at the beginning */
      if (lastidx > (idx = ixidx[i])) p = 0;
      lastidx = idx;
      for (; p<size; ++p) {
        if (idx >= layout->range[p] && idx < layout->range[p+1]) {
          plengths[p]++; 
          owner[count] = p; 
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
      ++count;
    }/* for(i=0; i < hunk->length; ++i) */
    link = link->next;
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
  ierr     = PetscMalloc(ni*chain->length*sizeof(PetscInt),&sindices);      CHKERRQ(ierr);
  if(nv) {
    ierr     = PetscMalloc(chain->length*sizeof(PetscScalar),&svalues);     CHKERRQ(ierr);
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
  link = chain->first;
  count = 0;
  while(link){
    hunk = link->hunk;
    if(mask == ISARRAY_I) {
      ixidx = hunk->i;
      iyidx = hunk->j;
    }
    else {
      ixidx = hunk->j;
      iyidx = hunk->i;
    }
    val = hunk->w;
    for(i = 0; i < hunk->length; ++i) {
      p = owner[count];
      sindices[ni*sstarts[p]+plengths[p]]                             = ixidx[i];
      if(ni==2) 
        sindices[ni*sstarts[p]+(sstarts[p+1]-sstarts[p])+plengths[p]] = iyidx[i];
      if(nv) 
        svalues[sstarts[p]+plengths[p]]                               = val[i];
      ++plengths[p];
      ++count;
    }
    link = link->next;
  }
  /* Allocate send requests: for the indices, and possibly one more for the scalar values, hence +nv */
  ierr     = PetscMalloc((1+nv)*nsends*sizeof(MPI_Request),&send_reqs);  CHKERRQ(ierr);

  /* Post sends */
  for (p=0,count=0; p<size; ++p) {
    if (plengths[p]) {
      ierr = MPI_Isend(sindices+ni*sstarts[p],ni*plengths[p],MPIU_INT,p,tag_i,comm,send_reqs+count++);CHKERRQ(ierr);
      if(nv) {
        ierr = MPI_Isend(svalues+sstarts[p],plengths[p],MPIU_SCALAR,p,tag_v,comm,send_reqs+count++);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFree2(plengths,owner);CHKERRQ(ierr);
  ierr = PetscFree(sstarts);CHKERRQ(ierr);

  /* Prepare to receive indices and values. */
  /* Compute the offsets of the individual received segments in the unified index/value arrays. */
  ierr = PetscMalloc(sizeof(PetscMPIInt)*(nrecvs+1), &rstarts); CHKERRQ(ierr);
  rstarts[0] = 0;
  for(j = 0; j < nrecvs; ++j) rstarts[j+1] = rstarts[j] + rlengths[j];

  alength = rstarts[nrecvs];
  /* Create a new ISArray for the received data segments */
  ierr = ISArrayCreate(chain->mask, &achain);               CHKERRQ(ierr);
  /* Get a hunk to pack the data into. */
  ierr = ISArrayGetHunk(achain, alength, &ahunk);   CHKERRQ(ierr);
  
  /* Use ahunk's data arrays as receive buffers. */
  if(mask == ISARRAY_I) {
    aixidx = ahunk->i;
    aiyidx = ahunk->j;
  }
  else {
    aiyidx = ahunk->i;
    aixidx = ahunk->j;
  }
  aval = ahunk->w;

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

  *_achain = achain;
  PetscFunctionReturn(0);
}/* ISArrayAssemble() */





