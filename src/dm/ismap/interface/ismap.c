#define PETSCDM_DLL

#include "private/ismapimpl.h"  /*I "petscmap.h"  I*/


PetscClassId  SA_MAPPING_CLASSID;
PetscLogEvent SA_MAPPING_Map, SA_MAPPING_MapLocal, SA_MAPPING_MapSplit, SA_MAPPING_MapSplitLocal;
PetscLogEvent SA_MAPPING_Bin, SA_MAPPING_BinLocal, SA_MAPPING_BinSplit, SA_MAPPING_BinSplitLocal;
PetscLogEvent SA_MAPPING_AssemblyBegin, SA_MAPPING_AssemblyEnd, SA_MAPPING_Invert, SA_MAPPING_Pushforward, SA_MAPPING_Pullback;

PetscFList SAMappingList               = PETSC_NULL;
PetscBool  SAMappingRegisterAllCalled  = PETSC_FALSE;
PetscBool  SAMappingPackageInitialized = PETSC_FALSE;

EXTERN_C_BEGIN
extern PetscErrorCode SAMappingCreate_Graph(SAMapping);
EXTERN_C_END


extern PetscErrorCode  SAMappingRegisterAll(const char *path);


#undef __FUNCT__  
#define __FUNCT__ "SAMappingFinalizePackage"
/*@C
  SAMappingFinalizePackage - This function destroys everything in the SAMapping package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode  SAMappingFinalizePackage(void)
{
  PetscFunctionBegin;
  SAMappingPackageInitialized = PETSC_FALSE;
  SAMappingRegisterAllCalled  = PETSC_FALSE;
  SAMappingList               = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SAMappingInitializePackage"
/*@C
  SAMappingInitializePackage - This function initializes everything in the SAMapping package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to SAMappingCreate()
  when using static libraries.

  Input Parameter:
. path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: SAMapping, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode  SAMappingInitializePackage(const char path[])
{
  char              logList[256];
  char              *className;
  PetscBool         opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (SAMappingPackageInitialized) PetscFunctionReturn(0);

  SAMappingPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("SAMapping",&SA_MAPPING_CLASSID);                                         CHKERRQ(ierr);
  /* Register Constructors */
  ierr = SAMappingRegisterAll(path);                                                                    CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("SAMappingMap",           SA_MAPPING_CLASSID,&SA_MAPPING_Map);           CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SAMappingMapLocal",      SA_MAPPING_CLASSID,&SA_MAPPING_MapLocal);      CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SAMappingBin",           SA_MAPPING_CLASSID,&SA_MAPPING_Bin);           CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SAMappingBinLocal",      SA_MAPPING_CLASSID,&SA_MAPPING_BinLocal);      CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SAMappingMap",           SA_MAPPING_CLASSID,&SA_MAPPING_MapSplit);      CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SAMappingMapLocal",      SA_MAPPING_CLASSID,&SA_MAPPING_MapSplitLocal); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SAMappingBin",           SA_MAPPING_CLASSID,&SA_MAPPING_BinSplit);      CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SAMappingBinLocal",      SA_MAPPING_CLASSID,&SA_MAPPING_BinSplitLocal); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SAMappingAssemblyBegin", SA_MAPPING_CLASSID,&SA_MAPPING_AssemblyBegin); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SAMappingAssemblyEnd",   SA_MAPPING_CLASSID,&SA_MAPPING_AssemblyEnd);   CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SAMappingPushforward",   SA_MAPPING_CLASSID,&SA_MAPPING_Pushforward);   CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SAMappingPullback",      SA_MAPPING_CLASSID,&SA_MAPPING_Pullback);      CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SAMappingInvert",        SA_MAPPING_CLASSID,&SA_MAPPING_Invert);        CHKERRQ(ierr);

  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);                        CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "is_mapping", &className);                                              CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(SA_MAPPING_CLASSID);                                              CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "is_mapping", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(SA_MAPPING_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(SAMappingFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SAMappingRegister"
/*@C
  SAMappingRegister - See SAMappingRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode  SAMappingRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(SAMapping))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&SAMappingList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SAMappingRegisterDestroy"
/*@C
   SAMappingRegisterDestroy - Frees the list of SAMapping methods that were
   registered by SAMappingRegisterDynamic).

   Not Collective

   Level: developer

.keywords: SAMapping, register, destroy

.seealso: SAMappingRegisterDynamic), SAMappingRegisterAll()
@*/
PetscErrorCode  SAMappingRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&SAMappingList);CHKERRQ(ierr);
  SAMappingRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SAMappingRegisterAll"
/*@C
  SAMappingRegisterAll - Registers all of the mapping constructors in the SAMapping package.

  Not Collective

  Level: developer

.keywords: SAMapping, register, all

.seealso:  SAMappingRegisterDestroy(), SAMappingRegisterDynamic), SAMappingCreate(), 
           SAMappingSetType()
@*/
PetscErrorCode  SAMappingRegisterAll(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  SAMappingRegisterAllCalled = PETSC_TRUE;
  ierr = SAMappingRegisterDynamic(SA_MAPPING_GRAPH,path,"SAMappingCreate_Graph",SAMappingCreate_Graph);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SAMappingMapLocal"
/*@C
    SAMappingMapLocal - maps an SA with local indices from the rank's support to global indices from the rank's range.
                        Since SAMapping is in general multivalued, some local indices are mapped to multiple global indices.
                        Only selected indices (I or J) are mapped; the other indices and weights, if any, are preserved on 
                        the images.


    Not collective

    Input Parameters:
+   map    - mapping of indices
.   inarr  - input SA
-   index   - selection of the index to map (SA_I or SA_J; PETSC_NULL is equivalent to SA_I)


    Output Parameters:
.   outarr - SA with the selected indices mapped


    Level: advanced

    Concepts: mapping^indices

.seealso: SAMappingGetSupport(), SAMappingGetImage(), SAMappingGetSupportSizeLocal(), SAMappingGetImageSizeLocal(),
          SAMappingMap(),        SAMappingBin(),      SAMappingBinLocal(),            SAMappingMapSplitLocal()

@*/
PetscErrorCode SAMappingMapLocal(SAMapping map, SA inarr, SAIndex index, SA outarr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID, 1);
  PetscValidPointer(outarr, 4);
  if(!index) index = SA_I;
  SAMappingCheckAssembled(map,PETSC_TRUE,1);
  SAMappingCheckMethod(map,map->ops->maplocal,"SAMappingMapLocal");
  ierr = PetscLogEventBegin(SA_MAPPING_MapLocal,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->maplocal)(map,inarr,index,outarr);      CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SA_MAPPING_MapLocal,map,0,0,0);   CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SAMappingMap"
/*@C
    SAMappingMap      - maps an SA with global indices from the rank's support to global indices from the rank's range.
                        Since SAMapping is in general multivalued, some indices are mapped to multiple global indices.
                        Only indices of the selected type (I or J) are mapped; the other indices and weights, if any, are 
                        preserved on the images.

    Not collective

    Input Parameters:
+   map    - mapping of indices
.   inarr  - input SA of indices and weights to map
-   index  - selection of the index to map (SA_I or SA_J; PETSC_NULL is equivalent to SA_I)


    Output Parameters:
.   outarr - SA with the selected indices mapped


    Level: advanced

    Concepts: mapping^indices global

.seealso: SAMappingGetSupport(), SAMappingGetImage(), SAMappingGetSupportSizeLocal(), SAMappingGetImageSizeLocal(),
          SAMappingMapLocal(),   SAMappingBin(),      SAMappingBinLocal(),            SAMappingMapSlit()

@*/
PetscErrorCode SAMappingMap(SAMapping map, SA inarr, SAIndex index, SA outarr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID, 1);
  PetscValidPointer(outarr, 4);
  SAMappingCheckAssembled(map,PETSC_TRUE,1);
  if(!index) index = SA_I;
  SAMappingCheckMethod(map,map->ops->map,"SAMappingMap");
  ierr = PetscLogEventBegin(SA_MAPPING_Map,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->map)(map,inarr,index,outarr);     CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SA_MAPPING_Map,map,0,0,0);   CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SAMappingBinLocal"
/*@C
    SAMappingBinLocal        - order local indices from the rank's support into n consecutive groups or "bins" (some possibly empty)
                               according to which of the n image indices they are mapped to on this rank. The groups are concatenated
                               and returned as a single array. See SAMappingBinSplitLocal() if separate bin output is desired. 
                               Since SAMapping is potentially multivalued, the same index can appear in multiple bins.
                               The binning is done on the indices of the selected type(I or J); the other indices and weights, if any, 
                               are moved to the appropriate bin together with the selected indices.


    Not collective

    Input Parameters:
+   map    - mapping of indices
.   array  - SA with indices to bin
-   index  - selection of the index to bin on (SA_I or SA_J; PETSC_NULL is equivalent to SA_I)


    Output Parameters:
.   bins    - SA containing concatenated binned indices; the number of bins is the same as the result of ISGetImageSizeLocal().

    Level: advanced

    Concepts: binning^local indices

.seealso: SAMappingGetSupport(), SAMappingGetImage(), SAMappingGetSupportSizeLocal(), SAMappingGetImageSizeLocal(),
          SAMappingBin(),        SAMappingMapLocal(), SAMappingMapLocal(),            SAMappingBinSplitLocal()

@*/
PetscErrorCode SAMappingBinLocal(SAMapping map, SA array, SAIndex index, SA bins)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID, 1);
  PetscValidPointer(bins,4);
  SAMappingCheckAssembled(map,PETSC_TRUE, 1);
  if(!index) index = SA_I;
  SAMappingCheckMethod(map,map->ops->binlocal,"SAMappingBinLocal");
  ierr = PetscLogEventBegin(SA_MAPPING_BinLocal,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->binlocal)(map,array,index,bins);       CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SA_MAPPING_BinLocal,map,0,0,0);   CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SAMappingBin"
/*@C
    SAMappingBin             - group local indices from the rank's support into n groups or "bins" (some possibly empty)
                               according to which of the n image indices they are mapped to on this rank. The groups are 
                               concatenated and returned as a single array. See SAMappingBinSplit() if separate bin output 
                               is desired.
                               Since SAMapping is potentially multivalued, the same index can appear in multiple bins.
                               The binning is done only on the indices of the selected type (I or J); the other indices and weights, 
                               if any, are moved to the appropriate bin together with the selected indices.


    Not collective

    Input Parameters:
+   map    - mapping of indices
.   array  - SA with indices to bin
-   index  - selection of the index to bin on (SA_I or SA_J; PETSC_NULL is equivalent to SA_I)


    Output Parameters:
.   bins    - SA containing the concatenated binned indices; the number of bins is the same as the result of ISGetImageSizeLocal().

    Level: advanced

    Concepts: binning^global indices

.seealso: SAMappingGetSupport(), SAMappingGetImage(), SAMappingGetSupportSizeLocal(), SAMappingGetImageSizeLocal(),
          SAMappingBinLocal(),   SAMappingMapLocal(), SAMappingMapLocal(),            SAMappingBinSplit()

@*/
PetscErrorCode SAMappingBin(SAMapping map, SA array, SAIndex index, SA bins)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID, 1);
  PetscValidPointer(bins,4);
  SAMappingCheckAssembled(map,PETSC_TRUE, 1);
  if(!index) index = SA_I;
  SAMappingCheckMethod(map,map->ops->bin,"SAMappingBin");
  ierr = PetscLogEventBegin(SA_MAPPING_Bin,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->bin)(map,array,index,bins);       CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SA_MAPPING_Bin,map,0,0,0);   CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SAMappingMapSplit"
/*@C
    SAMappingMapSplit      - maps an SA with global indices from the rank's support to global indices from the rank's range.
                             The image of each index is a separate SA. See SAMappingMap, if concatenated output is desired. 
                             Since SAMapping is in general multivalued, some global indices are mapped to multiple global indices.  
                             Only the indices of the selected type (I or J) are mapped; the other indices and weights, if any, 
                             are preserved on the images.


    Not collective

    Input Parameters:
+   map    - mapping of indices
.   inarr  - input SA
-   index  - selection of the index to map (SA_I or SA_J; PETSC_NULL is equivalent to SA_I)


    Output Parameters:
.   outarrs - SA list; the list length is the same as inarr's SA length.


    Level: advanced

    Concepts: mapping^indices global split

.seealso: SAMappingGetSupport(), SAMappingGetImage(), SAMappingGetSupportSizeLocal(), SAMappingGetImageSizeLocal(),
          SAMappingMap(),        SAMappingMapLocalSplit(), SAMappingBinSplit(),       SAMappingBinSplitLocal()

@*/
PetscErrorCode SAMappingMapSplit(SAMapping map, SA inarr, SAIndex index, SA *outarr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID, 1);
  PetscValidPointer(outarr, 4);
  if(!index) index = SA_I;
  SAMappingCheckAssembled(map,PETSC_TRUE,1);
  SAMappingCheckMethod(map,map->ops->mapsplit,"SAMappingMapSplit");
  ierr = PetscLogEventBegin(SA_MAPPING_MapSplit,map,0,0,0);  CHKERRQ(ierr);
  ierr = (*map->ops->mapsplit)(map,inarr,index,outarr); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SA_MAPPING_MapSplit,map,0,0,0);    CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SAMappingMapSplitLocal"
/*@C
    SAMappingMapSplitLocal - maps an SA with local indices from the rank's support to global indices from the rank's range.
                             The image of each index is a separate SA. Since SAMapping is in general multivalued, some local 
                             indices are mapped to multiple global indices.  Only the indices of the selected type (I or J) are mapped; 
                             the other indices and weights, if any, are preserved on the images.


    Not collective

    Input Parameters:
+   map    - mapping of indices
.   inarr  - input SA
-   index   - selection of the index to map (SA_I or SA_J; PETSC_NULL is equivalent to SA_I)


    Output Parameters:
.   outarrs - SA list; the list length is the same as inarr's SA length.


    Level: advanced

    Concepts: mapping^indices local split

.seealso: SAMappingGetSupport(), SAMappingGetImage(), SAMappingGetSupportSizeLocal(), SAMappingGetImageSizeLocal(),
          SAMappingMapLocal(),   SAMappingMapSplit(), SAMappingBinSplit(),            SAMappingBinSplitLocal()

@*/
PetscErrorCode SAMappingMapSplitLocal(SAMapping map, SA inarr, SAIndex index, SA *outarr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID, 1);
  PetscValidPointer(outarr, 4);
  if(!index) index = SA_I;
  SAMappingCheckAssembled(map,PETSC_TRUE,1);
  SAMappingCheckMethod(map,map->ops->mapsplitlocal,"SAMappingMapSplitLocal");
  ierr = PetscLogEventBegin(SA_MAPPING_MapSplitLocal,map,0,0,0);  CHKERRQ(ierr);
  ierr = (*map->ops->mapsplitlocal)(map,inarr,index,outarr);      CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SA_MAPPING_MapSplitLocal,map,0,0,0);    CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SAMappingBinSplitLocal"
/*@C
    SAMappingBinSplitLocal   - order local indices from the rank's support into n consecutive groups or "bins" (some possibly empty)
                               according to which of the n image indices they are mapped to on this rank. The bins are returned 
                               as individual SAs. See SAMappingBinLocal() if concatenated bin output is desired.
                               Since SAMapping is potentially multivalued, the same index can appear in multiple bins.
                               The binning is done on the indices of the selected type (I or J); the other indices and weights, if any, 
                               are moved to the appropriate bin together with the selected indices.


    Not collective

    Input Parameters:
+   map    - mapping of indices
.   array  - SA with indices to bin
-   index  - selection of the index to bin on (SA_I or SA_J; PETSC_NULL is equivalent to SA_I)


    Output Parameters:
.   bins    - SA list of bins; the number of bins is the same as the result of ISGetImageSizeLocal().

    Level: advanced

    Concepts: binning^local indices split

.seealso: SAMappingGetSupport(), SAMappingGetImage(), SAMappingGetSupportSizeLocal(), SAMappingGetImageSizeLocal(),
          SAMappingBinLocal(),   SAMappingMapSplit(), SAMappingMapSplitLocal(),       SAMappingBinSplit()

@*/
PetscErrorCode SAMappingBinSplitLocal(SAMapping map, SA array, SAIndex index, SA *bins)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID, 1);
  PetscValidPointer(bins,4);
  SAMappingCheckAssembled(map,PETSC_TRUE, 1);
  if(!index) index = SA_I;
  SAMappingCheckMethod(map,map->ops->binsplitlocal,"SAMappingBinSplitLocal");
  ierr = PetscLogEventBegin(SA_MAPPING_BinSplitLocal,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->binsplitlocal)(map,array,index,bins);       CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SA_MAPPING_BinSplitLocal,map,0,0,0);   CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SAMappingBinSplit"
/*@C
    SAMappingBinSplit        - group global indices from the rank's support into n groups or "bins" (some possibly empty)
                               according to which of the n image indices they are mapped to on this rank. The bins and 
                               returned as individual SAs. See SAMappingBin() if concatenated bin output is desired.
                               Since SAMapping is potentially multivalued, the same index can appear in multiple bins.
                               The binning is done on the indices of selected type (I or J); the other indices and weights, 
                               if any, are moved to the appropriate bin together with the selected indices.


    Not collective

    Input Parameters:
+   map    - mapping of indices
.   array  - SA with indices to bin
-   index  - selection of the index to bin on (SA_I or SA_J; PETSC_NULL is equivalent to SA_I)


    Output Parameters:
.   bins    - SA list of bins; the number of bins is the same as the result of ISGetImageSizeLocal().

    Level: advanced

    Concepts: binning^global indices split

.seealso: SAMappingGetSupport(), SAMappingGetImage(), SAMappingGetSupportSizeLocal(), SAMappingGetImageSizeLocal(),
          SAMappingBin(),        SAMappingMapSplit(), SAMappingMapSplitLocal(),       SAMappingBinSplitLocal()

@*/
PetscErrorCode SAMappingBinSplit(SAMapping map, SA array, SAIndex index, SA *bins)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID, 1);
  PetscValidPointer(bins,4);
  SAMappingCheckAssembled(map,PETSC_TRUE, 1);
  if(!index) index = SA_I;
  SAMappingCheckMethod(map,map->ops->binsplit,"SAMappingBinSplit");
  ierr = PetscLogEventBegin(SA_MAPPING_BinSplit,map,0,0,0); CHKERRQ(ierr);
  ierr = (*map->ops->binsplit)(map,array,index,bins);       CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SA_MAPPING_BinSplit,map,0,0,0);   CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SAMappingSetSizes"
/*@C
  SAMappingSetSizes - Sets the local and global sizes for the domain and range, and checks to determine compatibility

  Collective on SAMapping

  Input Parameters:
+  map     - the mapping
.  m       - number of local  domain indices (or PETSC_DETERMINE)
.  n       - number of local  range columns  (or PETSC_DETERMINE)
.  M       - number of global domain indices (or PETSC_DETERMINE or PETSC_IGNORE)
-  N       - number of global range  indices (or PETSC_DETERMINE or PETSC_IGNORE)

   Notes:
   The sizes specify (i) what the domain and range for the graph underlying SAMapping is, 
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
   Whether the domain is laid out in parallel (P) or not (S), determines the behavior of SAMapping
   during assembly.  In case (P), the edges of the underlying graph are migrated to the rank that
   owns the corresponding domain indices.  SAMapping can map indices lying in its local range, 
   which is a subset of its local domain.  This means that due to parallel assembly edges inserted
   by different ranks might be used during the mapping.  This is completely analogous to matrix 
   assembly.

   When the domain is not laid out in parallel, no migration takes place and the mapping of indices
   is done purely locally.
   
   

   Support/Image:
   Observe that the support and image of the graph may be strictly smaller than its domain and range,
   if no edges from some domain points (or to some range points) are added to SAMapping.
   
   Operator:
   Observe also that the linear operator defined by SAMapping will behave essentially as a VecScatter
   (i)   between MPI vectors with sizes (m,M) and (n,N), if both the domain and the range are (P),
   (ii)  between an MPI Vec with size (m,M) and a collection of SEQ Vecs (one per rank) of local size (n), 
         if the domain is (P) and the range is (S),
   (iii) between a collection of SEQ Vecs (one per rank) of local size (m) and an MPI Vec of size (n,N),
         if the domain is (S) and the range is (P),
   (iv)  between collections of SEQ Vecs (one per rank) or local sizes (m) and (n), if both the domain 
         and the range are (S).

  Level: beginner

.seealso: SAMappingGetSizes(), SAMappingGetSupportSize(), SAMappingGetImageSize(), SAMappingMapIndicesLocal()
@*/
PetscErrorCode  SAMappingSetSizes(SAMapping map, PetscInt m, PetscInt n, PetscInt M, PetscInt N)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,SA_MAPPING_CLASSID,1); 
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
#define __FUNCT__ "SAMappingGetSizes"
PetscErrorCode  SAMappingGetSizes(SAMapping map, PetscInt *m, PetscInt *n, PetscInt *M, PetscInt *N)
{

  PetscFunctionBegin;
  *m = map->xlayout->n;
  *n = map->ylayout->n;
  *M = map->xlayout->N;
  *N = map->ylayout->N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SAMappingSetUp_SAMapping"
PetscErrorCode SAMappingSetUp_SAMapping(SAMapping map)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscLayoutSetBlockSize(map->xlayout,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(map->ylayout,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(map->xlayout);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(map->ylayout);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}/* SAMappingSetUp_SAMapping() */

#undef __FUNCT__  
#define __FUNCT__ "SAMappingSetUp"
/*@C
   SAMappingSetUp - Sets up the internal mapping data structures for the later use.

   Collective on SAMapping

   Input Parameters:
.  map - the SAMapping context

   Notes:
   For basic use of the SAMapping classes the user need not explicitly call
   SAMappingSetUp(), since these actions will happen automatically.

   Level: advanced

.keywords: SAMapping, setup

.seealso: SAMappingCreate(), SAMappingDestroy(), SAMappingSetSizes()
@*/
PetscErrorCode SAMappingSetUp(SAMapping map)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID,1);
  SAMappingCheckMethod(map,map->ops->setup,"SAMappingSetUp");
  ierr = (*(map->ops->setup))(map); CHKERRQ(ierr);
  map->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}/* SAMappingGetSizes() */

#undef  __FUNCT__
#define __FUNCT__ "SAMappingAssemblyBegin"
PetscErrorCode SAMappingAssemblyBegin(SAMapping map)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID,1);

  if(map->assembled) PetscFunctionReturn(0);
  ierr = SAMappingSetUp(map); CHKERRQ(ierr);
  
  SAMappingCheckMethod(map,map->ops->assemblybegin, "SAMappingAsemblyBegin");
  ierr = PetscLogEventBegin(SA_MAPPING_AssemblyBegin, map,0,0,0); CHKERRQ(ierr);
  ierr = (*(map->ops->assemblybegin))(map); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SA_MAPPING_AssemblyBegin, map,0,0,0); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}/* SAMappingAssemblyBegin() */

#undef __FUNCT__
#define __FUNCT__ "SAMappingAssemblyEnd"
PetscErrorCode SAMappingAssemblyEnd(SAMapping map)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID, 1);
  SAMappingCheckMethod(map,map->ops->assemblyend, "SAMappingAsemblyEnd");
  ierr = PetscLogEventBegin(SA_MAPPING_AssemblyEnd, map,0,0,0); CHKERRQ(ierr);
  ierr = (*(map->ops->assemblyend))(map); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SA_MAPPING_AssemblyBegin, map,0,0,0); CHKERRQ(ierr);
  map->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}/* SAMappingAssemblyEnd() */


#undef  __FUNCT__
#define __FUNCT__ "SAMappingGetSupport"
PetscErrorCode SAMappingGetSupport(SAMapping map, PetscInt *_len, PetscInt *_supp[]) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID,1);
  SAMappingCheckAssembled(map,PETSC_TRUE,1);
  if(!_len && !_supp) PetscFunctionReturn(0);
  SAMappingCheckMethod(map,map->ops->getsupport,"SAMappingGetSupport");
  ierr = (*(map->ops->getsupport))(map,_len,_supp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAMappingGetSupportIS"
PetscErrorCode SAMappingGetSupportIS(SAMapping map, IS *supp) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID,1);
  SAMappingCheckAssembled(map,PETSC_TRUE,1);
  if(!supp) PetscFunctionReturn(0);
  SAMappingCheckMethod(map,map->ops->getsupportis,"SAMappingGetSupportIS");
  ierr = (*(map->ops->getsupportis))(map,supp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SAMappingGetSupportSA"
PetscErrorCode SAMappingGetSupportSA(SAMapping map, SA *supp) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID,1);
  SAMappingCheckAssembled(map,PETSC_TRUE,1);
  if(!supp) PetscFunctionReturn(0);
  SAMappingCheckMethod(map,map->ops->getsupportsa,"SAMappingGetSupportSA");
  ierr = (*(map->ops->getsupportsa))(map,supp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAMappingGetImage"
PetscErrorCode SAMappingGetImage(SAMapping map, PetscInt *_len, PetscInt *_image[]) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID,1);
  SAMappingCheckAssembled(map,PETSC_TRUE,1);
  if(!_len && !_image) PetscFunctionReturn(0);
  SAMappingCheckMethod(map,map->ops->getimage,"SAMappingGetImage");
  ierr = (*(map->ops->getimage))(map,_len,_image); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAMappingGetImageIS"
PetscErrorCode SAMappingGetImageIS(SAMapping map, IS *image) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID,1);
  SAMappingCheckAssembled(map,PETSC_TRUE,1);
  if(!image) PetscFunctionReturn(0);
  SAMappingCheckMethod(map,map->ops->getimageis,"SAMappingGetImageIS");
  ierr = (*(map->ops->getimageis))(map,image); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAMappingGetImageSA"
PetscErrorCode SAMappingGetImageSA(SAMapping map, SA *image) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(map, SA_MAPPING_CLASSID,1);
  SAMappingCheckAssembled(map,PETSC_TRUE,1);
  if(!image) PetscFunctionReturn(0);
  SAMappingCheckMethod(map,map->ops->getimagesa,"SAMappingGetImageSA");
  ierr = (*(map->ops->getimagesa))(map,image); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SAMappingGetMaxImageSize"
PetscErrorCode SAMappingGetMaxImageSize(SAMapping map, PetscInt *maxsize)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,SA_MAPPING_CLASSID,1);
  PetscValidIntPointer(maxsize,2);
  SAMappingCheckAssembled(map,PETSC_TRUE,1);
  SAMappingCheckMethod(map, map->ops->getmaximagesize,"SAMappingGetMaxImageSize");
  ierr = (*map->ops->getmaximagesize)(map,maxsize); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SAMappingGetOperator"
PetscErrorCode SAMappingGetOperator(SAMapping map, Mat *mat)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,SA_MAPPING_CLASSID,1);
  PetscValidPointer(mat,2);
  SAMappingCheckAssembled(map,PETSC_TRUE,1);
  SAMappingCheckMethod(map, map->ops->getoperator,"SAMappingGetOperator");
  ierr = (*map->ops->getoperator)(map,mat); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAMappingView"
PetscErrorCode SAMappingView(SAMapping map, PetscViewer v) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,SA_MAPPING_CLASSID,1);
  SAMappingCheckAssembled(map,PETSC_TRUE,1);
  SAMappingCheckMethod(map,map->ops->view, "SAMappingView");
  ierr = (*(map->ops->view))(map,v); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* SAMappingView() */

#undef  __FUNCT__
#define __FUNCT__ "SAMappingInvert"
PetscErrorCode SAMappingInvert(SAMapping map, SAMapping *imap) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,SA_MAPPING_CLASSID, 1);
  PetscValidPointer(imap,2);
  SAMappingCheckMethod(map,map->ops->invert, "SAMappingInvert");
  ierr = PetscLogEventBegin(SA_MAPPING_Invert, map, 0,0,0); CHKERRQ(ierr);
  ierr = (*(map->ops->invert))(map,imap); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SA_MAPPING_Invert, map, 0,0,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SAMappingPullback"
/*@C
   SAMappingPullback - compose mappings C = A*B

   Collective on SAMapping

   Input Parameters:
+  A - the left  mapping
-  B - the right mapping


   Output Parameters:
.  C - the product mapping: domain as in A, range as in B


   Level: intermediate

.seealso: SAMappingPushforward(), SAMappingInvert()
@*/
PetscErrorCode  SAMappingPullback(SAMapping A,SAMapping B, SAMapping *C) 
{
  PetscErrorCode ierr;
  PetscErrorCode (*pullback)(SAMapping,SAMapping,SAMapping*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,SA_MAPPING_CLASSID,1);
  PetscValidType(A,1);
  if (!A->assembled) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled mapping");
  PetscValidHeaderSpecific(B,SA_MAPPING_CLASSID,2);
  PetscValidType(B,2);
  if (!B->assembled) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled mapping");
  PetscValidPointer(C,3);

  /* dispatch based on the type of A and B */
  char  pullbackname[256];
  ierr = PetscStrcpy(pullbackname,"SAMappingPullback_");CHKERRQ(ierr);
  ierr = PetscStrcat(pullbackname,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = PetscStrcat(pullbackname,"_");    CHKERRQ(ierr);
  ierr = PetscStrcat(pullbackname,((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = PetscStrcat(pullbackname,"_C");   CHKERRQ(ierr); /* e.g., pullbackname = "ISPullback_ismappingis_ismappingis_C" */
  ierr = PetscObjectQueryFunction((PetscObject)B,pullbackname,(void (**)(void))&pullback);CHKERRQ(ierr);
  if (!pullback) SETERRQ2(((PetscObject)A)->comm,PETSC_ERR_ARG_INCOMP,"SAMappingPullback requires A, %s, to be compatible with B, %s",((PetscObject)A)->type_name,((PetscObject)B)->type_name);    
  ierr = PetscLogEventBegin(SA_MAPPING_Pullback, A,B,0,0); CHKERRQ(ierr);
  ierr = (*pullback)(A,B,C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SA_MAPPING_Pullback, A,B,0,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "SAMappingPushforward"
/*@C
   SAMappingPushforward - mapping from the range of A to the range of B, pointwise on the common domain

   Collective on SAMapping

   Input Parameters:
+  A - the left  mapping
-  B - the right mapping


   Output Parameters:
.  C - the product mapping: domain as the range of  A, range as the range of B


   Level: intermediate
.seealso: SAMappingPullback(), SAMappingInvert()
@*/
PetscErrorCode  SAMappingPushforward(SAMapping A,SAMapping B, SAMapping *C) 
{
  PetscErrorCode ierr;
  PetscErrorCode (*pushforward)(SAMapping,SAMapping,SAMapping*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,SA_MAPPING_CLASSID,1);
  PetscValidType(A,1);
  if (!A->assembled) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled mapping");
  PetscValidHeaderSpecific(B,SA_MAPPING_CLASSID,2);
  PetscValidType(B,2);
  if (!B->assembled) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled mapping");
  PetscValidPointer(C,3);

  /* dispatch based on the type of A and B */
  char  pushforwardname[256];
  ierr = PetscStrcpy(pushforwardname,"SAMappingPushforward_");CHKERRQ(ierr);
  ierr = PetscStrcat(pushforwardname,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = PetscStrcat(pushforwardname,"_");    CHKERRQ(ierr);
  ierr = PetscStrcat(pushforwardname,((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = PetscStrcat(pushforwardname,"_C");   CHKERRQ(ierr); /* e.g., pushforwardname = "ISPushforward_ismappingis_ismappingis_C" */
  ierr = PetscObjectQueryFunction((PetscObject)B,pushforwardname,(void (**)(void))&pushforward);CHKERRQ(ierr);
  if (!pushforward) SETERRQ2(((PetscObject)A)->comm,PETSC_ERR_ARG_INCOMP,"SAMappingPushforward requires A, %s, to be compatible with B, %s",((PetscObject)A)->type_name,((PetscObject)B)->type_name);    
  ierr = PetscLogEventBegin(SA_MAPPING_Pushforward, A,B,0,0); CHKERRQ(ierr);
  ierr = (*pushforward)(A,B,C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SA_MAPPING_Pushforward, A,B,0,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef  __FUNCT__
#define __FUNCT__ "SAMappingSetType"
PetscErrorCode SAMappingSetType(SAMapping map, const SAMappingType maptype) {
  PetscErrorCode ierr;
  PetscErrorCode (*ctor)(SAMapping);
  PetscBool sametype;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,SA_MAPPING_CLASSID, 1);
  ierr = PetscTypeCompare((PetscObject)map,maptype, &sametype); CHKERRQ(ierr);
  if(sametype) PetscFunctionReturn(0);

  if(!SAMappingRegisterAllCalled) {
    ierr = SAMappingRegisterAll(PETSC_NULL); CHKERRQ(ierr);
  }
  ierr =  PetscFListFind(ISList,((PetscObject)map)->comm,maptype,(void(**)(void))&ctor);CHKERRQ(ierr);
  if(!ctor) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unrecognized SAMapping type: %s", maptype); 

  /* destroy the old implementation, if it existed */
  if(map->ops->destroy) {
    ierr = (*(map->ops->destroy))(map); CHKERRQ(ierr);
    map->ops->destroy = PETSC_NULL;
  }
  
  /* create the new implementation */
  ierr = (*ctor)(map); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* SAMappingSetType() */

#undef  __FUNCT__
#define __FUNCT__ "SAMappingDestroy"
PetscErrorCode SAMappingDestroy(SAMapping map) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,SA_MAPPING_CLASSID,1);
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
}/* SAMappingDestroy() */

#undef  __FUNCT__
#define __FUNCT__ "SAMappingCreate"
PetscErrorCode SAMappingCreate(MPI_Comm comm, SAMapping *_map) 
{
  SAMapping map;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(map,2);
  *_map = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = SAMappingInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(map,_p_SAMapping,struct _SAMappingOps,SA_MAPPING_CLASSID,0,"SAMapping",comm,SAMappingDestroy,SAMappingView); CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm,&(map->xlayout)); CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm,&(map->ylayout)); CHKERRQ(ierr);
  *_map = map;
  PetscFunctionReturn(0);
}/* SAMappingCreate() */



#undef  __FUNCT__
#define __FUNCT__ "SAHunkCreate"
PetscErrorCode SAHunkCreate(PetscInt maxlength, SAComponents mask, SAHunk *_hunk) 
{
  PetscErrorCode ierr;
  SAHunk hunk;
  PetscFunctionBegin;
  PetscValidPointer(_hunk,3);

  if(maxlength <= 0) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Nonpositive SAHunk maxlength: %D", maxlength);

  ierr = PetscNew(struct _n_SAHunk, &hunk); CHKERRQ(ierr);
  hunk->mask = mask;
  hunk->maxlength = maxlength;
  if(mask & SA_I) {
    PetscInt *ia;
    ierr = PetscMalloc(sizeof(PetscInt)*maxlength, &ia);  CHKERRQ(ierr);
    ierr = PetscMemzero(ia, sizeof(PetscInt)*maxlength);  CHKERRQ(ierr);
    hunk->i = ia;
  }
  if(mask & SA_J) {
    PetscInt *ja;
    ierr = PetscMalloc(sizeof(PetscInt)*maxlength, &ja);  CHKERRQ(ierr);
    ierr = PetscMemzero(ja, sizeof(PetscInt)*maxlength); CHKERRQ(ierr);
    hunk->j = ja;
  }
  if(mask & SA_W) {
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
#define __FUNCT__ "SAHunkAddData"
PetscErrorCode SAHunkAddData(SAHunk hunk, PetscInt length, const PetscInt *i, const PetscScalar *w, const PetscInt *j) 
{
  PetscErrorCode ierr;
  PetscInt mask;
  PetscFunctionBegin;
  PetscValidPointer(hunk,1);
  mask = (i != PETSC_NULL) | ((j != PETSC_NULL)<<1) | ((w != PETSC_NULL)<<2);
  if(mask != hunk->mask) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Data components %D incompatible with the SAHunk mask", mask,hunk->mask);
  if(hunk->length + length > hunk->maxlength) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot add data of length %D, hunk only has %D space left", length, hunk->maxlength-hunk->length);
  if(mask & SA_I) {
    ierr = PetscMemcpy(hunk->i+hunk->length, i, sizeof(PetscInt)*length);
  }
  if(mask & SA_J) {
    ierr = PetscMemcpy(hunk->j+hunk->length, j, sizeof(PetscInt)*length);
  }
  if(mask & SA_W) {
    ierr = PetscMemcpy(hunk->w+hunk->length, w, sizeof(PetscScalar)*length);
  }
  hunk->length += length;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SAHunkGetSubHunk"
PetscErrorCode SAHunkGetSubHunk(SAHunk hunk, PetscInt start, PetscInt maxlength, PetscInt length, SAComponents mask, SAHunk *_subhunk) 
{
  PetscErrorCode ierr;
  SAHunk subhunk;
  PetscFunctionBegin;
  PetscValidPointer(hunk,1);
  PetscValidPointer(_subhunk,5);
  *_subhunk = PETSC_NULL;
  if(maxlength <= 0) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Nonpositive subhunk maxlength: %D", maxlength);
  if(length <= 0) 
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Nonpositive subhunk length: %D", length);
  if(length > maxlength) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "subhunk length %D exceeds maxlength %D", length, maxlength);
  if(start < 0 || start >= hunk->maxlength) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "subhunk start %D out of bounds [0,%D)", start,hunk->maxlength);
  if(start+maxlength > hunk->maxlength) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "subhunk end %D beyond the end of hunk %D",start+maxlength,hunk->maxlength);
  
  if(mask & (~(hunk->mask))) 
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Subhunk mask %D is not a submask of the hunk mask %D", mask, hunk->mask);
  ierr = PetscNew(struct _n_SAHunk, &subhunk); CHKERRQ(ierr);
  subhunk->mask = mask;
  subhunk->maxlength = maxlength;
  subhunk->length    = length;
  if(mask & SA_I) {
    subhunk->i = hunk->i+start;
  }
  if(mask & SA_J) {
    subhunk->j = hunk->j+start;
  }
  if(mask & SA_W) {
    subhunk->w = hunk->w+start;
  }
  subhunk->mode = PETSC_USE_POINTER;
  ++(hunk->refcnt);
  *_subhunk = subhunk;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SAHunkDestroy"
PetscErrorCode SAHunkDestroy(SAHunk hunk) 
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
    ierr = SAHunkDestroy(hunk->parent); CHKERRQ(ierr);
  }
  hunk->parent = PETSC_NULL;
  ierr = PetscFree(hunk); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SACreate"
PetscErrorCode SACreate(SAComponents mask, SA *_arr) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_arr,2);
  if(!(mask & SA_I) && !(mask & SA_J)) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "SAComponents %D must contain at least one of the indices: I or J", mask);
  }
  ierr = PetscNew(struct _n_SA, _arr); CHKERRQ(ierr);
  (*_arr)->mask = mask;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SADuplicate"
PetscErrorCode SADuplicate(SA arr, SA *_darr) 
{
  PetscErrorCode ierr;
  SA darr;
  SALink link;
  PetscFunctionBegin;
  PetscValidPointer(arr,1);
  PetscValidPointer(_darr,2);
  ierr = SACreate(arr->mask, &darr); CHKERRQ(ierr);
  link  = arr->first;
  while(link) {
    ierr = SAAddHunk(darr,link->hunk);  CHKERRQ(ierr);
  }
  *_darr = arr;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SACreateArrays"
PetscErrorCode SACreateArrays(SAComponents mask, PetscInt count, SA **_arrays) 
{
  PetscErrorCode ierr;
  PetscInt i;
  PetscFunctionBegin;
  PetscValidPointer(_arrays,3);
  if(!(mask & SA_I) && !(mask & SA_J)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "SAComponents %D must contain at least one of the indices: I or J", mask);
  if(count < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Negative array count: %D", count);
  ierr = PetscMalloc(sizeof(SA), _arrays); CHKERRQ(ierr);
  for(i = 0; i < count; ++i) {
    ierr = SACreate(mask, *_arrays+i); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAClear"
PetscErrorCode SAClear(SA arr) 
{
  PetscErrorCode ierr;
  SALink    link;
  PetscFunctionBegin;
  PetscValidPointer(arr,1);
  link = arr->first;
  while(link) {
    ierr = SAHunkDestroy(link->hunk); CHKERRQ(ierr);
    link = link->next;
  }
  arr->first = PETSC_NULL;
  arr->last  = PETSC_NULL;
  arr->length = 0;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SADestroy"
PetscErrorCode SADestroy(SA chain) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  ierr = SAClear(chain); CHKERRQ(ierr);
  chain->mask   = 0;
  ierr = PetscFree(chain);    CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* 
 Right now we merely allocate a new hunk.
 In the future, this can manage a pool of hunks, use a buffer to draw subhunks from, etc.
 */
#undef  __FUNCT__
#define __FUNCT__ "SAGetHunk"
PetscErrorCode SAGetHunk(SA chain, PetscInt length, SAHunk *_hunk) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = SAHunkCreate(length, chain->mask, _hunk); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAAddHunk"
PetscErrorCode SAAddHunk(SA chain, SAHunk hunk) 
{
  PetscErrorCode ierr;
  SALink    link;
  PetscFunctionBegin;
  if(chain->mask & (~(hunk->mask))) {
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Hunk mask %D incompatible with the array mask %D", hunk->mask, chain->mask);
  }
  ierr = PetscMalloc(sizeof(struct _n_SALink), &link);
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
#define __FUNCT__ "SAAddData"
PetscErrorCode SAAddData(SA chain, PetscInt length, const PetscInt *i, const PetscScalar *w, const PetscInt *j) 
{
  PetscErrorCode ierr;
  SAHunk hunk;
  SAComponents mask;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  mask = (i != PETSC_NULL) | ((j != PETSC_NULL)<<1) | ((w != PETSC_NULL)<<2);
  if(mask != chain->mask) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Data components %D incompatible with the SAComponents", mask,chain->mask);
  ierr = SAGetHunk(chain, length, &hunk);  CHKERRQ(ierr);
  ierr = SAHunkAddData(hunk, length, i,w,j);       CHKERRQ(ierr);
  ierr = SAAddHunk(chain, hunk); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SAAddI"
PetscErrorCode SAAddI(SA chain, PetscInt length, PetscInt i, const PetscScalar wa[], const PetscInt ja[]) 
{
  PetscErrorCode ierr;
  SAHunk hunk;
  PetscInt mask;
  PetscInt k;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  mask = (SA_I | (ja != PETSC_NULL)<<1 | (wa != PETSC_NULL)<<2);
  if(mask & (~(chain->mask))) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Array components provided %D incompatible with array mask %D", mask, chain->mask);
  if(!length) PetscFunctionReturn(0);
  ierr = SAGetHunk(chain, length, &hunk);           CHKERRQ(ierr);
  for(k = 0; k < length; ++k) hunk->i[k] = i;
  if(ja) {
    ierr = PetscMemcpy(hunk->j, ja, sizeof(PetscInt)*length);    CHKERRQ(ierr);
  }
  if(wa) {
    ierr = PetscMemcpy(hunk->w, wa, sizeof(PetscScalar)*length); CHKERRQ(ierr);
  }
  ierr = SAAddHunk(chain, hunk);                            CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAAddJ"
PetscErrorCode SAAddJ(SA chain, PetscInt length, const PetscInt ia[], const PetscScalar wa[], PetscInt j) 
{
  PetscErrorCode ierr;
  SAHunk hunk;
  PetscInt mask;
  PetscInt k;
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  mask = ((ia != PETSC_NULL) | SA_J | (wa != PETSC_NULL)<<2);
  if(mask & (~(chain->mask))) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Array components provided %D incompatible with array mask %D", mask, chain->mask);
  if(!length) PetscFunctionReturn(0);
  ierr = SAGetHunk(chain, length, &hunk);           CHKERRQ(ierr);
  for(k = 0; k < length; ++k) hunk->j[k] = j;
  if(ia) {
    ierr = PetscMemcpy(hunk->i, ia, sizeof(PetscInt)*length);    CHKERRQ(ierr);
  }
  if(wa) {
    ierr = PetscMemcpy(hunk->w, wa, sizeof(PetscScalar)*length); CHKERRQ(ierr);
  }
  ierr = SAAddHunk(chain, hunk);                            CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAMerge"
PetscErrorCode SAMerge(SA arr, SA *_marr) 
{
  PetscErrorCode ierr;
  SALink link;
  SAHunk merged;
  PetscInt count, offset;
  PetscFunctionBegin;
  PetscValidPointer(arr,1);
  PetscValidPointer(_marr,2);

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
    ierr = SAHunkCreate(arr->length, arr->mask, &merged);                           CHKERRQ(ierr);
    /* Copy the indices and weights into the merged arrays. */
    offset = 0;
    link = arr->first;
    while(link) {
      SAHunk hunk = link->hunk;
      if(arr->mask & SA_I) {
        ierr = PetscMemcpy(merged->i+offset, hunk->i, sizeof(PetscInt)*hunk->length);    CHKERRQ(ierr);
      }
      if(arr->mask & SA_J) {
        ierr = PetscMemcpy(merged->j+offset, hunk->j, sizeof(PetscInt)*hunk->length);    CHKERRQ(ierr);
      }
      if(arr->mask & SA_W) {
        ierr = PetscMemcpy(merged->w+offset, hunk->w, sizeof(PetscScalar)*hunk->length); CHKERRQ(ierr);
      }
      offset += hunk->length;
    }
  }/* if(arr->length) */
  merged->length = offset;
  ierr = SACreate(arr->mask, _marr); CHKERRQ(ierr);
  ierr = SAAddHunk(*_marr, merged);  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAGetLength"
PetscErrorCode SAGetLength(SA chain, PetscInt *_length) 
{
  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  PetscValidPointer(_length,2);
  *_length = chain->length;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAGetData"
PetscErrorCode SAGetData(SA chain, PetscInt *ia, PetscScalar *wa, PetscInt *ja) 
{
  PetscErrorCode ierr;
  PetscInt len, mask, off;
  SALink link;
  SAHunk hunk;

  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  mask = ((ia != PETSC_NULL) & (chain->mask&SA_I)) | ((ja != PETSC_NULL)<<1 & (chain->mask&SA_J)) | ((wa != PETSC_NULL)<<2 & (chain->mask&SA_W));
  len = chain->length;
  off = 0;
  link = chain->first;
  while(link) {
    hunk = link->hunk;
    if(mask&SA_I) {
      ierr = PetscMemcpy(ia+off, hunk->i, sizeof(PetscInt)*hunk->length); CHKERRQ(ierr);
    }
    if(mask&SA_J) {
      ierr = PetscMemcpy(ja+off, hunk->j, sizeof(PetscInt)*hunk->length); CHKERRQ(ierr);
    }
    if(mask&SA_W) {
      ierr = PetscMemcpy(wa+off, hunk->w, sizeof(PetscScalar)*hunk->length); CHKERRQ(ierr);
    }
    link = link->next;
    off += hunk->length;
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SAAddArray"
PetscErrorCode SAAddArray(SA chain, SA chain2) 
{
  PetscErrorCode ierr;
  SALink link;
  PetscFunctionBegin;
  PetscValidPointer(chain, 1);
  PetscValidPointer(chain2,2);
  if(chain->mask & (~(chain2->mask))) {
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "New mask %D ncompatible with original %D", chain2->mask, chain->mask);
  }
  link = chain2->first;
  while(link) {
    ierr = SAAddHunk(chain, link->hunk); CHKERRQ(ierr);
    link = link->next;
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SASplit"
PetscErrorCode SASplit(SA arr, PetscInt count, const PetscInt *lengths, PetscInt mask, SA *arrs)
{
  PetscErrorCode ierr;
  PetscInt i, lengthi, start, end, len;
  SALink link;
  SAHunk hunk, subhunk;

  PetscFunctionBegin;
  PetscValidPointer(arr,1);
  if(count < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Negative array length %D", count);
  PetscValidPointer(lengths, 3);
  PetscValidPointer(arrs, 5);
  if(!count) PetscFunctionReturn(0);
  link = arr->first;
  i = 0;        /* current subarray index */
  lengthi = 0;  /* current length of the current subarray */
  while(link) {
    hunk = link->hunk;
    start = 0;  /* start of unassigned hunk space */
    while(hunk->length-start) {
      end = PetscMin(hunk->length, lengths[i]-lengthi);
      len = start-end;
      ierr = SAHunkGetSubHunk(hunk,start,len,len,hunk->mask, &subhunk); CHKERRQ(ierr);
      ierr = SAAddHunk(arrs[i], subhunk);                               CHKERRQ(ierr);
      start   += len;
      lengthi += len;
      if(lengthi == lengths[i]) {
        lengthi = 0; ++i;
        if(i >= count) PetscFunctionReturn(0);
      }
    }
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
      waits), this could be made into an SAMapping method.  However, then an SAMapping of some sort must be instantiated
      for SA assembly.  Maybe it always happens in the common use cases, anyhow. 
 */
#undef __FUNCT__  
#define __FUNCT__ "SAAssemble"
PetscErrorCode SAAssemble(SA chain, PetscInt mask, PetscLayout layout, SA achain) 
{
  PetscErrorCode ierr;
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
  SALink link;
  SAHunk hunk, ahunk;

  PetscFunctionBegin;
  PetscValidPointer(chain,1);
  PetscValidPointer(achain, 4);

  /* Make sure that at least one of the indices -- I or J -- is being assembled on, and that the index being assembled on is present in the SA. */
  if((mask != SA_I && mask != SA_J)|| !(mask & chain->mask)) {
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot assemble SA with components %D on component %D", chain->mask, mask);
  }
  if(chain->mask != achain->mask) {
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Input and output array masks differ: %D and %D", chain->mask, achain->mask);
  }

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
  ni += ((chain->mask & SA_I) > 0);
  ni += ((chain->mask & SA_J) > 0);

  /* How many value arrays are being sent?  One or none? */
  nv =  ((chain->mask & SA_W) > 0);


  
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
      if(mask == SA_I) {
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
    if(mask == SA_I) {
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
  /* Clear the SA that will store the received data segments */
  ierr = SAClear(achain);                      CHKERRQ(ierr);
  /* Get a hunk to pack the data into. */
  ierr = SAGetHunk(achain, alength, &ahunk);   CHKERRQ(ierr);
  
  /* Use ahunk's data arrays as receive buffers. */
  if(mask == SA_I) {
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
  PetscFunctionReturn(0);
}/* SAAssemble() */





