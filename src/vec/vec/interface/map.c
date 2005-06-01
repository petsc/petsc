#define PETSCVEC_DLL
/*
     Provides the interface functions for all map operations.
   These are the map functions the user calls.
*/
#include "private/vecimpl.h"    /*I "petscvec.h" I*/

/* Logging support */
PetscCookie MAP_COOKIE = 0;

#undef __FUNCT__  
#define __FUNCT__ "PetscMapSetTypeFromOptions_Private"
/*
  PetscMapSetTypeFromOptions_Private - Sets the type of map from user options. Defaults to a PETSc sequential map on one
  processor and a PETSc MPI map on more than one processor.

  Collective on PetscMap

  Input Parameter:
. map - The map

  Level: intermediate

.keywords: PetscMap, set, options, database, type
.seealso: PetscMapSetFromOptions(), PetscMapSetType()
*/
static PetscErrorCode PetscMapSetTypeFromOptions_Private(PetscMap map)
{
  PetscTruth     opt;
  const char     *defaultType;
  char           typeName[256];
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (map->type_name) {
    defaultType = map->type_name;
  } else {
    ierr = MPI_Comm_size(map->comm, &size);CHKERRQ(ierr);
    if (size > 1) {
      defaultType = MAP_MPI;
    } else {
      defaultType = MAP_MPI;
    }
  }

  if (!PetscMapRegisterAllCalled) {
    ierr = PetscMapRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsList("-map_type", "PetscMap type"," PetscMapSetType", PetscMapList, defaultType, typeName, 256, &opt);
  CHKERRQ(ierr);
  if (opt) {
    ierr = PetscMapSetType(map, typeName);CHKERRQ(ierr);
  } else {
    ierr = PetscMapSetType(map, defaultType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapSetFromOptions"
/*@C
  PetscMapSetFromOptions - Configures the map from the options database.

  Collective on PetscMap

  Input Parameter:
. map - The map

  Notes:  To see all options, run your program with the -help option, or consult the users manual.
          Must be called after PetscMapCreate() but before the map is used.

  Level: intermediate

  Concepts: maptors^setting options
  Concepts: maptors^setting type

.keywords: PetscMap, set, options, database
.seealso: PetscMapCreate(), PetscMapPrintHelp(), PetscMaphSetOptionsPrefix()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapSetFromOptions(PetscMap map)
{
  PetscTruth opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,MAP_COOKIE,1);

  ierr = PetscOptionsBegin(map->comm, map->prefix, "PetscMap options", "PetscMap");CHKERRQ(ierr);

  /* Handle generic maptor options */
  ierr = PetscOptionsHasName(PETSC_NULL, "-help", &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscMapPrintHelp(map);CHKERRQ(ierr);
  }

  /* Handle map type options */
  ierr = PetscMapSetTypeFromOptions_Private(map);CHKERRQ(ierr);

  /* Handle specific maptor options */
  if (map->ops->setfromoptions) {
    ierr = (*map->ops->setfromoptions)(map);CHKERRQ(ierr);
  }

  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapPrintHelp"
/*@
  PetscMapPrintHelp - Prints all options for the PetscMap.

  Input Parameter:
. map - The map

  Options Database Keys:
$  -help, -h

  Level: intermediate

.keywords: PetscMap, help
.seealso: PetscMapSetFromOptions()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapPrintHelp(PetscMap map)
{
  char p[64];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, MAP_COOKIE,1);

  ierr = PetscStrcpy(p, "-");CHKERRQ(ierr);
  if (map->prefix) {
    ierr = PetscStrcat(p, map->prefix);CHKERRQ(ierr);
  }

  (*PetscHelpPrintf)(map->comm, "PetscMap options ------------------------------------------------\n");
  (*PetscHelpPrintf)(map->comm,"   %smap_type <typename> : Sets the map type\n", p);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapDestroy"
/*@
   PetscMapDestroy - Destroys a map object.

   Not Collective

   Input Parameter:
.  m - the map object

   Level: developer

.seealso: PetscMapCreateMPI()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapDestroy(PetscMap map)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(map, MAP_COOKIE,1); 
  if (--map->refct > 0) PetscFunctionReturn(0);
  if (map->range) {
    ierr = PetscFree(map->range);CHKERRQ(ierr);
  }
  if (map->ops->destroy) {
    ierr = (*map->ops->destroy)(map);CHKERRQ(ierr);
  }
  ierr = PetscHeaderDestroy(map);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapSetOptionsPrefix"
/*@C
   PetscMapSetOptionsPrefix - Sets the prefix used for searching for all 
   PetscMap options in the database.

   Collective on PetscMap

   Input Parameter:
+  map - the PetscMap context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: PetscMap, set, options, prefix, database

.seealso: PetscMapSetFromOptions()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapSetOptionsPrefix(PetscMap map,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,MAP_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)map,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapAppendOptionsPrefix"
/*@C
   PetscMapAppendOptionsPrefix - Appends to the prefix used for searching for all 
   PetscMap options in the database.

   Collective on PetscMap

   Input Parameters:
+  map - the PetscMap context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: PetscMap, append, options, prefix, database

.seealso: PetscMapGetOptionsPrefix()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapAppendOptionsPrefix(PetscMap map,const char prefix[])
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,MAP_COOKIE,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)map,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetOptionsPrefix"
/*@C
   PetscMapGetOptionsPrefix - Sets the prefix used for searching for all 
   PetscMap options in the database.

   Not Collective

   Input Parameter:
.  map - the PetscMap context

   Output Parameter:
.  prefix - pointer to the prefix string used

   Notes: On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.keywords: PetscMap, get, options, prefix, database

.seealso: PetscMapAppendOptionsPrefix()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapGetOptionsPrefix(PetscMap map,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,MAP_COOKIE,1);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)map,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapSetUp"
/*@
   PetscMapSetUp - Sets up the internal map data structures for the later use.

   Collective on PetscMap

   Input Parameters:
.  map - the PetscMap context

   Notes:
   For basic use of the PetscMap classes the user need not explicitly call
   PetscMapSetUp(), since these actions will happen automatically.

   Level: advanced

.keywords: PetscMap, setup

.seealso: PetscMapCreate(), PetscMapDestroy()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapSetUp(PetscMap map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,MAP_COOKIE,1);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapView"
/*@C
   PetscMapView - Visualizes a map object.

   Collective on PetscMap

   Input Parameters:
+  map - the map
-  viewer - visualization context

  Notes:
  The available visualization contexts include
+    PETSC_VIEWER_STDOUT_SELF - standard output (default)
.    PETSC_VIEWER_STDOUT_WORLD - synchronized standard
        output where only the first processor opens
        the file.  All other processors send their 
        data to the first processor to print. 
-     PETSC_VIEWER_DRAW_WORLD - graphical display of nonzero structure

   Level: beginner

.seealso: PetscViewerSetFormat(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(), 
          PetscViewerSocketOpen(), PetscViewerBinaryOpen(), PetscMapLoad()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapView(PetscMap map,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscInt          size;
  PetscTruth        iascii;
  const char        *cstr;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(map,MAP_COOKIE,1);
  PetscValidType(map,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(map->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  PetscCheckSameComm(map,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);  
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (map->prefix) {
        ierr = PetscViewerASCIIPrintf(viewer,"PetscMap Object:(%s)\n",map->prefix);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"PetscMap Object:\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscMapGetType(map,&cstr);CHKERRQ(ierr);
      ierr = PetscMapGetSize(map,&size);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"type=%s, size=%D\n",cstr,size);CHKERRQ(ierr);
    }
  }
  if (!iascii) {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)viewer)->type_name);
  } else {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);  
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapSetLocalSize"
/*@C
  PetscMapSetLocalSize - Sets the number of elements associated with this processor.

  Not Collective

  Input Parameters:
+ m - the map object
- n - the local size

  Level: developer

.seealso: PetscMapSetSize(), PetscMapGetLocalRange(), PetscMapGetGlobalRange()
Concepts: PetscMap^local size
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapSetLocalSize(PetscMap m,PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAP_COOKIE,1); 
  m->n = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetLocalSize"
/*@
   PetscMapGetLocalSize - Gets the number of elements associated with this processor.

   Not Collective

   Input Parameter:
.  m - the map object

   Output Parameter:
.  n - the local size

   Level: developer

.seealso: PetscMapGetSize(), PetscMapGetLocalRange(), PetscMapGetGlobalRange()

   Concepts: PetscMap^local size

@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapGetLocalSize(PetscMap m,PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAP_COOKIE,1); 
  PetscValidIntPointer(n,2);
  *n = m->n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapSetSize"
/*@C
  PetscMapSetSize - Sets the total number of elements associated with this map.

  Not Collective

  Input Parameters:
+ m - the map object
- N - the global size

  Level: developer

.seealso: PetscMapSetLocalSize(), PetscMapGetLocalRange(), PetscMapGetGlobalRange()
 Concepts: PetscMap^size
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapSetSize(PetscMap m,PetscInt N)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAP_COOKIE,1); 
  m->N = N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetSize"
/*@
   PetscMapGetSize - Gets the total number of elements associated with this map.

   Not Collective

   Input Parameter:
.  m - the map object

   Output Parameter:
.  N - the global size

   Level: developer

.seealso: PetscMapGetLocalSize(), PetscMapGetLocalRange(), PetscMapGetGlobalRange()

   Concepts: PetscMap^size
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapGetSize(PetscMap m,PetscInt *N)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAP_COOKIE,1); 
  PetscValidIntPointer(N,2);
  *N = m->N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetLocalRange"
/*@
   PetscMapGetLocalRange - Gets the local ownership range for this procesor.

   Not Collective

   Input Parameter:
.  m - the map object

   Output Parameter:
+  rstart - the first local index, pass in PETSC_NULL if not interested 
-  rend   - the last local index + 1, pass in PETSC_NULL if not interested

   Level: developer

.seealso: PetscMapGetLocalSize(), PetscMapGetGlobalRange()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapGetLocalRange(PetscMap m,PetscInt *rstart,PetscInt *rend)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAP_COOKIE,1); 
  if (rstart)  PetscValidIntPointer(rstart,2);
  if (rend) PetscValidIntPointer(rend,3);
  if (rstart) *rstart = m->rstart;
  if (rend)   *rend   = m->rend;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetGlobalRange"
/*@C
   PetscMapGetGlobalRange - Gets the ownership ranges for all processors.

   Not Collective

   Input Parameter:
.  m - the map object

   Output Parameter:
.  range - array of size + 1 where size is the size of the communicator 
           associated with the map. range[rank], range[rank+1] is the 
           range for processor 

   Level: developer

.seealso: PetscMapGetSize(), PetscMapGetLocalRange()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapGetGlobalRange(PetscMap m,PetscInt *range[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAP_COOKIE,1); 
  PetscValidPointer(range,2);
  *range = m->range;
  PetscFunctionReturn(0);
}
