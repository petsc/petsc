
/*
    We define the memory operations here. The reason we just do not use
  the standard memory routines in the PETSc code is that on some machines
  they are broken.

*/
#include <petsc/private/petscimpl.h>        /*I  "petscsys.h"   I*/
#include <petscbt.h>
#include <../src/sys/utils/ftn-kernels/fcopy.h>

/*@
   PetscMemcmp - Compares two byte streams in memory.

   Not Collective

   Input Parameters:
+  str1 - Pointer to the first byte stream
.  str2 - Pointer to the second byte stream
-  len  - The length of the byte stream
         (both str1 and str2 are assumed to be of length len)

   Output Parameters:
.   e - PETSC_TRUE if equal else PETSC_FALSE.

   Level: intermediate

   Note:
   PetscArraycmp() is preferred
   This routine is anologous to memcmp()

.seealso: `PetscMemcpy()`, `PetscMemcmp()`, `PetscArrayzero()`, `PetscMemzero()`, `PetscArraycmp()`, `PetscArraycpy()`, `PetscStrallocpy()`,
          `PetscArraymove()`
@*/
PetscErrorCode PetscMemcmp(const void *str1, const void *str2, size_t len, PetscBool *e)
{
  if (!len) {
    // if e is a bad ptr I guess we just die here then?
    *e = PETSC_TRUE;
    return 0;
  }

  PetscFunctionBegin;
  PetscValidPointer(str1,1);
  PetscValidPointer(str2,2);
  PetscValidBoolPointer(e,4);
  *e = memcmp((char*)str1,(char*)str2,len) ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HWLOC)
#include <petsc/private/petscimpl.h>
#include <hwloc.h>

/*@C
     PetscProcessPlacementView - display the MPI process placement by core

  Input Parameter:
.   viewer - ASCII viewer to display the results on

  Level: intermediate

  Notes:
    Requires that PETSc be installed with hwloc, for example using --download-hwloc
@*/
PetscErrorCode PetscProcessPlacementView(PetscViewer viewer)
{
  PetscBool        isascii;
  PetscMPIInt      rank;
  hwloc_bitmap_t   set;
  hwloc_topology_t topology;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  PetscCheck(isascii,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Only ASCII viewer is supported");

  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD,&rank));
  hwloc_topology_init ( &topology);
  hwloc_topology_load ( topology);
  set = hwloc_bitmap_alloc();

  PetscStackCallStandard(hwloc_get_proc_cpubind,topology, getpid(), set, HWLOC_CPUBIND_PROCESS);
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"MPI rank %d Process id: %d coreid %d\n",rank,getpid(),hwloc_bitmap_first(set)));
  PetscCall(PetscViewerFlush(viewer));
  hwloc_bitmap_free(set);
  hwloc_topology_destroy(topology);
  PetscFunctionReturn(0);
}
#endif
