/*$Id: dascatter.c,v 1.19 2000/04/09 04:39:49 bsmith Exp bsmith $*/
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/dm/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DAGetScatter"
/*@C
   DAGetScatter - Gets the local-to-global, local-to-global, and 
   local-to-local vector scatter contexts for a distributed array.

   Not Collective, but VecScatter is parallel if DA is parallel

   Input Parameter:
.  da - the distributed array

   Output Parameters:
+  ltog - local-to-global scatter context (may be PETSC_NULL)
.  gtol - global-to-local scatter context (may be PETSC_NULL) 
-  ltol - local-to-local scatter context (may be PETSC_NULL)

   Level: developer

   Notes:
   The output contexts are valid only as long as the input da is valid.
   If you delete the da, the scatter contexts will become invalid.

.keywords: distributed array, get, scatter, context, global-to-local,
           local-to-global, local-to-local

.seealso: DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DALocalToGlobal()
@*/
int DAGetScatter(DA da,VecScatter *ltog,VecScatter *gtol,VecScatter *ltol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (ltog) *ltog = da->ltog;
  if (gtol) *gtol = da->gtol;
  if (ltol) *ltol = da->ltol;
  PetscFunctionReturn(0);
}
 
