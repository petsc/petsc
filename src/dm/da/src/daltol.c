#ifndef lint
static char vcid[] = "$Id: da2.c,v 1.32 1996/01/27 04:56:41 bsmith Exp $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "daimpl.h"    /*I   "da.h"   I*/

/*@
   DALocalToLocalBegin - Maps from a local representation (including 
       ghostpoints) to another where the ghostpoints in the second are
       set correctly. Must be followed by DALocalToLocalEnd().

   Input Parameters:
.  da - the distributed array context
.  g - the original vector
.  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the vector with correct ghost values

.keywords: distributed array, global to local, begin

.seealso: DALocalToLocalEnd(), DALocalToGlobal(), DACreate2d()
@*/
int DALocalToLocalBegin(DA da,Vec g, InsertMode mode,Vec l)
{
  int ierr;
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  ierr = VecScatterBegin(g,l,mode,SCATTER_ALL,da->ltol); CHKERRQ(ierr);
  return 0;
}

/*@
   DALocalToLocalEnd - Maps from a local representation (including 
       ghostpoints) to another where the ghostpoints in the second are
       set correctly. Must be preceeded by DALocalToLocalBegin().

   Input Parameters:
.  da - the distributed array context
.  g - the original vector
.  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the vector with correct ghost values

.keywords: distributed array, global to local, end

.seealso: DALocalToLocalBegin(), DALocalToGlobal(), DAGlobalToLocal() DACreate2d()
@*/
int DALocalToLocalEnd(DA da,Vec g, InsertMode mode,Vec l)
{
  int ierr;
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  ierr = VecScatterEnd(g,l,mode,SCATTER_ALL,da->ltol); CHKERRQ(ierr);
  return 0;
}

