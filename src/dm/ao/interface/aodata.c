#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aodata.c,v 1.1 1997/09/20 00:26:41 bsmith Exp bsmith $";
#endif
/*  
   Defines the abstract operations on AOData
*/
#include "src/ao/aoimpl.h"      /*I "ao.h" I*/

#undef __FUNC__  
#define __FUNC__ "AODataView" 
/*@
   AODataView - Displays an application ordering.

   Input Parameters:
.  aodata - the application ordering context
.  viewer - viewer used to display the set, for example VIEWER_STDOUT_SELF.

.keywords:application ordering

.seealso: ViewerFileOpenASCII()
@*/
int AODataView(AOData aodata, Viewer viewer)
{
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  return (*aodata->view)((PetscObject)aodata,viewer);
}

#undef __FUNC__  
#define __FUNC__ "AODataDestroy" 
/*@
   AODataDestroy - Destroys an application ordering set.

   Input Parameters:
.  aodata - the application ordering context

.keywords: destroy, application ordering

.seealso: AODataCreateBasic()
@*/
int AODataDestroy(AOData aodata)
{
  if (!aodata) return 0;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  if (--aodata->refct > 0) return 0;
  return (*aodata->destroy)((PetscObject)aodata);
}





