#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ao.c,v 1.24 1999/01/31 16:11:10 bsmith Exp bsmith $";
#endif
/*  
   Defines the abstract operations on AO (application orderings) 
*/
#include "src/dm/ao/aoimpl.h"      /*I "ao.h" I*/

#undef __FUNC__  
#define __FUNC__ "AOView" 
/*@
   AOView - Displays an application ordering.

   Collective on AO and Viewer

   Input Parameters:
+  ao - the application ordering context
-  viewer - viewer used for display

   Level: intermediate

   Note:
   The available visualization contexts include
+     VIEWER_STDOUT_SELF - standard output (default)
-     VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   The user can open an alternative visualization context with
   ViewerASCIIOpen() - output to a specified file.

.keywords: application ordering

.seealso: ViewerASCIIOpen()
@*/
int AOView(AO ao, Viewer viewer)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  ierr = (*ao->ops->view)(ao,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODestroy" 
/*@
   AODestroy - Destroys an application ordering set.

   Collective on AO

   Input Parameters:
.  ao - the application ordering context

   Level: beginner

.keywords: destroy, application ordering

.seealso: AOCreateBasic()
@*/
int AODestroy(AO ao)
{
  int ierr;

  PetscFunctionBegin;
  if (!ao) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  if (--ao->refct > 0) PetscFunctionReturn(0);
  ierr = (*ao->ops->destroy)(ao); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* ---------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "AOPetscToApplicationIS" 
/*@
   AOPetscToApplicationIS - Maps an index set in the PETSc ordering to 
   the application-defined ordering.

   Collective on AO and IS

   Input Parameters:
+  ao - the application ordering context
-  is - the index set

   Level: intermediate

   Notes:
   The index set cannot be of type stride or block
   
   Any integers in ia[] that are negative are left unchanged. This 
         allows one to convert, for example, neighbor lists that use negative
         entries to indicate nonexistent neighbors due to boundary conditions
         etc.

.keywords: application ordering, mapping

.seealso: AOCreateBasic(), AOView(),AOApplicationToPetsc(),
          AOApplicationToPetscIS(),AOPetscToApplication()
@*/
int AOPetscToApplicationIS(AO ao,IS is)
{
  int        n,*ia,ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  ierr = ISBlock(is,&flag);CHKERRQ(ierr);
  if (flag) SETERRQ(1,1,"Cannot translate block index sets");
  ierr = ISStride(is,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = ISStrideToGeneral(is);CHKERRQ(ierr);
  }

  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&ia); CHKERRQ(ierr);
  ierr = (*ao->ops->petsctoapplication)(ao,n,ia); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&ia); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AOApplicationToPetscIS" 
/*@
   AOApplicationToPetscIS - Maps an index set in the application-defined
   ordering to the PETSc ordering.

   Collective on AO and IS

   Input Parameters:
+  ao - the application ordering context
-  is - the index set

   Level: beginner

   Note:
   The index set cannot be of type stride or block
   
   Any integers in ia[] that are negative are left unchanged. This 
   allows one to convert, for example, neighbor lists that use negative
   entries to indicate nonexistent neighbors due to boundary conditions, etc.

.keywords: application ordering, mapping

.seealso: AOCreateBasic(), AOView(), AOPetscToApplication(),
          AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
int AOApplicationToPetscIS(AO ao,IS is)
{
  int        n,*ia,ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  ierr = ISBlock(is,&flag);CHKERRQ(ierr);
  if (flag) SETERRQ(1,1,"Cannot translate block index sets");
  ierr = ISStride(is,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = ISStrideToGeneral(is);CHKERRQ(ierr);
  }

  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&ia); CHKERRQ(ierr);
  ierr = (*ao->ops->applicationtopetsc)(ao,n,ia); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&ia); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AOPetscToApplication" 
/*@
   AOPetscToApplication - Maps a set of integers in the PETSc ordering to 
   the application-defined ordering.

   Collective on AO

   Input Parameters:
+  ao - the application ordering context
.  n - the number of integers
-  ia - the integers

   Level: beginner

   Note:
   Any integers in ia[] that are negative are left unchanged. This 
   allows one to convert, for example, neighbor lists that use negative
   entries to indicate nonexistent neighbors due to boundary conditions, etc.

.keywords: application ordering, mapping

.seealso: AOCreateBasic(), AOView(),AOApplicationToPetsc(),
          AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
int AOPetscToApplication(AO ao,int n,int *ia)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  ierr = (*ao->ops->petsctoapplication)(ao,n,ia);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AOApplicationToPetsc" 
/*@
   AOApplicationToPetsc - Maps a set of integers in the application-defined
   ordering to the PETSc ordering.

   Collective on AO

   Input Parameters:
+  ao - the application ordering context
.  n - the number of integers
-  ia - the integers

   Level: beginner

   Note:
   Any integers in ia[] that are negative are left unchanged. This 
   allows one to convert, for example, neighbor lists that use negative
   entries to indicate nonexistent neighbors due to boundary conditions, etc.

.keywords: application ordering, mapping

.seealso: AOCreateBasic(), AOView(), AOPetscToApplication(),
          AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
int AOApplicationToPetsc(AO ao,int n,int *ia)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  ierr = (*ao->ops->applicationtopetsc)(ao,n,ia); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

