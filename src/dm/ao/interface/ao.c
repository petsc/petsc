#ifndef lint
static char vcid[] = "$Id: ao.c,v 1.1 1996/06/25 19:20:07 bsmith Exp bsmith $";
#endif
/*  
   Defines the abstract operations on AO (application orderings) 
*/
#include "aoimpl.h"      /*I "ao.h" I*/


/*@
   AOView - Displays an application ordering.

   Input Parameters:
.  ao - the application ordering context
.  viewer - viewer used to display the set, for example STDOUT_VIEWER_SELF.

.keywords:application ordering

.seealso: ViewerFileOpenASCII()
@*/
int AOView(AO ao, Viewer viewer)
{
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  return (*ao->view)((PetscObject)ao,viewer);
}

/*@
   AODestroy - Destroys an application ordering set.

   Input Parameters:
.  ao - the application ordering context

.keywords: destroy, application ordering

.seealso: AOCreateDebug(), AOCreateBasic()
@*/
int AODestroy(AO ao)
{
  if (!ao) return 0;
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  return (*ao->destroy)((PetscObject)ao);
}


/* ---------------------------------------------------------------------*/
/*@
   AOPetscToApplicationIS - Maps an index set in the PETSc ordering to 
     the application defined ordering.

   Input Parameters:
.  ao - the application ordering context
.  is - the index set

.keywords: application ordering, mapping

.seealso: AOCreateDebug(), AOCreateBasic(), AOView(),AOApplicationToPetsc(),
          AOApplicationToPetscIS(),AOPetscToApplication()
@*/
int AOPetscToApplicationIS(AO ao,IS is)
{
  int n,*ia,ierr;
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&ia); CHKERRQ(ierr);
  ierr = (*ao->ops.petsctoapplication)(ao,n,ia); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&ia); CHKERRQ(ierr);
  return 0;
}

/*@
   AOApplicationToPetscIS - Maps an index set in the application ordering to 
     the PETSc ordering.

   Input Parameters:
.  ao - the application ordering context
.  is - the index set

.keywords: application ordering, mapping

.seealso: AOCreateDebug(), AOCreateBasic(), AOView(), AOPetscToApplication(),
          AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
int AOApplicationToPetscIS(AO ao,IS is)
{
  int n,*ia,ierr;
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&ia); CHKERRQ(ierr);
  ierr = (*ao->ops.applicationtopetsc)(ao,n,ia); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&ia); CHKERRQ(ierr);
  return 0;
}

/*@
   AOPetscToApplication - Maps a set of integers in the PETSc ordering to 
     the application defined ordering.

   Input Parameters:
.  ao - the application ordering context
.  n - the number of integers
.  ia - the integers

.keywords: application ordering, mapping

.seealso: AOCreateDebug(), AOCreateBasic(), AOView(),AOApplicationToPetsc(),
          AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
int AOPetscToApplication(AO ao,int n,int *ia)
{
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  return (*ao->ops.petsctoapplication)(ao,n,ia);
}

/*@
   AOApplicationToPetsc - Maps a set of integers in the application ordering
                          to the PETSc ordering.

   Input Parameters:
.  ao - the application ordering context
.  n - the number of integers
.  ia - the integers

.keywords: application ordering, mapping

.seealso: AOCreateDebug(), AOCreateBasic(), AOView(), AOPetscToApplication(),
          AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
int AOApplicationToPetsc(AO ao,int n,int *ia)
{
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  return (*ao->ops.applicationtopetsc)(ao,n,ia);
}





