/*$Id: dagtona.c,v 1.10 2001/03/23 23:25:00 balay Exp $*/
 
/*
     Tools to help solve the coarse grid problem redundantly.
  Provides two scatter contexts that (1) map from the usual global vector
  to all processors the entire vector in NATURAL numbering and (2)
  from the entire vector on each processor in natural numbering extracts
  out this processors piece in GLOBAL numbering
*/

#include "src/dm/da/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DAGlobalToNaturalAllCreate"
/*
   DAGlobalToNaturalAllCreate - Creates a scatter context that maps from the 
     global vector the entire vector to each processor in natural numbering

   Collective on DA

   Input Parameter:
.  da - the distributed array context

   Output Parameter:
.  scatter - the scatter context

   Level: advanced

.keywords: distributed array, global to local, begin, coarse problem

.seealso: DAGlobalToNaturalEnd(), DALocalToGlobal(), DACreate2d(), 
          DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DACreateNaturalVector()
*/
int DAGlobalToNaturalAllCreate(DA da,VecScatter *scatter)
{
  int ierr,m;
  IS  from,to;
  Vec tmplocal;
  AO  ao;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = DAGetAO(da,&ao);CHKERRQ(ierr);

  /* create the scatter context */
  ierr = VecGetSize(da->global,&m);CHKERRQ(ierr);
  ierr = ISCreateStride(da->comm,m,0,1,&to);CHKERRQ(ierr);
  ierr = AOPetscToApplicationIS(ao,to);CHKERRQ(ierr);
  ierr = ISCreateStride(da->comm,m,0,1,&from);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,m,&tmplocal);CHKERRQ(ierr);
  ierr = VecScatterCreate(da->global,from,tmplocal,to,scatter);CHKERRQ(ierr);
  ierr = VecDestroy(tmplocal);CHKERRQ(ierr);  
  ierr = ISDestroy(from);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DANaturalAllToGlobalCreate"
/*
   DANaturalAllToGlobalCreate - Creates a scatter context that maps from a copy
     of the entire vector on each processor to its local part in the global vector.

   Collective on DA

   Input Parameter:
.  da - the distributed array context

   Output Parameter:
.  scatter - the scatter context

   Level: advanced

.keywords: distributed array, global to local, begin, coarse problem

.seealso: DAGlobalToNaturalEnd(), DALocalToGlobal(), DACreate2d(), 
          DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DACreateNaturalVector()
*/
int DANaturalAllToGlobalCreate(DA da,VecScatter *scatter)
{
  int ierr,M,m,start;
  IS  from,to;
  Vec tmplocal;
  AO  ao;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = DAGetAO(da,&ao);CHKERRQ(ierr);

  /* create the scatter context */
  ierr = VecGetSize(da->global,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(da->global,&m);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(da->global,&start,PETSC_NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(da->comm,m,start,1,&from);CHKERRQ(ierr);
  ierr = AOPetscToApplicationIS(ao,from);CHKERRQ(ierr);
  ierr = ISCreateStride(da->comm,m,start,1,&to);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,M,&tmplocal);CHKERRQ(ierr);
  ierr = VecScatterCreate(tmplocal,from,da->global,to,scatter);CHKERRQ(ierr);
  ierr = VecDestroy(tmplocal);CHKERRQ(ierr);  
  ierr = ISDestroy(from);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

