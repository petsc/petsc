/*$Id: aopart.c,v 1.14 2000/05/05 22:19:20 balay Exp bsmith $*/

#include "petscao.h"       /*I  "petscao.h"  I*/

#undef __FUNC__
#define __FUNC__ "AODataKeyPartition"
/*@C
    AODataKeyPartition - Partitions a key across the processors to reduce
    communication costs.

    Collective on AOData

    Input Parameters:
+   aodata - the database
-   key - the key you wish partitioned and renumbered

   Level: advanced

.seealso: AODataSegmentPartition()
@*/
int AODataKeyPartition(AOData aodata,char *key)
{
  AO              ao;
  Mat             adj;
  MatPartitioning part;
  IS              is,isg;
  int             ierr;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = PetscObjectGetComm((PetscObject)aodata,&comm);CHKERRQ(ierr);

  ierr = AODataKeyGetAdjacency(aodata,key,&adj);CHKERRA(ierr);
  ierr = MatPartitioningCreate(comm,&part);CHKERRA(ierr);
  ierr = MatPartitioningSetAdjacency(part,adj);CHKERRA(ierr);
  ierr = MatPartitioningSetFromOptions(part);CHKERRA(ierr);
  ierr = MatPartitioningApply(part,&is);CHKERRA(ierr);
  ierr = MatPartitioningDestroy(part);CHKERRA(ierr);
  ierr = MatDestroy(adj);CHKERRQ(ierr);
  ierr = ISPartitioningToNumbering(is,&isg);CHKERRA(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);

  ierr = AOCreateBasicIS(isg,PETSC_NULL,&ao);CHKERRA(ierr);
  ierr = ISDestroy(isg);CHKERRA(ierr);

  ierr = AODataKeyRemap(aodata,key,ao);CHKERRA(ierr);
  ierr = AODestroy(ao);CHKERRA(ierr);
  PetscFunctionReturn(0);
}
