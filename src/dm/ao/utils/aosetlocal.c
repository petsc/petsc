#define PETSCDM_DLL

#include "petscao.h"       /*I  "petscao.h"  I*/

#undef __FUNCT__
#define __FUNCT__ "AODataPartitionAndSetupLocal"
/*@C     
       AODataPartitionAndSetupLocal - Partitions across a given key (for example cells), then partitions a segment
         (for example vertices) subservient to that key.

    Collective on AOData

  Input Parameters:
+  ao           - the AO database
.  keyname      - the name of the key
-  segmentname  - the name of the segment 


  Output Parameters:
+  iskey         - the local indices in global numbering of the key entries (cells). Note that these are
                         contiguous from rstart to rend
.  issegment     - the local indices in global numbering of the segment entries (vertices)
-  ltog          - the local to global mapping for the segment entries (vertices)

  Level: advanced

  Notes: this renumbers the key and segment entries in the AO database to reflect the new partitioning.
  The ltog mapping is a mapping for the issegment indices, that is ltog applied to the indices
  0 to sizeof(issegment)-1 is the entries in issegment. 

.seealso: AODataKeyParition(), AODataSegmentPartition()
@*/
PetscErrorCode PETSCDM_DLLEXPORT AODataPartitionAndSetupLocal(AOData ao,const char keyname[],const char segmentname[],IS *iskey,IS *issegment,ISLocalToGlobalMapping *ltog)
{
  ISLocalToGlobalMapping ltogkey;
  PetscErrorCode         ierr;
  PetscInt               rstart,rend;
  MPI_Comm               comm;

  PetscFunctionBegin;  

  /*      Partition the keys (cells)   */
  ierr = AODataKeyPartition(ao,keyname);CHKERRQ(ierr);  

  /*      Partition the segment (vertices) subservient to the keys (cells)  */ 
  ierr = AODataSegmentPartition(ao,keyname,segmentname);CHKERRQ(ierr);  

 /*     Generate the list of key entries (cells) on this processor   */
  ierr = AODataKeyGetOwnershipRange(ao,"cell",&rstart,&rend);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ao,&comm);CHKERRQ(ierr);

  ierr = ISCreateStride(comm,rend-rstart,rstart,1,iskey);CHKERRQ(ierr);

 /*       Get the list of segment entries (vertices) used by these key entries (cells)   */
  ierr = AODataSegmentGetReducedIS(ao,keyname,segmentname,*iskey,issegment);CHKERRQ(ierr);

 /*     Make local to global mapping of key entries (cells)  */
  ierr = ISLocalToGlobalMappingCreateIS(*iskey,&ltogkey);CHKERRQ(ierr);

  /*       Make local to global mapping of segment entries (vertices)  */
  ierr = ISLocalToGlobalMappingCreateIS(*issegment,ltog);CHKERRQ(ierr);

  /*        Attach the local to global mappings to the database */
  ierr = AODataKeySetLocalToGlobalMapping(ao,keyname,ltogkey);CHKERRQ(ierr);
  ierr = AODataKeySetLocalToGlobalMapping(ao,segmentname,*ltog);CHKERRQ(ierr);

  /*      Dereference the ltogkey; we don't need a copy of it */
  ierr = PetscObjectDereference((PetscObject)ltogkey);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



