#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aoreduced.c,v 1.16 1999/09/27 21:32:22 bsmith Exp bsmith $";
#endif

#include "src/dm/ao/aoimpl.h"     /*I   "ao.h"  I*/
#include "sys.h"
#include "bitarray.h"

#undef __FUNC__  
#define __FUNC__ "AODataSegmentGetReduced_Basic"
int AODataSegmentGetReduced_Basic(AOData ao,char *name,char *segname,int n,int *keys,IS *is)
{
  AODataSegment *segment; 
  AODataKey     *key;
  int           ierr,dsize,i,bs,flag,*found,count,imin,imax,*out;
  char          *idata, *odata;
  BTPetsc       mask;

  PetscFunctionBegin;
  /* find the correct segment */
  ierr = AODataSegmentFind_Private(ao,name,segname,&flag,&key,&segment);CHKERRQ(ierr);
  if (flag != 1) SETERRQ(PETSC_ERR_ARG_WRONG,1,"Cannot locate segment");

  if (segment->datatype != PETSC_INT) SETERRQ(PETSC_ERR_ARG_WRONG,1,"Only for PETSC_INT data");

  /*
     Copy the found values into a contiguous location, keeping them in the 
     order of the requested keys
  */
  ierr  = PetscDataTypeGetSize(segment->datatype,&dsize);CHKERRQ(ierr);
  bs    = segment->bs;
  odata = (char *) PetscMalloc((n+1)*bs*dsize);CHKPTRQ(odata);
  idata = (char *) segment->data;
  for ( i=0; i<n; i++ ) {
    ierr = PetscMemcpy(odata + i*bs*dsize,idata + keys[i]*bs*dsize,bs*dsize);CHKERRQ(ierr);
  }

  found = (int *) odata;
  n     = n*bs;

  /*  Determine the max and min values */
  if (n) {
    imin = PETSC_MAX_INT;
    imax = 0;  
    for ( i=0; i<n; i++ ) {
      if (found[i] < 0) continue;
      imin = PetscMin(imin,found[i]);
      imax = PetscMax(imax,found[i]);
    }
  } else {
    imin = imax = 0;
  }
  ierr = PetscBTCreate(imax-imin,mask);CHKERRQ(ierr);
  /* Put the values into the mask and count them */
  count = 0;
  for ( i=0; i<n; i++ ) {
    if (found[i] < 0) continue;
    if (!PetscBTLoopupSet(mask,found[i] - imin)) count++;
  }
  ierr = PetscBTMemzero(imax-imin,mask);CHKERRQ(ierr);
  out = (int *) PetscMalloc((count+1)*sizeof(int));CHKPTRQ(out);
  count = 0;
  for ( i=0; i<n; i++ ) {
    if (found[i] < 0) continue;
    if (!PetscBTLoopupSet(mask,found[i] - imin)) {out[count++] = found[i];}
  }
  ierr = PetscBTDestroy(mask);CHKERRQ(ierr);
  ierr = PetscFree(found);CHKERRQ(ierr);

  ierr = ISCreateGeneral(ao->comm,count,out,is);CHKERRQ(ierr);
  ierr = PetscFree(out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}






