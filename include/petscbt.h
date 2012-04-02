
#if !defined(__PETSCBT_H)
#define __PETSCBT_H
PETSC_EXTERN_CXX_BEGIN

/*S
     PetscBT - PETSc bitarrays

     Level: advanced

     PetscBTCreate(m,&bt)       - creates a bit array with enough room to hold m values
     PetscBTDestroy(&bt)        - destroys the bit array
     PetscBTMemzero(m,bt)       - zeros the entire bit array (sets all values to false)
     PetscBTSet(bt,index)       - sets a particular entry as true
     PetscBTClear(bt,index)     - sets a particular entry as false
     PetscBTLookup(bt,index)    - returns the value 
     PetscBTLookupSet(bt,index) - returns the value and then sets it true
     PetscBTLength(m)           - returns number of bytes in array with m bits
     PetscBTView(m,bt,viewer)   - prints all the entries in a bit array

    We do not currently check error flags on PetscBTSet(), PetscBTClear(), PetscBTLookup(),
    PetcBTLookupSet(), PetscBTLength() cause error checking would cost hundreds more cycles then
    the operation.

S*/
typedef char* PetscBT;


PETSC_STATIC_INLINE PetscInt  PetscBTLength(PetscInt m) 
{
  return  ((m)/PETSC_BITS_PER_BYTE+1);
}

PETSC_STATIC_INLINE PetscErrorCode PetscBTMemzero(PetscInt m,PetscBT array)
{
  return PetscMemzero(array,sizeof(char)*((m)/PETSC_BITS_PER_BYTE+1));
}

PETSC_STATIC_INLINE PetscErrorCode PetscBTDestroy(PetscBT *array)
{
  return PetscFree(*array);
}

PETSC_STATIC_INLINE char PetscBTLookup(PetscBT array,PetscInt index) 
{
  char      BT_mask,BT_c;
  PetscInt  BT_idx;

 return  (BT_idx        = (index)/PETSC_BITS_PER_BYTE, 
          BT_c          = array[BT_idx], 
          BT_mask       = (char)1 << ((index)%PETSC_BITS_PER_BYTE), 
          BT_c & BT_mask);
}

PETSC_STATIC_INLINE PetscErrorCode PetscBTView(PetscInt m,const PetscBT bt,PetscViewer viewer)
{
  PetscInt       i;
  PetscErrorCode ierr; 

  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
  for (i=0; i<m; i++) { 
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%D %d\n",i,PetscBTLookup(bt,i));CHKERRQ(ierr);
  } 
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode PetscBTCreate(PetscInt m,PetscBT *array)
{
  return (PetscMalloc(((m)/PETSC_BITS_PER_BYTE+1)*sizeof(char),array) || PetscBTMemzero(m,*array));
}

PETSC_STATIC_INLINE char PetscBTLookupSet(PetscBT array,PetscInt index) 
{
  char      BT_mask,BT_c;
  PetscInt  BT_idx;

  return (BT_idx        = (index)/PETSC_BITS_PER_BYTE, 
          BT_c          = array[BT_idx], 
          BT_mask       = (char)1 << ((index)%PETSC_BITS_PER_BYTE), 
          array[BT_idx] = BT_c | BT_mask, 
          BT_c & BT_mask);
}

PETSC_STATIC_INLINE PetscErrorCode PetscBTSet(PetscBT array,PetscInt index)
{
  char      BT_mask,BT_c;
  PetscInt  BT_idx;

  BT_idx        = (index)/PETSC_BITS_PER_BYTE;
  BT_c          = array[BT_idx];
  BT_mask       = (char)1 << ((index)%PETSC_BITS_PER_BYTE);
  array[BT_idx] = BT_c | BT_mask;
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode PetscBTClear(PetscBT array,PetscInt index)
{
  char      BT_mask,BT_c;
  PetscInt  BT_idx;

  BT_idx        = (index)/PETSC_BITS_PER_BYTE;
  BT_c          = array[BT_idx];
  BT_mask       = (char)1 << ((index)%PETSC_BITS_PER_BYTE);
  array[BT_idx] = BT_c & (~BT_mask);
 return 0;
}


PETSC_EXTERN_CXX_END
#endif
