/* $Id: bitarray.h,v 1.14 1999/11/05 14:48:27 bsmith Exp bsmith $ */

/*    

          BT - Bit array objects: used to compactly store logical arrays of variables.

     PetscBTCreate(m,bt)        - creates a bit array with enough room to hold m values
     PetscBTDestroy(bt)         - destroys the bit array
     PetscBTMemzero(m,bt)       - zeros the entire bit array (sets all values to false)
     PetscBTSet(bt,index)       - sets a particular entry as true
     PetscBTClear(bt,index)     - sets a particular entry as false
     PetscBTLookup(bt,index)    - returns the value 
     PetscBTLookupSet(bt,index) - returns the value and then sets it true
     PetscBTLength(m)           - returns number of bytes in array with m bits
     PetscBTView(m,bt,viewer)   - prints all the entries in a bit array

    These routines do not currently have manual pages.

    The are all implemented as macros with the trivial data structure for efficiency.

    These are not thread safe since they use a few global variables.

    We do not currently check error flags on PetscBTSet(), PetscBTClear(), PetscBTLookup(),
    PetcBTLookupSet(), PetscBTLength() cause error checking would cost hundreds more cycles then
    the operation.

*/
#if !defined(__BITARRAY_H)
#define __BITARRAY_H

#if !defined(BITSPERBYTE)
#define BITSPERBYTE 8
#endif

typedef char* PetscBT;

extern char _BT_mask,_BT_c;
extern int  _BT_idx;

#define PetscBTView(m,bt,viewer) {\
  int    __i,__ierr; \
  Viewer __viewer = viewer; \
  if (!__viewer) __viewer = VIEWER_STDOUT_SELF;\
  for (__i=0; __i<m; __i++) { \
    __ierr = PetscPrintf(((PetscObject)__viewer)->comm,"%d %d\n",__i,PetscBTLookup(bt,__i));CHKERRQ(__ierr);\
  }}

#define PetscBTLength(m)        ((m)/BITSPERBYTE+1)*sizeof(char)

#define PetscBTCreate(m,array)  (array = (char *)PetscMalloc(((m)/BITSPERBYTE+1)*sizeof(char)),\
                           (!array) ? 1 : PetscBTMemzero(m,array))

#define PetscBTMemzero(m,array) PetscMemzero(array,(m)/BITSPERBYTE+1)

#define PetscBTLookupSet(array,index)    (_BT_idx           = (index)/BITSPERBYTE, \
                                        _BT_c           = array[_BT_idx], \
                                        _BT_mask        = (char)1 << ((index)%BITSPERBYTE), \
                                        array[_BT_idx]  = _BT_c | _BT_mask, \
                                        _BT_c & _BT_mask)

#define PetscBTSet(array,index)    (_BT_idx          = (index)/BITSPERBYTE, \
                                 _BT_c           = array[_BT_idx], \
                                 _BT_mask        = (char)1 << ((index)%BITSPERBYTE), \
                                 array[_BT_idx]  = _BT_c | _BT_mask,0)


#define PetscBTClear(array,index)  (_BT_idx          = (index)/BITSPERBYTE, \
                                 _BT_c           = array[_BT_idx], \
                                 _BT_mask        = (char)1 << ((index)%BITSPERBYTE), \
                                 array[_BT_idx]  = _BT_c & (~_BT_mask),0)

#define PetscBTLookup(array,index) (_BT_idx          = (index)/BITSPERBYTE, \
                                 _BT_c           = array[_BT_idx], \
                                 _BT_mask        = (char)1 << ((index)%BITSPERBYTE), \
                                 (_BT_c & _BT_mask) != 0)

#define PetscBTDestroy(array) PetscFree(array)

#endif


