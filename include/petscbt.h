/* $Id: bitarray.h,v 1.6 1997/10/01 22:43:45 bsmith Exp balay $ */

/*    
      BTSet - Expexts a charecter array -'array' as input, and 
      treats it as  an array of bits. It Checks if a given bit location
      ( specified by 'index') is marked, and later marks that location. 

      Input:
      array  - an array of char. Initially all bits are to be set to zero
               by using PetscMemzero().
      index  - specifies the index of the required bit in the bit array.
      
      Output:
      return val - 0 if the bit is not found,
                 - nonzero if found.
        
      Usage :
      BT_LOOKUP(char * array,  int index) ;

      Summary:
          The bit operations are equivalent to:
              1: retval = array[index];
              2: array[index] = 1;
              3: return retval;
*/
#if !defined(__BITARRAY_H)

#if !defined(BITSPERBYTE)
#define BITSPERBYTE 8
#endif

typedef char*  BT;

static char _mask, _BT_c;
static int  _BT_idx;

#define BTCreate(m,array) (array = (char *)PetscMalloc(((m)/BITSPERBYTE+1)*sizeof(char)),\
                           ( !array ) ? 1 : (BTMemzero(m,array),0) )

#define BTMemzero(m,array) PetscMemzero(array,(m)/BITSPERBYTE+1)

#define BTLookupSet(array, index)    (_BT_idx         = (index)/BITSPERBYTE, \
                                        _BT_c           = array[_BT_idx], \
                                        _mask           = (char)1 << ((index)%BITSPERBYTE), \
                                        array[_BT_idx]  = _BT_c | _mask, \
                                        _BT_c & _mask )

#define BTSet(array, index)    (_BT_idx         = (index)/BITSPERBYTE, \
                                 _BT_c           = array[_BT_idx], \
                                 _mask           = (char)1 << ((index)%BITSPERBYTE), \
                                 array[_BT_idx]  = _BT_c | _mask,0)


#define BTClear(array, index)  (_BT_idx         = (index)/BITSPERBYTE, \
                                 _BT_c           = array[_BT_idx], \
                                 _mask           = (char)1 << ((index)%BITSPERBYTE), \
                                 array[_BT_idx]  = _BT_c & (~_mask),0)

#define BTLookup(array, index) (_BT_idx         = (index)/BITSPERBYTE, \
                                 _BT_c           = array[_BT_idx], \
                                 _mask           = (char)1 << ((index)%BITSPERBYTE), \
                                 _BT_c & _mask )


#define BTDestroy(array) (PetscFree(array),0)

#endif
