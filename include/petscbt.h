/* $Id: bitarray.h,v 1.3 1996/02/01 15:36:30 balay Exp balay $ */

/*    
      BT_LOOKUP - Expexts a charecter array -'array' as input, and 
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
          The bit operations are euivalent to:
              1: retval = array[idx];
              2: array[index] = 1;
              3: return retval;
*/
#include <values.h>
static char _mask, _BT_c;
static int  _BT_idx;
#define BT_LOOKUP( array,  index) (_BT_idx         = index/BITSPERBYTE, \
                                   _BT_c           = array[_BT_idx], \
                                   _BT_idx         = index/BITSPERBYTE, \
                                   _mask           = (char)1 << (index%BITSPERBYTE), \
                                   array[_BT_idx]  = _BT_c|_mask, \
                                   _BT_c & _mask )


