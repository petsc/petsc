#!/usr/bin/env python1.5
#!/bin/env python1.5
# $Id: parseargs.py,v 1.1 1999/11/12 22:30:17 balay Exp $ 
#
#  Parses the argument list, and searches for the specified argument.
#  It returns the argument list minus the parsed argument.
# 
#  Calling sequence: 
#      ret_flag,ret_val = parseargs(search_arg,nvals,arg_list)
#  Input:
#     search_arg - argument to search for
#     nvals      - number of values associated with this argument.
#                  currently supports nval = 0, 1
#      arg_list  - the list of arguments to parse
#  Output:
#      ret_flag  - 0 => the search_arg is not present in arg_list
#                  1 => search_arg is present in the arg_list
#      ret_val   - None if nvals = 0
#                  Value corresponding to search_arg, if nvals = 1
#      arg_list  - initialial arg_list minus the search_arg and ret_val
#

def parseargs(search_arg,return_nargs,arg_list):
    import string
    import sys
    try:
        index = arg_list.index(search_arg)
    except:
        # Argument not found in list, hence return flag = 0,return val = None
        return 0,None
    
    if return_nargs == 0:
        arg_list.remove(search_arg)
        return 1,None
    
    if index+1 == len(arg_list):
        print 'Error! Option has no value!'
        print 'Expecting value with option: ' + search_arg
        sys.exit()
    else:
        ret_arg = arg_list[index+1]
        arg_list.remove(search_arg)
        arg_list.remove(ret_arg)
        return 1,ret_arg


