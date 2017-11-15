#!/bin/bash

##### The following commands do the following:
# Replace leading '/*E' and the like with '/**'  (Note: /*M denotes some text/chapter for the manual and does not comment a function)
# Same for trailing 'E*/'
# Replace .seealso: by \see    
# Replace 'Level: beginner' by '\addtogroup beginner \n Level: beginner'              (Note: [ \t] is more portable than \s)
# Replace 'Level: intermediate' by '\addtogroup intermediate \n Level: intermediate'  (Note: [ \t] is more portable than \s)
# Replace 'Level: advanced' by '\addtogroup advanced \n Level: advanced'              (Note: [ \t] is more portable than \s)
# Replace 'Level: developer' by '\addtogroup developer \n Level: developer'           (Note: [ \t] is more portable than \s)
# Replace '+ . . . -' enumeration list on left border for function parameters with @param
  sed -e 's/\/\*[EJS@]C*/\/\*\*/g' $2 \
| sed -e 's/[EJS@]\*\//\*\//g' \
| sed -e 's/\.seealso:/\\see/g' \
| sed -e "s/Level:[ \t][ \t]*beginner/\\\\ingroup ${1}-class-beginner\n Level: beginner/g" \
| sed -e "s/Level:[ \t][ \t]*intermediate/\\\\ingroup ${1}-class-intermediate\n Level: intermediate/g" \
| sed -e "s/Level:[ \t][ \t]*advanced/\\\\ingroup ${1}-class-advanced\n Level: advanced/g" \
| sed -e "s/Level:[ \t][ \t]*developer/\\\\ingroup ${1}-class-developer\n Level: developer/g" \
| sed -e 's/^+ /\\param /g' | sed -e 's/^\. /\\param /g' | sed -e 's/^- /\\param /g' 

