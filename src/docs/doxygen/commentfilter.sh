#!/bin/bash

##### The following commands do the following:
# Replace leading '/*M' and the like with '/**'
# Same for trailing 'M*/'
# Replace .seealso: by \see    
# Replace 'Level: beginner' by '\addtogroup beginner \n Level: beginner'              (Note: [ \t] is more portable than \s)
# Replace 'Level: intermediate' by '\addtogroup intermediate \n Level: intermediate'  (Note: [ \t] is more portable than \s)
# Replace 'Level: advanced' by '\addtogroup advanced \n Level: advanced'              (Note: [ \t] is more portable than \s)
# Replace 'Level: developer' by '\addtogroup developer \n Level: developer'           (Note: [ \t] is more portable than \s)
# Replace '+ . . . -' enumeration list on left border for function parameters with @param
  sed -e 's/\/\*[EJMS@]/\/\*\*/g' $1 \
| sed -e 's/[EJMS@]\*\//\*\//g' \
| sed -e 's/\.seealso:/\\see/g' \
| sed -e 's/Level:[ \t][ \t]*beginner/\\ingroup misc-class-beginner\n Level: beginner/g' \
| sed -e 's/Level:[ \t][ \t]*intermediate/\\ingroup misc-class-intermediate\n Level: intermediate/g' \
| sed -e 's/Level:[ \t][ \t]*advanced/\\ingroup misc-class-advanced\n Level: advanced/g' \
| sed -e 's/Level:[ \t][ \t]*developer/\\ingroup misc-class-developer\n Level: developer/g' \
| sed -e 's/^+ /\\param /g' | sed -e 's/^\. /\\param /g' | sed -e 's/^- /\\param /g' 
