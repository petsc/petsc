#!/bin/bash
#
#  Returns the SAWS published PETSc stack one function per line
#
${PETSC_DIR}/bin/saws/getSAWs.bash Stack | jsawk 'return this.directories' | jsawk 'if (this.name != "Stack") return null' | jsawk 'return this.variable' | jsawk 'return this[0]' | jsawk 'if (this.name != "functions") return null' | jsawk -n 'out(this.data.join("\n"))'

#  Notes:
#     jsawk applies to each entry in an array
#     jsawk 'if (this.name != "Stack") return null'  removes entries that do not satisfy some criteria
#     jsawk 'return this[0]'  takes the array of an array [[xxx]] and just returns the array [xxx]

