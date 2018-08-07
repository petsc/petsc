#!/bin/bash
date
if [[ ${named_test} =~ .*icc.* || ${named_test} =~ .*ifort.* ]]; then
 if [[ ${slave_label} != "macos" ]]; then
  eval `/software/common/adm/packages/softenv-1.4.2/bin/soft-dec sh add +intel`
  export INTEL_LICENSE_FILE
  export PATH
  export LD_LIBRARY_PATH
 fi
fi
export PATH=$PATH:/usr/local/bin
if [ ${named_test} != "none" ]; then
  if [[ ${slave_label} == "macos" ]]; then
      if [[ ! ${named_test} =~ osx.* ]]; then
         echo "Configuration requested does not match architecture. ignoring"
         exit 0
      fi
   fi
   if [[ ${slave_label} == "ia32" || ${slave_label} == "mcs" ]]; then
      if [[ ! ${named_test} =~ .*linux.* ]]; then
        echo "Configuration requested does not match architecture. ignoring"
        exit 0
      fi
   fi
fi
export PETSC_DIR=${WORKSPACE}
make
exit $?


