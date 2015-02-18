#!/bin/bash
export PETSC_DIR=${WORKSPACE}
function soft { eval `/software/common/adm/packages/softenv-1.4.2/bin/soft-dec sh $@`; }

if [[ ${named_test} =~ .*ifc.* ]]; then
  if [[ ${slave_label} != "macos" ]]; then
    soft add +intel
  fi
fi

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

make test
