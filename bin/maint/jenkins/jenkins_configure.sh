#!/bin/bash
#git checkout ${branch}
export PATH=${PATH}:/usr/local/bin

echo slave_label=${slave_label}
echo branch=${branch}
echo options=${configure_options}
echo mailto=${mailto} 
echo named_test=${named_test}
function soft { eval `/software/common/adm/packages/softenv-1.4.2/bin/soft-dec sh $@`; }

if [[ ${named_test} =~ .*ifc.* ]]; then
 if [[ ${slave_label} != "macos" ]]; then
  eval `/software/common/adm/packages/softenv-1.4.2/bin/soft-dec sh add +intel`
 fi
fi

export PETSC_DIR=${WORKSPACE} 
if [ ${named_test} == "none" ]; then
   ./configure ${configure_options}
else
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

   ./config/examples/arch-${named_test}.py ${configure_options}
fi

if [ $? -ne 0 ]; then
  cat configure.log
  exit 1
fi
