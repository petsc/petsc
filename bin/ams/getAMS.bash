#!/bin/bash
#
#  getAMS.bash [names or memoryname]
# 
#
if [ "${AMS_HOST}foo" == "foo" ]; then export AMS_HOST=localhost; fi
if [ "${AMS_PORT}foo" == "foo" ]; then export AMS_PORT=8080; fi
if [ $# == 1 ]; then
  if [ $1 == "names" ]; then export mem=""; else export mem=$1; fi;
  else export mem="*";
fi

curl --silent --show-error "${AMS_HOST}:${AMS_PORT}/ams/memory/${mem}"
