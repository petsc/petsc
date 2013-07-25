#!/bin/bash
#
#  getSAWs.bash [names or memoryname]
# 
#
if [ "${SAWS_HOST}foo" == "foo" ]; then export SAWS_HOST=localhost; fi
if [ "${SAWS_PORT}foo" == "foo" ]; then export SAWS_PORT=8080; fi
if [ $# == 1 ]; then
  if [ $1 == "names" ]; then export mem=""; else export mem=$1; fi;
  else export mem="*";
fi

curl --silent --show-error "${SAWS_HOST}:${SAWS_PORT}/SAWs/directory/${mem}"
