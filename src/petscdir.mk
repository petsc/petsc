#
#   Allows recursively including $PETSC_DIR/petscdir.mk from a any subdirectory
#
-include $(abspath $(dir $(lastword $(MAKEFILE_LIST))))/../petscdir.mk
