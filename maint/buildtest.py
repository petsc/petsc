#!/usr/bin/env python
#######################################################################
# Usage:
#  ./maint/builddist.py --with-config=linux-gnu
#
#  Options:
#    --with-config - specify the config file to use for the build (from $PETSC_DIR/config/*.py)
#  Notes:
#    invoke in PETSC_DIR
#
#######################################################################
def getoption(str):
  # Given a string, look for --option=val in argv and return val.
  # Once the option is found, remove it from argv
  import sys
  for arg in sys.argv:
    option = '--' + str + '='
    if arg.startswith(option):
      sys.argv.remove(arg)
      return arg.replace(option,'',1)


#######################################################################

common_test_options = [
  '--with-shared=0'
  ]

#######################################################################
def main():
  import os
  import sys

  # check if script invoked in PETSC_DIR
  if not os.path.exists(os.path.join('include','petscversion.h')):
    print 'Error! ' +  sys.argv[0] + ' not invoked in PETSC_DIR.'
    sys.exit(3)
  # look for user specified --with-config option
  confmodule = getoption('with-config')
  if confmodule == None:
    print 'Error! config not specified. use for eg: --with-config=linux-gnu'
    sys.exit(3)
  # now load the specified config file
  sys.path.insert(0,'config')
  try:
    conftest = __import__(confmodule)
  except:
    print 'Error! module ' + confmodule + ' not found!'
    sys.exit(3)

  #### need code ####
  # if conftest.configure_options or test_options do not exist flag error

  # Now construct a bunch of configure tests and run them
  # Using a simple algorithm now.
  # - run with configure_options
  # - for each test_options & common_test_options - keep adding to the current set and run
  # - construct a different PETSC_ARCH for each test
    
  import configure
  test_no = 1
  for opt in [''] + conftest.test_options + common_test_options:
    if opt != '' : conftest.configure_options.append(opt)
    conftest.configure_options.append('-PETSC_ARCH=' + confmodule + '-test-' + `test_no`)
    print 'Configuring with: ', conftest.configure_options
    configure.petsc_configure(conftest.configure_options)
    conftest.configure_options.pop()
    test_no += 1

#######################################################################
if __name__ ==  '__main__':
  main()
        
