function error = launch(program,np)
% $Id: launch.m,v 1.4 2000/02/02 20:07:58 bsmith Exp bsmith $
%
%  error = launch(program,np)
%  Starts up PETSc program
% see openprot, closeport and receive
disp('You must build the launch mex file by doing: cd ${PETSC_DIR}/src/sys/src/viewer/impls/socket/matlab; make BOPT=g matlabcodes')
