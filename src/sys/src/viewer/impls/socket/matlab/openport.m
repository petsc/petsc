function port = openport(number)
% $Id: openport.m,v 1.6 2000/02/02 20:07:58 bsmith Exp bsmith $
%
%  port = openport(number)
%  Opens a port to receive matrices from Petsc.
% see closeport and receive
disp('You must build the openport mex file by doing: cd ${PETSC_DIR}/src/sys/src/viewer/impls/socket/matlab; make BOPT=g matlabcodes')
