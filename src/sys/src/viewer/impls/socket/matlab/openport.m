function port = openport(number)
%
%  port = openport(number)
%  Opens a port to receive matrices from Petsc.
% see closeport and receive
disp('You must build the openport mex file by doing: cd ${PETSC_DIR}/src/sys/src/viewer/impls/socket/matlab; make BOPT=g matlabcodes')
