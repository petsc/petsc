function closeport(port)
%
%   closeport(port)
%   Closes a PETSc port opened with openport()
%see openport, receive
disp('You must build the closeport mex file by doing: cd ${PETSC_DIR}/src/sys/src/viewer/impls/socket/matlab; make BOPT=g matlabcodes')
