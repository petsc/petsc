function A = receive(port)
% $Id: receive.m,v 1.6 2000/02/02 20:07:58 bsmith Exp $
%
%   A = receive(port)
%   Receives a matrix from a port opened with openport()
%see openport and closeport
disp('You must build the receive mex file by doing: cd ${PETSC_DIR}; make BOPT=g matlabcodes')