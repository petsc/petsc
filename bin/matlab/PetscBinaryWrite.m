function PetscBinaryWrite(inarg,varargin)
%
%  Writes in PETSc binary file sparse matrices and vectors
%  if the array is multidimensional and dense it is saved
%  as a one dimensional array
%
%  Only works for square sparse matrices 
%
%   PetscBinaryWrite(inarg,args to write)
%   inarg may be:
%      filename 
%      socket number (0 for PETSc default)
%      the object returned from PetscOpenSocket or PetscOpenFile
%
if ischar(inarg) 
  fd = PetscOpenFile(inarg,'w');
else if isnumeric(inarg)
  if inarg == 0
    fd = PetscOpenSocket;
  else 
    fd = PetscOpenSocket(inarg);
  end
else 
  fd = inarg;
end
end

for l=1:nargin-1
  A = varargin{l}; 
  if issparse(A)
    % save sparse matrix in special Matlab format
    [m,n] = size(A);

    if min(size(A)) == 1     %a one-rank matrix will be compressed to a
                             %scalar instead of a vectory by sum
      n_nz = full(A' ~= 0);
    else
      n_nz = full(sum(A' ~= 0));
    end
    nz   = sum(n_nz);
    write(fd,[1211216,m,n,nz],'int32');

    write(fd,n_nz,'int32');   %nonzeros per row
    [i,j,s] = find(A');
    write(fd,i-1,'int32');
    if ~isreal(s)
      s = conj(s);
      l = length(s);
      sr = real(s);
      si = imag(s);
      s(1:2:2*l) = sr;
      s(2:2:2*l) = si;
    end
    write(fd,s,'double');
  else
    [m,n] = size(A);
    write(fd,[1211214,m*n],'int32');
    if ~isreal(A)
      l = length(A);
      sr = real(A);
      si = imag(A);
      A(1:2:2*l) = sr;
      A(2:2:2*l) = si;
    end
    write(fd,A,'double');
  end
end
if ischar(inarg) | isinteger(inarg) close(fd); end;
