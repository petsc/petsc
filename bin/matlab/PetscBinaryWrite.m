function PetscBinaryWrite(inarg,varargin)
%
%  Writes in PETSc binary file sparse matrices and vectors.
%  If the array is multidimensional and dense it is saved
%  as a one dimensional PETSc Vec. If you want to save the multidimensional
%  array as a matrix that MatLoad() will read you must first convert it to 
%  a sparse matrix: for example PetscBinaryWrite('myfile',sparse(A));
%
%
%   PetscBinaryWrite(inarg,args to write,['indices','int32' or 'int64'],['precision','float64' or 'float32'])
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

indices = 'int32';
precision = 'float64';
tnargin = nargin;
for l=1:nargin-2
  if ischar(varargin{l}) && strcmpi(varargin{l},'indices')
    tnargin = min(l,tnargin-1);
    indices = varargin{l+1};
  end
  if ischar(varargin{l}) && strcmpi(varargin{l},'precision')
    tnargin = min(l,tnargin-1);
    precision = varargin{l+1};
  end
end

for l=1:nargin-1
  A = varargin{l};
  if issparse(A) || min(size(A)) > 1
    % save sparse matrix in special Matlab format
    if ~issparse(A)
        A = sparse(A);
    end
    [m,n] = size(A);

    if min(size(A)) == 1     %a one-rank matrix will be compressed to a
                             %scalar instead of a vectory by sum
      n_nz = full(A' ~= 0);
    else
      n_nz = full(sum(A' ~= 0));
    end
    nz   = sum(n_nz);
    write(fd,[1211216,m,n,nz],indices);

    write(fd,n_nz,indices);   %nonzeros per row
    [i,j,s] = find(A');
    write(fd,i-1,indices);
    if ~isreal(s)
      s = conj(s);
      ll = length(s);
      sr = real(s);
      si = imag(s);
      s(1:2:2*ll) = sr;
      s(2:2:2*ll) = si;
    end
    write(fd,s,precision);
  else
    [m,n] = size(A);
    write(fd,[1211214,m*n],indices);
    if ~isreal(A)
      ll = length(A);
      sr = real(A);
      si = imag(A);
      A(1:2:2*ll) = sr;
      A(2:2:2*ll) = si;
    end
    write(fd,A,precision);
  end
end
if ischar(inarg) || isinteger(inarg)
    close(fd)
end
