function ex29view(v)
%
%   MATLAB script that plots solution from ex29.c
%
[m,n] = size(v);
n     = sqrt(m/4);
w     = zeros(n,n);
cnt   = 1;

for i=1:4
  w(:) = v(i:4:m);
  figure(i);
  surf(w);
end


