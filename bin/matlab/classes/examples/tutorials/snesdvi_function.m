function err = snesdvi_function(snes,xsol,func,user)
%
%  Minimal surface area problem nonlinear function as in
%  snes/examples/tests/ex8.c
%
err = 0;

mx = user.mx; my = user.my;
hx = 1/(mx+1); hy = 1/(my+1);
hydhx = hy/hx; hxdhy = hx/hy;

% Initialize f to zero
func.Set(0.0);

f = user.dm.VecGetArray(func);
x = user.dm.VecGetArray(xsol);

% Compute function over the mesh
for (j = 1:my)
  for (i = 1:mx)
    xc = x(i,j);
    xlt = xc; xr = xc;
    xrb = xc; xb = xc;
    xl = xc; xt  = xc;
        
    if (i == 1) % left side
      xl = user.left(j+1);
      xlt = user.left(j+2);
    else
      xl = x(i-1,j);
    end
        
    if (j == 1) % bottom side
      xb = user.bottom(i+1);
      xrb = user.bottom(i+2);
    else
      xb = x(i,j-1);
    end
      
    if (i == mx) %right side
      xr  = user.right(j+1);
      xrb = user.right(j);
    else
      xr = x(i+1,j);
    end
        
    if (j == my) % top side
      xt  = user.top(i+1);
      xlt = user.top(i);
    else
      xt = x(i,j+1);
    end
        
    if (i > 1 & j<my)
      xlt = x(i-1,j+1);
    end
        
    if (j > 1 & i<mx)
      xrb = x(i+1,j-1);
    end
        
    d1 = xc-xl; d2 = xc-xr;  d3 = xc-xt;
    d4 = xc-xb; d5 = xr-xrb; d6 = xrb-xb;
    d7 = xlt-xl; d8 = xt-xlt;
        
    df1dxc = d1*hydhx;
    df2dxc = ( d1*hydhx + d4*hxdhy );
    df3dxc = d3*hxdhy;
    df4dxc = ( d2*hydhx + d3*hxdhy );
    df5dxc = d2*hydhx;
    df6dxc = d4*hxdhy;
        
    d1 = d1/hx;
    d2 = d2/hx;
    d3 = d3/hy;
    d4 = d4/hy;
    d5 = d5/hy;
    d6 = d6/hx;
    d7 = d7/hy;
    d8 = d8/hx;

    f1 = sqrt( 1.0 + d1*d1 + d7*d7);
    f2 = sqrt( 1.0 + d1*d1 + d4*d4);
    f3 = sqrt( 1.0 + d3*d3 + d8*d8);
    f4 = sqrt( 1.0 + d3*d3 + d2*d2);
    f5 = sqrt( 1.0 + d2*d2 + d5*d5);
    f6 = sqrt( 1.0 + d4*d4 + d6*d6);

    df1dxc = df1dxc/f1;
    df2dxc = df2dxc/f2;
    df3dxc = df3dxc/f3;
    df4dxc = df4dxc/f4;
    df5dxc = df5dxc/f5;
    df6dxc = df6dxc/f6;

    f(i,j) = (df1dxc+df2dxc+df3dxc+df4dxc+df5dxc+df6dxc )/2.0;
  end
end
