/*$Id: PumpStream.java,v 1.1 2000/10/24 20:43:31 bsmith Exp bsmith $*/
/*
     Connects two streams together. Simply takes all input from first and
  passes it into second. This allows us to move stdout etc. from the server 
  back to the client
*/

import java.lang.Thread;
import java.io.*;

public class PumpStream {
  InputStream  in;
  OutputStream out;

  /*
     The constructor for the class.
  */
  public PumpStream(InputStream i,OutputStream o)
  {
    in  = i;
    out = o;
  }

  public int Pump() throws java.io.IOException
  {
    byte[] data = new byte[1024];
    int    fd   = 0,cnt = 0;
    while (true) {
      fd = in.read(data);
      if (fd == -1) break;
      cnt += fd;
      out.write(data);
    }
    out.close();
    return cnt;
  }
}




