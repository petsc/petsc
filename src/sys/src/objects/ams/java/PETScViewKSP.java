/*$Id: PETScViewKSP.java,v 1.2 2001/02/19 23:05:19 bsmith Exp $*/
/*
     Accesses the PETSc published objects
*/

/*  These are the Java GUI classes */
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.tree.*;

/* For the text input regions */
import javax.swing.text.*;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.Locale;
import java.util.*;

/*   This allows multiple threads */
import java.lang.Thread;

/*  These are the AMS API classes */
import gov.anl.mcs.ams.*;
import java.security.*;

import java.net.*;

/* for XY plots */
import ptolemy.plot.*;

/*
    This is the class that this file implements (must always be the same as
  the filename).
           Panel that displays information about a KSP object
*/
public class PETScViewKSP extends JInternalFrame {

  Container pane;

  public PETScViewKSP(AMS_Memory mem) {
    super("KSP",true,true,true);
    this.setVisible(true);
    this.setSize(300,300);

    pane = this.getContentPane();
    pane.setLayout(new BorderLayout());

    int n;
    double residuals[];
    String flist[] = mem.get_field_list();
    System.out.println("fields"+flist);

    n = mem.get_field("ResidualNormsCount").getIntData()[0];
    residuals = mem.get_field("ResidualNorms").getDoubleData();

    Plot plot = new Plot();
    int i;
    plot.setSize(300,300);
    plot.setYLog(true);
    plot.addLegend(1,"Residual norm");
    for (i=0; i<n; i++ ) {
      plot.addPoint(1,(double)i,residuals[i],true);
    }
    plot.fillPlot();
       pane.add(plot,BorderLayout.CENTER); 

  }

}









