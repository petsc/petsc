/*$Id: PETScView.java,v 1.9 2001/02/19 23:05:30 bsmith Exp bsmith $*/
/*
     Prints message indicating plugin is already installed and the returns to plugins.html
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

import java.security.*;

import java.net.*;



/*
    This is the class that this file implements (must always be the same as
  the filename).

    Applet is a subclass of PanelFrame, i.e. it is itself the base window we draw onto
*/
public class PETScPlugins extends JApplet {
    
  Container japplet;

  java.applet.AppletContext appletcontext;
  JApplet applet;

  public void init(){

    appletcontext = this.getAppletContext();
    applet        = this;
    japplet       = this.getContentPane();

    japplet.setLayout(new BorderLayout());
    japplet.add(new JTextField("Java Plugin installed on your system"),BorderLayout.NORTH);
    JButton button = new JButton("Return");
    button.addActionListener(new ActionListener(){
      public void actionPerformed(ActionEvent e) {
        try {
          appletcontext.showDocument(new URL("http://www.mcs.anl.gov/petsc/plugins.html"));
        } catch (java.net.MalformedURLException ex) {;}
      }
    }); 
    japplet.add(button,BorderLayout.CENTER);
    japplet.setVisible(true);
    japplet.validate(); 
    japplet.repaint(); 
  }

}









