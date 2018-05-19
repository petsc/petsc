<?xml version="1.0" encoding="UTF-8"?>
<!--**********************************************************************************
 *    M A R I T I M E  R E S E A R C H  I N S T I T U T E  N E T H E R L A N D S     *
 *************************************************************************************
 *    authors: Koos Huijssen, Christiaan M. Klaij                                    *
 *************************************************************************************
 *    content: XML to HTML Transformation script for XML-formatted performance       *
 *             reports with nested timers.                                           *
 ***********************************************************************************-->
<xsl:stylesheet id="rundata_xml2html"
  version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output method="html"/>

  <xsl:template match="root[1]">
    <!-- base html document-->
    <html>
      <head>
        <style type="text/css">
          *, html { font-family: Verdana, Arial, Helvetica, sans-serif; }

          table
          {
            table-layout:fixed;
          }

          th, td
          {
            white-space: nowrap;
            border: 1px solid black;
            padding-left: 10px;
            padding-right: 10px;
          }

          td.timername, th.timername
          {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }

          td.numeric, font.numeric
          {
            font-family:Consolas,Monaco,Lucida Console,Liberation Mono,DejaVu Sans Mono,Bitstream Vera Sans Mono,Courier New, monospace;
          }

          ol.tree {
            padding-left: 15px;
          }

          ol.tree li {
            list-style: none;          /* all list item li dots invisible */
            position: relative;
            margin-left: -20px;
          }

          label {
            cursor: pointer;        /* cursor changes when you mouse over this class */
          }

          ol.tree td, ol.tree th {
            border:none;
            /*border: 1px solid black;*/
            text-align: left;
          }
        </style>
        <title>PETSc Performance Summary</title>
        <meta charset="utf-8"/>
        <script  src="http://ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js"></script>
        <script>jQuery.fn.extend({
            toggleTree: function (tr, isClicked, init){
                var that = this;
                var thattext = that.text();
                if (!init) thattext = thattext.slice(2);
                if (isClicked) { that.html("&#9698;&#160;"+thattext); tr.show(); isClicked = false; }
                else {that.html("&#9655;&#160;"+thattext); tr.hide(); isClicked = true; }
                this.unbind('click.expandtree');
                this.bind('click.expandtree',function (){
                    if (isClicked) { that.html("&#9698;&#160;"+thattext); tr.show(); isClicked = false; }
                    else {that.html("&#9655;&#160;"+thattext); tr.hide(); isClicked = true; }
                });
                return this;
            }
        });</script>
      </head>
      <body>
        <!--xsl:variable name="numsiblings">
          <xsl:value-of select="count(preceding-sibling::root)+count(following-sibling::root)"/>
        </xsl:variable>
        <xsl:if test="$numsiblings>0">
          <xsl:variable name="irank">
            <xsl:value-of select="count(preceding-sibling::root)"/>
          </xsl:variable>
          <h1>Process rank <xsl:value-of select="$irank"/></h1>
        </xsl:if-->
        <xsl:apply-templates/>
      </body>
    </html>
  </xsl:template>

  <xsl:template match="root[position()>1]"/>

  <xsl:template match="applicationroot">
    <!-- application root -->
    <h2><xsl:value-of select="@desc"/></h2>
    <table>
      <xsl:apply-templates/>
    </table>
  </xsl:template>

  <xsl:template match="applicationroot/*">
    <!-- application root elements -->
    <tr><th align="left"><xsl:value-of select="@desc"/></th><td><xsl:apply-templates/></td></tr>
  </xsl:template>

  <xsl:template match="runspecification">
    <!-- run specification -->
    <h2><xsl:value-of select="@desc"/></h2>
    <table width="800">
      <xsl:apply-templates/>
    </table>
  </xsl:template>

  <xsl:template match="runspecification/*">
    <!-- run specification elements -->
    <tr><th align="left" width="300"><xsl:value-of select="@desc"/></th><td><xsl:apply-templates/></td></tr>
  </xsl:template>

  <xsl:template match="globalperformance">
    <!-- global performance -->
    <h2><xsl:value-of select="@desc"/></h2>
    <table width="800">
      <tr><th width="240"></th><th>Max</th><th>Max/Min</th><th>Avg</th><th>Total</th></tr>
      <xsl:apply-templates/>
    </table>
  </xsl:template>

  <xsl:template match="globalperformance/*">
    <!-- global performance elements -->
    <tr><th align="left"><xsl:value-of select="@desc"/></th>
      <td class="numeric"><xsl:value-of select="max"/></td>
      <td class="numeric"><xsl:value-of select="ratio"/></td>
      <td class="numeric"><xsl:value-of select="average"/></td>
      <td class="numeric"><xsl:value-of select="total"/></td>
    </tr>
  </xsl:template>

  <xsl:template match="timertree">
    <!-- timer tree -->
    <h2><xsl:value-of select="@desc"/>
      <xsl:if test="totaltime">
        (time in % of <xsl:value-of select="format-number(totaltime,'####0.##')"/> s, threshold = <xsl:value-of select="format-number(timethreshold,'##0.##')"/> %)
      </xsl:if>
    </h2>
    <xsl:variable name="iroot">
      <xsl:value-of select="count(preceding::root|ancestor::root)"/>
    </xsl:variable>
    <xsl:element name="button">
      <xsl:attribute name="onclick">setAllExpand<xsl:value-of select="$iroot"/>(true)</xsl:attribute>
      Expand all
    </xsl:element>
    <xsl:element name="button">
      <xsl:attribute name="onclick">setAllExpand<xsl:value-of select="$iroot"/>(false)</xsl:attribute>
      Collapse all
    </xsl:element>
    <ol class="tree">
      <xsl:call-template name="treeheader"/>
      <xsl:apply-templates select="event"/>
    </ol>
    <xsl:variable name="eventtreestart">
      <xsl:value-of select="count(preceding::events|ancestor::events)"/>
    </xsl:variable>
    <xsl:variable name="neventtree">
      <xsl:value-of select="count(descendant::events)"/>
    </xsl:variable>
    <script>
      $(document).ready(function()
      {
      <xsl:call-template name="toggleTreeLoop">
        <xsl:with-param name="i" select="$eventtreestart+1"/>
        <xsl:with-param name="limit" select="$eventtreestart+$neventtree"/>
        <xsl:with-param name="expandTree">false</xsl:with-param>
        <xsl:with-param name="init">true</xsl:with-param>
      </xsl:call-template>
      });
    </script>
    <script>
      function setAllExpand<xsl:value-of select="$iroot"/>(expandTree)
      {
      <xsl:call-template name="toggleTreeLoop">
        <xsl:with-param name="i" select="$eventtreestart+1"/>
        <xsl:with-param name="limit" select="$eventtreestart+$neventtree"/>
        <xsl:with-param name="expandTree">expandTree</xsl:with-param>
        <xsl:with-param name="init">false</xsl:with-param>
      </xsl:call-template>
      };
    </script>
  </xsl:template>

  <xsl:template match="selftimertable">
    <!-- Self timer table -->
    <h2><xsl:value-of select="@desc"/>
      <xsl:if test="totaltime">
        (time in % of <xsl:value-of select="format-number(totaltime,'####0.00')"/> s)
      </xsl:if>
    </h2>
    <ol class="tree">
      <xsl:call-template name="selftreeheader"/>
      <xsl:apply-templates select="event"/>
    </ol>
  </xsl:template>

  <xsl:template match="event[events]">
    <!--tree-->
    <li>
      <xsl:variable name="eventtreeid">
        <xsl:value-of select="count(preceding::events|ancestor::events)+1"/>
      </xsl:variable>
      <table width="1130">
        <tr>
          <xsl:element name="td">
            <xsl:attribute name="width">180</xsl:attribute>
            <xsl:attribute name="class">timername</xsl:attribute>
            <xsl:attribute name="title"><xsl:value-of select="name"/></xsl:attribute>
            <xsl:element name="label">
              <xsl:attribute name="class">h<xsl:value-of select="$eventtreeid"/></xsl:attribute>
              <xsl:value-of select="name"/>
            </xsl:element>
          </xsl:element>
          <xsl:call-template name="elemperfresults"/>
        </tr>
      </table>
      <xsl:element name="ol">
        <xsl:attribute name="class">t<xsl:value-of select="$eventtreeid"/></xsl:attribute>
        <xsl:apply-templates select="events/event"/>
      </xsl:element>
    </li>
  </xsl:template>

  <xsl:template match="event[not(events)]">
    <!--end-node-->
    <xsl:variable name="tm">
      <xsl:choose>
        <xsl:when test="time/value">
          <xsl:value-of select="time/value"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="time/maxvalue"/>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:variable>
    <xsl:if test="$tm &gt; 0">
      <li>
        <table width="1250">
          <tr>
            <xsl:element name="td">
              <xsl:attribute name="width">180</xsl:attribute>
              <xsl:attribute name="class">timername</xsl:attribute>
              <xsl:attribute name="title"><xsl:value-of select="name"/></xsl:attribute>
              &#160;&#160;&#160;&#160;<xsl:value-of select="name"/>
            </xsl:element>
            <xsl:call-template name="elemperfresults"/>
          </tr>
        </table>
      </li>
     </xsl:if>
  </xsl:template>

  <xsl:template name="treeheader">
    <li>
      <table width="1250">
        <tr>
          <th width="210" class="timername">&#160;&#160;&#160;Name</th>
          <th width="190">Time (%)</th>
          <th width="220"><font color="red" face="TrueType">Calls/parent call</font></th>
          <th width="220"><font color="blue" face="TrueType">Compute (Mflops)</font></th>
          <th width="220"><font color="green" face="TrueType">Transfers (MiB/s)</font></th>
          <th width="190"><font color="magenta" face="TrueType">Reductions/s</font></th>
        </tr>
      </table>
    </li>
  </xsl:template>

  <xsl:template name="elemperfresults">
    <td class="numeric" width="190">
      <xsl:call-template name="printperfelem_prct">
        <xsl:with-param name="varname" select="time"/>
      </xsl:call-template>
    </td>
    <td width="220"><font color="red" class="numeric">
      <xsl:call-template name="printperfelem_calls">
        <xsl:with-param name="varname" select="ncalls"/>
      </xsl:call-template>
    </font></td>
    <td width="220"><font color="blue" class="numeric">
      <xsl:call-template name="printperfelem">
        <xsl:with-param name="varname" select="mflops"/>
      </xsl:call-template>
    </font></td>
    <td width="220"><font color="green" class="numeric">
      <xsl:call-template name="printperfelem">
        <xsl:with-param name="varname" select="mbps"/>
      </xsl:call-template>
    </font></td>
    <td width="220"><font color="magenta" class="numeric">
      <xsl:call-template name="printperfelem">
        <xsl:with-param name="varname" select="nreductsps"/>
      </xsl:call-template>
    </font></td>
  </xsl:template>

  <xsl:template match="selftimertable/event">
    <xsl:variable name="tm1">
      <xsl:choose>
        <xsl:when test="time/value">
          <xsl:value-of select="time/value"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="time/maxvalue"/>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:variable>
    <xsl:if test="$tm1 &gt; 0">
      <li>
        <table width="1030">
          <tr>
            <xsl:element name="td">
              <xsl:attribute name="width">180</xsl:attribute>
              <xsl:attribute name="class">timername</xsl:attribute>
              <xsl:attribute name="title"><xsl:value-of select="name"/></xsl:attribute>
              &#160;&#160;&#160;<xsl:value-of select="name"/>
            </xsl:element>
            <td class="numeric" width="190">
              <xsl:call-template name="printperfelem_prct">
                <xsl:with-param name="varname" select="time"/>
              </xsl:call-template>
            </td>
            <td width="220"><font color="blue" class="numeric">
              <xsl:call-template name="printperfelem">
                <xsl:with-param name="varname" select="mflops"/>
              </xsl:call-template>
            </font></td>
            <td width="220"><font color="green" class="numeric">
              <xsl:call-template name="printperfelem">
                <xsl:with-param name="varname" select="mbps"/>
              </xsl:call-template>
            </font></td>
            <td width="220"><font color="magenta" class="numeric">
              <xsl:call-template name="printperfelem">
                <xsl:with-param name="varname" select="nreductsps"/>
              </xsl:call-template>
            </font></td>
          </tr>
        </table>
      </li>
    </xsl:if>
  </xsl:template>

  <xsl:template name="selftreeheader">
    <li>
      <table width="1030">
        <tr>
          <th width="210" class="timername">&#160;&#160;&#160;Name</th>
          <th width="190">Time (%)</th>
          <th width="220"><font color="blue" face="TrueType">Compute (Mflops)</font></th>
          <th width="220"><font color="green" face="TrueType">Transfers (MiB/s)</font></th>
          <th width="190"><font color="magenta" face="TrueType">Reductions/s</font></th>
        </tr>
      </table>
    </li>
  </xsl:template>

  <xsl:template name="append-pad">
    <xsl:param name="string"/>
    <xsl:param name="appendchar"/>
    <xsl:param name="length"/>
    <xsl:choose>
      <xsl:when test="string-length($string) &lt; $length">
        <xsl:call-template name="append-pad">
          <xsl:with-param name="string" select="concat($appendchar,$string)"/>
          <xsl:with-param name="appendchar" select="$appendchar"/>
          <xsl:with-param name="length" select="$length"/>
        </xsl:call-template>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="substring($string,1,$length)"/>
      </xsl:otherwise>
      </xsl:choose>
  </xsl:template>

  <xsl:template name="printperfelem_prct">
    <xsl:param name="varname"/>
    <xsl:if test="$varname">
      <xsl:choose>
        <xsl:when test="$varname/value">
          &#160;&#160;&#160;&#160;
          <xsl:call-template name="append-pad">
            <xsl:with-param name="string" select="format-number($varname/value,'##0')"/>
            <xsl:with-param name="appendchar" select="'&#160;'"/>
            <xsl:with-param name="length" select="2"/>
          </xsl:call-template>
        </xsl:when>
        <xsl:when test="$varname/minvalue">
          <xsl:call-template name="append-pad">
            <xsl:with-param name="string" select="format-number($varname/minvalue,'##0')"/>
            <xsl:with-param name="appendchar" select="'&#160;'"/>
            <xsl:with-param name="length" select="2"/>
          </xsl:call-template>
          -
          <xsl:if test="$varname/avgvalue">
            <xsl:call-template name="append-pad">
              <xsl:with-param name="string" select="format-number($varname/avgvalue,'##0')"/>
              <xsl:with-param name="appendchar" select="'&#160;'"/>
              <xsl:with-param name="length" select="2"/>
            </xsl:call-template>
            -
          </xsl:if>
          <xsl:call-template name="append-pad">
            <xsl:with-param name="string" select="format-number($varname/maxvalue,'##0')"/>
            <xsl:with-param name="appendchar" select="'&#160;'"/>
            <xsl:with-param name="length" select="2"/>
          </xsl:call-template>
        </xsl:when>
      </xsl:choose>
    </xsl:if>
  </xsl:template>

  <xsl:template name="printperfelem">
    <xsl:param name="varname"/>
    <xsl:if test="$varname">
      <xsl:choose>
        <xsl:when test="$varname/value">
          &#160;&#160;&#160;&#160;&#160;&#160;
          <xsl:call-template name="append-pad">
            <xsl:with-param name="string" select="format-number($varname/value,'####0')"/>
            <xsl:with-param name="appendchar" select="'&#160;'"/>
            <xsl:with-param name="length" select="4"/>
          </xsl:call-template>
        </xsl:when>
        <xsl:when test="$varname/minvalue">
          <xsl:call-template name="append-pad">
            <xsl:with-param name="string" select="format-number($varname/minvalue,'####0')"/>
            <xsl:with-param name="appendchar" select="'&#160;'"/>
            <xsl:with-param name="length" select="4"/>
          </xsl:call-template>
          -
          <xsl:if test="$varname/avgvalue">
            <xsl:call-template name="append-pad">
              <xsl:with-param name="string" select="format-number($varname/avgvalue,'####0')"/>
            <xsl:with-param name="appendchar" select="'&#160;'"/>
              <xsl:with-param name="length" select="4"/>
            </xsl:call-template>
            -
          </xsl:if>
          <xsl:call-template name="append-pad">
            <xsl:with-param name="string" select="format-number($varname/maxvalue,'####0')"/>
            <xsl:with-param name="appendchar" select="'&#160;'"/>
            <xsl:with-param name="length" select="4"/>
          </xsl:call-template>
        </xsl:when>
      </xsl:choose>
    </xsl:if>
  </xsl:template>

  <xsl:template name="printperfelem_calls">
    <xsl:param name="varname"/>
    <xsl:if test="$varname">
      <xsl:choose>
        <xsl:when test="$varname/value">
          &#160;&#160;&#160;&#160;&#160;&#160;
          <xsl:call-template name="append-pad">
            <xsl:with-param name="string" select="format-number($varname/value,'####0.00')"/>
            <xsl:with-param name="appendchar" select="'&#160;'"/>
            <xsl:with-param name="length" select="4"/>
          </xsl:call-template>
        </xsl:when>
        <xsl:when test="$varname/minvalue">
          <xsl:call-template name="append-pad">
            <xsl:with-param name="string" select="format-number($varname/minvalue,'####0.00')"/>
            <xsl:with-param name="appendchar" select="'&#160;'"/>
            <xsl:with-param name="length" select="4"/>
          </xsl:call-template>
          -
          <xsl:if test="$varname/avgvalue">
            <xsl:call-template name="append-pad">
              <xsl:with-param name="string" select="format-number($varname/avgvalue,'####0.00')"/>
            <xsl:with-param name="appendchar" select="'&#160;'"/>
              <xsl:with-param name="length" select="4"/>
            </xsl:call-template>
            -
          </xsl:if>
          <xsl:call-template name="append-pad">
            <xsl:with-param name="string" select="format-number($varname/maxvalue,'####0.00')"/>
            <xsl:with-param name="appendchar" select="'&#160;'"/>
            <xsl:with-param name="length" select="4"/>
          </xsl:call-template>
        </xsl:when>
      </xsl:choose>
    </xsl:if>
  </xsl:template>

  <xsl:template name="toggleTreeLoop">
    <xsl:param name="i"/>
    <xsl:param name="limit"/>
    <xsl:param name="expandTree"/>
    <xsl:param name="init"/>
    <xsl:if test="$i &lt;= $limit">
      $(".h<xsl:value-of select="$i"/>").toggleTree($(".t<xsl:value-of select="$i"/>"),<xsl:value-of select="$expandTree"/>,<xsl:value-of select="$init"/>)
      <xsl:call-template name="toggleTreeLoop">
        <xsl:with-param name="i" select="$i+1"/>
        <xsl:with-param name="limit" select="$limit"/>
        <xsl:with-param name="expandTree" select="$expandTree"/>
        <xsl:with-param name="init" select="$init"/>
      </xsl:call-template>
    </xsl:if>
  </xsl:template>

</xsl:stylesheet>
