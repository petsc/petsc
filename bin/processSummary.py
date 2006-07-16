#!/usr/bin/env python
import user
import script

class LogSummarizer(script.Script):
  transforms = [
    (', boost::tuples::null_type', ''),
    ('ALE::pair<ALE::Point, int>', 'Rank'),
    ('ALE::Sieve<ALE::Point, int, int>', 'Topology'),
    ('ALE::CoSifter<Topology, ALE::Point, ALE::Point, int>', 'Bundle'),
    ('ALE::CoSifter<Topology, ALE::Point, ALE::Point, double>', 'Field'),
    ('ALE::CoSifter<Topology, Rank, ALE::Point, int>', 'RankBundle'),
    ('ALE::CoSifter<Topology, Rank, ALE::Point, double>', 'RankField'),
    ('std::less<ALE::Point>', 'lessPoint'),
    ('std::less<Rank >', 'lessRank'),
    ('Bundle::trueFunc<ALE::Point>', 'trueFunc'),
    ('RankField::trueFunc<ALE::Point>', 'trueFunc'),
    ('Field::trueFunc<ALE::Point>', 'trueFunc'),
    ('boost::multi_index::composite_key_compare<lessPoint, lessPoint, lessPoint>', 'stdCompare'),
    ('boost::multi_index::composite_key_compare<lessPoint, std::less<int>, lessPoint>', 'stdCompareInt'),
    ('boost::multi_index::composite_key_compare<lessPoint, trueFunc, lessPoint>', 'orderCompare'),
    ('boost::multi_index::composite_key_compare<lessPoint, trueFunc, lessRank>', 'orderRankCompare'),
    ('ALE::array<ALE::Point>', 'PointArray'),
    ('ALE::set<ALE::Point>', 'PointSet'),
    ('std::vector<ALE::Point, std::allocator<ALE::Point> >', 'PointVec'),
    ('std::set<int, std::less<int>, std::allocator<int> >', 'IntSet'),
    ('ALE::SifterDef::Arrow<ALE::Point, ALE::Point, ALE::Point>', 'Arrow'),
    ('ALE::SifterDef::Arrow<ALE::Point, ALE::Point, int>', 'IntArrow'),
    ('ALE::SifterDef::Arrow<ALE::Point, Rank, ALE::Point>', 'RankArrow'),
    ('ALE::SifterDef::Arrow<int, ALE::Point, ALE::pair<ALE::Point, ALE::pair<int, int> > >', 'OverlapArrow'),
    ('ALE::SifterDef::Arrow<ALE::Point, int, ALE::pair<ALE::Point, ALE::pair<int, int> > >', 'FlipOverlapArrow'),
    ('ALE::SifterDef::Arrow<int, ALE::pair<int, ALE::Point>, ALE::pair<ALE::Point, ALE::pair<int, int> > >', 'BiOverlapArrow'),
    ('ALE::SifterDef::Arrow<ALE::pair<int, ALE::Point>, int, ALE::pair<ALE::Point, ALE::pair<int, int> > >', 'FlipBiOverlapArrow'),
    ('ALE::SifterDef::Rec<ALE::Point>', 'Point'),
    ('ALE::SifterDef::Rec<Rank >', 'RankPoint'),
    ('ALE::SifterDef::Rec<int>', 'IntPoint'),
    ('ALE::SifterDef::Rec<ALE::pair<int, ALE::Point> >', 'IPPairPoint'),
    ('ALE::SieveDef::Rec<ALE::Point, int>', 'Point&Int'),
    ('ALE::SieveDef::Rec<int, ALE::Point>', 'Int&Point'),
    ('ALE::SifterDef::RecContainerTraits<ALE::Point, Point >', 'SifterRecTraits'),
    ('ALE::SifterDef::RecContainerTraits<int, IntPoint >', 'OverlapRecTraits'),
    ('ALE::SieveDef::RecContainerTraits<ALE::Point, Point&Int >', ' SieveRecTraits'),
    ('ALE::SifterDef::RecContainer<ALE::Point, Point >', 'SifterRecCon'),
    ('ALE::SifterDef::RecContainer<Rank, RankPoint >', 'RankRecCon'),
    ('ALE::SifterDef::RecContainer<int, IntPoint >', 'OverlapRecCon'),
    ('ALE::SifterDef::RecContainer<ALE::pair<int, ALE::Point>, IPPairPoint >', 'BiOverlapRecCon'),
    ('ALE::SieveDef::RecContainer<ALE::Point, Point&Int >', 'SieveRecCon'),
    ('SifterRecTraits::PointSequence', 'SifPointSeq'),
    ('OverlapRecTraits::PointSequence', 'OvPointSeq'),
    ('SieveRecTraits::PointSequence', 'SivPointSeq'),
    ('SieveRecTraits::TwoValueSequence< SieveRecTraits::heightMarkerTag, int>', 'HeightSeq'),
    ('SieveRecTraits::TwoValueSequence< SieveRecTraits::depthMarkerTag, int>', 'DepthSeq'),
    ('ALE::RightSequenceDuplicator<ALE::ConeArraySequence<Arrow > >', 'SeqDup'),
    ('ALE::RightSequenceDuplicator<ALE::ConeArraySequence<IntArrow > >', 'IntSeqDup'),
    ('ALE::Sifter<ALE::Point, ALE::Point, ALE::Point, orderCompare, SifterRecCon, SifterRecCon >', 'Order'),
    ('ALE::Sifter<ALE::Point, ALE::Point, ALE::Point, stdCompare, SifterRecCon, SifterRecCon >', 'ReOrder'),
    ('ALE::Sifter<ALE::Point, Rank, ALE::Point, orderRankCompare, SifterRecCon, RankRecCon >', 'RankOrder'),
    ('ALE::ASifter<int, ALE::Point, ALE::pair<ALE::Point, ALE::pair<int, int> >, (ALE::SifterDef::ColorMultiplicity)1, boost::multi_index::composite_key_compare<std::less<int>, std::less<ALE::pair<ALE::Point, ALE::pair<int, int> > >, lessPoint>, OverlapRecCon, SifterRecCon >', 'Overlap'),
    ('ALE::ASifter<ALE::Point, int, ALE::pair<ALE::Point, ALE::pair<int, int> >, (ALE::SifterDef::ColorMultiplicity)1, boost::multi_index::composite_key_compare<lessPoint, std::less<ALE::pair<ALE::Point, ALE::pair<int, int> > >, std::less<int>>, SifterRecCon, OverlapRecCon >', 'FlipOverlap'),
    ('ALE::ASifter<int, ALE::pair<int, ALE::Point>, ALE::pair<ALE::Point, ALE::pair<int, int> >, (ALE::SifterDef::ColorMultiplicity)1, boost::multi_index::composite_key_compare<std::less<int>, std::less<ALE::pair<ALE::Point, ALE::pair<int, int> > >, std::less<ALE::pair<int, ALE::Point> >>, OverlapRecCon, BiOverlapRecCon >', 'BiOverlap'),
    ('ALE::ASifter<ALE::pair<int, ALE::Point>, int, ALE::pair<ALE::Point, ALE::pair<int, int> >, (ALE::SifterDef::ColorMultiplicity)1, boost::multi_index::composite_key_compare<std::less<ALE::pair<int, ALE::Point> >, std::less<ALE::pair<ALE::Point, ALE::pair<int, int> > >, std::less<int>>, BiOverlapRecCon, OverlapRecCon >', 'FlipBiOverlap'),
    ('ALE::ASifter<ALE::Point, ALE::Point, int, (ALE::SifterDef::ColorMultiplicity)1, stdCompareInt, SieveRecCon, SieveRecCon >::traits::coneSequence', 'ConeSeq'),
    ('ALE::ASifter<ALE::Point, ALE::Point, int, (ALE::SifterDef::ColorMultiplicity)1, stdCompareInt, SieveRecCon, SieveRecCon >::traits::supportSequence', 'SuppSeq'),
    ('ALE::ASifter<ALE::Point, ALE::Point, ALE::Point, (ALE::SifterDef::ColorMultiplicity)1, stdCompare, SifterRecCon, SifterRecCon >::traits::coneSequence', 'OrderFusionCone'),
    ('ALE::ASifter<ALE::Point, ALE::Point, ALE::Point, (ALE::SifterDef::ColorMultiplicity)1, stdCompare, SifterRecCon, SifterRecCon >::traits::supportSequence', 'OrderFusionSupp'),
    ('ALE::ASifter<ALE::Point, ALE::Point, ALE::Point, (ALE::SifterDef::ColorMultiplicity)1, orderCompare, SifterRecCon, SifterRecCon >::traits::coneSequence', 'OrderCone'),
    ('ALE::ASifter<ALE::Point, ALE::Point, ALE::Point, (ALE::SifterDef::ColorMultiplicity)1, orderCompare, SifterRecCon, SifterRecCon >::traits::supportSequence', 'OrderSupp'),
    ('BiOverlap::traits::coneSequence', 'BiOverlapCone'),
    ('BiOverlap::traits::supportSequence', 'BiOverlapSupp'),
    ('FlipBiOverlap::traits::coneSequence', 'FlipBiOverlapCone'),
    ('FlipBiOverlap::traits::supportSequence', 'FlipBiOverlapSupp'),
    ('ALE::Flip<Order >', 'FlipOrder'),
    ('ALE::Flip<ReOrder >', 'FlipReOrder'),
    ('boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<Arrow > > > >', 'bArrow'),
    ('boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<IntArrow > > > >', 'bIntArrow'),
    ('boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<RankArrow > > > >', 'bRankArrow'),
    ('boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<OverlapArrow > > > >', 'bOverlapArrow'),
    ('boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<FlipOverlapArrow > > > >', 'bFlipOverlapArrow'),
    ('boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<BiOverlapArrow > > > >', 'bBiOverlapArrow'),
    ('boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<FlipBiOverlapArrow > > > >', 'bFlipBiOverlapArrow'),
    ('boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<Point > >', 'bPoint'),
    ('boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<Point&Int > > > > >', 'bIntPoint'),
    ('boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<RankPoint > >', 'bRankPoint'),
    ('boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<OverlapPoint > > > >', 'bOverlapPoint'),
    ('boost::multi_index::detail::copy_map_entry<bArrow >', 'bArrowCopy'),
    ('boost::multi_index::detail::copy_map_entry<bPoint >', 'bPointCopy'),
    ('std::_Rb_tree_node<ALE::Point>', 'tPoint')
    ]

  removes = [
    'Registered event',
    'boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<ALE::SifterDef::Arrow<ALE::Point, ALE::Point, int> > > > >:',
    'boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<ALE::SieveDef::Rec<ALE::Point, int> > > > > >:',
    'boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<ALE::SifterDef::Arrow<ALE::Point, ALE::Point, ALE::Point> > > > >:',
    'boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<ALE::SifterDef::Rec<ALE::Point> > >:',
    'boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<ALE::SifterDef::Arrow<ALE::Point, ALE::pair<ALE::Point, int>, ALE::Point> > > > >:',
    'boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<ALE::SifterDef::Rec<ALE::pair<ALE::Point, int> > > >:',
    'boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<ALE::SifterDef::Rec<int> > >',
    'boost::multi_index::detail::ordered_index_node<boost::multi_index::detail::index_node_base<ALE::SifterDef::Rec<ALE::pair<int, ALE::Point> > > >',
    'std::_Rb_tree_node<ALE::Point>:',
    'Obj:std::set<std::string',
    'std::set<std::string'
    ]

  def setupHelp(self, help):
    help = script.Script.setupHelp(self, help)
    #help.addArgument('LogSummarizer', '-', nargs.ArgBool(None, 0, 'Print this help message', isTemporary = 1), ignoreDuplicates = 1)
    return help

  def filter(self, line):
    for r in LogSummarizer.removes:
      if line.startswith(r):
        return ''
    return line

  def transform(self, line):
    '''We could try to pick out just the event lines'''
    for longName, shortName in LogSummarizer.transforms:
      line = line.replace(longName, shortName)
    return line

  def process(self, inF, outF):
    for line in inF.readlines():
      line = self.filter(line)
      line = self.transform(line)
      outF.write(line)
    return

  def run(self):
    import sys

    self.setup()
    self.process(sys.stdin, sys.stdout)
    return

if __name__ == '__main__':
  LogSummarizer().run()
