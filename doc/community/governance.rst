.. _governance:

****************
PETSc Governance
****************

PETSc is developed by a distributed group of **contributors** (often called developers in PETSc documentation).
They include individuals who have contributed code, documentation, designs, user support,
or other work to PETSc. Anyone can be a contributor. The PETSc **community** consists of the contributors, users, supporters, and anyone else interested in PETSc,
its use, or its future. [#source_footnote]_


Consensus-based decision making by the community
================================================

Most project decisions are made by consensus of the PETSc community. The primary goal of this approach is to ensure that the people who are
most affected by and involved in any given change can contribute their knowledge in the confidence that their voices will be heard, because thoughtful
review from a broad community is the best mechanism we know of for creating high-quality software. Anyone in the PETSc community can participate in and express their opinion
on decisions; they need not be a contributor.

The mechanism we use to accomplish this goal may be unfamiliar for those who are not experienced with the cultural norms around free/open-source software development.
We provide a summary here, and suggest reading `Chapter 4: Social and Political Infrastructure <https://producingoss.com/en/social-infrastructure.html>`__  of Karl Fogel’s
`Producing Open Source Software <https://producingoss.com/en/index.html>`__, and in particular the section on Consensus-based Democracy, for a more detailed discussion.

In this context, consensus does not require:

* that we wait to solicit everybody’s opinion on every change,
* that we hold explicit votes on anything,
* or that everybody is happy or agrees with every decision.

Consensus means that we entrust everyone with the right to request a decision by the PETSc Council.
The mere specter of a council review request ensures that community members
are motivated from the start to find a solution that everyone can live with – accomplishing our stated goal
of ensuring that all interested perspectives are taken into account.

How do we know when consensus has been achieved? In principle, this is rather difficult, since consensus
is defined by the absence of requests for a PETSc Council decision, which requires us to somehow prove a negative.
In practice, we use a combination of our best judgment
(e.g., a simple and uncontroversial bug fix posted on GitLab and reviewed by another developer is probably fine)
and best efforts (e.g., all substantive API changes must be posted to the mailing list in order to give the broader
community a chance to catch any problems and suggest improvements; we assume that anyone who cares enough about
PETSc should be on the mailing list). If no-one bothers to comment on the mailing list
after a few days, then it’s probably fine. And worst case, if a change is more controversial than expected, or a crucial critique
is delayed because someone was on vacation, then it’s no big deal: we apologize for misjudging the situation, back up, and sort things out.

PETSc Council
=============

When a decision cannot be made by community consensus, community members may request a formal vote by the **PETSc Council**.
The role of the council is as follows.

* Vote on decisions that cannot be made by consensus; with a simple majority vote of all council members being binding.

* Vote on changes to the NumFOCUS signatories which will be conducted by the council using the Schulze Method of ranked choice voting.

* Vote on the addition and removal of PETSc Council members; with a 2/3 majority vote of all council members. Anyone in the PETSc community can
  be on the PETSc Council, one need not be a contributor. The initial council will consist of the 15 most active code contributors,
  plus two long-term contributors who now play important non-coding roles in the community. The initial high bias in the council towards contributors
  is simply due to the few non-contributors who are heavily actively engaged in the community.

* Vote on :any:`changes to the governance policy<governance_changes>` (this document) with a 2/3 majority vote of all council members.

Votes are public, presented in the usual discussion venues, and the voting period must remain open for at least seven days or until a required majority has been achieved.

.. _numfocus_signatories:


NumFOCUS signatories
====================

As a requirement of fiscal sponsorship by PETSc's planned membership in **NumFOCUS** there are five initial NumFOCUS signatories from five institutions.
Their role is to manage interactions with NumFOCUS and any project funding that comes through NumFOCUS.
It is expected that such funds will be spent in a manner that is consistent with the non-profit mission of NumFOCUS. Changes in the signatories will
be done by a vote of the PETSc Council.

.. _governance_changes:

Changes to the Governance Document
==================================

Merge requests to https://gitlab.com/petsc/petsc on this file (``doc/community/governance.rst``) constitute proposed changes to the governance document.
After a community discussion of the proposed changes, the PETSc Council can pass changes to the document with a 2/3 majority vote of all members.

.. rubric:: Footnotes

.. [#source_footnote] Material in this document, including occasionally exact wording, is partially based on https://github.com/dask/governance/blob/main/governance.md and https://numpy.org/doc/stable/dev/governance/governance.html.

