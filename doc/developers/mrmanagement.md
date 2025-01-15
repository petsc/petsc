(mr_management)=

# Merge request management

At any given time, at least one of the {any}`sec_core_developers` is an
"MR Shepherd" and ensures that open merge
requests progress through the {any}`review process <sec_mr_reviewing>`,
examining open merge requests and taking appropriate action.

```{eval-rst}
.. list-table:: MR Shepherd Checks
      :widths: 50 50
      :align: left
      :header-rows: 1

      * - MR State
        - Action
      * - Missing a :any:`workflow label <sec_workflow_labels>` and other labels
        - Add an appropriate label, or label ``workflow::Waiting-on-Submitter`` and ask the submitter/assignee to update
      * - Without an assignee
        - Assign the submitter (if the MR is from a fork, also list an appropriate developer)
      * - Without reviewers
        - Assign reviewers
```

If MRs are inactive for too long, remind the submitter/assignee, reviewer(s), or integrator(s) of actions to take.
If the submitter/assignee must take action, change the label to `workflow::Waiting-on-Submitter`.

```{eval-rst}
.. list-table:: MR Inactivity Thresholds
      :widths: 50 50
      :align: left
      :header-rows: 1

      * - MR state
        - Inactivity threshold
      * - ``workflow:Pipeline-Testing``
        - One week
      * - ``workflow::Review``
        - One week
      * - ``workflow::Ready-for-Merge``
        - Three days
      * - ``workflow::Waiting-on-Submitter``
        - One month
      * - ``workflow::Request-for-Comment``
        - One month
      * - ``workflow::Requires-Discussion``
        - One month
      * - All others
        - One year
```

If a submitter has been unresponsive for a year,
close the MR, label `workflow::Inactive-closed`,
and let the submitter know that they may reopen if desired.
