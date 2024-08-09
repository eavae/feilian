# SWDE Failed Case

There are some mistakes in SWDE dataset:

* `job-careerbuilder`, the location is identicated from commented html, which supposed to incorrect. The ids are `[44128,44129,44130,44596,44597,44598,44599,44600,44601,44609,44610,44611,44612,44613,44614,44615,44616,44617,44618,44619,45090]`.
* `restaurant-urbanspoon`, a corner case of html escape issue, like `address: S Lamar Blvd &amp;amp; Barton Skwy`. The id is 103190.
* `restaurant-usdiners`, the answer is reformatted. Original `American /American` but the ground truth is `American/American`. The id are `[104334,104751,104973]`