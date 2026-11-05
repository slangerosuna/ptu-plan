# HILG and Weisfeiler–Leman search control

The default portable trainer uses the HTN Instance Learning Graph (HILG) from
[Höller et al., *Learning Heuristic Functions for HTN Planning*](https://fai.cs.uni-saarland.de/hoeller/papers/2026-Hoeller-AAAI.pdf).
This implementation adapts the representation and WL features to partition
search control. It does not reproduce the paper's domain-specific SVR/GPR
remaining-plan-distance heuristic or its lifted LION search engine.

## Representation

`src/learning/hilg.rs` builds object nodes, current/goal atom nodes, and distinct
primitive/compound task occurrences from the original AST. Argument edges carry
one-based positions; repeated arguments and repeated task calls remain distinct.
Goal atoms distinguish current, goal, achieved, and negative roles. Object names
are used only to connect references, never as colors. Formula nodes retain goal
connectives and variable scopes; their features do not certify satisfaction.

Our extensions add partition membership, precedence, and the twelve directed
interaction channels. These retain information needed by the partially ordered
partitioner. This graph represents the initial planning instance at each merge
state; it is not an evolving progression-state heuristic. Schema summaries and
numeric partition features remain available alongside the instance graph.

## Fixed features and learned heads

`src/learning/wl.rs` computes directed, edge-labelled 1-WL refinement for rounds
zero through three. Each color combines its previous color with a sorted
multiset of labelled neighbor colors. Refinement uses full SHA-256 identifiers;
final histogram bins use bounded feature hashing, which permits collisions.

Two views use exact predicate/task colors and structural roles respectively.
Each round/view has 128 bins. Partition-local histograms count member task nodes
and their partition node; global histograms count instance nodes. Log-counts
produce 2,048 fixed features per partition, followed by twelve numeric features.
No attention or learned message passing is used in this encoder.

A width-32, three-layer dense projection feeds shared nonlinear merge/value
heads. Unlike the previous linear decoder, these heads can condition action
rankings on search budgets. The interface classifier retains its symbolic atom
features, and latent transitions learn against the final frozen WL encoder.
There are 73,154 merge/value, 643 interface, and 5,248 latent parameters.

## Compatibility and checks

Merge model schema 2 selects WL; schema 1 preserves existing relational
checkpoints and their frozen-encoder hashes. `train export --legacy-encoder`
exports the previous architecture. Existing immutable rollout records can be
reused: export reconstructs HILG/WL inputs from their audited source ASTs.
New cached counts are also checked against symbolic reconstruction during audit.

Tests cover finite-difference gradients, graph reindexing and object renaming,
argument positions, duplicate occurrences, goal roles, cached-feature tampering,
and checkpoint compatibility. GPU training checks native float64 losses and all
parameter gradients before fitting, then verifies exported native predictions.
Model selection uses held-out validation domains; benchmark results, rather than
prediction loss alone, determine whether the architecture improves planning.
