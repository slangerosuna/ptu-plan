use super::{FormulaFeature, Literal, TaskSummary};
use crate::parsing::json::Formula;

pub(super) fn summarize(conditions: Option<&Formula>, effects: Option<&Formula>) -> TaskSummary {
    let mut summary = TaskSummary::default();
    if let Some(formula) = conditions {
        collect(formula, true, false, &mut summary);
    }
    if let Some(formula) = effects {
        collect(formula, true, true, &mut summary);
    }
    summary
}

fn collect(formula: &Formula, positive: bool, effect: bool, summary: &mut TaskSummary) {
    use Formula::*;
    match formula {
        Empty => {}
        Atom(atom) => {
            let target = if effect {
                &mut summary.effects
            } else {
                &mut summary.conditions
            };
            target.insert(Literal {
                predicate: atom.name.clone(),
                positive,
            });
        }
        Not(inner) => collect(inner, !positive, effect, summary),
        And(terms) | Or(terms) => {
            if matches!(formula, Or(_)) {
                summary.formula_features.insert(FormulaFeature::Disjunction);
            }
            for term in terms {
                collect(term, positive, effect, summary);
            }
        }
        Xor(terms) => {
            summary
                .formula_features
                .insert(FormulaFeature::ExclusiveDisjunction);
            // Either polarity can participate in an exclusive disjunction.
            for term in terms {
                collect(term, true, effect, summary);
                collect(term, false, effect, summary);
            }
        }
        Imply(left, right) => {
            summary.formula_features.insert(FormulaFeature::Implication);
            collect(left, !positive, effect, summary);
            collect(right, positive, effect, summary);
        }
        Exists(_, inner) | ForAll(_, inner) => {
            summary
                .formula_features
                .insert(FormulaFeature::Quantification);
            collect(inner, positive, effect, summary);
        }
        Probabilistic(_, inner) => {
            summary
                .formula_features
                .insert(FormulaFeature::Probabilistic);
            collect(inner, positive, effect, summary);
        }
        Equals(_, _) => {
            summary.formula_features.insert(FormulaFeature::Equality);
        }
    }
}
