//! Supervised search control. Predictions rank legal choices, never prove facts.
pub mod audit;
mod autodiff;
pub mod beam;
pub mod features;
pub mod hilg;
pub mod interface;
pub mod latent;
pub mod latent_evaluation;
pub mod latent_training;
pub mod model;
pub mod online;
pub mod pipeline;
pub mod policy;
pub mod rollout;
pub mod training;
pub mod wl;

pub mod rl;
