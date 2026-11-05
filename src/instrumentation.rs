//! Versioned observer events; collecting traces never supplies symbolic rules.

use crate::master::{MasterStatistics, MasterStatus, RootPartition, VerifiedNogood};
use crate::solver::{LocalRequest, PlanCertificate, ProofStatus, SearchBudget, SearchStatistics};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OraclePurpose {
    #[default]
    Solve,
    CoreMinimization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceTrainingContext {
    pub boundary: Vec<crate::solver::GroundAtom>,
    pub future_reads: crate::solver::State,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceLevel {
    Summary,
    Full,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MasterEvent {
    InitialNetwork {
        initial_network: usize,
        partitions: Vec<RootPartition>,
    },
    BoundaryLimit {
        initial_network: usize,
        partition: usize,
        atoms: usize,
    },
    OracleStarted {
        call: usize,
        initial_network: usize,
        partition: usize,
        budget: SearchBudget,
        request: Option<LocalRequest>,
        #[serde(default)]
        purpose: OraclePurpose,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        interface_context: Option<InterfaceTrainingContext>,
    },
    OracleFinished {
        call: usize,
        status: ProofStatus,
        statistics: SearchStatistics,
        certificate: Option<PlanCertificate>,
    },
    CutVerified {
        call: usize,
        nogood: VerifiedNogood,
    },
    ConflictCoreVerified {
        core: crate::conflicts::VerifiedConflictCore,
    },
    ConflictCoreApplied {
        core_index: usize,
        initial_network: usize,
        partition: usize,
        initial_state: Option<crate::solver::State>,
        goal: Option<crate::solver::Condition>,
        boundary: Option<Vec<crate::solver::GroundAtom>>,
        ordinal: usize,
        next_ordinal: usize,
    },
    InterfaceProposed {
        initial_network: usize,
        partition: usize,
        proposal: crate::solver::BoundaryState,
        inference_time_us: u128,
    },
    Finished {
        status: MasterStatus,
        statistics: MasterStatistics,
    },
}
