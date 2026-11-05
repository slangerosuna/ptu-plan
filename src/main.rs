use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use serde::Serialize;

use ptu_plan::agglomeration::{Agglomeration, AgglomerationOptions, greedy_agglomerate};
use ptu_plan::analysis::{HtnAnalysis, NetworkId, analyze_program};
use ptu_plan::instrumentation::{MasterEvent, TraceLevel};
use ptu_plan::interaction::{GraphView, InteractionGraph};
use ptu_plan::interleaving::{InterleavingOptions, InterleavingResult, solve_interleaved};
use ptu_plan::learning::{
    beam::{BeamOptions, beam_agglomerate_with_latent},
    interface::{InterfaceCheckpoint, InterfaceModel},
    latent_training::LatentCheckpoint,
    model::MergeModel,
    policy::{PolicyTrace, learned_agglomerate_with_context},
    training::Checkpoint,
};
use ptu_plan::master::{
    MasterGuidance, MasterOptions, MasterResult, MasterStatus, initial_partitions,
    solve_sequential_guided,
};
use ptu_plan::parsing::discovery::{discover_sibling_problems, expand_problem_arg};
use ptu_plan::parsing::json::Program;
use ptu_plan::parsing::parser::{
    is_json_file, parse_program, read_program_json, resolve_parser_config,
};
use ptu_plan::partition::{MicroPartitioning, SpectralOptions, micropartition_network};
use ptu_plan::solver::{
    ExactnessMode, GroundingOptions, ProofStatus, SearchBudget, SolveResult,
    ground_reachable_program, solve_problem,
};

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Analyze, partition, and solve HDDL task networks",
    group(clap::ArgGroup::new("planning").args(["solve", "sequential", "interleave"]))
)]
struct Options {
    /// Domain file (.hddl), or a whole-program JSON export
    #[arg(value_name = "DOMAIN")]
    domain_path: PathBuf,

    /// Problem files or directories; defaults to the domain's sibling HDDL files
    #[arg(value_name = "PROBLEM")]
    problem_paths: Vec<PathBuf>,

    /// Parser executable (otherwise HDDL_PARSER, then hddl_analyzer on PATH)
    #[arg(long, value_name = "PATH")]
    parser: Option<PathBuf>,

    /// Validate HTN structure and print static features and graph view counts
    #[arg(long, conflicts_with = "analysis_json")]
    analyze: bool,

    /// Emit one JSON analysis/graph record per input program
    #[arg(long)]
    analysis_json: bool,

    /// Solve with exact progression search and replay the returned certificate
    #[arg(long, conflicts_with = "sequential")]
    solve: bool,

    /// Spectrally partition the problem's initial task occurrences
    #[arg(long)]
    micropartition: bool,

    /// Greedily merge spectral seeds while retaining a DAG of partition order
    #[arg(long)]
    agglomerate: bool,

    /// Learned merge-policy checkpoint; predictions select legal merges or STOP
    #[arg(long, requires = "agglomerate")]
    merge_model: Option<PathBuf>,
    #[arg(long, default_value_t = 0.1)]
    merge_value_weight: f64,
    /// Use receding-horizon beam agglomeration instead of one-step learned choices
    #[arg(long, requires = "merge_model")]
    merge_beam: bool,
    /// Frozen-encoder transition checkpoint for shallow latent beam lookahead
    #[arg(long, requires = "merge_beam")]
    latent_model: Option<PathBuf>,
    #[arg(long, default_value_t = 8)]
    beam_width: usize,
    #[arg(long, default_value_t = 3)]
    beam_depth: usize,
    #[arg(long, default_value_t = 8)]
    beam_branching: usize,
    #[arg(long, default_value_t = 10_000)]
    beam_max_evaluations: usize,
    /// Trained interface predictor; reorders exhaustive boundary valuations
    #[arg(long, requires = "sequential")]
    interface_model: Option<PathBuf>,
    /// Deletion probes per infeasible local contract; zero disables core cuts
    #[arg(long, default_value_t = 0, requires = "sequential")]
    conflict_core_checks: usize,

    /// Search sequential partial-state contracts and replay the composed plan
    #[arg(long)]
    sequential: bool,
    /// Repair failed partitions within one shared master time/call/node budget.
    #[arg(long, requires = "sequential")]
    repartition: bool,
    #[arg(long, default_value_t = 6, requires = "repartition")]
    repartition_attempts: usize,
    #[arg(long, default_value_t = 350, requires = "repartition")]
    repartition_stage_ms: u64,
    #[arg(long, default_value_t = 4000, requires = "repartition")]
    repartition_max_stage_ms: u64,
    #[arg(long, requires = "repartition")]
    no_monolithic_fallback: bool,
    /// Reconcile event-local causal requirements across interleaved partitions
    #[arg(long)]
    interleave: bool,
    #[arg(long, default_value_t = 256, requires = "interleave")]
    max_pocl_events: usize,
    #[arg(long, default_value_t = 1024, requires = "interleave")]
    max_pocl_frontier: usize,

    /// Include per-oracle requests, results, and proof events in JSON output
    #[arg(long,requires_all=["sequential","analysis_json"])]
    master_trace: bool,

    #[arg(long, default_value_t = 32)]
    merge_max_size: usize,
    #[arg(long, default_value_t = 1)]
    target_clusters: usize,
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    merge_min_score: f64,
    #[arg(long, default_value_t = 12)]
    max_boundary_atoms: usize,
    #[arg(long, default_value_t = 1_000)]
    max_local_calls: usize,
    #[arg(long, default_value_t = 10_000)]
    max_master_nodes: usize,
    #[arg(long, default_value_t = 30_000)]
    master_time_limit_ms: u64,

    #[arg(long, default_value_t = 8)]
    micro_max_size: usize,
    #[arg(long, default_value_t = 1)]
    micro_min_size: usize,

    /// View weights: ordering,support,threat,resource,hierarchy,shared-state
    #[arg(
        long,
        value_delimiter = ',',
        num_args = 1,
        default_value = "1,1,1,1,1,1"
    )]
    view_weights: Vec<f64>,

    #[arg(long, default_value_t = 100_000)]
    max_ground_instances: usize,
    #[arg(long, default_value_t = 100_000)]
    max_expansions: usize,
    #[arg(long, default_value_t = 500_000)]
    max_generated: usize,
    #[arg(long, default_value_t = 5_000)]
    time_limit_ms: u64,

    /// Bound refinement depth; without this flag, search is unbounded in depth
    #[arg(long, requires = "planning")]
    max_decomposition_depth: Option<usize>,
}

fn main() -> ExitCode {
    match run(&Options::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("ptu-plan: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(options: &Options) -> Result<(), String> {
    if is_json_file(&options.domain_path) {
        if !options.problem_paths.is_empty() {
            return Err("a .json input already contains its problem; use an .hddl domain to pair it with problem files".into());
        }
        return display_program(&read_program_json(&options.domain_path)?, options);
    }

    let config = resolve_parser_config(options.parser.as_deref())?;
    let domain_program = parse_program(&config, &options.domain_path, None)?;

    let problem_paths = if options.problem_paths.is_empty() {
        discover_sibling_problems(&options.domain_path)
            .map_err(|err| format!("cannot list problem files: {err}"))?
    } else {
        let mut paths = Vec::new();
        for argument in &options.problem_paths {
            paths.extend(
                expand_problem_arg(argument)
                    .map_err(|err| format!("cannot read {}: {err}", argument.display()))?,
            );
        }
        paths
    };

    if problem_paths.is_empty() {
        return display_program(&domain_program, options);
    }
    if !processing(options) {
        print_program(&domain_program);
    }
    for path in problem_paths {
        let program = parse_program(&config, &options.domain_path, Some(&path))?;
        let problem = program
            .problem
            .as_ref()
            .ok_or_else(|| format!("the parser produced no problem for {}", path.display()))?;
        if processing(options) {
            display_program(&program, options)?;
        } else {
            println!(
                "  Problem {} ({} objects)",
                problem.name,
                problem.objects.len()
            );
        }
    }
    Ok(())
}

fn display_program(program: &Program, options: &Options) -> Result<(), String> {
    if !processing(options) {
        print_program(program);
        return Ok(());
    }
    let analysis = analyze_program(program).map_err(|err| err.to_string())?;
    let graph = InteractionGraph::from_analysis(&analysis);
    let micropartitions = if options.micropartition
        || options.agglomerate
        || options.sequential
        || options.interleave
    {
        if analysis.initial_network.is_none() {
            return Err("micropartitioning requires an initial task network".into());
        }
        if options.view_weights.len() != 6 {
            return Err("--view-weights requires six comma-separated values: ordering,support,threat,resource,hierarchy,shared-state".into());
        }
        let w = &options.view_weights;
        let weights = ptu_plan::interaction::ViewWeights {
            ordering: w[0],
            support: w[1],
            threat: w[2],
            resource: w[3],
            hierarchy: w[4],
            shared_state: w[5],
        };
        Some(
            micropartition_network(
                &graph,
                NetworkId::Initial,
                &SpectralOptions {
                    max_cluster_size: options.micro_max_size,
                    min_cluster_size: options.micro_min_size,
                    weights,
                    ..SpectralOptions::default()
                },
            )
            .map_err(|err| err.to_string())?,
        )
    } else {
        None
    };
    let budget = SearchBudget {
        max_expansions: options.max_expansions,
        max_generated: options.max_generated,
        max_time_ms: options.time_limit_ms,
        exactness: options.max_decomposition_depth.map_or(
            ExactnessMode::Unbounded,
            |max_decomposition_depth| ExactnessMode::Bounded {
                max_decomposition_depth,
            },
        ),
    };
    let master_options = MasterOptions {
        max_master_nodes: options.max_master_nodes,
        max_local_calls: options.max_local_calls,
        max_boundary_atoms: options.max_boundary_atoms,
        max_time_ms: options.master_time_limit_ms,
        local_budget: budget,
    };
    let w = if options.view_weights.len() == 6 {
        options.view_weights.as_slice()
    } else {
        &[1.0; 6]
    };
    let merge_options = AgglomerationOptions {
        max_cluster_size: options.merge_max_size,
        target_clusters: options.target_clusters,
        min_score: options.merge_min_score,
        weights: ptu_plan::interaction::ViewWeights {
            ordering: w[0],
            support: w[1],
            threat: w[2],
            resource: w[3],
            hierarchy: w[4],
            shared_state: w[5],
        },
        ..Default::default()
    };
    let mut merge_policy = None;
    let agglomeration = if options.agglomerate {
        let config = merge_options;
        let seeds = &micropartitions.as_ref().unwrap().clusters;
        if let Some(path) = &options.merge_model {
            let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
            let value: serde_json::Value =
                serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
            let model: MergeModel = if value.get("model").is_some() {
                let checkpoint: Checkpoint =
                    serde_json::from_value(value).map_err(|e| e.to_string())?;
                checkpoint.validate().map_err(|e| e.to_string())?;
                checkpoint.model
            } else {
                serde_json::from_value(value).map_err(|e| e.to_string())?
            };
            let (partitioning, trace) = if options.merge_beam {
                let latent: Option<LatentCheckpoint> = options
                    .latent_model
                    .as_ref()
                    .map(|path| -> Result<_, String> {
                        let checkpoint: LatentCheckpoint = serde_json::from_slice(
                            &std::fs::read(path).map_err(|e| e.to_string())?,
                        )
                        .map_err(|e| e.to_string())?;
                        checkpoint.validate().map_err(|e| e.to_string())?;
                        checkpoint
                            .model
                            .validate(&model)
                            .map_err(|e| e.to_string())?;
                        Ok(checkpoint)
                    })
                    .transpose()?;
                beam_agglomerate_with_latent(
                    &analysis,
                    &graph,
                    seeds,
                    &config,
                    &model,
                    options.merge_value_weight,
                    &master_options,
                    &BeamOptions {
                        width: options.beam_width,
                        depth: options.beam_depth,
                        branching: options.beam_branching,
                        max_evaluations: options.beam_max_evaluations,
                    },
                    latent.as_ref().map(|c| &c.model),
                )
            } else {
                learned_agglomerate_with_context(
                    &analysis,
                    &graph,
                    seeds,
                    &config,
                    &model,
                    options.merge_value_weight,
                    &master_options,
                )
            }
            .map_err(|e| e.to_string())?;
            merge_policy = Some(trace);
            Some(partitioning)
        } else {
            Some(
                greedy_agglomerate(&analysis, &graph, NetworkId::Initial, seeds, &config)
                    .map_err(|e| e.to_string())?,
            )
        }
    } else {
        None
    };
    let model = if options.solve || options.sequential || options.interleave {
        Some(
            ground_reachable_program(
                program,
                &GroundingOptions {
                    max_instances: options.max_ground_instances,
                },
            )
            .map_err(|err| err.to_string())?,
        )
    } else {
        None
    };
    let mut repartition = None;
    let mut master_events = None;
    let mut interface_model_sha256 = None;
    let sequential = if options.sequential {
        let clusters = agglomeration
            .as_ref()
            .map(|a| &a.clusters)
            .unwrap_or(&micropartitions.as_ref().unwrap().clusters);
        let partitions = initial_partitions(&graph, clusters).map_err(|e| e.to_string())?;
        let interface_model: Option<InterfaceModel> = options
            .interface_model
            .as_ref()
            .map(|path| -> Result<_, String> {
                let value: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?;
                if value.get("model").is_some() {
                    let checkpoint: InterfaceCheckpoint =
                        serde_json::from_value(value).map_err(|e| e.to_string())?;
                    checkpoint.validate().map_err(|e| e.to_string())?;
                    Ok(checkpoint.model)
                } else {
                    let model: InterfaceModel =
                        serde_json::from_value(value).map_err(|e| e.to_string())?;
                    model.validate().map_err(|e| e.to_string())?;
                    Ok(model)
                }
            })
            .transpose()?;
        if let Some(model) = &interface_model {
            interface_model_sha256 = Some(ptu_plan::training::sha256(
                &serde_json::to_vec(model).map_err(|e| e.to_string())?,
            ));
        }
        let guidance = MasterGuidance {
            interface: interface_model
                .as_ref()
                .map(|m| m as &dyn ptu_plan::master::InterfaceGuide),
            conflict_core_checks: options.conflict_core_checks,
        };
        let result = if options.repartition {
            let outcome = ptu_plan::escalation::solve_escalating(
                model.as_ref().unwrap(),
                &analysis,
                &graph,
                clusters,
                &merge_options,
                &master_options,
                &guidance,
                &ptu_plan::escalation::EscalationOptions {
                    max_attempts: options.repartition_attempts,
                    attempt_time_ms: options.repartition_stage_ms,
                    max_attempt_time_ms: options.repartition_max_stage_ms,
                    monolithic_fallback: !options.no_monolithic_fallback,
                    ..Default::default()
                },
                if options.master_trace {
                    TraceLevel::Full
                } else {
                    TraceLevel::Summary
                },
            )
            .map_err(|e| e.to_string())?;
            let result = outcome.result.clone();
            repartition = Some(outcome);
            Ok(result)
        } else if options.master_trace {
            let mut events = Vec::new();
            let result = solve_sequential_guided(
                model.as_ref().unwrap(),
                &partitions,
                &master_options,
                &guidance,
                TraceLevel::Full,
                Some(&mut |event| events.push(event)),
            );
            master_events = Some(events);
            result
        } else {
            solve_sequential_guided(
                model.as_ref().unwrap(),
                &partitions,
                &master_options,
                &guidance,
                TraceLevel::Summary,
                None,
            )
        };
        Some(result.map_err(|e| e.to_string())?)
    } else {
        None
    };
    let solution = if options.solve {
        Some(solve_problem(model.as_ref().unwrap(), &budget).map_err(|err| err.to_string())?)
    } else {
        None
    };
    let interleaving = if options.interleave {
        let clusters = agglomeration
            .as_ref()
            .map(|a| &a.clusters)
            .unwrap_or(&micropartitions.as_ref().unwrap().clusters);
        let partitions = initial_partitions(&graph, clusters).map_err(|e| e.to_string())?;
        Some(
            solve_interleaved(
                model.as_ref().unwrap(),
                &partitions,
                &InterleavingOptions {
                    max_nodes: options.max_master_nodes,
                    max_generated: options.max_generated,
                    max_time_ms: options.master_time_limit_ms,
                    max_events: options.max_pocl_events,
                    max_frontier: options.max_pocl_frontier,
                    exactness: budget.exactness,
                },
            )
            .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    if options.analysis_json {
        #[derive(Serialize)]
        struct Record<'a> {
            analysis: &'a HtnAnalysis,
            graph: &'a InteractionGraph,
            #[serde(skip_serializing_if = "Option::is_none")]
            micropartitions: Option<&'a MicroPartitioning>,
            #[serde(skip_serializing_if = "Option::is_none")]
            agglomeration: Option<&'a Agglomeration>,
            #[serde(skip_serializing_if = "Option::is_none")]
            sequential: Option<&'a MasterResult>,
            #[serde(skip_serializing_if = "Option::is_none")]
            repartition: Option<&'a ptu_plan::escalation::EscalationResult>,
            #[serde(skip_serializing_if = "Option::is_none")]
            interleaving: Option<&'a InterleavingResult>,
            #[serde(skip_serializing_if = "Option::is_none")]
            solution: Option<&'a SolveResult>,
            #[serde(skip_serializing_if = "Option::is_none")]
            merge_policy: Option<&'a PolicyTrace>,
            #[serde(skip_serializing_if = "Option::is_none")]
            master_events: Option<&'a [MasterEvent]>,
            #[serde(skip_serializing_if = "Option::is_none")]
            interface_model_sha256: Option<&'a str>,
        }
        println!(
            "{}",
            serde_json::to_string(&Record {
                analysis: &analysis,
                graph: &graph,
                micropartitions: micropartitions.as_ref(),
                agglomeration: agglomeration.as_ref(),
                sequential: sequential.as_ref(),
                repartition: repartition.as_ref(),
                interleaving: interleaving.as_ref(),
                solution: solution.as_ref(),
                merge_policy: merge_policy.as_ref(),
                master_events: master_events.as_deref(),
                interface_model_sha256: interface_model_sha256.as_deref(),
            })
            .map_err(|err| err.to_string())?
        );
    } else {
        print_program(program);
        println!(
            "  HTN: {:?}, {:?}; {} reachable tasks, {} methods",
            analysis.signature.recursion,
            analysis.signature.ordering,
            analysis.signature.reachable_tasks,
            analysis.signature.reachable_methods
        );
        println!(
            "  Interaction graph: {} nodes, {} edges",
            graph.nodes.len(),
            graph.edges.len()
        );
        for view in [
            GraphView::Ordering,
            GraphView::Support,
            GraphView::Threat,
            GraphView::Resource,
            GraphView::Hierarchy,
            GraphView::SharedState,
        ] {
            println!("    {view:?}: {}", graph.edges_in_view(view).count());
        }
        for warning in &analysis.warnings {
            eprintln!("ptu-plan: analysis warning: {warning}");
        }
        if let Some(partitions) = &micropartitions {
            println!(
                "  Micro-partitions: {} (cut weight {:.3})",
                partitions.clusters.len(),
                partitions.cut_weight
            );
            for cluster in &partitions.clusters {
                println!("    Cluster {}: {:?}", cluster.id, cluster.nodes);
            }
        }
        if let Some(result) = &solution {
            println!(
                "  Solver: {:?} ({} expanded, {} generated)",
                result.status, result.statistics.expanded, result.statistics.generated
            );
            if let Some(certificate) = &result.certificate {
                for action in &certificate.actions {
                    println!(
                        "    ({}{})",
                        action.name,
                        if action.arguments.is_empty() {
                            String::new()
                        } else {
                            format!(" {}", action.arguments.join(" "))
                        }
                    );
                }
            }
        }
        if let Some(merged) = &agglomeration {
            println!(
                "  Agglomeration: {} clusters, {} merges, {} typed quotient edges",
                merged.clusters.len(),
                merged.merges.len(),
                merged.quotient_edges.len()
            );
        }
        if let Some(result) = &repartition {
            println!(
                "  Repartition escalation: {} attempts, {} total local calls, {} ms ({})",
                result.attempts.len(),
                result.work.local_calls,
                result.work.elapsed_ms,
                result.stop_reason
            );
        }
        if let Some(result) = &sequential {
            println!(
                "  Sequential master: {:?} ({} local calls, {} verified no-goods, {} active conflict cores)",
                result.status,
                result.statistics.local_calls,
                result.nogoods.len(),
                result.statistics.core_active
            );
            if let Some(certificate) = &result.certificate {
                println!(
                    "    {} actions; whole-problem replay passed",
                    certificate.actions.len()
                );
                for action in &certificate.actions {
                    println!(
                        "    ({}{})",
                        action.name,
                        if action.arguments.is_empty() {
                            String::new()
                        } else {
                            format!(" {}", action.arguments.join(" "))
                        }
                    );
                }
            }
        }
        if let Some(result) = &interleaving {
            println!(
                "  POCL interleaving master: {:?} ({} states, {} support choices, {} threat choices)",
                result.status,
                result.statistics.nodes,
                result.statistics.support_choices,
                result.statistics.threat_choices
            );
            if let Some(certificate) = &result.certificate {
                println!(
                    "    {} actions; whole-problem replay passed",
                    certificate.plan.actions.len()
                );
                for action in &certificate.plan.actions {
                    println!(
                        "    ({}{})",
                        action.name,
                        if action.arguments.is_empty() {
                            String::new()
                        } else {
                            format!(" {}", action.arguments.join(" "))
                        }
                    );
                }
            }
        }
    }
    if let Some(result) = solution
        && result.status != ProofStatus::Feasible
    {
        return Err(format!("local solver returned {:?}", result.status));
    }
    if let Some(result) = sequential
        && result.status != MasterStatus::Feasible
    {
        return Err(format!(
            "sequential master returned {:?}; this does not prove global infeasibility",
            result.status
        ));
    }
    if let Some(result) = interleaving
        && result.status != ProofStatus::Feasible
    {
        return Err(format!(
            "POCL interleaving master returned {:?}",
            result.status
        ));
    }
    Ok(())
}

fn processing(options: &Options) -> bool {
    options.analyze
        || options.analysis_json
        || options.solve
        || options.micropartition
        || options.agglomerate
        || options.sequential
        || options.interleave
}

fn print_program(program: &Program) {
    let domain = &program.domain;
    println!(
        "Domain {} ({} actions, {} compound tasks, {} methods)",
        domain.name,
        domain.actions.len(),
        domain.compound_tasks.len(),
        domain.methods.len()
    );
    if let Some(problem) = &program.problem {
        println!(
            "  Problem {} ({} objects)",
            problem.name,
            problem.objects.len()
        );
    }
}
