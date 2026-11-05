//! PANDA's cost-aware progression search, translated back to original HDDL,
//! verified by PANDA, and independently replayed through this project's solver.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::{Provenance, TrainingSample, make_sample, sha256};
use crate::parsing::parser::{ParserConfig, parse_program};
use crate::solver::*;

#[derive(Debug, Clone)]
pub struct PandaOptions {
    pub root: PathBuf,
    pub work: PathBuf,
    pub stage_seconds: u64,
    pub search_seconds: u64,
    pub memory_bytes: u64,
    pub grounding: GroundingOptions,
    pub validation_budget: SearchBudget,
}

impl Default for PandaOptions {
    fn default() -> Self {
        Self {
            root: PathBuf::from("target/panda"),
            work: PathBuf::from("target/panda-training"),
            stage_seconds: 60,
            search_seconds: 30,
            memory_bytes: 2 * 1024 * 1024 * 1024,
            grounding: GroundingOptions::default(),
            validation_budget: SearchBudget::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PandaEvidence {
    pub objective: String,
    pub engine_arguments: Vec<String>,
    pub tool_sha256: BTreeMap<String, String>,
    pub tool_revisions: BTreeMap<String, String>,
    pub domain_sha256: String,
    pub problem_sha256: String,
    pub domain_hddl: String,
    pub problem_hddl: String,
    pub reported_cost: usize,
    pub search_log: String,
    pub converted_plan: String,
    pub verification_log: String,
}

impl PandaEvidence {
    pub fn validate(&self, action_cost: usize) -> Result<()> {
        let prefix = [
            "-p", "-H", "zero", "-g", "action", "-w", "1", "-S", "42", "-t",
        ];
        if self.objective != "unit_cost_primitive_actions"
            || self.engine_arguments.len() != 12
            || !self
                .engine_arguments
                .iter()
                .take(10)
                .map(String::as_str)
                .eq(prefix)
            || self.engine_arguments[10]
                .parse::<u64>()
                .ok()
                .is_none_or(|x| x == 0)
        {
            bail!("PANDA run was not configured for unit-cost optimal search");
        }
        let log = strip_ansi(&self.search_log);
        require_solved(&log)?;
        for marker in [
            "Heuristic #0 = zero()",
            " - type: cost optimal",
            " - weight: 1",
            " - suboptimal: false",
            "- Status: Solved",
        ] {
            if !log.lines().any(|line| line.trim() == marker.trim()) {
                bail!("missing PANDA optimal-search evidence: {marker}");
            }
        }
        if log.contains("Status: Timeout")
            || reported_cost(&log)? != action_cost
            || self.reported_cost != action_cost
        {
            bail!("PANDA cost or completion evidence disagrees with the original action plan");
        }
        if !strip_ansi(&self.verification_log)
            .lines()
            .any(|line| line.trim() == "Plan verification result: true")
        {
            bail!("PANDA did not verify the original HDDL plan");
        }
        if primitive_plan(&self.converted_plan)?.len() != action_cost {
            bail!("converted plan cost mismatch");
        }
        if sha256(self.domain_hddl.as_bytes()) != self.domain_sha256
            || sha256(self.problem_hddl.as_bytes()) != self.problem_sha256
        {
            bail!("source HDDL checksum mismatch");
        }
        require_unit_cost(&self.domain_hddl, &self.problem_hddl)?;
        for name in ["parser", "grounder", "engine"] {
            if self
                .tool_sha256
                .get(name)
                .is_none_or(|hash| hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()))
            {
                bail!("missing PANDA binary checksum for {name}");
            }
        }
        Ok(())
    }
}

fn strip_ansi(text: &str) -> String {
    let mut clean = String::new();
    let mut escape = false;
    for c in text.chars() {
        if c == '\x1b' {
            escape = true;
        } else if escape {
            if c.is_ascii_alphabetic() {
                escape = false;
            }
        } else {
            clean.push(c);
        }
    }
    clean
}

fn require_unit_cost(domain: &str, problem: &str) -> Result<()> {
    for contents in [domain, problem] {
        let text = contents
            .lines()
            .map(|line| line.split(';').next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n")
            .to_ascii_lowercase();
        if [":functions", ":metric", ":action-costs", ":cost-bound"]
            .iter()
            .any(|token| text.contains(token))
        {
            bail!("optimal training import supports implicit unit action costs only");
        }
    }
    Ok(())
}

fn reported_cost(log: &str) -> Result<usize> {
    let line = log
        .lines()
        .rev()
        .find_map(|line| line.trim().strip_prefix("- Total costs of actions: "))
        .context("PANDA did not report an action cost")?;
    Ok(line.trim().parse()?)
}

fn require_solved(log: &str) -> Result<()> {
    let statuses: Vec<_> = log
        .lines()
        .filter_map(|line| line.trim().strip_prefix("- Status: "))
        .collect();
    if statuses != ["Solved"] {
        bail!(
            "PANDA search did not certify a solution: {}",
            if statuses.is_empty() {
                "missing status".into()
            } else {
                statuses.join(", ")
            }
        );
    }
    Ok(())
}

/// The converter writes original actions as `id name arg ...`, before `root`.
/// Never read the unconverted engine plan: it contains compiled helper actions.
pub fn primitive_plan(converted: &str) -> Result<Vec<GroundTask>> {
    let mut started = false;
    let mut seen = std::collections::BTreeSet::new();
    let mut actions = Vec::new();
    for line in converted
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        if line == "==>" {
            if started {
                bail!("multiple plan headers");
            }
            started = true;
            continue;
        }
        if !started {
            bail!("converted PANDA plan has no header");
        }
        if line.starts_with("root ") || line == "root" {
            return Ok(actions);
        }
        let mut tokens = line.split_whitespace();
        let id: usize = tokens.next().context("missing action ID")?.parse()?;
        if !seen.insert(id) {
            bail!("duplicate primitive occurrence ID");
        }
        let name = tokens.next().context("missing primitive task name")?;
        if name.contains('[') || name.starts_with("__") {
            bail!("plan still contains compiled PANDA task names");
        }
        actions.push(GroundTask {
            name: name.into(),
            arguments: tokens.map(str::to_string).collect(),
        });
    }
    bail!("converted PANDA plan has no root line")
}

fn read_log(path: &Path) -> Result<String> {
    if fs::metadata(path)?.len() > 16 * 1024 * 1024 {
        bail!("PANDA log exceeds 16 MiB: {}", path.display());
    }
    Ok(fs::read_to_string(path)?)
}

fn stage(tool: &Path, args: &[String], log: &Path, seconds: u64, memory: u64) -> Result<()> {
    if seconds == 0 || memory == 0 {
        bail!("PANDA time and memory limits must be positive");
    }
    let output = File::create(log)?;
    let status = Command::new("timeout")
        .args([
            "--foreground",
            "--signal=TERM",
            "--kill-after=2s",
            &format!("{seconds}s"),
        ])
        .arg("prlimit")
        .arg(format!("--as={memory}"))
        .arg("--")
        .arg(tool)
        .args(args)
        .stdout(output.try_clone()?)
        .stderr(output)
        .status()
        .with_context(|| format!("running {}", tool.display()))?;
    if !status.success() {
        bail!(
            "PANDA stage {} exited {status}; see {}",
            tool.display(),
            log.display()
        );
    }
    Ok(())
}

fn revision(path: &Path) -> String {
    Command::new("git")
        .arg("-C")
        .arg(path)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

pub fn collect(
    domain: &Path,
    problem: &Path,
    parser: &ParserConfig,
    options: &PandaOptions,
) -> Result<TrainingSample> {
    let domain = fs::canonicalize(domain)?;
    let problem = fs::canonicalize(problem)?;
    let domain_hddl = fs::read_to_string(&domain)?;
    let problem_hddl = fs::read_to_string(&problem)?;
    require_unit_cost(&domain_hddl, &problem_hddl)?;
    let program = parse_program(parser, &domain, Some(&problem)).map_err(anyhow::Error::msg)?;
    // Validate supported exact semantics before spending external search time.
    let model = ground_reachable_program(&program, &options.grounding)?;
    let key = sha256(&serde_json::to_vec(&(&domain_hddl, &problem_hddl))?);
    let work = options.work.join(key);
    fs::create_dir_all(&work)?;
    let work = fs::canonicalize(work)?;
    let root = fs::canonicalize(&options.root)?;
    let tools = [
        root.join("pandaPIparser/pandaPIparser"),
        root.join("pandaPIgrounder/pandaPIgrounder"),
        root.join("pandaPIengine/build/pandaPIengine"),
    ];
    let str_path = |p: &Path| p.to_string_lossy().into_owned();
    let htn = work.join("input.htn");
    let sas = work.join("input.sas");
    let raw = work.join("search.log");
    let converted = work.join("plan.txt");
    stage(
        &tools[0],
        &[
            "-C".into(),
            str_path(&domain),
            str_path(&problem),
            str_path(&htn),
        ],
        &work.join("parse.log"),
        options.stage_seconds,
        options.memory_bytes,
    )?;
    stage(
        &tools[1],
        &["-q".into(), str_path(&htn), str_path(&sas)],
        &work.join("ground.log"),
        options.stage_seconds,
        options.memory_bytes,
    )?;
    let mut arguments: Vec<String> = [
        "-p", "-H", "zero", "-g", "action", "-w", "1", "-S", "42", "-t",
    ]
    .map(str::to_string)
    .into();
    arguments.extend([options.search_seconds.to_string(), str_path(&sas)]);
    stage(
        &tools[2],
        &arguments,
        &raw,
        options.search_seconds + 3,
        options.memory_bytes,
    )?;
    let search_log = read_log(&raw)?;
    require_solved(&strip_ansi(&search_log))?;
    stage(
        &tools[0],
        &[
            "-C".into(),
            "-c".into(),
            str_path(&raw),
            str_path(&converted),
        ],
        &work.join("convert.log"),
        options.stage_seconds,
        options.memory_bytes,
    )?;
    stage(
        &tools[0],
        &[
            "-C".into(),
            "-v".into(),
            str_path(&domain),
            str_path(&problem),
            str_path(&converted),
        ],
        &work.join("verify.log"),
        options.stage_seconds,
        options.memory_bytes,
    )?;
    let mut hashes = BTreeMap::new();
    let mut revisions = BTreeMap::new();
    for (i, name) in ["parser", "grounder", "engine"].into_iter().enumerate() {
        hashes.insert(name.into(), sha256(&fs::read(&tools[i])?));
        revisions.insert(name.into(), revision(&root.join(format!("pandaPI{name}"))));
    }
    let evidence = PandaEvidence {
        objective: "unit_cost_primitive_actions".into(),
        engine_arguments: arguments,
        tool_sha256: hashes,
        tool_revisions: revisions,
        domain_sha256: sha256(domain_hddl.as_bytes()),
        problem_sha256: sha256(problem_hddl.as_bytes()),
        domain_hddl,
        problem_hddl,
        reported_cost: reported_cost(&search_log)?,
        search_log,
        converted_plan: read_log(&converted)?,
        verification_log: read_log(&work.join("verify.log"))?,
    };
    let plan = primitive_plan(&evidence.converted_plan)?;
    evidence.validate(plan.len())?;
    let result = solve_with_plan(&model, &plan, &options.validation_budget)?;
    let certificate = result.certificate.context(format!(
        "local HTN replay did not certify PANDA plan: {:?}",
        result.status
    ))?;
    make_sample(
        program,
        &model,
        certificate,
        Provenance::PandaOptimal {
            domain_path: domain,
            problem_path: problem,
            evidence: Box::new(evidence),
        },
    )
}
