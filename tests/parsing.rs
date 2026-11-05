use std::path::{Path, PathBuf};
use std::process::Command;

use ptu_plan::parsing::json::{TaskOrdering, decode_program};
use ptu_plan::parsing::parser::{ParserConfig, parse_program, read_program_json};

// A concurrent spawn can briefly inherit another thread's writable fixture FD,
// causing ETXTBSY when that fixture is executed. Keep fixture writes and child
// launches in these tests in one critical section.
static PROCESS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn decodes_optional_fields_and_task_orderings() {
    let minimal = read_program_json(&fixture("minimal.json")).unwrap();
    assert_eq!(minimal.domain.types, None);
    assert_eq!(minimal.domain.constants, None);
    assert_eq!(minimal.problem, None);

    let program = read_program_json(&fixture("program.json")).unwrap();
    assert_eq!(program.domain.name, "transport");
    assert_eq!(program.problem.unwrap().name, "p1");
    assert_eq!(
        program.domain.methods[0].tn.orderings,
        TaskOrdering::Partial(vec![("t1".into(), "t2".into())])
    );
    assert_eq!(program.domain.methods[1].tn.orderings, TaskOrdering::Total);
}

#[test]
fn json_input_does_not_launch_parser() {
    let config = ParserConfig {
        executable: PathBuf::from("/nonexistent/parser"),
    };
    let program = parse_program(&config, &fixture("minimal.json"), None).unwrap();
    assert_eq!(program.domain.name, "minimal");
    assert!(parse_program(&config, &fixture("minimal.json"), Some(Path::new("p.hddl"))).is_err());
    assert!(decode_program(b"{invalid}").is_err());
}

#[test]
fn cli_loads_json_without_parser_and_rejects_extra_problem() {
    let _guard = PROCESS_TEST_LOCK.lock().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ptu-plan"))
        .arg(fixture("minimal.json"))
        .env("HDDL_PARSER", "/nonexistent/parser")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Domain minimal"));

    let output = Command::new(env!("CARGO_BIN_EXE_ptu-plan"))
        .arg(fixture("minimal.json"))
        .arg("p.hddl")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("already contains its problem"));
}

#[cfg(unix)]
fn fake_parser(script: &str) -> (tempfile::TempDir, ParserConfig) {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempfile::tempdir().unwrap();
    let executable = directory.path().join("fake parser");
    std::fs::write(&executable, script).unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
    (directory, ParserConfig { executable })
}

#[cfg(unix)]
#[test]
fn invokes_convert_with_intact_paths_and_decodes_status_prefix() {
    let _guard = PROCESS_TEST_LOCK.lock().unwrap();
    let json = std::fs::read_to_string(fixture("program.json")).unwrap();
    let script = format!(
        r#"#!/bin/sh
[ "$#" -eq 6 ] && [ "$1" = convert ] &&
[ "$2" = 'domain with spaces.hddl' ] && [ "$3" = -p ] &&
[ "$4" = 'problem with spaces.hddl' ] && [ "$5" = --to ] &&
[ "$6" = json ] || exit 9
printf '[Ok]\n'
cat <<'JSON'
{json}
JSON
"#
    );
    let (_directory, config) = fake_parser(&script);
    let program = parse_program(
        &config,
        Path::new("domain with spaces.hddl"),
        Some(Path::new("problem with spaces.hddl")),
    )
    .unwrap();
    assert_eq!(program.domain.name, "transport");
    assert_eq!(program.problem.unwrap().name, "p1");
}

#[cfg(unix)]
#[test]
fn propagates_parser_errors_even_when_exit_status_is_zero() {
    let _guard = PROCESS_TEST_LOCK.lock().unwrap();
    for exit in [0, 7] {
        let (_directory, config) = fake_parser(&format!(
            "#!/bin/sh\nprintf 'invalid HDDL' >&2\nexit {exit}\n"
        ));
        let error = parse_program(&config, Path::new("domain.hddl"), None).unwrap_err();
        assert!(error.contains("invalid HDDL"), "{error}");
        if exit == 0 {
            assert!(error.contains("did not produce JSON"), "{error}");
        } else {
            assert!(error.contains("status 7"), "{error}");
        }
    }
}
