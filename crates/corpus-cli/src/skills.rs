//! What a checkout can do beyond what the prompt says. A skill is a directory holding a
//! `SKILL.md` and, usually, the package it describes; every skills root is on the
//! interpreter's path, so the directory name is what `import` takes. The prompt carries the
//! description and the path, and the instructions themselves are read by the model with the
//! tool it already has, in the session where they are needed.

use std::path::{Path, PathBuf};

/// The one file that makes a directory a skill rather than a package that happens to sit
/// there.
const MANIFEST: &str = "SKILL.md";

/// Where a project and a person each keep their own, under the directory corpus already
/// works out of.
const OWN: &str = ".corpus/skills";

/// Every place a session looks, most specific first: this project's, this person's, and the
/// ones the checkout shipped with. The order is the whole of the precedence — a name found
/// twice is the first one — and it is the order the interpreter's path is given, so the
/// skill the prompt describes is the module an `import` reaches.
pub fn roots(kernel_dir: &Path) -> Vec<PathBuf> {
    // Absolute, because these go on `PYTHONPATH`: a relative entry there is resolved against
    // whatever directory the interpreter is in when the import happens, and a cell can move.
    [std::env::current_dir().ok(), std::env::home_dir()]
        .into_iter()
        .flatten()
        .map(|dir| dir.join(OWN))
        .chain([kernel_dir.join("skills")])
        .collect()
}

pub struct Skill {
    name: String,
    description: String,
    manifest: PathBuf,
}

/// Every skill under `roots`, by name. The order is the prompt's order, and a prompt that
/// reorders itself between runs is a prompt no cache can be kept for. A root that is not
/// there is no skills rather than a failure: most sessions have only the one the checkout
/// shipped with, and a checkout without even that still has a session.
pub fn discover(roots: &[PathBuf]) -> Vec<Skill> {
    let mut skills: Vec<Skill> = Vec::new();
    for root in roots {
        // Resolved here, because the path is written into the prompt for the model to open:
        // whatever the caller was holding — a relative path, or one written through `..` — is
        // not something a cell can hand to `open()` and be sure of.
        let Ok(root) = root.canonicalize() else {
            continue;
        };
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        let mut found: Vec<Skill> = entries
            .flatten()
            .filter_map(|entry| read(&entry.path()))
            .collect();
        // The earlier root keeps the name, which is what lets a project ship its own version
        // of a skill the checkout has. Two roots resolving to one directory — corpus run out
        // of a home directory, say — falls out here as the same rule.
        found.retain(|skill| !skills.iter().any(|had| had.name == skill.name));
        skills.append(&mut found);
    }
    skills.sort_by(|one, other| one.name.cmp(&other.name));
    skills
}

fn read(dir: &Path) -> Option<Skill> {
    let manifest = dir.join(MANIFEST);
    let text = std::fs::read_to_string(&manifest).ok()?;
    // ponytail: the directory name is handed straight to `import`, and the skills root sits
    // ahead of site-packages, so a skill sharing a name with a package it installs would
    // shadow it outright — a directory called `docx` here hides python-docx and its
    // `Document` with it. Naming them apart is the whole of the guard. The upgrade, the day
    // a name has to be free of what python will take, is to put each skill's own directory
    // on the path rather than their parent, which lets the package inside be called
    // something else entirely.
    let name = dir.file_name()?.to_string_lossy().into_owned();

    let mut description = None;
    for (key, value) in frontmatter(&text) {
        match key {
            "description" => description = Some(value.to_string()),
            // The single source of truth is the directory, because that is what `import`
            // will accept; a `name` that says otherwise is a typo worth hearing about.
            "name" if value != name => {
                eprintln!("skill {name} calls itself `{value}`; the directory name is what imports")
            }
            _ => {}
        }
    }
    match description.filter(|line| !line.is_empty()) {
        Some(description) => Some(Skill {
            name,
            description,
            manifest,
        }),
        // Nothing in the prompt would say what this is for, so the model would never open
        // it: skipping it says so out loud rather than leaving it silently unreachable.
        None => {
            eprintln!("skill {name} has no description and would be invisible; skipping it");
            None
        }
    }
}

/// The `key: value` lines a manifest opens with, if it opens with any. Parsed here rather
/// than with a yaml library: two fields of flat text are not worth a dependency, and a
/// manifest that needs more than this is a manifest that has outgrown being read at a
/// glance.
fn frontmatter(text: &str) -> impl Iterator<Item = (&str, &str)> {
    text.strip_prefix("---")
        .into_iter()
        .flat_map(|body| body.lines().skip(1).take_while(|line| line.trim() != "---"))
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.trim(), value.trim().trim_matches(['"', '\'']).trim()))
}

/// What the prompt says about them: enough to know which one a task calls for, and where
/// to read the rest. Nothing at all when there are none, so the prompt of a checkout
/// without skills is the prompt as it was.
pub fn block(skills: &[Skill]) -> String {
    if skills.is_empty() {
        return String::new();
    }
    let mut block = String::from(
        "\n
Not all of what you can do is written here. Each file below holds what this prompt does not: \
what its subject is for, where the library underneath it catches people out, and whether there \
is a module of that name to import or only the instructions themselves. Read the one your task \
calls for before you start — `print(open(path).read())` — and write against what it tells you \
rather than against what the name suggests.\n",
    );
    for skill in skills {
        block.push_str(&format!(
            "\n  {} — {}\n    {}",
            skill.name,
            skill.description,
            skill.manifest.display()
        ));
    }
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("corpus-skills-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn put(root: &Path, name: &str, manifest: Option<&str>) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        if let Some(text) = manifest {
            std::fs::write(dir.join(MANIFEST), text).unwrap();
        }
    }

    #[test]
    fn a_skill_is_its_directory_its_description_and_its_manifest() {
        let root = scratch("found");
        put(
            &root,
            "documents",
            Some("---\nname: documents\ndescription: \"writes documents\"\n---\n\nthe body\n"),
        );

        let found = discover(&[root]);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "documents");
        assert_eq!(found[0].description, "writes documents");
        assert!(found[0].manifest.is_absolute());
        assert!(found[0].manifest.ends_with("documents/SKILL.md"));
        assert!(block(&found).contains(&found[0].manifest.display().to_string()));
    }

    #[test]
    fn a_directory_without_a_manifest_is_not_a_skill() {
        let root = scratch("bare");
        put(&root, "__pycache__", None);
        assert!(discover(&[root]).is_empty());
    }

    #[test]
    fn a_skill_without_a_description_is_skipped() {
        let root = scratch("mute");
        put(&root, "shell", Some("---\nname: shell\n---\n\nthe body\n"));
        assert!(discover(&[root]).is_empty());
    }

    #[test]
    fn the_order_is_the_same_whatever_the_directory_hands_back() {
        let root = scratch("order");
        for name in ["documents", "charts", "shell"] {
            put(
                &root,
                name,
                Some(&format!("---\ndescription: {name}\n---\n")),
            );
        }
        let names: Vec<String> = discover(&[root])
            .into_iter()
            .map(|found| found.name)
            .collect();
        assert_eq!(names, ["charts", "documents", "shell"]);
    }

    /// The point of having more than one root: what the checkout ships is the fallback, and
    /// a project that has its own version of a skill is the one the session is told about.
    #[test]
    fn the_earlier_root_keeps_the_name_and_the_later_one_adds_what_it_alone_has() {
        let project = scratch("project");
        let builtin = scratch("builtin");
        put(
            &project,
            "documents",
            Some("---\ndescription: the project's own\n---\n"),
        );
        for name in ["documents", "charts"] {
            put(
                &builtin,
                name,
                Some(&format!("---\ndescription: the checkout's {name}\n---\n")),
            );
        }

        let found = discover(&[project.clone(), builtin.clone()]);
        let named: Vec<(&str, &str)> = found
            .iter()
            .map(|skill| (skill.name.as_str(), skill.description.as_str()))
            .collect();
        assert_eq!(
            named,
            [
                ("charts", "the checkout's charts"),
                ("documents", "the project's own")
            ]
        );
        // And the manifest the prompt names is the one that won, since that is the file the
        // model is about to open.
        assert!(
            found[1]
                .manifest
                .starts_with(project.canonicalize().unwrap())
        );
    }

    #[test]
    fn a_root_that_is_not_there_is_no_skills_and_no_block() {
        let missing =
            std::env::temp_dir().join(format!("corpus-skills-{}-none", std::process::id()));
        assert!(discover(&[missing]).is_empty());
        assert!(block(&[]).is_empty());
    }

    /// A person's and a project's, then the checkout's — and all three absolute, because
    /// they are handed to an interpreter that can be told to change directory.
    #[test]
    fn the_roots_are_the_project_the_person_and_the_checkout_in_that_order() {
        let roots = roots(Path::new("/opt/corpus/kernel"));
        assert_eq!(
            roots.last().unwrap(),
            Path::new("/opt/corpus/kernel/skills")
        );
        assert!(roots.iter().all(|root| root.is_absolute()));
        assert_eq!(
            roots[0],
            std::env::current_dir().unwrap().join(OWN),
            "the project's own comes first"
        );
    }
}
