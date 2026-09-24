// Release builds on Windows are GUI apps: no console window beside the UI.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// NAM Mixer desktop shell.
//
// This is a thin Tauri wrapper, not a reimplementation: it starts the
// existing Flask backend (app.py) as a child process on a free localhost
// port, waits for it to answer /api/health, then points a native window at
// that URL. The web UI served is byte-for-byte the same as `python app.py`
// in a browser -- see CLAUDE.md's desktop-architecture note. The backend
// child is a REAL separate process; it is never asked to relaunch this GUI
// executable (that relaunch-loop bug is what killed the previous
// pywebview-based desktop app -- see docs/history/).
use std::io::Read;
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_dialog::DialogExt;

struct BackendProcess(Mutex<Option<Child>>);

// A native confirm command (tauri-plugin-dialog's blocking_show(), called
// from an async command the same way the save commands below call
// blocking_save_file()) was tried here for delete confirmations and
// removed: it hung indefinitely with no dialog and no error, silently
// breaking every delete button. window.confirm() is what static/app.js
// uses instead -- WKWebView (Tauri's engine on macOS) implements it
// natively via its own UI delegate, no Rust command needed.

// WKWebView (Tauri's engine on macOS, and WebView2 has similar quirks) does
// not honor the HTML `download` attribute -- clicking a download link just
// silently does nothing instead of showing a save dialog. The frontend
// detects it is running inside Tauri and calls these commands instead of
// relying on the browser's native download handling.
// Every command below returns the chosen path on success (never just `()`)
// so the frontend can tell the user exactly where the file went, rather
// than a save silently succeeding with no feedback -- see
// desktopSaveResult()/the single call site each of these has in app.js.
// "save cancelled" is a distinguished error string both here and in
// app.js's handling of it: cancelling the OS dialog is not a failure.
#[tauri::command]
async fn save_file_from_url(app: tauri::AppHandle, url: String, filename: String) -> Result<String, String> {
    let bytes = ureq::get(&url)
        .call()
        .map_err(|e| format!("download request failed: {e}"))?
        .into_reader()
        .bytes()
        .collect::<Result<Vec<u8>, _>>()
        .map_err(|e| format!("failed reading response body: {e}"))?;
    save_bytes_via_dialog(app, filename, bytes)
}

#[tauri::command]
async fn save_bytes(app: tauri::AppHandle, filename: String, data_base64: String) -> Result<String, String> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data_base64)
        .map_err(|e| format!("invalid base64 payload: {e}"))?;
    save_bytes_via_dialog(app, filename, bytes)
}

fn save_bytes_via_dialog(app: tauri::AppHandle, filename: String, bytes: Vec<u8>) -> Result<String, String> {
    let path = app
        .dialog()
        .file()
        .set_file_name(&filename)
        .blocking_save_file()
        .ok_or_else(|| "save cancelled".to_string())?;
    let path = path.into_path().map_err(|e| format!("invalid save path: {e}"))?;
    std::fs::write(&path, bytes).map_err(|e| format!("failed writing {}: {e}", path.display()))?;
    Ok(path.display().to_string())
}

// WKWebView (and WebView2) do not open a real new window/tab for
// `target="_blank"` links the way a normal browser does -- clicking one
// just silently does nothing. The frontend detects it is running inside
// Tauri and routes external links through this instead, which hands the
// URL to the OS's own "open" mechanism (the same thing a browser's "open
// link" ultimately does), so it opens in the user's actual default browser.
// Restricted to http(s) specifically -- never used to open an arbitrary
// local path or shell out to anything else.
#[tauri::command]
fn open_external_url(url: String) -> Result<(), String> {
    if !url.starts_with("https://") && !url.starts_with("http://") {
        return Err("only http(s) URLs can be opened this way".to_string());
    }
    if url.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err("URL contains whitespace or control characters".to_string());
    }
    let result = if cfg!(target_os = "macos") {
        Command::new("open").arg(&url).status()
    } else if cfg!(target_os = "windows") {
        // Never `cmd /C start`: cmd treats & | ^ % in the URL as shell syntax,
        // so a link could run another command. FileProtocolHandler receives
        // the URL as a plain argument.
        Command::new("rundll32").args(["url.dll,FileProtocolHandler", &url]).status()
    } else {
        Command::new("xdg-open").arg(&url).status()
    };
    match result {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!("opener exited with status {status}")),
        Err(e) => Err(format!("failed to launch opener: {e}")),
    }
}

fn find_free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("failed to bind an ephemeral port")
        .local_addr()
        .expect("failed to read local addr")
        .port()
}

fn repo_root() -> std::path::PathBuf {
    // In dev, this binary runs from desktop/src-tauri/target/debug/.
    // Walk up to the repo root (the directory containing app.py).
    let mut dir = std::env::current_exe().expect("failed to locate current exe");
    loop {
        if dir.join("app.py").is_file() {
            return dir;
        }
        if !dir.pop() {
            panic!("could not locate repo root (app.py) above the executable");
        }
    }
}

// A GUI app launched by double-click/Finder/`open` inherits launchd's
// minimal PATH (just /usr/bin:/bin:/usr/sbin:/sbin on macOS) -- NOT the
// user's real shell PATH from their .zshrc/.bashrc. This silently breaks
// anything the backend tries to find via PATH: a real Python 3.10+ for
// local A2 training's own venv setup (which only ever found macOS's
// ancient bundled Python 3.9 as a result, always refusing with a version
// error no matter what newer Python the user actually has installed), and
// the standalone `kaggle` CLI binary lookup. Running the user's own login
// shell once at startup to ask what PATH it would actually set is the
// standard fix for this whole class of problem (the same one Electron
// apps reach for via the "fix-path" package) -- inherited by every
// subprocess the backend itself spawns afterward, so this one fix also
// covers Kaggle CLI discovery, not just Python version detection.
fn user_shell_path() -> Option<String> {
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".to_string());
    // `env` prints the exported PATH colon-joined in every shell (fish's
    // "$PATH" would be space-separated), and taking the last PATH= line
    // ignores anything the user's rc files print first.
    let mut child = Command::new(&shell)
        .args(["-ilc", "env"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    // A user's shell rc file could in principle hang (a blocking network
    // call, a prompt waiting on input that never arrives) -- bounded so
    // that can never freeze the whole app at launch; falls back to
    // whatever PATH this process already has, same as before this fix.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match child.try_wait().ok()? {
            Some(status) => {
                if !status.success() {
                    return None;
                }
                let mut stdout = String::new();
                child.stdout.take()?.read_to_string(&mut stdout).ok()?;
                return path_from_env_output(&stdout);
            }
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                return None;
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

/// The PATH value from `env` output: the last `PATH=` line, so text printed
/// by the user's rc files before it is ignored.
fn path_from_env_output(stdout: &str) -> Option<String> {
    let path = stdout.lines().rev().find_map(|line| line.strip_prefix("PATH="))?.trim().to_string();
    (!path.is_empty()).then_some(path)
}

// The bundled backend (packaging/backend/nam_mixer_backend.spec, built with
// PyInstaller) is copied into the app bundle as a Tauri RESOURCE, not a
// sidecar: Tauri's sidecar mechanism expects one standalone executable file,
// but PyInstaller's onedir output is an exe plus a required sibling
// `_internal/` folder of its own -- onedir was chosen over onefile so
// DI_DIR/nam_render's relative-path lookups inside app.py resolve against a
// stable directory instead of a fresh temp-extraction path on every launch.
// Bundling the whole folder as a resource and spawning the exe inside it
// directly sidesteps that mismatch entirely.
fn bundled_backend_exe(app: &tauri::App) -> Option<std::path::PathBuf> {
    let exe_name = if cfg!(windows) { "nam-mixer-backend.exe" } else { "nam-mixer-backend" };
    let candidate = app
        .path()
        .resource_dir()
        .ok()?
        .join("nam-mixer-backend")
        .join(exe_name);
    candidate.is_file().then_some(candidate)
}

// In production this is the bundled backend's own executable (no
// interpreter needed at all -- see bundled_backend_exe above); in dev
// (`cargo build`/`cargo run`, no bundled resource present yet) this falls
// back to shelling out to a real `python3 app.py` exactly as before.
// Either way this is a REAL SEPARATE process, never asked to relaunch this
// GUI executable itself (that relaunch-loop bug is what killed the
// previous pywebview-based desktop app -- see docs/history/).
fn spawn_backend(app: &tauri::App, port: u16, data_dir: &std::path::Path) -> Child {
    let mut command = match bundled_backend_exe(app) {
        Some(exe) => {
            let mut cmd = Command::new(&exe);
            let exe_dir = exe.parent().unwrap();
            cmd.current_dir(exe_dir);
            // Local A2 training's OWN separate venv subprocess (see
            // hybrid/training/local_training.py) needs real loose scripts/hybrid
            // source + requirements-training.txt on disk -- bundled at
            // training_support/ alongside this exe (see the .spec file's
            // datas), never inside this frozen process's own packed
            // modules. Without this, LocalTrainingManager falls back to
            // TRAINING_ROOT=BASE_DIR (this exe's own directory), which
            // has none of that.
            cmd.env("NAM_MIXER_TRAINING_ROOT", exe_dir.join("_internal").join("training_support"));
            cmd
        }
        None => {
            let root = repo_root();
            let python = std::env::var("NAM_MIXER_PYTHON").unwrap_or_else(|_| "python3".to_string());
            let mut cmd = Command::new(python);
            cmd.arg("app.py").current_dir(&root);
            cmd
        }
    };
    if let Some(path) = user_shell_path() {
        command.env("PATH", path);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW); // the console backend gets no window of its own
    }
    command
        .env("PORT", port.to_string())
        .env("NAM_MIXER_DATA_DIR", data_dir)
        .env("NAM_MIXER_ENV_FILE", data_dir.join(".env"))
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("failed to start the NAM Mixer backend (is python3 on PATH? -- see NAM_MIXER_PYTHON)")
}

/// Ok once /api/health answers; Err with the reason if the backend exits
/// first (reported at once rather than after the full timeout) or times out.
fn wait_for_health(port: u16, timeout: Duration, child: &mut Child) -> Result<(), String> {
    let url = format!("http://127.0.0.1:{port}/api/health");
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!("The local backend exited during startup ({status})."));
        }
        if let Ok(resp) = ureq::get(&url).timeout(Duration::from_millis(500)).call() {
            if resp.status() == 200 {
                return Ok(());
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    Err(format!("The local backend did not become ready within {} seconds.", timeout.as_secs()))
}

/// Ask the backend to exit (SIGTERM lets it stop its own local-training
/// subprocess, which runs in a separate session), then force it if needed.
fn stop_backend(mut child: Child) {
    #[cfg(unix)]
    {
        let _ = Command::new("kill").args(["-TERM", &child.id().to_string()]).status();
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn percent_encode(text: &str) -> String {
    text.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![save_file_from_url, save_bytes, open_external_url])
        .manage(BackendProcess(Mutex::new(None)))
        .setup(|app| {
            let port = find_free_port();
            // A real per-OS user data directory (e.g. ~/Library/Application
            // Support/com.nammixer.desktop on macOS) -- NEVER the bundle's
            // own (read-only, ephemeral-on-reinstall) directory. app.py
            // already creates this path itself (WORK_DIR.mkdir(...)) once
            // told about it via NAM_MIXER_DATA_DIR.
            let data_dir = app.path().app_data_dir().expect("failed to resolve app data dir");
            let mut child = spawn_backend(app, port, &data_dir);
            let ready = wait_for_health(port, Duration::from_secs(30), &mut child);
            *app.state::<BackendProcess>().0.lock().unwrap() = Some(child);
            let url = match ready {
                Ok(()) => format!("http://127.0.0.1:{port}/"),
                // A blank about: page looks like a frozen launch and gives
                // the user no recovery path. Keep the failure in the same
                // window and make it explicit instead.
                Err(reason) => format!(
                    "data:text/html,{}",
                    percent_encode(&format!(
                        "<html><body style=\"font-family:sans-serif;padding:2rem\"><h1>NAM Mixer could not start</h1><p>{reason} Quit and try again.</p></body></html>"
                    ))
                ),
            };
            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url.parse().unwrap()))
                .title("NAM Mixer")
                .inner_size(1280.0, 860.0)
                .build()?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building NAM Mixer desktop shell")
        .run(|app_handle, event| {
            // Covers every quit path (window close, Cmd+Q, dock quit) --
            // whichever fires first kills the backend exactly once.
            if let tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit = event {
                let state = app_handle.state::<BackendProcess>();
                let taken = state.0.lock().unwrap().take();
                if let Some(child) = taken {
                    stop_backend(child);
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_comes_from_env_output_even_after_rc_file_greetings() {
        let zsh = "Welcome back!\nHOME=/Users/x\nPATH=/opt/homebrew/bin:/usr/bin:/bin\nSHELL=/bin/zsh\n";
        assert_eq!(path_from_env_output(zsh).as_deref(), Some("/opt/homebrew/bin:/usr/bin:/bin"));
        // fish exports PATH colon-joined through `env` too
        let fish = "PATH=/opt/homebrew/bin:/usr/bin\nfish_greeting=hi\n";
        assert_eq!(path_from_env_output(fish).as_deref(), Some("/opt/homebrew/bin:/usr/bin"));
        assert_eq!(path_from_env_output("no path here\n"), None);
    }

    #[test]
    fn startup_error_page_is_fully_encoded() {
        let encoded = percent_encode("<p>exited (exit status: 1). Quit</p>");
        assert!(!encoded.contains('<') && !encoded.contains(' ') && !encoded.contains('('));
        assert!(encoded.starts_with("%3Cp%3Eexited"));
    }
}

