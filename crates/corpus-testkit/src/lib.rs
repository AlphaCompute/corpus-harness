//! A scripted OpenAI-compatible endpoint: one canned response per request, in order.

use std::sync::{Arc, Mutex};

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Where the kernel's python shim lives. Every crate in the workspace sits the same
/// distance from it, so one spelling of the path serves all of them.
pub fn kernel_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernel")
}

pub struct Endpoint {
    /// Hand this to `Provider::new`.
    pub url: String,
    seen: Arc<Mutex<Vec<String>>>,
}

impl Endpoint {
    /// The request bodies the model actually received, in order.
    pub fn requests(&self) -> Vec<String> {
        self.seen.lock().unwrap().clone()
    }
}

/// Serves one canned response per request, in order.
pub async fn serve(responses: Vec<Vec<u8>>) -> Endpoint {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    tokio::spawn(async move {
        let mut responses = responses.into_iter().peekable();
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let (head, body) = drain_request(&mut socket).await;
            // A real provider answers its model listing whether or not anyone scripted
            // one, and callers ask for it to learn the context window. A script that
            // opens with a listing means the listing itself is what is under test.
            let scripted_listing = responses
                .peek()
                .is_some_and(|next| String::from_utf8_lossy(next).contains(r#""object":"list""#));
            if head.starts_with("GET /models") && !scripted_listing {
                let _ = socket
                    .write_all(&json(r#"{"object":"list","data":[]}"#))
                    .await;
                let _ = socket.flush().await;
                continue;
            }
            let Some(response) = responses.next() else {
                return;
            };
            recorder.lock().unwrap().push(body);
            if response.is_empty() {
                // An empty response means: take the request and never answer it.
                tokio::time::sleep(std::time::Duration::from_secs(120)).await;
                continue;
            }
            let _ = socket.write_all(&response).await;
            let _ = socket.flush().await;
        }
    });
    Endpoint {
        url: format!("http://{addr}"),
        seen,
    }
}

/// The request line and headers, then the body it announced. Read a line at a time rather
/// than a byte at a time: a header block is a few hundred bytes, and every test request
/// pays for however many syscalls that takes.
async fn drain_request(socket: &mut TcpStream) -> (String, String) {
    let mut reader = tokio::io::BufReader::new(socket);
    let mut head = Vec::new();
    loop {
        let mut line = Vec::new();
        match reader.read_until(b'\n', &mut line).await {
            Ok(0) | Err(_) => return (String::new(), String::new()),
            Ok(_) => {}
        }
        let blank = line == b"\r\n" || line == b"\n";
        head.extend_from_slice(&line);
        if blank {
            break;
        }
    }
    let length: usize = String::from_utf8_lossy(&head)
        .to_lowercase()
        .split("content-length:")
        .nth(1)
        .and_then(|rest| rest.split("\r\n").next())
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(0);
    // Off the reader, not the socket: the buffer it filled looking for the blank line has
    // the first of the body in it already.
    let mut body = vec![0u8; length];
    let _ = reader.read_exact(&mut body).await;
    (
        String::from_utf8_lossy(&head).into_owned(),
        String::from_utf8_lossy(&body).into_owned(),
    )
}

fn http(status: &str, content_type: &str, body: &str) -> Vec<u8> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    [head.as_bytes(), body.as_bytes()].concat()
}

pub fn sse(events: &[&str]) -> Vec<u8> {
    let mut body = String::new();
    for event in events {
        body.push_str(&format!("data: {event}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    http("200 OK", "text/event-stream", &body)
}

pub fn json(body: &str) -> Vec<u8> {
    http("200 OK", "application/json", body)
}

/// A provider that turned the request down, with whatever it said about why.
pub fn refused(status: &str, body: &str) -> Vec<u8> {
    http(status, "application/json", body)
}

/// Announces more body than it sends, then hangs up: a stream that broke mid-answer.
pub fn truncated(events: &[&str]) -> Vec<u8> {
    let mut full = sse(events);
    full.truncate(full.len() - 30);
    full
}

pub fn delta(fields: &str) -> String {
    format!(r#"{{"choices":[{{"index":0,"delta":{{{fields}}}}}]}}"#)
}

/// The empty delta that carries the reason a turn ended.
pub fn finish(reason: &str) -> String {
    format!(r#"{{"choices":[{{"index":0,"delta":{{}},"finish_reason":"{reason}"}}]}}"#)
}

pub fn says(text: &str) -> Vec<u8> {
    sse(&[
        &delta(&format!(r#""content":{}"#, quote(text))),
        &finish("stop"),
    ])
}

/// The `python(code)` call as it rides inside a delta. Take this one where the turn puts
/// something else in the same delta; [`runs_python`] is the whole turn.
pub fn python_call(call_id: &str, code: &str) -> serde_json::Value {
    serde_json::json!({
        "index": 0,
        "id": call_id,
        "type": "function",
        "function": { "name": "python", "arguments": serde_json::json!({ "code": code }).to_string() },
    })
}

/// A turn that calls `python(code)` and nothing else.
pub fn runs_python(call_id: &str, code: &str) -> Vec<u8> {
    let call = python_call(call_id, code);
    sse(&[
        &delta(&format!(r#""tool_calls":[{call}]"#)),
        &finish("tool_calls"),
    ])
}

fn quote(text: &str) -> String {
    serde_json::Value::String(text.to_string()).to_string()
}
