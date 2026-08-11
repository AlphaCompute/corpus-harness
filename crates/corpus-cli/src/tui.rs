use std::time::{Duration, Instant};

use anyhow::Result;
use corpus_agent::Event;
use ratatui::Frame;
use ratatui::crossterm::event::{Event as Term, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Margin, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use tokio::sync::mpsc;

use crate::Sink;
use crate::session::Session;

const ACCENT: Color = Color::Cyan;
const BRAND: Color = Color::Rgb(0x00, 0x00, 0xFF);
const MUTED: Color = Color::DarkGray;
/// Assumes a dark terminal, which is the only thing a single constant can assume.
const BAND: Color = Color::Indexed(236);
/// The one row above the transcript.
const HEAD: u16 = 1;
/// Which span of a ready-made line gives way when the terminal is too narrow: what the
/// cell did and how long it took must survive, so it is the glimpse of code that goes.
const FLEX_SPAN: usize = 2;
const PULSE: [&str; 4] = ["◇", "◈", "◆", "◈"];
const PULSE_MS: u64 = 250;

fn muted() -> Style {
    Style::new().fg(MUTED)
}

/// The line of a cell a reader would point at: not a comment, not an import, and a call
/// if there is one. Prime Agent scores every line for this; one rule covers our cells.
fn preview(code: &str) -> String {
    const LOW_SIGNAL: [&str; 6] = ["print(", "len(", "str(", "repr(", "int(", "float("];
    let lines: Vec<&str> = code
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    // A cell that does something is best described by what it calls; a cell that only
    // probes imports is best described by the import it probes.
    let rank = |line: &&str| {
        if line.starts_with("import ") || line.starts_with("from ") {
            1
        } else if !line.contains('(') {
            3
        } else if LOW_SIGNAL.iter().any(|dull| line.starts_with(dull)) {
            2
        } else {
            0
        }
    };
    let chosen = lines.iter().copied().min_by_key(rank).unwrap_or_default();

    let line: String = chosen.split_whitespace().collect::<Vec<_>>().join(" ");
    match line.chars().count() > 64 {
        true => ellipsize(&line, 63),
        false => line,
    }
}

fn ellipsize(text: &str, keep: usize) -> String {
    text.chars().take(keep).chain("…".chars()).collect()
}

/// One step of the transcript, running or finished. The preview is `FLEX_SPAN`, so a
/// narrow terminal eats the glimpse of code and keeps the mark, the name and the tail.
fn step(mark: &str, colour: Color, name: &str, code: &str, tail: String) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{mark} "), Style::new().fg(colour)),
        Span::styled(name.to_string(), Style::new().fg(ACCENT)),
        Span::styled(format!(" · {}", preview(code)), Style::new()),
        Span::styled(tail, muted()),
    ])
}

fn took(ms: u64) -> String {
    match ms < 1000 {
        true => format!("{ms}ms"),
        false => format!("{:.1}s", ms as f64 / 1000.0),
    }
}

/// Thinking is set apart from the answer by weight, not by position: it sits in the
/// transcript where it happened.
fn thinking() -> Style {
    Style::new().fg(MUTED).add_modifier(Modifier::ITALIC)
}

/// A logical line of the transcript. Wrapping happens at draw time against the current
/// width, so a resize reflows the whole history instead of leaving ragged old lines.
struct Entry {
    indent: u16,
    style: Style,
    text: String,
    /// Still being streamed into: the next delta of the same kind extends it.
    open: bool,
    /// Already styled span by span, so it is placed rather than wrapped.
    ready: Option<Line<'static>>,
}

#[derive(Default)]
struct Transcript {
    entries: Vec<Entry>,
    wrapped: Vec<Line<'static>>,
    /// What `wrapped` was last laid out from, so a redraw that changed neither the text
    /// nor the width can keep it.
    at_width: u16,
    dirty: bool,
}

impl Transcript {
    /// A separator, unless the transcript already ends in one.
    fn blank(&mut self) {
        if self
            .entries
            .last()
            .is_some_and(|entry| entry.text.is_empty() && entry.ready.is_none())
        {
            self.close();
            return;
        }
        self.line(0, Style::new(), "");
    }

    fn line(&mut self, indent: u16, style: Style, text: impl Into<String>) {
        self.close();
        self.dirty = true;
        self.entries.push(Entry {
            indent,
            style,
            text: text.into(),
            open: false,
            ready: None,
        });
    }

    fn stream(&mut self, indent: u16, style: Style, text: &str) {
        self.dirty = true;
        for (n, piece) in text.split('\n').enumerate() {
            let extend = n == 0
                && matches!(self.entries.last(), Some(e) if e.open && e.indent == indent && e.style == style);
            if !extend {
                // The placeholder left by a trailing newline was waiting for more of the
                // same kind; text of another kind means it never arrived. Text of the
                // same kind means the writer meant a blank line, and it stays.
                let stale = self.entries.last().is_some_and(|e| {
                    e.open && e.text.is_empty() && (e.indent != indent || e.style != style)
                });
                if stale {
                    self.entries.pop();
                }
                self.entries.push(Entry {
                    indent,
                    style,
                    text: String::new(),
                    open: true,
                    ready: None,
                });
            }
            self.entries
                .last_mut()
                .expect("just pushed")
                .text
                .push_str(piece);
        }
    }

    /// The final answer replaces what was streamed for it: a provider can correct itself
    /// mid-stream, and the log is what the transcript should agree with.
    fn replace_from(&mut self, at: usize, indent: u16, style: Style, text: &str) {
        self.dirty = true;
        self.entries.truncate(at);
        if !text.is_empty() {
            for piece in text.split('\n') {
                self.entries.push(Entry {
                    indent,
                    style,
                    text: piece.to_string(),
                    open: false,
                    ready: None,
                });
            }
        }
    }

    fn push_ready(&mut self, line: Line<'static>) {
        self.dirty = true;
        self.entries.push(Entry {
            indent: 0,
            style: Style::new(),
            text: String::new(),
            open: false,
            ready: Some(line),
        });
    }

    /// Drops everything from `at` and puts one ready-made line in its place.
    fn replace_line(&mut self, at: usize, line: Line<'static>) {
        self.entries.truncate(at);
        self.push_ready(line);
    }

    fn close(&mut self) {
        let Some(entry) = self.entries.last_mut() else {
            return;
        };
        self.dirty = true;
        // A block that ended with a newline left an empty line waiting for text that
        // never came; it would read as a gap nobody asked for.
        if entry.open && entry.text.is_empty() {
            self.entries.pop();
        } else {
            entry.open = false;
        }
    }

    fn rewrap(&mut self, width: u16) {
        // Redrawn far more often than it changes: a pulse tick, a keystroke or a scroll
        // leaves every wrapped line exactly where it was.
        if !self.dirty && self.at_width == width {
            return;
        }
        self.dirty = false;
        self.at_width = width;
        self.wrapped.clear();
        for entry in &self.entries {
            if let Some(ready) = &entry.ready {
                let mut line = ready.clone();
                let room = width as usize;
                let total: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
                if total > room
                    && let Some(span) = line.spans.get_mut(FLEX_SPAN)
                {
                    let keep = span
                        .content
                        .chars()
                        .count()
                        .saturating_sub(total - room + 1);
                    span.content = span
                        .content
                        .chars()
                        .take(keep)
                        .chain("…".chars())
                        .collect::<String>()
                        .into();
                }
                self.wrapped.push(line);
            } else {
                let room = (width as usize)
                    .saturating_sub(entry.indent as usize)
                    .max(8);
                for piece in wrap(&entry.text, room) {
                    let mut text = " ".repeat(entry.indent as usize);
                    text.push_str(&piece);
                    if entry.style.bg.is_some() {
                        // A band has to reach both edges, or it reads as a highlighted word.
                        let pad = (width as usize).saturating_sub(text.chars().count());
                        text.push_str(&" ".repeat(pad));
                    }
                    self.wrapped.push(Line::styled(text, entry.style));
                }
            }
        }
    }
}

/// Greedy wrap that breaks on spaces but never drops leading whitespace: Python cells
/// are in here, and their indentation is part of the meaning.
fn wrap(text: &str, width: usize) -> Vec<String> {
    // Bytes are never fewer than characters, so the common line settles without being
    // laid out character by character first.
    if text.len() <= width || text.chars().count() <= width {
        return vec![text.to_string()];
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + width).min(chars.len());
        let mut cut = end;
        if end < chars.len()
            && let Some(space) = chars[start..end].iter().rposition(|c| *c == ' ')
            && space > 0
        {
            cut = start + space + 1;
        }
        out.push(chars[start..cut].iter().collect());
        start = cut;
    }
    out
}

enum Msg {
    Event(Event),
    TurnDone,
    Failed(String),
}

enum Action {
    Nothing,
    Submit(String),
    Interrupt,
    Quit,
}

struct App {
    model: String,
    transcript: Transcript,
    input: Vec<char>,
    cursor: usize,
    queued: Vec<String>,
    working: Option<String>,
    since: Instant,
    offset: usize,
    follow: bool,
    tick: usize,
    /// Where the answer being streamed right now started in the transcript.
    answer_at: Option<usize>,
    /// Where the cell being written right now started. It is watched line by line while
    /// it is written, because that is when stopping it is still worth something.
    writing_at: Option<usize>,
    /// A running cell shows its code and its output; once it is done all of that
    /// collapses into one line, the way a finished step reads in a transcript.
    tool: Option<RunningTool>,
}

struct RunningTool {
    at: usize,
    name: String,
    code: String,
    output_lines: usize,
}

impl App {
    fn new() -> App {
        App {
            model: String::new(),
            transcript: Transcript::default(),
            input: Vec::new(),
            cursor: 0,
            queued: Vec::new(),
            working: None,
            since: Instant::now(),
            offset: 0,
            follow: true,
            tick: 0,
            answer_at: None,
            writing_at: None,
            tool: None,
        }
    }

    fn busy(&self) -> bool {
        self.working.is_some()
    }

    fn on_event(&mut self, event: Event) {
        match event {
            Event::SessionStart { model, .. } => self.model = model,
            Event::UserMessage { text, .. } => {
                self.answer_at = None;
                self.transcript.blank();
                self.transcript.line(
                    0,
                    Style::new().bg(BAND).add_modifier(Modifier::BOLD),
                    format!("› {text}"),
                );
            }
            Event::ThinkingDelta { text, .. } => self.transcript.stream(2, thinking(), &text),
            Event::MessageDelta { text, .. } => {
                self.answer_at.get_or_insert(self.transcript.entries.len());
                self.transcript.stream(2, Style::new(), &text);
            }
            Event::Answer { text, .. } => {
                if let Some(at) = self.answer_at.take() {
                    self.transcript.replace_from(at, 2, Style::new(), &text);
                }
            }
            Event::CodeDelta { text, .. } => {
                if self.writing_at.is_none() {
                    self.answer_at = None;
                    self.transcript.blank();
                    self.writing_at = Some(self.transcript.entries.len());
                    self.busy_with("writing");
                }
                self.transcript.stream(2, Style::new().fg(ACCENT), &text);
            }
            Event::ToolStart { name, args, .. } => {
                self.answer_at = None;
                let code = corpus_agent::python_code(&args)
                    .unwrap_or_default()
                    .to_string();
                // A running cell reads the same as a finished one. Its code is worth the
                // room only when it failed, and that is not known yet.
                let running = step(
                    "·",
                    ACCENT,
                    &name,
                    &code,
                    format!(" · ↑ {} lines", code.lines().count()),
                );
                // The cell was watched as it was written, so the step takes that place
                // back rather than repeating the code a second time below it.
                let at = match self.writing_at.take() {
                    Some(at) => {
                        self.transcript.replace_line(at, running);
                        at
                    }
                    None => {
                        self.transcript.blank();
                        let at = self.transcript.entries.len();
                        self.transcript.push_ready(running);
                        at
                    }
                };
                self.tool = Some(RunningTool {
                    at,
                    name: name.clone(),
                    code,
                    output_lines: 0,
                });
                self.busy_with(name);
            }
            Event::ToolStream { text, .. } => {
                if let Some(tool) = &mut self.tool {
                    tool.output_lines += text.matches('\n').count();
                }
                self.transcript.stream(4, muted(), &text);
            }
            Event::ToolEnd {
                ok, summary, ms, ..
            } => {
                self.finish_tool(ok, &summary, ms);
                self.answer_at = None;
                self.busy_with("thinking");
            }
            Event::Compaction { dropped, .. } => {
                self.transcript
                    .line(0, muted(), format!("· compacted {dropped} tool results"))
            }
            Event::TurnEnd { stop, usage, .. } => {
                // A turn stopped mid-cell keeps the code it had written on screen, but
                // the place it was going to stays behind with it.
                self.writing_at = None;
                self.transcript.line(
                    0,
                    muted(),
                    format!("· {stop:?} · {} in / {} out", usage.input, usage.output),
                );
            }
            _ => {}
        }
    }

    /// A cell that worked leaves the line it already had; a cell that failed unfolds,
    /// because its code is what you need in front of you to fix it.
    fn finish_tool(&mut self, ok: bool, summary: &str, ms: u64) {
        let Some(tool) = self.tool.take() else {
            return;
        };
        let sent = tool.code.lines().count();
        let back = match tool.output_lines {
            0 => String::new(),
            lines => format!(" ↓ {lines}"),
        };
        let (mark, colour) = match ok {
            true => ("✓", Color::Green),
            false => ("✗", Color::Red),
        };
        let head = step(
            mark,
            colour,
            &tool.name,
            &tool.code,
            format!(" · ↑ {sent}{back} lines · {}", took(ms)),
        );
        self.transcript.replace_line(tool.at, head);
        if !ok {
            for line in tool.code.lines() {
                // A gutter mark on an empty line is a mark with nothing to point at.
                let shown = match line.trim().is_empty() {
                    true => String::new(),
                    false => format!("› {line}"),
                };
                self.transcript.line(2, Style::new().fg(ACCENT), shown);
            }
            self.transcript
                .line(2, Style::new().fg(Color::Red), summary.to_string());
        }
    }

    fn busy_with(&mut self, what: impl Into<String>) {
        self.working = Some(what.into());
        self.since = Instant::now();
    }

    fn on_key(&mut self, key: ratatui::crossterm::event::KeyEvent) -> Action {
        if key.kind == KeyEventKind::Release {
            return Action::Nothing;
        }
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            // While a turn is running Ctrl+C stops the turn, not corpus: the session,
            // the kernel and everything in its namespace stay alive.
            KeyCode::Char('c') if control => {
                return match self.busy() {
                    true => Action::Interrupt,
                    false => Action::Quit,
                };
            }
            KeyCode::Char('d') if control && self.input.is_empty() => return Action::Quit,
            KeyCode::Char('u') if control => {
                self.input.drain(..self.cursor);
                self.cursor = 0;
            }
            KeyCode::Char('k') if control => {
                self.input.truncate(self.cursor);
            }
            KeyCode::Char('a') if control => self.cursor = 0,
            KeyCode::Char('e') if control => self.cursor = self.input.len(),
            KeyCode::Char(c) if !control => {
                self.input.insert(self.cursor, c);
                self.cursor += 1;
            }
            KeyCode::Backspace if self.cursor > 0 => {
                self.cursor -= 1;
                self.input.remove(self.cursor);
            }
            KeyCode::Delete if self.cursor < self.input.len() => {
                self.input.remove(self.cursor);
            }
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => self.cursor = (self.cursor + 1).min(self.input.len()),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.input.len(),
            KeyCode::Up => self.scroll(-1),
            KeyCode::Down => self.scroll(1),
            KeyCode::PageUp => self.scroll(-10),
            KeyCode::PageDown => self.scroll(10),
            KeyCode::Enter if !self.input.is_empty() => {
                let text: String = self.input.drain(..).collect();
                self.cursor = 0;
                self.follow = true;
                if self.busy() {
                    self.queued.push(text);
                } else {
                    return Action::Submit(text);
                }
            }
            _ => {}
        }
        Action::Nothing
    }

    fn scroll(&mut self, by: isize) {
        self.offset = self.offset.saturating_add_signed(by);
        self.follow = false;
    }

    /// One row: who this is, which model, and how far the view is from the end.
    fn header(&self, frame: &mut Frame, head: Rect, bottom: usize) {
        let mut spans = vec![Span::styled("corpus", Style::new().fg(BRAND))];
        if !self.model.is_empty() {
            spans.push(Span::styled(format!(" · {}", self.model), muted()));
        }
        spans.push(Span::styled(
            format!(" · v{}", env!("CARGO_PKG_VERSION")),
            muted(),
        ));
        frame.render_widget(Paragraph::new(Line::from(spans)), head);

        // Styling the paragraph would paint its whole rect, and this sits on top of the
        // header: the style belongs to the words, not to the row they land on.
        if self.offset < bottom {
            frame.render_widget(
                Paragraph::new(Line::styled(
                    format!(" ↓ {} more ", bottom - self.offset),
                    muted(),
                ))
                .right_aligned(),
                head,
            );
        }
    }

    fn draw(&mut self, frame: &mut Frame) {
        let [head, body, status, prompt] = Layout::vertical([
            Constraint::Length(HEAD),
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        let body = body.inner(Margin::new(1, 0));
        self.transcript.rewrap(body.width);
        let working = self.working.as_ref().map(|what| {
            Line::styled(
                format!(
                    "{} {what} · {:.1}s",
                    PULSE[self.tick % PULSE.len()],
                    self.since.elapsed().as_secs_f32()
                ),
                Style::new().fg(ACCENT),
            )
        });
        let settled = self.transcript.wrapped.len();
        let total = settled + usize::from(working.is_some());
        let height = body.height as usize;
        let bottom = total.saturating_sub(height);
        if self.follow {
            self.offset = bottom;
        }
        self.offset = self.offset.min(bottom);
        let mut view =
            self.transcript.wrapped[self.offset..(self.offset + height).min(settled)].to_vec();
        if let Some(working) = working
            && self.offset + height > settled
        {
            view.push(working);
        }
        frame.render_widget(Paragraph::new(view), body);

        self.header(frame, head, bottom);

        let hints = match self.busy() {
            true => " enter send · ↑↓ scroll · ^C interrupt",
            false => " enter send · ↑↓ scroll · ^C quit",
        };
        frame.render_widget(Paragraph::new(Line::styled(hints, muted())), status);
        if !self.queued.is_empty() {
            frame.render_widget(
                Paragraph::new(Line::styled(
                    format!("{} queued ", self.queued.len()),
                    muted(),
                ))
                .right_aligned(),
                status,
            );
        }

        // The input is one line: long text scrolls sideways rather than growing the box.
        let room = prompt.width.saturating_sub(3) as usize;
        let from = self.cursor.saturating_sub(room);
        let shown: String = self.input[from..].iter().take(room).collect();
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(" › ", Style::new().fg(ACCENT)),
                Span::raw(shown),
            ])),
            prompt,
        );
        frame.set_cursor_position(Position::new(
            prompt.x + 3 + (self.cursor - from) as u16,
            prompt.y,
        ));
    }
}

pub async fn run(session: Box<dyn Session>, sink: Sink) -> Result<()> {
    let (prompt_tx, prompt_rx) = mpsc::channel::<String>(8);
    let (msg_tx, mut msgs) = mpsc::unbounded_channel::<Msg>();

    let interrupt = session.interrupt();
    drive(session, sink, prompt_rx, msg_tx);

    let mut terminal = ratatui::init();

    // Raw mode has to be on before anything reads keys, or the first line is line-buffered.
    let (key_tx, mut keys) = mpsc::channel::<Term>(64);
    std::thread::spawn(move || {
        while let Ok(event) = ratatui::crossterm::event::read() {
            if key_tx.blocking_send(event).is_err() {
                return;
            }
        }
    });

    let mut app = App::new();
    let mut pulse = tokio::time::interval(Duration::from_millis(PULSE_MS));

    let outcome = loop {
        terminal.draw(|frame| app.draw(frame))?;
        tokio::select! {
            _ = pulse.tick() => app.tick += 1,
            Some(message) = msgs.recv() => {
                let mut message = Some(message);
                // Deltas arrive far faster than the eye: fold everything pending into
                // one redraw instead of one frame per token.
                while let Some(current) = message.take() {
                    match current {
                        Msg::Event(event) => app.on_event(event),
                        Msg::TurnDone => {
                            app.working = None;
                            if !app.queued.is_empty() {
                                let next = app.queued.remove(0);
                                app.busy_with("thinking");
                                let _ = prompt_tx.send(next).await;
                            }
                        }
                        Msg::Failed(error) => {
                            app.working = None;
                            app.transcript.line(0, Style::new().fg(Color::Red), error);
                        }
                    }
                    message = msgs.try_recv().ok();
                }
            }
            Some(event) = keys.recv() => {
                // A resize needs no marking: the transcript reflows itself when the width
                // it was wrapped against is no longer the width it is drawn into.
                if let Term::Key(key) = event {
                    match app.on_key(key) {
                        Action::Quit => break Ok(()),
                        Action::Interrupt => interrupt.raise(),
                        Action::Submit(text) => {
                            app.busy_with("thinking");
                            if prompt_tx.send(text).await.is_err() {
                                break Ok(());
                            }
                        }
                        Action::Nothing => {}
                    }
                }
            }
        }
    };

    ratatui::restore();
    drop(prompt_tx);
    outcome
}

fn drive(
    mut session: Box<dyn Session>,
    mut sink: Sink,
    mut prompts: mpsc::Receiver<String>,
    tx: mpsc::UnboundedSender<Msg>,
) {
    tokio::spawn(async move {
        while let Some(prompt) = prompts.recv().await {
            let result = session
                .run(&prompt, &mut |event| {
                    let _ = tx.send(Msg::Event(sink.emit(event)));
                })
                .await;
            let _ = match result {
                Ok(()) => tx.send(Msg::TurnDone),
                Err(error) => tx.send(Msg::Failed(format!("{error:#}"))),
            };
        }
        let _ = session
            .finish(&mut |event| {
                let _ = tx.send(Msg::Event(sink.emit(event)));
            })
            .await;
    });
}

#[cfg(test)]
mod tests {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::KeyEvent;
    use serde_json::json;
    use uuid::Uuid;

    use super::*;

    fn buffer(app: &mut App, width: u16, height: u16) -> ratatui::buffer::Buffer {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        terminal.backend().buffer().clone()
    }

    fn screen(app: &mut App, width: u16, height: u16) -> String {
        let width = width as usize;
        buffer(app, width as u16, height)
            .content
            .chunks(width)
            .map(|row| {
                row.iter()
                    .map(|cell| cell.symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Screen row of a transcript line: crossterm counts from zero, and the header takes
    /// the row above.
    fn line_row(line: u16) -> u16 {
        HEAD + line
    }

    fn press(app: &mut App, code: KeyCode) -> Action {
        app.on_key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn type_in(app: &mut App, text: &str) {
        for ch in text.chars() {
            press(app, KeyCode::Char(ch));
        }
    }

    #[test]
    fn wrapping_keeps_indentation_and_splits_words_that_do_not_fit() {
        assert_eq!(wrap("    x = 1", 20), ["    x = 1"]);
        assert_eq!(
            wrap("one two three four", 8),
            ["one two ", "three ", "four"]
        );
        assert_eq!(
            wrap("https://averylongurl", 10),
            ["https://av", "erylongurl"]
        );
    }

    #[test]
    fn a_turn_renders_as_a_transcript() {
        let agent = Uuid::now_v7();
        let mut app = App::new();
        for event in [
            Event::SessionStart {
                session_id: agent,
                model: "test-model".into(),
            },
            Event::UserMessage {
                turn_id: agent,
                text: "say hi".into(),
            },
            Event::ToolStart {
                agent,
                call_id: "c1".into(),
                name: "python".into(),
                args: json!({ "code": "print('hi')" }),
            },
            Event::ToolStream {
                agent,
                call_id: "c1".into(),
                text: "hi\n".into(),
            },
            Event::ToolEnd {
                agent,
                call_id: "c1".into(),
                ok: true,
                summary: "hi".into(),
                ms: 12,
            },
            Event::MessageDelta {
                agent,
                text: "Said hi.".into(),
            },
        ] {
            app.on_event(event);
        }

        let screen = screen(&mut app, 70, 16);
        assert!(screen.starts_with("corpus · test-model"), "{screen}");
        assert!(screen.contains("› say hi"), "{screen}");
        assert!(
            screen.contains("✓ python · print('hi') · ↑ 1 ↓ 1 lines · 12ms"),
            "{screen}"
        );
        assert!(screen.contains("  Said hi."), "{screen}");
    }

    #[test]
    fn what_the_agent_is_doing_sits_at_the_end_of_the_transcript() {
        let agent = Uuid::now_v7();
        let mut app = App::new();
        app.on_event(Event::UserMessage {
            turn_id: agent,
            text: "go".into(),
        });
        app.on_event(Event::MessageDelta {
            agent,
            text: "On it.".into(),
        });
        app.busy_with("python");

        let rows: Vec<String> = screen(&mut app, 60, 12)
            .lines()
            .map(str::to_string)
            .collect();
        let message = rows
            .iter()
            .position(|row| row.contains("On it."))
            .expect("the message");
        assert!(
            rows[message + 1].contains("python ·"),
            "it belongs right after the message: {rows:#?}"
        );
        assert!(
            rows[rows.len() - 2].contains("^C interrupt"),
            "the bottom row is hints now"
        );
    }

    #[test]
    fn a_cell_is_described_by_what_it_does() {
        assert_eq!(
            preview("results = fetch_url(url='x')\nprint(results)"),
            "results = fetch_url(url='x')"
        );
        assert_eq!(preview("print(71 * 93)"), "print(71 * 93)");
        assert_eq!(
            preview(
                "# probe\ntry:\n    from fpdf import FPDF\n    print('yes')\nexcept:\n    pass"
            ),
            "from fpdf import FPDF",
            "a cell that only probes imports is about the import"
        );
    }

    #[test]
    fn a_running_cell_does_not_spell_out_its_code() {
        let agent = Uuid::now_v7();
        let mut app = App::new();
        app.on_event(Event::ToolStart {
            agent,
            call_id: "c1".into(),
            name: "python".into(),
            args: json!({
                "code": "page = fetch_url(url='https://tokio.rs')\ntext = page['text']\nprint(text[:200])"
            }),
        });

        let screen = screen(&mut app, 70, 16);
        assert!(
            screen.contains("· python · page = fetch_url(url='https://tokio.rs') · ↑ 3 lines"),
            "{screen}"
        );
        assert!(
            !screen.contains("print(text"),
            "the body stays folded: {screen}"
        );
    }

    #[test]
    fn a_cell_is_watched_as_it_is_written_and_folds_once_it_runs() {
        let agent = Uuid::now_v7();
        let code = "page = fetch_url(url='https://tokio.rs')\nprint(page['text'][:200])";
        let mut app = App::new();
        for piece in [
            "page = fetch_url(url=",
            "'https://tokio.rs')\nprint(page['text'][:200])",
        ] {
            app.on_event(Event::CodeDelta {
                agent,
                text: piece.into(),
            });
        }

        let writing = screen(&mut app, 70, 16);
        assert!(
            writing.contains("print(page['text'][:200])"),
            "a long cell must be readable while it is written: {writing}"
        );

        app.on_event(Event::ToolStart {
            agent,
            call_id: "c1".into(),
            name: "python".into(),
            args: json!({ "code": code }),
        });

        let running = screen(&mut app, 70, 16);
        assert!(
            running.contains("· python · page = fetch_url(url='https://tokio.rs') · ↑ 2 lines"),
            "{running}"
        );
        assert!(
            !running.contains("print(page"),
            "the step takes back the room the code was watched in: {running}"
        );
    }

    #[test]
    fn a_failed_cell_unfolds_its_code() {
        let agent = Uuid::now_v7();
        let mut app = App::new();
        app.on_event(Event::ToolStart {
            agent,
            call_id: "c1".into(),
            name: "python".into(),
            args: json!({ "code": "print(1/0)" }),
        });
        app.on_event(Event::ToolEnd {
            agent,
            call_id: "c1".into(),
            ok: false,
            summary: "ZeroDivisionError: division by zero".into(),
            ms: 8,
        });

        let screen = screen(&mut app, 70, 16);
        assert!(
            screen.contains("✗ python · print(1/0) · ↑ 1 lines · 8ms"),
            "{screen}"
        );
        assert!(
            screen.contains("› print(1/0)"),
            "a failure unfolds its code: {screen}"
        );
        assert!(
            screen.contains("ZeroDivisionError: division by zero"),
            "{screen}"
        );
    }

    /// Anything drawn on top of the header must not repaint the row it lands on.
    #[test]
    fn the_header_keeps_its_colour_under_what_is_written_over_it() {
        let agent = Uuid::now_v7();
        let mut app = App::new();
        app.on_event(Event::SessionStart {
            session_id: agent,
            model: "nemotron-3".into(),
        });
        for n in 0..40 {
            app.on_event(Event::MessageDelta {
                agent,
                text: format!("line {n}\n"),
            });
        }
        app.scroll(-20);

        let rendered = buffer(&mut app, 60, 12);
        assert_eq!(rendered[(0, 0)].style().fg, Some(BRAND), "the name");
        assert_eq!(
            rendered[(10, 0)].style().fg,
            Some(MUTED),
            "the model beside it"
        );
        assert!(
            screen(&mut app, 60, 12)
                .lines()
                .next()
                .unwrap()
                .contains("↓"),
            "and how far from the end it is"
        );
    }

    #[test]
    fn a_question_is_banded_across_the_whole_width() {
        let agent = Uuid::now_v7();
        let mut app = App::new();
        app.on_event(Event::UserMessage {
            turn_id: agent,
            text: "hi".into(),
        });

        let rendered = buffer(&mut app, 40, 12);
        let row = line_row(1);
        assert_eq!(
            rendered[(2, row)].style().bg,
            Some(BAND),
            "the text sits on the band"
        );
        assert_eq!(
            rendered[(37, row)].style().bg,
            Some(BAND),
            "and so does the far end"
        );
    }

    /// Whatever the terminal is, every character the model wrote has to be on screen.
    #[test]
    fn nothing_streamed_is_lost_at_any_width() {
        let thought = "The user wants me to create a fancy PDF about \"alpha compute\". \
            I need to first understand what it refers to: a company, a concept, a product.\n\
            \n\
            Let me search for information about it, then find a way to render a document.";
        for width in [40u16, 61, 78, 150] {
            let agent = Uuid::now_v7();
            let mut app = App::new();
            for chunk in thought.as_bytes().chunks(7) {
                app.on_event(Event::ThinkingDelta {
                    agent,
                    text: String::from_utf8_lossy(chunk).into_owned(),
                });
            }
            let shown: String = screen(&mut app, width, 30)
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            let flat: String = thought.split_whitespace().collect::<Vec<_>>().join(" ");
            assert!(
                shown.contains(&flat),
                "text lost or reordered at {width} columns:\n  want: {flat}\n  got:  {shown}"
            );
        }
    }

    #[test]
    fn a_blank_line_inside_a_thought_survives() {
        let agent = Uuid::now_v7();
        let mut app = App::new();
        app.on_event(Event::ThinkingDelta {
            agent,
            text: "first\n".into(),
        });
        app.on_event(Event::ThinkingDelta {
            agent,
            text: "\nsecond".into(),
        });

        let rows: Vec<String> = screen(&mut app, 40, 14)
            .lines()
            .map(str::to_string)
            .collect();
        let first = rows.iter().position(|r| r.contains("first")).unwrap();
        let second = rows.iter().position(|r| r.contains("second")).unwrap();
        assert_eq!(
            second - first,
            2,
            "the empty line between them is part of the text"
        );
    }

    #[test]
    fn thinking_reads_as_thinking_not_as_answer() {
        let agent = Uuid::now_v7();
        let mut app = App::new();
        app.on_event(Event::ThinkingDelta {
            agent,
            text: "weighing it up".into(),
        });
        app.on_event(Event::MessageDelta {
            agent,
            text: "done".into(),
        });

        let rendered = buffer(&mut app, 40, 12);
        let thought = rendered[(3, line_row(0))].style();
        let answer = rendered[(3, line_row(1))].style();
        assert!(
            thought.add_modifier.contains(Modifier::ITALIC),
            "{thought:?}"
        );
        assert_eq!(thought.fg, Some(MUTED));
        assert!(
            !answer.add_modifier.contains(Modifier::ITALIC),
            "the answer is not italic"
        );
    }

    #[test]
    fn the_final_answer_replaces_what_was_streamed_for_it() {
        let agent = Uuid::now_v7();
        let mut app = App::new();
        app.on_event(Event::UserMessage {
            turn_id: agent,
            text: "71*93?".into(),
        });
        app.on_event(Event::MessageDelta {
            agent,
            text: " leaked thinking.</think>6603".into(),
        });
        app.on_event(Event::Answer {
            agent,
            text: "6603".into(),
        });

        let screen = screen(&mut app, 40, 16);
        assert!(screen.contains("6603"), "{screen}");
        assert!(
            !screen.contains("leaked thinking"),
            "the transcript must match the log: {screen}"
        );
    }

    #[test]
    fn the_newest_line_stays_visible_until_you_scroll_away() {
        let agent = Uuid::now_v7();
        let mut app = App::new();
        for n in 0..40 {
            app.on_event(Event::MessageDelta {
                agent,
                text: format!("line {n}\n"),
            });
        }

        assert!(
            screen(&mut app, 40, 14).contains("line 39"),
            "a new line must follow the bottom"
        );
        app.scroll(-30);
        let scrolled = screen(&mut app, 40, 14);
        assert!(
            !scrolled.contains("line 39"),
            "scrolling up must stay put: {scrolled}"
        );
        app.on_event(Event::MessageDelta {
            agent,
            text: "line 40\n".into(),
        });
        assert!(
            !screen(&mut app, 40, 14).contains("line 40"),
            "new output must not yank the view back down"
        );
    }

    /// The wrapped lines are kept between redraws, so both things that invalidate them
    /// have to actually do it.
    #[test]
    fn a_redrawn_transcript_shows_what_arrived_since_the_last_draw() {
        let agent = Uuid::now_v7();
        let mut app = App::new();
        app.on_event(Event::MessageDelta {
            agent,
            text: "first".into(),
        });
        assert!(screen(&mut app, 40, 12).contains("first"));

        app.on_event(Event::MessageDelta {
            agent,
            text: "\nsecond".into(),
        });
        assert!(
            screen(&mut app, 40, 12).contains("second"),
            "new text at an unchanged width"
        );
        assert!(
            screen(&mut app, 12, 12).contains("seco"),
            "and the same text laid out again when the width changes"
        );
    }

    #[test]
    fn typing_while_the_agent_works_waits_for_the_next_turn() {
        let mut app = App::new();
        type_in(&mut app, "first");
        assert!(matches!(press(&mut app, KeyCode::Enter), Action::Submit(text) if text == "first"));

        app.busy_with("python");
        type_in(&mut app, "second");
        assert!(matches!(press(&mut app, KeyCode::Enter), Action::Nothing));
        assert_eq!(app.queued, ["second"]);
        assert!(screen(&mut app, 40, 12).contains("1 queued"));
    }

    #[test]
    fn ctrl_c_stops_the_turn_while_it_runs_and_leaves_when_idle() {
        let mut app = App::new();
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(matches!(app.on_key(ctrl_c), Action::Quit));

        app.busy_with("python");
        assert!(matches!(app.on_key(ctrl_c), Action::Interrupt));
        assert!(screen(&mut app, 60, 12).contains("^C interrupt"));
    }

    #[test]
    fn the_input_edits_like_a_line_of_text() {
        let mut app = App::new();
        type_in(&mut app, "helo");
        press(&mut app, KeyCode::Left);
        type_in(&mut app, "l");
        assert_eq!(app.input.iter().collect::<String>(), "hello");
        press(&mut app, KeyCode::Home);
        assert!(matches!(
            app.on_key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::CONTROL)),
            Action::Nothing
        ));
        assert!(app.input.is_empty());
    }
}
