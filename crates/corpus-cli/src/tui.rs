use std::time::{Duration, Instant};

use anyhow::Result;
use corpus_agent::Event;
use ratatui::Frame;
use ratatui::crossterm::event::{Event as Term, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Margin, Position};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use tokio::sync::mpsc;

use crate::session::{Prompt, Session, deliver};
use crate::{MARK, Opening, Sink, WORDMARK};

const ACCENT: Color = Color::Cyan;
const BRAND: Color = Color::Rgb(0x00, 0x00, 0xFF);
const MUTED: Color = Color::DarkGray;
/// Assumes a dark terminal, which is the only thing a single constant can assume.
const BAND: Color = Color::Indexed(236);
/// Past this share of the window, how much room is left stops being trivia.
const TIGHT: f32 = 0.75;
const FULL: f32 = 0.9;
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

/// A task as one line of a transcript. Unlike a cell, what a child was asked is prose:
/// there is no line in it worth picking out, only a beginning worth showing.
pub fn one_line(text: &str) -> String {
    let line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match line.chars().count() > 64 {
        true => ellipsize(&line, 63),
        false => line,
    }
}

fn ellipsize(text: &str, keep: usize) -> String {
    text.chars().take(keep).chain("…".chars()).collect()
}

/// One step of the transcript, running or finished — a cell that ran or a child that was
/// sent off. The glimpse of what it does is `FLEX_SPAN`, so a narrow terminal eats that
/// and keeps the mark, the name and the tail.
fn step(mark: &str, colour: Color, name: &str, glimpse: String, tail: String) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{mark} "), Style::new().fg(colour)),
        Span::styled(name.to_string(), Style::new().fg(ACCENT)),
        Span::styled(format!(" · {glimpse}"), Style::new()),
        Span::styled(tail, muted()),
    ])
}

/// Round token counts the way a reader says them out loud.
fn tokens(count: u32) -> String {
    match count {
        0..1_000 => count.to_string(),
        1_000..100_000 => format!("{:.1}k", count as f32 / 1000.0),
        _ => format!("{}k", count / 1000),
    }
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
enum Entry {
    Text {
        indent: u16,
        style: Style,
        text: String,
        /// Still being streamed into: the next delta of the same kind extends it.
        open: bool,
    },
    /// Already styled span by span, so it is placed rather than wrapped. `flex` says the
    /// line has a span it will give up when the terminal is too narrow, and belongs to
    /// whoever built it: a lockup has nothing it is willing to lose.
    Ready { line: Line<'static>, flex: bool },
}

#[derive(Default)]
struct Transcript {
    entries: Vec<Entry>,
    wrapped: Vec<Line<'static>>,
    /// Where each entry's wrapped lines begin in `wrapped`. Streaming only ever touches
    /// the end of the transcript, so this is what lets a redraw lay out the last entry
    /// again instead of the whole history.
    starts: Vec<usize>,
    /// What `wrapped` was last laid out from, so a redraw that changed neither the text
    /// nor the width can keep it.
    at_width: u16,
    /// The earliest entry whose layout is no longer good.
    dirty: Option<usize>,
}

impl Transcript {
    fn touch(&mut self, at: usize) {
        self.dirty = Some(self.dirty.map_or(at, |first| first.min(at)));
    }

    /// A separator, unless the transcript already ends in one.
    fn blank(&mut self) {
        if self
            .entries
            .last()
            .is_some_and(|entry| matches!(entry, Entry::Text { text, .. } if text.is_empty()))
        {
            self.close();
            return;
        }
        self.line(0, Style::new(), "");
    }

    /// One entry per newline, because nothing downstream breaks a line for us: `wrap`
    /// splits on width alone, and a `\n` left inside an entry reaches the screen as a
    /// control character.
    fn push_text(&mut self, indent: u16, style: Style, text: &str) {
        self.touch(self.entries.len());
        for piece in text.split('\n') {
            self.entries.push(Entry::Text {
                indent,
                style,
                text: piece.to_string(),
                open: false,
            });
        }
    }

    fn line(&mut self, indent: u16, style: Style, text: impl Into<String>) {
        self.close();
        self.push_text(indent, style, &text.into());
    }

    fn stream(&mut self, indent: u16, style: Style, text: &str) {
        // The last entry is either extended, replaced or followed; nothing before it moves.
        self.touch(self.entries.len().saturating_sub(1));
        for (n, piece) in text.split('\n').enumerate() {
            let extend = n == 0
                && matches!(self.entries.last(), Some(Entry::Text { open: true, indent: at, style: with, .. }) if *at == indent && *with == style);
            if !extend {
                // The placeholder left by a trailing newline was waiting for more of the
                // same kind; text of another kind means it never arrived. Text of the
                // same kind means the writer meant a blank line, and it stays.
                let stale = self.entries.last().is_some_and(|entry| {
                    matches!(entry, Entry::Text { open: true, text, indent: at, style: with }
                        if text.is_empty() && (*at != indent || *with != style))
                });
                if stale {
                    self.entries.pop();
                }
                self.entries.push(Entry::Text {
                    indent,
                    style,
                    text: String::new(),
                    open: true,
                });
            }
            let Some(Entry::Text { text, .. }) = self.entries.last_mut() else {
                unreachable!("just pushed");
            };
            text.push_str(piece);
        }
    }

    /// The final answer replaces what was streamed for it: a provider can correct itself
    /// mid-stream, and the log is what the transcript should agree with.
    fn replace_from(&mut self, at: usize, indent: u16, style: Style, text: &str) {
        self.touch(at);
        self.entries.truncate(at);
        if !text.is_empty() {
            self.push_text(indent, style, text);
        }
    }

    fn push_ready(&mut self, line: Line<'static>, flex: bool) {
        self.touch(self.entries.len());
        self.entries.push(Entry::Ready { line, flex });
    }

    /// Drops everything from `at` and puts one ready-made line in its place. `push_ready`
    /// touches what the truncate left behind, which is this same index.
    ///
    /// What a cell printed collapses into the line that replaces it; ready-made lines do
    /// not, because they are not the cell's output. A child sent off mid-cell leaves one
    /// of those, and it has to outlive the cell that sent it: otherwise the only word of
    /// a child that runs for minutes disappears the moment its parent's cell ends.
    fn replace_line(&mut self, at: usize, line: Line<'static>, flex: bool) {
        let kept: Vec<Entry> = self
            .entries
            .drain(at.min(self.entries.len())..)
            .filter(|entry| matches!(entry, Entry::Ready { .. }))
            .collect();
        self.push_ready(line, flex);
        self.entries.extend(kept);
    }

    fn close(&mut self) {
        let Some(Entry::Text { open, text, .. }) = self.entries.last_mut() else {
            return;
        };
        // A block that ended with a newline left an empty line waiting for text that
        // never came; it would read as a gap nobody asked for.
        if *open && text.is_empty() {
            self.entries.pop();
            self.touch(self.entries.len());
        } else {
            *open = false;
        }
    }

    fn rewrap(&mut self, width: u16) {
        // Redrawn far more often than it changes: a pulse tick, a keystroke or a scroll
        // leaves every wrapped line exactly where it was. A change part-way through costs
        // only the entries from there on, so a token arriving into a long session does not
        // lay the whole history out again.
        let from = match self.dirty {
            _ if self.at_width != width => 0,
            Some(first) => first.min(self.entries.len()).min(self.starts.len()),
            None => return,
        };
        self.dirty = None;
        self.at_width = width;
        let cut = self.starts.get(from).copied().unwrap_or(self.wrapped.len());
        self.wrapped.truncate(cut);
        self.starts.truncate(from);
        for entry in &self.entries[from..] {
            self.starts.push(self.wrapped.len());
            match entry {
                Entry::Ready { line, flex } => {
                    let mut line = line.clone();
                    let room = width as usize;
                    let total: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
                    if total > room
                        && let Some(span) = flex.then(|| line.spans.get_mut(FLEX_SPAN)).flatten()
                    {
                        let keep = span
                            .content
                            .chars()
                            .count()
                            .saturating_sub(total - room + 1);
                        span.content = ellipsize(&span.content, keep).into();
                    }
                    self.wrapped.push(line);
                }
                Entry::Text {
                    indent,
                    style,
                    text,
                    ..
                } => {
                    let room = (width as usize).saturating_sub(*indent as usize).max(8);
                    for piece in wrap(text, room) {
                        let mut line = " ".repeat(*indent as usize);
                        line.push_str(&piece);
                        if style.bg.is_some() {
                            // A band has to reach both edges, or it reads as a highlighted word.
                            let pad = (width as usize).saturating_sub(line.chars().count());
                            line.push_str(&" ".repeat(pad));
                        }
                        self.wrapped.push(Line::styled(line, *style));
                    }
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
    Window(u32),
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
    /// Tokens the last turn sent, and how many the model takes at once. A window of zero
    /// is a provider that never said, and then only the count is worth showing.
    context: u32,
    window: u32,
    transcript: Transcript,
    input: Vec<char>,
    cursor: usize,
    queued: Vec<String>,
    /// What the agent is doing and since when. `None` is an idle session, and there is no
    /// moment to read off one.
    working: Option<(String, Instant)>,
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
    /// What each child was asked, kept so the line saying it finished can say which one
    /// finished. A child's own stream never reaches the screen: it is in the log, and
    /// what a reader needs here is that one started and that one came back.
    kids: std::collections::HashMap<uuid::Uuid, String>,
}

struct RunningTool {
    at: usize,
    name: String,
    code: String,
    output_lines: usize,
}

impl App {
    /// The lockup belongs to opening the window, not to the session: it is on screen
    /// before the first prompt is typed.
    fn new() -> App {
        let mut app = App {
            model: String::new(),
            context: 0,
            window: 0,
            transcript: Transcript::default(),
            input: Vec::new(),
            cursor: 0,
            queued: Vec::new(),
            working: None,
            offset: 0,
            follow: true,
            tick: 0,
            answer_at: None,
            writing_at: None,
            tool: None,
            kids: std::collections::HashMap::new(),
        };
        let brand = Style::new().fg(BRAND);
        app.transcript
            .push_ready(Line::styled(MARK[0], brand), false);
        app.transcript.push_ready(
            Line::from(vec![
                Span::styled(MARK[1], brand),
                Span::raw(format!("  corpus v{}", env!("CARGO_PKG_VERSION"))),
            ]),
            false,
        );
        app.transcript.push_ready(
            Line::from(vec![
                Span::styled(MARK[2], brand),
                Span::raw("  by "),
                Span::styled(WORDMARK, Style::new().add_modifier(Modifier::BOLD)),
            ]),
            false,
        );
        app
    }

    fn busy(&self) -> bool {
        self.working.is_some()
    }

    fn on_event(&mut self, event: Event) {
        // A child's deltas are the child's business: the parent's transcript carries the
        // line that says one started and the line that says it came back, and the log
        // keeps everything in between for whoever wants it.
        if event
            .agent()
            .is_some_and(|agent| self.kids.contains_key(&agent))
        {
            return;
        }
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
            // News from the children reads as the transcript's own voice, not as the
            // person's: nobody typed it, and a band would say somebody did.
            Event::Notice { text, .. } => {
                self.answer_at = None;
                self.transcript.blank();
                self.transcript.line(0, muted(), format!("· {text}"));
                // Nobody pressed anything, so nothing else has said the session is working.
                self.busy_with("thinking");
            }
            Event::AgentStart { agent, task, .. } => {
                self.answer_at = None;
                self.transcript.blank();
                self.transcript.push_ready(
                    step("·", ACCENT, "agent", one_line(&task), String::new()),
                    true,
                );
                self.kids.insert(agent, task);
            }
            Event::AgentEnd {
                agent, ok, chars, ..
            } => {
                let task = self.kids.get(&agent).cloned().unwrap_or_default();
                let (mark, colour) = match ok {
                    true => ("✓", Color::Green),
                    false => ("✗", Color::Red),
                };
                self.answer_at = None;
                self.transcript.push_ready(
                    step(
                        mark,
                        colour,
                        "agent",
                        one_line(&task),
                        format!(" · {chars} chars"),
                    ),
                    true,
                );
            }
            Event::ThinkingDelta { text, .. } => self.transcript.stream(2, thinking(), &text),
            Event::MessageDelta { text, .. } => {
                self.answer_at.get_or_insert(self.transcript.entries.len());
                self.transcript.stream(2, Style::new(), &text);
            }
            Event::Answer { text, .. } => match self.answer_at.take() {
                Some(at) => self.transcript.replace_from(at, 2, Style::new(), &text),
                // An answer nobody streamed is the loop giving up and saying why, which
                // is the moment it is most worth reading. Without this the turn just
                // stops after a cell, and a ceiling reads as a hang.
                None if !text.is_empty() => {
                    self.transcript.blank();
                    self.transcript.line(2, Style::new(), text);
                }
                None => {}
            },
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
                    preview(&code),
                    format!(" · ↑ {} lines", code.lines().count()),
                );
                // The cell was watched as it was written, so the step takes that place
                // back rather than repeating the code a second time below it.
                let at = match self.writing_at.take() {
                    Some(at) => {
                        self.transcript.replace_line(at, running, true);
                        at
                    }
                    None => {
                        self.transcript.blank();
                        let at = self.transcript.entries.len();
                        self.transcript.push_ready(running, true);
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
            Event::TurnEnd { context, .. } => {
                // A turn stopped mid-cell keeps the code it had written on screen, but
                // the place it was going to stays behind with it.
                self.writing_at = None;
                self.context = context;
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
            preview(&tool.code),
            format!(" · ↑ {sent}{back} lines · {}", took(ms)),
        );
        self.transcript.replace_line(tool.at, head, true);
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
        self.working = Some((what.into(), Instant::now()));
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

    /// How much of the window the session is carrying. A known window is worth showing
    /// from the start, empty; an unknown one has nothing to say until a turn has come
    /// back with a count, and then only the count.
    fn fill(&self) -> Line<'static> {
        if self.context == 0 && self.window == 0 {
            return Line::default();
        }
        let mut spans = vec![Span::styled(format!("{} ", tokens(self.context)), muted())];
        if self.window > 0 {
            let share = self.context as f32 / self.window as f32;
            let colour = match share {
                _ if share >= FULL => Color::Red,
                _ if share >= TIGHT => Color::Yellow,
                _ => MUTED,
            };
            spans.push(Span::styled(
                format!("/ {} · ", tokens(self.window)),
                muted(),
            ));
            spans.push(Span::styled(
                format!("{:.0}% ", share * 100.0),
                Style::new().fg(colour),
            ));
        }
        Line::from(spans)
    }

    fn draw(&mut self, frame: &mut Frame) {
        let [body, prompt, status] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        let body = body.inner(Margin::new(1, 0));
        self.transcript.rewrap(body.width);
        let working = self.working.as_ref().map(|(what, since)| {
            Line::styled(
                format!(
                    "{} {what} · {:.1}s",
                    PULSE[self.tick % PULSE.len()],
                    since.elapsed().as_secs_f32()
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

        let mut left = match self.model.is_empty() {
            true => String::new(),
            false => format!(" {}", self.model),
        };
        if !self.queued.is_empty() {
            left.push_str(&format!(" · {} queued", self.queued.len()));
        }
        frame.render_widget(Paragraph::new(Line::styled(left, muted())), status);
        frame.render_widget(Paragraph::new(self.fill()).right_aligned(), status);

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

pub async fn run(session: Box<dyn Session>, sink: Sink, opening: Opening) -> Result<()> {
    let (prompt_tx, prompt_rx) = mpsc::channel::<String>(8);
    let (msg_tx, mut msgs) = mpsc::unbounded_channel::<Msg>();

    // Whoever can name the window is still being asked; the screen opens without it.
    let Opening { model, window } = opening;
    let asked = msg_tx.clone();
    tokio::spawn(async move {
        if let Ok(tokens) = window.await {
            let _ = asked.send(Msg::Window(tokens));
        }
    });

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
    app.model = model;
    let mut pulse = tokio::time::interval(Duration::from_millis(PULSE_MS));
    // The ticks missed while the session sat idle are not owed to anybody: catching up on
    // them would spin the mark through a burst the moment a turn starts.
    pulse.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let outcome = loop {
        terminal.draw(|frame| app.draw(frame))?;
        tokio::select! {
            // Only the mark beside a running turn animates, so an idle session waits for
            // something to happen rather than redrawing four times a second forever.
            _ = pulse.tick(), if app.busy() => app.tick += 1,
            Some(message) = msgs.recv() => {
                let mut message = Some(message);
                // Deltas arrive far faster than the eye: fold everything pending into
                // one redraw instead of one frame per token.
                while let Some(current) = message.take() {
                    match current {
                        Msg::Event(event) => app.on_event(event),
                        Msg::Window(tokens) => app.window = tokens,
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
        loop {
            // A person typing and the children reporting back are the two things that
            // start a turn, and between turns both are waited on at once.
            let mut show = |event| {
                let _ = tx.send(Msg::Event(sink.emit(event)));
            };
            let next = tokio::select! {
                prompt = prompts.recv() => match prompt {
                    Some(text) => Some(Prompt::Human(text)),
                    None => break,
                },
                news = session.idle(&mut show) => match news {
                    Ok(news) => news.map(Prompt::Notice),
                    Err(error) => {
                        let _ = tx.send(Msg::Failed(format!("{error:#}")));
                        None
                    }
                },
            };
            let Some(prompt) = next else { continue };
            let result = deliver(session.as_mut(), prompt, &mut |event| {
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
    use corpus_provider::{StopReason, Usage};
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

    /// How many transcript lines the lockup takes before anything a turn writes.
    const LOCKUP: u16 = 3;

    /// The row under the prompt, as a reader sees it: the outer padding is trimmed but
    /// the gap that separates the two ends of the row is not.
    fn status(app: &mut App) -> String {
        let screen = screen(app, 60, 12);
        screen.lines().next_back().unwrap().trim().to_string()
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
                agent,
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
        assert!(
            screen.starts_with(&format!(
                "  \\|/\n  -*-  corpus v{}",
                env!("CARGO_PKG_VERSION")
            )),
            "{screen}"
        );
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
            agent,
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
    fn a_turn_that_gives_up_says_so_on_screen() {
        let agent = Uuid::now_v7();
        let mut app = App::new();
        // The shape a ceiling leaves behind: a cell ran, no answer was ever streamed.
        app.on_event(Event::ToolEnd {
            agent,
            call_id: "c1".into(),
            ok: true,
            summary: "1 lines".into(),
            ms: 4,
        });
        app.on_event(Event::Answer {
            agent,
            text: "Stopped after 200 steps without reaching an answer.".into(),
        });

        let screen = screen(&mut app, 70, 16);
        assert!(
            screen.contains("Stopped after 200 steps"),
            "why the turn ended must reach the screen, or it reads as a hang: {screen}"
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

    /// A child is two lines in the parent's transcript and nothing else on screen: eight
    /// of them streaming at once is not something a reader can follow, and the log keeps
    /// every word of it anyway.
    #[test]
    fn a_child_is_two_lines_and_its_stream_stays_in_the_log() {
        let root = Uuid::now_v7();
        let child = Uuid::now_v7();
        let mut app = App::new();
        app.context = 4_000;
        for event in [
            Event::AgentStart {
                agent: child,
                parent: root,
                task: "Read the second quarterly report and summarise it".into(),
            },
            Event::MessageDelta {
                agent: child,
                text: "margins fell".into(),
            },
            Event::ToolStart {
                agent: child,
                call_id: "c9".into(),
                name: "python".into(),
                args: json!({ "code": "print('child at work')" }),
            },
            Event::ToolEnd {
                agent: child,
                call_id: "c9".into(),
                ok: true,
                summary: "child at work".into(),
                ms: 7,
            },
            Event::TurnEnd {
                turn_id: child,
                agent: child,
                stop: StopReason::Stop,
                usage: Usage::default(),
                context: 190_000,
            },
            Event::AgentEnd {
                agent: child,
                parent: root,
                ok: true,
                chars: 4210,
                preview: "margins fell".into(),
            },
        ] {
            app.on_event(event);
        }

        let screen = screen(&mut app, 70, 16);
        assert!(
            screen.contains("· agent · Read the second quarterly report and summarise it"),
            "the line saying a child was sent off: {screen}"
        );
        assert!(
            screen.contains("✓ agent · Read the second quarterly report")
                && screen.contains("4210 chars"),
            "the line saying it came back: {screen}"
        );
        assert!(
            !screen.contains("margins fell") && !screen.contains("child at work"),
            "a child's own stream must stay off the parent's screen: {screen}"
        );
        assert_eq!(app.context, 4_000, "a child's context is not the session's");
    }

    /// The parent keeps writing while a child works, and a child's step must not be taken
    /// for the parent's: the running cell it would fold is not the one it started.
    #[test]
    fn a_child_does_not_disturb_the_cell_the_parent_is_running() {
        let root = Uuid::now_v7();
        let child = Uuid::now_v7();
        let mut app = App::new();
        for event in [
            Event::ToolStart {
                agent: root,
                call_id: "c1".into(),
                name: "python".into(),
                args: json!({ "code": "answers = [kid.result() for kid in agents()]" }),
            },
            Event::AgentStart {
                agent: child,
                parent: root,
                task: "dig".into(),
            },
            Event::ToolStart {
                agent: child,
                call_id: "c2".into(),
                name: "python".into(),
                args: json!({ "code": "print('elsewhere')" }),
            },
            Event::ToolEnd {
                agent: child,
                call_id: "c2".into(),
                ok: false,
                summary: "NameError: elsewhere".into(),
                ms: 3,
            },
            Event::ToolEnd {
                agent: root,
                call_id: "c1".into(),
                ok: true,
                summary: "3 answers".into(),
                ms: 900,
            },
        ] {
            app.on_event(event);
        }

        let screen = screen(&mut app, 70, 16);
        assert!(
            screen.contains("✓ python · answers = [kid.result() for kid in a"),
            "the parent's own cell must be the one that folds: {screen}"
        );
        assert!(
            !screen.contains("NameError"),
            "a child's failure unfolded into the parent's transcript: {screen}"
        );
        assert!(
            screen.contains("· agent · dig"),
            "a child sent off mid-cell must outlive the cell that sent it: {screen}"
        );
    }

    /// News from the children is not a question a person asked, and the band is how the
    /// transcript says who is talking.
    #[test]
    fn news_from_the_children_does_not_read_as_a_person() {
        let root = Uuid::now_v7();
        let mut app = App::new();
        app.on_event(Event::Notice {
            turn_id: root,
            agent: root,
            text: "an agent finished · 4210 chars".into(),
        });

        let rendered = buffer(&mut app, 40, 12);
        assert_ne!(
            rendered[(2, LOCKUP + 1)].style().bg,
            Some(BAND),
            "only a person's own question gets the band"
        );
        assert!(
            screen(&mut app, 40, 12).contains("· an agent finished"),
            "the news itself belongs on screen"
        );
    }

    /// The row above the prompt: which model answers, and how full its window is.
    #[test]
    fn the_status_row_names_the_model_and_how_full_the_window_is() {
        let agent = Uuid::now_v7();
        let mut app = App::new();
        let end = |context| Event::TurnEnd {
            turn_id: agent,
            agent,
            stop: StopReason::Stop,
            usage: Usage {
                input: 99_999,
                output: 12,
            },
            context,
        };

        // The model is on screen from the moment the window opens, because the session
        // says nothing about itself until a turn is under way.
        app.model = "nemotron-3".into();
        assert_eq!(
            status(&mut app),
            "nemotron-3",
            "an unanswered window leaves nothing to report"
        );

        app.on_event(end(12_400));
        let row = status(&mut app);
        assert!(row.starts_with("nemotron-3"), "{row}");
        assert!(
            row.ends_with("12.4k") && !row.contains('%'),
            "the count stands alone until the provider has answered for the window: {row}"
        );

        // Asking for the window runs behind the session rather than ahead of it, so the
        // share appears mid-conversation, against what is already on screen.
        app.window = 200_000;
        let row = status(&mut app);
        assert!(row.ends_with("12.4k / 200k · 6%"), "{row}");

        // The turn's whole bill counts a long turn's context once per step, so the row
        // must read the last step instead of `usage.input`.
        app.on_event(end(184_000));
        assert!(status(&mut app).ends_with("184k / 200k · 92%"), "{row}");
    }

    /// A window known before the first prompt is worth showing empty, so the row is not
    /// blank for the whole of the first turn.
    #[test]
    fn a_known_window_shows_before_anything_has_been_asked() {
        let mut app = App::new();
        app.model = "nemotron-3".into();
        app.window = 200_000;

        let row = status(&mut app);
        assert!(row.starts_with("nemotron-3"), "{row}");
        assert!(row.ends_with("0 / 200k · 0%"), "{row}");
    }

    /// A provider that never said how big the window is gets no invented percentage.
    #[test]
    fn an_unknown_window_leaves_the_count_to_speak_for_itself() {
        let agent = Uuid::now_v7();
        let mut app = App::new();
        app.on_event(Event::SessionStart {
            session_id: agent,
            model: "gpt-4o".into(),
        });
        app.on_event(Event::TurnEnd {
            turn_id: agent,
            agent,
            stop: StopReason::Stop,
            usage: Usage::default(),
            context: 900,
        });

        let row = status(&mut app);
        assert!(row.starts_with("gpt-4o") && row.ends_with("900"), "{row}");
        assert!(!row.contains('%'), "no window, no share: {row}");
    }

    /// The mark opens the window, whole and in the brand, before anything is typed.
    #[test]
    fn the_window_opens_under_the_mark() {
        let mut app = App::new();

        let screen = screen(&mut app, 60, 12);
        let rows: Vec<&str> = screen.lines().take(3).collect();
        assert_eq!(
            rows,
            [
                "  \\|/",
                &format!("  -*-  corpus v{}", env!("CARGO_PKG_VERSION")),
                "  /|\\  by ALPHA COMPUTE"
            ],
            "{screen}"
        );
        assert_eq!(
            buffer(&mut app, 60, 12)[(3, 1)].style().fg,
            Some(BRAND),
            "the star of the mark"
        );
    }

    /// Only a line that named a span it can give up loses one. Every ready-made line used
    /// to lose its third span whatever it held, which for the lockup is the company's name.
    #[test]
    fn only_the_span_a_line_named_gives_way_when_it_does_not_fit() {
        let crowded = || {
            Line::from(vec![
                Span::raw("aaaa"),
                Span::raw("bbbb"),
                Span::raw("cccccccc"),
            ])
        };

        let mut step = Transcript::default();
        step.push_ready(crowded(), true);
        step.rewrap(10);
        assert!(
            step.wrapped[0].spans[FLEX_SPAN].content.ends_with('…'),
            "a step gives up its glimpse of code: {:?}",
            step.wrapped[0]
        );

        let mut lockup = Transcript::default();
        lockup.push_ready(crowded(), false);
        lockup.rewrap(10);
        assert_eq!(
            lockup.wrapped[0].spans[FLEX_SPAN].content, "cccccccc",
            "a line that named no flexible span keeps all of them"
        );

        // And the lockup is such a line.
        let mut app = App::new();
        app.transcript.rewrap(12);
        assert_eq!(
            app.transcript.wrapped[2].spans[FLEX_SPAN].content, WORDMARK,
            "the wordmark is not something the mark is willing to lose"
        );
    }

    /// A provider's failure arrives as one string with newlines in it, and `wrap` breaks
    /// on width alone: an entry that kept its `\n` would draw it as a control character.
    #[test]
    fn a_multi_line_error_becomes_one_transcript_line_each() {
        let mut app = App::new();
        app.transcript.line(
            0,
            Style::new(),
            "provider returned 500:\n<html>\nbad gateway",
        );

        let rows: Vec<String> = screen(&mut app, 40, 12)
            .lines()
            .map(str::to_string)
            .collect();
        assert!(
            rows.iter().any(|row| row.trim() == "<html>"),
            "each line stands on its own row: {rows:#?}"
        );
    }

    #[test]
    fn a_question_is_banded_across_the_whole_width() {
        let agent = Uuid::now_v7();
        let mut app = App::new();
        app.on_event(Event::UserMessage {
            turn_id: agent,
            agent,
            text: "hi".into(),
        });

        let rendered = buffer(&mut app, 40, 12);
        let row = LOCKUP + 1;
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
        let thought = rendered[(3, LOCKUP)].style();
        let answer = rendered[(3, LOCKUP + 1)].style();
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
            agent,
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

    /// Only the entries from the first changed one are laid out again, so what that
    /// leaves behind has to be what laying the whole transcript out afresh would give.
    #[test]
    fn a_kept_layout_matches_one_done_from_scratch() {
        let agent = Uuid::now_v7();
        let mut app = App::new();
        for event in [
            Event::UserMessage {
                turn_id: agent,
                agent,
                text: "go".into(),
            },
            Event::ThinkingDelta {
                agent,
                text: "weighing\nit ".into(),
            },
            Event::ThinkingDelta {
                agent,
                text: "up".into(),
            },
            Event::MessageDelta {
                agent,
                text: "a partial answer long enough to wrap more than once".into(),
            },
            Event::CodeDelta {
                agent,
                text: "print(1)".into(),
            },
            Event::ToolStart {
                agent,
                call_id: "c1".into(),
                name: "python".into(),
                args: json!({ "code": "print(1)" }),
            },
            Event::ToolStream {
                agent,
                call_id: "c1".into(),
                text: "one\ntwo\n".into(),
            },
            Event::ToolEnd {
                agent,
                call_id: "c1".into(),
                ok: false,
                summary: "boom".into(),
                ms: 3,
            },
            Event::MessageDelta {
                agent,
                text: "leaked".into(),
            },
            Event::Answer {
                agent,
                text: "corrected\nanswer".into(),
            },
        ] {
            app.on_event(event);
            app.transcript.rewrap(30);
            let kept = app.transcript.wrapped.clone();

            // Nothing carried over: every entry laid out again.
            app.transcript.dirty = Some(0);
            app.transcript.rewrap(30);
            assert_eq!(
                kept, app.transcript.wrapped,
                "the kept layout drifted from a fresh one"
            );
            assert_eq!(
                app.transcript.starts.len(),
                app.transcript.entries.len(),
                "one recorded start per entry"
            );
        }
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
