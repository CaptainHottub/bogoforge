use std::io::{self, Stdout};
use std::sync::atomic::Ordering;
use std::time::Duration;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph},
    Frame, Terminal,
};
use tokio_util::sync::CancellationToken;

use crate::metrics::SharedMetrics;
use log::error;

/// Retro-terminal palette: cyan for "right", hot pink for "wrong", amber as
/// the accent that ties the whole thing together. Deliberately a different
/// mood than a literal green/orange dashboard clone.
const CORRECT: Color = Color::Rgb(56, 224, 196);
const WRONG: Color = Color::Rgb(255, 92, 141);
const ACCENT: Color = Color::Rgb(255, 184, 76);
const DIM: Color = Color::Rgb(96, 100, 112);
const INK: Color = Color::Rgb(18, 19, 24);

pub fn run(metrics: SharedMetrics, cancel: CancellationToken) {
    let sampler = spawn_sampler(metrics.clone(), cancel.clone());

    if let Err(e) = run_inner(&metrics, &cancel) {
        error!("[tui] error: {e}");
    }
    cancel.cancel();

    let _ = sampler.join();
}

// ── Background sampler: host CPU/RAM via `sysinfo`, GPU via `nvidia-smi` ──────
//
// Runs on its own OS thread (sysinfo and spawning subprocesses are blocking),
// polling roughly once a second and writing straight into `Metrics` so the
// render loop just reads atomics — no channels, no extra synchronisation.

fn spawn_sampler(metrics: SharedMetrics, cancel: CancellationToken) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        use sysinfo::System;

        let mut sys = System::new_all();
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        // The first CPU reading is meaningless until a second sample lands.
        std::thread::sleep(Duration::from_millis(300));

        // Determine which GPU telemetry tool to use (1 = NVIDIA, 2 = AMD, 0 = None)
        // If `nvidia-smi` or `rocm-smi` isn't on PATH there's
        // no point spawning a subprocess every second forever — probe once,
        // and only keep polling if it actually answered.
        let mut active_gpu_tool = if sample_nvidia_smi().is_some() {
            1
        } else if sample_rocm_smi().is_some() {
            2
        } else {
            0
        };

        if active_gpu_tool == 0 {
            metrics.clear_gpu_stats();
        }

        loop {
            if cancel.is_cancelled() {
                return;
            }

            sys.refresh_cpu_usage();
            sys.refresh_memory();
            metrics.set_host_stats(sys.global_cpu_usage() as f64, sys.used_memory(), sys.total_memory());

            match active_gpu_tool {
                1 => {
                    if let Some((name, util, used_mb, total_mb, temp)) = sample_nvidia_smi() {
                        metrics.set_gpu_stats(name, util, used_mb, total_mb, temp);
                    } else {
                        active_gpu_tool = 0;
                        metrics.clear_gpu_stats();
                    }
                }
                2 => {
                    if let Some((name, util, used_mb, total_mb, temp)) = sample_rocm_smi() {
                        metrics.set_gpu_stats(name, util, used_mb, total_mb, temp);
                    } else {
                        active_gpu_tool = 0;
                        metrics.clear_gpu_stats();
                    }
                }
                _ => {}
            }

            // Sleep in short slices so cancellation lands within ~100ms
            // instead of stalling shutdown for up to a full sample interval.
            for _ in 0..10 {
                if cancel.is_cancelled() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    })
}

/// Runs `nvidia-smi` once and parses a single line of CSV telemetry:
/// `name, utilization.gpu, memory.used, memory.total, temperature.gpu`.
/// Returns `None` if the binary is missing, errors, or replies with anything
/// we don't understand — the caller treats that as "nothing to monitor here".
fn sample_nvidia_smi() -> Option<(String, i32, u64, u64, i32)> {
    let mut cmd = std::process::Command::new("nvidia-smi");
    cmd.args([
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]);

    // Keep a console window from flashing into existence on every poll.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let output = cmd.output().ok()?;
    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next()?;
    let mut parts = line.split(',').map(|s| s.trim());

    let name = parts.next()?.to_string();
    let util: i32 = parts.next()?.parse().ok()?;
    let used_mb: u64 = parts.next()?.parse().ok()?;
    let total_mb: u64 = parts.next()?.parse().ok()?;
    let temp: i32 = parts.next()?.parse().ok()?;

    Some((name, util, used_mb, total_mb, temp))
}

/// Runs `rocm-smi` once and parses the CSV telemetry.
/// Maps the AMD output perfectly into the expected format:
/// `name, utilization.gpu, memory.used (MB), memory.total (MB), temperature.gpu`.
fn sample_rocm_smi() -> Option<(String, i32, u64, u64, i32)> {
    let mut cmd = std::process::Command::new("rocm-smi");
    cmd.args([
        "--showproductname",
        "--showuse",
        "--showmeminfo", "vram",
        "--showtemp",
        "--csv",
    ]);

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let output = cmd.output().ok()?;
    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut lines = text.lines();

    // Parse headers to find column indices dynamically
    let header_line = lines.next()?;
    let headers: Vec<&str> = header_line.split(',').collect();

    let mut idx_name = None;
    let mut idx_util = None;
    let mut idx_used = None;
    let mut idx_total = None;
    let mut idx_temp = None;

    for (i, h) in headers.iter().enumerate() {
        let h_lower = h.to_lowercase();
        if h_lower.contains("card series") || h_lower.contains("product") {
            idx_name = Some(i);
        } else if h_lower.contains("use (%)") {
            idx_util = Some(i);
        } else if h_lower.contains("used memory") {
            idx_used = Some(i);
        } else if h_lower.contains("total memory") {
            idx_total = Some(i);
        } else if h_lower.contains("temp") && h_lower.contains("edge") {
            idx_temp = Some(i);
        }
    }

    let data_line = lines.next()?;
    let parts: Vec<&str> = data_line.split(',').collect();

    // Extract and clean the strings (stripping quotes and non-numeric chars)
    let name = parts.get(idx_name?)?.replace('"', "").trim().to_string();

    let util_str = parts.get(idx_util?)?.chars().filter(|c| c.is_ascii_digit()).collect::<String>();
    let util: i32 = util_str.parse().ok()?;

    // Convert AMD's Byte output into Megabytes to match NVIDIA
    let used_bytes_str = parts.get(idx_used?)?.chars().filter(|c| c.is_ascii_digit()).collect::<String>();
    let used_mb = used_bytes_str.parse::<u64>().ok()? / (1024 * 1024);

    let total_bytes_str = parts.get(idx_total?)?.chars().filter(|c| c.is_ascii_digit()).collect::<String>();
    let total_mb = total_bytes_str.parse::<u64>().ok()? / (1024 * 1024);

    // AMD sometimes outputs floats for temp (e.g., 45.0), so we parse to f64 first
    let temp_str = parts.get(idx_temp?)?.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect::<String>();
    let temp: i32 = temp_str.parse::<f64>().ok()? as i32; 

    Some((name, util, used_mb, total_mb, temp))
}

fn run_inner(metrics: &SharedMetrics, cancel: &CancellationToken) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = render_loop(&mut terminal, metrics, cancel);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn render_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    metrics: &SharedMetrics,
    cancel: &CancellationToken,
) -> io::Result<()> {
    let mut view = View::Board;

    loop {
        if cancel.is_cancelled() {
            break;
        }

        terminal.draw(|frame| draw(frame, metrics, view))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                // Windows (and terminals with the keyboard-enhancement flags
                // on) report both press *and* release as separate `Event::Key`
                // values for a single physical keystroke. Acting on both
                // toggled the view on press and immediately back on release —
                // only react to the press (and OS auto-repeat) edge.
                if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                    continue;
                }

                match (key.code, key.modifiers) {
                    (KeyCode::Char('q') | KeyCode::Char('Q'), _)
                    | (KeyCode::Char('c'), KeyModifiers::CONTROL) => break,
                    (KeyCode::Char('p') | KeyCode::Char('P') | KeyCode::Tab, _) => {
                        view = view.toggled();
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}

/// Which body the main area is currently showing — toggled with `p`/`Tab`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Board,
    Stats,
}

impl View {
    fn toggled(self) -> Self {
        match self {
            View::Board => View::Stats,
            View::Stats => View::Board,
        }
    }

    fn hint(self) -> &'static str {
        match self {
            View::Board => "p/tab: performance view",
            View::Stats => "p/tab: board view",
        }
    }
}

// ── Layout ─────────────────────────────────────────────────────────────────────
//
// A wide masthead up top, a swappable body (board+dials, or a performance
// page — toggle with p/Tab), and a marquee-style ticker along the bottom.

fn draw(frame: &mut Frame, metrics: &SharedMetrics, view: View) {
    let area = frame.area();

    let [header, body, footer] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(18), Constraint::Length(3)])
        .areas(area);

    draw_masthead(frame, header, metrics, view);

    match view {
        View::Board => {
            let [board_area, dial_area] = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(46), Constraint::Length(32)])
                .areas(body);

            draw_board(frame, board_area, metrics);
            draw_dials(frame, dial_area, metrics);
        }
        View::Stats => draw_stats(frame, body, metrics),
    }

    draw_ticker(frame, footer, metrics);
}

// ── Masthead ───────────────────────────────────────────────────────────────────

fn draw_masthead(frame: &mut Frame, area: Rect, metrics: &SharedMetrics, view: View) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let [left, right] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(20), Constraint::Length(34)])
        .areas(inner);

    let status = metrics.status.lock().clone();
    let title = Line::from(vec![
        Span::styled(":: ", Style::default().fg(ACCENT)),
        Span::styled(
            "BOGOFORGE",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  shuffle furnace  ", Style::default().fg(DIM)),
        Span::styled("[", Style::default().fg(DIM)),
        Span::styled(status, Style::default().fg(ACCENT)),
        Span::styled("]  ", Style::default().fg(DIM)),
        Span::styled(format!("({})", view.hint()), Style::default().fg(DIM)),
    ]);
    frame.render_widget(Paragraph::new(title).alignment(Alignment::Left), left);

    let last_best = metrics.last_report_best.load(Ordering::Relaxed);
    let badge = if last_best >= 0 {
        Line::from(vec![
            Span::styled(format!("{last_best}"), Style::default().fg(badge_color(last_best)).add_modifier(Modifier::BOLD)),
            Span::styled("/25 latest ", Style::default().fg(DIM)),
            Span::styled("#", Style::default().fg(badge_color(last_best))),
        ])
    } else {
        Line::from(Span::styled("warming up .", Style::default().fg(DIM)))
    };
    frame.render_widget(Paragraph::new(badge).alignment(Alignment::Right), right);
}

// ── Board: the permutation rendered as a grid of chunky tiles ─────────────────

fn draw_board(frame: &mut Frame, area: Rect, metrics: &SharedMetrics) {
    let arr = *metrics.last_best_arr.lock();
    let score = metrics.last_report_best.load(Ordering::Relaxed);

    let title = if score >= 0 {
        format!(" board - {score} of 25 seated correctly ")
    } else {
        " board - awaiting first result ".to_string()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .title_style(Style::default().fg(Color::White))
        .border_style(Style::default().fg(DIM));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if score < 0 {
        frame.render_widget(
            Paragraph::new("the furnace hasn't reported a shuffle yet...")
                .style(Style::default().fg(DIM))
                .alignment(Alignment::Center),
            centered_row(inner),
        );
        return;
    }

    let [label_area, chart_area, legend_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(6), Constraint::Length(1)])
        .areas(inner);

    // Stretch each bar to fill the available width rather than rendering
    // thin hairlines: with a 1-column gap between each of the 25 slots,
    // whatever's left over gets divided up as the per-bar thickness. The
    // vertical split above doesn't change the width, so this applies equally
    // to the label row and the bars beneath it — they line up column for
    // column.
    let gap_cols = 24usize;
    let bar_width = ((inner.width as usize).saturating_sub(gap_cols) / 25).max(1);

    frame.render_widget(
        Paragraph::new(build_value_labels(&arr, bar_width)).alignment(Alignment::Center),
        label_area,
    );

    let lines = build_ascii_bars(&arr, chart_area.height as usize, bar_width);
    frame.render_widget(Paragraph::new(lines).alignment(Alignment::Center), chart_area);

    let legend = Line::from(vec![
        Span::styled("# ", Style::default().fg(CORRECT).add_modifier(Modifier::BOLD)),
        Span::styled("seated correctly      ", Style::default().fg(DIM)),
        Span::styled("# ", Style::default().fg(WRONG).add_modifier(Modifier::BOLD)),
        Span::styled("out of place      ", Style::default().fg(DIM)),
        Span::styled("(bar height = seated value, slots 1-25 left to right)", Style::default().fg(DIM)),
    ]);
    frame.render_widget(Paragraph::new(legend).alignment(Alignment::Center), legend_area);
}

/// Renders the 25 board slots as plain-ASCII vertical bars (`#` stacked from
/// the baseline), one column per slot, `bar_width` characters thick, height
/// proportional to the seated value. Colour-coded green/pink by whether that
/// slot landed correctly. Deliberately avoids any Unicode block-element
/// glyphs (eighths blocks etc.) so it renders identically regardless of the
/// terminal's font coverage — thickness, not finer-grained glyphs, is what
/// makes these fill the area.
fn build_ascii_bars(arr: &[u8; 25], rows: usize, bar_width: usize) -> Vec<Line<'static>> {
    let rows = rows.max(1);
    let bar_width = bar_width.max(1);

    // How many of the `rows` cells (counted from the bottom) should be filled
    // for each bar, rounded to the nearest whole row.
    let filled: Vec<usize> = arr
        .iter()
        .map(|&v| {
            let v = (v as usize).min(25);
            ((v * rows * 2) + 25) / 50
        })
        .collect();

    let mut lines = Vec::with_capacity(rows);
    for r in 0..rows {
        let level_from_bottom = rows - 1 - r;
        let mut spans = Vec::with_capacity(25 * 2);
        for (i, &v) in arr.iter().enumerate() {
            let correct = v == (i + 1) as u8;
            let color = if correct { CORRECT } else { WRONG };
            let ch = if level_from_bottom < filled[i] { '#' } else { ' ' };
            spans.push(Span::styled(
                ch.to_string().repeat(bar_width),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ));
            if i < 24 {
                spans.push(Span::raw(" "));
            }
        }
        lines.push(Line::from(spans));
    }
    lines
}

/// One line of seated values, right-aligned over each bar's column so the
/// numbers read as sitting "on top of" the bar chart beneath them. Falls
/// back to a single digit when the columns are too narrow for "25" — still
/// readable, just terser, on cramped terminals.
fn build_value_labels(arr: &[u8; 25], bar_width: usize) -> Line<'static> {
    let bar_width = bar_width.max(1);
    let mut spans = Vec::with_capacity(25 * 2);
    for (i, &v) in arr.iter().enumerate() {
        let correct = v == (i + 1) as u8;
        let color = if correct { CORRECT } else { WRONG };
        let text = if bar_width >= 2 {
            format!("{v:>bar_width$}")
        } else {
            format!("{}", v % 10)
        };
        spans.push(Span::styled(text, Style::default().fg(color).add_modifier(Modifier::BOLD)));
        if i < 24 {
            spans.push(Span::raw(" "));
        }
    }
    Line::from(spans)
}

fn centered_row(area: Rect) -> Rect {
    let mid = area.height / 2;
    Rect { x: area.x, y: area.y + mid, width: area.width, height: 1 }
}

// ── Dials: gauge + sparkline + headline figures + seed ────────────────────────

fn draw_dials(frame: &mut Frame, area: Rect, metrics: &SharedMetrics) {
    let score = metrics.last_report_best.load(Ordering::Relaxed);
    let all_best = metrics.all_time_best.load(Ordering::Relaxed);
    let rate = metrics.compute_rate();
    let session = metrics.session_shuffles.load(Ordering::Relaxed);
    let lifetime = metrics.lifetime_shuffles.load(Ordering::Relaxed);
    let uptime = fmt_uptime(metrics.started_at.elapsed().as_secs());
    let seed = metrics.current_seed.lock().clone();
    let history: Vec<i32> = metrics.recent_bests.lock().iter().copied().collect();

    let [gauge_area, spark_area, fig_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Length(6), Constraint::Min(10)])
        .areas(area);

    // -- progress-toward-perfect gauge --
    let ratio = if score >= 0 { (score as f64 / 25.0).clamp(0.0, 1.0) } else { 0.0 };
    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" closeness to a clean board ")
                .title_style(Style::default().fg(DIM))
                .border_style(Style::default().fg(DIM)),
        )
        .gauge_style(Style::default().fg(badge_color(score)).bg(INK))
        .ratio(ratio)
        .label(if score >= 0 { format!("{score}/25") } else { "-".to_string() });
    frame.render_widget(gauge, gauge_area);

    // -- trend ramp of recent ticks, drawn with plain ASCII so it renders
    //    identically regardless of the terminal's glyph coverage (no reliance
    //    on the block-element characters ratatui's own Sparkline draws with) --
    let spark_block = Block::default()
        .borders(Borders::ALL)
        .title(" recent trend ")
        .title_style(Style::default().fg(DIM))
        .border_style(Style::default().fg(DIM));
    let spark_inner = spark_block.inner(spark_area);
    frame.render_widget(spark_block, spark_area);
    if history.is_empty() {
        frame.render_widget(
            Paragraph::new("-").style(Style::default().fg(DIM)).alignment(Alignment::Center),
            centered_row(spark_inner),
        );
    } else {
        let spans: Vec<Span> = history
            .iter()
            .map(|&v| {
                Span::styled(
                    ascii_ramp_char(v).to_string(),
                    Style::default().fg(badge_color(v)).add_modifier(Modifier::BOLD),
                )
            })
            .collect();
        frame.render_widget(
            Paragraph::new(Line::from(spans)).alignment(Alignment::Center),
            centered_row(spark_inner),
        );
    }

    // -- headline figures + seed --
    let label = Style::default().fg(DIM);
    let value = Style::default().fg(Color::White).add_modifier(Modifier::BOLD);

    let lines = vec![
        Line::styled("uptime", label),
        Line::styled(uptime, value),
        Line::raw(""),
        Line::styled("throughput", label),
        Line::styled(fmt_rate(rate), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Line::raw(""),
        Line::styled("session shuffles", label),
        Line::styled(fmt_count(session), value),
        Line::raw(""),
        Line::styled("lifetime shuffles", label),
        Line::styled(fmt_count(lifetime), value),
        Line::raw(""),
        Line::styled("personal best", label),
        Line::styled(fmt_best(all_best), Style::default().fg(WRONG).add_modifier(Modifier::BOLD)),
        Line::raw(""),
        Line::styled("seed", label),
        Line::styled(
            if seed.is_empty() { "-".to_string() } else { seed },
            Style::default().fg(Color::White),
        ),
    ];

    let fig_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM));
    let fig_inner = fig_block.inner(fig_area);
    frame.render_widget(fig_block, fig_area);
    frame.render_widget(Paragraph::new(lines), fig_inner);
}

// ── Performance view: a grid of headline stat cards ───────────────────────────

fn draw_stats(frame: &mut Frame, area: Rect, metrics: &SharedMetrics) {
    let rate = metrics.compute_rate();
    let session = metrics.session_shuffles.load(Ordering::Relaxed);
    let lifetime = metrics.lifetime_shuffles.load(Ordering::Relaxed);
    let session_best = metrics.session_best.load(Ordering::Relaxed);
    let all_best = metrics.all_time_best.load(Ordering::Relaxed);
    let uptime_secs = metrics.started_at.elapsed().as_secs();
    let seed = metrics.current_seed.lock().clone();
    let history: Vec<i32> = metrics.recent_bests.lock().iter().copied().collect();

    // Host + GPU telemetry, sampled in the background by `spawn_sampler`.
    let cpu_pct = metrics.cpu_usage_pct();
    let mem_used = metrics.mem_used_bytes.load(Ordering::Relaxed);
    let mem_total = metrics.mem_total_bytes.load(Ordering::Relaxed);
    let gpu_name = metrics.gpu_name.lock().clone();
    let gpu_util = metrics.gpu_util_pct.load(Ordering::Relaxed);
    let gpu_mem_used = metrics.gpu_mem_used_mb.load(Ordering::Relaxed);
    let gpu_mem_total = metrics.gpu_mem_total_mb.load(Ordering::Relaxed);
    let gpu_temp = metrics.gpu_temp_c.load(Ordering::Relaxed);

    let session_avg_rate = if uptime_secs > 0 {
        session as f64 / uptime_secs as f64
    } else {
        0.0
    };
    let valid: Vec<i32> = history.iter().copied().filter(|&v| v >= 0).collect();
    let avg_score = if valid.is_empty() {
        None
    } else {
        Some(valid.iter().sum::<i32>() as f64 / valid.len() as f64)
    };

    let title = match &gpu_name {
        Some(name) => format!(" performance - {name} "),
        None => " performance ".to_string(),
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .title_style(Style::default().fg(Color::White))
        .border_style(Style::default().fg(DIM));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let [grid_area, info_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(12), Constraint::Length(2)])
        .areas(inner);

    let (gpu_load_value, gpu_load_color) = match (&gpu_name, gpu_util) {
        (Some(_), u) if u >= 0 => (format!("{u}% util\n{gpu_temp}C"), load_color(u as f64)),
        (Some(_), _) => ("n/a".to_string(), DIM),
        (None, _) => ("no telemetry\n(nvidia-smi or rocm-smi not found)".to_string(), DIM),
    };
    let (gpu_mem_value, gpu_mem_color) = match &gpu_name {
        Some(_) if gpu_mem_total > 0 => {
            (format!("{gpu_mem_used} / {gpu_mem_total} MB"), Color::White)
        }
        Some(_) => ("n/a".to_string(), DIM),
        None => ("n/a".to_string(), DIM),
    };

    let cards: [(&str, String, Color); 9] = [
        ("live throughput", fmt_rate(rate), ACCENT),
        ("session avg rate", fmt_rate(session_avg_rate), ACCENT),
        (
            "shuffles (session / lifetime)",
            format!("{}\n{}", fmt_count(session), fmt_count(lifetime)),
            Color::White,
        ),
        (
            "session best / all-time",
            format!("{} / {}", fmt_score(session_best), fmt_score(all_best)),
            badge_color(session_best.max(all_best)),
        ),
        (
            "avg of recent ticks",
            match avg_score {
                Some(a) => format!("{a:.1} / 25"),
                None => "- / 25".to_string(),
            },
            badge_color(avg_score.map(|a| a.round() as i32).unwrap_or(-1)),
        ),
        ("uptime", fmt_uptime(uptime_secs), Color::White),
        (
            "host cpu / ram",
            format!("{cpu_pct:.0}% cpu\n{} / {} ram", fmt_bytes(mem_used), fmt_bytes(mem_total)),
            load_color(cpu_pct),
        ),
        ("gpu load", gpu_load_value, gpu_load_color),
        ("gpu memory", gpu_mem_value, gpu_mem_color),
    ];

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Ratio(1, 3), Constraint::Ratio(1, 3), Constraint::Ratio(1, 3)])
        .split(grid_area);

    for (r, row_area) in rows.iter().enumerate() {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Ratio(1, 3), Constraint::Ratio(1, 3), Constraint::Ratio(1, 3)])
            .split(*row_area);
        for (c, col_area) in cols.iter().enumerate() {
            let (label, value, color) = &cards[r * 3 + c];
            stat_card(frame, *col_area, label, value, *color);
        }
    }

    let info = Line::from(vec![
        Span::styled("seed ", Style::default().fg(DIM)),
        Span::styled(
            if seed.is_empty() { "-".to_string() } else { seed },
            Style::default().fg(Color::White),
        ),
    ]);
    frame.render_widget(Paragraph::new(info).alignment(Alignment::Center), info_area);
}

/// One bordered card: a dim label up top, a big bold figure below it. The
/// value may contain `\n` to stack a couple of related readings (e.g. GPU
/// utilisation and temperature) in the same card.
fn stat_card(frame: &mut Frame, area: Rect, label: &str, value: &str, color: Color) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines = vec![
        Line::styled(format!(" {label}"), Style::default().fg(DIM)),
        Line::raw(""),
    ];
    for part in value.split('\n') {
        lines.push(Line::styled(
            format!(" {part}"),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

fn fmt_score(v: i32) -> String {
    if v < 0 {
        "-".to_string()
    } else {
        v.to_string()
    }
}

/// Colour-codes a 0-100 load percentage: calm when idle, amber when busy,
/// pink when pegged.
fn load_color(pct: f64) -> Color {
    if pct >= 90.0 {
        WRONG
    } else if pct >= 60.0 {
        ACCENT
    } else {
        CORRECT
    }
}

fn fmt_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else if b >= MB {
        format!("{:.0} MB", b / MB)
    } else if b >= KB {
        format!("{:.0} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

// ── Ticker: a single scrolling-style line of recent results ───────────────────

fn draw_ticker(frame: &mut Frame, area: Rect, metrics: &SharedMetrics) {
    let history: Vec<i32> = metrics.recent_bests.lock().iter().copied().collect();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if history.is_empty() {
        frame.render_widget(
            Paragraph::new(Span::styled(
                " quiet so far - the first results will scroll through here ",
                Style::default().fg(DIM),
            )),
            inner,
        );
        return;
    }

    let valid: Vec<i32> = history.iter().copied().filter(|&v| v >= 0).collect();
    let avg = if valid.is_empty() {
        0.0
    } else {
        valid.iter().sum::<i32>() as f64 / valid.len() as f64
    };

    let mut spans = vec![
        Span::styled(" history ", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled(">> ", Style::default().fg(DIM)),
    ];
    for (i, &v) in history.iter().enumerate() {
        let text = if v < 0 { "-".to_string() } else { v.to_string() };
        spans.push(Span::styled(text, Style::default().fg(badge_color(v)).add_modifier(Modifier::BOLD)));
        if i + 1 < history.len() {
            spans.push(Span::styled(" | ", Style::default().fg(DIM)));
        }
    }
    spans.push(Span::styled("   running avg ", Style::default().fg(DIM)));
    spans.push(Span::styled(
        format!("{avg:.1}"),
        Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
    ));

    frame.render_widget(Paragraph::new(Line::from(spans)), inner);
}

// ── Formatting helpers ─────────────────────────────────────────────────────────

/// Plain-ASCII intensity ramp (low -> high) used in place of Unicode block
/// glyphs, so the trend strip renders the same on any font/terminal.
const ASCII_RAMP: &[u8] = b" .-:=+*#%@";

fn ascii_ramp_char(score: i32) -> char {
    if score < 0 {
        return '.';
    }
    let clamped = (score as usize).min(25);
    let idx = (clamped * (ASCII_RAMP.len() - 1)) / 25;
    ASCII_RAMP[idx] as char
}

fn badge_color(best: i32) -> Color {
    match best {
        n if n < 0 => DIM,
        n if n >= 20 => CORRECT,
        n if n >= 12 => ACCENT,
        _ => WRONG,
    }
}

fn fmt_best(best: i32) -> String {
    if best < 0 {
        "- / 25".into()
    } else {
        format!("{best} / 25")
    }
}

fn fmt_rate(rate: f64) -> String {
    if rate <= 0.0 {
        return "-".into();
    }
    if rate >= 1e12 {
        format!("{:.2}T/s", rate / 1e12)
    } else if rate >= 1e9 {
        format!("{:.2}B/s", rate / 1e9)
    } else if rate >= 1e6 {
        format!("{:.2}M/s", rate / 1e6)
    } else if rate >= 1e3 {
        format!("{:.2}K/s", rate / 1e3)
    } else {
        format!("{rate:.0}/s")
    }
}

fn fmt_count(n: u64) -> String {
    if n == 0 {
        return "0".into();
    }
    if n >= 1_000_000_000_000 {
        format!("{:.2}T", n as f64 / 1e12)
    } else if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{n}")
    }
}

fn fmt_uptime(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{h}h {m}m {s}s")
}
