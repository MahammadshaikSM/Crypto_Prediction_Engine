// Ichnora landing page: reads the same JSON the dashboards publish.
(() => {
const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const D = (iso) => new Date(iso.slice(0, 10) + "T12:00:00");
const fmtDate = (iso, y = true) => D(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", ...(y ? { year: "numeric" } : {}) });
const mean = (a) => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
const pct1 = (n) => `${n.toFixed(1)}%`;
const signed = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1, minimumFractionDigits: 1, signDisplay: "exceptZero" });
const UP = '<svg width="10" height="10" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5l8 12H4z" fill="currentColor"/></svg>';
const DOWN = '<svg width="10" height="10" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19L4 7h16z" fill="currentColor"/></svg>';
const chg = (v) => {
  if (v == null || isNaN(v)) return '<span class="chg flat">–</span>';
  const dir = v > 0.05 ? "up" : v < -0.05 ? "down" : "flat";
  return `<span class="chg ${dir}">${dir === "up" ? UP : dir === "down" ? DOWN : ""}${signed.format(v)}%<span class="sr">${dir === "up" ? " gain" : dir === "down" ? " loss" : ""}</span></span>`;
};
const set = (sel, v) => { const el = $(sel); if (el) el.textContent = v; };
$("#yr").textContent = new Date().getFullYear();

// ---------- generic UI ----------
const io = new IntersectionObserver((es) => es.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } }), { threshold: .15 });
const observe = () => $$(".reveal:not(.in), .card:not(.in)").forEach((el) => io.observe(el));
observe();
$$(".card").forEach((c) => c.addEventListener("pointermove", (e) => { const r = c.getBoundingClientRect(); c.style.setProperty("--mx", `${e.clientX - r.left}px`); c.style.setProperty("--my", `${e.clientY - r.top}px`); }));

const toggle = (btn, target, cls = "paused") => {
  const apply = (p) => { target.classList.toggle(cls, p); btn.setAttribute("aria-pressed", p); btn.textContent = p ? "Play" : "Pause"; };
  btn.addEventListener("click", () => apply(!target.classList.contains(cls)));
  if (RM) apply(true);
};
toggle($("#marquee-ctl"), $("#marquee"));
toggle($("#orbit-ctl"), $("#orbit"));

// ---------- crypto data (site/summary.json, built daily by site/build_summary.py) ----------
function renderHero(sum) {
  const coins = sum.today.slice(0, 10);
  const latest = { date: sum.date, maturity_date: sum.maturity, fetched_at: sum.fetched_at };
  const orbit = $("#orbit");
  const narrow = matchMedia("(max-width: 960px)").matches, R = 232;
  orbit.innerHTML = coins.map((c, i) => `
    <div class="pick3d" role="listitem" style="${narrow ? "" : `transform: rotateY(${i * 36}deg) translateZ(${R}px)`}">
      <span class="rk">#${c.rank ?? i + 1}</span>
      <img src="${esc(c.image)}" alt="" width="58" height="58" loading="eager" referrerpolicy="no-referrer">
      <b>${esc(c.symbol)}</b><span>${esc(c.name)}</span>
      <span class="sc">score <strong class="num">${Number(c.score).toFixed(0)}</strong></span>
    </div>`).join("") + `<div class="orbit-core" aria-hidden="true">Today's<br>10 picks<small>${esc(fmtDate(sum.date, false).toUpperCase())}</small></div>`;
  const d = latest.date;
  set("#updated", `updated ${fmtDate((latest.fetched_at || d).slice(0, 10))}`);
  const lt = sum.light, light = $("#light");
  if (lt && light) {
    light.className = `chip light-chip fade-in ${lt.state}`;
    light.innerHTML = `<span class="ld" aria-hidden="true"></span>Market light <b>${lt.state === "trade" ? "Trade" : "Caution"}</b> · BTC ${lt.btc7 >= 0 ? "+" : ""}${Number(lt.btc7).toFixed(1)}% 7d`;
    light.title = lt.state === "trade" ? "Bitcoin is flat or up over 7 days" : "Bitcoin is down over 7 days";
    light.hidden = false;
  }
  const ta = sum.trader && sum.trader.all;
  if (ta) set("#p-ret", `${ta.avg >= 0 ? "+" : ""}${(ta.avg * 100).toFixed(1)}%`);
  set("#pk-date", fmtDate(d, false));
  set("#pk-due", fmtDate(latest.maturity_date, false));
  $("#picks").innerHTML = coins.map((c, i) => `
    <article class="pick">
      <div class="top"><span class="rk">#${c.rank ?? i + 1}</span><img src="${esc(c.image)}" alt="" width="40" height="40" loading="lazy" referrerpolicy="no-referrer"></div>
      <h3>${esc(c.symbol)}</h3><p class="name">${esc(c.name)}</p>
      <div class="bar" aria-hidden="true"><i style="transform:scaleX(${Math.min(1, (c.score || 0) / 100)})"></i></div>
      <dl>
        <dt>Score</dt><dd class="num">${Number(c.score).toFixed(1)}</dd>
        <dt>Confidence</dt><dd class="num">${Number(c.confidence).toFixed(0)}%</dd>
        <dt>24h</dt><dd>${chg(c.mom24h)}</dd>
        <dt>RSI</dt><dd class="num">${c.rsi != null ? Number(c.rsi).toFixed(0) : "–"}</dd>
      </dl>
    </article>`).join("");
}

function renderReceipts(sum) {
  const st = { rows: sum.rows.map((x) => ({ day: x.d, hit: x.hit, base: x.base, pos: x.pos, mkt: x.mkt, hc: x.hc })), days: sum.days };
  const r = st.rows, hit = mean(r.map((x) => x.hit)), base = mean(r.map((x) => x.base));
  const pos = mean(r.map((x) => x.pos)), mk = r.filter((x) => x.mkt != null).map((x) => x.mkt);
  const lift = hit / base;
  set("#p-lift", `${lift.toFixed(1)}×`); set("#r-lift", lift.toFixed(1));
  set("#p-days", st.days); set("#r-days", st.days); set("#r-graded", r.length);
  set("#r-since", fmtDate(r[0].day));
  set("#r-hit", pct1(hit)); set("#r-base", pct1(base));
  set("#r-hits-per", (hit / 10).toFixed(1)); set("#r-rand-per", (base / 10).toFixed(1));
  const top = Math.max(hit, base) * 1.25;
  $("#bar-me").style.setProperty("--w", hit / top); $("#bar-rand").style.setProperty("--w", base / top);
  set("#r-pos", pct1(pos)); if (mk.length) set("#r-mkt", pct1(mean(mk)));
  const best = r.reduce((a, b) => (b.hit > a.hit ? b : a));
  set("#r-best", `${best.hit.toFixed(0)}%`);
  $("#r-best-date").parentElement.innerHTML = `<b>${esc(fmtDate(best.day))}</b>: ${best.hc} of 10 picks landed in the top 10. The worst days score zero, and those are in the record too.`;

  // monthly table
  const months = {};
  r.forEach((x) => { const m = x.day.slice(0, 7); (months[m] ||= []).push(x); });
  $("#month-table tbody").innerHTML = Object.entries(months).map(([m, xs]) =>
    `<tr><td>${D(m + "-15").toLocaleDateString("en-US", { month: "short", year: "numeric" })}</td><td>${xs.length}</td><td>${pct1(mean(xs.map((x) => x.hit)))}</td><td>${pct1(mean(xs.map((x) => x.base)))}</td><td>${pct1(mean(xs.map((x) => x.pos)))}</td></tr>`).join("");
  drawChart(r);
}

// ---------- chart ----------
function drawChart(rows) {
  const cv = $("#chart"), g = cv.getContext("2d");
  const W = 14, roll = rows.map((_, i) => { const s = rows.slice(Math.max(0, i - W + 1), i + 1); return { day: rows[i].day, hit: mean(s.map((x) => x.hit)), base: mean(s.map((x) => x.base)) }; });
  const maxY = Math.max(30, Math.ceil(Math.max(...roll.map((x) => x.hit)) / 5) * 5);
  const above = roll.filter((x) => x.hit > x.base).length;
  set("#chart-summary", `The 14-day average beat the random baseline on ${above} of ${roll.length} graded days (${Math.round(100 * above / roll.length)}%). Method: see footnote 1.`);
  let prog = RM ? 1 : 0, started = false;
  const paint = () => {
    const dpr = Math.min(devicePixelRatio, 2), w = cv.clientWidth, h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr; g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    const L = 40, B = 26, T = 10, X = (i) => L + (i / (roll.length - 1)) * (w - L - 8), Y = (v) => T + (1 - v / maxY) * (h - T - B);
    g.font = "12px JetBrains Mono, monospace"; g.fillStyle = "#A1A1AC"; g.strokeStyle = "#FFFFFF14"; g.lineWidth = 1;
    for (let v = 0; v <= maxY; v += 10) { g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(w - 8, Y(v)); g.stroke(); g.fillText(`${v}%`, 0, Y(v) + 4); }
    let lastM = "";
    roll.forEach((p, i) => { const m = p.day.slice(0, 7); if (m !== lastM && p.day.slice(8) <= "07") { lastM = m; g.fillText(D(p.day).toLocaleDateString("en-US", { month: "short" }), X(i), h - 6); } });
    const n = Math.max(2, Math.floor(roll.length * prog));
    const line = (key, color, width, fill) => {
      g.beginPath(); roll.slice(0, n).forEach((p, i) => (i ? g.lineTo(X(i), Y(p[key])) : g.moveTo(X(i), Y(p[key]))));
      if (fill) { const gr = g.createLinearGradient(0, T, 0, h - B); gr.addColorStop(0, "#D4FF3A40"); gr.addColorStop(1, "#D4FF3A00");
        g.save(); g.lineTo(X(n - 1), Y(0)); g.lineTo(X(0), Y(0)); g.closePath(); g.fillStyle = gr; g.fill(); g.restore();
        g.beginPath(); roll.slice(0, n).forEach((p, i) => (i ? g.lineTo(X(i), Y(p[key])) : g.moveTo(X(i), Y(p[key])))); }
      g.strokeStyle = color; g.lineWidth = width; g.lineJoin = "round"; g.stroke();
    };
    g.setLineDash([5, 5]); line("base", "#A1A1AC", 1.5); g.setLineDash([]); line("hit", "#D4FF3A", 2.5, true);
  };
  const animate = () => { prog = Math.min(1, prog + .02); paint(); if (prog < 1) requestAnimationFrame(animate); };
  new IntersectionObserver(([en], obs) => { if (en.isIntersecting && !started) { started = true; obs.disconnect(); RM ? paint() : requestAnimationFrame(animate); } }, { threshold: .3 }).observe(cv);
  addEventListener("resize", () => started && paint());
}

// ---------- time machine (site/history.json, lazy-loaded) ----------
function renderExplorer(hist) {
  const rows = hist.days, img = hist.images || {}, range = $("#ex-range");
  range.max = rows.length - 1; range.value = rows.length - 1;
  const show = (i) => {
    const x = rows[i]; range.value = i;
    $("#ex-date").innerHTML = `${esc(fmtDate(x.d))}<small>Picked ${esc(fmtDate(x.d, false))} · graded ${esc(fmtDate(x.e, false))}</small>`;
    $("#ex-score").innerHTML = `<span class="chip"><b>${x.hc}/10</b> in top 10</span><span class="chip"><b>${x.pc}/10</b> went up</span>`;
    $("#ex-list").innerHTML = x.p.map(([rank, id, sym, name, ret, hit]) => `<li class="${hit ? "hit" : ""}"><span class="r">#${rank}</span><img src="${esc(img[id] || "")}" alt="" width="26" height="26" loading="lazy" referrerpolicy="no-referrer"><span><b>${esc(String(sym).toUpperCase())}</b> <span style="color:var(--fg-2)">${esc(name)}</span></span>${chg(ret)}<span class="h">${hit ? '✓<span class="sr"> in top 10</span>' : ""}</span></li>`).join("");
    $("#ex-actual").innerHTML = x.a.map(([sym, ret, was]) => `<li class="${was ? "was" : ""}">${esc(String(sym).toUpperCase())} ${chg(ret)}${was ? " · picked" : ""}</li>`).join("");
  };
  range.addEventListener("input", () => show(+range.value));
  $("#ex-prev").addEventListener("click", () => show(Math.max(0, +range.value - 1)));
  $("#ex-next").addEventListener("click", () => show(Math.min(rows.length - 1, +range.value + 1)));
  $("#ex-best").addEventListener("click", () => show(rows.reduce((bi, x, i) => (x.hc > rows[bi].hc ? i : bi), 0)));
  $("#ex-worst").addEventListener("click", () => show(rows.reduce((wi, x, i) => (x.hc < rows[wi].hc || (x.hc === rows[wi].hc && x.pc < rows[wi].pc) ? i : wi), 0)));
  show(rows.length - 1);
}

// ---------- stocks ----------
function renderStocks(s) {
  if (!s) return;
  const top = Math.max(s.hit, s.base) * 1.25;
  set("#s-lift", (s.hit / s.base).toFixed(1)); set("#s-hit", pct1(s.hit)); set("#s-base", pct1(s.base));
  $("#s-bar-me").style.setProperty("--w", s.hit / top); $("#s-bar-rand").style.setProperty("--w", s.base / top);
  set("#s-days", s.days); set("#s-n", `~${s.universe}`); set("#s-pos", pct1(s.pos));
}

// ---------- load ----------
const load = (url) => fetch(url, { cache: "no-cache" }).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
load("site/summary.json").then((sum) => {
  renderHero(sum); renderReceipts(sum); renderStocks(sum.stocks);
}).catch((err) => {
  console.warn("summary unavailable", err);
  $("#picks").innerHTML = '<p class="intro">Live picks are taking a moment. <a href="prediction_dashboard.html">Open the dashboard</a> to see them now.</p>';
  set("#chart-summary", "History is unavailable right now. The live dashboard has the full record.");
});
new IntersectionObserver(([en], obs) => {
  if (!en.isIntersecting) return; obs.disconnect();
  load("site/history.json").then(renderExplorer).catch((err) => { console.warn("history unavailable", err); $("#ex-date").textContent = "History unavailable right now"; });
}, { rootMargin: "600px" }).observe($("#explorer"));

// ---------- forecast (site/forecast.json, built daily by site/build_forecast.py) ----------
const FEAT = [["vol_adj_mom", "Vol-adj momentum", "#9BC425"], ["mom24h", "24h momentum", "#D4FF3A"], ["btc_rel", "BTC-relative", "#B9E62E"], ["rsi_s", "RSI", "#C9B8FF"], ["vol_surge", "Volume surge", "#8B6CFF"]];
const OLD_W = { vol_adj_mom: 22, mom24h: 28, btc_rel: 25, vol_surge: 10, rsi_s: 15 };
function renderForecast(fc) {
  const tabs = $$(".fc-tab");
  // target_at is UTC; show it in the viewer's own time zone (falls back to the UTC date)
  const localDay = (h) => h.target_at ? new Date(h.target_at).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : fmtDate(h.target_date, false);
  $$(".fc-date").forEach((el) => { const h = fc.horizons[el.dataset.h]; if (h) el.textContent = localDay(h); });
  const show = (h) => {
    const f = fc.horizons[h]; if (!f) return;
    tabs.forEach((t) => t.setAttribute("aria-selected", t.dataset.h === String(h)));
    $("#fc-panel").setAttribute("aria-labelledby", `tab-${h}`);
    $("#fc-list").innerHTML = f.picks.map((c) => `<li><span class="r">#${c.rank}</span><img src="${esc(c.image)}" alt="" width="34" height="34" loading="lazy" referrerpolicy="no-referrer"><span class="nm"><b>${esc(c.symbol)}</b><span>${esc(c.name)} · RSI ${c.rsi == null ? "n/a" : Math.round(c.rsi)}</span></span><span class="m24">${chg(c.mom24h)}</span><span class="sc num" title="Model score">${c.score.toFixed(1)}</span></li>`).join("");
    const t = f.backtest.test, old = f.backtest.live_test, top = Math.max(t.hit, old.hit, t.base) * 1.2;
    set("#fc-lift", (t.hit / t.base).toFixed(1)); set("#fc-hit", pct1(t.hit)); set("#fc-old", pct1(old.hit)); set("#fc-base", pct1(t.base));
    [["#fc-bar-me", t.hit], ["#fc-bar-old", old.hit], ["#fc-bar-rand", t.base]].forEach(([s, v]) => $(s).style.setProperty("--w", v / top));
    $(".fc-side .card").classList.add("in");
    $("#fc-note").innerHTML = `Graded ${h} days later over <b>${t.days} days</b> (Oct 2025 to Sep 2026) the tuning never saw. Picks rose <b>${pct1(t.pos)}</b> of the time and beat the average coin by <b>${signed.format(t.edge)} pts</b> per ${h}-day window.`;
    const w = f.weights;
    $("#fc-weights").style.display = w ? "" : "none";
    if (!w) {  // model v2: no fixed weights, show what drives it instead
      set("#fc-wtitle", "What drives " + (f.model && f.model.name || "the model"));
      $("#fc-wlist").innerHTML = (f.model && f.model.drivers || []).map((d) => `<li><span>${esc(d)}</span></li>`).join("");
      return;
    }
    set("#fc-wtitle", "Tuned weights");
    $("#fc-weights").innerHTML = FEAT.map(([k, , col]) => w[k] > 0 ? `<div style="flex:${w[k]};background:${col}">${w[k] >= 7 ? w[k].toFixed(1) + "%" : ""}</div>` : "").join("");
    $("#fc-weights").setAttribute("aria-label", "Tuned weights: " + FEAT.map(([k, n]) => `${n} ${w[k].toFixed(1)}%`).join(", "));
    $("#fc-wlist").innerHTML = FEAT.map(([k, n, col]) => `<li><span><span class="sw" style="background:${col};display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:8px"></span>${n}</span><span><del>${OLD_W[k]}%</del><b>${w[k].toFixed(1)}%</b></span></li>`).join("");
  };
  tabs.forEach((t) => t.addEventListener("click", () => show(t.dataset.h)));
  $(".fc-tabs").addEventListener("keydown", (e) => { if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const i = tabs.findIndex((t) => t.getAttribute("aria-selected") === "true"); const n = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length]; n.focus(); show(n.dataset.h); } });
  show("3");
}
load("site/forecast.json").then(renderForecast).catch((err) => { console.warn("forecast unavailable", err); $("#fc-list").innerHTML = '<li class="intro">The forecast is being rebuilt. Check back after the next daily update.</li>'; });
})();
