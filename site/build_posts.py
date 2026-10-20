"""Daily post kit + live call data for CryptoTracker Pro.

Writes
  site/calls.json           recent 3-day and 7-day calls with pick prices, latest prices and grades
                            (read by call.html, the page every post links to)
  posts/<date>/             ready-to-post captions (.txt) and share images (.png):
      1_call_7d   today's 7-day call            2_call_3d   today's 3-day call
      3_update_*  calls graded tomorrow         4_result_*  calls graded today
  posts/PINNED.txt          the "how it works" post for your profile

Post kits older than 14 days are removed. Runs in the daily GitHub Action after the trackers.
"""
import json, os, re, shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
# Public website address for posts. Empty = posts carry no link at all.
# Set the repository variable PUBLIC_SITE_URL (Settings > Secrets and variables > Actions > Variables)
# once you have your own domain; nothing else needs to change.
SITE = os.environ.get("PUBLIC_SITE_URL", "").strip()
if SITE and not SITE.endswith("/"):
    SITE += "/"
SHORT = SITE.replace("https://", "").replace("http://", "").rstrip("/")
KEEP_DAYS, CALL_DAYS = 14, 40
TAG = {7: "#7DayCall", 3: "#3DayCall"}

PINNED = f"""📌 How CryptoTracker Pro works

Every day we publish 10 coins for 3 days and 10 for 7 days.
Each call shows the start and end date. On the end date we post the results, good or bad.

• Prices are locked at pick time
• Every call is graded in public: wins and misses
• Every call and its result stays public

{f"Track everything at {SITE}" if SITE else "Follow for a new call every day and every result."}
#CryptoTrackerPro #3DayCall #7DayCall
Not financial advice. Crypto is volatile and you can lose money."""


PROFILE = ("Daily crypto calls: 10 coins for 3 days + 10 for 7 days. Every call graded in public, wins and misses. NFA."
           + (f"\n{SITE}" if SITE else ""))


# ---------- formatting ----------
def d(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def md(s):            # "Oct 19"
    return d(s).strftime("%b %d").replace(" 0", " ")


def wmd(s):           # "Sun, Oct 26"
    return d(s).strftime("%a, %b %d").replace(" 0", " ")


def price(p):
    if p is None:
        return "–"
    if p >= 1000:
        return f"${p:,.0f}"
    if p >= 1:
        return f"${p:,.2f}"
    return f"${p:.4g}" if p >= 0.0001 else f"${p:.2e}"


def pct(v):
    if v is None:
        return "–"
    return ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v):.1f}%"


# ---------- calls ----------
def build_calls(data):
    today = max(data["predictions"])
    latest = data["predictions"][today].get("price_snapshot", {})
    calls = []
    for h, preds, evals in ((7, data["predictions"], data.get("evaluations", {})),
                            (3, data.get("predictions_3d", {}), data.get("evaluations_3d", {}))):
        for day in sorted(preds):
            if d(day) < d(today) - timedelta(days=CALL_DAYS):
                continue
            p, ev = preds[day], evals.get(day)
            res = {c["id"]: c for c in (ev or {}).get("coin_results", [])}
            picks = []
            for c in p["predicted_coins"]:
                r = res.get(c["id"])
                if r is None:   # older picks may have been saved under a different id; match by symbol
                    r = next((x for x in res.values() if x["symbol"].lower() == c["symbol"].lower()), None)
                start = c.get("price_at_prediction")
                now = r["price_at_evaluation"] if r else latest.get(c["id"])
                chg = r["actual_return_pct"] if r else ((now / start - 1) * 100 if now and start else None)
                picks.append(dict(rank=c["rank"], symbol=c["symbol"].upper(), name=c["name"], image=c.get("image", ""),
                                  start=start, now=now, change=None if chg is None else round(chg, 2),
                                  hit=bool(r["hit"]) if r else None))
            calls.append(dict(date=day, end=p.get("maturity_date") or (d(day) + timedelta(days=h)).isoformat(),
                              horizon=h, model=p.get("model", "v1" if h == 7 else "3d-tuned"),
                              status="graded" if ev else "open", graded_on=(ev or {}).get("evaluation_date"),
                              hits=(ev or {}).get("hit_count"), up=(ev or {}).get("positive_count"),
                              actual_top10=[a["symbol"].upper() for a in (ev or {}).get("actual_top10", [])], picks=picks))
    calls.sort(key=lambda c: (c["date"], -c["horizon"]), reverse=True)
    return today, calls


# ---------- images ----------
BG, SURF, FG, FG2, LINE = "#07070A", "#111117", "#F4F4EF", "#A1A1AC", "#24242C"
LIME, ONLIME, DOWN = "#D4FF3A", "#0B0B0F", "#FF7A6B"
FONT_PATHS = {
    "bold": ["C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"],
    "reg": ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
    "mono": ["C:/Windows/Fonts/consolab.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"],
}


def font(kind, size):
    for p in FONT_PATHS[kind]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def card(call, mode, today):
    """mode: call | update | result. 1080x1350 portrait card."""
    W, H = 1080, 1350
    im = Image.new("RGB", (W, H), BG); g = ImageDraw.Draw(im)
    h = call["horizon"]; X = 70
    g.rectangle([0, 0, W, 12], fill=LIME)
    g.text((X, 62), "CRYPTOTRACKER PRO", font=font("bold", 26), fill=LIME)
    title = {"call": f"{h}-DAY CALL", "update": "GRADED TOMORROW", "result": f"{h}-DAY RESULTS"}[mode]
    g.text((X, 100), title, font=font("bold", 92), fill=FG)
    sub = f"{h}-Day Call · {md(call['date'])} → {md(call['end'])}"
    if mode == "result":
        sub = f"{call['hits']}/10 finished in the top 10 · {call['up']}/10 went up"
    elif mode == "update":
        priced = [p for p in call["picks"] if p["change"] is not None]
        up = sum(1 for p in priced if p["change"] > 0)
        sub = f"{h}-Day Call {md(call['date'])} → {md(call['end'])} · {up}/{len(priced)} priced coins up"
    g.text((X, 212), sub, font=font("reg", 34), fill=FG2)

    # timeline
    y, x0, x1 = 300, X + 14, W - X - 14
    total = (d(call["end"]) - d(call["date"])).days
    done = 1.0 if mode == "result" else min(1.0, max(0.0, (d(today) - d(call["date"])).days / total))
    g.line([x0, y, x1, y], fill=LINE, width=8)
    g.line([x0, y, x0 + (x1 - x0) * done, y], fill=LIME, width=8)
    g.ellipse([x0 - 14, y - 14, x0 + 14, y + 14], fill=LIME)
    g.ellipse([x1 - 14, y - 14, x1 + 14, y + 14], fill=LIME if done >= 1 else BG, outline=LIME, width=5)
    lab = font("bold", 24)
    g.text((X, y + 28), f"{md(call['date']).upper()} · PICKED", font=lab, fill=FG)
    rt = f"RESULTS · {md(call['end']).upper()}"
    g.text((W - X - g.textlength(rt, font=lab), y + 28), rt, font=lab, fill=FG)

    # rows
    top, rh = 400, 76
    head = {"call": "PRICE AT PICK", "update": "SO FAR", "result": f"{h}-DAY RETURN"}[mode]
    small = font("bold", 22)
    g.text((X, top - 6), "COIN", font=small, fill=FG2)
    g.text((W - X - 300 - g.textlength(head, font=small) + 300, top - 6), head, font=small, fill=FG2)
    for i, p in enumerate(call["picks"]):
        y0 = top + 36 + i * rh
        g.rounded_rectangle([X - 16, y0, W - X + 16, y0 + rh - 10], radius=16, fill=SURF)
        hit = mode == "result" and p["hit"]
        if hit:
            g.rounded_rectangle([X - 16, y0, W - X + 16, y0 + rh - 10], radius=16, outline=LIME, width=3)
        g.text((X + 4, y0 + 17), f"#{p['rank']}", font=font("mono", 28), fill=FG2)
        g.text((X + 80, y0 + 8), p["symbol"], font=font("bold", 36), fill=FG)
        name = p["name"] if len(p["name"]) <= 26 else p["name"][:25] + "…"
        g.text((X + 80 + g.textlength(p["symbol"], font=font("bold", 36)) + 16, y0 + 20), name, font=font("reg", 24), fill=FG2)
        if mode == "call":
            val, col = price(p["start"]), FG
        else:
            val = pct(p["change"]); col = LIME if (p["change"] or 0) > 0 else DOWN if (p["change"] or 0) < 0 else FG2
        vf = font("mono", 32); right = W - X - (150 if mode == "result" else 0)
        g.text((right - g.textlength(val, font=vf), y0 + 16), val, font=vf, fill=col)
        if mode == "result":
            pill = "HIT" if hit else "MISS"
            g.rounded_rectangle([W - X - 116, y0 + 14, W - X, y0 + 54], radius=20, fill=LIME if hit else LINE)
            g.text((W - X - 58 - g.textlength(pill, font=small) / 2, y0 + 20), pill, font=small, fill=ONLIME if hit else FG2)

    # footer
    fy = H - 118
    g.line([X, fy, W - X, fy], fill=LINE, width=2)
    g.text((X, fy + 22), SHORT or "CRYPTOTRACKER PRO · NEW CALLS DAILY", font=font("bold", 26), fill=LIME)
    note = "Not financial advice · Data: CoinMarketCap"
    g.text((X, fy + 64), note, font=font("reg", 22), fill=FG2)
    tag = f"#CryptoTrackerPro {TAG[h]}"
    g.text((W - X - g.textlength(tag, font=font("bold", 22)), fy + 64), tag, font=font("bold", 22), fill=FG)
    return im


# ---------- captions ----------
def link(call):
    return f"{SITE}call.html?d={call['date']}&h={call['horizon']}" if SITE else ""


def arrow(call):
    """' 👉 <link>' when a public site URL is configured, else nothing."""
    return f" 👉 {link(call)}" if SITE else ""


def coin_tag(name):
    """'Basic Attention Token' -> '#BasicAttentionToken' (letters and digits only)."""
    words = re.findall(r"[A-Za-z0-9]+", name)
    return "#" + "".join(w[:1].upper() + w[1:] for w in words) if words else ""


def x_len(text):
    """Length as X counts it: most symbols/emoji above U+10FF weigh 2, links weigh 23."""
    text = re.sub(r"https?://\S+", "x" * 23, text)
    return sum(2 if ord(ch) > 0x10FF else 1 for ch in text)


def caption(call, mode, limit=None):
    """Short daily captions; the explanation lives once on the profile (PROFILE.txt / PINNED.txt).
    With `limit` (X = 280), coin hashtags are added only while the post still fits."""
    all_tags = list(dict.fromkeys(t for t in (coin_tag(p["name"]) for p in call["picks"]) if t))
    if limit:
        kept = []
        for t in all_tags:
            trial = caption_with(call, mode, kept + [t])
            if x_len(trial) > limit:
                break
            kept.append(t)
        return caption_with(call, mode, kept)
    return caption_with(call, mode, all_tags)


def caption_with(call, mode, coin_tags):
    h = call["horizon"]
    tags = " ".join([f"#CryptoTrackerPro {TAG[h]}"] + coin_tags)
    cash = " ".join(f"${p['symbol']}" for p in call["picks"])
    rng = f"{md(call['date'])} → {md(call['end'])}"
    if mode == "call":
        head, body = f"🎯 {h}-Day Call · {rng}", f"{cash}\nResults {md(call['end'])}{arrow(call)}"
    else:
        ranked = sorted([p for p in call["picks"] if p["change"] is not None], key=lambda p: -p["change"])
        if mode == "update":
            lead = ", ".join(f"${p['symbol']} {pct(p['change'])}" for p in ranked[:3])
            head, body = f"⏳ Results tomorrow · {h}-Day Call {rng}", f"Leading: {lead}{arrow(call)}"
        else:
            hits = ", ".join(f"${p['symbol']} {pct(p['change'])}" for p in ranked if p["hit"]) or "none"
            head = f"✅ {h}-Day results · {rng}"
            body = f"{call['hits']}/10 hit the top 10: {hits} · {call['up']}/10 up{arrow(call)}"
    return f"{head}\n{body}\n{tags}\nNFA"


# ---------- main ----------
def main():
    data = json.loads((ROOT / "prediction_data.json").read_text(encoding="utf-8"))
    today, calls = build_calls(data)
    (ROOT / "site" / "calls.json").write_text(json.dumps(dict(as_of=today, calls=calls), separators=(",", ":")), encoding="utf-8")

    posts = ROOT / "posts"; out = posts / today
    out.mkdir(parents=True, exist_ok=True)
    (posts / "PINNED.txt").write_text(PINNED, encoding="utf-8")
    (posts / "PROFILE.txt").write_text(PROFILE, encoding="utf-8")
    for old in posts.iterdir():
        try:
            if old.is_dir() and d(old.name) < d(today) - timedelta(days=KEEP_DAYS):
                shutil.rmtree(old)
        except ValueError:
            pass

    plan = []
    for c in calls:
        tag = f"{c['horizon']}d_{c['date']}"
        if c["date"] == today:
            plan.append((f"{1 if c['horizon'] == 7 else 2}_call_{c['horizon']}d", c, "call"))
        elif c["status"] == "open" and c["end"] == (d(today) + timedelta(days=1)).isoformat():
            plan.append((f"3_update_{tag}", c, "update"))
        elif c["status"] == "graded" and c.get("graded_on") == today:
            plan.append((f"4_result_{tag}", c, "result"))
    lines = [f"# Post kit · {today}", "", "Post in this order. Each post: paste the caption and attach the image.", ""]
    for name, c, mode in sorted(plan, key=lambda x: x[0]):
        (out / f"{name}.txt").write_text(caption(c, mode), encoding="utf-8")
        (out / f"{name}.x.txt").write_text(caption(c, mode, limit=280), encoding="utf-8")
        card(c, mode, today).save(out / f"{name}.png", optimize=True)
        lines.append(f"- **{name}**: {mode} for the {c['horizon']}-day call {md(c['date'])} → {md(c['end'])}")
    (out / "README.md").write_text("\n".join(lines + ["", "Pinned profile post: ../PINNED.txt"]), encoding="utf-8")
    print(f"calls.json: {len(calls)} calls | post kit {out.relative_to(ROOT)}: {[p[0] for p in plan]}")


if __name__ == "__main__":
    main()
