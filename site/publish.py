"""Publish today's post kit (posts/<date>/) after site/build_posts.py.

Destinations (each switches on only when its secrets are set):
  Telegram DM to you      TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID       the whole kit, to copy into the CMC app
  Telegram channel        TELEGRAM_BOT_TOKEN + TELEGRAM_CHANNEL_ID    e.g. @CryptoTrackerPro (bot must be an admin)
  X (Twitter)             X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET   (app with Read+Write)

Rules
  * Updates/results are published only for calls whose call post we published (no orphan results).
  * Results and updates are sent as replies to the original call post (threads on X and Telegram).
  * posts/ledger.json remembers what went where, so a second run on the same day never double-posts.
"""
import json, os, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
POSTS = ROOT / "posts"
LEDGER = POSTS / "ledger.json"


def env(name):
    return (os.environ.get(name) or "").strip()


def load_ledger():
    try:
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    except Exception:
        return {"sent": {}, "calls": {}}


def items_for(day_dir):
    """[(name, mode, call_key, caption, x_caption, image)] in posting order."""
    out = []
    for png in sorted(day_dir.glob("*.png")):
        name = png.stem
        txt, xtxt = day_dir / f"{name}.txt", day_dir / f"{name}.x.txt"
        if not txt.exists():
            continue
        parts = name.split("_")
        mode = parts[1]
        if mode == "call":
            key = f"{parts[2]}_{day_dir.name}"                        # e.g. 7d_2026-10-20
        else:
            key = f"{parts[2]}_{parts[3]}"                             # e.g. 7d_2026-10-14
        out.append((name, mode, key, txt.read_text(encoding="utf-8"),
                    xtxt.read_text(encoding="utf-8") if xtxt.exists() else txt.read_text(encoding="utf-8"), png))
    return out


# ---------- Telegram ----------
def tg(method, token, data, files=None):
    r = requests.post(f"https://api.telegram.org/bot{token}/{method}", data=data, files=files, timeout=60)
    body = r.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram {method}: {body.get('description')}")
    return body["result"]


def tg_photo(token, chat, caption, image, reply_to=None):
    data = {"chat_id": chat, "caption": caption[:1024]}
    if reply_to:
        data["reply_to_message_id"] = reply_to
        data["allow_sending_without_reply"] = "true"
    with open(image, "rb") as fh:
        return tg("sendPhoto", token, data, {"photo": fh})["message_id"]


# ---------- X ----------
def x_clients():
    keys = [env(k) for k in ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET")]
    if not all(keys):
        return None
    import tweepy
    v1 = tweepy.API(tweepy.OAuth1UserHandler(*keys))
    v2 = tweepy.Client(consumer_key=keys[0], consumer_secret=keys[1], access_token=keys[2], access_token_secret=keys[3])
    return v1, v2


def x_post(clients, text, image, reply_to=None):
    v1, v2 = clients
    media_ids = None
    try:
        media_ids = [v1.media_upload(filename=str(image)).media_id]
    except Exception as e:  # some API tiers block media upload; post text-only rather than nothing
        print(f"    X media upload failed ({e}); posting text only")
    kw = {"text": text, "media_ids": media_ids}
    if reply_to:
        kw["in_reply_to_tweet_id"] = reply_to
    return str(v2.create_tweet(**kw).data["id"])


def main():
    days = sorted(p for p in POSTS.iterdir() if p.is_dir())
    if not days:
        print("no post kit found"); return
    day = days[-1]
    ledger = load_ledger()
    sent = ledger["sent"].setdefault(day.name, {})
    items = items_for(day)

    token, me, channel = env("TELEGRAM_BOT_TOKEN"), env("TELEGRAM_CHAT_ID"), env("TELEGRAM_CHANNEL_ID")
    x = x_clients()
    targets = [t for t, on in (("telegram_me", token and me), ("telegram_channel", token and channel), ("x", x)) if on]
    status = {"date": day.name, "secrets_visible": {k: bool(env(k)) for k in (
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TELEGRAM_CHANNEL_ID", "X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET")},
        "targets": targets, "errors": {}}
    def save_status():
        (POSTS / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    save_status()
    if not targets:
        print("no destinations configured (set Telegram / X secrets to enable)"); return

    def publishable(mode, key):
        # results/updates only for calls we actually published
        return mode == "call" or key in ledger["calls"]

    for target in targets:
        done = set(sent.get(target, []))
        todo = [it for it in items if it[0] not in done and publishable(it[1], it[2])]
        if not todo:
            print(f"{target}: nothing new"); continue
        try:
            if target == "telegram_me":
                tg("sendMessage", token, {"chat_id": me, "text": f"📦 Post kit · {day.name} · {len(todo)} posts\n"
                                          "Long-press a caption to copy it, save the image, post on CMC in this order."})
            for name, mode, key, cap, xcap, img in todo:
                reply = ledger["calls"].get(key, {}).get(target) if mode != "call" else None
                if target == "x":
                    msg_id = x_post(x, xcap, img, reply)
                else:
                    msg_id = tg_photo(token, me if target == "telegram_me" else channel, cap, img, reply)
                if mode == "call":
                    ledger["calls"].setdefault(key, {})[target] = msg_id
                sent.setdefault(target, []).append(name)
                print(f"{target}: posted {name}")
                time.sleep(2)
        except Exception as e:
            print(f"{target}: FAILED: {e}")
            status["errors"][target] = f"{type(e).__name__}: {e}"[:400]
        finally:
            LEDGER.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
            save_status()


if __name__ == "__main__":
    main()
