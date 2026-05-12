"""
utils/display.py — Terminal display for GeoSentiFake pipeline results.
No external dependencies — pure stdlib.
"""

import sys
import os


# ── ANSI colours (auto-disabled on Windows without colour support) ────────────
def _supports_colour():
    return (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
            and os.name != "nt" or os.environ.get("TERM") == "xterm-256color")


USE_COLOUR = _supports_colour()


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if USE_COLOUR else text


def red(t):     return _c("91", t)
def green(t):   return _c("92", t)
def yellow(t):  return _c("93", t)
def blue(t):    return _c("94", t)
def magenta(t): return _c("95", t)
def cyan(t):    return _c("96", t)
def bold(t):    return _c("1",  t)
def dim(t):     return _c("2",  t)
def amber(t):   return _c("33", t)


class PipelineDisplay:
    WIDTH = 72

    def print_header(self):
        print()
        print("═" * self.WIDTH)
        print(bold(cyan("  GeoSentiFake  —  Emotional Skew Quantification Pipeline")))
        print(dim("  Fake News Detection + ESS + Geopolitical Sentiment Analysis"))
        print("═" * self.WIDTH)
        print()

    def print_progress(self, current: int, total: int, title: str):
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        pct = int(100 * current / total)
        print(f"\r  [{bar}] {pct:3d}%  Article {current}/{total}: {title[:45]:<45}", end="", flush=True)
        if current == total:
            print()

    def _ess_bar(self, score: float, width: int = 30) -> str:
        filled = int(width * score / 100)
        bar = "█" * filled + "░" * (width - filled)
        if score >= 67:
            colour = red
        elif score >= 34:
            colour = amber
        else:
            colour = green
        return colour(f"[{bar}] {score:.1f}")

    def _label_colour(self, label: str) -> str:
        m = {"FAKE": red, "REAL": green, "PARTIAL": amber}
        fn = m.get(label.upper(), dim)
        return fn(f"  {label.upper():<10}")

    def _band_colour(self, band: str) -> str:
        m = {"HIGH": red, "MODERATE": amber, "LOW": green}
        fn = m.get(band.upper(), dim)
        return fn(band.upper())

    def print_result_card(self, result: dict):
        w = self.WIDTH
        print()
        print("─" * w)
        print(bold(f"  [{result['article_id']}]  {result['title'][:60]}"))
        print(dim(f"  {result['source']}  ·  {result['date']}  ·  {result['latency_ms']} ms"))
        print()

        # Label + confidence
        label_str = self._label_colour(result["label"])
        conf_str  = f"{result['confidence']:.1%}"
        print(f"  {'VERDICT':<16} {label_str}  confidence: {bold(conf_str)}")
        print(f"  {'STANCE':<16}  {cyan(result['stance'].upper())}  ({result['stance_conf']:.1%})")
        print(f"  {'CREDIBILITY':<16}  {result['credibility']:.2f} / 1.00")
        print()

        # ESS
        ess = result["ess_score"]
        gsm = result["gsm"]
        print(f"  {bold('EMOTIONAL SKEW SCORE (ESS)')}")
        print(f"    {self._ess_bar(ess)}   Band: {self._band_colour(result['ess_band'])}")
        print(f"    EIS={result['eis']:.3f}  DBS={result['dbs']:.3f}  CDS={result['cds']:.3f}  "
              f"GSM={gsm}× ({result['gsm_level']})")
        print()

        # Emotions
        emos = result["emotions"]
        dom  = result["dominant_emotion"]
        vad  = result["vad"]
        print(f"  {bold('EMOTION PROFILE')}  (dominant: {magenta(dom.upper())})")
        emo_line = ""
        for emo, prob in sorted(emos.items(), key=lambda x: -x[1]):
            bar_w = int(prob * 15)
            bar   = "▓" * bar_w + "░" * (15 - bar_w)
            emo_line += f"    {emo:<13} {bar}  {prob:.3f}\n"
        print(emo_line.rstrip())
        print(f"    VAD  →  Valence={vad['valence']:+.3f}  "
              f"Arousal={vad['arousal']:.3f}  Dominance={vad['dominance']:+.3f}")
        print()

        # SHAP top features
        print(f"  {bold('TOP FEATURES (SHAP)')}")
        for feat, imp in result["shap_top_features"][:5]:
            bar_w = int(imp * 50)
            bar = "█" * bar_w + "░" * max(0, 10 - bar_w)
            print(f"    {feat:<28} {bar}  {imp:.3f}")
        print()

        # Verdict text
        print(f"  {bold('VERDICT')}")
        verdict = result["verdict_text"]
        # wrap at 66 chars
        words = verdict.split()
        line, lines = "", []
        for w in words:
            if len(line) + len(w) + 1 > 66:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        for ln in lines:
            print(f"    {ln}")
        print()

    def print_summary(self, results: list):
        print("═" * self.WIDTH)
        print(bold("  BATCH SUMMARY"))
        print("═" * self.WIDTH)
        total = len(results)
        labels  = [r["label"] for r in results]
        fake_n  = labels.count("FAKE")
        real_n  = labels.count("REAL")
        part_n  = labels.count("PARTIAL")
        avg_ess = sum(r["ess_score"] for r in results) / total
        max_ess = max(results, key=lambda r: r["ess_score"])
        min_ess = min(results, key=lambda r: r["ess_score"])

        print(f"  Articles analysed : {total}")
        print(f"  FAKE              : {red(str(fake_n))}  "
              f"({fake_n/total:.0%})")
        print(f"  REAL              : {green(str(real_n))}  "
              f"({real_n/total:.0%})")
        print(f"  PARTIAL           : {amber(str(part_n))}  "
              f"({part_n/total:.0%})")
        print()
        print(f"  Avg ESS           : {self._ess_bar(avg_ess, 20)}")
        max_ess_score = max_ess["ess_score"]
        min_ess_score = min_ess["ess_score"]
        print(f"  Highest ESS       : {red(f'{max_ess_score:.1f}')}  "
              f"\u2192  {max_ess['title'][:50]}")
        print(f"  Lowest ESS        : {green(f'{min_ess_score:.1f}')}  "
              f"\u2192  {min_ess['title'][:50]}")
        avg_lat = sum(r["latency_ms"] for r in results) / total
        print(f"  Avg latency       : {avg_lat:.0f} ms/article")
        print()

        # ESS distribution
        print(f"  {bold('ESS DISTRIBUTION')}")
        high = [r for r in results if r["ess_band"] == "HIGH"]
        mod  = [r for r in results if r["ess_band"] == "MODERATE"]
        low  = [r for r in results if r["ess_band"] == "LOW"]
        print(f"    {red('HIGH     [67-100]')}  : {len(high)} articles")
        print(f"    {amber('MODERATE [34-66]')}  : {len(mod)} articles")
        print(f"    {green('LOW      [0-33]')}   : {len(low)} articles")
        print()
        print("═" * self.WIDTH)
        print()
