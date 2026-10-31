"""Build a self-contained SVG card from public GitHub profile data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


USER = "pxhdsg"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "stats.svg"


def get_json(url: str) -> object:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "pxhdsg-profile-stats",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def public_star_count() -> int:
    page = 1
    total = 0
    while True:
        repos = get_json(
            f"https://api.github.com/users/{USER}/repos?per_page=100&page={page}&type=owner"
        )
        if not isinstance(repos, list):
            raise TypeError("Unexpected repositories response")
        total += sum(int(repo["stargazers_count"]) for repo in repos if not repo["fork"])
        if len(repos) < 100:
            return total
        page += 1


def build_svg(repos: int, stars: int, followers: int) -> str:
    metrics = [
        ("PUBLIC REPOS", repos, "01"),
        ("STARS EARNED", stars, "02"),
        ("FOLLOWERS", followers, "03"),
    ]
    cells = []
    for index, (label, value, marker) in enumerate(metrics):
        x = 58 + index * 310
        cells.append(
            f"""
  <g transform="translate({x} 0)">
    <text x="0" y="70" class="label">{label}</text>
    <text x="0" y="134" class="value">{value:,}</text>
    <text x="258" y="134" class="marker">{marker}</text>
    <path d="M 0 153 H 260" stroke="#58718b" stroke-opacity=".55"/>
  </g>"""
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="210" viewBox="0 0 1000 210" role="img" aria-label="GitHub public statistics for {USER}: {repos} repositories, {stars} stars, {followers} followers">
  <defs>
    <linearGradient id="bg" x2="1" y2="1">
      <stop offset="0" stop-color="#101831"/>
      <stop offset="1" stop-color="#182b42"/>
    </linearGradient>
    <style>
      .label {{ fill:#7ce8da; font:700 14px 'Segoe UI',Arial,sans-serif; letter-spacing:2px; }}
      .value {{ fill:#f5f6ff; font:800 57px 'Segoe UI',Arial,sans-serif; }}
      .marker {{ fill:#7c8ca8; font:700 13px 'Segoe UI',Arial,sans-serif; }}
      .footer {{ fill:#9bb0c9; font:500 13px 'Segoe UI',Arial,sans-serif; letter-spacing:1px; }}
    </style>
  </defs>
  <rect width="1000" height="210" rx="20" fill="url(#bg)"/>
  <rect x="12" y="12" width="976" height="186" rx="13" fill="none" stroke="#61829e" stroke-opacity=".35"/>
  <circle cx="947" cy="31" r="5" fill="#78eddd"/>
  <circle cx="947" cy="31" r="10" fill="none" stroke="#78eddd" stroke-opacity=".3"/>
  {''.join(cells)}
  <text x="58" y="185" class="footer">LIVE TELEMETRY  /  every universe begins at zero</text>
</svg>
"""


def main() -> None:
    user = get_json(f"https://api.github.com/users/{USER}")
    if not isinstance(user, dict):
        raise TypeError("Unexpected user response")
    svg = build_svg(
        repos=int(user["public_repos"]),
        stars=public_star_count(),
        followers=int(user["followers"]),
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(svg, encoding="utf-8")
    print(f"Updated {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
