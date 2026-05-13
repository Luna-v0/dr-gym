"""Canonical catalog of DeepRacer world names + their human-readable labels.

The keys are what you put in ``WorldsConfig.names`` (and what the simapp
reads from the ``WORLD_NAME`` env var); the values are the friendly names
shown in the AWS console.

To train on every track in turn::

    from gym_dr import ALL_TRACKS, WorldsConfig
    worlds = WorldsConfig(names=ALL_TRACKS, chunk_steps=10_000, rotations=1)

Not every name in ``TRACKS`` is guaranteed to be present in every simapp
build — the upstream image ships a subset under
``.deepracer-env-upstream/tracks.txt``. Use ``existing_tracks()`` below to
filter against your local checkout if you want to avoid hitting missing
worlds at runtime.
"""
from __future__ import annotations

from pathlib import Path


TRACKS: dict[str, str] = {
    # 2024 tracks
    "2024_reinvent_champ_ccw": "Forever Raceway CCW",
    "2024_reinvent_champ_cw": "Forever Raceway CW",

    # 2022 tracks
    "2022_reinvent_champ": "2022 re:Invent Championship",
    "2022_reinvent_champ_ccw": "2022 re:Invent Championship CCW",
    "2022_reinvent_champ_cw": "2022 re:Invent Championship CW",
    "2022_october_pro": "Jennens Super Speedway",
    "2022_october_pro_ccw": "Jennens Super Speedway CCW",
    "2022_october_pro_cw": "Jennens Super Speedway CW",
    "2022_october_open": "Jennens Family Speedway",
    "2022_october_open_ccw": "Jennens Family Speedway CCW",
    "2022_october_open_cw": "Jennens Family Speedway CW",
    "2022_september_pro": "Roger Super Raceway",
    "2022_september_pro_ccw": "Roger Super Raceway CCW",
    "2022_september_pro_cw": "Roger Super Raceway CW",
    "2022_september_open": "Roger Ring",
    "2022_september_open_ccw": "Roger Ring CCW",
    "2022_september_open_cw": "Roger Ring CW",
    "2022_august_pro": "Jochem Highway",
    "2022_august_pro_ccw": "Jochem Highway CCW",
    "2022_august_pro_cw": "Jochem Highway CW",
    "2022_august_open": "Jochem Turnpike",
    "2022_august_open_ccw": "Jochem Turnpike CCW",
    "2022_august_open_cw": "Jochem Turnpike CW",
    "2022_july_pro": "DBro Super Raceway",
    "2022_july_pro_ccw": "DBro Super Raceway CCW",
    "2022_july_pro_cw": "DBro Super Raceway CW",
    "2022_july_open": "DBro Raceway",
    "2022_june_pro": "BreadCentric Speedway",
    "2022_june_pro_ccw": "BreadCentric Speedway CCW",
    "2022_june_pro_cw": "BreadCentric Speedway CW",
    "2022_june_open": "BreadCentric Loop",
    "2022_june_open_ccw": "BreadCentric Loop CCW",
    "2022_june_open_cw": "BreadCentric Loop CW",
    "2022_may_pro": "Ross Super Speedway",
    "2022_may_pro_ccw": "Ross Super Speedway CCW",
    "2022_may_pro_cw": "Ross Super Speedway CW",
    "2022_may_open": "Ross Raceway",
    "2022_may_open_ccw": "Ross Raceway CCW",
    "2022_may_open_cw": "Ross Raceway CW",
    "2022_april_pro": "Ace Super Speedway",
    "2022_april_pro_ccw": "Ace Super Speedway CCW",
    "2022_april_pro_cw": "Ace Super Speedway CW",
    "2022_april_open": "Ace Speedway",
    "2022_april_open_ccw": "Ace Speedway CCW",
    "2022_april_open_cw": "Ace Speedway CW",
    "2022_march_pro": "Rogue Raceway",
    "2022_march_pro_ccw": "Rogue Raceway CCW",
    "2022_march_pro_cw": "Rogue Raceway CW",
    "2022_march_open": "Rogue Circuit",
    "2022_march_open_ccw": "Rogue Circuit CCW",
    "2022_march_open_cw": "Rogue Circuit CW",
    "2022_summit_speedway": "RL Speedway",
    "2022_summit_speedway_ccw": "RL Speedway CCW",
    "2022_summit_speedway_cw": "RL Speedway CW",
    "2022_summit_speedway_mini": "RL Speedway Mini",

    # 2021 tracks
    "caecer_loop": "Vivalas Loop",
    "caecer_gp": "Vivalas Speedway",
    "red_star_open": "Expedition Loop",
    "red_star_pro": "Expedition Super Loop",
    "red_star_pro_ccw": "Expedition Super Loop CCW",
    "red_star_pro_cw": "Expedition Super Loop CW",
    "morgan_pro": "Playa Super Raceway",
    "morgan_open": "Playa Raceway",
    "arctic_pro": "Hot Rod Super Speedway",
    "arctic_pro_ccw": "Hot Rod Super Speedway CCW",
    "arctic_pro_cw": "Hot Rod Super Speedway CW",
    "arctic_open": "Hot Rod Speedway",
    "arctic_open_ccw": "Hot Rod Speedway CCW",
    "arctic_open_cw": "Hot Rod Speedway CW",
    "dubai_pro": "Baja Highway",
    "dubai_open": "Baja Turnpike",
    "dubai_open_ccw": "Baja Turnpike CCW",
    "dubai_open_cw": "Baja Turnpike CW",
    "hamption_open": "Kuei Raceway",
    "hamption_pro": "Kuei Super Raceway",
    "jyllandsringen_pro": "Cosmic Circuit",
    "jyllandsringen_pro_ccw": "Cosmic Circuit CCW",
    "jyllandsringen_pro_cw": "Cosmic Circuit CW",
    "jyllandsringen_open": "Cosmic Loop",
    "jyllandsringen_open_ccw": "Cosmic Loop CCW",
    "jyllandsringen_open_cw": "Cosmic Loop CW",
    "thunder_hill_pro": "Lars Circuit",
    "thunder_hill_pro_ccw": "Lars Circuit CCW",
    "thunder_hill_pro_cw": "Lars Circuit CW",
    "thunder_hill_open": "Lars Loop",
    "penbay_pro": "Po-Chun Super Speedway",
    "penbay_pro_ccw": "Po-Chun Super Speedway CCW",
    "penbay_pro_cw": "Po-Chun Super Speedway CW",
    "penbay_open": "Po-Chun Speedway",
    "penbay_open_ccw": "Po-Chun Speedway CCW",
    "penbay_open_cw": "Po-Chun Speedway CW",

    # 2020 tracks
    "Monaco_building": "European Seaside Circuit - Buildings",
    "Singapore_building": "Asia Pacific Bay Loop - Buildings",
    "Austin": "American Hills Speedway",
    "Singapore": "Asia Pacific Bay Loop",
    "Singapore_f1": "Asia Pacific Bay Loop F1",
    "Monaco": "European Seaside Circuit",
    "Aragon": "Stratus Loop",
    "Belille": "Cumulo Turnpike",
    "Albert": "Yun Speedway",
    "July_2020": "Roger Raceway",
    "FS_June2020": "Fumiaki Loop",
    "Spain_track": "Circuit de Barcelona-Catalunya",
    "Spain_track_f1": "Circuit de Barcelona-Catalunya F1",
    "reInvent2019_track": "Smile Speedway",
    "reInvent2019_track_ccw": "Smile Speedway CCW",
    "reInvent2019_track_cw": "Smile Speedway CW",

    # 2019 and earlier tracks
    "reinvent_base": "re:Invent 2018",
    "reinvent_base_jeremiah": "re:Invent 2018 (Jeremiah)",
    "reinvent_carpet": "re:Invent 2018 (Carpet)",
    "reinvent_concrete": "re:Invent 2018 (Concrete)",
    "reinvent_wood": "re:Invent 2018 (Wood)",
    "AmericasGeneratedInclStart": "Baadal Track",
    "LGSWide": "SOLA Speedway",
    "Vegas_track": "AWS Summit Raceway",
    "Canada_Training": "Toronto Turnpike Training",
    "Canada_Eval": "Toronto Turnpike Eval",
    "Mexico_track": "Cumulo Carrera Training",
    "Mexico_track_eval": "Cumulo Carrera Eval",
    "China_track": "Shanghai Sudu Training",
    "China_eval_track": "Shanghai Sudu Eval",
    "New_York_Track": "Empire City Training",
    "New_York_Eval_Track": "Empire City Eval",
    "Tokyo_Training_track": "Kumo Torakku Training",
    "Virtual_May19_Train_track": "London Loop Training",
    "Bowtie_track": "Bowtie Track",
    "Oval_track": "Oval Track",
    "reInvent2019_wide": "A to Z Speedway",
    "reInvent2019_wide_ccw": "A to Z Speedway CCW",
    "reInvent2019_wide_cw": "A to Z Speedway CW",
    "reInvent2019_wide_mirrored": "A to Z Speedway Mirrored",
    "H_track": "H track",
    "Straight_track": "Straight track",
    "AWS_track": "AWS Track",
}


ALL_TRACKS: list[str] = list(TRACKS.keys())
"""All known DeepRacer world names. Pass to ``WorldsConfig.names`` to rotate
through every track. Order is the insertion order in ``TRACKS``."""


def display_name(world_name: str) -> str:
    """Return the human-readable label for a world name (or itself if unknown)."""
    return TRACKS.get(world_name, world_name)


def existing_tracks(project_dir: str | Path | None = None) -> list[str]:
    """Filter ``ALL_TRACKS`` to the subset whose route files exist in the
    upstream simapp source on disk.

    Reads ``<project_dir>/.deepracer-env-upstream/tracks.txt`` (cloned by
    ``./bootstrap.sh``). If the file is missing, returns ``ALL_TRACKS``
    unchanged.
    """
    root = Path(project_dir) if project_dir else Path.cwd()
    tracks_file = root / ".deepracer-env-upstream" / "tracks.txt"
    if not tracks_file.exists():
        return list(ALL_TRACKS)
    existing = {line.strip() for line in tracks_file.read_text().splitlines() if line.strip()}
    return [w for w in ALL_TRACKS if w in existing]
