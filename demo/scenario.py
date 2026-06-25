"""
Demo scenario for the ZipperChat static demo.

Edit this file to change what the demo shows.
Then run:  python demo/build.py
"""

# ── Input email ──────────────────────────────────────────────────────────────

EMAIL = """\
From: Maya Patel <maya.patel@applied-research.example.com>
Subject: Question about ZipperGen for our research group

Hi ZipperGen team,

I am part of a small applied research group working on internal AI assistants \
for project coordination.

ZipperGen looks relevant to us because it makes coordination explicit and keeps \
human approval in the workflow. Would you have time for a short call next week \
to discuss whether it could fit this kind of setup?

Tuesday afternoon or Thursday morning would work well for me.

Best regards,
Maya\
"""

# ── LLM outputs ──────────────────────────────────────────────────────────────

CLASSIFY_ROUTE   = "scheduling"
SCHED_KIND       = "propose_slots"
AVAILABILITY     = "Tuesday 3 June at 14:00 or Thursday 5 June at 10:00"
TODAY            = "Sunday 1 June 2026"

EVENT_DETAILS_JSON = """\
{"title": "ZipperGen Research Call", "start": "2026-06-03T14:00:00", \
"end": "2026-06-03T15:00:00", \
"attendee": "maya.patel@applied-research.example.com"}\
"""

EVENT_SUMMARY = "ZipperGen Research Call — Tuesday 3 June at 14:00"

SCHEDULING_REPLY = """\
Hi Maya,

Tuesday 3 June at 14:00 works well for us. Looking forward to hearing more \
about your project.

Best,
ZipperGen team\
"""

# ── Human responses ───────────────────────────────────────────────────────────
# These are the values auto-submitted in the demo after a short pause.

SLOT_CHOICE       = "Tuesday 3 June at 14:00"
REPLY_EDIT        = SCHEDULING_REPLY   # approved as-is
