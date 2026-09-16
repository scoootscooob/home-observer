# Teacher distillation: a frontier model decides when the observer should speak

Status: protocol v1, 2026-09-16. Written after a first-hand teacher pass over the four
workflow clips and a 29-second untrimmed segment (section 4). Companion to
`docs/model-roadmap.md`.

## 1. The idea

The local observer is one small vision-language model that watches the stream and, at every
tick, either stays silent, pushes one structured observation to the cloud coordinator, or takes
a pre-authorized action. The hard part is not describing frames; it is knowing *when* something
is worth saying and *when to stay quiet*. That judgment is what a frontier model has and a 4B
model does not, so we distill it: the frontier watches footage tick by tick under the same
contract the student will use, its decisions become the training targets, and the student learns
to reproduce them from the same frames.

This keeps the production boundary intact. In production, media never leaves the host and the
coordinator receives only what the observer pushes. The frontier sees media only in research,
on licensed public footage and consented recordings, to build the dataset.

## 2. The per-tick contract (teacher and student share it)

At tick `t` the decision is exactly one of:

| Decision | Meaning | Payload |
|---|---|---|
| `silent` | nothing the coordinator needs; the default | none |
| `context` | one observation the coordinator could not infer from the journal so far | `observation` |
| `act` | a pre-authorized rule matched with high confidence and waiting would be wrong | `action` |
| `unsure` | teacher only: needs more frames or resolution; resolved by a second look | none |

Observation kinds: `person_presence`, `activity_started`, `activity_ended`, `object_taken`,
`object_placed`, `object_moved` (repositioned without a lift), `container_state` (lid, door,
drawer), `appliance_state`, `food_state`, `hazard`, `other`. Fields: `subject`, `location`,
`detail` (under 20 words), `confidence`. Every observation is emitted once, at the first tick
where it is visibly established, never in anticipation.

Rules the teacher follows, learned from the pass in section 4:

- Silence is the default. Ongoing activity that was already reported stays silent; each cut,
  stir or reach inside an activity stays silent; camera motion stays silent.
- A repeated report is a false report. The observer's own recent journal is in the context so
  that both teacher and student can see what was already said.
- Report state that persists: an object changed surface or container, a lid or door changed
  state, food moved between containers, a person changed area. Do not report transient
  handling that returns to the same state.
- Push context rather than act unless an explicit rule in the action policy matches. The action
  policy is part of the context and is normally empty for public footage.
- The goal shapes relevance but never creates events. With a recipe goal, "plate moved, not
  lifted" is worth one line; without it, it is silence.

## 3. Dataset row and student rendering

One row per chunk (`schema_version: home-observer.stream-target/1`): source video and hash,
chunk range, tick rate, the context the teacher saw (goal, action policy, device-state text,
the journal so far), the ordered tick decisions with payloads and a short `why`, the chunk
summary, and a teacher record (model, mode, frames sent, second looks, prompt version). Rows
are produced by `scripts/teacher_label.py`; an interactive review by a person or by Claude in a
session is written in the same format with `teacher.mode = "interactive"`.

Student rendering: the frames for tick `t` enter the stream with the tick time; the target at
that position is the silence token, or a JSON line for a `context` observation, or an action
line. The student's own emitted lines are appended to the stream, so the journal-in-context the
teacher saw is exactly what the student will see. Blind mode (no narrations shown to the
teacher) is the default; a narration-assisted mode exists only for datasets with dense
timestamps and is recorded in the row so the two are never mixed silently.

## 4. First teacher pass (interactive, this session)

Footage: the four workflow clips at 2 ticks per second, 400 px tiles, and P22_105 from 40 to
69 s at 1 tick per second. Narrations were hidden during the pass and compared afterwards.

| Clip | What the teacher saw and decided | Narration check |
|---|---|---|
| take pan (P27_03, 52.5 to 61.0 s) | ticks 0 to 2.5 s: hands at the pan drawer, read as the pan being lifted; 3 to 5 s camera swing, silent; 5.5 s a pan in hand at the hob; 6 to 8 s pan set on the hob, one `object_placed` | wrong on the first two seconds: that was the lid being put on the pan (put down lid ends 2.53 s); take pan is 4.04 to 5.49 s, during the swing; put down pan 5.40 to 9.67 s |
| take lid (P27_03, 44.5 to 53.0 s) | 3.5 to 4.0 s hand takes a lid at the sink area, one `object_taken`; 6 to 8 s back at the drawer, hand on the pan, silent | take lid 3.18 to 5.40 s, correct; put down lid 4.48 to 10.53 s |
| take spoon (P28_07, 16.0 to 27.0 s) | 0.5 s roasting tray being pulled out of the open oven with a glove, one `appliance_state` (oven open, tray out); 6.5 s spoon picked up from the counter, one `object_taken`; 8.5 s stirring starts, one `activity_started`; all else silent | take spoon 6.15 to 6.90 s, stir food 7.99 to 14.73 s, both correct |
| move plate (P22_105, 55.0 to 67.0 s) | 3.0 to 3.5 s right plate slid toward the center (`object_moved`, only under a plate-related goal); 6.0 s mozzarella transferred to the right plate, one `food_state`; fork put down, silent; 9.5 s plate moved again, silent as a repeat | move plate 2.95 to 4.00 s and 9.27 to 10.25 s; pick up and put down mozzarella 4.76 to 6.83 s; put down cutlery 7.08 to 8.94 s; all correct |
| P22_105 40 to 69 s | 40 to 42 s person arrives at the dining table, one `person_presence`; 44 s cutting mozzarella begins, one `activity_started`; 45 to 57 s silent; 58 s plate slid, silent; 60 to 61 s mozzarella served onto the second plate, one `food_state`; 63 s fork down, silent; 66 to 67 s person leaves for the kitchen counter, one `person_presence` | cut mozzarella 43.25 to 57.44 s, then the plate sequence above; correct |

What this calibrates:

- Rate of speech. 3 to 4 pushes in 29 seconds of dense cooking activity, 1 to 2 per 8 to
  12 second clip, silence at roughly 90 percent of ticks even in busy footage. Quiet footage
  should be silent at nearly all ticks; that is the distribution the student must learn.
- Frame rate. 1 tick per second is enough for activity-level and presence context; a lift that
  lasts under a second (the lid) needs 2 ticks per second, and disambiguating "lid placed on
  pan" from "pan lifted" needed either 4 ticks per second or the hidden narration. The teacher
  therefore labels at 2 Hz and takes a second look at 4 Hz and higher resolution on any tick
  it marks `unsure`. The student's own second look at higher resolution is the same mechanism.
- Egocentric footage inflates motion. The camera swings are silence in the teacher's contract,
  but in a fixed-camera home the same seconds would show a person walking, which is a
  `person_presence` change when they leave the area. Public egocentric sets remain useful for
  object and food events; presence and location targets need fixed-camera footage.
- One event, one line. The mozzarella pickup and put-down within two seconds is one served
  observation, not two events; the two plate slides are one report or none.

## 5. Building the dataset

1. Sources, in order: the licensed egocentric kitchen footage already on disk (diagnostics and
   object events), fixed multi-camera home sets as their licenses allow (presence, location,
   appliance state), simulator replays with ground-truth object state (dense, cheap, exact
   timestamps), and later our own consented recordings. `docs/research/streaming-data-2026-09.md`
   holds the survey.
2. Teacher pass with `scripts/teacher_label.py`: 20 second chunks, 2 Hz, 512 px frames, a
   2 second lead-in with no decisions, the journal chained across chunks, second looks at 4 Hz
   and 896 px for `unsure` ticks, strict JSON validated against the row schema, usage recorded.
3. Audit: on sets with dense narrations, compare teacher pushes to narrated events (onset
   collar 1 s) without ever showing the teacher the narrations; on all sets, a person reviews a
   random 5 percent of chunks and every `act`. Teacher precision and the rate of speech per
   hour are reported per source before any training run.
4. Negatives are not sampled away. Whole hours of quiet footage are labeled, and the student
   sees them in proportion; the silence token must dominate the training stream.
5. Versioning as in `data/continuous-v2`: manifests with hashes, participant-disjoint splits
   frozen before labeling, and a held-out set the teacher labels but no training run reads.

## 6. Costs (to be measured on the first batch)

Frames at 512 px cost roughly 200 input tokens each; at 2 Hz a 20 second chunk is about 40
frames or 8K tokens plus context, and one hour of footage is about 180 chunks. Through the
research proxy the cost is subscription quota; through a metered key it is dollars per hour
of footage that the first batch will measure and record in the run manifest.
