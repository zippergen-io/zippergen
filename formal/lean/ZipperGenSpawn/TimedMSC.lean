import ZipperGenParallel.MSC

namespace ZipperGenSpawn

open ZipperGenParallel

universe u v

/-- Message matching together with a linearization witness.  The timestamp is
    proof-only: it records one scheduler order in which the finite prefix was
    produced.  Local-successor and message edges must both increase it. -/
structure TimedWordMSCData {Lifeline : Type u} {Channel : Type v}
    (words : Lifeline -> Word (LocalLetter Lifeline Channel)) where
  msg : WordEvent words -> WordEvent words -> Prop
  msg_send_recv :
    forall {s r}, msg s r ->
      exists A B ch,
        s.owner = A /\ r.owner = B /\
        s.kind = EventKind.send B ch /\
        r.kind = EventKind.recv A ch
  msg_letter_match :
    forall {s r}, msg s r ->
      LocalLetter.MessageMatches s.owner r.owner s.letter r.letter
  msg_lifeline_ne :
    forall {s r}, msg s r -> Not (s.owner = r.owner)
  msg_send_unique :
    forall {s r1 r2}, msg s r1 -> msg s r2 -> r1 = r2
  msg_recv_unique :
    forall {s1 s2 r}, msg s1 r -> msg s2 r -> s1 = s2
  recv_matched :
    forall {r}, (exists A ch, r.kind = EventKind.recv A ch) ->
      exists s, msg s r
  channel_fifo :
    forall {s1 s2 r1 r2 A B ch},
      msg s1 r1 -> msg s2 r2 ->
      s1.owner = A -> s2.owner = A ->
      r1.owner = B -> r2.owner = B ->
      s1.kind = EventKind.send B ch ->
      s2.kind = EventKind.send B ch ->
      r1.kind = EventKind.recv A ch ->
      r2.kind = EventKind.recv A ch ->
      s1.localIndex < s2.localIndex ->
      r1.localIndex < r2.localIndex
  time : WordEvent words -> Nat
  local_time :
    forall {e f},
      LocalSucc (fun x : WordEvent words => x.owner)
        (fun x : WordEvent words => x.localIndex) e f ->
      time e < time f
  msg_time : forall {s r}, msg s r -> time s < time r

namespace TimedWordMSCData

variable {Lifeline : Type u} {Channel : Type v}
variable {words : Lifeline -> Word (LocalLetter Lifeline Channel)}

private theorem path_increases (D : TimedWordMSCData words) :
    forall {e f},
      Relation.TransGen
        (BaseEdge
          (fun x : WordEvent words => x.owner)
          (fun x : WordEvent words => x.localIndex)
          D.msg) e f ->
      D.time e < D.time f := by
  intro e f path
  induction path with
  | single hedge =>
      rcases hedge with hlocal | hmsg
      · exact D.local_time hlocal
      · exact D.msg_time hmsg
  | tail _ hedge ih =>
      rcases hedge with hlocal | hmsg
      · exact Nat.lt_trans ih (D.local_time hlocal)
      · exact Nat.lt_trans ih (D.msg_time hmsg)

/-- A scheduler linearization discharges the only non-local MSC obligation:
    acyclicity.  All remaining word-MSC fields are copied verbatim. -/
def toWordMSCData (D : TimedWordMSCData words) : WordMSCData words where
  msg := D.msg
  msg_send_recv := D.msg_send_recv
  msg_letter_match := D.msg_letter_match
  msg_lifeline_ne := D.msg_lifeline_ne
  msg_send_unique := D.msg_send_unique
  msg_recv_unique := D.msg_recv_unique
  recv_matched := D.recv_matched
  acyclic := by
    intro e cycle
    exact (Nat.lt_irrefl (D.time e)) (path_increases D cycle)
  channel_fifo := D.channel_fifo

theorem wordIsMSC (D : TimedWordMSCData words) : WordIsMSC words :=
  Exists.intro D.toWordMSCData True.intro

end TimedWordMSCData

end ZipperGenSpawn
