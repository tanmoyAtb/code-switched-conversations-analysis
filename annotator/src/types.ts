/**
 * Message language mode. Unordered categories — kappa is unweighted.
 *
 * There is deliberately no `mixed`. Romanized Bangla routinely carries an English word or
 * two, which is simply how Bangla is written in Latin script — not a separate mode. A
 * `mixed` class would have no non-arbitrary boundary against that population, so annotators
 * would split it differently and the disagreement would land on the largest class.
 * Code-switching is measured at conversation level (a thread carrying more than one
 * language), which needs no per-message mixing judgement.
 *
 * There is deliberately no `other` either. Messages too short or too content-free to carry a
 * language (emoji-only, numeric-only) are filtered out upstream, so the class would have had
 * no population; an escape hatch nobody can use is worse than none, because it collects
 * whatever an annotator finds awkward and quietly hides disagreement inside itself.
 */
export const LABELS = ['english', 'bangla', 'romanized_bangla'] as const;

export type Label = (typeof LABELS)[number];

export function isLabel(value: unknown): value is Label {
  return typeof value === 'string' && (LABELS as readonly string[]).includes(value);
}

export const LABEL_DEFINITIONS: Record<Label, string> = {
  english: 'Predominantly English',
  bangla: 'Bangla written in Bengali script',
  romanized_bangla: 'Bangla written in Latin script',
};

/** Shown on buttons. The full definition is in the guidelines, not in the UI. */
export const LABEL_SHORT: Record<Label, string> = {
  english: 'English',
  bangla: 'Bangla',
  romanized_bangla: 'Romanized',
};

/**
 * A small integer typed at session start. No login, no passwords.
 *
 * This identifies an honest participant, not an authenticated one: anyone can type `2`.
 * With a handful of known annotators that is the right trade, but it is a property of the
 * data collection and belongs in the write-up rather than being assumed away.
 */
export type AnnotatorId = number;

export function parseAnnotatorId(value: unknown): AnnotatorId | null {
  const n = Number(String(value ?? '').trim());
  return Number.isInteger(n) && n >= 1 && n <= 99 ? n : null;
}

/**
 * The source distinguishes only "sent through the business account" from "sent by the
 * customer", so this is the finest split the data actually supports. Whether a business
 * message was automated is `is_bot`, which is deliberately three-valued.
 */
export type Role = 'customer' | 'business';

/**
 * Mirrors a row of data/texts_no_annotation.jsonl, which the server reads directly.
 * Text arrives already redacted: typed placeholders, no sender or business identifiers,
 * no timestamps. `business_id` identifies one business and exists for the business-level split.
 */
export interface Message {
  message_id: number;
  /** Conversation this message belongs to. Annotation is conversation-at-a-time. */
  thread_id: number;
  /**
   * Position in the ORIGINAL conversation, not in the retained one. Turns are therefore
   * gappy: a jump from 7 to 11 means messages were dropped in curation, and the UI shows
   * that gap rather than hiding it.
   */
  turn: number;
  text: string;
  role: Role;
  /** null = undetermined, NOT "human". */
  is_bot: boolean | null;
  business_id: number;
}

/**
 * One judgement. Every entry carries a label: there is no skip, so a message an annotator
 * cannot decide gets their closest guess rather than a recorded abstention. That guess is
 * indistinguishable from genuine disagreement in the agreement figures, which matters most
 * on very short messages where `ok` or `thanks` is equally English and romanized Bangla.
 */
export interface AnnotationEntry {
  message_id: number;
  turn: number;
  label: Label;
  /** Milliseconds to the first decision on this message. Noisy — for filtering, not scoring. */
  ms_spent: number;
}

/**
 * One file per (annotator, conversation), at
 * data/annotations/annotator_<id>/thread_<thread_id>.json.
 *
 * The path is the key. An annotator voting twice on the same message is not merely rejected,
 * it is unrepresentable — the directory fixes the annotator, the filename fixes the thread,
 * and `annotations` holds one entry per message_id. No index, no uniqueness check, no drift.
 *
 * There is no `revisions` counter and no `completed` flag. Git already stores every version
 * of this file, and completion is derivable from the entries; a second copy of either fact
 * would only be one that can go stale.
 */
export interface AnnotationFile {
  annotator_id: AnnotatorId;
  thread_id: number;
  business_id: number;
  is_control: boolean;
  updated_at: string;
  /** Sorted by message_id so a diff shows the labels that changed, not a reshuffle. */
  annotations: AnnotationEntry[];
}
